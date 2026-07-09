import os
import copy
from typing import List, Tuple, Dict, Set

import numpy as np
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from config import get_args
from utils import read_feat_graph, split_data, get_gpu_memory, Stop_args, EarlyStopping
from memory_count import FunctionPerformanceMonitor
import random
import time
import builtins

def extract_candidate_subgraph_nodes(G: nx.Graph, query: List[int], topn: int = 500) -> List[int]:
    if not query:
        return []

    visited: Set[int] = set(query)
    frontier: List[int] = list(query)

    while frontier and len(visited) < topn:
        next_frontier: List[int] = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
                    if len(visited) >= topn:
                        break
            if len(visited) >= topn:
                break
        if not next_frontier:
            break
        frontier = next_frontier

    nodes = list(visited)[:topn]
    return nodes


def get_graph_input_for_query(G: nx.Graph, feature: np.ndarray, query_nodes: List[int], args, device=None) -> Tuple:
    sub_nodes = extract_candidate_subgraph_nodes(G, query_nodes, topn=int(getattr(args, 'gnn_topn', 500)))
    if not sub_nodes:
        return None
        
    edge_index = build_subgraph(G, sub_nodes)
    
    add_pagerank = getattr(args, 'add_pagerank_feat', True)
    x, dist_idx = build_subgraph_features(
        G, sub_nodes, feature, query=query_nodes,
        add_pagerank_feat=add_pagerank,
        pagerank_alpha=getattr(args, 'pagerank_alpha', 0.85),
        pagerank_max_iter=getattr(args, 'pagerank_max_iter', 200),
        max_dist=int(getattr(args, 'max_dist', 20))
    )
    
    node_to_idx = {n: i for i, n in enumerate(sub_nodes)}
    q_idx = [node_to_idx[n] for n in query_nodes if n in node_to_idx]
    
    if device is not None:
        x = x.to(device)
        edge_index = edge_index.to(device)
        dist_idx = dist_idx.to(device)
        
    return x, edge_index, dist_idx, q_idx, sub_nodes


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.bincount(dst, minlength=num_nodes).clamp_min(1).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        x_norm = x * deg_inv_sqrt.view(-1, 1)
        agg = torch.zeros_like(x_norm)
        agg = agg.index_add(0, dst, x_norm[src])
        agg = agg * deg_inv_sqrt.view(-1, 1)
        h = self.linear(agg)
        return h


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3, max_dist: int = 20):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout
        
        self.dist_embedding = nn.Embedding(num_embeddings=max_dist + 1, embedding_dim=hidden_dim)
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        for i in range(num_layers):
            self.layers.append(GCNLayer(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
        self.out_dim = 2 * hidden_dim
    
    def forward(self, x, edge_index, query_idx=None, dist_idx=None, return_node_emb=False):
        if x.size(0) == 0:
            zero = torch.zeros((self.out_dim,), device=x.device)
            return zero, zero if return_node_emb else zero

        h = self.layers[0].linear(x)
        
        if dist_idx is not None:
            dist_emb = self.dist_embedding(dist_idx)
            h = h + dist_emb
        
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.bincount(dst, minlength=num_nodes).clamp_min(1).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        h_norm = h * deg_inv_sqrt.view(-1, 1)
        agg = torch.zeros_like(h_norm)
        agg = agg.index_add(0, dst, h_norm[src])
        h = agg * deg_inv_sqrt.view(-1, 1)
        
        if self.num_layers > 1:
            h = self.batch_norms[0](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        for i in range(1, self.num_layers):
            h = self.layers[i](h, edge_index)
            if i < self.num_layers - 1:
                h = self.batch_norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        node_emb = h

        global_pool = node_emb.mean(dim=0)
        
        if query_idx is not None and len(query_idx) > 0:
            query_node_embs = node_emb[query_idx]
            query_pool = query_node_embs.mean(dim=0)
        else:
            query_pool = global_pool
        
        graph_emb = torch.cat([query_pool, global_pool], dim=-1)

        if return_node_emb:
            return graph_emb, node_emb
        return graph_emb


def build_subgraph(G: nx.Graph, nodes: List[int]) -> torch.Tensor:
    if not nodes:
        return torch.zeros((2, 0), dtype=torch.long)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_set = set()
    for u in nodes:
        for v in G.neighbors(u):
            if v in node_to_idx:
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                edges_set.add((u_idx, v_idx))
                edges_set.add((v_idx, u_idx))
    num_nodes = len(nodes)
    for i in range(num_nodes):
        edges_set.add((i, i))
    
    if not edges_set:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = list(edges_set)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def compute_integer_distance_indices(G: nx.Graph, nodes: List[int], query: List[int], max_dist: int = 20) -> np.ndarray:
    if not nodes or not query:
        return np.full(len(nodes), max_dist, dtype=np.int64)
    
    subgraph = G.subgraph(nodes).copy()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    distances = np.full(len(nodes), float('inf'), dtype=np.float32)
    
    for q_node in query:
        if q_node not in subgraph:
            continue
        try:
            path_lengths = nx.single_source_shortest_path_length(subgraph, q_node)
            for node, length in path_lengths.items():
                if node in node_to_idx:
                    idx = node_to_idx[node]
                    distances[idx] = min(distances[idx], float(length))
        except Exception:
            continue
    
    dist_indices = distances.astype(np.int64)
    dist_indices[dist_indices == float('inf')] = max_dist
    dist_indices[dist_indices > max_dist] = max_dist
    
    return dist_indices


def compute_subgraph_pagerank(G: nx.Graph, nodes: List[int], query: List[int] = None,
                              alpha: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
    if not nodes:
        return np.zeros(0, dtype=np.float32)
    
    try:
        subgraph = G.subgraph(nodes).copy()
        
        if subgraph.number_of_nodes() == 0 or subgraph.number_of_edges() == 0:
            return np.ones(len(nodes), dtype=np.float32) / len(nodes)
        
        if query and len(query) > 0:
            personalization = {node: 1.0 / len(query) if node in query and node in subgraph else 0.0 
                             for node in subgraph.nodes()}
            if sum(personalization.values()) == 0:
                personalization = None
        else:
            personalization = None
        
        pagerank_scores = nx.pagerank(subgraph, alpha=alpha, 
                                     personalization=personalization,
                                     max_iter=max_iter, 
                                     tol=tol)
        
        scores_array = np.array([pagerank_scores.get(node, 0.0) for node in nodes], dtype=np.float32)
        
        if scores_array.sum() > 0:
            scores_array = scores_array / scores_array.sum()
        
        return scores_array
    except Exception as e:
        print(f"Warning: PageRank computation failed in feature building: {e}")
        return np.ones(len(nodes), dtype=np.float32) / len(nodes)


def sample_training_data(train_samples, Y_model, Y_model_f1_scores, Y_param_scores_list, 
                         Y_param_mask_list, metadata_list, sample_ratio: np.float16):
    n_samples = len(train_samples)
    n_selected = int(n_samples * sample_ratio)
    
    if n_selected >= n_samples:
        return train_samples, Y_model, Y_model_f1_scores, Y_param_scores_list, Y_param_mask_list, metadata_list
    
    selected_indices = list(range(n_selected))
    
    sampled_train_samples = [train_samples[i] for i in selected_indices]

    if Y_model is not None:
        if isinstance(Y_model, np.ndarray):
            sampled_Y_model = Y_model[selected_indices]
        else:
            sampled_Y_model = [Y_model[i] for i in selected_indices]
    else:
        sampled_Y_model = None

    if Y_model_f1_scores is not None:
        if isinstance(Y_model_f1_scores, np.ndarray):
            sampled_Y_model_f1_scores = Y_model_f1_scores[selected_indices]
        else:
            sampled_Y_model_f1_scores = [Y_model_f1_scores[i] for i in selected_indices]
    else:
        sampled_Y_model_f1_scores = None

    sampled_Y_param_scores_list = [Y_param_scores_list[i] for i in selected_indices] if Y_param_scores_list is not None else None
    sampled_Y_param_mask_list = [Y_param_mask_list[i] for i in selected_indices] if Y_param_mask_list is not None else None
    sampled_metadata_list = [metadata_list[i] for i in selected_indices] if metadata_list is not None else None
    
    return sampled_train_samples, sampled_Y_model, sampled_Y_model_f1_scores, sampled_Y_param_scores_list, sampled_Y_param_mask_list, sampled_metadata_list


def build_subgraph_features(G: nx.Graph, nodes: List[int], feature: np.ndarray, query: List[int] = None, add_pagerank_feat: bool = True,
                            pagerank_alpha: float = 0.85, pagerank_max_iter: int = 200, max_dist: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    if not nodes:
        feat_dim = feature.shape[1] if feature is not None and len(feature.shape) > 1 else 1
        extra_dims = 1 + (1 if add_pagerank_feat else 0)
        x = torch.zeros((0, feat_dim + extra_dims), dtype=torch.float32)
        dist_idx = torch.zeros((0,), dtype=torch.long)
        return x, dist_idx
    
    x0 = feature[nodes].astype(np.float32)
    
    extra_features = []
    
    if query:
        query_set = set(query)
        query_indicator = np.array([1.0 if n in query_set else 0.0 for n in nodes], dtype=np.float32)
        extra_features.append(query_indicator.reshape(-1, 1))
    else:
        query_indicator = np.zeros(len(nodes), dtype=np.float32)
        extra_features.append(query_indicator.reshape(-1, 1))
    
    if add_pagerank_feat:
        pagerank_feat = compute_subgraph_pagerank(G, nodes, query, 
                                                  alpha=pagerank_alpha, 
                                                  max_iter=pagerank_max_iter)
        
        pagerank_feat = np.clip(pagerank_feat, a_min=1e-10, a_max=None)
        pagerank_feat_log = np.log(pagerank_feat)
        pagerank_mean = np.mean(pagerank_feat_log)
        pagerank_std = np.std(pagerank_feat_log)
        if pagerank_std > 1e-8:
            pagerank_feat_normalized = (pagerank_feat_log - pagerank_mean) / pagerank_std
        else:
            pagerank_feat_normalized = np.zeros_like(pagerank_feat_log)
        
        extra_features.append(pagerank_feat_normalized.reshape(-1, 1))
    
    if extra_features:
        x_combined = np.concatenate([x0] + extra_features, axis=1).astype(np.float32)
    else:
        x_combined = x0
    
    x = torch.from_numpy(x_combined).float()
    
    if query:
        dist_indices = compute_integer_distance_indices(G, nodes, query, max_dist=max_dist)
    else:
        dist_indices = np.full(len(nodes), max_dist, dtype=np.int64)
    dist_idx = torch.from_numpy(dist_indices).long()
    
    return x, dist_idx


def _load_rec_data(args, suffix: str = 'recLabel') -> Dict[str, pd.DataFrame]:
    tables = {}
    for model_name in getattr(args, 'all_models', []):
        xlsx = os.path.join(args.process_path, f"{args.dataset}_{model_name}_{suffix}.xlsx")
        if os.path.exists(xlsx):
            try:
                tables[model_name] = pd.read_excel(xlsx, engine="openpyxl")
            except Exception:
                tables[model_name] = pd.read_excel(xlsx)
    return tables


def _parse_thr_value(thr_value) -> List[float]:
    if pd.isna(thr_value):
        return [0.0]
    s = str(thr_value).strip()
    parts = s.split('_')
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    return out if out else [0.0]


def _get_param_values_for_model(model_name: str, args) -> List[tuple]:
    if model_name not in args.thr_dict or args.thr_dict[model_name] is None:
        return [tuple([0.0])]
    
    td = args.thr_dict[model_name]
    min_thr = td.get('min_thr', [0])
    max_thr = td.get('max_thr', [1])
    thr_step = td.get('thr_step', [1])
    thr_type = td.get('type', ['float'])
    
    if len(min_thr) == 1:
        lo = float(min_thr[0])
        hi = float(max_thr[0])
        step = float(thr_step[0])
        is_int = (thr_type[0] == 'int') if len(thr_type) > 0 else False
        
        if step <= 0:
            return [tuple([lo])]
        
        param_values = []
        current = lo
        while current <= hi + 1e-9:
            if is_int:
                param_values.append(tuple([int(round(current))]))
            else:
                param_values.append(tuple([round(current, 2)]))
            current += step
        
        return param_values
    
    param_ranges = []
    for i in range(len(min_thr)):
        lo = float(min_thr[i])
        hi = float(max_thr[i])
        step = float(thr_step[i])
        is_int = (thr_type[i] == 'int') if i < len(thr_type) else False
        
        if step <= 0:
            param_range = [lo]
        else:
            param_range = []
            current = lo
            while current <= hi + 1e-9:
                if is_int:
                    param_range.append(int(round(current)))
                else:
                    param_range.append(round(current, 2))
                current += step
        param_ranges.append(param_range)
    
    import itertools
    param_values = list(itertools.product(*param_ranges))
    return [tuple(p) for p in param_values]


def _align_param_to_discrete(param_tuple: tuple, model_name: str, args) -> tuple:
    if model_name not in args.thr_dict or args.thr_dict[model_name] is None:
        if isinstance(param_tuple, (list, tuple)):
            return tuple(round(float(x), 2) for x in param_tuple)
    
    param_values_list = _get_param_values_for_model(model_name, args)
    if not param_values_list:
        if isinstance(param_tuple, (list, tuple)):
            return tuple(round(float(x), 2) for x in param_tuple)
    
    param_normalized = tuple(float(x) for x in param_tuple) if isinstance(param_tuple, (list, tuple)) else (float(param_tuple),)
    
    for val in param_values_list:
        val_normalized = tuple(float(x) for x in val)
        if len(param_normalized) == len(val_normalized):
            if all(abs(a - b) < 1e-6 for a, b in zip(param_normalized, val_normalized)):
                return val
    
    min_dist = float('inf')
    best_val = param_values_list[0]
    for val in param_values_list:
        val_normalized = tuple(float(x) for x in val)
        if len(param_normalized) == len(val_normalized):
            dist = sum(abs(a - b) for a, b in zip(param_normalized, val_normalized))
            if dist < min_dist:
                min_dist = dist
                best_val = val
    
    return best_val


def _param_to_bin_index(param_tuple: tuple, model_name: str, args, param_values_list: List[tuple] = None) -> int:
    if param_values_list is None:
        param_values_list = _get_param_values_for_model(model_name, args)
    
    if not param_values_list:
        return 0
    
    param_normalized = tuple(float(x) for x in param_tuple) if isinstance(param_tuple, (list, tuple)) else (float(param_tuple),)
    
    for idx, val in enumerate(param_values_list):
        val_normalized = tuple(float(x) for x in val)
        if len(param_normalized) == len(val_normalized):
            if all(abs(a - b) < 1e-6 for a, b in zip(param_normalized, val_normalized)):
                return idx
    
    min_dist = float('inf')
    best_idx = 0
    for idx, val in enumerate(param_values_list):
        val_normalized = tuple(float(x) for x in val)
        if len(param_normalized) == len(val_normalized):
            dist = sum(abs(a - b) for a, b in zip(param_normalized, val_normalized))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
    
    return best_idx


def prepare_end_to_end_data(args, G: nx.Graph, feature: np.ndarray, suffix: str = 'recLabel') -> Tuple:
    tables = _load_rec_data(args, suffix=suffix)
    all_models = getattr(args, 'all_models', [])
    thr_dict = getattr(args, 'thr_dict', {})  
    param_models = [m for m in all_models if m in thr_dict and thr_dict[m] is not None]
    model_to_idx = {name: i for i, name in enumerate(all_models)}
    print(f"Model filtering: {len(all_models)} total models, {len(param_models)} models need parameter recommendation")
    if len(all_models) > len(param_models):
        excluded = [m for m in all_models if m not in param_models]
    
    query_keys = set()
    for df in tables.values():
        if 'query' in df.columns:
            query_keys.update(df['query'].astype(str).tolist())
    query_keys = sorted(list(query_keys))
    
    mapping = {data['old_id']: node for node, data in G.nodes(data=True)}
    
    train_samples = []
    Y_model_list = []
    Y_model_f1_scores_list = []
    Y_param_scores_list = []
    Y_param_mask_list = []
    metadata_list = []
    
    param_values_dict = {}
    param_num_bins_list = []
    for model_name in param_models:
        param_values = _get_param_values_for_model(model_name, args)
        param_values_dict[model_name] = param_values
        param_num_bins_list.append(len(param_values))
    
    if not param_num_bins_list:
        param_num_bins_list = [10]
    
    for qkey in query_keys:
        parts = str(qkey).split('_') if '_' in str(qkey) else [str(qkey)]
        old_ids = [int(p) for p in parts if p.isdigit()]
        q_nodes = [mapping[o] for o in old_ids if o in mapping]
        if not q_nodes:
            continue
        result = get_graph_input_for_query(G, feature, q_nodes, args, device=None)
        if result is None:
            continue
        x, edge_index, dist_idx, q_idx, sub_nodes = result
        
        query_results = []
        for model_name, df in tables.items():
            rows = df[df['query'].astype(str) == str(qkey)] if 'query' in df.columns else None
            if rows is None or rows.empty:
                continue
            for _, row in rows.iterrows():
                if 'f1' in row and not pd.isna(row['f1']):
                    f1_score = float(row['f1']); f1_score = np.clip(f1_score, 0.0, 1.0)
                    thr = _parse_thr_value(row['thr']) if 'thr' in row and model_name in getattr(args, 'thr_dict', {}) else [0.0]
                    query_results.append({'model': model_name, 'threshold': tuple(thr), 'f1_score': f1_score, 'model_idx': model_to_idx.get(model_name, 0)})
        if not query_results:
            continue
        
        train_samples.append((x, edge_index, dist_idx, q_idx, q_nodes))
        
        model_best_f1 = {}
        for res in query_results:
            model_name = res['model']
            model_best_f1[model_name] = max(model_best_f1.get(model_name, 0.0), res['f1_score'])
        
        temperature = float(getattr(args, 'model_label_temperature', 0.2))
        f1_scores_array = np.array([model_best_f1.get(name, 0.0) for name in all_models], dtype=np.float32)
        
        Y_model_f1_scores_list.append(f1_scores_array.copy())
        
        if temperature > 0:
            exp_scores = np.exp(f1_scores_array / temperature)
            y_model_soft = exp_scores / (exp_scores.sum() + 1e-10)
        else:
            y_model_soft = np.zeros(len(all_models), dtype=np.float32)
            y_model_soft[np.argmax(f1_scores_array)] = 1.0
        Y_model_list.append(y_model_soft)
        
        per_model_f1_data = {mname: {} for mname in param_models}
        
        for res in query_results:
            mname = res['model']
            if mname not in param_models:
                continue
            
            param_values = param_values_dict.get(mname, [tuple([0.0])])
            b = _param_to_bin_index(res['threshold'], mname, args, param_values)
            if b < len(param_values):
                param_tuple = param_values[b]
                if param_tuple not in per_model_f1_data[mname]:
                    per_model_f1_data[mname][param_tuple] = []
                per_model_f1_data[mname][param_tuple].append(res['f1_score'])
        
        per_model_param_targets = {}
        per_model_param_masks = {}
        
        for mname in param_models:
            param_values = param_values_dict.get(mname, [tuple([0.0])])
            num_params = len(param_values[0]) if param_values and len(param_values[0]) > 0 else 1
            
            if num_params == 1:
                num_bins = len(param_values)
                f1_scores = np.zeros(num_bins, dtype=np.float32)
                mask = np.zeros(num_bins, dtype=np.float32)
                
                for b, param_tuple in enumerate(param_values):
                    if param_tuple in per_model_f1_data[mname]:
                        f1_scores[b] = np.mean(per_model_f1_data[mname][param_tuple])
                        mask[b] = 1.0
                
                if f1_scores.sum() > 1e-6:
                    f1_scores = f1_scores / (f1_scores.sum() + 1e-10)
                else:
                    valid_bins = mask.sum()
                    if valid_bins > 0:
                        f1_scores = mask / valid_bins
                    else:
                        f1_scores = np.ones_like(f1_scores) / len(f1_scores)
                
                per_model_param_targets[mname] = {0: f1_scores}
                per_model_param_masks[mname] = {0: mask}
            else:
                param_targets = {}
                param_masks = {}
                
                for param_idx in range(num_params):
                    param_dim_values = sorted(list(set([
                        float(pv[param_idx]) if isinstance(pv, (tuple, list)) and len(pv) > param_idx 
                        else 0.0 for pv in param_values
                    ])))
                    num_bins = len(param_dim_values)
                    
                    f1_scores = np.zeros(num_bins, dtype=np.float32)
                    mask = np.zeros(num_bins, dtype=np.float32)
                    
                    for bin_idx, param_dim_val in enumerate(param_dim_values):
                        max_f1 = 0.0
                        has_data = False
                        
                        for param_tuple, f1_list in per_model_f1_data[mname].items():
                            if isinstance(param_tuple, (tuple, list)) and len(param_tuple) > param_idx:
                                if abs(float(param_tuple[param_idx]) - param_dim_val) < 1e-6:
                                    if param_idx == 0:
                                        max_f1 = max(max_f1, np.mean(f1_list))
                                        has_data = True
                                    else:
                                        best_param_0 = None
                                        best_f1_param_0 = 0.0
                                        for other_tuple, other_f1_list in per_model_f1_data[mname].items():
                                            if isinstance(other_tuple, (tuple, list)) and len(other_tuple) > 0:
                                                if abs(float(other_tuple[param_idx]) - param_dim_val) < 1e-6:
                                                    if len(other_tuple) > 0:
                                                        other_f1 = np.mean(other_f1_list)
                                                        if other_f1 > best_f1_param_0:
                                                            best_f1_param_0 = other_f1
                                                            best_param_0 = float(other_tuple[0])
                                        
                                        if len(param_tuple) > 0:
                                            param_0_val = float(param_tuple[0])
                                            if best_param_0 is None or abs(param_0_val - best_param_0) < 1e-6:
                                                max_f1 = max(max_f1, np.mean(f1_list))
                                                has_data = True
                        
                        if has_data:
                            f1_scores[bin_idx] = max_f1
                            mask[bin_idx] = 1.0
                    
                    if f1_scores.sum() > 1e-6:
                        f1_scores = f1_scores / (f1_scores.sum() + 1e-10)
                    else:
                        valid_bins = mask.sum()
                        if valid_bins > 0:
                            f1_scores = mask / valid_bins
                        else:
                            f1_scores = np.ones_like(f1_scores) / len(f1_scores)
                    
                    param_targets[param_idx] = f1_scores
                    param_masks[param_idx] = mask
                
                per_model_param_targets[mname] = param_targets
                per_model_param_masks[mname] = param_masks
        
        max_param_dims = {}
        for mname in param_models:
            if mname in per_model_param_targets:
                max_param_dims[mname] = max([len(targets) for targets in per_model_param_targets[mname].values()])
        
        Y_param_scores_list.append(per_model_param_targets)
        Y_param_mask_list.append(per_model_param_masks)
        
        best_result = max(query_results, key=lambda x: x['f1_score'])
        metadata_list.append({'query': str(qkey), 'query_nodes': q_nodes, 'best_model': best_result['model'], 
                             'best_threshold': best_result['threshold'], 'best_f1': best_result['f1_score']})
    
    if not train_samples:
        raise ValueError("No training data found in recLabel files!")
    
    Y_model = np.array(Y_model_list, dtype=np.float32)
    Y_model_f1_scores = np.array(Y_model_f1_scores_list, dtype=np.float32)
    
    return train_samples, Y_model, Y_model_f1_scores, Y_param_scores_list, Y_param_mask_list, param_num_bins_list, all_models, param_models, param_values_dict

def compute_param_validation_loss(param_model, gnn_model, valid_samples, Y_param_scores_list_valid, 
                                 Y_param_mask_list_valid, param_models, args, feature, device,
                                 use_ablation_gnn=False, use_zero_gnn=False,
                                 gnn_hidden_dim=128, lambda_param_KL=1.0):
    if not use_ablation_gnn:
        gnn_model.eval()
    param_model.eval()
    
    use_ablation_param_ce = getattr(args, 'ablation_mode', None) is not None and 'param_ce' in getattr(args, 'ablation_mode', '').lower()
    
    if use_ablation_param_ce:
        ce_loss_fn = nn.CrossEntropyLoss()
    else:
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    valid_loss_total = 0.0
    valid_kl_total = 0.0
    valid_mse_total = 0.0
    valid_ce_total = 0.0
    num_valid_samples = 0
    
    with torch.no_grad():
        for sample_idx, (x, edge_index, dist_idx, q_idx, q_nodes) in enumerate(valid_samples):
            if use_ablation_gnn:
                if use_zero_gnn:
                    query_feat = torch.zeros((1, 2 * gnn_hidden_dim), dtype=torch.float32).to(device)
            else:
                x = x.to(device)
                edge_index = edge_index.to(device)
                dist_idx = dist_idx.to(device)
                if use_zero_gnn:
                    query_feat = torch.zeros((1, 2 * gnn_hidden_dim), dtype=torch.float32).to(device)
                else:
                    query_feat = gnn_model(x, edge_index, query_idx=q_idx, dist_idx=dist_idx).unsqueeze(0)
            
            outputs = param_model(query_feat)
            
            sample_kl_list = []
            sample_mse_list = []
            sample_ce_list = []
            
            for model_name in param_models:
                if model_name not in outputs or sample_idx >= len(Y_param_scores_list_valid):
                    continue
                if model_name not in Y_param_scores_list_valid[sample_idx]:
                    continue
                
                model_outputs = outputs[model_name]
                param_names = sorted(model_outputs.keys())
                
                for param_name in param_names:
                    if param_name not in model_outputs:
                        continue
                    
                    output_dict = model_outputs[param_name]
                    log_probs = output_dict['log_probs']
                    recommended_val = output_dict.get('recommended_val', None)
                    
                    if sample_idx < len(Y_param_scores_list_valid) and model_name in Y_param_scores_list_valid[sample_idx]:
                        param_idx = int(param_name.split('_')[1]) if '_' in param_name else 0
                        
                        if param_idx in Y_param_scores_list_valid[sample_idx][model_name]:
                            true_f1_scores = Y_param_scores_list_valid[sample_idx][model_name][param_idx]
                            true_mask = Y_param_scores_list_valid[sample_idx][model_name].get(param_idx, np.zeros_like(true_f1_scores))
                            
                            true_dist = torch.from_numpy(true_f1_scores).float().to(device)
                            true_mask_t = torch.from_numpy(true_mask).float().to(device)
                            
                            if true_mask_t.sum() > 0:
                                if use_ablation_param_ce:
                                    true_best_bin = torch.argmax(true_dist).item()
                                    label_tensor = torch.tensor([true_best_bin], dtype=torch.long, device=device)
                                    ce_loss_m = ce_loss_fn(log_probs, label_tensor)
                                    sample_ce_list.append(ce_loss_m)
                                else:
                                    kl_loss_m = kl_loss_fn(log_probs, true_dist.unsqueeze(0))
                                    sample_kl_list.append(kl_loss_m)
                                
                                if not use_ablation_param_ce and recommended_val is not None:
                                    bins_key = f"{model_name}_{param_name}_bins"
                                    bin_values = None
                                    if hasattr(param_model, bins_key):
                                        bin_values = getattr(param_model, bins_key).to(device)
                                    
                                    if bin_values is not None:
                                        true_best_bin = torch.argmax(true_dist).item()
                                        true_best_param = bin_values[true_best_bin].unsqueeze(0)
                                        param_range = bin_values.max() - bin_values.min()
                                        if param_range > 1e-6:
                                            mse_loss_m = F.mse_loss(recommended_val, true_best_param) / (param_range ** 2)
                                        else:
                                            mse_loss_m = F.mse_loss(recommended_val, true_best_param)
                                        sample_mse_list.append(mse_loss_m)
            
            if use_ablation_param_ce:
                if sample_ce_list:
                    valid_loss_total += torch.stack(sample_ce_list).mean().item()
                    valid_ce_total += torch.stack(sample_ce_list).mean().item()
                    num_valid_samples += 1
            else:
                if sample_kl_list or sample_mse_list:
                    kl_loss = torch.stack(sample_kl_list).mean() if sample_kl_list else torch.tensor(0.0, device=device)
                    mse_loss = torch.stack(sample_mse_list).mean() if sample_mse_list else torch.tensor(0.0, device=device)
                    valid_loss_total += (lambda_param_KL * kl_loss + mse_loss).item()
                    valid_kl_total += kl_loss.item()
                    valid_mse_total += mse_loss.item()
                    num_valid_samples += 1
    
    if num_valid_samples > 0:
        return valid_loss_total / num_valid_samples, valid_kl_total / num_valid_samples, valid_mse_total / num_valid_samples, valid_ce_total / num_valid_samples
    else:
        return float('inf'), 0.0, 0.0, 0.0


def compute_model_validation_loss(model_model, gnn_model, valid_samples, Y_model_valid, Y_model_f1_scores_valid,
                                  args, feature, device, use_ablation_gnn=False,
                                  use_zero_gnn=False, gnn_hidden_dim=128):
    if not use_ablation_gnn:
        gnn_model.eval()
    model_model.eval()
    
    use_ablation_model_ce = getattr(args, 'ablation_mode', None) is not None and 'model_ce' in getattr(args, 'ablation_mode', '').lower()
    
    if use_ablation_model_ce:
        ce_loss_fn = nn.CrossEntropyLoss()
        if len(Y_model_valid) > 0:
            Y_model_valid_t = torch.from_numpy(Y_model_valid).float().to(device)
            Y_model_labels_valid = torch.argmax(Y_model_valid_t, dim=1).long()
        else:
            Y_model_labels_valid = None
    else:
        lambda_model_KL = float(getattr(args, 'lambda_model_KL', 1.0))
        temp_target = float(getattr(args, 'hybrid_loss_temp_target', 10.0))
        top_k = int(getattr(args, 'rec_train_topk', 3))
        margin = float(getattr(args, 'hybrid_loss_margin', 0.1))
        hybrid_loss_fn = HybridTopKLoss(lambda_model_KL=lambda_model_KL, temp_target=temp_target, top_k=top_k, margin=margin)
        if len(Y_model_f1_scores_valid) > 0:
            Y_model_f1_scores_valid_t = torch.from_numpy(Y_model_f1_scores_valid).float().to(device)
        else:
            Y_model_f1_scores_valid_t = None
    
    valid_loss_total = 0.0
    valid_kl_total = 0.0
    valid_rank_total = 0.0
    valid_ce_total = 0.0
    num_valid_batches = 0
    
    with torch.no_grad():
        gnn_embs_batch = []
        batch_indices = []
        
        for sample_idx, (x, edge_index, dist_idx, q_idx, q_nodes) in enumerate(valid_samples):
            if use_ablation_gnn:
                if use_zero_gnn:
                    query_feat = torch.zeros(2 * gnn_hidden_dim, dtype=torch.float32).to(device)
            else:
                x = x.to(device)
                edge_index = edge_index.to(device)
                dist_idx = dist_idx.to(device)
                if use_zero_gnn:
                    query_feat = torch.zeros(2 * gnn_hidden_dim, dtype=torch.float32).to(device)
                else:
                    query_feat = gnn_model(x, edge_index, query_idx=q_idx, dist_idx=dist_idx)
            
            gnn_embs_batch.append(query_feat)
            batch_indices.append(sample_idx)
            
            end_of_batch = (len(gnn_embs_batch) == 64) or ((sample_idx + 1) == len(valid_samples))
            if not end_of_batch:
                continue
            
            gnn_emb_batch = torch.stack(gnn_embs_batch, dim=0)
            logits = model_model(gnn_emb_batch)
            
            if use_ablation_model_ce:
                if Y_model_labels_valid is not None and len(batch_indices) > 0:
                    batch_labels = Y_model_labels_valid[batch_indices]
                    model_loss = ce_loss_fn(logits, batch_labels)
                    valid_loss_total += float(model_loss.item())
                    valid_ce_total += float(model_loss.item())
                    num_valid_batches += 1
            else:
                if Y_model_f1_scores_valid_t is not None and len(batch_indices) > 0:
                    true_f1_scores = Y_model_f1_scores_valid_t[batch_indices]
                    model_loss, loss_kl, loss_rank = hybrid_loss_fn(logits, true_f1_scores)
                    valid_loss_total += float(model_loss.item())
                    valid_kl_total += float(loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl)
                    valid_rank_total += float(loss_rank.item() if isinstance(loss_rank, torch.Tensor) else loss_rank)
                    num_valid_batches += 1
            
            gnn_embs_batch = []
            batch_indices = []
    
    if num_valid_batches > 0:
        return valid_loss_total / num_valid_batches, valid_kl_total / num_valid_batches, valid_rank_total / num_valid_batches, valid_ce_total / num_valid_batches
    else:
        return float('inf'), 0.0, 0.0, 0.0


def recommend_train(args, G: nx.Graph, feature: np.ndarray, device):
    ablation_mode = getattr(args, 'ablation_mode', None)
    use_ablation_gnn = ablation_mode is not None and 'gnn' in ablation_mode.lower()
    use_zero_gnn = getattr(args, 'use_zero_gnn', False)
    model_dir = os.path.join(args.model_path, "recommend")
    os.makedirs(model_dir, exist_ok=True)
    if args.rec_exp_mod == 1 or args.rec_exp_mod == 2:
        meta_path = os.path.join(model_dir, f"{args.dataset}_1_0_meta.npz")
        model_path = os.path.join(model_dir, f"{args.dataset}_1_0_model.pth")
    else:
        meta_path = os.path.join(model_dir, f"{args.dataset}_{args.rec_exp_mod}_{args.exp_param}_meta.npz")
        model_path = os.path.join(model_dir, f"{args.dataset}_{args.rec_exp_mod}_{args.exp_param}_model.pth")

    model_size = 0
    data_time = 0
    train_time = 0
    train_cpu_memory = 0
    train_gpu_memory = 0
    model_param_num = 0
    global_epoch = int(getattr(args, 'epoch', 200))
    patience = int(getattr(args, 'patience', 50))
    max_cpu_memory = 0
    max_gpu_memory = 0

    if os.path.exists(meta_path) and os.path.exists(model_path):
        meta = np.load(meta_path, allow_pickle=True)
        gnn_hidden_dim = int(meta['gnn_hidden_dim'])
        num_models = int(meta['num_models'])
        gnn_input_dim = int(meta['gnn_input_dim'])
        
        saved_param_models = meta.get('param_model_names', [])
        if isinstance(saved_param_models, np.ndarray):
            saved_param_models = saved_param_models.tolist()
        
        saved_param_num_bins_list = meta.get('param_num_bins_list', [])
        if isinstance(saved_param_num_bins_list, np.ndarray):
            saved_param_num_bins_list = saved_param_num_bins_list.tolist()
        
        all_models = getattr(args, 'all_models', [])
        thr_dict = getattr(args, 'thr_dict', {})
        current_param_models = [m for m in all_models if m in thr_dict and thr_dict[m] is not None]
        current_num_param_models = len(current_param_models) if current_param_models else 0
        
        current_param_num_bins_list = []
        for model_name in current_param_models:
            param_values = _get_param_values_for_model(model_name, args)
            current_param_num_bins_list.append(len(param_values))
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_input_dim' in meta:
            saved_model_input_dim = int(meta['model_input_dim'])
        else:
            if use_ablation_gnn:
                if use_zero_gnn:
                    saved_model_input_dim = 2 * gnn_hidden_dim
                else:
                    saved_model_input_dim = 1
            else:
                saved_model_input_dim = 2 * gnn_hidden_dim
        model_model = SimpleMLPModelRecommender(input_dim=saved_model_input_dim, num_models=num_models).to(device)
        if use_ablation_gnn:
            gnn_param_model = None
            gnn_model_selection_model = None
        else:
            max_dist = int(getattr(args, 'max_dist', 20))
            gnn_param_model = GraphEncoder(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                           num_layers=int(getattr(args, 'gnn_layers', 2)), dropout=0.3, max_dist=max_dist).to(device)
            gnn_param_model.load_state_dict(checkpoint['gnn_param'])
            
            gnn_model_selection_model = GraphEncoder(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                                      num_layers=int(getattr(args, 'gnn_layers', 2)), dropout=0.3, max_dist=max_dist).to(device)
            gnn_model_selection_model.load_state_dict(checkpoint['gnn_model'])
        
        model_model.load_state_dict(checkpoint['model'])
        
        param_values_dict = {}
        for model_name in current_param_models:
            param_values_dict[model_name] = _get_param_values_for_model(model_name, args)
        
        param_model = None
        if current_num_param_models > 0 and checkpoint.get('param') is not None:
            if 'param_input_dim' in meta and int(meta['param_input_dim']) > 0:
                saved_param_input_dim = int(meta['param_input_dim'])
            else:
                if use_ablation_gnn:
                    if use_zero_gnn:
                        saved_param_input_dim = 2 * gnn_hidden_dim
                    else:
                        saved_param_input_dim = 1
                else:
                    saved_param_input_dim = 2 * gnn_hidden_dim
            
            param_model = SimpleMLPParamRecommender(input_dim=saved_param_input_dim, num_models=current_num_param_models, 
                                                        param_num_bins_list=current_param_num_bins_list,
                                                        param_values_dict=param_values_dict,
                                                        param_model_names=current_param_models).to(device)
            param_model.load_state_dict(checkpoint['param'])
        
        models = {
            'gnn_param': gnn_param_model,
            'gnn_model': gnn_model_selection_model,
            'param': param_model,
            'model': model_model,
            'param_models': current_param_models,
            'param_values_dict': param_values_dict
        }
        for m in models.values():
            if m is not None and hasattr(m, 'eval'):
                m.eval()
        
        model_size = os.path.getsize(model_path) / 1024 / 1024 + os.path.getsize(meta_path) / 1024 / 1024
        model_list = [model_model]
        if gnn_param_model is not None:
            model_list.append(gnn_param_model)
        if gnn_model_selection_model is not None:
            model_list.append(gnn_model_selection_model)
        if param_model is not None:
            model_list.append(param_model)
        model_param_num = sum(p.numel() for m in model_list for p in m.parameters())
        print(f"Loaded trained model from {model_path}")
        return models, data_time, train_time, train_cpu_memory, train_gpu_memory, model_size, model_param_num
    
    data_start = time.time()

    in_dim = feature.shape[1] if len(feature.shape) > 1 else 1
    add_pagerank_feat = getattr(args, 'add_pagerank_feat', True)
    extra_dims = 1 + (1 if add_pagerank_feat else 0)
    gnn_input_dim = in_dim + extra_dims
    gnn_hidden_dim = int(getattr(args, 'gnn_emb_dim', 128))
    num_models = len(getattr(args, 'all_models', [])) if hasattr(args, 'all_models') and len(args.all_models)>0 else 1
    
    gnn_train_time = 0.0
    if not use_ablation_gnn:
        max_dist = int(getattr(args, 'max_dist', 20))
        gnn_model = GraphEncoder(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                 num_layers=int(getattr(args, 'gnn_layers', 2)), dropout=0.3, max_dist=max_dist).to(device)

    
    train_samples, Y_model, Y_model_f1_scores, Y_param_scores_list, Y_param_mask_list, param_num_bins_list, all_models, param_models, param_values_dict = prepare_end_to_end_data(args, G, feature)
    
    valid_samples, Y_model_valid, Y_model_f1_scores_valid, Y_param_scores_list_valid, Y_param_mask_list_valid, _, _, _, _ = prepare_end_to_end_data(args, G, feature, suffix='recValid')
    
    args.all_models = all_models
    num_models = len(all_models)
    num_param_models = len(param_models) if param_models else 0
    
    print(f"Model recommender: {num_models} models")
    print(f"Parameter recommender: {num_param_models} models")
    print(f"Training samples: {len(train_samples)}, Validation samples: {len(valid_samples)}")
    
    data_end = time.time()
    data_time = data_end - data_start - gnn_train_time
    print(f"Data preparation time: {data_time:.2f}s")

    train_start = time.time()

    if use_ablation_gnn:
        if use_zero_gnn:
            param_input_dim = 2 * gnn_hidden_dim
        else:
            param_input_dim = 1
    else:
        param_input_dim = 2 * gnn_hidden_dim
    if num_param_models > 0:
        param_model = SimpleMLPParamRecommender(input_dim=param_input_dim, num_models=num_param_models, 
                                                   param_num_bins_list=param_num_bins_list,
                                                   param_values_dict=param_values_dict,
                                                   param_model_names=param_models).to(device)
    else:
        param_model = None
    
    if use_ablation_gnn:
        if use_zero_gnn:
            model_input_dim = 2 * gnn_hidden_dim
        else:
            model_input_dim = 1
    else:
        model_input_dim = 2 * gnn_hidden_dim
    model_model = SimpleMLPModelRecommender(input_dim=model_input_dim, num_models=num_models).to(device)
    
    param_train_samples = train_samples
    param_Y_param_scores_list = Y_param_scores_list
    param_Y_param_mask_list = Y_param_mask_list
    
    if param_model is not None and num_param_models > 0:
        print(f"Starting decoupled fine-tuning for Parameter Recommendation")
        
        if not use_ablation_gnn:
            print(f"Training GNN ")
            param_opt = torch.optim.Adam([
                {'params': gnn_model.parameters(), 'lr': 1e-3}, 
                {'params': param_model.parameters(), 'lr': 1e-3}
            ], weight_decay=1e-4)
        else:
            param_opt = torch.optim.Adam(param_model.parameters(), lr=1e-3, weight_decay=1e-4)

        param_epochs = global_epoch
        param_all_loss = []
        param_best_loss = float('inf')
        param_best_state = None
        gnn_param_best_state = None
        stopping_args = Stop_args(patience=patience, max_epochs=param_epochs)
        early_stopping_param = EarlyStopping(param_model, **stopping_args)
        
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        lambda_param_KL = getattr(args, 'lambda_param_KL', 1.0)
        print(f"[Param] Using KL Divergence + MSE Loss")
        
        for ep in range(param_epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            monitor = FunctionPerformanceMonitor()
            
            def train_param_epoch():
                if not use_ablation_gnn:
                    gnn_model.train()
                param_model.train()
                batch_size = 64
                batch_losses = []
                batch_kl_list = []
                batch_mse_list = []
                n_samples = len(param_train_samples)

                param_opt.zero_grad()
                for sample_idx, (x, edge_index, dist_idx, q_idx, q_nodes) in enumerate(param_train_samples):
                    if use_ablation_gnn:
                        if use_zero_gnn:
                            query_feat = torch.zeros((1, 2 * gnn_hidden_dim), dtype=torch.float32).to(device)
                    else:
                        x = x.to(device)
                        edge_index = edge_index.to(device)
                        dist_idx = dist_idx.to(device)
                        if use_zero_gnn:
                            query_feat = torch.zeros((1, 2 * gnn_hidden_dim), dtype=torch.float32).to(device)
                        else:
                            query_feat = gnn_model(x, edge_index, query_idx=q_idx, dist_idx=dist_idx).unsqueeze(0) 
                    
                    outputs = param_model(query_feat) 
                    

                    sample_kl_list = []
                    sample_mse_list = []
                    for model_name in param_models:
                        if model_name not in outputs:
                            continue
                        
                        model_outputs = outputs[model_name]
                        param_names = sorted(model_outputs.keys()) 
                        
                        for param_name in param_names:
                            if param_name not in model_outputs:
                                continue
                            
                            output_dict = model_outputs[param_name]
                            log_probs = output_dict['log_probs'] 
                            recommended_val = output_dict['recommended_val']
                            param_idx = int(param_name.split('_')[1])
                            
                            if (sample_idx < len(param_Y_param_scores_list) and 
                                model_name in param_Y_param_scores_list[sample_idx] and
                                param_idx in param_Y_param_scores_list[sample_idx][model_name]):
                                
                                true_f1_scores = param_Y_param_scores_list[sample_idx][model_name][param_idx]
                                true_mask = param_Y_param_mask_list[sample_idx][model_name][param_idx] 
                                
                                true_dist = torch.from_numpy(true_f1_scores).float().to(device)
                                true_mask_t = torch.from_numpy(true_mask).float().to(device)
                                
                                if true_mask_t.sum() > 0:
                                    kl_loss_m = kl_loss_fn(log_probs, true_dist.unsqueeze(0))
                                    sample_kl_list.append(kl_loss_m)
                                    
                                    bins_key = f"{model_name}_{param_name}_bins"
                                    bin_values = None
                                    if hasattr(param_model, bins_key):
                                        bin_values = getattr(param_model, bins_key).to(device)
                                    
                                    if bin_values is not None:
                                        true_best_bin = torch.argmax(true_dist).item()
                                        true_best_param = bin_values[true_best_bin].unsqueeze(0)
                                        param_range = bin_values.max() - bin_values.min()
                                        if param_range > 1e-6:
                                            mse_loss_m = F.mse_loss(recommended_val, true_best_param) / (param_range ** 2)
                                        else:
                                            mse_loss_m = F.mse_loss(recommended_val, true_best_param)
                                        sample_mse_list.append(mse_loss_m)
                                            
                    if n_samples % batch_size == 0:
                        current_batch_div = batch_size
                    else:
                        last_batch_size = n_samples % batch_size
                        if sample_idx < n_samples - last_batch_size:
                            current_batch_div = batch_size
                        else:
                            current_batch_div = last_batch_size

                    if not sample_kl_list and not sample_mse_list:
                        pass
                    else:
                        kl_loss = torch.stack(sample_kl_list).mean() if sample_kl_list else torch.tensor(0.0, device=device)
                        mse_loss = torch.stack(sample_mse_list).mean() if sample_mse_list else torch.tensor(0.0, device=device)
                        sample_loss = lambda_param_KL * kl_loss + mse_loss
                        batch_losses.append(float(sample_loss.item()))
                        batch_kl_list.append(float(kl_loss.item()))
                        batch_mse_list.append(float(mse_loss.item()))
                        (sample_loss / current_batch_div).backward()

                    end_of_batch = ((sample_idx + 1) % batch_size == 0) or ((sample_idx + 1) == len(param_train_samples))
                    if end_of_batch:
                        param_opt.step()
                        param_opt.zero_grad()

                if not batch_losses:
                    return {'total': 0.0, 'kl': 0.0, 'mse': 0.0}

                avg_total = float(np.mean(batch_losses))
                avg_kl = float(np.mean(batch_kl_list)) if batch_kl_list else 0.0
                avg_mse = float(np.mean(batch_mse_list)) if batch_mse_list else 0.0
                return {'total': avg_total, 'kl': avg_kl, 'mse': avg_mse}
            
            metrics = monitor.monitor_function(train_param_epoch)
            loss_dict = metrics['result']
            param_loss_val = loss_dict['total']
            cur_cpu_memory = metrics['max_memory_mb']
            cur_gpu_memory = get_gpu_memory()
            max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
            max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
            
            param_valid_loss, param_valid_kl, param_valid_mse, param_valid_ce = compute_param_validation_loss(
                param_model, gnn_model if not use_ablation_gnn else None, valid_samples,
                Y_param_scores_list_valid, Y_param_mask_list_valid, param_models, args, feature, device,
                use_ablation_gnn=use_ablation_gnn,
                use_zero_gnn=use_zero_gnn, gnn_hidden_dim=gnn_hidden_dim, lambda_param_KL=lambda_param_KL
            )
            
            param_all_loss.append(param_valid_loss)
            
            if param_valid_loss < param_best_loss:
                param_best_loss = param_valid_loss
                param_best_state = {k: v.cpu() for k, v in param_model.state_dict().items()}
                if not use_ablation_gnn:
                    gnn_param_best_state = {k: v.cpu() for k, v in gnn_model.state_dict().items()}
                else:
                    gnn_param_best_state = None
        
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"[Param] epoch {ep+1}/{param_epochs}, Total Loss={param_loss_val:.4f}")
            
            if getattr(args, 'early_stopping', 0) != 0 and early_stopping_param.simple_check(param_all_loss):
                print(f"[Param] Early stopping at epoch {ep+1}")
                break
        
        if param_best_state is not None:
            param_model.load_state_dict(param_best_state)
        if gnn_param_best_state is not None and not use_ablation_gnn and not use_zero_gnn:
            max_dist = int(getattr(args, 'max_dist', 20))
            gnn_param_model = GraphEncoder(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                           num_layers=int(getattr(args, 'gnn_layers', 2)), dropout=0.3, max_dist=max_dist).to(device)
            gnn_param_model.load_state_dict(gnn_param_best_state)
            print(f"Saved fine-tuned GNN for parameter recommendation")
        else:
            gnn_param_model = None
    else:
        gnn_param_model = None
        
    print(f"Starting decoupled fine-tuning for Model Recommendation")
    
    if not use_ablation_gnn:
        max_dist = int(getattr(args, 'max_dist', 20))
        gnn_model = GraphEncoder(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                 num_layers=int(getattr(args, 'gnn_layers', 2)), dropout=0.3, max_dist=max_dist).to(device)
        model_opt = torch.optim.Adam([
            {'params': gnn_model.parameters(), 'lr': 1e-3}, 
            {'params': model_model.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
    else:
        model_opt = torch.optim.Adam(model_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model_epochs = global_epoch
    model_all_loss = []
    model_best_loss = float('inf')
    model_best_state = None
    gnn_model_best_state = None
    stopping_args = Stop_args(patience=patience, max_epochs=model_epochs)
    early_stopping_model = EarlyStopping(model_model, **stopping_args)
    
    lambda_model_KL = float(getattr(args, 'lambda_model_KL', 1.0))
    temp_target = float(getattr(args, 'hybrid_loss_temp_target', 10.0))
    top_k = int(getattr(args, 'rec_train_topk', 3))
    margin = float(getattr(args, 'hybrid_loss_margin', 0.1))
    hybrid_loss_fn = HybridTopKLoss(lambda_model_KL=lambda_model_KL,  temp_target=temp_target, top_k=top_k, margin=margin)
    Y_model_f1_scores_t = torch.from_numpy(Y_model_f1_scores).float().to(device)  # [N, num_models]
    print(f"[Model] Using HybridTopKLoss ")
    
    for ep in range(model_epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        
        def train_model_epoch():
                if not use_ablation_gnn:
                    gnn_model.train()
                model_model.train()

                batch_size = 64
                n_samples = len(train_samples)
                total_loss = 0.0
                total_kl = 0.0
                total_rank = 0.0
                num_batches = 0

                model_opt.zero_grad()

                gnn_embs_batch = []
                batch_indices = []

                for sample_idx, (x, edge_index, dist_idx, q_idx, q_nodes) in enumerate(train_samples):
                    if use_ablation_gnn:
                        if use_zero_gnn:
                            query_feat = torch.zeros(2 * gnn_hidden_dim, dtype=torch.float32).to(device)
                    else:
                        x = x.to(device)
                        edge_index = edge_index.to(device)
                        dist_idx = dist_idx.to(device)
                        if use_zero_gnn:
                            query_feat = torch.zeros(2 * gnn_hidden_dim, dtype=torch.float32).to(device)
                        else:
                            query_feat = gnn_model(x, edge_index, query_idx=q_idx, dist_idx=dist_idx)

                    gnn_embs_batch.append(query_feat)
                    batch_indices.append(sample_idx)

                    end_of_batch = (len(gnn_embs_batch) == batch_size) or ((sample_idx + 1) == n_samples)
                    if not end_of_batch:
                        continue

                    gnn_emb_batch = torch.stack(gnn_embs_batch, dim=0)  # [B, hidden_dim]
                    logits = model_model(gnn_emb_batch)  # [B, num_models]

                    true_f1_scores = Y_model_f1_scores_t[batch_indices]  # [B, num_models]
                    model_loss, loss_kl, loss_rank = hybrid_loss_fn(logits, true_f1_scores)

                    model_loss.backward()
                    model_opt.step()
                    model_opt.zero_grad()

                    total_loss += float(model_loss.item())
                    total_kl += float(loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl)
                    total_rank += float(loss_rank.item() if isinstance(loss_rank, torch.Tensor) else loss_rank)
                    num_batches += 1

                    gnn_embs_batch = []
                    batch_indices = []

                if num_batches == 0:
                    return {
                        'total': 0.0,
                        'kl': 0.0,
                        'rank': 0.0,
                    }

                return {
                    'total': total_loss / num_batches,
                    'kl': total_kl / num_batches,
                    'rank': total_rank / num_batches,
                }
        
        metrics = monitor.monitor_function(train_model_epoch)
        loss_dict = metrics['result']
        model_loss_val = loss_dict['total']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
        max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
        
        model_valid_loss, model_valid_kl, model_valid_rank, model_valid_ce = compute_model_validation_loss(
            model_model, gnn_model if not use_ablation_gnn else None, valid_samples,
            Y_model_valid, Y_model_f1_scores_valid, args, feature, device,
            use_ablation_gnn=use_ablation_gnn,
            use_zero_gnn=use_zero_gnn, gnn_hidden_dim=gnn_hidden_dim
        )
        
        model_all_loss.append(model_valid_loss)
        
        if model_valid_loss < model_best_loss:
            model_best_loss = model_valid_loss
            model_best_state = {k: v.cpu() for k, v in model_model.state_dict().items()}
            if not use_ablation_gnn:
                gnn_model_best_state = {k: v.cpu() for k, v in gnn_model.state_dict().items()}
            else:
                gnn_model_best_state = None
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[Model] epoch {ep+1}/{model_epochs}, Total Loss={model_loss_val:.4f}")
        
        if getattr(args, 'early_stopping', 0) != 0 and early_stopping_model.simple_check(model_all_loss):
            print(f"[Model] Early stopping at epoch {ep+1}")
            break
    
    if model_best_state is not None:
        model_model.load_state_dict(model_best_state)
    if gnn_model_best_state is not None and not use_ablation_gnn and not use_zero_gnn:
        max_dist = int(getattr(args, 'max_dist', 20))
        gnn_model_selection_model = GraphEncoder(input_dim=gnn_input_dim, hidden_dim=gnn_hidden_dim, 
                                                  num_layers=int(getattr(args, 'gnn_layers', 2)), dropout=0.3, max_dist=max_dist).to(device)
        gnn_model_selection_model.load_state_dict(gnn_model_best_state)
        print(f"Saved fine-tuned GNN for model recommendation with KL Loss: {model_best_loss:.4f}")
    else:
        gnn_model_selection_model = None
    
    train_end = time.time()
    train_time = train_end - train_start + gnn_train_time
    
    train_cpu_memory = max_cpu_memory
    train_gpu_memory = max_gpu_memory
    
    if gnn_param_model is not None:
        gnn_param_model.eval()
    if gnn_model_selection_model is not None:
        gnn_model_selection_model.eval()
    if param_model is not None:
        param_model.eval()
    model_model.eval()
    
    checkpoint = {
        'gnn_param': gnn_param_model.state_dict() if gnn_param_model is not None else None,
        'gnn_model': gnn_model_selection_model.state_dict() if gnn_model_selection_model is not None else None,
        'model': model_model.state_dict()
    }
    if param_model is not None:
        checkpoint['param'] = param_model.state_dict()
    else:
        checkpoint['param'] = None
    torch.save(checkpoint, model_path)
    
    np.savez_compressed(
        meta_path,
        gnn_input_dim=gnn_input_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        num_models=num_models,
        num_param_models=num_param_models, 
        param_num_bins_list=param_num_bins_list, 
        model_names=getattr(args, 'all_models', []),
        param_model_names=param_models if param_models else [],  
        model_input_dim=model_input_dim, 
        param_input_dim=param_input_dim if param_model is not None else 0 
    )
    
    model_size = os.path.getsize(model_path) / 1024 / 1024 + os.path.getsize(meta_path) / 1024 / 1024
    model_list = [model_model]
    if gnn_param_model is not None:
        model_list.append(gnn_param_model)
    if gnn_model_selection_model is not None:
        model_list.append(gnn_model_selection_model)
    if param_model is not None:
        model_list.append(param_model)
    model_param_num = sum(p.numel() for m in model_list for p in m.parameters())
    
    models = {
        'gnn_param': gnn_param_model, 
        'gnn_model': gnn_model_selection_model,
        'param': param_model,
        'model': model_model,
        'param_models': param_models,
        'param_values_dict': param_values_dict
    }
    print(f"Model saved to {model_path}")
    print(f"Training completed - Data: {data_time:.2f}s, Train: {train_time:.2f}s, Model size: {model_size:.2f}MB, Params: {model_param_num}")
    
    return models, data_time, train_time, train_cpu_memory, train_gpu_memory, model_size, model_param_num


def recommend_test(G: nx.Graph, feature: np.ndarray, query: List[int], args, models: Dict, device, k: int = 5):
    ablation_mode = getattr(args, 'ablation_mode', None)
    use_ablation_gnn = ablation_mode is not None and 'gnn' in ablation_mode.lower()
    use_zero_gnn = getattr(args, 'use_zero_gnn', False)
    
    gnn_param = models['gnn_param']  
    gnn_model_selection = models['gnn_model'] 
    gnn_hidden_dim = getattr(args, 'gnn_emb_dim', 128) 
    
    param_model = models.get('param') 
    model_model = models['model']
    param_models = models.get('param_models', []) 
    
    if gnn_param is not None:
        gnn_param.eval()
    if gnn_model_selection is not None:
        gnn_model_selection.eval()
    if param_model is not None:
        param_model.eval()
    model_model.eval()
    
    q_idx = []
    dist_idx = None
    x = None
    edge_index = None
    if not use_ablation_gnn:
        result = get_graph_input_for_query(G, feature, query, args, device=device)
        if result is None:
            return []
        x, edge_index, dist_idx, q_idx, _ = result
    else:
        q_idx = query
    
    with torch.no_grad():
        if param_model is not None and len(param_models) > 0:
            if use_ablation_gnn:
                if use_zero_gnn:
                    if hasattr(param_model, 'shared_mlp') and len(param_model.shared_mlp) > 0:
                        param_input_dim = param_model.shared_mlp[0].weight.shape[1]
                    else:
                        param_input_dim = 2 * gnn_hidden_dim
                    graph_emb_param = torch.zeros((1, param_input_dim), dtype=torch.float32).to(device)
            else:
                if use_zero_gnn:
                    graph_emb_param = torch.zeros((1, 2 * gnn_hidden_dim), dtype=torch.float32).to(device)
                else:
                    graph_emb_param = gnn_param(x, edge_index, query_idx=q_idx, dist_idx=dist_idx).unsqueeze(0)
            outputs = param_model(graph_emb_param)
        else:
            outputs = None
        
        if use_ablation_gnn:
            if use_zero_gnn:
                model_input_dim = model_model.net[0].weight.shape[1]
                graph_emb_model = torch.zeros((1, model_input_dim), dtype=torch.float32).to(device)
        else:
            if use_zero_gnn:
                graph_emb_model = torch.zeros((1, 2 * gnn_hidden_dim), dtype=torch.float32).to(device)
            else:
                graph_emb_model = gnn_model_selection(x, edge_index, query_idx=q_idx, dist_idx=dist_idx).unsqueeze(0)
        
        logits = model_model(graph_emb_model)
        probs = F.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy().flatten()
    
    top_indices = np.argsort(-probs_np)[:k]
    outputs_list = []
    for idx in top_indices:
        model_name = args.all_models[idx]
        if param_model is not None and model_name in param_models and outputs is not None and model_name in outputs:
            model_outputs = outputs[model_name]
            param_values_list = []
            for param_name in sorted(model_outputs.keys()):  # param_0, param_1, ...
                output_dict = model_outputs[param_name]
                
                if 'recommended_val' in output_dict:
                    param_value = float(output_dict['recommended_val'].item())
                    param_values_list.append(param_value)
                else:
                    log_probs = output_dict['log_probs']  # [1, bins]
                    probs = torch.exp(log_probs)  # [1, bins]
                    best_bin_idx = torch.argmax(probs, dim=1).item()
                    bins_key = f"{model_name}_{param_name}_bins"
                    bin_values = None
                    if hasattr(param_model, bins_key):
                        bin_values = getattr(param_model, bins_key).to(graph_emb_param.device)       
                    if bin_values is not None:
                        param_value = float(bin_values[best_bin_idx].item())
                        param_values_list.append(param_value)
                    else:
                        print(f"[WARN] Cannot get param value for {bins_key}, using 0.0")
                        param_values_list.append(0.0)
            
            if param_values_list:
                param_tuple = tuple(param_values_list)
            else:
                param_tuple = tuple([0.0])
            
            param_tuple = _align_param_to_discrete(param_tuple, model_name, args)
        else:
            param_tuple = tuple([0.0])
        outputs_list.append((model_name, param_tuple, float(probs_np[idx])))
    
    return outputs_list

class SimpleMLPParamRecommender(nn.Module):
    def __init__(self, input_dim: int, num_models: int, param_num_bins_list: List[int], 
                 param_values_dict: Dict[str, List[tuple]], param_model_names: List[str],
                 hidden_dim: int = 64):
        super().__init__()
        self.param_model_names = param_model_names
        self.model_heads = nn.ModuleDict()
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        for model_name in param_model_names:
            param_values = param_values_dict.get(model_name, [tuple([0.0])])
            num_params = len(param_values[0]) if param_values else 1
            
            for param_idx in range(num_params):
                dim_vals = sorted(list(set([float(pv[param_idx]) for pv in param_values])))
                num_bins = len(dim_vals)
                
                head_key = f"{model_name}_param_{param_idx}"
                self.model_heads[head_key] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), 
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, num_bins)
                )
                self.register_buffer(f"{head_key}_bins", torch.tensor(dim_vals))

    def forward(self, graph_features: torch.Tensor):
        h = self.shared_mlp(graph_features)
        outputs = {}
        for model_name in self.param_model_names:
            model_outputs = {}
            # Find relevant heads
            relevant_keys = [k for k in self.model_heads.keys() if k.startswith(f"{model_name}_param_")]
            for key in relevant_keys:
                param_name = key.split(f"{model_name}_")[1]
                head = self.model_heads[key]
                logits = head(h)
                log_probs = F.log_softmax(logits, dim=1)
                
                # Get recommended val (argmax)
                bins = getattr(self, f"{key}_bins")
                best_idx = torch.argmax(logits, dim=1)
                rec_val = bins[best_idx]
                
                model_outputs[param_name] = {
                    'log_probs': log_probs,
                    'recommended_val': rec_val,
                }
            outputs[model_name] = model_outputs
        return outputs

class SimpleMLPModelRecommender(nn.Module):
    def __init__(self, input_dim: int, num_models: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_models)
        )
    def forward(self, graph_features):
        return self.net(graph_features)

class HybridTopKLoss(nn.Module):
    def __init__(self, lambda_model_KL: float = 1.0, temp_target: float = 10.0, 
                 top_k: int = 3, margin: float = 0.1):
        super().__init__()
        self.lambda_model_KL = lambda_model_KL
        self.temp_target = temp_target
        self.top_k = top_k
        self.margin = margin
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
    
    def forward(self, logits: torch.Tensor, true_perfs: torch.Tensor) -> torch.Tensor:
        batch_size, num_models = logits.shape
        
        log_probs = F.log_softmax(logits, dim=1)  # [B, num_models]
        
        target_probs = F.softmax(true_perfs * self.temp_target, dim=1)  # [B, num_models]
        
        loss_kl = self.kl_loss(log_probs, target_probs)
        
        loss_rank_list = []
        
        for b in range(batch_size):
            batch_logits = logits[b]  # [num_models]
            batch_perfs = true_perfs[b]  # [num_models]
            
            sorted_indices = torch.argsort(batch_perfs, descending=True) 
            sorted_perfs = batch_perfs[sorted_indices]
            
            k = min(self.top_k, num_models)
            
            for rank in range(k):
                anchor_idx = sorted_indices[rank]
                anchor_score = batch_logits[anchor_idx] 
                
                negative_indices = sorted_indices[rank + 1:]
                
                if len(negative_indices) == 0:
                    continue 
                
                negative_scores = batch_logits[negative_indices]
                
                num_negatives = len(negative_indices)
                anchor_scores_expanded = anchor_score.expand(num_negatives) 
                targets = torch.ones(num_negatives, device=logits.device) 
                
                rank_losses = self.ranking_loss(
                    anchor_scores_expanded, 
                    negative_scores, 
                    targets
                )  # [num_negatives]
                
                non_zero_mask = rank_losses > 0
                if non_zero_mask.sum() > 0:
                    avg_rank_loss = rank_losses[non_zero_mask].mean()
                else:
                    continue 
                
                weight = 1.0 / (rank + 1)
                weighted_rank_loss = weight * avg_rank_loss
                
                loss_rank_list.append(weighted_rank_loss)
        
        if loss_rank_list:
            loss_rank = torch.stack(loss_rank_list).mean()
        else:
            loss_rank = torch.tensor(0.0, device=logits.device)
        
        total_loss = self.lambda_model_KL * loss_kl + loss_rank
        
        return total_loss, loss_kl, loss_rank