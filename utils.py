from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import os
import csv
import scipy.sparse as sp
from typing import List
import copy
import operator
from enum import Enum, auto
from typing import List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Module
import networkx as nx
import pandas as pd
import random
import torch
def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

# only compute ground-truth-based metrics
def evaluate_gt_communities(communities, labels):
    all_precisions, all_recalls, all_f1s, jaccard_scores = [], [], [], []

    for pred_community, true_community in zip(communities, labels):

        # Precision, Recall, F1
        tp = len(true_community & pred_community)
        precision = tp / len(pred_community) if len(pred_community) > 0 else 0
        recall = tp / len(true_community) if len(true_community) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        jaccard = tp / len(true_community | pred_community) if (true_community | pred_community) else 0
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        jaccard_scores.append(jaccard)

    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    avg_jaccard = np.mean(jaccard_scores)

    print(f"\nPrecision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
    print(f"Jaccard similarity: {avg_jaccard:.4f}")
    return all_precisions, all_recalls, all_f1s, jaccard_scores

# approximate diameter with single bfs
def single_bfs(graph, start):
    lengths = nx.single_source_shortest_path_length(graph, start)
    max_dist = max(lengths.values())
    return max_dist

# approximate diameter with double bfs
def double_bfs(graph, start):
    lengths = nx.single_source_shortest_path_length(graph, start)
    u = max(lengths, key=lengths.get)
    lengths_u = nx.single_source_shortest_path_length(graph, u)
    max_dist = max(lengths_u.values())
    return max_dist

def estimate_diameter(graph, n):
    nodes = list(graph.nodes)
    if len(nodes) <= n:
        return nx.diameter(graph)
    else:
        print(f"out of diameter limit, community size {len(nodes)}")
        max_dist = 0
        sampled_nodes = random.sample(nodes, n)
        for start in sampled_nodes:
            dist = single_bfs(graph, start)
            if dist > max_dist:
                max_dist = dist
        return max_dist

# compute all metrics
def evaluate_communities(graph, communities, labels, diam_limit = 500, forbid_diam = True):
    num_node = graph.number_of_nodes()
    assert num_node == (
                max(graph.nodes) + 1), f"The number of node is {num_node} while the maxid+1 is {(max(graph.nodes) + 1)}"
    all_precisions, all_recalls, all_f1s, jaccard_scores, nmis, aris = [], [], [], [], [], []
    all_degree, all_density, all_diameter, all_conductance = [], [], [], []
    com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus = [], [], [], [], [], []

    count = 1
    for pred_community, true_community in zip(communities, labels):
        print(f"evaluate query {count}/{len(communities)}")
        count += 1

        # Precision, Recall, F1
        tp = len(true_community & pred_community)
        precision = tp / len(pred_community) if len(pred_community) > 0 else 0
        recall = tp / len(true_community) if len(true_community) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        jaccard = tp / len(true_community | pred_community) if (true_community | pred_community) else 0
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        jaccard_scores.append(jaccard)

        # NMI
        pred_labels = [1 if node in pred_community else 0 for node in range(num_node)]
        true_labels = [1 if node in true_community else 0 for node in range(num_node)]
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        nmis.append(nmi)

        # ARI
        ari = adjusted_rand_score(true_labels, pred_labels)
        aris.append(ari)

        subgraph = graph.subgraph(pred_community)
        # Average degree
        degrees = [deg for _, deg in subgraph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        all_degree.append(avg_degree)

        # Density
        density = 2.0 * subgraph.number_of_edges() / (subgraph.number_of_nodes() * (subgraph.number_of_nodes() - 1))  if subgraph.number_of_nodes() > 1 else 0
        all_density.append(density)

        # Community Size
        c_size = len(pred_community)
        com_size.append(c_size)

        # Diameter
        if not forbid_diam:
            if c_size == 0:
                diameter = float('inf')
            elif nx.is_connected(subgraph):
                diameter = estimate_diameter(subgraph, diam_limit)
            else:
                diameter = float('inf')  # if there is no connected component containing query
                # for component in nx.connected_components(subgraph):
                #     if any(node in component for node in query):
                #         diameter = nx.diameter(subgraph.subgraph(component))
                #         break
        else: # forbid computation of diameter
            diameter = -1
        all_diameter.append(diameter)

        # External Conductance
        m = graph.number_of_edges()
        cut_size = len(list(nx.edge_boundary(graph, pred_community)))
        volume = sum(dict(graph.degree(pred_community)).values())
        if volume > 0 and volume != 2*m:
            conductance = cut_size / min(volume, 2 * m - volume)
        else:
            conductance = 0
        all_conductance.append(conductance)

        # Modularity
        L_C = subgraph.number_of_edges()
        if m > 0:
            Q_C = (L_C / m) - (volume / (2 * m)) ** 2
        else:
            Q_C = 0
        modularity.append(Q_C)

        # Density Modularity
        n = subgraph.number_of_nodes()
        if m > 0 and n > 0:
            d_mdu = (L_C / n) - (volume ** 2 / (4 * m * n))
        else:
            d_mdu = 0
        density_modu.append(d_mdu)

        # Local Modularity
        boundary_nodes = set() # the vertex that have at least one neighbor outside the community
        for node in pred_community:
            assert node in graph, f"{node} is not in graph"
            neighbors = set(graph.neighbors(node))
            if neighbors - pred_community:
                boundary_nodes.add(node)
        if len(boundary_nodes) > 0: # the community has boundary nodes
            internal_edges = graph.subgraph(boundary_nodes).number_of_edges() # edges between boundary nodes
            out_edges = len(list(nx.edge_boundary(graph, boundary_nodes))) # edges from boundary nodes to outside
            denominator = internal_edges + out_edges
            local_mdu = (len(list(nx.edge_boundary(graph, pred_community, boundary_nodes)))) / denominator if denominator > 0 else 0
        else:
            local_mdu = 0
        local_modu.append(local_mdu)

        # Subgraph Modularity
        if cut_size > 0:
            sub_mdu = L_C / cut_size
        else:
            sub_mdu = 0
        sub_modu.append(sub_mdu)

        # Edge surplus
        f = subgraph.number_of_edges() - (1 / 3) * c_size * (c_size - 1) / 2 # \alpha is 1/3
        edge_surplus.append(f)
        
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    avg_jaccard = np.mean(jaccard_scores)
    avg_nmi = np.mean(nmis)
    avg_ari = np.mean(aris)
    avg_degree = np.mean(all_degree)
    avg_density = np.mean(all_density)
    avg_diameter = np.mean(all_diameter)
    avg_conductance = np.mean(all_conductance)
    avg_comsize = np.mean(com_size)
    avg_modularity = np.mean(modularity)
    avg_density_modu = np.mean(density_modu)
    avg_local_modu = np.mean(local_modu)
    avg_sub_modu = np.mean(sub_modu)
    avg_edge_surplus = np.mean(edge_surplus)
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
    print(f"Jaccard similarity: {avg_jaccard:.4f}")
    return all_precisions, all_recalls, all_f1s, jaccard_scores, nmis, aris, all_degree, all_density, all_diameter, all_conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus

def get_gpu_memory():
    if torch.cuda.is_available():
        # allocated = torch.cuda.memory_allocated() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        return max_allocated
    else:
        return 0.0

def store_result(path, t, q, pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["query", "time", "f1", "pre", "rec", "jac", "nmi", "ari", "com_size", "degree", "density", "diameter", "conductance", "modularity", "density_modu", "local_modu", "sub_modu", "edge_surplus"])
    with open(path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, query in enumerate(q):
            writer.writerow([
                query,
                round(t[idx], 6),
                round(f1[idx], 6),
                round(pre[idx], 6),
                round(rec[idx], 6),
                round(jac[idx], 6),
                round(nmi[idx], 6),
                round(ari[idx], 6),
                round(com_size[idx], 6),
                round(degree[idx], 6),
                round(density[idx], 6),
                round(diameter[idx], 6),
                round(conductance[idx], 6),
                round(modularity[idx], 6),
                round(density_modu[idx], 6),
                round(local_modu[idx], 6),
                round(sub_modu[idx], 6),
                round(edge_surplus[idx], 6)
            ])

def store_summary(path, data, model, data_time, train_time, valid_time, test_time, model_size, model_param, test_cpu_memory, test_gpu_memory, train_cpu_memory, train_gpu_memory, phase = None, exp_mod = None, experiment_type = None):
    xlsx_path = path + ".xlsx"
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)
    if phase is not None:
        df = pd.DataFrame([{
        "Dataset": data,
        "Model": model,
        "Data Time (s)": round(data_time, 6),
        "Train Time (s)": round(train_time, 6),
        "Valid Time (s)": round(valid_time, 6),
        "Test Time (s)": round(test_time, 6),
        "Total Time (s)": round(data_time + train_time + valid_time + test_time, 6),
        "Model Params": model_param,
        "Model Size (MB)": round(model_size, 6),
        "test CPU Memory (MB)": round(test_cpu_memory, 6),
        "test GPU Memory (MB)": round(test_gpu_memory, 6),
        "train CPU Memory (MB)": round(train_cpu_memory, 6),
        "train GPU Memory (MB)": round(train_gpu_memory, 6),
        "Phase": phase
    }])
    else:
        df = pd.DataFrame([{
            "Dataset": data,
            "Model": model,
            "Data Time (s)": round(data_time, 6),
            "Train Time (s)": round(train_time, 6),
            "Valid Time (s)": round(valid_time, 6),
            "Test Time (s)": round(test_time, 6),
            "Total Time (s)": round(data_time + train_time + valid_time + test_time, 6),
            "Model Params": model_param,
            "Model Size (MB)": round(model_size, 6),
            "test CPU Memory (MB)": round(test_cpu_memory, 6),
            "test GPU Memory (MB)": round(test_gpu_memory, 6),
            "train CPU Memory (MB)": round(train_cpu_memory, 6),
            "train GPU Memory (MB)": round(train_gpu_memory, 6)
        }])
    if phase is not None:
        df["Phase"] = [phase]
    if exp_mod is not None and experiment_type is not None and exp_mod < len(experiment_type):
        df["exp_mod"] = [experiment_type[exp_mod]]
    if os.path.exists(xlsx_path):
        old_df = pd.read_excel(xlsx_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(xlsx_path, index=False)

def store_result_xlsx(path, t, q, pre, rec, f1, jac, nmi, ari, degree, density, diameter,
                      conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, cpu_memory, gpu_memory, thrs = None, base_cpu = 0, rec_time = None, gt_size = None, model_name = None):
    if thrs is not None:
        data = {
            "query": q,
            "thr":thrs,
            "f1": [round(x, 6) for x in f1],
            "pre": [round(x, 6) for x in pre],
            "rec": [round(x, 6) for x in rec],
            "jac": [round(x, 6) for x in jac],
            "nmi": [round(x, 6) for x in nmi],
            "ari": [round(x, 6) for x in ari],
            "com_size": [round(x, 6) for x in com_size],
            "time": [round(x, 6) for x in t],
            "degree": [round(x, 6) for x in degree],
            "density": [round(x, 6) for x in density],
            "diameter": [round(x, 6) for x in diameter],
            "conductance": [round(x, 6) for x in conductance],
            "modularity": [round(x, 6) for x in modularity],
            "density_modu": [round(x, 6) for x in density_modu],
            "local_modu": [round(x, 6) for x in local_modu],
            "sub_modu": [round(x, 6) for x in sub_modu],
            "edge_surplus": [round(x, 6) for x in edge_surplus],
            "cpu_memory": [round(x-base_cpu, 6) if (x-base_cpu) > 0 else 0 for x in cpu_memory],
            "gpu_memory": [round(x, 6) for x in gpu_memory]
        }
    else:
        data = {
            "query": q,
            "f1": [round(x, 6) for x in f1],
            "pre": [round(x, 6) for x in pre],
            "rec": [round(x, 6) for x in rec],
            "jac": [round(x, 6) for x in jac],
            "nmi": [round(x, 6) for x in nmi],
            "ari": [round(x, 6) for x in ari],
            "com_size": [round(x, 6) for x in com_size],
            "time": [round(x, 6) for x in t],
            "degree": [round(x, 6) for x in degree],
            "density": [round(x, 6) for x in density],
            "diameter": [round(x, 6) for x in diameter],
            "conductance": [round(x, 6) for x in conductance],
            "modularity": [round(x, 6) for x in modularity],
            "density_modu": [round(x, 6) for x in density_modu],
            "local_modu": [round(x, 6) for x in local_modu],
            "sub_modu": [round(x, 6) for x in sub_modu],
            "edge_surplus": [round(x, 6) for x in edge_surplus],
            "cpu_memory": [round(x-base_cpu, 6) if (x-base_cpu) > 0 else 0 for x in cpu_memory],
            "gpu_memory": [round(x, 6) for x in gpu_memory]
        }
    if rec_time is not None:
        data["rec_time"] = [round(x, 6) for x in rec_time]
    if gt_size is not None:
        data["gt_size"] = [round(x, 6) for x in gt_size]
    if model_name is not None:
        data['model'] = model_name
    df = pd.DataFrame(data)
    xlsx_path = path + ".xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

def query_onehot2array(onehot_array):
    return [np.nonzero(row)[0][0] for row in onehot_array]

def query_onehot2list(onehot_array):
    return [np.nonzero(row)[0] for row in onehot_array]

def gt_onehot2arrayset(onehot_array):
    return [set(np.nonzero(row)[0]) for row in onehot_array]

def gt_onehot2array(onehot_array):
    return [np.nonzero(row)[0] for row in onehot_array]

def single_query(queries):
    return [query[0] for query in queries]

def normalize_features(features):
    features = features.astype(np.float32)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()


class Best(Enum):
    RANKED = auto()
    ALL = auto()


def Stop_args(patience=100, max_epochs=2000):
    return dict(stop_varnames=[StopVariable.ACCURACY, StopVariable.LOSS], patience=patience, max_epochs=max_epochs,
                remember=Best.RANKED)


class EarlyStopping:
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 10, max_epochs: int = 200, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int) -> bool:
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [
                self.comp_ops[i](val, self.remembered_vals[i])
                for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    self.best_state = {
                        key: value.cpu() for key, value
                        in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not (self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            self.best_state = {
                                key: value.cpu() for key, value
                                in self.model.state_dict().items()}
                            # print('**********')
                            break
                        # else:
                        #     print('$$$$$$$$$$')
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0

    def simple_check(self, loss_list):
        if len(loss_list) <= self.patience:
            return 0
        else:
            flag = 1
            for i in range(self.patience):
                if loss_list[-1] < loss_list[-1 - i]:
                    flag = 0
            return flag

def deduplicate_communities_list(com):
    seen = set()
    deduped = []
    for group in com:
        fs = frozenset(group)
        if fs not in seen:
            seen.add(fs)
            deduped.append(np.array(sorted(group)))
    return deduped

def maskGT(q, nxg, ids, remain_num):
    dist_map = {}
    for node in ids:
        try:
            dist = nx.shortest_path_length(nxg, source=q, target=node)
        except nx.NetworkXNoPath:
            dist = float('inf')
        dist_map[node] = dist

    sorted_nodes = sorted(dist_map.items(), key=lambda x: x[1])
    community = [node for node, _ in sorted_nodes[:remain_num]]

    return community

# remove some vertices and keep the remaining vertices connected
def mask_keep_connected(nxg, community, remain_num):
    if remain_num >= len(community):
        return list(community), []
    subg = nxg.subgraph(community).copy()
    remain_nodes = set(community)
    while len(remain_nodes) > remain_num:
        cut_points = set(nx.articulation_points(subg))
        candidates = list(remain_nodes - cut_points)
        if not candidates:
            break
        to_remove = random.choice(candidates)
        remain_nodes.remove(to_remove)
        subg.remove_node(to_remove)
    removed_nodes = set(community) - remain_nodes
    return remain_nodes, removed_nodes

def read_Q_GT(query_file, gt_file, N, size, nxg, node_id_map = None, param = None, return_list = None):
    # read query
    false_query = set() # some queries are not in the graph under scalability experiment, remove corresponding groundtruth
    query_nodes = []
    with open(query_file, "r") as f:
        i = 0
        for line in f:
            ids = list(map(int, line.strip().split()))
            if node_id_map is not None:
                ids = [node_id_map[node] for node in ids if node in node_id_map]
            if not ids:
                false_query.add(i)
            query_nodes.append(ids)
            i = i + 1

    # read groundtruth
    gt_nodes = []
    new_queries = []
    with open(gt_file, "r") as f:
        i = 0
        for line in f:
            if i in false_query:
                i += 1
                continue
            ids = list(map(int, line.strip().split()))
            if node_id_map is not None:
                ids = [node_id_map[node] for node in ids if node in node_id_map]
            assert len(ids) > 0
            # filter out disconnected community
            query = query_nodes[i]
            i += 1
            gt_set = set(ids)
            subgraph = nxg.subgraph(gt_set)
            connected_subsets = list(nx.connected_components(subgraph))
            judge = False
            for comp in connected_subsets:
                if set(query).issubset(comp):
                    ids = list(comp)
                    judge = True
                    break
            if not judge: # the query vertices are not in the same groundtruth
                continue

            gt_nodes.append(ids)
            new_queries.append(query)
    query_nodes = new_queries
    assert len(gt_nodes) == len(query_nodes)

    # mask community
    new_queries = []
    new_gt_nodes = []
    if param is not None:
        n_percent = 1 - param / 100
        unique_comms = {frozenset(row) for row in gt_nodes}
        query_map_gt = {}
        gt = []
        for i, com in enumerate(unique_comms):
            n_keep = int(len(com) * n_percent)
            new_com, _ = mask_keep_connected(nxg, com, n_keep)
            for nd in new_com:
                query_map_gt[nd] = i
            gt.append(list(new_com))
        for query in query_nodes:
            assert len(query) == 1, "The number of query vertex must be 1 in mask experiment"
            if query[0] in query_map_gt:
                new_queries.append(query)
                new_gt_nodes.append(gt[query_map_gt[query[0]]])
        query_nodes = new_queries
        gt_nodes = new_gt_nodes

    assert len(gt_nodes) == len(query_nodes)

    if return_list is None:
        # query onehot
        query_nodes_onehot = torch.zeros((len(query_nodes), N))
        for i, nodelist in enumerate(query_nodes):
            for node in nodelist:
                query_nodes_onehot[i, node] = 1
        query_nodes = query_nodes_onehot.numpy().astype(np.float32)[0: size, :]

        # groundtruth onehot
        gt_nodes_onehot = torch.zeros((len(gt_nodes), N))
        for i, nodelist in enumerate(gt_nodes):
            for node in nodelist:
                gt_nodes_onehot[i, node] = 1
        gt_nodes = gt_nodes_onehot.numpy().astype(np.float32)[0: size, :]

    else:
        # return as list of sets
        query_nodes = query_nodes[:size]
        gt_nodes = [set(nodelist) for nodelist in gt_nodes][:size]

    return query_nodes, gt_nodes

def query_to_onehot(query_nodes, N):
    query_nodes_onehot = torch.zeros((len(query_nodes), N), dtype=torch.float32)
    for i, nodelist in enumerate(query_nodes):
        query_nodes_onehot[i, nodelist] = 1
    return query_nodes_onehot.numpy()

def gt_to_onehot(gt_nodes, N):
    gt_nodes_onehot = torch.zeros((len(gt_nodes), N), dtype=torch.float32)
    for i, nodeset in enumerate(gt_nodes):
        node_list = list(nodeset)
        gt_nodes_onehot[i, node_list] = 1

    return  gt_nodes_onehot.numpy()

def write_train_info(path, epoch, epoch_time, sum_time, cur_cpu_memory, cur_gpu_memory, train_loss, valid_loss, type = 0):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch/query", "time", "sum_time", "cpu_memory_MB", "gpu_memory_MB", "train_loss", "valid_loss"])
    if type == 0:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch + 1,
                round(epoch_time, 6),
                round(sum_time, 6),
                round(cur_cpu_memory, 6),
                round(cur_gpu_memory, 6),
                round(train_loss, 6),
                round(valid_loss, 6)
            ])
    elif type == 1:
        with open(path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch, # query
                round(epoch_time, 6), # train time
                round(sum_time, 6), # test time
                round(cur_cpu_memory, 6),
                round(cur_gpu_memory, 6),
                float('inf')
            ])

def read_DynamicEdge(file, G, mod = 'add'):
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            u, v = map(int, parts[:2])
            if mod == 'add':
                G.add_edge(u, v)
            elif mod == 'del':
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
    return G

def read_feat_graph(args, return_feature = True):
    # load data
    data_file = args.data_path + "/" + args.dataset

    # read feat
    if args.attr == 0:
        feature = []
        feat_file = data_file + "/feat"
        with open(feat_file, 'r') as file:
            feature = np.array([[float(line.split()[1])] for line in file],
                               dtype=np.float32)  # suppose that the feature has been normalized
    elif args.attr == 1:
        assert args.dataset not in args.dataset_snap, f"the dataset {args.dataset} does not have feature"
        feat_file = data_file + "/feat_attr.npy"
        feature = np.load(feat_file)

    # read graph
    graph_file = data_file + "/graph"
    G = nx.Graph()
    ori_N = feature.shape[0]
    G.add_nodes_from(range(ori_N))
    with open(graph_file, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)

    if args.exp_mod == 4:  # scalability
        vertex_file = data_file + f"/{args.exp_param}_vertex"
        vertex_set = set()
        with open(vertex_file, 'r') as f:
            for line in f:
                node = int(line.strip())
                vertex_set.add(node)
        G = G.subgraph(vertex_set).copy()

    if args.exp_mod == 5:  # dynamic
        if args.exp_param < 0:
            edge_file = data_file + f"/del_{abs(args.exp_param)}_edge"
            G = read_DynamicEdge(edge_file, G, "del")
        else:
            edge_file = data_file + f"/add_{abs(args.exp_param)}_edge"
            G = read_DynamicEdge(edge_file, G, "add")

    G = nx.convert_node_labels_to_integers(
        G,
        first_label=0,
        ordering="default",
        label_attribute="old_id"
    )
    node_id_map = {
        data["old_id"]: node for node, data in G.nodes(data=True)
    }
    N = G.number_of_nodes()
    max_node_id = max(G.nodes)
    assert max_node_id == N - 1
    if args.exp_mod == 4 or args.exp_mod == 5: # since the graph has changed, the initial core number of vertices should be updated
        feature = initialize_feature(G, N)
    else:
        feature_remapped = np.zeros((N, feature.shape[1]), dtype=feature.dtype)
        for old_id, new_id in node_id_map.items():
            feature_remapped[new_id] = feature[old_id]
        feature = feature_remapped
    if return_feature:
        return feature, G, node_id_map, N
    else:
        return G, node_id_map, N


def split_data(args, N, node_id_map, g, return_list = None, return_query = 'all', input_train_size = None):
    # read query and labels
    data_file = args.data_path + "/" + args.dataset
    train_query_file = data_file + "/" + args.train_query_file
    train_gt_file = data_file + "/" + args.train_gt_file
    valid_query_file = data_file + "/" + args.valid_query_file
    valid_gt_file = data_file + "/" + args.valid_gt_file
    test_query_file = data_file + "/" + args.test_query_file
    test_gt_file = data_file + "/" + args.test_gt_file
    if input_train_size is None:
        input_train_size = args.train_size
    if return_query == 'all':
        if args.exp_mod == 8:
            train_queries, train_labels = read_Q_GT(train_query_file, train_gt_file, N, input_train_size, g, node_id_map, param = args.exp_param, return_list = return_list)
        else:
            train_queries, train_labels = read_Q_GT(train_query_file, train_gt_file, N, input_train_size, g, node_id_map, return_list = return_list)
        valid_queries, valid_labels = read_Q_GT(valid_query_file, valid_gt_file, N, args.valid_size, g, node_id_map, return_list = return_list)
        test_queries, test_labels = read_Q_GT(test_query_file, test_gt_file, N, args.test_size, g, node_id_map, return_list = return_list)
        return train_queries, train_labels, valid_queries, valid_labels, test_queries, test_labels
    elif return_query == 'train':
        if args.exp_mod == 8:
            train_queries, train_labels = read_Q_GT(train_query_file, train_gt_file, N, input_train_size, g, node_id_map, param = args.exp_param, return_list = return_list)
        else:
            train_queries, train_labels = read_Q_GT(train_query_file, train_gt_file, N, input_train_size, g, node_id_map, return_list = return_list)
        return train_queries, train_labels
    elif return_query == 'valid':
        return read_Q_GT(valid_query_file, valid_gt_file, N, args.valid_size, g, node_id_map, return_list = return_list)
    elif return_query == 'test':
        return read_Q_GT(test_query_file, test_gt_file, N, args.test_size, g, node_id_map, return_list = return_list)


# compute normalized core number
def initialize_feature(graphx, max_id, root = None, write = 0):
    core_numbers = nx.core_number(graphx)
    core_values = [core_numbers[node] if node in core_numbers else 0 for node in range(max_id)] # some vertices do not have neighbor, thus not in graphx
    core_values = torch.tensor(core_values, dtype=torch.float32).view(-1, 1)
    scaler = MinMaxScaler()
    normalized_core_values = scaler.fit_transform(core_values).astype(np.float32)

    # write features
    if write == 1:
        feature_file = root + "/feat"
        with open(feature_file, "w") as out_file:
            for idx, value in enumerate(normalized_core_values):
                out_file.write(f"{idx}  {value[0]}\n")

    return normalized_core_values