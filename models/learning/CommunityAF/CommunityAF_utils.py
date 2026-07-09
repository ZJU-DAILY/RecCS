import pathlib
import collections
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
from .graph import Graph


class PretrainComDataset(Dataset):
    def __init__(self, com, train_size):
        self.com = com
        self.n_com = train_size

    def __len__(self):
        return self.n_com

    def __getitem__(self, idx):
        return torch.tensor(idx, dtype=torch.long)

def connected_components(g, nodes):
    remaining = set(nodes)
    ccs = []
    cc = set()
    queue = collections.deque()
    while len(remaining) or len(queue):
        # print(queue, remaining)
        if len(queue) == 0:
            if len(cc):
                ccs.append(cc)
            v = remaining.pop()
            cc = {v}
            queue.extend(set(g.neighbors(v)) & remaining)
            remaining -= {v}
            remaining -= set(g.neighbors(v))
        else:
            v = queue.popleft()
            queue.extend(set(g.neighbors(v)) & remaining)
            cc |= (set(g.neighbors(v)) & remaining) | {v}
            remaining -= set(g.neighbors(v))
    if len(cc):
        ccs.append(cc)
    return ccs

def preprocess_nodefeats(conv, nodefeats, hidden_size=64):
    features = np.vstack(nodefeats)
    sp_feats = sp.csr_matrix(features)
    print(sp_feats.shape)
    convolved_feats = conv(sp_feats)
    N, F = convolved_feats.shape
    if F >= hidden_size:
        svd = TruncatedSVD(n_components=hidden_size, algorithm='arpack')
        x = svd.fit_transform(convolved_feats)
        x = (x - x.mean(0, keepdims=True)) / x.std(0, keepdims=True)
    else:
        dense_feats = convolved_feats.toarray()
        mean = dense_feats.mean(axis=0)
        std = dense_feats.std(axis=0) + 1e-6
        x = (dense_feats - mean) / std
        padding = np.zeros((N, hidden_size - F))
        x = np.hstack([x, padding])
    return x

def myshortest_path(nxg, seed):
    path_map = {}
    vis = [seed]
    path_map[seed] = [seed]
    cnt = 0
    while cnt < len(vis):
        u = vis[cnt]
        cnt += 1
        if len(path_map[u]) > 5:
            continue
        for v in nxg.neighbors(u):
            if v in path_map.keys():
                continue
            else:
                path_map[v] = path_map[u] + [v]
                vis.append(v)
    return path_map

def get_augment(seeds,graph,nxg,conv):
    bs = len(seeds)
    feat = np.zeros((bs, graph.n_nodes), dtype=np.float32)
    for i, seed in enumerate(seeds):
        shortest_path = myshortest_path(nxg, seed)
        len_path = [(key, 1 / len(shortest_path[key])) for key in shortest_path.keys() if len(shortest_path[key]) < 5]
        data = [v for k, v in len_path]
        ind = [k for k, v in len_path]
        feat[i][ind] = data
    return conv(sp.csr_matrix(feat.T)).T


def get_data(idx,args,roll_mapper,pre_dataset):
    result = {
        "batch_neighbor": [],
        "batch_next_onehot": [],
        "batch_z_seed": [],
        "batch_z_node": [],
        "batch_com": [],
        "batch_labels": []
    }
    if args.rankingloss:
        result["batch_neg"] = []
        result["batch_neg_neighbors"] = []
        result["batch_neg_z_seed"] = []
        result["batch_neg_z_node"] = []
        result["batch_neg_z_augment"] = []

    if args.augment:
        result["batch_z_augment"] = []
    for j in idx:
        for i in roll_mapper[j]:
            result["batch_neighbor"].extend(pre_dataset[i]['neighbor'])
            result["batch_next_onehot"].extend(pre_dataset[i]['next_onehot'])
            result["batch_z_seed"].extend(pre_dataset[i]['z_seed'])
            result["batch_z_node"].extend(pre_dataset[i]['z_node'])
            result["batch_com"].extend(pre_dataset[i]['now_com'])
            result["batch_labels"].extend(pre_dataset[i]['label'])
            if args.rankingloss:
                result["batch_neg"].append(pre_dataset[i]['neg'])
                result["batch_neg_neighbors"].append(pre_dataset[i]['neg_neighbors'])
                result["batch_neg_z_seed"].append(pre_dataset[i]['neg_z_seed'])
                result["batch_neg_z_node"].append(pre_dataset[i]['neg_z_node'])
                result["batch_neg_z_augment"].append(pre_dataset[i]['neg_z_augment'])

            if args.augment:
                result["batch_z_augment"].extend(pre_dataset[i]['z_augment'])
    return result

def make_single_node_encoding(new_node, graph,conv):
    bs = len(new_node)
    n_nodes = graph.n_nodes
    ind = np.array([[v, i] for i, v in enumerate(new_node) if v is not None], dtype=np.int64).T
    if len(ind):
        data = np.ones(ind.shape[1], dtype=np.float32)
        x_nodes = conv(sp.csc_matrix((data, ind), shape=[n_nodes, bs])).T
    else:
        x_nodes = conv(sp.csc_matrix((n_nodes, bs), dtype=np.float32)).T
    return x_nodes

def make_nodes_encoding(new_node, graph, conv):
    bs = len(new_node)
    n_nodes = graph.n_nodes
    ind = [[v, i] for i, vs in enumerate(new_node) for v in vs]
    if len(ind):
        ind = np.asarray(ind, dtype=np.int64).T
        data = np.ones(ind.shape[1], dtype=np.float32)
        x_nodes = conv(sp.csc_matrix((data, ind), shape=[n_nodes, bs])).T
    else:
        x_nodes = conv(sp.csc_matrix((n_nodes, bs), dtype=np.float32)).T
    return x_nodes


