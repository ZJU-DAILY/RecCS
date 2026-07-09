import numpy as np
import networkx as nx
import scipy.sparse as sp
# from .data_utils import rand_train_test_idx, partition_splits, partition_splits_avg, remove_version_splits
import torch
import time
from tqdm import tqdm
import pymetis as metis
from torch_geometric.utils import subgraph, remove_self_loops
import math

class Dataset(object):
    def __init__(self):
        self.graph = {}
        self.label = None

    def indexSpliteByPartition(self):
        pass
    
    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def get_dataset(G, args, hindex = None, max_coreness = None):
    # get graph
    num_nodes = G.number_of_nodes()
    edges_undirect = np.array(list(G.edges()), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges_undirect.shape[0]), (edges_undirect[:, 0], edges_undirect[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    edges_list = edges_undirect.tolist()
    graph_dict = get_graph_dict(edges_list, num_nodes)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_coo = adj.tocoo()
    edge_index = torch.from_numpy(np.vstack([adj_coo.row, adj_coo.col])).long()
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    dataset = Dataset()
    dataset.graph = {'edge_index': edge_index,
                     'adj': adj,
                     'num_nodes': num_nodes}

    # get feature
    coreness_dict = calculateGraphCoreness(G)
    # dataset.graph["max_coreness"] = max(list(coreness_dict.values()))
    dataset.graph["coreness"] = coreness_dict
    if hindex is None:
        full_hindex_max = get_max_hindex(dataset, graph_dict)
        dataset.graph["full_hindex_max"] = full_hindex_max
    else:
        dataset.graph["full_hindex_max"] = hindex
    feature = getHindexVector(dataset, args, graph_dict=graph_dict)
    dataset.graph['node_feat'] = torch.as_tensor(feature).float()

    # get label
    coreness_values = list(set(list(dataset.graph["coreness"].values())))
    coreness_values.sort()
    if max_coreness is None:
        max_coreness = max(coreness_values)
    dataset.graph["max_coreness"] = max_coreness
    coreness_list = np.array([coreness_dict[i] for i in range(dataset.graph['num_nodes'])])
    dataset.label = torch.tensor(coreness_list, dtype=torch.long)

    return dataset

def getHindexVector(dataset, args, graph_dict=None):
    def hIndex(citations):
        citations.sort(reverse = True)
        for i, j in enumerate(citations):
            if i+1 > j:
                return i
        return len(citations)

    edges = dataset.graph['edge_index'].T
    edges = edges.tolist()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    degrees_dict = dict(nx.degree(graph))
    nodes_num = dataset.graph['num_nodes']
    isolate_nodes = set(range(nodes_num)) - set(degrees_dict.keys())
    for node in isolate_nodes:
        degrees_dict[node] = 0
    dataset.graph["degree"] = degrees_dict
    degrees_np = np.array([degrees_dict[i] for i in range(nodes_num)])

    # get all nodes h-Index
    hIndex_dict = {}
    for i in tqdm(range(nodes_num)):
        # graph_dict[i] = list(map(coreness_dict.get, graph_dict[i]))
        citations = np.array(degrees_np)[graph_dict[i][1:]]
        hIndex_dict[i] = hIndex(citations.tolist())
    dataset.graph["hIndex_dict"] = hIndex_dict
    hIndex_np = np.array([hIndex_dict[i] for i in range(nodes_num)])

    # high order h-index
    if args.h_index_order == 1:
        pass
    else:
        for i in range(args.h_index_order-1):
            tmp_index_dict = hIndex_dict
            tmp_index_dict_np = np.array([tmp_index_dict[i] for i in range(nodes_num)])
            new_hIndex_dict = {}
            for i in tqdm(range(nodes_num)):
                # graph_dict[i] = list(map(coreness_dict.get, graph_dict[i]))
                citations = np.array(tmp_index_dict_np)[graph_dict[i][1:]]
                new_hIndex_dict[i] = hIndex(citations.tolist())
            hIndex_dict = new_hIndex_dict

    max_hindex =  dataset.graph["full_hindex_max"]

    feature = np.zeros((nodes_num, max_hindex+1))
    for i in range(nodes_num):
        graph_dict[i] = np.array(hIndex_np)[graph_dict[i]]
    del hIndex_np
    for i in tqdm(range(feature.shape[0])):
        hIndex_val = graph_dict[i]
        for j in hIndex_val:
            feature[i][j] += 1
        tmp_sum = np.sum(feature[i])
        if tmp_sum != 0:
            feature[i] = feature[i] / tmp_sum
    feature = torch.as_tensor(feature).float()
    return feature

def get_max_hindex(dataset, graph_dict):
    def hIndex(citations):
        citations.sort(reverse=True)
        for i, j in enumerate(citations):
            if i + 1 > j:
                return i
        return len(citations)

    edges = dataset.graph['edge_index'].T
    edges = edges.tolist()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    degrees_dict = dict(nx.degree(graph))
    nodes_num = dataset.graph['num_nodes']
    isolate_nodes = set(range(nodes_num)) - set(degrees_dict.keys())
    for node in isolate_nodes:
        degrees_dict[node] = 0
    dataset.graph["degree"] = degrees_dict
    degrees_np = np.array([degrees_dict[i] for i in range(nodes_num)])

    # get all nodes h-Index
    hIndex_dict = {}
    for i in range(nodes_num):
        # graph_dict[i] = list(map(coreness_dict.get, graph_dict[i]))
        citations = np.array(degrees_np)[graph_dict[i]]
        hIndex_dict[i] = hIndex(citations.tolist())
    dataset.graph["hIndex_dict"] = hIndex_dict

    hIndex_np = np.array([hIndex_dict[i] for i in range(nodes_num)])
    max_hindex = math.ceil(max(hIndex_np) / 10) * 10
    return max_hindex

def calculateGraphCoreness(G):
    start = time.time()
    coreness_dict = nx.algorithms.core.core_number(G)
    end = time.time()
    print(f"compute coreness time: {end - start}")
    for nd in G.nodes():
        if G.degree[nd] == 0:
            coreness_dict[nd] = 0
    return coreness_dict

def normalize(mx): 
    """
    Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    # adj = adj.to_dense().cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    n_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return n_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


# a->b and b->a
def get_graph_dict(edges, num):
    graph_dict = {}
    for i in range(num):
        graph_dict[i] = [i]
        # graph_dict[i] = []
    for edge in edges:
        a, b = edge
        # if a not in graph_dict:
        #     graph_dict[a] = []
        # if b not in graph_dict:
        #     graph_dict[b] = []
        graph_dict[a].append(b)
        graph_dict[b].append(a)
    return graph_dict
