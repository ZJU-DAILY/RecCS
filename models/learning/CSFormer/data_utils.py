import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import random
import time
import pymetis as metis
import json

def rand_train_test_idx(label, train_num):
    """ randomly splits label into train/valid/test splits """
    print("Start random split:")
    label = torch.Tensor(label)
    labeled_nodes = torch.where(label != -1)[0]  # get labeled Node

    n = labeled_nodes.shape[0]
    assert train_num <= n
    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]

    train_idx = labeled_nodes[train_indices]

    return train_idx

def rand_train_test_idx_(label, args, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if not args.use_exist_data_splits:
        print("Start random split:")
        label = torch.Tensor(label)
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]  # get labeled Node
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]-1
        train_num = int(n * args.train_prop)
        valid_num = int(n * args.valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]
        
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/random_train_idx.npy", train_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/random_valid_idx.npy", valid_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/random_test_idx.npy", test_idx)
        
    else:
        print("Loading from exist parts")
        train_idx = torch.from_numpy(np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/random_train_idx.npy")).int()
        valid_idx = torch.from_numpy(np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/random_valid_idx.npy")).int()
        test_idx = torch.from_numpy(np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/random_test_idx.npy")).int()

    return train_idx, valid_idx, test_idx

def partition_splits(label,  graph, args, load_exist_graphfile=True, graphFilePath=None):
    print("Start partition split:")
    if not args.use_exist_data_splits:
        parts_num = args.partition_num
        if not load_exist_graphfile:
            edges_np = graph['edge_index'].T
            edges = edges_np.tolist()
            nodes_num = graph['num_nodes']
            graph_file = np.empty((nodes_num, 0))
            graph_file = graph_file.tolist()
            for edge in edges:
                graph_file[edge[0]].append(edge[1])
            graph_file = np.array(graph_file)
            np.save(graphFilePath+"graph_file.npy", graph_file)
        else:
            nodes_num = graph['num_nodes']
            graph_file = np.load(graphFilePath+"graph_file.npy", allow_pickle=True)

        print("Start metis [graph partition]")
        if not args.Load_exist_parts:
            start_time = time.time()
            (edgecuts, parts) = metis.part_graph(parts_num, adjacency=graph_file)
            end_time = time.time()
            print("[Graph partitioning] time consumption: {:.2f}s".format(end_time - start_time))
            assert len(parts) == nodes_num
            print(len(parts))
            print("End")

            nodes_parts = {}
            for i in range(parts_num):
                nodes_parts[i] = np.argwhere(np.array(parts) == i).ravel()
                nodes_parts[i] = torch.from_numpy(nodes_parts[i]).int()
        else:
            print("Loading from exist parts")
            nodes_parts = np.load(graphFilePath + str(parts_num) + "-parts_dict.npy", allow_pickle=True).item()

        parts_list = list(range(parts_num))
        
        train_num = int(parts_num * args.train_prop)
        valid_num = int(parts_num * args.valid_prop)
        
        train_indices = parts_list[:train_num]
        val_indices = parts_list[train_num:train_num + valid_num]
        test_indices = parts_list[train_num + valid_num:]

        train_idx = np.array([])
        valid_idx = np.array([])
        test_idx = np.array([])
        
        for i in train_indices:
            train_idx = np.concatenate((train_idx, nodes_parts[i].numpy()), axis=0)
        for i in val_indices:
            valid_idx = np.concatenate((valid_idx, nodes_parts[i].numpy()), axis=0)
        for i in test_indices:
            test_idx = np.concatenate((test_idx, nodes_parts[i].numpy()), axis=0)

        trainAndvalid = np.concatenate((train_idx, valid_idx), axis=0)
        random.shuffle(trainAndvalid)
        train_idx = trainAndvalid[:len(train_idx)]
        valid_idx = trainAndvalid[len(train_idx):]
        
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/part_train_idx.npy", train_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/part_valid_idx.npy", valid_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/part_test_idx.npy", test_idx)
        
        train_idx = torch.from_numpy(train_idx).int()
        valid_idx = torch.from_numpy(valid_idx).int()
        test_idx = torch.from_numpy(test_idx).int()
    else:
        train_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/part_train_idx.npy")
        valid_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/part_valid_idx.npy")
        test_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/part_test_idx.npy")
        
        train_idx = torch.from_numpy(train_idx).int()
        valid_idx = torch.from_numpy(valid_idx).int()
        test_idx = torch.from_numpy(test_idx).int()
    return train_idx, valid_idx, test_idx

def partition_splits_avg(label,  graph, args, load_exist_graphfile=True, graphFilePath=None):
    if not args.use_exist_data_splits:
        print("Start partition split(avg):")
        parts_num = args.partition_num
        if not load_exist_graphfile:
            edges_np = graph['edge_index'].T
            edges = edges_np.tolist()
            nodes_num = graph['num_nodes']
            graph_file = np.empty((nodes_num, 0))
            graph_file = graph_file.tolist()
            for edge in edges:
                graph_file[edge[0]].append(edge[1])
            graph_file = np.array(graph_file)
            np.save(graphFilePath+"graph_file.npy", graph_file)
        else:
            nodes_num = graph['num_nodes']
            graph_file = np.load(graphFilePath+"graph_file.npy", allow_pickle=True)

        print("Start metis [graph partition]")
        if not args.Load_exist_parts:
            start_time = time.time()
            (edgecuts, parts) = metis.part_graph(parts_num, adjacency=graph_file)
            end_time = time.time()
            print("[Graph partitioning] time consumption: {:.2f}s".format(end_time - start_time))
            assert len(parts) == nodes_num
            print(len(parts))
            print("End")

            nodes_parts = {}
            for i in range(parts_num):
                nodes_parts[i] = np.argwhere(np.array(parts) == i).ravel()
                nodes_parts[i] = torch.from_numpy(nodes_parts[i]).int()
                # nodes_parts[i].astype(np.int64)
        else:
            print("Loading from exist parts")
            nodes_parts = np.load(graphFilePath + str(parts_num) + "-parts_dict.npy", allow_pickle=True).item()

        parts_list = list(range(parts_num))
        
        train_idx = np.array([])
        valid_idx = np.array([])
        test_idx = np.array([])
        
        for i in parts_list:
            parti_len = len(nodes_parts[i])
            train_num = int(parti_len * args.train_prop)
            valid_num = int(parti_len * args.valid_prop)
            
            train_idx = np.concatenate((train_idx, nodes_parts[i].numpy()[:train_num]), axis=0)
            valid_idx = np.concatenate((valid_idx, nodes_parts[i].numpy()[train_num:train_num + valid_num]), axis=0)
            test_idx = np.concatenate((test_idx, nodes_parts[i].numpy()[train_num + valid_num:]), axis=0)
        
        train_idx = torch.from_numpy(train_idx).int()
        valid_idx = torch.from_numpy(valid_idx).int()
        test_idx = torch.from_numpy(test_idx).int()

        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/partavg_train_idx.npy", train_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/partavg_valid_idx.npy", valid_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/partavg_test_idx.npy", test_idx)
    else:
        print("Loading from exist parts")
        train_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/partavg_train_idx.npy")
        valid_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/partavg_valid_idx.npy")
        test_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/partavg_test_idx.npy")
        
        train_idx = torch.from_numpy(train_idx).int()
        valid_idx = torch.from_numpy(valid_idx).int()
        test_idx = torch.from_numpy(test_idx).int()

    return train_idx, valid_idx, test_idx

def remove_version_splits( graph, args, load_exist_graphfile=True, graphFilePath=None):
    if not args.use_exist_data_splits:
        print("Start random split(remove version):")
        node_index = torch.nonzero(graph['nodes_mask']).squeeze(1)
        n = len(node_index)
        train_num = int(n * args.train_prop)
        valid_num = int(n * args.valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]


        train_idx = node_index[train_indices]
        valid_idx = node_index[val_indices]
        test_idx = node_index[test_indices]
        
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/(remove version)random_train_idx.npy", train_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/(remove version)random_valid_idx.npy", valid_idx)
        np.save(args.work_path + "/data_splits/"+ args.dataset_name +"/(remove version)random_test_idx.npy", test_idx)
        print("Finish random split(remove version)")
    else:
        print("Loading from exist parts")
        train_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/(remove version)random_train_idx.npy")
        valid_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/(remove version)random_valid_idx.npy")
        test_idx = np.load(args.work_path + "/data_splits/"+ args.dataset_name +"/(remove version)random_test_idx.npy")
        print("Finish random split(remove version)")
    return train_idx, valid_idx, test_idx

def class_rand_splits(label, label_num_per_class):
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    valid_num, test_num = 500, 1000
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:valid_num + test_num]
    
    return train_idx, valid_idx, test_idx



def get_neigbors(graph, source_node, depth=1):
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(graph["edge_index"].permute(1, 0).tolist())
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    output = {}
    layers = dict(nx.bfs_successors(nx_graph, source=source_node, depth_limit=depth))
    nodes = [source_node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
        
    neighbors = []
    for i in output.keys():
        neighbors.extend(output[i])
    return neighbors


def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j


def re_features(adj, features, K):
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]
    x = features + torch.zeros_like(features)
    for i in range(K):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]        
    nodes_features = nodes_features.squeeze()
    # print(nodes_features)
    # print(nodes_features.shape)
    return nodes_features

def sample_node_subgraph(centra_node, hop_num, graph_edge_index):
    subgraph, edge_index, mapping, edge_mask = k_hop_subgraph(
        centra_node, hop_num, graph_edge_index, relabel_nodes=False)
    return subgraph, edge_index

def sample_batch_indices(node):
    pass


if __name__ == "__main__":
    # edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    #                             [2, 2, 4, 4, 6, 6]])
    # subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    #     6, 1, edge_index, relabel_nodes=False)
    # print(subset)
    # # tensor([2, 3, 4, 5, 6])
    # print(edge_index)
    # # tensor([[0, 1, 2, 3],[2, 2, 4, 4]])
    # print(mapping)
    # # tensor([4])
    # print(edge_mask)
    # # tensor([False, False,  True,  True,  True,  True])
    # print(subset[mapping])
    
    import matplotlib.pyplot as plt
    import networkx as nx
    import pymetis as metis
    from ogb.nodeproppred import NodePropPredDataset
    from dataset import Dataset
    import time
    
    
    data_dir = "/home/Hanano/XXXX-4/LGCS/dataset/amazon2m"
    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=f'{data_dir}/ogb')
    dataset = Dataset('amazon2m')
    dataset.graph = ogb_dataset.graph

    edges_np = dataset.graph['edge_index'].T
    edges = edges_np.tolist()

    G = nx.Graph()
    start_time = time.time()
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    end_time = time.time()
    print("NX graph construction time: {:.2f}s".format(end_time - start_time))

    print("Start metis graph partition")
    start_time = time.time()
    (edgecuts, parts) = metis.part_graph(G, 64)
    end_time = time.time()
    print("Graph partitioning time: {:.2f}s".format(end_time - start_time))
    print("End")
    print("Done")

    # G = nx.Graph()
    # edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    #                             [2, 2, 4, 4, 6, 6]])
    # edge_index = edge_index.permute(1, 0)
    # edge_index = edge_index.tolist()
    # G.add_edges_from(edge_index)

    # G = metis.example_networkx()
    # (edgecuts, parts) = metis.part_graph(G, 4)
    # print("Done")
    # colors = ['red', 'blue', 'pink', 'yellow']
    # color_map = []
    # for i, p in enumerate(parts):
    #     color_map.append(colors[p])
    # print(edgecuts)
    # print(parts)
    # nx.draw_networkx(G, node_color=color_map)
    # plt.savefig('./test.jpg')

    pass