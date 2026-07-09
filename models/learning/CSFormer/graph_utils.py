import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import json
import random
import copy
import time
from tqdm import tqdm

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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)     

def calculateAddGraphCoreness(edges, num_nodes, data_dir, args):
    if args.Load_exist_addcoreness:
        print("Load from exists!!!")
        with open(data_dir+'addcoreness_dict.json', 'r') as file:
            coreness_dict = json.load(file)
            coreness_dict = {int(k): coreness_dict[k] for k in coreness_dict}
    else:
        graph = nx.Graph()
        isolate_nodes = list(set(list(range(num_nodes))) - set(np.array(edges).flatten()))
        print("assert start")
        assert len(isolate_nodes) + len(set(np.array(edges).flatten())) == num_nodes
        print("assert End")
        graph.add_edges_from(edges)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        print("start caculate coreness")
        start = time.time()
        coreness_dict = nx.algorithms.core.core_number(graph)
        end = time.time()
        print("End caculate coreness")
        print(f"cost{(end - start)}s")
        if len(isolate_nodes):
            for i in isolate_nodes:
                coreness_dict[i] = 0
        else:
            pass
        with open(data_dir+'addcoreness_dict.json', 'w') as file:
            json.dump(coreness_dict, file)
        # np.save(data_dir+"coreness_dict.npy", coreness_dict)
    return coreness_dict

def add_nodes2graph(dataset, add_num, edge_num, data_dir, args):
    def hIndex(citations):
        citations.sort(reverse = True)
        for i, j in enumerate(citations):
            if i+1 > j:
                return i
        return len(citations)
    
    data_dir = data_dir + "/" + args.dataset_name + "/"
    edges_np = dataset.graph['edge_index'].T
    # nodes = list(range(dataset.graph['num_nodes']))
    nodes = list(set(edges_np.flatten().tolist()))
    
    coreness_choosed = None
    if args.nodes_select_type:
        coreness_types = list(set(list(dataset.graph["coreness"].values())))
        coreness_types.sort(reverse=True)
        corenessMapnodes = {}
        for type in coreness_types:
            corenessMapnodes[type] = [k for k, v in dataset.graph["coreness"].items() if v == type]
        # select_coreness_types = coreness_types[:int(len(coreness_types)/args.coreness_part)]
        # select_coreness_types = coreness_types[4:9]
        select_coreness_types =coreness_types[args.sample_coreness_lower:args.sample_coreness_upper]
        coreness_choosed = select_coreness_types
        candidate_nodes_idx_pool = []
        for i in select_coreness_types:
            candidate_nodes_idx_pool.extend(corenessMapnodes[i])

    edges = edges_np.tolist()
    graph_dict = get_graph_dict(edges, dataset.graph['num_nodes'])
    
    degrees_dict = dataset.graph["degree"]
    # degrees_np = np.array([degrees_dict[i] for i in range(nodes_num)])

    new_nodes = np.array(list(range(add_num))) + dataset.graph['num_nodes']
    new_nodes = new_nodes.tolist()
    change_nodes = []
    change_nodes.extend(new_nodes)
    for new_node in new_nodes:
        # update degree
        degrees_dict[new_node] = edge_num
        if args.nodes_select_type:
            edge_nodes = random.sample(candidate_nodes_idx_pool, edge_num)
        else:
            edge_nodes = random.sample(nodes, edge_num)
        
        change_nodes.extend(edge_nodes)
        for node in edge_nodes:
            degrees_dict[node] = degrees_dict[node]+1

        # update graph
        graph_dict[new_node] = [new_node]
        graph_dict[new_node].extend(edge_nodes)
        for node in edge_nodes:
            graph_dict[node].append(new_node)
        
        # update edges
        for origin_node in edge_nodes:
            edges.append([new_node, origin_node])
            edges.append([origin_node, new_node])
    graph_dict_copy = copy.deepcopy(graph_dict)
    add_edges = np.array(edges)
    dataset.graph["add_edge_index"] = torch.from_numpy(add_edges.T).long()
    newgraph_nodes_num = dataset.graph['num_nodes'] + add_num
    dataset.graph['add_num_nodes'] = newgraph_nodes_num
    degrees_np = np.array([degrees_dict[i] for i in range(newgraph_nodes_num)])
    # get all nodes h-Index
    print("Calculate hIndex[add nodes][Start]")
    hIndex_dict = dataset.graph["hIndex_dict"]
    for i in change_nodes:
        citations = np.array(degrees_np)[graph_dict[i]]
        hIndex_dict[i] = hIndex(citations.tolist())
    print("Calculate hIndex[End]")
    print("Max hIndex", max(list(hIndex_dict.values())))
    hIndex_np = np.array([hIndex_dict[i] for i in range(newgraph_nodes_num)])
    # np.save(dir + "hIndex_np.npy", hIndex_np)
    
    
    print("h-Index vector[add nodes] construct[start]")
    start_time = time.time()
    feature_dim = dataset.graph['node_feat'].shape[1]
    feature_add = torch.from_numpy(np.zeros((len(new_nodes), feature_dim)))
    feature = torch.cat([dataset.graph['node_feat'], feature_add], dim=0)
    feature = feature.numpy()
    for i in new_nodes:
        # graph_dict[i] = list(map(coreness_dict.get, graph_dict[i]))
        graph_dict[i] = np.array(hIndex_np)[graph_dict[i]]
    for i in new_nodes:
        hIndex_val = graph_dict[i]
        for j in hIndex_val:
            if j>(feature_dim-1):
                feature[i][feature_dim-1] += 1
            else:
                feature[i][j] += 1
    norm = np.sum(feature, axis=1, keepdims=True)
    feature = feature/norm
    end_time = time.time()
    print(f"hIndex_vector[add nodes] time consumption: {end_time - start_time:.2f}s")
    # np.save(dir + "hIndex_vector[add nodes].npy", feature)
    feature = torch.as_tensor(feature).float()
    dataset.graph['add_node_feat'] = feature
    
    print("Start construct Adj[add]")
    adj = sp.coo_matrix((np.ones(add_edges.shape[1]), (add_edges[0, :], add_edges[1, :])),
                        shape=(newgraph_nodes_num, newgraph_nodes_num),
                        dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    print("End construct Adj[add]")
    dataset.graph["add_adj"] = adj
    
    coreness_dict= calculateAddGraphCoreness(edges, newgraph_nodes_num, data_dir, args)
    dataset.graph["add_coreness_"] = copy.deepcopy(coreness_dict)
    for i in range(newgraph_nodes_num):
        if coreness_dict[i] > dataset.graph["max_coreness"]:
            coreness_dict[i] = dataset.graph["max_coreness"]
    dataset.graph["add_coreness"] = coreness_dict
    coreness_values = list(set(list(dataset.graph["add_coreness"].values())))
    coreness_values.sort()
    corenessMapindex = {j:i for i,j in enumerate(coreness_values)}
    coreness_list = np.array([corenessMapindex[coreness_dict[i]] for i in range(newgraph_nodes_num)])
    dataset.graph["add_label"] = torch.tensor(coreness_list, dtype=torch.long)
    
    return graph_dict_copy, coreness_choosed

def remove_nodes_from_graph(dataset, args, data_dir, device, remove_type="coreness", prob=0.05):
    all_nodes_idx = list(range(dataset.graph["num_nodes"]))
    test_num = int(dataset.graph["num_nodes"] * prob)
    
    if remove_type == "Random":
        print("Start remove nodes[Random]")
        remove_nodes = random.sample(all_nodes_idx, test_num)
        print("End remove nodes[Random]")
    elif remove_type == "coreness":
        print("Start remove nodes[coreness]")
        if not args.load_exist_remove_info:
            coreness_types = list(set(list(dataset.graph["coreness"].values())))
            coreness_types.sort(reverse=True)
            corenessMapnodes = {}
            for type in coreness_types:
                corenessMapnodes[type] = [k for k, v in dataset.graph["coreness"].items() if v == type]
            select_coreness_types =coreness_types[args.sample_coreness_lower:args.sample_coreness_upper]
            # coreness_choosed = select_coreness_types
            candidate_nodes_idx_pool = []
            for i in select_coreness_types:
                candidate_nodes_idx_pool.extend(corenessMapnodes[i])
            if test_num>len(candidate_nodes_idx_pool):
                test_num = len(candidate_nodes_idx_pool)
            else:
                pass
            remove_nodes = random.sample(candidate_nodes_idx_pool, test_num)
            try:
                remove_nodes.remove(dataset.graph["num_nodes"])
            except ValueError:
                pass
            np.save(data_dir + "remove_nodes"+ str(prob*100) +".npy", remove_nodes)
        else:
            print("saving [remove_nodes.npy]")
            remove_nodes = np.load(data_dir + "remove_nodes"+ str(prob*100) +".npy")
        print("End remove nodes[coreness]")
    if not args.load_exist_remove_info:
        nodes_mask = torch.ones([dataset.graph["num_nodes"]])
        nodes_mask[remove_nodes] = 0
        
        print("Start mask edges:")
        # edge_mask1 = torch.ones([dataset.graph['edge_index'].shape[1]])
        edge_mask = torch.zeros([dataset.graph['edge_index'].shape[1]]).to(device)
        dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
        remove_nodes = torch.tensor(remove_nodes).to(device)
        for node in tqdm(remove_nodes):
            remove_edges_0 = dataset.graph['edge_index'][0] == node
            remove_edges_1 = dataset.graph['edge_index'][1] == node
            remove_edges_0 = remove_edges_0.float()
            remove_edges_1 = remove_edges_1.float()
            edge_mask += remove_edges_0
            edge_mask += remove_edges_1

        # start_time = time.time()
        # edge_mask = torch.zeros([dataset.graph['edge_index'].shape[1]]).to(device)
        # batch_size = 50
        # batch_num = len(remove_nodes)
        # remove_nodes = torch.tensor(remove_nodes).to(device)
        # for i,j in tqdm(enumerate(range(batch_num))):
        #     batch_remove_nodes = remove_nodes[i*batch_size:(batch_size+i*batch_size)]
        #     batch_remove_nodes = torch.tensor(batch_remove_nodes).to(device)
        #     batch_remove_nodes = batch_remove_nodes.unsqueeze(1)
        #     remove_edges_0 = dataset.graph['edge_index'][0] == batch_remove_nodes
        #     remove_edges_1 = dataset.graph['edge_index'][1] == batch_remove_nodes
        #     edge_mask_ = sum(remove_edges_0) + sum(remove_edges_1)
        #     edge_mask += edge_mask_

        edge_mask = (edge_mask == 0).int()
        edge_mask = edge_mask.cpu()
        np.save(data_dir + "edge_mask"+ str(prob*100) +".npy", edge_mask)
        # end_time = time.time()
        # print("Mask edges time consumption: {:.2f}s".format(end_time - start_time))
    
        
        dataset.graph['is_removed'] = True
        dataset.graph['nodes_mask'] = nodes_mask.long()
        remove_nodes = remove_nodes.cpu()
        dataset.graph['edge_index'] = dataset.graph['edge_index'].cpu()
        
    else:
        nodes_mask = torch.ones([dataset.graph["num_nodes"]])
        nodes_mask[remove_nodes] = 0
        dataset.graph['nodes_mask'] = nodes_mask.int()
        edge_mask = np.load(data_dir + "edge_mask"+ str(prob*100) +".npy")
        edge_mask = torch.from_numpy(edge_mask)
    return remove_nodes, nodes_mask, edge_mask