import csv
import os
import random
import tracemalloc
import time
from types import NoneType
import metis
import torch
import torch as th
import numpy as np
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
from memory_count import FunctionPerformanceMonitor
from .model_GCN import ConRC
import copy
from utils import *

def select_pos(queries, ground_truth, pos_size):
    data_lists = []
    for i in range(len(queries)):
        q = queries[i][0]
        query_nodes = set()
        query_nodes.add(q)
        comm = ground_truth[i]
        gt_nodes = comm
        non_query_nodes = list(gt_nodes - query_nodes)
        if pos_size == -1:
            pos = non_query_nodes
        else:
            pos = random.sample(non_query_nodes, min(pos_size, len(non_query_nodes)))
        data_lists.append((q, pos, comm))
    return data_lists

def metis_clustering(graph, cluster_number):
    (st, parts) = metis.part_graph(graph, cluster_number)
    clusters = list(set(parts))
    cluster_membership = {}
    for i, node in enumerate(graph.nodes):
        cluster = parts[i]
        cluster_membership[node] = cluster
    return clusters, cluster_membership

def unified_load_data(args):
    # STEP1: get graph
    feature, G, node_id_map, N = read_feat_graph(args)
    feature = torch.from_numpy(feature)
    node_in_dim = feature.shape[1]
    n_nodes = feature.shape[0]

    # STEP2: get clusters
    clusters, cluster_membership = metis_clustering(G, args.cluster_number)

    # STEP3: get adj
    nodes_adj = {}
    for id1, id2 in G.edges:
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)

    # cluster: [vertices in cluster]
    sg_nodes = {}
    for u, c in cluster_membership.items():
        if c not in sg_nodes:
            sg_nodes[c] = []
        sg_nodes[c].append(u)


    return feature, node_in_dim, n_nodes, G, sg_nodes, clusters, cluster_membership, nodes_adj, node_id_map, N

class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def hypergraph_construction(edge_index, num_nodes, k=1):
    if k == 1:
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    else:
        neighbor_augment = TwoHopNeighbor()
        hop_data = Data(edge_index=edge_index, edge_attr=None)
        hop_data.num_nodes = num_nodes
        for _ in range(k - 1):
            hop_data = neighbor_augment(hop_data)
        hop_edge_index = hop_data.edge_index
        hop_edge_attr = hop_data.edge_attr
        edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    return edge_index, edge_attr

def decompose_train(g, train, nodes_feats, n_nodes, cluster_membership,
                    sg_nodes, k, nodes_adj):
    train_lists = []
    for q, pos, com in train:
        # obtain the cluster of q
        nodeslists = sg_nodes[cluster_membership[q]]

        # find the bound of each cluster
        nodelistb = []
        if q in nodes_adj:
            neighbor = nodes_adj[q]
            nodelistb = [x for x in neighbor if x not in nodeslists]
        nodelistb = list(set(nodelistb))
        nodeslists = nodeslists+nodelistb

        # obtain the features of node in cluster and its bound
        mask = [False]*n_nodes
        for u in nodeslists:
            mask[u] = True
        feats = nodes_feats[mask]

        # map the vertex id
        nodeslists = sorted(nodeslists)
        nodes_ = {}
        for i, u in enumerate(nodeslists):
            nodes_[u]=i
        sub = g.subgraph(nodeslists)
        src = []
        dst = []
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]
            id2_ = nodes_[id2]
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst])
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k = k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]

        # STEP6: process positive samples: some positive samples may not in the cluster
        pos_ = [nodes_[x] for x in pos if x in nodeslists]
        train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))

    return train_lists

def COCLEP_parameter(input_args):
    args = copy.deepcopy(input_args)
    args.batch_size = 64
    args.hidden_dim = 256
    args.num_layers = 3
    args.drop_out = 0.1
    args.lr = 0.001
    args.weight_decay = 0.0005
    args.tau = 0.2
    args.pos_size = -1 # If pos_size == -1, all the vertices in the community are positive samples.
    args.k = 1
    args.min_thr = 0
    args.max_thr = 0.95
    args.thr_step = 0.05
    alpha_list = {
        "citeseer": 0.0001,
        "cora": 0.001,
        "photo": 0.2,
        "cs": 0.2,
        "Physics": 0.2,
        "reddit": 0.2,
        "amazon": 0.2,
        "dblp_snap": 0.2,
        "youtube": 100,
        "livejournal": 0.2,
        "orkut": 0.2,
        "email": 0.2
    }
    lam_list = {
        "citeseer": 0.0001,
        "cora": 0.001,
        "photo": 0.2,
        "cs": 0.2,
        "Physics": 0.2,
        "reddit": 0.2,
        "amazon": 0.2,
        "dblp_snap": 0.2,
        "youtube": 100,
        "livejournal": 0.2,
        "orkut": 0.2,
        "email": 0.2
    }
    cluster_list = {
        "citeseer": 3,
        "cora": 3,
        "photo": 5,
        "cs": 5,
        "Physics": 10,
        "reddit": 1000,
        "amazon": 100,
        "dblp_snap": 100,
        "youtube": 300,
        "livejournal": 1000,
        "orkut": 2000,
        "email": 3
    }
    args.alpha = alpha_list[args.dataset]
    args.lam = lam_list[args.dataset]
    args.cluster_number = cluster_list[args.dataset]
    if args.exp_mod == 5 and args.fix:
        args.save_model = args.model_path + "/" + args.dataset + f"_COCLEP_1_0.pth"  # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_COCLEP_log_1_0"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_COCLEP_thr_1_0"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_COCLEP_{args.exp_mod}_{args.exp_param}_fix" # the file to save result
    else:
        args.save_model = args.model_path + "/" + args.dataset + f"_COCLEP_{args.exp_mod}_{args.exp_param}.pth" # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_COCLEP_log_{args.exp_mod}_{args.exp_param}"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_COCLEP_thr_{args.exp_mod}_{args.exp_param}"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_COCLEP_{args.exp_mod}_{args.exp_param}" # the file to save result
    args.save_processData = args.process_path + "/" + args.dataset + f"_COCLEP_{args.exp_mod}_{args.exp_param}.pth"  # the file to save DataLoader of model
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_COCLEP_recLabel" # the file to save recommendation label
    return args

def COCLEP_RecGenerate(args, model_queries, model_gts, model, device):
    # Pack data
    nodes_feats, node_in_dim, n_nodes, g, \
        sg_nodes, clusters, cluster_membership, nodes_adj, node_id_map, N = unified_load_data(args)

    # load model
    if next(model.parameters()).device != device:
            model = model.to(device)
    model.eval()
    pred_com = []
    labels = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    nodes_feats = nodes_feats.to(device)
    n_nodes = nodes_feats.shape[0]
    model_thrs = []
    str_queries = []
    single_queries = single_query(model_queries)
    for i, q in enumerate(single_queries):
        # Generate threshold values using numpy to avoid floating point precision issues
        thr_values = np.arange(args.min_thr, args.max_thr + args.thr_step/2, args.thr_step)
        # Round to avoid floating point precision issues
        thr_values = np.round(thr_values, decimals=2)
        
        for thr in thr_values:
            t1 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            monitor = FunctionPerformanceMonitor()
            metrics = monitor.monitor_function(community_search, args, q, model, device, nodes_feats, g, n_nodes, nodes_adj, sg_nodes, cluster_membership, thr)
            community = metrics['result']
            cur_cpu_memory = metrics['max_memory_mb']
            cur_gpu_memory = get_gpu_memory()
            cpu_memory.append(cur_cpu_memory)
            gpu_memory.append(cur_gpu_memory)
            pred_com.append(community)
            str_queries.append(g.nodes[q]['old_id'])
            labels.append(model_gts[i])
            t2 = time.time()
            all_t.append(t2 - t1)
            model_thrs.append(thr)
    return g, pred_com, labels, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def COCLEP_RecQuery(args, model_queries, model_gts, model, device, best_thr = None):
    # Pack data
    nodes_feats, node_in_dim, n_nodes, g, \
        sg_nodes, clusters, cluster_membership, nodes_adj, node_id_map, N = unified_load_data(args)

    # load model
    if next(model.parameters()).device != device:
            model = model.to(device)
    model.eval()
    pred_com = []
    queries = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    nodes_feats = nodes_feats.to(device)
    n_nodes = nodes_feats.shape[0]
    single_queries = single_query(model_queries)
    for i, q in enumerate(single_queries):
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        queries.append({q})
        metrics = monitor.monitor_function(community_search, args, q, model, device, nodes_feats, g, n_nodes, nodes_adj, sg_nodes, cluster_membership, best_thr[i])
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        pred_com.append(community)
        t2 = time.time()
        all_t.append(t2 - t1)
    str_queries = ['_'.join(str(g.nodes[node]['old_id']) for node in query) for query in queries]
    return pred_com, str_queries, all_t, cpu_memory, gpu_memory

def COCLEP_DataLoader(args, phase = 'all'):
    # Package data
    nodes_feats, node_in_dim, n_nodes, g, \
        sg_nodes, clusters, cluster_membership, nodes_adj, node_id_map, N = unified_load_data(args)

    trainloader, validloader, testloader = None, None, None
    if phase == 'all' or phase == 'train':
        # split data
        train_cur_in, train_cur_out = split_data(args, N, node_id_map, g, return_list = True, return_query = 'train')
        valid_cur_in, valid_cur_out = split_data(args, N, node_id_map, g, return_list = True, return_query = 'valid')
        train = select_pos(train_cur_in, train_cur_out, args.pos_size)
        val = select_pos(valid_cur_in, valid_cur_out, args.pos_size)

        # Handling training sets affected by partitioning
        partited_train = decompose_train(g, train, nodes_feats, n_nodes, cluster_membership,
                                                sg_nodes, args.k, nodes_adj)

        partited_valid = decompose_train(g, val, nodes_feats, n_nodes, cluster_membership,
                                                sg_nodes, args.k, nodes_adj)

        trainloader = (partited_train, partited_valid, node_in_dim)
        validloader = (val, cluster_membership, sg_nodes, nodes_feats, g, nodes_adj)
    if phase == 'all' or phase == 'test':
        test_cur_in, test_cur_out = split_data(args, N, node_id_map, g, return_list = True, return_query = 'test')
        testloader = (single_query(test_cur_in), test_cur_out, cluster_membership, sg_nodes, nodes_feats, g, nodes_adj)
    return trainloader, validloader, testloader

def eval_loss(valid, model):
    model.eval()
    epoch_loss = 0.0
    i = 0
    with torch.no_grad():
        for q, pos, edge_index, edge_index_aug, feats in valid:
            device = next(model.parameters()).device
            edge_index = edge_index.to(device)
            edge_index_aug = edge_index_aug.to(device)
            feats = feats.to(device)
            if len(pos) == 0:
                # print(f"The positive samples {q} is null")
                i = i + 1
                continue
            loss = model.compute_loss((q, pos, edge_index, edge_index_aug, feats), valid = True)
            epoch_loss = epoch_loss + loss.item()
            i = i + 1
    return epoch_loss

def train_one_epoch(train, model, batch_size, optimizer):
    model.train()
    epoch_loss = 0.0
    start = time.time()
    i = 0
    for q, pos, edge_index, edge_index_aug, feats in train:
        device = next(model.parameters()).device
        edge_index = edge_index.to(device)
        edge_index_aug = edge_index_aug.to(device)
        feats = feats.to(device)
        if len(pos) == 0:
            # print(f"The positive samples {q} is null")
            i = i + 1
            continue
        loss = model((q, pos, edge_index, edge_index_aug, feats))
        epoch_loss = epoch_loss + loss.item()
        loss.backward()
        if (i + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        i = i + 1
    if i % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()
    end = time.time()
    epoch_time = end - start
    return epoch_loss, epoch_time

def save_model_complete(model, args, trainloader=None):
    node_in_dim = None
    if trainloader is not None:
        _, _, node_in_dim = trainloader
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_in_dim': node_in_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'drop_out': args.drop_out,
            'tau': args.tau,
            'alpha': args.alpha,
            'lam': args.lam,
            'k': args.k,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    }
    torch.save(model_info, args.save_model)
    print(f"Model saved to {args.save_model}")

def load_model_complete(args, device):
    if not os.path.exists(args.save_model):
        print(f"Model file {args.save_model} not found")
        return None
    
    try:
        model_info = torch.load(args.save_model, map_location=device)
        
        model_config = model_info['model_config']
        
        model = ConRC(
            node_in_dim=model_config['node_in_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['drop_out'],
            tau=model_config['tau'],
            device=device,
            alpha=model_config['alpha'],
            lam=model_config['lam'],
            k=model_config['k']
        )
        
        model.load_state_dict(model_info['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from {args.save_model}")
        print(f"Model config: {model_config}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_pretrained_model(args, device):
    print("Loading pre-trained model...")
    model = load_model_complete(args, device)
    if model is not None:
        print("Pre-trained model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"No pre-trained model found at {args.save_model}")

def COCLEP_train(args, trainloader, device):
    ini_time = time.time()
    
    if trainloader is None:
        print("Empty trainloader provided, attempting to load pre-trained model...")
        model = load_model_complete(args, device)
        if model is not None:
            print("Loaded pre-trained model successfully")
            return model, 0, 0
        else:
            raise ValueError("No pre-trained model found and trainloader is empty. Cannot proceed.")
    
    train_ori, valid_ori, node_in_dim = trainloader
    
    model = load_model_complete(args, device)
    if model is not None:
        print("Loaded pre-trained model, skipping training")
        return model, 0, 0
    
    model = ConRC(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,
                  device, args.alpha, args.lam, args.k)
    max_cpu_memory = 0
    max_gpu_memory = 0
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    optimizer.zero_grad()
    print("starting training...")
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epoch)
    early_stopping = EarlyStopping(model, **stopping_args)
    all_loss = []
    
    save_model_complete(model, args, trainloader)
    
    if args.train_size == 0:
        print("train set is null")
        model = load_model_complete(args, device)
        return model, max_cpu_memory, max_gpu_memory
    best_loss = float('inf')
    best_model_state = None
    for epoch in range(args.epoch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(train_one_epoch,train_ori, model, args.batch_size, optimizer)
        train_loss, epoch_time = metrics['result']
        valid_loss = eval_loss(valid_ori, model)
        all_loss.append(valid_loss)
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
        max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
        # save model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_state = model.state_dict().copy()
            save_model_complete(model, args, trainloader)
        avg_loss = float('inf') if len(train_ori) == 0 else train_loss / len(train_ori)
        avg_valid_loss = float('inf') if len(valid_ori) == 0 else valid_loss / len(valid_ori)
        # record information
        sum_time = time.time() - ini_time
        write_train_info(args.save_trainInfo, epoch, epoch_time, sum_time, cur_cpu_memory, cur_gpu_memory, avg_loss, avg_valid_loss)
        if (epoch + 1) % args.save_per_epoch == 0:
            print(f"Epoch [{epoch + 1}/{args.epoch}], Time: {epoch_time:.6f}s, "
                  f"CPU Memory: {cur_cpu_memory:.6f} MB, "
                  f"GPU Memory: {cur_gpu_memory:.6f} MB, "
                  f"Train Loss: {avg_loss:.6f}, "
                  f"Valid Loss: {avg_valid_loss:.6f}")
        # early stop
        if args.early_stopping != 0 and early_stopping.simple_check(all_loss):
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_loss:.6f}")
    
    return model, max_cpu_memory, max_gpu_memory

def search(args, q, model ,device, nodes_feats, g, n_nodes, nodeslists):
    mask = [False] * n_nodes
    for u in nodeslists:
        mask[u] = True
    feats = nodes_feats[mask]
    nodeslists = sorted(nodeslists)
    nodes_ = {}
    nodes_reverse = {}
    for i, u in enumerate(nodeslists):
        nodes_[u] = i
        nodes_reverse[i] = u
    sub = g.subgraph(nodeslists)
    src = []
    dst = []
    for id1, id2 in sub.edges:
        id1_ = nodes_[id1]
        id2_ = nodes_[id2]
        src.append(id1_)
        dst.append(id2_)
        src.append(id2_)
        dst.append(id1_)
    edge_index = torch.tensor([src, dst]).to(device)
    edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=args.k)
    edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]
    with torch.no_grad():
        h = model((nodes_[q], None, edge_index, edge_index_aug, feats))
    numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
    norm = torch.norm(h, dim=-1, keepdim=True)
    denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
    sim = numerator / denominator
    simlists = torch.sigmoid(sim.squeeze(0)).to(
        torch.device('cpu')).numpy().tolist()
    sim_map = {}
    for i, score in enumerate(simlists):
        sim_map[nodes_reverse[i]] = score
    return sim_map, h[nodes_[q]]

def COCLEP_valid(args, model, validloader, device):
    if os.path.exists(args.save_thr):
        with open(args.save_thr, "r") as file:
            best_thr = float(file.read().strip())
            return best_thr
    else:
        if next(model.parameters()).device != device:
            model = model.to(device)
    model.eval()
    val, cluster_membership, sg_nodes, nodes_feats, g, nodes_adj = validloader
    scorelists = []
    queries = []
    labels = []
    n_nodes = nodes_feats.shape[0]
    nodes_feats = nodes_feats.to(device)
    with torch.no_grad():
        for q, pos, comm in val:
            nodeslists = sg_nodes[cluster_membership[q]]
            nodelistb = []
            if q in nodes_adj:
                neighbor = nodes_adj[q]
                nodelistb = [x for x in neighbor if x not in nodeslists]
            nodeslists = nodeslists + nodelistb
            sim_map,_ = search(args, q, model ,device, nodes_feats, g, n_nodes, nodeslists)
            queries.append({q})
            labels.append(comm)
            scorelists.append(sim_map)

    # Generate threshold values using numpy to avoid floating point precision issues
    thr_values = np.arange(args.min_thr, args.max_thr + args.thr_step/2, args.thr_step)
    # Round to avoid floating point precision issues
    thr_values = np.round(thr_values, decimals=2)
    
    best_f1 = 0.0
    best_thr = thr_values[0]
    
    for s_ in thr_values:
        communities = []
        for sim_map in scorelists:
            community = set()
            for node, sim in sim_map.items():
                if sim >= s_:
                    community.add(node)
            communities.append(community)
        pre, rec, f1, jac = evaluate_gt_communities(communities, labels)
        avg_f1 = np.mean(f1)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thr = s_
    print(f"best f1 {best_f1}, best_thr {best_thr}")
    with open(args.save_thr, "w") as file:
        file.write(f"{best_thr:.2f}")
    return best_thr

def community_search(args, q, model, device, nodes_feats, g, n_nodes, nodes_adj, sg_nodes, cluster_membership, best_thr):
    nodeslists = sg_nodes[cluster_membership[q]]
    nodelistb = []
    if q in nodes_adj:
        neighbor = nodes_adj[q]
        nodelistb = [x for x in neighbor if x not in nodeslists]
    nodeslists = nodeslists + nodelistb
    sim_map, hq = search(args, q, model, device, nodes_feats, g, n_nodes, nodeslists)
    community = set()
    for node, sim in sim_map.items():
        if sim >= best_thr:
            community.add(node)

    # expand community
    cluster_set = set()
    for qb in nodelistb:
        if cluster_membership[qb] in cluster_set:
            continue
        cluster_set.add(cluster_membership[qb])
        if sim_map[qb] < best_thr:
            continue
        nodeslists_ = sg_nodes[cluster_membership[qb]]

        nodelistb_ = []
        if qb in nodes_adj:
            neighbor_ = nodes_adj[qb]
            nodelistb_ = [x for x in neighbor_ if x not in nodeslists_]
        nodelistb_ = list(set(nodelistb_))
        nodeslists_ = nodeslists_ + nodelistb_
        qb = q

        mask_ = [False] * n_nodes
        for u in nodeslists_:
            mask_[u] = True
        feats = nodes_feats[mask_]
        nodeslists_ = sorted(nodeslists_)
        nodes__ = {}
        nodes_reverse__ = {}
        for i, u in enumerate(nodeslists_):
            nodes__[u] = i
            nodes_reverse__[i] = u
        sub_ = g.subgraph(nodeslists_)
        src_ = []
        dst_ = []
        for id1_, id2_ in sub_.edges:
            id1__ = nodes__[id1_]
            id2__ = nodes__[id2_]
            src_.append(id1__)
            dst_.append(id2__)
            src_.append(id2__)
            dst_.append(id1__)
        edge_index_ = torch.tensor([src_, dst_]).to(device)
        edge_index_aug_, egde_attr_ = hypergraph_construction(edge_index_, len(nodeslists_), k=args.k)
        edge_index_ = add_remaining_self_loops(edge_index_)[0]
        with torch.no_grad():
            h_ = model((nodes__[qb], None, edge_index_, edge_index_aug_, feats))
        h_[nodes__[qb]] = h_[nodes__[qb]] + hq

        numerator_ = torch.mm(h_[nodes__[qb]].unsqueeze(0), h_.t())
        norm_ = torch.norm(h_, dim=-1, keepdim=True)
        denominator_ = torch.mm(norm_[nodes__[qb]].unsqueeze(0), norm_.t())
        sim_ = numerator_ / denominator_
        simlists_ = torch.sigmoid(sim_.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()

        for i, score in enumerate(simlists_):
            if score >= best_thr and nodeslists_[i] not in community:
                community.add(nodeslists_[i])
    return community

def COCLEP_test(args, model, testloader, best_thr, device):
    model.eval()
    query, gt, cluster_membership, sg_nodes, nodes_feats, g, nodes_adj = testloader
    pred_com = []
    queries = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    nodes_feats = nodes_feats.to(device)
    n_nodes = nodes_feats.shape[0]
    for q in query:
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        queries.append({q})
        metrics = monitor.monitor_function(community_search, args, q, model, device, nodes_feats, g, n_nodes, nodes_adj, sg_nodes, cluster_membership, best_thr)
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        pred_com.append(community)
        t2 = time.time()
        all_t.append(t2 - t1)
    str_queries = ['_'.join(str(g.nodes[node]['old_id']) for node in query) for query in queries]
    return g, pred_com, gt, str_queries, all_t, cpu_memory, gpu_memory












