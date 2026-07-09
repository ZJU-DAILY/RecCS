from .data_loader import get_dataset
import time
from .transzero_utils import *
from utils import *
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from .model import PretrainModel
from .lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import copy
import csv
import tracemalloc
from memory_count import FunctionPerformanceMonitor
import networkx as nx

def save_model_complete(model, args, inputloader=None):
    input_dim = None
    if inputloader is not None:
        trainloader, _, _ = inputloader
        input_dim = trainloader.dataset.shape[2]
    
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'hops': args.hops,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'dropout': args.dropout,
            'attention_dropout': args.attention_dropout,
            'readout': args.readout,
            'peak_lr': args.peak_lr,
            'end_lr': args.end_lr,
            'weight_decay': args.weight_decay,
            'warmup_updates': args.warmup_updates,
            'tot_updates': args.tot_updates,
            'epoch': args.epoch
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
        
        temp_args = copy.deepcopy(args)
        temp_args.input_dim = model_config['input_dim']
        temp_args.hidden_dim = model_config['hidden_dim']
        temp_args.hops = model_config['hops']
        temp_args.n_layers = model_config['n_layers']
        temp_args.n_heads = model_config['n_heads']
        temp_args.dropout = model_config['dropout']
        temp_args.attention_dropout = model_config['attention_dropout']
        temp_args.readout = model_config['readout']
        temp_args.peak_lr = model_config['peak_lr']
        temp_args.end_lr = model_config['end_lr']
        temp_args.weight_decay = model_config['weight_decay']
        temp_args.warmup_updates = model_config['warmup_updates']
        temp_args.tot_updates = model_config['tot_updates']
        temp_args.epoch = model_config['epoch']
        
        model = PretrainModel(
            input_dim=model_config['input_dim'],
            config=temp_args
        )
        
        model.load_state_dict(model_info['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from {args.save_model}")
        print(f"Model config: {model_config}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def subgraph_density_controled(candidate_score, graph_score):
    weight_gain = (sum(candidate_score) - sum(graph_score) * (len(candidate_score) ** 1) / (len(graph_score) ** 1)) / (
                len(candidate_score) ** 0.50)
    return weight_gain

# binary search
def GlobalSearch(query_index, graph_score):

    candidates = query_index
    selected_candidate = candidates

    graph_score=np.array(graph_score)
    max2min_index = np.argsort(-graph_score)
    
    startpoint = 0
    endpoint = int(0.50*len(max2min_index))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:
        candidates_left = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]
        candidate_score_left = [graph_score[i] for i in candidates_left]
        candidates_density_left = subgraph_density_controled(candidate_score_left, graph_score)

        candidate_score_half = candidate_score_left+[graph_score[max2min_index[int((startpoint+endpoint)/2)]]]
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

        
        candidate_score_right = candidate_score_half+[graph_score[max2min_index[int((startpoint+endpoint)/2)+1]]]
        candidates_density_right = subgraph_density_controled(candidate_score_right, graph_score)

        if candidates_density_half>candidates_density_left and candidates_density_half>candidates_density_right:
            break
        elif candidates_density_half<candidates_density_right:
            startpoint = int((startpoint+endpoint)/2)+1
        else:
            endpoint = int((startpoint+endpoint)/2)-1
        
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, endpoint)] 
    # selected_candidate = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))] 
    
    return selected_candidate

# no binary
# def GlobalSearch(query_index, graph_score):
#     # candidates = query_index
#     # selected_candidate = candidates

#     graph_score = np.array(graph_score)
#     max2min_index = np.argsort(-graph_score)

#     startpoint = 0
#     endpoint = int(0.50 * len(max2min_index))
#     if endpoint >= 10000:
#         endpoint = 10000

#     while True:
#         candidates_half = query_index + [max2min_index[i] for i in range(0, int((startpoint + endpoint) / 2))]
#         candidate_score_half = [graph_score[i] for i in candidates_half]
#         candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

#         candidates = query_index + [max2min_index[i] for i in range(0, endpoint)]
#         candidate_score = [graph_score[i] for i in candidates]
#         candidates_density = subgraph_density_controled(candidate_score, graph_score)

#         if candidates_density >= candidates_density_half:
#             startpoint = int((startpoint + endpoint) / 2)
#             endpoint = endpoint
#         else:
#             startpoint = startpoint
#             endpoint = int((startpoint + endpoint) / 2)

#         if startpoint == endpoint or startpoint + 1 == endpoint:
#             break

#     selected_candidate = query_index + [max2min_index[i] for i in range(0, startpoint)]

#     return selected_candidate


def TransZero_parameter (input_args):
    args = copy.deepcopy(input_args)
    args.hops = 5 # Hop of neighbors to be calculated
    args.pe_dim = 15 # position embedding size
    args.hidden_dim = 512 # Hidden layer size
    args.ffn_dim = 64 # FFN layer size
    args.n_layers = 1 # Number of Transformer layers
    args.n_heads = 8 # Number of Transformer heads
    args.dropout = 0.1 # Dropout
    args.attention_dropout = 0.1 # Dropout in the attention layer
    args.readout = 'mean'
    args.alpha = 0.1 # the value the balance the loss
    args.batch_size = 4000 # batch size
    args.group_epoch_gap = 20
    args.tot_updates = 1000 # used for optimizer learning rate scheduling
    args.warmup_updates = 400 # warmup steps
    args.peak_lr = 0.001 # peak learning rate
    args.end_lr = 0.0001 # end learning rate
    args.weight_decay = 0.00001 # weight decay

    args.min_thr = -1
    args.max_thr = -1
    args.thr_step = -1

    if args.exp_mod == 5 and args.fix:
        args.save_model = args.model_path + "/" + args.dataset + f"_TransZero_1_0.pth"  # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_TransZero_log_1_0"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_TransZero_thr_1_0"
        args.save_result = args.result_path + "/" + args.dataset + f"_TransZero_{args.exp_mod}_{args.exp_param}_fix"
        args.save_embedding = args.result_path + "/" + args.dataset + f"_TransZero_embd_{args.exp_mod}_{args.exp_param}_fix.npy"
    else:
        args.save_model = args.model_path + "/" + args.dataset + f"_TransZero_{args.exp_mod}_{args.exp_param}.pth"
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_TransZero_log_{args.exp_mod}_{args.exp_param}"
        args.save_thr = args.model_path + "/" + args.dataset + f"_TransZero_thr_{args.exp_mod}_{args.exp_param}"
        args.save_result = args.result_path + "/" + args.dataset + f"_TransZero_{args.exp_mod}_{args.exp_param}"
        args.save_embedding = args.result_path + "/" + args.dataset + f"_TransZero_embd_{args.exp_mod}_{args.exp_param}.npy"
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_TransZero_recLabel" # the file to save recommendation label
    return args

def TransZero_DataLoader(args, phase = 'all'):
    feature, G, node_id_map, N = read_feat_graph(args)
    row = []
    col = []
    for uid, vid in G.edges():
        row.extend([uid, vid])
        col.extend([vid, uid])
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.ones(len(row), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()
    adj_scipy = torch_adj_to_scipy(adj)
    graph = dgl.from_scipy(adj_scipy)
    lpe = laplacian_positional_encoding(graph, args.pe_dim)

    # remap feature according to new id
    features = torch.from_numpy(feature)
    features = torch.cat((features, lpe), dim=1)

    start_feature_processing = time.time()
    processed_features = re_features(adj, features, args.hops)  # return (N, hops+1, d)
    if processed_features.shape[0] < 10000:
        indicator = conductance_hop(adj, args.hops)  # return (N, hops+1)
        indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1])
        processed_features = processed_features * indicator
    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))
    start = time.time()
    print("starting transformer to coo")
    adj = transform_coo_to_csr(adj)
    print("start mini batch processing")
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0])  # transform to coo to support tensor operation
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("adj process time: {:.4f}s".format(time.time() - start))
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle=False)

    processed_train, processed_valid, processed_test = None, None, None
    if phase == 'all' or phase == 'train':
        processed_train = (data_loader,adj_batch, minus_adj_batch)
    if phase == 'all' or phase == 'test':
        test_cur_in, test_cur_out = split_data(args,N,node_id_map, G, return_list = True, return_query = 'test')
        queries = query_to_onehot(test_cur_in, N)
        processed_test = (data_loader, torch.Tensor(queries), test_cur_out, G)
    return processed_train, processed_valid, processed_test

def community_search(single_query, embedding_tensor, G=None):
    # compute query feature
    query_feature = torch.mm(single_query, embedding_tensor)  # (1, embedding_dim)
    query_num = torch.sum(single_query, dim=1)  # (1,)
    query_feature = torch.div(query_feature, query_num.view(-1, 1))  # (1, embedding_dim)

    # compute query similarity
    query_score = cosin_similarity(query_feature, embedding_tensor)  # (1, node_num)
    query_score = torch.nn.functional.normalize(query_score, dim=1, p=1).reshape(-1)  # (1, node_num)

    query_index = torch.nonzero(single_query.squeeze()).reshape(-1).tolist()
    selected_candidates = set(GlobalSearch(query_index, query_score.tolist()))
    
    if G is not None:
        query_set = set(query_index)
        induced_subgraph = G.subgraph(selected_candidates)
        connected_components = list(nx.connected_components(induced_subgraph))
        for component in connected_components:
            if query_set.issubset(component):
                return query_index, component
        return query_index, query_set
    
    return query_index, selected_candidates

def TransZero_RecGenerate(args, input_queries, model_gts, model, device):
    feature, G, node_id_map, N = read_feat_graph(args)
    model_queries = query_to_onehot(input_queries, N)
    row = []
    col = []
    for uid, vid in G.edges():
        row.extend([uid, vid])
        col.extend([vid, uid])
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.ones(len(row), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()
    adj_scipy = torch_adj_to_scipy(adj)
    graph = dgl.from_scipy(adj_scipy)
    lpe = laplacian_positional_encoding(graph, args.pe_dim)

    # remap feature according to new id
    features = torch.from_numpy(feature)
    features = torch.cat((features, lpe), dim=1)
    query = torch.Tensor(model_queries)
    labels = model_gts

    start_feature_processing = time.time()
    processed_features = re_features(adj, features, args.hops)  # return (N, hops+1, d)
    if processed_features.shape[0] < 10000:
        indicator = conductance_hop(adj, args.hops)  # return (N, hops+1)
        indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1])
        processed_features = processed_features * indicator
    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))
    start = time.time()
    print("starting transformer to coo")
    adj = transform_coo_to_csr(adj)
    print("start mini batch processing")
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0])  # transform to coo to support tensor operation
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("adj process time: {:.4f}s".format(time.time() - start))
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle=False)

    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()
    node_embedding = []
    with torch.no_grad():
        for _, item in enumerate(data_loader):
            nodes_features = item.to(device)
            node_tensor, neighbor_tensor = model(nodes_features)
            if len(node_embedding) == 0:
                node_embedding = np.concatenate(
                    (node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                # node_embedding = node_tensor.cpu().detach().numpy()
            else:
                new_node_embedding = np.concatenate(
                    (node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                # new_node_embedding = node_tensor.cpu().detach().numpy()
                node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
    embedding_tensor = torch.from_numpy(node_embedding)

    all_labels = []
    all_queries = []
    all_communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    for i in range(query.shape[0]):
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        single_query = query[i].unsqueeze(0)  # (1, some_dim)
        metrics = monitor.monitor_function(community_search, single_query, embedding_tensor, G)
        query_index, selected_candidates = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        all_labels.append(labels[i])
        all_communities.append(selected_candidates)
        all_queries.append(set(query_index))
        end = time.time()
        all_t.append(end - start)
    str_queries = ['_'.join(str(G.nodes[node]['old_id']) for node in query) for query in all_queries]
    model_thrs = [-1 for i in range(len(all_communities))]
    if os.path.exists(args.save_embedding):
        os.remove(args.save_embedding)
    return G, all_communities, all_labels, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def TransZero_RecQuery(args, input_queries, model_gts, model, device, best_thr = None):
    feature, G, node_id_map, N = read_feat_graph(args)
    model_queries = query_to_onehot(input_queries, N)
    row = []
    col = []
    for uid, vid in G.edges():
        row.extend([uid, vid])
        col.extend([vid, uid])
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.ones(len(row), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()
    adj_scipy = torch_adj_to_scipy(adj)
    graph = dgl.from_scipy(adj_scipy)
    lpe = laplacian_positional_encoding(graph, args.pe_dim)

    # remap feature according to new id
    features = torch.from_numpy(feature)
    features = torch.cat((features, lpe), dim=1)
    query = torch.Tensor(model_queries)

    start_feature_processing = time.time()
    processed_features = re_features(adj, features, args.hops)  # return (N, hops+1, d)
    if processed_features.shape[0] < 10000:
        indicator = conductance_hop(adj, args.hops)  # return (N, hops+1)
        indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1])
        processed_features = processed_features * indicator
    t_feature_precessing = time.time() - start_feature_processing
    print("feature process time: {:.4f}s".format(t_feature_precessing))
    start = time.time()
    print("starting transformer to coo")
    adj = transform_coo_to_csr(adj)
    print("start mini batch processing")
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0])  # transform to coo to support tensor operation
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("adj process time: {:.4f}s".format(time.time() - start))
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle=False)

    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()
    node_embedding = []
    with torch.no_grad():
        for _, item in enumerate(data_loader):
            nodes_features = item.to(device)
            node_tensor, neighbor_tensor = model(nodes_features)
            if len(node_embedding) == 0:
                node_embedding = np.concatenate(
                    (node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                # node_embedding = node_tensor.cpu().detach().numpy()
            else:
                new_node_embedding = np.concatenate(
                    (node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                # new_node_embedding = node_tensor.cpu().detach().numpy()
                node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
    embedding_tensor = torch.from_numpy(node_embedding)

    all_queries = []
    all_communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    for i in range(query.shape[0]):
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        single_query = query[i].unsqueeze(0)  # (1, some_dim)
        metrics = monitor.monitor_function(community_search, single_query, embedding_tensor, G)
        query_index, selected_candidates = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        all_communities.append(selected_candidates)
        all_queries.append(set(query_index))
        end = time.time()
        all_t.append(end - start)
    str_queries = ['_'.join(str(G.nodes[node]['old_id']) for node in query) for query in all_queries]
    if os.path.exists(args.save_embedding):
        os.remove(args.save_embedding)
    return all_communities, str_queries, all_t, cpu_memory, gpu_memory

def TransZero_test(args, model, testloader, best_thr, device):
    data_loader, query, labels, G = testloader
    if not os.path.exists(args.save_embedding) and args.save_data == 1:
        embedding_tensor = torch.from_numpy(np.load(args.save_embedding))
    else:
        model = model.to(device)
        model.eval()
        node_embedding = []
        with torch.no_grad():
            for _, item in enumerate(data_loader):
                nodes_features = item.to(device)
                node_tensor, neighbor_tensor = model(nodes_features)
                if len(node_embedding) == 0:
                    node_embedding = np.concatenate(
                        (node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                    # node_embedding = node_tensor.cpu().detach().numpy()
                else:
                    new_node_embedding = np.concatenate(
                        (node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                    # new_node_embedding = node_tensor.cpu().detach().numpy()
                    node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
        embedding_tensor = torch.from_numpy(node_embedding)
    all_queries = []
    all_communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    for i in range(query.shape[0]):
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        single_query = query[i].unsqueeze(0)  # (1, some_dim)
        metrics = monitor.monitor_function(community_search, single_query, embedding_tensor, G)
        query_index, selected_candidates = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        all_communities.append(selected_candidates)
        all_queries.append(set(query_index))
        end = time.time()
        all_t.append(end - start)
    str_queries = ['_'.join(str(G.nodes[node]['old_id']) for node in query) for query in all_queries]
    if os.path.exists(args.save_embedding):
        os.remove(args.save_embedding)
    return G, all_communities, labels, str_queries, all_t, cpu_memory, gpu_memory

def train_one_epoch(trainloader, batch_size, device, adj_batch, minus_adj_batch, optimizer, model, lr_scheduler):
    model.train()
    epoch_loss = 0.0
    start = time.time()
    tracemalloc.start()
    for index, item in enumerate(trainloader):
        start_index = index * batch_size
        nodes_features = item.to(device)
        adj_ = adj_batch[index].to(device)
        minus_adj = minus_adj_batch[index].to(device)
        # print(nodes_features.shape)
        optimizer.zero_grad()
        node_tensor, neighbor_tensor = model(nodes_features)
        # print(node_tensor.shape, neighbor_tensor.shape, adj_.shape, minus_adj.shape)
        loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_, minus_adj)
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss_train.item()
    end = time.time()
    epoch_time = end - start
    return epoch_loss, epoch_time

def TransZero_train(args, inputloader, device):
    ini_time = time.time()
    
    if inputloader is None or (isinstance(inputloader, tuple) and len(inputloader) == 0):
        print("Empty inputloader provided, attempting to load pre-trained model...")
        model = load_model_complete(args, device)
        if model is not None:
            print("Loaded pre-trained model successfully")
            return model, 0, 0
        else:
            raise ValueError("No pre-trained model found and inputloader is empty. Cannot proceed.")
    
    trainloader, adj_batch, minus_adj_batch = inputloader
    
    model = load_model_complete(args, device)
    if model is not None:
        print("Loaded pre-trained model, skipping training")
        return model, 0, 0
    
    model = PretrainModel(input_dim=trainloader.dataset.shape[2], config=args).to(device)
    max_cpu_memory = 0
    max_gpu_memory = 0
    
    # train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epoch)
    early_stopping = EarlyStopping(model, **stopping_args)
    print("starting training...")
    t_start = time.time()
    all_loss = []
    
    save_model_complete(model, args, inputloader)
    
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
        metrics = monitor.monitor_function(train_one_epoch, trainloader, args.batch_size, device, adj_batch, minus_adj_batch, optimizer, model, lr_scheduler)
        epoch_loss, epoch_time = metrics['result']
        all_loss.append(epoch_loss)
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
        max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
        # save model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
            save_model_complete(model, args, inputloader)
        avg_loss = float('inf') if len(trainloader) == 0 else epoch_loss / len(trainloader)
        sum_time = time.time() - ini_time
        write_train_info(args.save_trainInfo, epoch, epoch_time, sum_time, cur_cpu_memory, cur_gpu_memory, avg_loss, avg_loss)
        if (epoch + 1) % args.save_per_epoch == 0:
            print(f"Epoch [{epoch + 1}/{args.epoch}], Time: {epoch_time:.6f}s, "
                  f"CPU Memory: {cur_cpu_memory / (1024 * 1024):.6f} MB, "
                  f"GPU Memory: {cur_gpu_memory:.6f} MB, "
                  f"Loss: {avg_loss:.6f}")
        if args.early_stopping !=0 and early_stopping.simple_check(all_loss):
            break
    print("Optimization Finished!")
    print("Train time: {:.4f}s".format(time.time() - t_start))

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with loss: {best_loss:.6f}")

    # save embedding
    if not os.path.exists(args.save_embedding) and args.save_data == 1:
        model.eval()
        node_embedding = []
        for _, item in enumerate(trainloader):
            nodes_features = item.to(device)
            node_tensor, neighbor_tensor = model(nodes_features)
            if len(node_embedding) == 0:
                node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                # node_embedding = node_tensor.cpu().detach().numpy()
            else:
                new_node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
                # new_node_embedding = node_tensor.cpu().detach().numpy()
                node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
        np.save(args.save_embedding, node_embedding)

    return model, max_cpu_memory, max_gpu_memory
    


