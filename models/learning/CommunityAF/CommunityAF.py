import argparse
import os
import datetime
import random
from .conv import GraphConv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import time
from memory_count import FunctionPerformanceMonitor
from utils import *
from .CommunityAF_utils import preprocess_nodefeats,PretrainComDataset,get_augment,get_data,make_single_node_encoding,make_nodes_encoding
from .model import CSAFModel
import copy
from .graph import Graph
import networkx as nx

def margin_loss(x1,x2,target,margin,reduce):
    if reduce =='mean':
        return torch.mean(torch.sqrt(torch.relu(((x2-x1+1)**2-(1-margin)**2))))
    elif reduce=='sum':
        return torch.mean(torch.sqrt(torch.relu(((x2 - x1 + 1) ** 2 - (1 - margin) ** 2))))

def eval_loss(args,device,epoch_cnt,model,valid_dataloader,valid_mapper,valid_dataset,attr):
    batch_losses = []
    model.eval()
    batch_cnt = 0
    with torch.no_grad():
        for i_batch, batch_data in (enumerate(valid_dataloader)):
            #print('batch_data', batch_data[:3])
            batch_data = get_data(batch_data.tolist(),args,valid_mapper,valid_dataset)
            batch_cnt += 1
            batch_result, ln_var = model(attr, batch_data)
            out_z, out_logdet = batch_result['out_z'], batch_result['out_logdet']
            loss = model.log_prob(out_z, out_logdet)
            if args.rankingloss and epoch_cnt >= args.cnt_rank:
                stop_score = batch_result['stop_score']
                pospair = batch_result['pospair']
                negpair = batch_result['negpair']
                target = [1] * len(pospair)
                target = torch.LongTensor(target).to(device)
                if args.squrank:
                    rank_loss = margin_loss(stop_score[pospair], stop_score[negpair], target,margin=args.margin,reduce=args.reduce)
                else:
                    rank_loss = torch.nn.functional.margin_ranking_loss(stop_score[pospair], stop_score[negpair], target,margin=args.margin,reduce='sum')
                loss += rank_loss*args.gamma
            batch_losses.append(loss.item())
    epoch_loss = sum(batch_losses)
    return epoch_loss

def train_one_epoch(args,device,epoch_cnt,model,train_dataloader,optimizer,roll_mapper,pre_dataset,attr):
    t_start = time.time()
    batch_losses = []
    model.train()
    batch_cnt = 0
    for i_batch, batch_data in (enumerate(train_dataloader)):
        #print('batch_data', batch_data[:3])
        batch_data = get_data(batch_data.tolist(),args,roll_mapper,pre_dataset)
        optimizer.zero_grad()
        batch_cnt += 1
        batch_result, ln_var = model(attr, batch_data)
        out_z, out_logdet = batch_result['out_z'], batch_result['out_logdet']
        loss = model.log_prob(out_z, out_logdet)
        if args.rankingloss and epoch_cnt >= args.cnt_rank:
            stop_score = batch_result['stop_score']
            pospair = batch_result['pospair']
            negpair = batch_result['negpair']
            target = [1] * len(pospair)
            target = torch.LongTensor(target).to(device)
            if args.squrank:
                rank_loss = margin_loss(stop_score[pospair], stop_score[negpair], target,margin=args.margin,reduce=args.reduce)
            else:
                rank_loss = torch.nn.functional.margin_ranking_loss(stop_score[pospair], stop_score[negpair], target,margin=args.margin,reduce='sum')
            loss += rank_loss*args.gamma
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    epoch_loss = sum(batch_losses)
    epoch_time = time.time() - t_start
    return epoch_loss, epoch_time

def preprocess(idx, seed, args, train_comms, graph, conv, nxg):
    com = train_comms[idx].tolist()
    labels = [0] * len(com)
    labels[-1] = 1
    left = set(com) - set([seed])
    bfs_com = [seed]
    bfs_com_set = set(bfs_com)
    neighbor = []
    next_nodes = []
    batch_now_com = [[seed]]
    neighbor.append(graph.neighbors[seed])
    # community must be connected
    while left:
        union = neighbor[-1] & left
        assert union != set()
        left -= union
        union = list(union)
        while union:
            if False and args.rollouts != 1:
                random.shuffle(union)
            top = union.pop()
            bfs_com.append(top)
            bfs_com_set.add(top)
            batch_now_com.append(batch_now_com[-1] + [top])
            neighbor.append((graph.neighbors[top] | neighbor[-1]) - bfs_com_set)
            next_nodes.append([top])
    next_nodes.append([-1])
    for i in range(len(neighbor)):
        next_node = next_nodes[i]

        if (len(neighbor[i]) > args.max_neighbor):
            neighbor[i] = np.random.choice(list(neighbor[i] - set(next_node)),
                                         args.max_neighbor - len(next_node)).tolist() + next_node


        neighbor[i] = sorted(list(neighbor[i]))

    neighbor_map = [{node: i for i, node in enumerate(ner)} for ner in neighbor]
    next_onehot = np.zeros((len(next_nodes), args.max_neighbor + 1))
    for i in range(len(next_nodes) - 1):
        tmpnext = next_nodes[i]
        for j in tmpnext:
            next_onehot[i][neighbor_map[i][j]] = 1

    next_onehot[-1][args.max_neighbor] = 1

    z_seed = make_single_node_encoding([seed], graph,conv)
    z_node = make_nodes_encoding([[seed]] + [node for node in next_nodes[:-1]], graph,conv)

    z_seed_dense = np.asarray(z_seed.todense())
    z_node_dense = np.asarray(z_node.todense())
    z_node_dense = z_node_dense.cumsum(axis=0)

    batch_z_seed = [
        torch.as_tensor(z_seed_dense[0, batch_now_com[i] + neighbor[i]]).unsqueeze(0)
        for i in range(len(next_nodes))
    ]
    batch_z_node = [
        torch.as_tensor(z_node_dense[i, batch_now_com[i] + neighbor[i]]).unsqueeze(0)
        for i in range(len(next_nodes))
    ]

    # batch_z_seed = []
    # batch_z_node = []
    # for i in range(1, len(next_nodes)):
    #     z_node[i] += z_node[i - 1]
    # for i in range(len(next_nodes)):
    #     batch_z_seed.append(torch.from_numpy(z_seed[0, (batch_now_com[i]) + neighbor[i]].todense()))
    #     batch_z_node.append(torch.from_numpy(z_node[i, (batch_now_com[i]) + neighbor[i]].todense()))
    #     assert batch_z_node[-1].shape[1] == len(batch_now_com[i]) + len(neighbor[i]) and batch_z_seed[-1].shape[1] == len(
    #         batch_now_com[i]) + len(neighbor[i])

    assert len(neighbor) == len(batch_now_com)
    result = {"neighbor": neighbor,
              "next_onehot": next_onehot,
              "now_com": batch_now_com,
              "z_seed": batch_z_seed,
              "z_node": batch_z_node,
              "label": labels
              }
    if args.augment:
        batch_z_augment = []
        z_augment = get_augment([seed],graph,nxg,conv)
        for i in range(len(next_nodes)):
            batch_z_augment.append(torch.from_numpy(z_augment[0, (batch_now_com[i]) + neighbor[i]].todense()))
        result["z_augment"] = batch_z_augment
    if args.rankingloss:
        neg_neighbor = neighbor[-1][:]
        if -1 in neg_neighbor:
            neg_neighbor.remove(-1)
        if len(neg_neighbor) < args.neg_num:
            neg_neighbor += np.random.choice(list(graph.nodes - set(com)), args.neg_num - len(neg_neighbor)).tolist()
        batch_neg = np.random.choice(neg_neighbor, args.neg_num).tolist()
        assert len(batch_neg) >= args.neg_num
        neg_z_node = make_nodes_encoding([[node] for node in batch_neg], graph,conv)
        for i in range(neg_z_node.shape[0]):
            neg_z_node[i] += z_node[-1]
        batch_neg_z_node = []
        batch_neg_z_seed = []
        neg_neighbors = []
        neg_batch_z_augment = []
        for i in range(len(batch_neg)):
            neg = batch_neg[i]
            neg_neighbor = list(set(neg_neighbor) | graph.neighbors[neg] - set(com + [neg]))
            if (len(neg_neighbor) > args.max_neighbor):
                neg_neighbor = np.random.choice(neg_neighbor, args.max_neighbor).tolist()
            batch_neg_z_seed.append(torch.from_numpy(z_seed[0, com + [neg] + sorted(
                neg_neighbor)].todense()))
            batch_neg_z_node.append(torch.from_numpy(neg_z_node[i, com + [neg] + sorted(neg_neighbor)].todense()))
            neg_neighbors.append(neg_neighbor)
            if args.augment:
                neg_batch_z_augment.append(torch.from_numpy(z_augment[0, com + [neg] + sorted(neg_neighbor)].todense()))

        result["neg"] = batch_neg
        result["neg_neighbors"] = neg_neighbors
        result["neg_z_seed"] = batch_neg_z_seed
        result["neg_z_node"] = batch_neg_z_node
        result["neg_z_augment"] = neg_batch_z_augment
    return result

def load_pre_dataset(args,train_comms, graph, conv, nxg, train_queries):
    pre_dataset = []
    roll_mapper = {}
    st = time.time()
    assert args.rollouts == 1, "args.rollouts must be 1"
    for i in tqdm(range(len(train_comms))):
        seed = train_queries[i]
        result = preprocess(i, seed, args, train_comms, graph, conv, nxg)
        roll_mapper[i] = []
        roll_mapper[i].append(len(pre_dataset))
        pre_dataset.append(result)
        # for j in range(args.rollouts):
        #     if len(train_comms[i]) > j:
        #         seed = train_comms[i][j]
        #         result = preprocess(i, seed, args, train_comms, graph, conv, nxg)
        #         roll_mapper[i].append(len(pre_dataset))
        #         pre_dataset.append(result)
    pre_dataset.append(roll_mapper)
    print('end preprocess', time.time() - st)
    return pre_dataset[:-1], pre_dataset[-1]

# load train data only containing community
def load_pre_dataset_com(args,train_comms, graph, conv, nxg):
    pre_dataset = []
    roll_mapper = {}
    st = time.time()
    assert args.rollouts == 1, "args.rollouts must be 1"
    for i in tqdm(range(len(train_comms))):
        roll_mapper[i] = []
        for j in range(args.rollouts):
            if len(train_comms[i]) > j:
                seed = train_comms[i][j]
                result = preprocess(i, seed, args, train_comms, graph, conv, nxg)
                roll_mapper[i].append(len(pre_dataset))
                pre_dataset.append(result)
    pre_dataset.append(roll_mapper)
    print('end preprocess', time.time() - st)
    return pre_dataset[:-1], pre_dataset[-1]

def CommunityAF_RecGenerate(args, model_queries, model_gts, model, device):
    # pack data
    feature, G, node_id_map, N = read_feat_graph(args)
    graph = Graph(G)
    args.max_size = model.args.max_size

    # reduce dimension of feature
    conv = GraphConv(graph)
    attr = preprocess_nodefeats(conv, feature, args.nfeat)
    attr = torch.from_numpy(attr).float()
    get_labels = model_gts
    queries = single_query(model_queries)

    # load model
    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()

    # search parameters
    attr = attr.to(device)
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    str_queries = []
    labels = []
    model_thrs = []
    count = 1
    for i, query in enumerate(queries):
        ms = args.min_ms
        while ms <= args.max_ms:
            me = args.min_me
            while me <= args.max_me:
                args.m_s = ms
                args.m_e = me
                print(f"test query {count}/{len(queries)}  ms: {ms} me:{me}")
                t1 = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                monitor = FunctionPerformanceMonitor()
                metrics = monitor.monitor_function(community_search, args, model, query, graph, G, conv, attr)
                community = metrics['result']
                cur_cpu_memory = metrics['max_memory_mb']
                cur_gpu_memory = get_gpu_memory()
                cpu_memory.append(cur_cpu_memory)
                gpu_memory.append(cur_gpu_memory)
                communities.append(community)
                t2 = time.time()
                all_t.append(t2 - t1)
                model_thrs.append(f"{ms}_{me}")
                str_queries.append(str(G.nodes[query]['old_id']))
                labels.append(get_labels[i])
                me += args.thr_step
            ms += args.thr_step
        count += 1
    return G, communities, labels, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def CommunityAF_RecQuery(args, model_queries, model_gts, model, device, best_thr = None):
    # pack data
    feature, G, node_id_map, N = read_feat_graph(args)
    graph = Graph(G)
    args.max_size = model.args.max_size

    # reduce dimension of feature
    conv = GraphConv(graph)
    attr = preprocess_nodefeats(conv, feature, args.nfeat)
    attr = torch.from_numpy(attr).float()
    queries = single_query(model_queries)

    # load model
    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()

    attr = attr.to(device)
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    str_queries = []
    count = 1
    for i, query in enumerate(queries):
        print(f"test query {count}/{len(queries)}")
        count += 1
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        thrs = best_thr[i]
        ms = int(thrs[0])
        me = int(thrs[1])
        args.m_s = ms
        args.m_e = me
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(community_search, args, model, query, graph, G, conv, attr)
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1)
    str_queries = [str(G.nodes[node]['old_id']) for node in queries]
    return communities, str_queries, all_t, cpu_memory, gpu_memory

def CommunityAF_parameter(input_args):
    args = copy.deepcopy(input_args)
    args.max_neighbor = 200
    args.num_flow_layer = 8
    args.dropout = 0
    args.rollouts = 1
    args.augment = False
    args.neg_num = 10
    args.nfeat = 128
    args.nhid = 128
    args.nout = 128
    args.lr = 0.001
    args.batch_size = 64
    args.single_mlp = True
    args.is_batchNorm = False
    args.normalization = None
    args.is_bn_before = False
    args.init_emb = False
    args.com_feature = True
    args.rankingloss = True
    args.best_score = True
    args.m_e = 1 # the number of communities generated for one query [1,5]
    args.m_s = 2 # control the early stop when expanding community [1,10]
    args.gamma = 1
    args.squrank = False
    args.margin = 0
    args.reduce = "sum"
    args.cache = True
    args.rule = False
    args.multiprocessing = True
    args.stopmode = 1
    # args.chunk_size = 2000
    args.att_mode = 1
    args.cnt_rank = 0
    args.community_min = 1
    args.max_size = 1000
    args.min_me = 1
    args.max_me = 5
    args.min_ms = 1
    args.max_ms = 10
    args.thr_step = 1
    if args.exp_mod == 5 and args.fix:
        args.save_model = args.model_path + "/" + args.dataset + f"_CommunityAF_1_0.pth"  # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_CommunityAF_log_1_0"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_CommunityAF_thr_1_0"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_CommunityAF_{args.exp_mod}_{args.exp_param}_fix" # the file to save result
    else:
        args.save_model = args.model_path + "/" + args.dataset + f"_CommunityAF_{args.exp_mod}_{args.exp_param}.pth" # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_CommunityAF_log_{args.exp_mod}_{args.exp_param}"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_CommunityAF_thr_{args.exp_mod}_{args.exp_param}"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_CommunityAF_{args.exp_mod}_{args.exp_param}" # the file to save result
    args.save_processData = args.process_path + "/" + args.dataset + f"_CommunityAF_{args.exp_mod}_{args.exp_param}.pth"  # the file to save DataLoader of model
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_CommunityAF_recLabel" # the file to save recommendation label
    return args

def CommunityAF_DataLoader(args, phase = 'all'):
    # read graph and feature
    feature, G, node_id_map, N = read_feat_graph(args)
    graph = Graph(G)

    # reduce dimension of feature
    conv = GraphConv(graph)
    attr = preprocess_nodefeats(conv, feature, args.nfeat)
    attr = torch.from_numpy(attr).float()
    
    processed_train, processed_valid, processed_test = None, None, None
    if phase == 'all' or phase == 'train':
        # split data
        train_cur_in, train_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'train', input_train_size = args.default_train_size)
        valid_cur_in, valid_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'valid')

        train_comms = deduplicate_communities_list(train_cur_out)
        num_communities = int((args.train_size/args.default_train_size)*len(train_comms))
        train_comms = train_comms[:num_communities]

        valid_comms = deduplicate_communities_list(valid_cur_out)
        print(f"The number of community for training and validation:{len(train_comms)} and {len(valid_comms)}")

        pre_dataset, roll_mapper = load_pre_dataset_com(args, train_comms, graph, conv, G)
        if len(train_comms)>0:
            train_dataloader = DataLoader(PretrainComDataset(roll_mapper, len(train_comms)),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=0)
        else:
            train_dataloader = None

        # pack valid data for train
        valid_num = len(valid_comms)
        valid_comms = valid_comms[:valid_num]
        valid_dataset, valid_mapper = load_pre_dataset_com(args, valid_comms, graph, conv, G)
        if len(valid_comms) > 0:
            args.max_com_size = max(len(x) for x in valid_comms)
        args.max_size = args.max_com_size
        if len(valid_comms)>0:
            valid_dataloader = DataLoader(PretrainComDataset(valid_mapper, len(valid_comms)),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0)
        else:
            valid_dataloader = None

        processed_train = (train_dataloader, roll_mapper, pre_dataset, valid_dataloader, valid_mapper, valid_dataset, attr)

        # pack valid data for parameter search
        valid_comms_dup = valid_cur_out
        valid_queries = single_query(valid_cur_in)
        processed_valid = (graph, conv, attr, valid_queries, G, valid_comms_dup)

    if phase == 'all' or phase == 'test':
        # pack test data
        test_cur_in, test_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'test')
        test_comms_dup = test_cur_out
        test_queries = single_query(test_cur_in)
        processed_test = (graph, conv, attr, test_queries, G, test_comms_dup)
    return processed_train, processed_valid, processed_test

def save_model_complete(model, args, trainloader=None):
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'nfeat': args.nfeat,
            'nhid': args.nhid,
            'nout': args.nout,
            'num_flow_layer': args.num_flow_layer,
            'max_neighbor': args.max_neighbor,
            'att_mode': args.att_mode,
            'dropout': args.dropout,
            'single_mlp': args.single_mlp,
            'is_batchNorm': args.is_batchNorm,
            'normalization': args.normalization,
            'init_emb': args.init_emb,
            'lr': args.lr,
            'weight_decay': 0
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
        
        import copy
        temp_args = copy.deepcopy(args)
        temp_args.nfeat = model_config['nfeat']
        temp_args.nhid = model_config['nhid']
        temp_args.nout = model_config['nout']
        temp_args.num_flow_layer = model_config['num_flow_layer']
        temp_args.max_neighbor = model_config['max_neighbor']
        temp_args.att_mode = model_config['att_mode']
        temp_args.dropout = model_config['dropout']
        temp_args.single_mlp = model_config['single_mlp']
        temp_args.is_batchNorm = model_config['is_batchNorm']
        temp_args.normalization = model_config['normalization']
        temp_args.init_emb = model_config['init_emb']
        
        model = CSAFModel(args=temp_args, device=device)
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

def CommunityAF_train(args, trainloader, device):
    ini_time = time.time()
    
    if trainloader is None:
        print("Empty trainloader provided, attempting to load pre-trained model...")
        model = load_model_complete(args, device)
        if model is not None:
            print("Loaded pre-trained model successfully")
            return model, 0, 0
        else:
            raise ValueError("No pre-trained model found and trainloader is empty. Cannot proceed.")
    
    train_dataloader, roll_mapper, pre_dataset, valid_dataloader, valid_mapper, valid_dataset, attr = trainloader
    attr = attr.to(device)
    
    model = load_model_complete(args, device)
    if model is not None:
        print("Loaded pre-trained model, skipping training")
        return model, 0, 0
    
    model = CSAFModel(args=args, device=device).to(device)
    max_cpu_memory = 0
    max_gpu_memory = 0
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epoch)
    early_stopping = EarlyStopping(model, **stopping_args)
    all_loss = []
    
    save_model_complete(model, args, trainloader)
    
    if train_dataloader is None:
        print("train set is null")
        model = load_model_complete(args, device)
        return model, max_cpu_memory, max_gpu_memory
    best_loss = float('inf')
    best_model_state = None
    for epoch in (range(args.epoch)):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(train_one_epoch, args,device,epoch,model,train_dataloader,optimizer,roll_mapper,pre_dataset,attr)
        train_loss, epoch_time = metrics['result']
        valid_loss = eval_loss(args,device,epoch,model,valid_dataloader,valid_mapper,valid_dataset,attr)
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
        avg_loss = float('inf') if len(train_dataloader) == 0 else train_loss / len(train_dataloader)
        avg_valid_loss = float('inf') if len(valid_dataloader) == 0 else valid_loss / len(valid_dataloader)
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

def community_search(args, model, query, graph, nxg, conv, attr):
    seeds = [query for _ in range(args.m_e)]
    n_seed = len(seeds)
    with torch.no_grad():
        decline = args.m_s
        stop = args.stopmode
        neribor_max = args.max_neighbor
        temperature = 0.75
        prior_node_dist = [torch.distributions.normal.Normal(torch.zeros([(neribor_max + 1)]),
                                                             temperature * torch.ones((neribor_max + 1))) for i in
                           range(n_seed)]

        batch_scores = [[-1] * decline for i in range(n_seed)]
        batch_nowcom = [[seed] for seed in seeds]  # one query
        batch_neribor = [list(graph.neighbors[seed]) for seed in seeds]
        batch_bestcom = [None for i in range(n_seed)]
        bacth_bests = [-2 for i in range(n_seed)]
        z_seeds = make_single_node_encoding([seed for seed in seeds], graph, conv)
        z_nodes = make_single_node_encoding([seed for seed in seeds], graph, conv)
        batch_z_seed = [0] * n_seed
        batch_z_node = [0] * n_seed
        if args.augment:
            batch_z_augment = [0] * n_seed
            z_augment = get_augment(seeds, graph, nxg, conv)
        batch_ok = [0] * n_seed
        ok = 0
        latent_nodes = [0] * n_seed
        for size in (range(args.max_size)):
            if (ok == n_seed):
                break
            for i in (range(n_seed)):
                if batch_ok[i] == 1:
                    continue
                latent_nodes[i] = prior_node_dist[i].sample().view(1, -1)
                if len(batch_neribor[i]) > neribor_max:
                    batch_neribor[i] = np.random.choice(batch_neribor[i], neribor_max)

                batch_neribor[i] = sorted(batch_neribor[i])
                batch_z_seed[i] = torch.from_numpy(z_seeds[i, (batch_nowcom[i]) + batch_neribor[i]].todense())
                batch_z_node[i] = torch.from_numpy(z_nodes[i, (batch_nowcom[i]) + batch_neribor[i]].todense())
                if args.augment:
                    batch_z_augment[i] = torch.from_numpy(
                        z_augment[i, (batch_nowcom[i]) + batch_neribor[i]].todense())

            latent_node = torch.stack(latent_nodes).view(n_seed, -1)
            batch_data = {"batch_com": batch_nowcom,
                          "batch_neighbor": batch_neribor,
                          "batch_deq": latent_node,
                          "batch_z_seed": batch_z_seed,
                          "batch_z_node": batch_z_node
                          }
            if args.augment:
                batch_data['batch_z_augment'] = batch_z_augment

            result = model.flow_core.batch_revser(attr, batch_data)
            out_z = result['out_z']
            newnodes = []
            for i in range(n_seed):
                if batch_ok[i] == 1:
                    newnodes.append(None)
                    continue
                latent_node = out_z[i]
                neribor_num = len(batch_neribor[i])

                if stop == 1:
                    score = result['stop_score'][i].item()
                    if args.rule:
                        score = 1 - nx.conductance(nxg, batch_nowcom[i])
                    if score > bacth_bests[i]:
                        bacth_bests[i] = score
                        batch_bestcom[i] = batch_nowcom[i]
                    flag = 0
                    for d in range(decline):
                        if score < batch_scores[i][-1 - d]:
                            flag += 1
                    if flag == decline and len(batch_nowcom[i]) >= args.community_min:
                        newnodes.append(None)
                        batch_ok[i] = 1
                        ok += 1
                    elif neribor_num == 0:
                        newnodes.append(None)
                        batch_ok[i] = 1
                        ok += 1
                    elif flag < decline:
                        index = torch.argmax(latent_node[:neribor_num]).item()
                        newnode = batch_neribor[i][index]
                        batch_nowcom[i].append(newnode)
                        batch_neribor[i].extend(list(graph.neighbors[newnode]))
                        batch_neribor[i] = list(set(batch_neribor[i]) - set(batch_nowcom[i]))
                        batch_scores[i].append(score)
                        newnodes.append(newnode)
                    else:
                        batch_scores[i].append(score)
                        newnodes.append(None)
                else:
                    if neribor_num != 0:
                        index = torch.argmax(latent_node[:neribor_num]).item()
                        newnode = batch_neribor[i][index]
                    else:
                        index = None
                    if index == None or (
                            latent_node[-1] > latent_node[index] and len(batch_nowcom[i]) >= args.community_min):
                        newnodes.append(None)
                        batch_ok[i] = 1
                        ok += 1
                    elif latent_node[-1] <= latent_node[index]:
                        batch_nowcom[i].append(newnode)
                        batch_neribor[i].extend(list(graph.neighbors[newnode]))
                        batch_neribor[i] = list(set(batch_neribor[i]) - set(batch_nowcom[i]))
                        newnodes.append(newnode)
                    else:
                        newnodes.append(None)
            z_nodes += make_single_node_encoding(newnodes, graph, conv)
        coms = []
        for i in range(0, len(batch_nowcom), args.m_e):
            if stop == 1:
                bestc = None
                bests = -1
                for j in range(args.m_e):
                    com = batch_nowcom[i + j]
                    s = bacth_bests[i + j]
                    if args.best_score == True:
                        com = batch_bestcom[i + j]
                        s = bacth_bests[i + j]
                    if s > bests:
                        bestc = com
                coms.append(bestc)
            else:
                coms.append(batch_nowcom[i])

        # only one query vertex
        return set(coms[0])

def CommunityAF_valid(args, model, validloader, device):
    if os.path.exists(args.save_thr):
        with open(args.save_thr, "r") as file:
            line = file.readline()
            best_ms, best_me = map(int, line.strip().split())
            return (best_ms, best_me)
    else:
        if next(model.parameters()).device != device:
            model = model.to(device)
    model.eval()
    graph, conv, attr, queries, nxg, labels = validloader
    attr = attr.to(device)
    best_f1 = 0.0
    best_ms = 1
    best_me = 1
    for ms in range(args.min_ms, args.max_ms + args.thr_step, args.thr_step):
        for me in range(args.min_me, args.max_me + args.thr_step, args.thr_step):
            args.m_e = me
            args.m_s = ms
            communities = []
            for query in queries:
                community = community_search(args, model, query, graph, nxg, conv, attr)
                communities.append(community)
            pre, rec, f1, jac = evaluate_gt_communities(communities, labels)
            avg_f1 = np.mean(f1)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_ms = ms
                best_me = me
            print(f"f1 {avg_f1}, ms {ms}, me {me}")
    print(f"best f1 {best_f1}, best_ms {best_ms}, best_me {best_me}")
    with open(args.save_thr, "w") as file:
        file.write(f"{best_ms} {best_me}")
    return (best_ms, best_me)

def CommunityAF_test(args, model, testloader, best_thr, device):
    ms, me = best_thr
    args.m_s = ms
    args.m_e = me
    model.eval()
    graph, conv, attr, queries, nxg, labels = testloader
    attr = attr.to(device)
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    # test community
    count = 1
    for query in queries:
        print(f"test query {count}/{len(queries)}")
        count += 1
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(community_search, args, model, query, graph, nxg, conv, attr)
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1)
    str_queries = [str(nxg.nodes[node]['old_id']) for node in queries]
    return nxg, communities, labels, str_queries, all_t, cpu_memory, gpu_memory