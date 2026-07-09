import copy
from .subgraph import SubGraph
import numpy as np
import os
from utils import *
from memory_count import FunctionPerformanceMonitor
import torch

def ICS_GNN_parameter(input_args):
    args = copy.deepcopy(input_args)
    args.dropout = 0.5 # Dropout parameter. Default is 0.5.
    args.learning_rate = 0.01 # Learning rate. Default is 0.01.
    # args.community_size = 30 # The size of final community. Default is 30.
    args.ics_train_ratio = 0.02 # Train ratio of ics-gnn. Default is 0.02.
    if args.exp_mod == 7:
        if args.train_size == 0:
            args.ics_train_ratio = 0
        else:
            args.ics_train_ratio = args.ics_train_ratio * (args.train_size/args.default_train_size)
    if args.exp_mod == 8:
        if args.exp_param == 100:
            args.ics_train_ratio = 0
        else:
            args.ics_train_ratio = args.ics_train_ratio * (1-args.exp_param/100)
    args.subgraph_size = 400 # The size of subgraphs. Default is 400.

    args.min_thr = 10
    args.max_thr = args.subgraph_size
    args.thr_step = 10

    args.layers = [16] # The size of hidden layers.
    args.save_model = args.model_path + "/" + args.dataset + f"_ICS_GNN_{args.exp_mod}_{args.exp_param}.pth" # the file to save the parameter
    args.save_thr = args.model_path + "/" + args.dataset + f"_ICS_GNN_thr_{args.exp_mod}_{args.exp_param}"  # the file to save the best threshold
    args.save_trainInfo = args.train_path + "/" + args.dataset + f"_ICS_GNN_log_{args.exp_mod}_{args.exp_param}" # the file to save log during training
    args.save_processData = args.process_path + "/" + args.dataset + f"_ICS_GNN_{args.exp_mod}_{args.exp_param}.pth" # the file to save DataLoader of model
    args.save_result = args.result_path + "/" + args.dataset + f"_ICS_GNN_{args.exp_mod}_{args.exp_param}" # the file to save result
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_ICS_GNN_recLabel" # the file to save recommendation label
    return args

def pack_data(args, g, feature, input_gt, input_query):
    queries = single_query(input_query)
    sub = SubGraph(args, g, feature)
    return (queries, input_gt, g, sub)

def ICS_GNN_DataLoader(args, phase = 'all'):
    feature, g, node_id_map, N = read_feat_graph(args)
    processed_train, processed_valid, processed_test = None, None, None
    if phase == 'all' or phase == 'train':
        valid_cur_in, valid_cur_out = split_data(args, N, node_id_map, g, return_list = True, return_query = 'valid')
        processed_valid = (g, feature, valid_cur_out, valid_cur_in)
    if phase == 'all' or phase == 'test':
        test_cur_in, test_cur_out = split_data(args, N, node_id_map, g, return_list = True, return_query = 'test')
        processed_test = (g, feature, test_cur_out, test_cur_in)
    return processed_train, processed_valid, processed_test

def ICS_GNN_valid(args, model, validloader, device):
    if os.path.exists(args.save_thr):
        with open(args.save_thr, "r") as file:
            best_thr = int(file.read().strip())
            return best_thr
    g, feature, valid_cur_out, valid_cur_in = validloader
    queries, labels, g, sub = pack_data(args, g, feature, valid_cur_out, valid_cur_in)
    best_thr = args.min_thr
    best_f1 = 0.0
    for thr in range(args.min_thr, args.max_thr + args.thr_step, args.thr_step):
        sub.TOPK_SIZE = thr
        communities = []
        for idx, q in enumerate(queries):
            gt = labels[idx]
            train_time, test_time, model_size, model_param, community, train_gpu_memory, train_cpu_memory = sub.community_search(q, gt, device)
            communities.append(set(community))
        pre, rec, f1, jac = evaluate_gt_communities(communities, labels)
        avg_f1 = np.mean(f1)
        if avg_f1 > best_f1:
            best_thr = thr
            best_f1 = avg_f1
    print(f"best f1 {best_f1}, best_thr {best_thr}", best_f1, best_thr)
    with open(args.save_thr, "w") as file:
        file.write(f"{best_thr}")
    return best_thr

def ICS_GNN_RecQuery(args, model_queries, model_gts, model, device, best_thr = None):
    # get graph
    feature, g, node_id_map, N = read_feat_graph(args)
    queries, comms, g, sub = pack_data(args, g, feature, model_gts, model_queries)
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    str_queries = []
    model_thrs = None
    for idx, q in enumerate(queries):
        sub.TOPK_SIZE = best_thr[idx]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gt = comms[idx]
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(sub.community_search, q, gt, device)
        train_time, test_time, model_size, model_param, community, train_gpu_memory, train_cpu_memory = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = max(get_gpu_memory(),train_gpu_memory)
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        communities.append(set(community))
        all_t.append(test_time)
        str_queries.append(str(g.nodes[q]['old_id']))
    return communities, str_queries, all_t, cpu_memory, gpu_memory

def ICS_GNN_RecGenerate(args, model_queries, model_gts, model, device):
    # get graph
    feature, g, node_id_map, N = read_feat_graph(args)
    queries, comms, g, sub = pack_data(args, g, feature, model_gts, model_queries)
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    str_queries = []
    model_thrs = []
    all_labels = []
    for idx, q in enumerate(queries):
        for thr in range(args.min_thr, args.max_thr + args.thr_step, args.thr_step):
            sub.TOPK_SIZE = thr
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            gt = comms[idx]
            monitor = FunctionPerformanceMonitor()
            metrics = monitor.monitor_function(sub.community_search, q, gt, device)
            train_time, test_time, model_size, model_param, community, train_gpu_memory, train_cpu_memory = metrics['result']
            cur_cpu_memory = metrics['max_memory_mb']
            cur_gpu_memory = max(get_gpu_memory(),train_gpu_memory)
            cpu_memory.append(cur_cpu_memory)
            gpu_memory.append(cur_gpu_memory)
            communities.append(set(community))
            all_t.append(test_time)
            all_labels.append(gt)
            model_thrs.append(thr)
            str_queries.append(str(g.nodes[q]['old_id']))
    return g, communities, all_labels, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def ICS_GNN_test(args, model, testloader, best_thr, device):
    g, feature, test_cur_out, test_cur_in = testloader
    queries, labels, g, sub = pack_data(args, g, feature, test_cur_out, test_cur_in)
    sub.TOPK_SIZE = best_thr
    sum_train = 0
    sum_query = 0
    sum_model_size = 0
    sum_model_param = 0
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    all_train_gpu_memory = 0
    all_train_cpu_memory = 0
    for idx, q in enumerate(queries):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gt = labels[idx]
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(sub.community_search, q, gt, device)
        train_time, test_time, model_size, model_param, community, train_gpu_memory, train_cpu_memory = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = max(get_gpu_memory(),train_gpu_memory)
        all_train_gpu_memory = max(all_train_gpu_memory, train_gpu_memory)
        all_train_cpu_memory = max(all_train_cpu_memory, train_cpu_memory)
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        communities.append(set(community))
        sum_train += train_time
        sum_query += test_time
        sum_model_size += model_size
        sum_model_param += model_param
        all_t.append(test_time)
    avg_train = sum_train/len(queries)
    avg_model_size = sum_model_size/len(queries)
    avg_model_param = sum_model_param/len(queries)
    str_queries = [str(g.nodes[query]['old_id']) for query in queries]
    return g, communities, labels, str_queries, all_t, cpu_memory, gpu_memory, avg_train, sum_query, avg_model_size, avg_model_param, all_train_gpu_memory, all_train_cpu_memory