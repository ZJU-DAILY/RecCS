import gc
import shutil
from config import get_args
from models.learning.QD_GNN.QD_GNN import *
from models.learning.TransZero.TransZero import *
from models.learning.COCLEP.COCLEP import *
from models.learning.ICS_GNN.ICS_GNN import *
from models.learning.CommunityAF.CommunityAF import *
from models.learning.CommunityDF.CommunityDF import *
from models.learning.CSFormer.CSFormer import *
from models.non_learning.QDC.QDC_wrapper import QDCWrapper
from models.non_learning.LM.LM_wrapper import LMWrapper
from models.non_learning.DMCS.DMCS_wrapper import DMCSWrapper
from models.non_learning.PPR.PPR_wrapper import PPRWrapper
from models.non_learning.kcore.kcore_wrapper import KCoreWrapper
from models.non_learning.ktruss.ktruss_wrapper import KTrussWrapper
from models.non_learning.kclique.kclique_wrapper import KCliqueWrapper
from models.non_learning.kecc.kecc_wrapper import KECCWrapper
from models.non_learning.SGM.SGM_wrapper import SGMWrapper
from typing import List, Dict, Set, Tuple
import numpy as np
import torch
import os
import random
import psutil
from utils import *
import copy
import builtins
import time
from datetime import datetime
import pandas as pd
import networkx as nx
method_implementations = {
    'QDC': QDCWrapper('QDC'), 
    'LM': LMWrapper('LM'),
    'DMCS': DMCSWrapper('DMCS'),
    'PPR': PPRWrapper('PPR'),
    'kcore': KCoreWrapper('kcore'),
    'ktruss': KTrussWrapper('ktruss'),
    'kclique': KCliqueWrapper('kclique'),
    'kecc': KECCWrapper('kecc'),
    'SGM': SGMWrapper('SGM')
}
k_dict = {
    'ktruss': {'min_thr': 3, 'max_thr': 20, 'thr_step': 1},
    'kclique': {'min_thr': 1, 'max_thr': 7, 'thr_step': 1},
}
def tune_k_parameter(method_name, method_instance,
                    train_queries: List[List[int]], train_gts: List[Set[int]],
                    min_k, max_k, thr_step) -> int:
    print(f"[PARAMETER TUNING] Tuning k parameter for {method_name} on {method_instance.dataset_name}")
    print(f"[PARAMETER TUNING] Testing k values from {min_k} to {max_k}")
    print(f"[PARAMETER TUNING] Using {len(train_queries)} training queries")
    
    best_k = min_k
    best_f1 = 0.0
    print(f"[PARAMETER TUNING] Method: {method_name}, max_k: {max_k}")
    
    k_values_to_test = np.arange(min_k, max_k + thr_step, thr_step).tolist()
    print(f"[PARAMETER TUNING] Testing k values: {k_values_to_test}")
    
    for k in k_values_to_test:
        print(f"[PARAMETER TUNING] Testing k = {k}")
        communities = []
        for i, query in enumerate(train_queries):
            community = method_instance.search_community(query, k=k, silent_mode=True)
            if community and len(community) > 0:
                communities.append(community)
            else:
                communities.append(set(query))
        
        if communities and train_gts and len(communities) == len(train_gts):
            try:
                precisions, recalls, f1s, jaccards = evaluate_gt_communities(
                    communities, train_gts
                )
                avg_f1 = np.mean(f1s) if f1s else 0.0
                print(f"[PARAMETER TUNING] k={k}: F1={avg_f1:.4f} ({len(communities)} communities)")
                
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_k = k
                    print(f"[PARAMETER TUNING] New best k={k} with F1={avg_f1:.4f}")
            except Exception as e:
                print(f"[PARAMETER TUNING] Error evaluating k={k}: {e}")
        else:
            print(f"[PARAMETER TUNING] Invalid results for k={k}, skipping")
    
    print(f"[PARAMETER TUNING] Best k = {best_k} with F1-score = {best_f1:.4f}")
    
    return best_k

def nonLearning_CS(query, method_instance, optimal_k = None):
    if optimal_k is not None:
        community = method_instance.search_community(query, k=optimal_k)
    else:
        community = method_instance.search_community(query)
    return community

def process(model, args, device):
    old_exp_mod = -1
    if args.exp_mod == 9 or args.exp_mod == 10:
        old_exp_mod = args.exp_mod
        args.exp_mod = 1
    if args.phase == 'train' and model in ["QDC", "LM", "SGM", "DMCS", "PPR", "OQC"]: # do not need to train or valid
        return
    process = psutil.Process(os.getpid())
    base_cpu_memory = process.memory_info().rss / 1024 / 1024
    if model in args.non_learning_models:
        nonLearning_args = copy.deepcopy(args)
        nonLearning_args.method_name = model
        method_instance = method_implementations[model]
        # get graph
        before = process.memory_info().rss / 1024 / 1024
        graph, node_id_map, n_nodes = read_feat_graph(nonLearning_args, return_feature = False)
        after = process.memory_info().rss / 1024 / 1024

        # split data
        pre_start = time.time()
        valid_queries, valid_gts = split_data(nonLearning_args, n_nodes, node_id_map, graph, return_list= True, return_query = 'valid')
        test_queries, test_gts = split_data(nonLearning_args, n_nodes, node_id_map, graph, return_list= True, return_query = 'test')
        nonLearning_args.max_com_size = min(500, sum(len(x) for x in valid_gts) / len(valid_gts)) if valid_gts else 100
        print(f"valid size: {len(valid_queries)}, test size: {len(test_queries)}")
        pre_end = time.time()
        pre_time = pre_end - pre_start

        method_name = nonLearning_args.method_name
        if method_name in nonLearning_args.single_query and nonLearning_args.exp_mod == 3:
            print(f"{method_name} only supports single query vertex.")
            return None

        method_instance.dataset_name = nonLearning_args.dataset
        method_instance.n_nodes = n_nodes
        method_instance.max_com_size = nonLearning_args.max_com_size

        ori_dataset_name = nonLearning_args.dataset
        if nonLearning_args.exp_mod == 4 or nonLearning_args.exp_mod == 5:
            nonLearning_args.dataset = f'{nonLearning_args.dataset}_{nonLearning_args.exp_mod}_{nonLearning_args.exp_param}'
            method_instance.dataset_name = nonLearning_args.dataset
            tmp_path = os.path.join(nonLearning_args.data_path, nonLearning_args.dataset)
            os.makedirs(tmp_path, exist_ok=True)
            graph_path = os.path.join(tmp_path, 'graph')
            with open(graph_path, 'w') as f:
                for u, v in graph.edges():
                    f.write(f"{u} {v}\n")
        else:
            graph_path = os.path.join(nonLearning_args.data_path, nonLearning_args.dataset, 'graph')

        model_size = 0
        train_cpu_memory = 0
        test_time = 0
        test_cpu = 0
        data_time = 0
        train_time = 0
        valid_time = 0
        # train phase
        train_start = time.time()
        if method_name in nonLearning_args.index_based_methods and (nonLearning_args.phase == 'all' or nonLearning_args.phase == 'train'):
            if nonLearning_args.exp_mod == 5 and nonLearning_args.fix:
                method_instance.index_manager.args = True
            print(f"[{method_name}]begin construct index for {method_name} in {nonLearning_args.dataset} under {nonLearning_args.exp_mod}_{nonLearning_args.exp_param}, time {datetime.now()}")
            def index_func():
                monitor = FunctionPerformanceMonitor()
                return monitor.monitor_function(method_instance.build_index_only,graph_path, method_instance.dataset_name)
            metrics = index_func()
            train_cpu_memory = metrics['max_memory_mb']
            if nonLearning_args.exp_mod == 5 and nonLearning_args.fix:
                parts = nonLearning_args.dataset.split('_')
                index_path = os.path.join(nonLearning_args.model_path, f"{parts[0]}_{method_name}_index.pkl")
            else:
                index_path = os.path.join(nonLearning_args.model_path, f"{nonLearning_args.dataset}_{method_name}_index.pkl")
            if os.path.exists(index_path):
                model_size = os.path.getsize(index_path) / 1024 / 1024
            print(
                f"[{method_name}]end construct index for {method_name} in {nonLearning_args.dataset} under {nonLearning_args.exp_mod}_{nonLearning_args.exp_param}, time {datetime.now()}")
        train_end = time.time()
        train_time = train_end - train_start

        # data reading
        if method_name in nonLearning_args.index_based_methods and nonLearning_args.exp_mod == 5 and nonLearning_args.fix:
            method_instance.index_manager.args = True
        data_start = time.time()
        try:
            load_success = method_instance.load_graph(graph_path, nonLearning_args.dataset)
        except TypeError:
            load_success = method_instance.load_graph(graph_path)
        data_end = time.time()
        data_time = data_end - data_start + pre_time

        # valid phase
        valid_start_time = time.time()
        optimal_k = None
        if method_name in nonLearning_args.k_based_methods:
            if nonLearning_args.exp_mod == 5 and nonLearning_args.fix:
                save_thr = nonLearning_args.model_path + "/" + ori_dataset_name + "_" + method_name + f"_thr_1_0"
            else:
                save_thr = nonLearning_args.model_path + "/" + ori_dataset_name + "_" + method_name + f"_thr_{nonLearning_args.exp_mod}_{nonLearning_args.exp_param}"
            if os.path.exists(save_thr):
                with open(save_thr, "r") as file:
                    optimal_k = int(file.read().strip())
            else:
                print(f"[K-TUNING] Using validation data for k-parameter tuning ({len(valid_queries)} queries)")

                optimal_k = tune_k_parameter(
                    method_name, method_instance, valid_queries, valid_gts,
                    k_dict[method_name]['min_thr'], k_dict[method_name]['max_thr'], k_dict[method_name]['thr_step']
                )
                with open(save_thr, "w") as file:
                    file.write(f"{optimal_k}")
        valid_end_time = time.time()
        valid_time = valid_end_time - valid_start_time

        del valid_queries, valid_gts
        gc.collect()

        if nonLearning_args.phase == 'all' or nonLearning_args.phase == 'test':
            # test phase
            all_t = []
            cpu_memory = []
            gpu_memory = []
            test_communities = []
            str_queries = []

            test_start = time.time()
            for i,query in enumerate(test_queries):
                t1 = time.time()
                str_query = '_'.join(str(graph.nodes[node]['old_id']) for node in query)
                str_queries.append(str_query)
                if old_exp_mod != -1:
                    print(
                        f"[{method_name}]begin query {str_query} for {method_name} in {nonLearning_args.dataset} under {old_exp_mod}_{nonLearning_args.exp_param}, time {datetime.now()}")
                else:
                    print(
                        f"[{method_name}]begin query {str_query} for {method_name} in {nonLearning_args.dataset} under {nonLearning_args.exp_mod}_{nonLearning_args.exp_param}, time {datetime.now()}")
                def query_func(query, method_instance, optimal_k):
                    monitor = FunctionPerformanceMonitor()
                    return monitor.monitor_function(nonLearning_CS, query, method_instance, optimal_k)
                metrics = query_func(query, method_instance, optimal_k)
                community = metrics['result']
                cur_cpu_memory = metrics['max_memory_mb']
                cpu_memory.append(cur_cpu_memory)
                gpu_memory.append(0)
                test_communities.append(community if community else set(query))
                if old_exp_mod != -1:
                    print(
                        f"[{method_name}]end query {str_query} for {method_name} in {nonLearning_args.dataset} under {old_exp_mod}_{nonLearning_args.exp_param}, time {datetime.now()}")
                else:
                    print(
                        f"[{method_name}]end query {str_query} for {method_name} in {nonLearning_args.dataset} under {nonLearning_args.exp_mod}_{nonLearning_args.exp_param}, time {datetime.now()}")
                print(f"[{method_name}]Found community for vertex {str_query} (query {i + 1}/{len(test_queries)}):")
                t2 = time.time()
                t = t2 - t1
                all_t.append(t)
            test_end = time.time()
            test_time = test_end - test_start
            test_cpu = np.max(cpu_memory) - base_cpu_memory if len(cpu_memory) > 0 else 0

            if test_communities and test_gts:
                precisions, recalls, f1s, jaccards, nmi, ari, degree, density, diameter, conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus = evaluate_communities(
                        graph, test_communities, test_gts, nonLearning_args.diam_limit)
                if old_exp_mod != -1:
                    nonLearning_args.exp_mod = old_exp_mod
                if nonLearning_args.exp_mod == 5 and nonLearning_args.fix:
                    save_result = nonLearning_args.result_path + "/" + ori_dataset_name + "_" + method_name + f"_{nonLearning_args.exp_mod}_{nonLearning_args.exp_param}_fix"
                else:
                    save_result = nonLearning_args.result_path + "/" + ori_dataset_name + "_" + method_name + f"_{nonLearning_args.exp_mod}_{nonLearning_args.exp_param}"
                store_result_xlsx(save_result, all_t, str_queries, precisions, recalls, f1s, jaccards, nmi, ari, degree,
                                density, diameter, conductance,
                                com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, cpu_memory,
                                gpu_memory, base_cpu= base_cpu_memory)

        if train_cpu_memory > 0:
            train_cpu_memory = train_cpu_memory - base_cpu_memory
        if nonLearning_args.exp_mod == 5 and nonLearning_args.fix:
            summary_path = nonLearning_args.result_path + f"summary_{ori_dataset_name}_{nonLearning_args.exp_mod}_{nonLearning_args.exp_param}_fix"
        else:
            summary_path = nonLearning_args.result_path + f"summary_{ori_dataset_name}_{nonLearning_args.exp_mod}_{nonLearning_args.exp_param}"
        if old_exp_mod == -1 or nonLearning_args.exp_mod == 9 or nonLearning_args.exp_mod == 10:
            store_summary(summary_path, ori_dataset_name, method_name, data_time, train_time, valid_time, test_time,
                        model_size, 0, test_cpu, 0, train_cpu_memory, 0, nonLearning_args.phase, nonLearning_args.exp_mod, nonLearning_args.experiment_types)

        if nonLearning_args.exp_mod == 4 or nonLearning_args.exp_mod == 5:
            tmp_path = os.path.join(nonLearning_args.data_path, nonLearning_args.dataset)
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)

    elif model in args.learning_models:
        best_thr = None
        # set model parameters
        model_param = model + "_parameter"
        model_args = globals()[model_param](args)
        md = None
        model_size = 0
        model_param_num = 0
        train_cpu_memory = 0
        train_gpu_memory = 0
        test_time = 0
        test_cpu_memory = 0
        test_gpu_memory = 0
        data_time = 0
        train_time = 0
        valid_time = 0
        print(model_args)
        # load Data
        print(
            f"[{model}] begin loading data of {model} in {model_args.dataset} under {model_args.exp_mod}_{model_args.exp_param}, time {datetime.now()}")
        data_start = time.time()
        model_data = model + "_DataLoader"
        trainloader, validloader, testloader = globals()[model_data](model_args, model_args.phase)
        data_end = time.time()
        data_time = data_end - data_start
        print(
            f"[{model}]end loading data of {model} in {model_args.dataset} under {model_args.exp_mod}_{model_args.exp_param}, time {datetime.now()}")

        # train model
        train_start = time.time()
        if model != 'ICS_GNN':
            print(
                f"[{model}]begin training {model} in {model_args.dataset} under {model_args.exp_mod}_{model_args.exp_param}, time {datetime.now()}")
            model_train = model + "_train"
            md, train_cpu_memory, train_gpu_memory = globals()[model_train](model_args, trainloader, device)
            model_size = os.path.getsize(model_args.save_model) / 1024 / 1024
            if model == "CommunityDF":
                model_param_num = builtins.sum(p.numel() for p in md.gcnmodel.parameters()) + builtins.sum(p.numel() for p in md.pre_model.parameters())
            else:
                model_param_num = builtins.sum(p.numel() for p in md.parameters())
            print(
                f"[{model}]end training {model} in {model_args.dataset} under {model_args.exp_mod}_{model_args.exp_param}, time {datetime.now()}")
        train_end = time.time()
        train_time = train_end - train_start
        del trainloader
        gc.collect()

        # valid model
        valid_start = time.time()
        print("begin validation")
        model_valid = model + "_valid"
        try:
            best_thr = globals()[model_valid](model_args, md, validloader, device)
        except KeyError:
            print(f"Warning: {model_valid} function not found, setting best_thr to None")
            best_thr = None
        valid_end = time.time()
        valid_time = valid_end - valid_start

        del validloader
        gc.collect()
            
        if model_args.phase == 'all' or model_args.phase == 'test':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
            gc.collect()
            # test model
            test_start = time.time()
            if old_exp_mod != -1:
                print(
                    f"[{model}]begin testing {model} in {model_args.dataset} under {old_exp_mod}_{model_args.exp_param}, time {datetime.now()}")
            else:
                print(
                    f"[{model}]begin testing {model} in {model_args.dataset} under {model_args.exp_mod}_{model_args.exp_param}, time {datetime.now()}")
            model_test = model + "_test"

            if model == "ICS_GNN":
                g, communities, labels, str_queries, all_t, cpu_memory, gpu_memory, train_time, query_time, model_size, model_param_num, train_gpu_memory, train_cpu_memory = globals()[model_test](model_args, md, testloader, best_thr, device)
            else:
                g, communities, labels, str_queries, all_t, cpu_memory, gpu_memory = globals()[model_test](model_args, md, testloader, best_thr, device)
            test_end = time.time()
            test_time = test_end - test_start
            if old_exp_mod != -1:
                print(
                    f"[{model}]end testing {model} in {model_args.dataset} under {old_exp_mod}_{model_args.exp_param}, time {datetime.now()}")
            else:
                print(
                    f"[{model}]end testing {model} in {model_args.dataset} under {model_args.exp_mod}_{model_args.exp_param}, time {datetime.now()}")

            test_cpu_memory = np.max(cpu_memory) - base_cpu_memory if len(cpu_memory) > 0 else 0
            test_gpu_memory = np.max(gpu_memory) if len(gpu_memory) > 0 else 0

            # evaluate and store result
            if old_exp_mod != -1:
                args.exp_mod = old_exp_mod
                model_args = globals()[model_param](args)
            assert len(communities) == len(labels), f"the length of community and label is different"
            pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus = evaluate_communities(
                g, communities, labels, model_args.diam_limit)
            store_result_xlsx(model_args.save_result, all_t, str_queries, pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance,
                            com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, cpu_memory, gpu_memory, base_cpu = base_cpu_memory)

        if train_cpu_memory > 0:
            train_cpu_memory = train_cpu_memory - base_cpu_memory
        if model_args.exp_mod == 5 and model_args.fix:
            summary_path = model_args.result_path + f"summary_{model_args.dataset}_{model_args.exp_mod}_{model_args.exp_param}_fix"
        else:
            summary_path = model_args.result_path + f"summary_{model_args.dataset}_{model_args.exp_mod}_{model_args.exp_param}"
        if old_exp_mod == -1 or model_args.exp_mod == 9 or model_args.exp_mod == 10:
            store_summary(summary_path, model_args.dataset, model, data_time, train_time, valid_time, test_time, model_size, model_param_num, test_cpu_memory, test_gpu_memory, train_cpu_memory, train_gpu_memory, model_args.phase, model_args.exp_mod, model_args.experiment_types)

        # print(model_args)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    after_mem = process.memory_info().rss / 1024 / 1024
    print()

def inductive_exp(args):
    print("begin induct experiments")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        args.train_query_file = "1_induct_train_query"
        args.train_gt_file = "1_induct_train_gt"
        args.valid_query_file = "1_induct_valid_query"
        args.valid_gt_file = "1_induct_valid_gt"
        args.test_query_file = "1_induct_test_query"
        args.test_gt_file = "1_induct_test_gt"
        process(model, args, device)

def querysize_exp(args):
    print("begin query size experiments")
    # single_query = {"COCLEP","CGNP","ICS_GNN","CommunityAF","CommunityDF","CSFormer"}
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        if model in args.single_query:
            print(f"{model} only supports single query vertex.")
            continue
        qlist = args.params
        for qsize in qlist:
            args.exp_param = qsize
            args.train_query_file = f"{args.exp_param}_qsize_train_query"
            args.train_gt_file = f"{args.exp_param}_qsize_train_gt"
            args.valid_query_file = f"{args.exp_param}_qsize_valid_query"
            args.valid_gt_file = f"{args.exp_param}_qsize_valid_gt"
            args.test_query_file = f"{args.exp_param}_qsize_test_query"
            args.test_gt_file = f"{args.exp_param}_qsize_test_gt"
            process(model, args, device)
    print("end query size experiments")

def attr_exp(args):
    print("begin attribute experiments")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        args.attr = 1
        process(model, args, device)
    print("end attribute experiments")

def scalability_exp(args):
    print("begin scalability experiments")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        scala_list = args.params
        for graph_size in scala_list:
            args.exp_param = graph_size
            process(model, args, device)
    print("end scalability experiments")

def dynamic_exp(args):
    # ensure that you have trained all models under exp_mod = 1
    print("begin dynamic experiments")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        dynamic_list = args.params
        if model in args.learning_models:
            if model == 'ICS_GNN' and args.fix:
                continue
            for edge_size in dynamic_list:
                args.exp_param = edge_size
                process(model, args, device)
        elif model in args.non_learning_models:
            for edge_size in dynamic_list:
                args.exp_param = edge_size
                process(model, args, device)
    print("end dynamic experiments")

def boundary_exp(args):
    print("begin boundary experiments")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        args.train_query_file = "1_induct_train_query"
        args.train_gt_file = "1_induct_train_gt"
        args.valid_query_file = "1_induct_valid_query"
        args.valid_gt_file = "1_induct_valid_gt"
        args.test_query_file = "1_boundary_test_query"
        args.test_gt_file = "1_boundary_test_gt"
        process(model, args, device)
    print("end boundary experiments")

def noboundary_exp(args):
    print("begin noboundary experiments")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for model in args.all_models:
        args.train_query_file = "1_induct_train_query"
        args.train_gt_file = "1_induct_train_gt"
        args.valid_query_file = "1_induct_valid_query"
        args.valid_gt_file = "1_induct_valid_gt"
        args.test_query_file = "1_noboundary_test_query"
        args.test_gt_file = "1_noboundary_test_gt"
        process(model, args, device)
    print("end noboundary experiments")


def _compute_rank_quantiles(values):
    if not values:
        return None, None
    unique_vals = sorted(set(values))
    
    if len(unique_vals) <= 5:
        thresholds = unique_vals
    else:
        n_unique = len(unique_vals)
        indices = [int(n_unique * q) - 1 for q in [0.2, 0.4, 0.6, 0.8, 1.0]]
        indices = [max(0, idx) for idx in indices] 
        indices[-1] = n_unique - 1
        thresholds = [unique_vals[idx] for idx in indices]
    
    labels = [f"{t:.3f}" for t in thresholds]
    return thresholds, labels

def _cumulative_mean(pairs, thresholds):
    results = []
    for thr in thresholds:
        total = 0.0
        cnt = 0
        for v, m in pairs:
            if v <= thr:
                total += m
                cnt += 1
        avg = total / cnt if cnt > 0 else np.nan
        results.append(round(avg, 4) if not np.isnan(avg) else np.nan)
    return results

def _analyze_metric_performance(metric_name, query_to_metric, method_to_f1, all_methods):
    valid_values = [v for v in query_to_metric.values() 
                    if v != float('inf') and not np.isnan(v)]
    
    if not valid_values:
        print(f"[WARN] No valid {metric_name} values")
        return None
    
    rank_thresholds, rank_labels = _compute_rank_quantiles(valid_values)
    
    if rank_thresholds is None:
        return None
    
    print(f"[INFO] {metric_name} rank quantiles: {rank_labels}")
    
    method_pairs = {}
    for method in all_methods:
        pairs = []
        if method in method_to_f1:
            for q_str, f1 in method_to_f1[method].items():
                if q_str in query_to_metric:
                    metric_val = query_to_metric[q_str]
                    if metric_val != float('inf') and not np.isnan(metric_val) and not np.isnan(f1):
                        pairs.append((metric_val, f1))
        method_pairs[method] = pairs
    
    all_metric_values = sorted(valid_values)
    
    rank_cum_counts = []
    print(f"\n[INFO] {metric_name} - Rank quantile sample counts:")
    print(f"  Cumulative (<= threshold):")
    for idx, thr in enumerate(rank_thresholds):
        count = sum(1 for v in all_metric_values if v <= thr)
        rank_cum_counts.append(count)
        print(f"    <= {rank_labels[idx]}: {count} samples")
    
    rank_cum_labels = [f"{label}_{count}" for label, count in zip(rank_labels, rank_cum_counts)]
    
    records_rank_cum = []
    
    for method in all_methods:
        pairs = method_pairs[method]
        if not pairs:
            record = {"method": method}
            for label in rank_cum_labels:
                record[label] = np.nan
            records_rank_cum.append(record)
            continue
        
        pairs.sort(key=lambda x: x[0])
        
        rank_cum_vals = _cumulative_mean(pairs, rank_thresholds)
        record = {"method": method}
        for label, val in zip(rank_cum_labels, rank_cum_vals):
            record[label] = val
        records_rank_cum.append(record)
    
    rank_cum_df = pd.DataFrame(records_rank_cum, columns=["method"] + rank_cum_labels)
    
    return rank_cum_df

def query_best(result_dir, dataset, all_methods, metric='f1', exclude_mod=0):
    exp_mod = 1
    exp_param = 0
    
    print(f"\n[INFO] Processing dataset: {dataset}")
    method_dfs = {}
    
    for method in all_methods:
        result_path = os.path.join(result_dir, f"{dataset}_{method}_{exp_mod}_{exp_param}.xlsx")
        if not os.path.exists(result_path):
            print(f"[WARN] Result file not found: {result_path}")
            continue
        
        try:
            df = pd.read_excel(result_path)
            if "query" not in df.columns:
                print(f"[ERROR] Missing 'query' column in {result_path}")
                continue
            if metric not in df.columns:
                print(f"[WARN] Missing '{metric}' column in {result_path}, skipping")
                continue
            
            df = df.set_index("query")
            method_dfs[method] = df
            print(f"[INFO] Loaded {method}: {len(df)} queries")
        except Exception as e:
            print(f"[ERROR] Failed to load {result_path}: {e}")
            continue
    
    if not method_dfs:
        print(f"[ERROR] No valid results found for dataset {dataset}")
        return
    
    all_queries = set()
    for df in method_dfs.values():
        all_queries.update(df.index)
    
    print(f"[INFO] Total queries: {len(all_queries)}")
    
    best_model_count = {method: 0 for method in method_dfs.keys()}
    query_best_models = {}
    excluded_queries = 0
    
    for query in all_queries:
        best_value = -1
        best_models = []
        
        for method, df in method_dfs.items():
            if query in df.index:
                value = df.loc[query, metric]
                if pd.notna(value):
                    if value > best_value:
                        best_value = value
                        best_models = [method]
                    elif value == best_value:
                        best_models.append(method)
        
        if exclude_mod == 1 and len(best_models) > 1:
            excluded_queries += 1
            continue
        
        if best_models:
            for method in best_models:
                best_model_count[method] += 1
            query_best_models[query] = best_models
    
    valid_queries = len(all_queries) - excluded_queries
    
    total_best_count = sum(list(best_model_count.values()))
    
    print(f"\n[INFO] Best model statistics for dataset {dataset} (metric: {metric}):")
    print(f"  Total queries: {len(all_queries)}")
    if exclude_mod == 1:
        print(f"  Excluded queries (multiple best models): {excluded_queries}")
    print(f"  Valid queries: {valid_queries}")
    print(f"  Total best counts: {total_best_count}")
    print(f"  Best model counts:")
    sorted_counts = sorted(best_model_count.items(), key=lambda x: x[1], reverse=True)
    for method, count in sorted_counts:
        percentage = (count / total_best_count * 100) if total_best_count > 0 else 0
        print(f"    {method}: {count} ({percentage:.2f}%)")
    
    summary_data = []
    for method in all_methods:
        count = best_model_count.get(method, 0)
        percentage = (count / total_best_count * 100) if total_best_count > 0 else 0
        summary_data.append({
            'method': method,
            'best_count': count,
            'percentage': round(percentage, 3)
        })
    summary_df = pd.DataFrame(summary_data)
    out_path = os.path.join(result_dir, f"{dataset}_best_{exp_mod}_{exp_param}.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"  Saved statistics to: {out_path}")

def gt_metrics(args, result_dir, dataset, all_methods, metric='f1', exp_mod=1, exp_param=0):
    print(f"\n{'='*60}")
    print(f"[INFO] Analyzing GT metrics for dataset: {dataset} (metric: {metric})")
    print(f"{'='*60}")
    
    args.dataset = dataset
    args.exp_mod = exp_mod
    args.exp_param = exp_param
    
    args.train_query_file = "1_induct_train_query"
    args.train_gt_file = "1_induct_train_gt"
    args.valid_query_file = "1_induct_valid_query"
    args.valid_gt_file = "1_induct_valid_gt"
    args.test_query_file = "1_induct_test_query"
    args.test_gt_file = "1_induct_test_gt"
    
    try:
        graph, node_id_map, N = read_feat_graph(args, return_feature=False)
    except Exception as e:
        print(f"[ERROR] Failed to read graph for dataset {dataset}: {e}")
        return
    
    try:
        test_queries, test_gts = split_data(args, N, node_id_map, graph,
                                          return_list=True, return_query='test')
    except Exception as e:
        print(f"[ERROR] Failed to read Q/GT for dataset {dataset}: {e}")
        return
    
    if not test_queries or not test_gts:
        print(f"[WARN] Empty test queries or groundtruth for dataset {dataset}")
        return
    
    assert len(test_queries) == len(test_gts)
    
    query_to_size = {}
    query_to_density = {}
    
    for query, gt in zip(test_queries, test_gts):
        str_query = '_'.join(str(graph.nodes[node]['old_id']) for node in query)
        nodes_set = set(gt)
        
        size = len(gt)
        query_to_size[str_query] = size
        
        if len(gt) <= 1:
            dens = 0.0
        else:
            sub = graph.subgraph(nodes_set)
            m = sub.number_of_edges()
            n = len(gt)
            dens = 2.0 * m / (n * (n - 1))
        query_to_density[str_query] = dens
    
    method_to_f1 = {}
    for method in all_methods:
        result_path = os.path.join(result_dir, f"{dataset}_{method}_{exp_mod}_{exp_param}.xlsx")
        if not os.path.exists(result_path):
            continue
        
        try:
            df = pd.read_excel(result_path, dtype={"query": str})
        except Exception as e:
            print(f"[ERROR] Failed to load {result_path}: {e}")
            continue
        
        if "query" not in df.columns or metric not in df.columns:
            continue
        
        df["query"] = df["query"].astype(str)
        df["query"] = df["query"].str.replace(r'\.0$', '', regex=True)
        
        f1_dict = {}
        for _, row in df.iterrows():
            q_str = str(row["query"]).strip()
            value = row[metric]
            if pd.notna(value):
                f1_dict[q_str] = float(value)
        
        if f1_dict:
            method_to_f1[method] = f1_dict
    
    metrics_data = {
        'size': query_to_size,
        'density': query_to_density
    }
    
    for metric_name, query_to_metric in metrics_data.items():
        print(f"\n[INFO] Processing {metric_name}...")
        result = _analyze_metric_performance(metric_name, query_to_metric, method_to_f1, all_methods)
        
        if result is None:
            print(f"[WARN] Skipping {metric_name} due to no valid data")
            continue
        
        rank_cum_sorted = result.sort_values('method').reset_index(drop=True)
        
        method_order = {method: idx for idx, method in enumerate(all_methods)}
        rank_cum_sorted['_sort_order'] = rank_cum_sorted['method'].map(method_order)
        rank_cum_sorted = rank_cum_sorted.sort_values('_sort_order').drop('_sort_order', axis=1).reset_index(drop=True)
        
        out_path = os.path.join(result_dir, f"{dataset}_gt_{metric_name}_{exp_mod}_{exp_param}.csv")
        rank_cum_sorted.to_csv(out_path, index=False)
        print(f"[INFO] Saved {metric_name} analysis to: {out_path}")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    
    if args.eval_bestQ or args.eval_gt_property:
        all_methods = ["QDC", "LM", "SGM", "DMCS", "PPR", "kcore", "ktruss", "kclique", "kecc", 'ICS_GNN', 'QD_GNN', 'CommunityAF', 'CommunityDF','COCLEP','CSFormer','TransZero']
        result_dir = args.result_path
        
        if args.eval_bestQ:
            query_best(result_dir, args.dataset, all_methods, metric='f1', exclude_mod=0)
        
        if args.eval_gt_property:
            gt_metrics(args, result_dir, args.dataset, all_methods, metric='f1', exp_mod=1, exp_param=0)
        return
    
    if args.exp_mod == 1:
        inductive_exp(args)
    elif args.exp_mod == 3:
        if len(args.params) == 1 and args.params[0] == -1:
            args.params = [2, 4, 6, 8]
        querysize_exp(args)
    elif args.exp_mod == 4:
        if len(args.params) == 1 and args.params[0] == -1:
            args.params = [20, 40, 60, 80]
        scalability_exp(args)
    elif args.exp_mod == 5:
        if len(args.params) == 1 and args.params[0] == -1:
            args.params = [10, 20, 30]
        dynamic_exp(args)
    elif args.exp_mod == 6:
        if args.dataset not in args.dataset_snap:
            attr_exp(args)
    elif args.exp_mod == 9:
        boundary_exp(args)
    elif args.exp_mod == 10:
        noboundary_exp(args)

if __name__ == "__main__":
    main()

