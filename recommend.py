import gc
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
import networkx as nx
import torch
import os
import random
import psutil
from models.recommend.RecCS import recommend_train, recommend_test
from utils import *
import copy
import time
from datetime import datetime
import pandas as pd
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
thr_dict = {
    'ktruss': {'min_thr': [3], 'max_thr': [20], 'thr_step': [1], 'type': ['int']},
    'kclique': {'min_thr': [1], 'max_thr': [7], 'thr_step': [1], 'type': ['int']},
    'ICS_GNN': {'min_thr': [10], 'max_thr': [400], 'thr_step': [10], 'type': ['int']},
    'COCLEP': {'min_thr': [0], 'max_thr': [0.95], 'thr_step': [0.05], 'type': ['float']},
    'QD_GNN': {'min_thr': [0], 'max_thr': [0.95], 'thr_step': [0.05], 'type': ['float']},
    'CommunityAF': {'min_thr': [1,1], 'max_thr': [10,5], 'thr_step': [1,1], 'type': ['int','int']}
}

def Recgenerate_nonLearning(G, method_name, method_instance,
                    queries: List[List[int]], gts: List[Set[int]], k_dict = None):
    all_t = []
    cpu_memory = []
    gpu_memory = []
    test_communities = []
    str_queries = []
    labels = []
    thrs = []
    if k_dict is None:
        for i, query in enumerate(queries):
            t1 = time.time()
            str_query = '_'.join(str(G.nodes[node]['old_id']) for node in query)
            str_queries.append(str_query)
            def query_func(query, method_instance):
                monitor = FunctionPerformanceMonitor()
                return monitor.monitor_function(method_instance.search_community, query, silent_mode=True)
            metrics = query_func(query, method_instance)
            community = metrics['result']
            cur_cpu_memory = metrics['max_memory_mb']
            cpu_memory.append(cur_cpu_memory)
            gpu_memory.append(0)
            test_communities.append(community if community else set(query))
            labels.append(gts[i])
            print(f"[{method_name}]Found community for vertex {str_query} (query {i + 1}/{len(queries)})")
            t2 = time.time()
            t = t2 - t1
            all_t.append(t)
            thrs.append(-1)
    else:
        min_thr = k_dict[method_name]['min_thr'][0]
        max_thr = k_dict[method_name]['max_thr'][0]
        thr_step = k_dict[method_name]['thr_step'][0]
        k_values_to_test = np.arange(min_thr, max_thr + thr_step, thr_step).tolist()
        print(f"[PARAMETER TUNING] Testing k values: {k_values_to_test}")
        for i, query in enumerate(queries):
            for k in k_values_to_test:
                t1 = time.time()
                str_query = '_'.join(str(G.nodes[node]['old_id']) for node in query)
                str_queries.append(str_query)
                def query_func(query, method_instance, optimal_k):
                    monitor = FunctionPerformanceMonitor()
                    return monitor.monitor_function(method_instance.search_community, query, k = optimal_k, silent_mode=True)
                metrics = query_func(query, method_instance, k)
                community = metrics['result']
                cur_cpu_memory = metrics['max_memory_mb']
                cpu_memory.append(cur_cpu_memory)
                gpu_memory.append(0)
                test_communities.append(community if community else set(query))
                labels.append(gts[i])
                print(f"[{method_name}]Found community for vertex {str_query} with k={k} (query {i + 1}/{len(queries)})")
                t2 = time.time()
                t = t2 - t1
                all_t.append(t)
                thrs.append(k)
    return test_communities, labels, str_queries, all_t, cpu_memory, gpu_memory, thrs

def test_recommend(G, query_dict, args):
    proc_dir = args.process_path
    dataset = args.dataset
    model_to_df = {}
    for model_name in getattr(args, 'all_models', []):
        xlsx = os.path.join(proc_dir, f"{dataset}_{model_name}_recExact.xlsx")
        if os.path.exists(xlsx):
            try:
                df = pd.read_excel(xlsx, engine="openpyxl")
            except Exception:
                df = pd.read_excel(xlsx)
            model_to_df[model_name] = df

    def format_query_str(q):
        try:
            return '_'.join(str(G.nodes[node]['old_id']) for node in q)
        except Exception:
            return str(q)

    def format_thr_str(model_name, thr_val):
        if model_name not in args.thr_dict:
            return "-1"
        td = args.thr_dict[model_name]
        types = td.get('type', ['float'])
        if isinstance(thr_val, (list, tuple)):
            parts = []
            for i, v in enumerate(thr_val):
                if i < len(types) and types[i] == 'int':
                    parts.append(str(int(round(v))))
                else:
                    parts.append(f"{float(v):.2f}")
            return '_'.join(parts)
        else:
            if len(types) > 0 and types[0] == 'int':
                return str(int(round(thr_val)))
            else:
                return f"{float(thr_val):.2f}"
    
    def parse_thr_to_values(thr_str):
        if pd.isna(thr_str) or str(thr_str).strip() == '':
            return []
        try:
            parts = str(thr_str).strip().split('_')
            values = []
            for p in parts:
                try:
                    values.append(float(p))
                except ValueError:
                    continue
            return values
        except Exception:
            return []
    
    def thr_values_match(thr1_val, thr2_str, tolerance=1e-6):
        if isinstance(thr1_val, (list, tuple)):
            val1_list = [float(v) for v in thr1_val]
        else:
            val1_list = [float(thr1_val)]
        
        val2_list = parse_thr_to_values(thr2_str)
        
        if len(val1_list) != len(val2_list):
            return False
        
        for v1, v2 in zip(val1_list, val2_list):
            if abs(v1 - v2) > tolerance:
                return False
        return True

    per_query = {}
    for model_name, query_data in query_dict.items():
        df = model_to_df.get(model_name)
        if df is None:
            print(f"recExact not found for model {model_name}")
            continue
        for (thr, test_query, test_gt, rec_time) in query_data:
            qstr = format_query_str(test_query)
            if 'query' not in df.columns or 'thr' not in df.columns:
                print(f"recExact for {model_name} missing columns")
                continue
            
            query_rows = df[df['query'].astype(str) == qstr]
            if query_rows.empty:
                print(f"Not found in recExact: model={model_name}, query={qstr}")
                continue
            
            model_needs_thr_match = model_name in getattr(args, 'thr_dict', {})
            
            matched_row = None
            if not model_needs_thr_match:
                for idx, row in query_rows.iterrows():
                    thr_str_val = str(row['thr']).strip()
                    if thr_str_val == '-1' or thr_str_val == '-1.0':
                        matched_row = row
                        break
                if matched_row is None:
                    matched_row = query_rows.iloc[0]
            else:
                for idx, row in query_rows.iterrows():
                    if thr_values_match(thr, row['thr']):
                        matched_row = row
                        break
                
                if matched_row is None:
                    thr_str = format_thr_str(model_name, thr)
                    print(f"Threshold not matched in recExact: model={model_name}, query={qstr}, "
                          f"recommended_thr={thr}, formatted_thr={thr_str}")
                    continue
            
            row = matched_row

            def get_float(series, candidates, default=0.0):
                for name in candidates:
                    if name in series and pd.notna(series[name]):
                        try:
                            return float(series[name])
                        except Exception:
                            pass
                return default

            f1_val = get_float(row, ['f1'])
            pre_val = get_float(row, ['pre'])
            rec_val = get_float(row, ['rec'])
            jac_val = get_float(row, ['jac'])
            nmi_val = get_float(row, ['nmi'])
            ari_val = get_float(row, ['ari'])
            degree_val = get_float(row, ['degree'])
            density_val = get_float(row, ['density'])
            diameter_val = get_float(row, ['diameter'])
            conductance_val = get_float(row, ['conductance'])
            com_size_val = get_float(row, ['com_size'])
            modularity_val = get_float(row, ['modularity'])
            density_modu_val = get_float(row, ['density_modu'])
            local_modu_val = get_float(row, ['local_modu'])
            sub_modu_val = get_float(row, ['sub_modu'])
            edge_surplus_val = get_float(row, ['edge_surplus'])
            time_val = get_float(row, ['time'])
            cpu_val = get_float(row, ['cpu_memory'])
            gpu_val = get_float(row, ['gpu_memory'])
            key = tuple(test_query)
            if key not in per_query:
                per_query[key] = {
                    'query_str': qstr,
                    'cands': [],
                    'rec_time': rec_time,
                    'time':0.0,
                    'cpu': cpu_val,
                    'gpu': gpu_val
                }
            per_query[key]['cands'].append({
                'model': model_name,
                'thr': thr,
                'f1': f1_val,
                'pre': pre_val,
                'rec': rec_val,
                'jac': jac_val,
                'nmi': nmi_val,
                'ari': ari_val,
                'degree': degree_val,
                'density': density_val,
                'diameter': diameter_val,
                'conductance': conductance_val,
                'com_size': com_size_val,
                'modularity': modularity_val,
                'density_modu': density_modu_val,
                'local_modu': local_modu_val,
                'sub_modu': sub_modu_val,
                'edge_surplus': edge_surplus_val,
            })
            per_query[key]['time'] += time_val
            per_query[key]['cpu'] = max(per_query[key]['cpu'], cpu_val)
            per_query[key]['gpu'] = max(per_query[key]['gpu'], gpu_val)

    all_queries, all_t, all_cpu, all_gpu, all_rec_t, all_model_name, all_thrs = [], [], [], [], [], [], []
    f1_list, pre_list, rec_list = [], [], []
    jac_list, nmi_list, ari_list = [], [], []
    degree_list, density_list, diameter_list = [], [], []
    conductance_list, com_size_list, modularity_list = [], [], []
    density_modu_list, local_modu_list, sub_modu_list, edge_surplus_list = [], [], [], []

    for key, acc in per_query.items():
        if len(acc['cands']) == 0:
            continue
        best_idx = int(np.argmax([c['f1'] for c in acc['cands']]))
        best = acc['cands'][best_idx]
        all_queries.append(acc['query_str'])
        all_rec_t.append(acc['rec_time'])
        all_t.append(acc['time'])
        all_model_name.append(best['model'])
        model_name = best['model']
        thr_val = best['thr']
        if model_name in getattr(args, 'thr_dict', {}):
            all_thrs.append(format_thr_str(model_name, thr_val))
        else:
            all_thrs.append(str(thr_val))
        f1_list.append(best['f1'])
        pre_list.append(best['pre'])
        rec_list.append(best['rec'])
        jac_list.append(best.get('jac', 0.0))
        nmi_list.append(best.get('nmi', 0.0))
        ari_list.append(best.get('ari', 0.0))
        degree_list.append(best.get('degree', 0.0))
        density_list.append(best.get('density', 0.0))
        diameter_list.append(best.get('diameter', 0.0))
        conductance_list.append(best.get('conductance', 0.0))
        com_size_list.append(best.get('com_size', 0.0))
        modularity_list.append(best.get('modularity', 0.0))
        density_modu_list.append(best.get('density_modu', 0.0))
        local_modu_list.append(best.get('local_modu', 0.0))
        sub_modu_list.append(best.get('sub_modu', 0.0))
        edge_surplus_list.append(best.get('edge_surplus', 0.0))
        all_cpu.append(acc['cpu'])
        all_gpu.append(acc['gpu'])

    return (
        all_t,
        all_queries,
        pre_list,
        rec_list,
        f1_list,
        jac_list,
        nmi_list,
        ari_list,
        degree_list,
        density_list,
        diameter_list,
        conductance_list,
        com_size_list,
        modularity_list,
        density_modu_list,
        local_modu_list,
        sub_modu_list,
        edge_surplus_list,
        all_cpu,
        all_gpu,
        all_rec_t,
        all_model_name,
        all_thrs,
    )

def recommend_query(args):
    # initialize parameters
    process = psutil.Process(os.getpid())
    base_cpu_memory = process.memory_info().rss / 1024 / 1024
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.thr_dict = thr_dict

    feature, G, node_id_map, N = read_feat_graph(args)
    test_queries, test_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'test')
    rec_model, data_time, train_time, train_cpu_memory, train_gpu_memory, model_size, model_param_num = recommend_train(args, G, feature, device)
    train_cpu_memory = train_cpu_memory - base_cpu_memory if train_cpu_memory > 0 else 0

    test_time = 0
    rec_cpu_memory = 0
    rec_gpu_memory = 0
    if args.phase == 'test' or args.phase == 'all':
        # recommend model (top-k)
        test_start = time.time()
        query_dict = {}
        top_k = int(getattr(args, 'rec_topk', 5))
        for i, query in enumerate(test_queries):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            def recommend_func():
                monitor = FunctionPerformanceMonitor()
                return monitor.monitor_function(recommend_test, G, feature, query, args, rec_model, device, k=top_k)
            metrics = recommend_func()
            outputs = metrics['result']
            rec_cpu_memory = max(rec_cpu_memory, metrics['max_memory_mb'])
            rec_gpu_memory = max(rec_gpu_memory, get_gpu_memory())
            t2 = time.time()
            rec_time = t2 - t1
            for (pre_md, pred_thr, prob) in outputs:
                # print(f"recommend model: {pre_md} (prob={prob:.4f})")
                # print(f"recommend parameter: {pred_thr}")
                if pre_md not in query_dict:
                    query_dict[pre_md] = []
                norm_pred_thr = pred_thr
                if isinstance(pred_thr, (list, tuple)) and len(pred_thr) == 1:
                    norm_pred_thr = pred_thr[0]
                if pre_md in args.learning_models:
                    query_dict[pre_md].append((norm_pred_thr, test_queries[i], test_labels[i], rec_time))
                elif pre_md in args.non_learning_models:
                    query_dict[pre_md].append((norm_pred_thr, query, test_labels[i], rec_time))
        test_end = time.time()
        test_time = test_end - test_start
        rec_cpu_memory = rec_cpu_memory - base_cpu_memory if rec_cpu_memory > 0 else 0
        print("------------------end recommendation------------------")

        # evaluate the quality based on recommend exact result
        result = test_recommend(G, query_dict, args)
        save_path = args.result_path + "/" + args.dataset + f"_recommend_{args.rec_exp_mod}_{args.exp_param}"
        input_base_cpu = 0
        all_t, all_queries, pre_list, rec_list, f1_list, jac, nmi, ari, degree, density, diameter, conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, all_cpu, all_gpu, all_rec_t, all_model_name, all_thrs = result
        avg_f1 = np.mean(f1_list) if len(f1_list) > 0 else 0.0
        print(f"Average F1-score: {avg_f1:.4f}")
        print(f"Top-k parameter: {top_k}")
        # record result of recommendation + query 
        store_result_xlsx(save_path, all_t, all_queries, pre_list, rec_list, f1_list, jac, nmi, ari, degree, density, diameter, conductance,
                      com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, all_cpu, all_gpu, base_cpu= input_base_cpu, rec_time = all_rec_t, model_name = all_model_name, thrs = all_thrs)
    
    summary_path = args.result_path + f"summary_{args.dataset}_recommend_{args.rec_exp_mod}_{args.exp_param}"
    # record statistics of recommendation (do not contain query phase)
    store_summary(summary_path, args.dataset, f"recommend", data_time, train_time, 0, test_time, model_size, model_param_num, rec_cpu_memory, rec_gpu_memory, train_cpu_memory, train_gpu_memory, args.phase)

def recommend_generateLabel(args):
    # initialize parameters
    process = psutil.Process(os.getpid())
    base_cpu_memory = process.memory_info().rss / 1024 / 1024
    model_name = args.recommend_model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load graph
    G, node_id_map, N = read_feat_graph(args, return_feature = False)

    # load non-learning models
    if model_name in args.non_learning_models:
        nonLearning_args = copy.deepcopy(args)
        nonLearning_args.save_recLabel = nonLearning_args.process_path  + "/" + nonLearning_args.dataset + f"_{model_name}_{args.generate_mod}" # the file to save recommendation label
        method_instance = method_implementations[model_name]
        valid_queries, valid_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'valid')
        nonLearning_args.max_com_size = min(500, sum(len(x) for x in valid_labels) / len(valid_labels)) if valid_labels else 100
        method_instance.dataset_name = nonLearning_args.dataset
        method_instance.n_nodes = N
        method_instance.max_com_size = nonLearning_args.max_com_size
        graph_path = os.path.join(nonLearning_args.data_path, nonLearning_args.dataset, 'graph')
        
        # train model
        if nonLearning_args.phase == "all" or nonLearning_args.phase == "train":
            if model_name in nonLearning_args.index_based_methods:
                def index_func():
                    monitor = FunctionPerformanceMonitor()
                    return monitor.monitor_function(method_instance.build_index_only,graph_path, method_instance.dataset_name)
                metrics = index_func()

        if nonLearning_args.phase == "all" or nonLearning_args.phase == "test":
            # load data
            try:
                load_success = method_instance.load_graph(graph_path, nonLearning_args.dataset)
            except TypeError:
                load_success = method_instance.load_graph(graph_path)
            if args.generate_mod == 'recLabel':
                model_queries, model_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'train')
            elif args.generate_mod == 'recValid':
                model_queries, model_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'valid')
            elif args.generate_mod == 'recExact':
                model_queries, model_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'test')
            # generate recommend labels (if exists, load from file)
            if model_name in nonLearning_args.k_based_methods:
                pred_com, labels, str_queries, all_t, cpu_memory, gpu_memory, thrs = Recgenerate_nonLearning(G, model_name, method_instance, model_queries, model_labels, thr_dict)
            else:
                pred_com, labels, str_queries, all_t, cpu_memory, gpu_memory, thrs = Recgenerate_nonLearning(G, model_name, method_instance, model_queries, model_labels)

            # evaluate and store the quality
            gt_size = [len(label) for label in labels]
            pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus = evaluate_communities(G, pred_com, labels)
            file_path = nonLearning_args.save_recLabel
            store_result_xlsx(file_path, all_t, str_queries, pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance,com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, cpu_memory, gpu_memory, thrs, base_cpu=base_cpu_memory, gt_size = gt_size)

    # load learning models
    elif model_name in args.learning_models:
        model_param = model_name + "_parameter"
        model_args = globals()[model_param](args)
        model_data = model_name + "_DataLoader"

        md = None
        model_train = model_name + "_train"
        if model_name != 'ICS_GNN':
            if model_args.phase == "all" or model_args.phase == "train":
                trainloader, validloader, testloader = globals()[model_data](model_args, model_args.phase)
                md, train_cpu_memory, train_gpu_memory = globals()[model_train](model_args, trainloader, device)
                del trainloader, validloader, testloader
                gc.collect()
            else:
                md, train_cpu_memory, train_gpu_memory = globals()[model_train](model_args, None, device)

        if model_args.phase == "all" or model_args.phase == "test":
            model_recommend = model_name + "_RecGenerate"
            if args.generate_mod == 'recLabel':
                model_queries, model_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'train')
            elif args.generate_mod == 'recValid':
                model_queries, model_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'valid')
            elif args.generate_mod == 'recExact':
                model_queries, model_labels = split_data(args, N, node_id_map, G, return_list = True, return_query = 'test')
            g, pred_com, labels, str_queries, all_t, cpu_memory, gpu_memory, thrs = globals()[model_recommend](model_args, model_queries, model_labels, md, device)
            # evaluate and store the quality
            gt_size = [len(label) for label in labels]
            pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance, com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus = evaluate_communities(g, pred_com, labels)
            file_path = os.path.join(model_args.process_path, f"{model_args.dataset}_{model_name}_{args.generate_mod}")
            store_result_xlsx(file_path, all_t, str_queries, pre, rec, f1, jac, nmi, ari, degree, density, diameter, conductance,com_size, modularity, density_modu, local_modu, sub_modu, edge_surplus, cpu_memory, gpu_memory, thrs, base_cpu=base_cpu_memory, gt_size = gt_size)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    args.train_query_file = "1_induct_train_query"
    args.train_gt_file = "1_induct_train_gt"
    args.valid_query_file = "1_induct_valid_query"
    args.valid_gt_file = "1_induct_valid_gt"
    args.test_query_file = "1_induct_test_query"
    args.test_gt_file = "1_induct_test_gt"

    if args.rec_task == 'generate':
        recommend_generateLabel(args)
    elif args.rec_task == 'recommend':
        if args.rec_exp_mod == 1:
            recommend_query(args) 
        elif args.rec_exp_mod == 2:
            args.rec_topk = args.exp_param
            recommend_query(args) 
        elif args.rec_exp_mod == 3:
            args.gnn_topn = args.exp_param  
            recommend_query(args) 
        elif args.rec_exp_mod == 11:
            args.ablation_mode = 'gnn'
            args.use_zero_gnn = True
            recommend_query(args)
        elif args.rec_exp_mod == 14:
            args.lambda_param_KL = 0
            args.lambda_model_KL = 0
            recommend_query(args)