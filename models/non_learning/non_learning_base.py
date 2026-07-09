import os
import time
import copy
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple, Union, Optional
from collections import defaultdict
import json

class NonLearningBase(ABC):
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.graph = None
        self.graph_path = None
        self.query_nodes = None
        self.result_path = None
        self.dataset_name = None
        self.max_com_size = None
    @abstractmethod
    def load_graph(self, graph_path: str) -> bool:
        pass
    
    @abstractmethod
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict:
        pass
    
    def set_parameters(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def evaluate_community(self, predicted_community: Set[int], 
                          ground_truth_community: Set[int]) -> Dict[str, float]:
        if not predicted_community or not ground_truth_community:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 
                'jaccard': 0.0, 'nmi': 0.0
            }
        
        tp = len(predicted_community & ground_truth_community)
        precision = tp / len(predicted_community) if predicted_community else 0.0
        recall = tp / len(ground_truth_community) if ground_truth_community else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard = tp / len(predicted_community | ground_truth_community) if (predicted_community | ground_truth_community) else 0.0
        
        if hasattr(self, 'n_nodes'):
            pred_labels = [1 if node in predicted_community else 0 for node in range(self.n_nodes)]
            true_labels = [1 if node in ground_truth_community else 0 for node in range(self.n_nodes)]
            from sklearn.metrics import normalized_mutual_info_score
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
        else:
            nmi = 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'jaccard': jaccard,
            'nmi': nmi
        }
    
    def save_results(self, query_nodes: List[int], community: Set[int], 
                    execution_time: float = 0.0, metrics: Dict[str, float] = None,
                    test_id: str = None):
        output_dir = os.path.join("output", "result")
        os.makedirs(output_dir, exist_ok=True)
        
        if test_id:
            filename = f"{self.method_name}_{self.dataset_name}_{test_id}.txt"
        else:
            filename = f"{self.method_name}_{self.dataset_name}.txt"
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'a') as f:
            community_str = ' '.join(map(str, sorted(list(community))))
            f.write(f"{community_str}\n")
    
    def save_single_result(self, community: Set[int], output_path: str):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'a') as f:
            community_str = ' '.join(map(str, sorted(list(community))))
            f.write(f"{community_str}\n")


class NonLearningDataLoader:
    
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.data_path = args.data_path
        
    def load_graph_data(self) -> Tuple[str, int]:
        if self.dataset_name in self.args.dataset_pyg:
            graph_path = os.path.join(self.data_path, self.dataset_name, "graph")
        elif self.dataset_name in self.args.dataset_snap: 
            graph_path = os.path.join(self.data_path, self.dataset_name, "graph")
        else:
            dataset_dir = os.path.join(self.data_path, self.dataset_name)
            possible_files = ["graph.txt", "edges.txt", "graph", "network.txt"]
            graph_path = None
            for filename in possible_files:
                full_path = os.path.join(dataset_dir, filename)
                if os.path.exists(full_path):
                    graph_path = full_path
                    break
            
            if not graph_path:
                raise FileNotFoundError(f"Graph file not found for dataset {self.dataset_name}")
        
        data_file = self.args.data_path + "/" + self.args.dataset
        if self.args.attr == 0:
            feature = []
            feat_file = data_file + "/feat"
            with open(feat_file, 'r') as file:
                feature = np.array([[float(line.split()[1])] for line in file],
                                   dtype=np.float32)  # suppose that the feature has been normalized
        elif self.args.attr == 1:
            feat_file = data_file + "/feat_attr.npy"
            feature = np.load(feat_file)
        n_nodes = feature.shape[0]
        return graph_path, n_nodes
    
    def load_query_data(self, num_queries: int = 100) -> List[List[int]]:
        possible_query_files = ["query", "query.txt", "queries.txt"]
        for filename in possible_query_files:
            query_path = os.path.join(self.data_path, self.dataset_name, filename)
            if os.path.exists(query_path):
                queries = self._load_queries_from_file(query_path, num_queries)
                if queries:
                    print(f"[DATA] Loaded {len(queries)} queries from {filename}")
                    return queries
        
        print(f"[DATA] No predefined queries found, generating {num_queries} random queries")
        graph_path, n_nodes = self.load_graph_data()
        return self._generate_random_queries(graph_path, num_queries)
    
    def load_ground_truth(self) -> Dict[str, Set[int]]:
        gt_path = os.path.join(self.data_path, self.dataset_name, "gt")
        if os.path.exists(gt_path):
            return self._load_ground_truth_from_file(gt_path)
        return {}
    
    def _count_nodes(self, graph_path: str) -> int:
        nodes = set()
        try:
            with open(graph_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            nodes.add(int(parts[0]))
                            nodes.add(int(parts[1]))
        except:
            return 0
        return len(nodes)
    
    def _load_queries_from_file(self, query_path: str, num_queries: int) -> List[List[int]]:
        queries = []
        try:
            with open(query_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= num_queries:
                        break
                    if line.strip():
                        nodes = [int(x) for x in line.strip().split()]
                        queries.append(nodes)
        except Exception as e:
            print(f"[ERROR] Failed to load queries from {query_path}: {e}")
        return queries
    
    def _generate_random_queries(self, graph_path: str, num_queries: int) -> List[List[int]]:
        nodes = set()
        try:
            with open(graph_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            nodes.add(int(parts[0]))
                            nodes.add(int(parts[1]))
        except:
            return []
        
        if not nodes:
            return []
        
        nodes = list(nodes)
        queries = []
        
        np.random.seed(self.args.seed)
        for _ in range(num_queries):
            query_size = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            query_nodes = np.random.choice(nodes, size=query_size, replace=False).tolist()
            queries.append(query_nodes)
        
        return queries
    
    def _load_ground_truth_from_file(self, gt_path: str) -> Dict[str, Set[int]]:
        return {}


def get_method_parameters(method_name: str, args) -> Dict:
    base_params = {
        'dataset': args.dataset,
        'result_path': os.path.join(args.result_path, f"{method_name}_{args.dataset}.txt"),
        'seed': args.seed,
    }
    
    # Method-specific parameters
    method_specific_params = {
        'CD': {'omega_weights': [1.0] * 10, 'max_iterations': 100},
        'QDC': {'decay_factor': 0.9, 'method': 'auto'},
        'LM': {'k': 25000, 'max_neighbors': 10000},
        'DMCS': {'distance_threshold': 3, 'modularity_threshold': 0.1},
        'PPR': {'alpha': 0.15, 'conductance_target': 0.3},
        'OQC': {'alpha': 0.8, 'beta': 0.5},
        'kcore': {'min_core': 2, 'max_core': None},
        'ktruss': {'k': 3, 'max_truss': 10},
        'kclique': {'k': 3, 'overlap_threshold': 0.5},
        'kecc': {'k': 2, 'edge_connectivity': True}
    }
    
    if method_name in method_specific_params:
        base_params.update(method_specific_params[method_name])
    return base_params


def import_non_learning_methods():
    methods = {}
    return methods 