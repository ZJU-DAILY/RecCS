import os
import sys
from typing import List, Set, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from models.non_learning.index_manager import IndexManager
from kcore import TreeIndex, Graph


class KCoreWrapper(NonLearningBase):
    def __init__(self, method_name: str = 'kcore'):
        super().__init__(method_name)
        self.tree_index = None
        self.n_nodes = 0
        self.max_core = 3
        self.k_value = 3  # Default k value
        self.min_core = 2  # Add min_core attribute
        self.index_manager = IndexManager()
        self.dataset_name = None
        self.force_rebuild_index = False  # Force rebuild index flag
    def load_graph(self, graph_path: str, dataset_name: str = None) -> bool:
        try:
            self.graph_path = graph_path
            self.dataset_name = dataset_name or os.path.basename(os.path.dirname(graph_path))
            
            experiment_type = getattr(self, 'experiment_type', 'standard')
            
            if not self.force_rebuild_index and self.index_manager.index_exists('kcore', self.dataset_name, experiment_type=experiment_type):
                success, index_data = self.index_manager.load_index('kcore', self.dataset_name, experiment_type=experiment_type)
                if success and index_data:
                    self.tree_index = index_data['tree_index']
                    self.n_nodes = index_data['stats']['n_nodes']
                    self.max_core = index_data['stats']['max_core']
                    print(f"[KCORE] Loaded pre-built index for {self.dataset_name}")
                    return True
            
            print(f"[KCORE] Building new index for {self.dataset_name}...")
            self.tree_index = TreeIndex(graph_path)
            self.n_nodes = self.tree_index.graph.n
            self.max_core = self.tree_index.graph.core_max
            
            success, stats = self.index_manager.build_and_save_index('kcore', self.dataset_name, graph_path, experiment_type=experiment_type)
            if success:
                print(f"[KCORE] Index saved successfully")
            
            return True
        except Exception as e:
            print(f"Error loading graph for K-Core: {e}")
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.tree_index:
            return set()

        min_core = kwargs.get('min_core', self.min_core)
        silent_mode = kwargs.get('silent_mode', False)
        
        try:
            if not query_nodes:
                return set()
            
            valid_query_nodes = []
            for node in query_nodes:
                if node in self.tree_index.core_index:
                    valid_query_nodes.append(node)
                elif not silent_mode:
                    print(f"[KCORE WARNING] Query node {node} not found in graph, skipping")
            
            if not valid_query_nodes:
                if not silent_mode:
                    print(f"[KCORE WARNING] No valid query nodes found in graph, returning query nodes")
                return set(query_nodes)
            
            nodes, edges = self.tree_index.search_with_shell(valid_query_nodes)
            
            if isinstance(nodes, set):
                return self._get_connected_component(nodes, query_nodes)
            elif isinstance(nodes, list):
                return self._get_connected_component(set(nodes), query_nodes)
            else:
                return self._fallback_search(valid_query_nodes, min_core)
                
        except Exception as e:
            if not silent_mode:
                print(f"Error in K-Core community search: {e}")
            return self._fallback_search(query_nodes, min_core)
    
    def _search_with_shell(self, query_nodes: List[int], k_value: int) -> Set[int]:
        try:
            if not query_nodes:
                return set()
            
            primary_query = query_nodes[0]
            result = self.tree_index.search_with_shell([primary_query])
            
            if isinstance(result, list):
                return set(result)
            elif isinstance(result, set):
                return result
            else:
                return set()
        except:
            return set()
    
    def _fallback_search(self, query_nodes: List[int], k_value: int) -> Set[int]:
        try:
            if not query_nodes:
                return set()
            query_cores = []
            for node in query_nodes:
                if node in self.tree_index.core_index:
                    core_idx = self.tree_index.core_index[node]
                    core_value = self.tree_index.core_minimum_degree.get(core_idx, 0)
                    query_cores.append(core_value)
            
            if not query_cores:
                return set(query_nodes)
            
            target_core = max(query_cores)
            
            community = set()
            for node, core_idx in self.tree_index.core_index.items():
                core_value = self.tree_index.core_minimum_degree.get(core_idx, 0)
                if core_value >= target_core:
                    community.add(node)
            
            return self._get_connected_component(community, query_nodes)
            
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return set(query_nodes) if query_nodes else set()
    
    def _get_connected_component(self, candidates: Set[int], query_nodes: List[int]) -> Set[int]:
        try:
            from collections import deque
            
            if not query_nodes or not candidates:
                return set()
            
            start_node = query_nodes[0]
            if start_node not in candidates:
                return set(query_nodes)
            
            visited = set()
            queue = deque([start_node])
            visited.add(start_node)
            
            while queue:
                current = queue.popleft()
                if current in self.tree_index.graph.adj:
                    for neighbor in self.tree_index.graph.adj[current]:
                        if neighbor in candidates and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            if all(q in visited for q in query_nodes):
                return visited
            else:
                return set(query_nodes)
            
        except:
            return set(query_nodes) if query_nodes else set()
    
    def get_default_parameters(self) -> Dict:
        return {
            'k': 3,
            'min_core': 2,
            'max_core': self.max_core if self.max_core else 10,
            'use_index': True
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'k' in params:
            self.k_value = params['k']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'kcore',
            'full_name': 'K-Core Community Search',
            'description': 'Community search based on k-core decomposition',
            'supports_multiple_queries': True,
            'requires_index': True,
            'parameters': self.get_default_parameters()
        }
    
    def build_index_only(self, graph_path: str, dataset_name: str):
        try:
            success, stats = self.index_manager.build_and_save_index('kcore', dataset_name, graph_path)
            return success
        except Exception as e:
            print(f"Error building K-Core index: {e}")
            return False