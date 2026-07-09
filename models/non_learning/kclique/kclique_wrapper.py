import os
import sys
import gc
from typing import List, Set, Dict
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from models.non_learning.index_manager import IndexManager

try:
    from Index import KCliqueIndex
    KCLIQUE_AVAILABLE = True
    # print("[KCLIQUE] Original KCliqueIndex available from Index.py")
except ImportError as e:
    KCLIQUE_AVAILABLE = False
    print(f"[KCLIQUE] ERROR: Original KCliqueIndex not available: {e}")
    raise ImportError("K-Clique requires the original KCliqueIndex implementation from Index.py")


class KCliqueDetector:
    def __init__(self, index_data=None):
        self.kclique_index = None
        self.graph_path = None
        
        if index_data:
            self.kclique_index = index_data.get('kclique_index')
            self.graph_path = index_data.get('graph_path')
            
            if self.kclique_index:
                print(f"[KCLIQUE] Loaded KCliqueIndex from pkl: {len(self.kclique_index.all_cliques)} cliques")
            else:
                print(f"[KCLIQUE] Warning: No kclique_index in index_data")
        
    def load_from_file(self, path):
        try:
            self.graph_path = path
            self.kclique_index = KCliqueIndex(path)
            print(f"[KCLIQUE] Successfully built index from {path}")
            print(f"[KCLIQUE] Index contains {len(self.kclique_index.all_cliques)} cliques")
            return True
        except Exception as e:
            print(f"[KCLIQUE] Failed to build index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def clear_cache(self):
        if self.kclique_index:
            self.kclique_index.clear_cache()
    
    def get_memory_info(self):
        if self.kclique_index:
            return self.kclique_index.get_memory_info()
        return {'status': 'no_index'}
    
    def find_k_clique_community(self, query_nodes, k=3):
        if not self.kclique_index:
            print("[KCLIQUE] Index not available")
            return set()
        
        try:
            query_set = set(query_nodes)
            community = self.kclique_index.dcpc_query(query_set)
            
            if community:
                print(f"[KCLIQUE] Found k-clique community of size {len(community)}")
                return community
            else:
                print("[KCLIQUE] No valid k-clique community found, returning query nodes")
                return set(query_nodes)
                
        except Exception as e:
            print(f"[KCLIQUE] Error in DCPC query: {e}")
            import traceback
            traceback.print_exc()
            return set(query_nodes)
    
    def find_k_clique_community_with_k(self, query_nodes, k=3):
        if not self.kclique_index:
            print("[KCLIQUE] Index not available")
            return set()
        
        try:
            query_set = set(query_nodes)
            community = self.kclique_index.kcpc_query_with_k(query_set, k)
            
            if community:
                return community
            else:
                return set(query_nodes)
                
        except Exception as e:
            print(f"[KCLIQUE] Error in KCPC query: {e}")
            import traceback
            traceback.print_exc()
            return set(query_nodes)


class KCliqueWrapper(NonLearningBase):
    def __init__(self, method_name: str = 'kclique'):
        super().__init__(method_name)
        self.detector = None
        self.k = 3
        self.use_kcpc = True 
        self.index_manager = IndexManager()
        self.dataset_name = None
        self._query_count = 0 
        self.force_rebuild_index = False 

    def load_graph(self, graph_path: str, dataset_name: str = None) -> bool:
        try:
            self.graph_path = graph_path
            self.dataset_name = dataset_name or os.path.basename(os.path.dirname(graph_path))
            if not self.force_rebuild_index and self.index_manager.index_exists('kclique', self.dataset_name):
                print(f"[KCLIQUE] Loading existing index for {self.dataset_name}...")
                success, index_data = self.index_manager.load_index('kclique', self.dataset_name)
                if success and index_data and index_data.get('kclique_index'):
                    self.detector = KCliqueDetector(index_data)
                    self.n_nodes = self.detector.kclique_index.graph.n
                    print(f"[KCLIQUE] Loaded pre-built index for {self.dataset_name}")
                    print(f"[KCLIQUE] Graph has {self.n_nodes} nodes, {len(self.detector.kclique_index.all_cliques)} cliques")
                    return True
            
            print(f"[KCLIQUE] Index not found, building new index for {self.dataset_name}...")
            print(f"[KCLIQUE] (Recommended: run phase='train' separately for better control)")
            self.detector = KCliqueDetector()
            success = self.detector.load_from_file(graph_path)
            
            if success:
                self.n_nodes = self.detector.kclique_index.graph.n
                print(f"[KCLIQUE] Successfully loaded graph with {self.n_nodes} nodes")
                
                success, stats = self.index_manager.build_and_save_index('kclique', self.dataset_name, graph_path)
                if success:
                    print(f"[KCLIQUE] Index saved successfully")
                
                return True
            else:
                print(f"[KCLIQUE] Failed to build index")
                return False
                
        except Exception as e:
            print(f"Error loading graph for K-Clique: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.detector:
            print("[KCLIQUE] Detector not initialized")
            return set()
        k = round(kwargs.get('k', self.k))
        use_kcpc = kwargs.get('use_kcpc', True)
        silent_mode = kwargs.get('silent_mode', False)
        
        self._query_count += 1
        
        try:
            if not query_nodes:
                return set()
            if hasattr(self.detector, 'kclique_index') and hasattr(self.detector.kclique_index, 'graph'):
                graph = self.detector.kclique_index.graph
                if hasattr(graph, 'adj'):
                    valid_query_nodes = []
                    for node in query_nodes:
                        if node < len(graph.adj) and graph.adj[node]: 
                            valid_query_nodes.append(node)
                        elif not silent_mode:
                            print(f"[KCLIQUE WARNING] Query node {node} not found in graph, skipping")
                    
                    if not valid_query_nodes:
                        if not silent_mode:
                            print(f"[KCLIQUE WARNING] No valid query nodes found in graph, returning query nodes")
                        return set(query_nodes)
                    
                    query_nodes = valid_query_nodes
            
            if use_kcpc:
                community = self.detector.find_k_clique_community_with_k(query_nodes, k)
            else:
                community = self.detector.find_k_clique_community(query_nodes, k)
            
            return community if isinstance(community, set) else set(community)
                
        except Exception as e:
            if not silent_mode:
                print(f"Error in K-Clique community search: {e}")
                import traceback
                traceback.print_exc()
            return set(query_nodes) if query_nodes else set()
    
    def get_default_parameters(self) -> Dict:
        return {
            'k': 3,
            'use_kcpc': False 
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'k' in params:
            self.k = params['k']
        if 'use_kcpc' in params:
            self.use_kcpc = params['use_kcpc']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'kclique',
            'full_name': 'K-Clique Percolation',
            'description': 'Index-based densest clique percolation community search using original Index.py implementation',
            'supports_multiple_queries': True,
            'requires_index': True,
            'parameters': self.get_default_parameters()
        }
    
    def get_memory_info(self) -> Dict:
        if self.detector:
            info = self.detector.get_memory_info()
            info['query_count'] = self._query_count
            return info
        return {'status': 'no_detector', 'query_count': self._query_count}
    
    def clear_cache(self):
        if self.detector:
            self.detector.clear_cache()
        gc.collect()
    
    def build_index_only(self, graph_path: str, dataset_name: str):
        try:
            success, stats = self.index_manager.build_and_save_index('kclique', dataset_name, graph_path)
            return success
        except Exception as e:
            print(f"Error building K-Clique index: {e}")
            return False