import os
import sys
from typing import List, Set, Dict
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from models.non_learning.index_manager import IndexManager

try:
    from ktruss import MyGraph, TecIndexSB, MyEdge
    KTRUSS_AVAILABLE = True
except ImportError:
    print("[KTRUSS] Warning: ktruss module not available")
    KTRUSS_AVAILABLE = False
    class MyGraph:
        def __init__(self):
            self.g = {}
            self.number_of_edge = 0
            
    class TecIndexSB:
        def __init__(self):
            pass
            
    class MyEdge:
        def __init__(self, s, t):
            self.s = s
            self.t = t

class KTrussGraph:
    def __init__(self, index_data=None):
        if index_data:
            self.my_graph = index_data.get('my_graph')
            self.tec_index = index_data.get('tec_index')
            self.trussd = index_data.get('trussd')
            self.klistdict = index_data.get('klistdict')
            self.graph_path = index_data.get('graph_path')
            if self.my_graph:
                self.n_nodes = len(self.my_graph.g)
                self.graph = {v: set(self.my_graph.g[v].keys()) for v in self.my_graph.g}
            else:
                self.graph = {}
                self.n_nodes = 0
        else:
            self.my_graph = None
            self.tec_index = None
            self.trussd = None
            self.klistdict = None
            self.graph = {}
            self.n_nodes = 0
            self.graph_path = None
    
    def load_from_file(self, path):
        """Load graph from file and build index"""
        if not KTRUSS_AVAILABLE:
            print("[KTRUSS] Cannot load graph: ktruss module not available")
            return
            
        self.graph_path = path
        self.my_graph = MyGraph()
        self.my_graph.read_graph_edgelist(path)
        
        self.klistdict = {}
        self.trussd = {}
        self.klistdict = self.my_graph.compute_truss("", self.trussd)
        
        self.tec_index = TecIndexSB()
        self.tec_index.construct_index(self.klistdict, self.trussd, self.my_graph)
        
        self.graph = {v: set(self.my_graph.g[v].keys()) for v in self.my_graph.g}
        self.n_nodes = len(self.my_graph.g)
    
    def find_k_truss_community(self, query_node, k=3):
        if not self.tec_index or not self.my_graph:
            return set()
        
        try:
            communities_edges = self.tec_index.find_k_community_for_query(query_node, k)
            
            if not communities_edges:
                return {query_node}
            
            community_nodes = set()
            for community_edges in communities_edges:
                for edge in community_edges:
                    if hasattr(edge, 's') and hasattr(edge, 't'):
                        community_nodes.add(edge.s)
                        community_nodes.add(edge.t)
            
            if community_nodes:
                community_nodes.add(query_node)
                return community_nodes
            else:
                return {query_node}
                
        except Exception as e:
            print(f"[KTRUSS] Error in find_k_truss_community: {e}")
            return {query_node}


class KTrussWrapper(NonLearningBase):
    
    def __init__(self, method_name: str = 'ktruss'):
        super().__init__(method_name)
        self.k_truss_graph = None
        self.k = 3
        self.max_truss = 10
        self.index_manager = IndexManager()
        self.dataset_name = None
        self.force_rebuild_index = False  # Force rebuild index flag

    def load_graph(self, graph_path: str, dataset_name: str = None) -> bool:
        try:
            self.graph_path = graph_path
            self.dataset_name = dataset_name or os.path.basename(os.path.dirname(graph_path))
            
            if not self.force_rebuild_index and self.index_manager.index_exists('ktruss', self.dataset_name):
                success, index_data = self.index_manager.load_index('ktruss', self.dataset_name)
                if success and index_data and index_data.get('my_graph'):
                    self.k_truss_graph = KTrussGraph(index_data)
                    self.n_nodes = len(self.k_truss_graph.my_graph.g)
                    self.max_truss = index_data['stats'].get('max_truss', 10)
                    print(f"[KTRUSS] Loaded pre-built index for {self.dataset_name}")
                    print(f"[KTRUSS] Graph has {self.n_nodes} nodes, max truss: {self.max_truss}")
                    return True
            
            print(f"[KTRUSS] Index not found, building new index for {self.dataset_name}...")
            self.k_truss_graph = KTrussGraph()
            self.k_truss_graph.load_from_file(graph_path)
            self.n_nodes = self.k_truss_graph.n_nodes
            
            success, stats = self.index_manager.build_and_save_index('ktruss', self.dataset_name, graph_path)
            if success:
                self.max_truss = stats.get('max_truss', 10)
                print(f"[KTRUSS] Index saved successfully, max truss: {self.max_truss}")
            
            return True
        except Exception as e:
            print(f"Error loading graph for K-Truss: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.k_truss_graph:
            return set()
        
        k = round(kwargs.get('k', self.k))
        silent_mode = kwargs.get('silent_mode', False)
        
        try:
            if not query_nodes:
                return set()
            
            primary_query = query_nodes[0]
            
            if hasattr(self.k_truss_graph, 'graph') and primary_query not in self.k_truss_graph.graph:
                if not silent_mode:
                    print(f"[KTRUSS WARNING] Query node {primary_query} not found in graph, returning query nodes")
                return set(query_nodes)
            
            community = self.k_truss_graph.find_k_truss_community(primary_query, k)
            
            return community if isinstance(community, set) else set(community)
                
        except Exception as e:
            if not silent_mode:
                print(f"Error in K-Truss community search: {e}")
            return set(query_nodes) if query_nodes else set()
    
    def get_default_parameters(self) -> Dict:
        return {
            'k': 3,
            'max_truss': 10,
            'triangle_threshold': 1
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'k' in params:
            self.k = params['k']
        if 'max_truss' in params:
            self.max_truss = params['max_truss']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'ktruss',
            'full_name': 'K-Truss Community Search',
            'description': 'Community search based on k-truss decomposition and triangle connectivity',
            'supports_multiple_queries': False, 
            'requires_index': True,
            'parameters': self.get_default_parameters()
        }
    
    def build_index_only(self, graph_path: str, dataset_name: str):
        try:
            success, stats = self.index_manager.build_and_save_index('ktruss', dataset_name, graph_path)
            return success
        except Exception as e:
            print(f"Error building K-Truss index: {e}")
            return False