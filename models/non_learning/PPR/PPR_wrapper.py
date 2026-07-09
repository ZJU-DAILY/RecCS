import os
import sys
import time
import networkx as nx
from typing import List, Set, Dict
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from PPR import Graph, find_community_for_vertex, read_query_from_file


class PPRWrapper(NonLearningBase):
    def __init__(self, method_name: str = 'PPR'):
        super().__init__(method_name)
        self.ppr_graph = None
        self.conductance_target = 0.3
        
    def load_graph(self, graph_path: str) -> bool:
        try:
            self.graph_path = graph_path
            self.ppr_graph = Graph()
            self.ppr_graph.read_from_file(graph_path)
            self.n_nodes = self.ppr_graph.n_vertices
            return True
        except Exception as e:
            print(f"Error loading graph for PPR: {e}")
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.ppr_graph:
            return set()
        conductance_target = kwargs.get('k', self.conductance_target)
        
        query_vertex = query_nodes[0] if query_nodes else 0
        
        community = self._find_community_silent(
                query_vertex, 
                phi=conductance_target
            )
        
        if community is None:
            return {query_vertex} if query_nodes else set()
        elif isinstance(community, list):
            return set(community)
        elif isinstance(community, set):
            return community
        else:
            return {query_vertex} if query_nodes else set()
    
    def _find_community_silent(self, query_vertex: int, phi: float = None):
        from PPR import pagerank_nibble, CONDUCTANCE_TARGET
        import math
        
        start_time = time.time()
        max_runtime = 1000.0
        
        if phi is None:
            phi = CONDUCTANCE_TARGET  
            
        if query_vertex not in self.ppr_graph.neighbors:
            return {query_vertex}
        
        if self.ppr_graph.degrees.get(query_vertex, 0) == 0:
            return {query_vertex}
        
        B = math.ceil(math.log2(self.ppr_graph.n_edges)) if self.ppr_graph.n_edges > 0 else 1
        
        best_community = None
        best_conductance = float('inf')
        
        for b in range(1, B + 1):
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime:
                if best_community is not None:
                    return best_community
                else:
                    return None
            
            community, conductance = pagerank_nibble(self.ppr_graph, query_vertex, phi, b)
            
            if community is not None and conductance !=-1 and query_vertex in community:
                if conductance < best_conductance:
                    best_community = community
                    best_conductance = conductance
                
                if conductance <= phi:
                    break
        
        return best_community
    
    def get_default_parameters(self) -> Dict:
        return {
            'alpha': 0.15,
            'conductance_target': 0.3,
            'max_iterations': 10000,
            'tolerance': 1e-6
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'alpha' in params:
            self.alpha = params['alpha']
        if 'conductance_target' in params:
            self.conductance_target = params['conductance_target']
    