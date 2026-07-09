import os
import sys
import networkx as nx
from typing import List, Set, Dict
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from QDC import FreeRiderDetector, find_community_for_vertex, read_graph_from_file


class QDCWrapper(NonLearningBase):
    def __init__(self, method_name: str = 'QDC'):
        super().__init__(method_name)
        self.graph = None
        self.detector = None
        self.decay_factor = 0.9
        self.method = 'heuristic2'
        
    def load_graph(self, graph_path: str) -> bool:
        try:
            self.graph_path = graph_path
            self.graph = read_graph_from_file(graph_path)
            self.n_nodes = self.graph.number_of_nodes()
            self.detector = FreeRiderDetector(
                self.graph, 
                decay_factor=self.decay_factor
            )
            return True
        except Exception as e:
            print(f"Error loading graph for QDC: {e}")
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.graph:
            return set()
        
        decay_factor = kwargs.get('decay_factor', self.decay_factor)
        method = kwargs.get('method', self.method)
        silent_mode = kwargs.get('silent_mode', False)
        k = kwargs.get('k', self.max_com_size)

        try:
            self._silent_mode = silent_mode
            
            if not query_nodes:
                return set()
            
            valid_query_nodes = [node for node in query_nodes if node in self.graph.nodes()]
            if not valid_query_nodes:
                return set(query_nodes)
            
            current_detector = FreeRiderDetector(
                self.graph, 
                decay_factor=decay_factor
            )
            
            
            if self.max_com_size is not None:
                result = current_detector.detect_community(valid_query_nodes, method=method, k = k)
            else:
                result = current_detector.detect_community(valid_query_nodes, method=method)
            community = result.get('community', set())
            
            if community:
                community = set(community) | set(valid_query_nodes)
            
            if isinstance(community, list):
                return set(community)
            elif isinstance(community, set):
                return community
            else:
                return set(valid_query_nodes) 
                
        except Exception as e:
            print(f"Error in QDC community search: {e}")
            try:
                primary_query = valid_query_nodes[0]
                community_list, info = find_community_for_vertex(
                    self.graph, 
                    primary_query, 
                    decay_factor=decay_factor
                )
                
                if community_list:
                    return set(community_list)
                else:
                    return set([primary_query])  
            except Exception as backup_e:
                print(f"Backup method also failed: {backup_e}")
                return set(valid_query_nodes) 
    
    def get_default_parameters(self) -> Dict:
        return {
            'decay_factor': 0.9,
            'method': 'auto',
            'max_iterations': 100,
            'tolerance': 1e-6,
            'alpha': 1.0  
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'decay_factor' in params:
            self.decay_factor = params['decay_factor']
        if 'method' in params:
            self.method = params['method']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'QDC',
            'full_name': 'Query-biased Density Connected',
            'description': 'FreeRider algorithm for robust local community detection with query-biased density',
            'supports_multiple_queries': True,
            'requires_index': False,
            'parameters': self.get_default_parameters()
        } 