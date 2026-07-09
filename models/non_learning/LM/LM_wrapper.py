import os
import sys
import tempfile
from typing import List, Set, Dict
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from greedy_algorithm import GREEDYALGORITHM
from file import ReadGraph, ReadQueries, WriteCommunity


class LMWrapper(NonLearningBase):
    def __init__(self, method_name: str = 'LM'):
        super().__init__(method_name)
        self.algorithm = None
        self.k = 50
        self.max_neighbors = 1000
        
    def load_graph(self, graph_path: str) -> bool:
        try:
            self.graph_path = graph_path
            self.algorithm = GREEDYALGORITHM()
            converted_graph_path = self._convert_graph_format(graph_path)
            ReadGraph(converted_graph_path, self.algorithm)
            self.n_nodes = self.algorithm.n
            if converted_graph_path != graph_path:
                try:
                    os.unlink(converted_graph_path)
                except:
                    pass
            
            return True
        except Exception as e:
            print(f"Error loading graph for LM: {e}")
            return False
    
    def _convert_graph_format(self, input_path: str) -> str:
        try:
            edges = []
            nodes = set()
            
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                u, v = int(parts[0]), int(parts[1])
                                if u != v:
                                    edges.append((u, v))
                                    nodes.add(u)
                                    nodes.add(v)
                            except ValueError:
                                continue
            
            node_mapping = {node: i+1 for i, node in enumerate(sorted(nodes))}
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_path = temp_file.name
                
                temp_file.write(f"{len(nodes)} {len(edges)}\n")
                
                for u, v in edges:
                    new_u = node_mapping[u]
                    new_v = node_mapping[v]
                    temp_file.write(f"{new_u} {new_v}\n")
            
            self.reverse_mapping = {v: k for k, v in node_mapping.items()}
            self.node_mapping = node_mapping
            
            return temp_path
            
        except Exception as e:
            print(f"Error converting graph format: {e}")
            return input_path
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.algorithm:
            return set()
        
        if self.max_com_size is not None:
            self.k = self.max_com_size
        k = kwargs.get('k', self.k)
        max_neighbors = kwargs.get('max_neighbors', self.max_neighbors)
        
        try:
            if not query_nodes:
                return set()
            
            mapped_query_nodes = []
            for node in query_nodes:
                if hasattr(self, 'node_mapping') and node in self.node_mapping:
                    mapped_query_nodes.append(self.node_mapping[node])
            
            if not mapped_query_nodes:
                return set(query_nodes)
            
            self.algorithm.community_set.clear()
            self.algorithm.boundary_set.clear()
            self.algorithm.community_neighbour_set.clear()
            self.algorithm.k = k
            
            for node in mapped_query_nodes:
                self.algorithm.community_set.add(node)
            
            self.algorithm.init()
            self.algorithm.run()
            
            community = set(self.algorithm.community_set)
            
            if hasattr(self, 'reverse_mapping'):
                original_community = set()
                for node in community:
                    if node in self.reverse_mapping:
                        original_community.add(self.reverse_mapping[node])
                return original_community
            else:
                return community
                
        except Exception as e:
            print(f"Error in LM community search: {e}")
            return set()
    
    def get_default_parameters(self) -> Dict:
        return {
            'k': 50,
            'max_neighbors': 1000,
            'modularity_threshold': 0.1
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'k' in params:
            self.k = params['k']
        if 'max_neighbors' in params:
            self.max_neighbors = params['max_neighbors']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'LM',
            'full_name': 'Local Modularity',
            'description': 'Greedy local modularity optimization for community detection',
            'supports_multiple_queries': False,  
            'requires_index': False,
            'parameters': self.get_default_parameters()
        } 