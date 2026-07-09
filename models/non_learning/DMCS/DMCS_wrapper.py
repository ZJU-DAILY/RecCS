import os
import sys
import tempfile
from typing import List, Set, Dict
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase
from DMCS import DMCS, MyGraph, read_query_nodes


class DMCSWrapper(NonLearningBase):

    def __init__(self, method_name: str = 'DMCS'):
        super().__init__(method_name)
        self.my_graph = None
        self.distance_threshold = 3
        self.modularity_threshold = 0.1
        
    def load_graph(self, graph_path: str) -> bool:

        try:
            self.graph_path = graph_path
            self.my_graph = MyGraph()
            self.my_graph.read_graph_from_file(graph_path)
            self.n_nodes = self.my_graph.i_num_nodes
            return True
        except Exception as e:
            print(f"Error loading graph for DMCS: {e}")
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:

        if not self.my_graph:
            return set()
        
        
        if not query_nodes:
            return set()
        
        valid_query_nodes = [node for node in query_nodes if node in self.my_graph.umap_graph]
        
        if not valid_query_nodes:
            return set(query_nodes)
        
        graph_copy = MyGraph()
        graph_copy.read_graph_from_file(self.graph_path)
        
        dmcs = DMCS(graph_copy)
        
        dmcs.get_a_community_by_fpa(valid_query_nodes)
        
        community = set()
        if hasattr(dmcs, 'returned_community') and dmcs.returned_community:
            for node, neighbors in dmcs.returned_community.items():
                if neighbors: 
                    community.add(node)
                    for neighbor in neighbors:
                        community.add(neighbor)
            
        return community
    
    def _read_community_from_output(self, output_path: str) -> Set[int]:
        community = set()
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                u, v = int(parts[0]), int(parts[1])
                                community.add(u)
                                community.add(v)
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error reading community from output file: {e}")
        
        return community
    
    def get_default_parameters(self) -> Dict:
        return {
            'distance_threshold': 3,
            'modularity_threshold': 0.1,
            'max_layers': 10,
            'use_fpa': True  # Fixed Point Algorithm
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        
        if 'distance_threshold' in params:
            self.distance_threshold = params['distance_threshold']
        if 'modularity_threshold' in params:
            self.modularity_threshold = params['modularity_threshold']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'DMCS',
            'full_name': 'Density Modularity Community Search',
            'description': 'Parameter-free data-driven community search using density modularity',
            'supports_multiple_queries': True,
            'requires_index': False,
            'parameters': self.get_default_parameters()
        } 