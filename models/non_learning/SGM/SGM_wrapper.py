import os
import sys
import time
import networkx as nx
from typing import List, Set, Dict
from pathlib import Path
from collections import deque

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.non_learning.non_learning_base import NonLearningBase


class SGMWrapper(NonLearningBase):
    
    def __init__(self, method_name: str = 'SGM'):
        super().__init__(method_name)
        self.graph = None
        self.nx_graph = None
        self.n_nodes = 0
        self.n_edges = 0
        
    def load_graph(self, graph_path: str, dataset_name: str = None) -> bool:
        try:
            self.graph_path = graph_path
            self.dataset_name = dataset_name or os.path.basename(os.path.dirname(graph_path))
            
            edges = []
            max_node = 0
            with open(graph_path, 'r') as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        u, v = map(int, line.split())
                        edges.append((u, v))
                        max_node = max(max_node, u, v)
            self.n_nodes = max_node + 1 
            self.n_edges = len(edges)
            
            self.graph = [[] for _ in range(self.n_nodes)]
            for u, v in edges:
                self.graph[u].append(v)
                self.graph[v].append(u)
            
            print(f"[SGM] Loaded graph with {self.n_nodes} nodes and {self.n_edges} edges")
            return True
        except Exception as e:
            print(f"Error loading graph for SGM: {e}")
            return False
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if not self.graph:
            return set()
        
        try:
            if not query_nodes:
                return set()
            valid_query_nodes = [node for node in query_nodes if 0 <= node < self.n_nodes]
            if not valid_query_nodes:
                return set(query_nodes)
            
            community = set(valid_query_nodes)
            
            final_community = self._local_optimal_algorithm(community)
            
            return final_community
            
        except Exception as e:
            print(f"Error in SGM community search: {e}")
            return set(valid_query_nodes) 
    
    def _local_optimal_algorithm(self, initial_community: Set[int]) -> Set[int]:
        start_time = time.time()
        max_runtime = 1000.0
        
        community = initial_community.copy()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime:
                return community
            
            improved = False
            current_modularity = self._calculate_modularity(community)
            
            neigh_set = self._calculate_neighbor_set(community)
            for degree, node in sorted(neigh_set, reverse=True):
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime:
                    return community
                
                if node not in community:
                    test_community = community | {node}
                    test_modularity = self._calculate_modularity(test_community)
                    if test_modularity > current_modularity:
                        community.add(node)
                        current_modularity = test_modularity
                        improved = True
            
            community_copy = community.copy()
            for node in community_copy:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime:
                    return community
                
                if len(community) > 1:
                    test_community = community - {node}
                    if self._is_connected(test_community):
                        test_modularity = self._calculate_modularity(test_community)
                        if test_modularity > current_modularity:
                            community.remove(node)
                            current_modularity = test_modularity
                            improved = True
            
            if not improved:
                break
        
        return community
    
    def _calculate_modularity(self, community: Set[int]) -> float:
        if not community:
            return 0.0
        
        ind = 0  
        outd = 0  
        
        for node in community:
            for neighbor in self.graph[node]:
                if neighbor in community:
                    if node < neighbor: 
                        ind += 1
                else:
                    outd += 1
        
        if outd == 0:
            return float('inf')
        
        return ind / outd
    
    def _calculate_neighbor_set(self, community: Set[int]) -> List[tuple]:
        neigh_set = set()
        for node in community:
            for neighbor in self.graph[node]:
                if neighbor not in community:
                    degree = len(self.graph[neighbor])
                    neigh_set.add((degree, neighbor))
        
        return list(neigh_set)
    
    def _is_connected(self, community: Set[int]) -> bool:
        if not community:
            return False
        if len(community) == 1:
            return True
        
        start = next(iter(community))
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            for neighbor in self.graph[current]:
                if neighbor in community and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(community)
    
    def get_default_parameters(self) -> Dict:

        return {
            'max_iterations': 100,
            'convergence_threshold': 1e-6
        }
    
    def set_parameters(self, params: Dict):
        super().set_parameters(params)
        if 'max_iterations' in params:
            self.max_iterations = params['max_iterations']
        if 'convergence_threshold' in params:
            self.convergence_threshold = params['convergence_threshold']
    
    def get_method_info(self) -> Dict:
        return {
            'name': 'SGM',
            'full_name': 'Subgraph Modularity',
            'description': 'Local optimal algorithm for community search based on subgraph modularity',
            'supports_multiple_queries': True,
            'requires_index': False,
            'parameters': self.get_default_parameters()
        } 