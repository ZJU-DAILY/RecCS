# greedy_algorithm.py

import time
from define import PII, pair_hash  
from typing import List, Set

class GREEDYALGORITHM:
    def __init__(self):
        self.n = 0  
        self.m = 0  
        self.graph: List[List[int]] = []  
        self.community_set: Set[int] = set()
        self.boundary_set: Set[int] = set()
        self.community_neighbour_set: Set[int] = set()
        self.k = 0  
        self.R = 0.0  
        self.T = 0.0  
        self.I = 0  
        self.edges_inT: Set[PII] = set()
        self.edges_inI: Set[PII] = set()  

    def cal_boundary_set(self):
        self.boundary_set.clear()
        for u in self.community_set:
            for neighbor in self.graph[u]:
                if neighbor not in self.community_set:
                    self.boundary_set.add(u)
                    break
        # print("boundary_set:")
        # self.print(self.boundary_set)

    def cal_community_neighbour_set(self):
        self.community_neighbour_set.clear()
        for u in self.community_set:
            for neighbor in self.graph[u]:
                if neighbor not in self.community_set:
                    self.community_neighbour_set.add(neighbor)
        # print("community_neighbour_set:")
        # self.print(self.community_neighbour_set)

    def cal_local_modularity(self):
        self.edges_inT.clear()
        edges_inI: Set[PII] = set()
        for bd_node in self.boundary_set:
            for neighbor in self.graph[bd_node]:
                u = min(bd_node, neighbor)
                v = max(bd_node, neighbor)
                self.edges_inT.add((u, v))
                if neighbor in self.community_set:
                    edges_inI.add((u, v))
        self.T = len(self.edges_inT)
        self.I = len(edges_inI)
        self.edges_inI = edges_inI
        self.R = self.I / self.T if self.T != 0 else 0.0

    def cal_delta_local_modularity(self, vj: int) -> float:
        x = 0 
        y = 0  
        z = 0  
        for neighbor in self.graph[vj]:
            if neighbor in self.boundary_set:
                x += 1
        vj_neighbors_in_community = 0
        for neighbor in self.graph[vj]:
            if neighbor in self.community_set:
                vj_neighbors_in_community += 1
        y = len(self.graph[vj]) - vj_neighbors_in_community
        
        affected_nodes = {vj}
        for neighbor in self.graph[vj]:
            if neighbor in self.community_set:
                affected_nodes.add(neighbor)
        
        new_boundary_nodes = set()
        for node in affected_nodes:
            has_external_neighbor = False
            for nb in self.graph[node]:
                if nb not in self.community_set and nb != vj:
                    has_external_neighbor = True
                    break
            if has_external_neighbor:
                new_boundary_nodes.add(node)
        
        for bd_node in self.boundary_set:
            if bd_node not in affected_nodes:
                new_boundary_nodes.add(bd_node)
        
        edges_to_keep = set()
        
        for edge in self.edges_inT:
            u, v = edge[0], edge[1]
            if u not in affected_nodes and v not in affected_nodes:
                edges_to_keep.add(edge)
        
        for bd_node in new_boundary_nodes:
            if bd_node in affected_nodes: 
                for neighbor in self.graph[bd_node]:
                    u_temp = min(bd_node, neighbor)
                    v_temp = max(bd_node, neighbor)
                    edges_to_keep.add((u_temp, v_temp))
        
        z = len(self.edges_inT - edges_to_keep)
        
        delta_local_modularity = x - self.R * y - z * (1 - self.R)
        denominator = self.T - z + y
        if denominator != 0:
            delta_local_modularity /= denominator
        else:
            delta_local_modularity = 0.0
        
        return delta_local_modularity

    def init(self):
        self.cal_community_neighbour_set()
        self.cal_boundary_set()
        self.cal_local_modularity()

    def update_boundary_set_incremental(self, vj: int):
        has_external_neighbor = False
        for neighbor in self.graph[vj]:
            if neighbor not in self.community_set:
                has_external_neighbor = True
                break
        if has_external_neighbor:
            self.boundary_set.add(vj)
        
        affected_nodes = set()
        for neighbor in self.graph[vj]:
            if neighbor in self.community_set:
                affected_nodes.add(neighbor)
        
        for node in affected_nodes:
            has_external = False
            for nb in self.graph[node]:
                if nb not in self.community_set:
                    has_external = True
                    break
            if has_external:
                self.boundary_set.add(node)
            else:
                self.boundary_set.discard(node)
    
    def update_community_neighbour_set_incremental(self, vj: int):
        self.community_neighbour_set.discard(vj)
        for neighbor in self.graph[vj]:
            if neighbor not in self.community_set:
                self.community_neighbour_set.add(neighbor)
    
    def update_local_modularity_incremental(self, vj: int):
        affected_nodes = {vj}
        for neighbor in self.graph[vj]:
            if neighbor in self.community_set:
                affected_nodes.add(neighbor)
        edges_to_remove_fromT = set()
        edges_to_remove_fromI = set()
        
        for edge in self.edges_inT:
            u, v = edge[0], edge[1]
            if u in affected_nodes or v in affected_nodes:
                edges_to_remove_fromT.add(edge)
                if edge in self.edges_inI:
                    edges_to_remove_fromI.add(edge)
        self.edges_inT -= edges_to_remove_fromT
        self.edges_inI -= edges_to_remove_fromI
        
        new_edges_inI = set()
        for bd_node in self.boundary_set:
            if bd_node in affected_nodes: 
                for neighbor in self.graph[bd_node]:
                    u = min(bd_node, neighbor)
                    v = max(bd_node, neighbor)
                    self.edges_inT.add((u, v))
                    if neighbor in self.community_set:
                        new_edges_inI.add((u, v))
        
        self.edges_inI |= new_edges_inI
        
        self.T = len(self.edges_inT)
        self.I = len(self.edges_inI)
        self.R = self.I / self.T if self.T != 0 else 0.0
    
    def run(self):
        start_time = time.time()
        max_runtime = 1000.0
        
        while len(self.community_set) < self.k:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime:
                return
            
            if not self.community_neighbour_set:
                break
                
            vj = -1
            max_delta_local_modularity = float('-inf')
            for u in self.community_neighbour_set:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime:
                    return
                
                delta_local_modularity = self.cal_delta_local_modularity(u)
                if delta_local_modularity > max_delta_local_modularity:
                    max_delta_local_modularity = delta_local_modularity
                    vj = u
                # print(u, delta_local_modularity)
            if vj == -1:
                break
            self.community_set.add(vj)
            self.update_boundary_set_incremental(vj)
            self.update_community_neighbour_set_incremental(vj)
            self.update_local_modularity_incremental(vj)

    def print(self, node_set: Set[int]):
        for u in node_set:
            print(u, end=" ")
        print()

    def print_edges(self, edge_set: Set[PII]):
        for edge in edge_set:
            print(edge[0], edge[1])
        print()
