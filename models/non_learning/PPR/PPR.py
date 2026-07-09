import numpy as np
import heapq
from collections import deque
import math
import random
import os
import datetime
import json
import sys
import time
CONDUCTANCE_TARGET = 0.3          #  Target conductance for cuts (lower = better quality)
ALPHA_MIN = 0.01                 #  Minimum teleport probability (higher = faster convergence)
MAX_ITERATIONS_APPROX_PR = 100000  #  Max iterations for approximate PageRank
MAX_ITERATIONS_PARTITION = 10     #  Max iterations for finding partitions
RANDOM_SEED = 0                  #  Random seed for reproducibility


class Graph:
    def __init__(self):
        self.neighbors = {}  # adjacency list
        self.degrees = {}    # degree of each vertex
        self.n_vertices = 0  # number of vertices
        self.n_edges = 0     # number of edges

    def read_from_file(self, path):
        edges = []
        max_node = -1
        
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue 
                parts = line.split()
                if len(parts) >= 2:
                    u, v = map(int, parts[:2])
                    if u == v:
                        continue 
                    edges.append((u, v))
                    max_node = max(max_node, u, v)
        
        if max_node >= 0:
            self.n_vertices = max_node + 1
            self.neighbors = {i: [] for i in range(self.n_vertices)}
            self.degrees = {i: 0 for i in range(self.n_vertices)}

        for u, v in edges:
            if v not in self.neighbors[u]: 
                self.neighbors[u].append(v)
                self.neighbors[v].append(u)
                self.degrees[u] += 1
                self.degrees[v] += 1
                self.n_edges += 1

    def add_vertex(self, v):
        if v not in self.neighbors:
            self.neighbors[v] = []
            self.degrees[v] = 0
            self.n_vertices += 1

    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)
        self.neighbors[u].append(v)
        self.neighbors[v].append(u)
        self.degrees[u] += 1
        self.degrees[v] += 1
        self.n_edges += 1

    def volume(self, vertices=None):
        if vertices is None:
            return 2 * self.n_edges
        
        return sum(self.degrees[v] for v in vertices if v in self.degrees)

    def edge_boundary(self, vertices):
        boundary = 0
        vertices_set = set(vertices)
        
        for v in vertices:
            if v not in self.neighbors:
                continue
            for u in self.neighbors[v]:
                if u not in vertices_set:
                    boundary += 1
                    
        return boundary

    def conductance(self, vertices):
        if not vertices:
            return float('inf')
            
        vol_vertices = self.volume(vertices)
        vol_graph = self.volume()
        
        # If the set is the entire graph, conductance is 0
        if vol_vertices == vol_graph:
            return 0
            
        edge_boundary = self.edge_boundary(vertices)
        denominator = min(vol_vertices, vol_graph - vol_vertices)
        
        if denominator == 0:
            return float('inf')
            
        return edge_boundary / denominator


def approximate_pagerank(graph, seed, alpha, epsilon):
    p = {v: 0 for v in graph.neighbors}
    r = seed.copy()
    
    queue = deque([v for v in r if r[v] >= epsilon * graph.degrees.get(v, 0)])
    
    max_iterations = MAX_ITERATIONS_APPROX_PR     
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        u = queue.popleft()
        
        if r[u] < epsilon * graph.degrees.get(u, 0):
            continue
        
        push_value = r[u]
        p[u] = p.get(u, 0) + alpha * push_value
        r[u] = (1 - alpha) * push_value / 2
        
        if graph.degrees[u] == 0:   
            p[u] += r[u]            
            r[u] = 0
            continue
            
        neighbor_push = (1 - alpha) * push_value / (2 * graph.degrees[u])

        for v in graph.neighbors[u]:
            old_r_v = r.get(v, 0)
            r[v] = old_r_v + neighbor_push
            
            if old_r_v < epsilon * graph.degrees[v] <= r[v] and v not in queue:
                queue.append(v)
        
        if r[u] >= epsilon * graph.degrees[u] and u not in queue:
            queue.append(u)
    
    if iterations >= max_iterations:
        print(f"Warning: Maximum iterations ({max_iterations}) reached in approximate_pagerank.")
    
    return p, r


def pagerank_nibble(graph, v, phi, b):
    if graph.n_edges == 0:
        return None, -1

    if v not in graph.neighbors or len(graph.neighbors[v]) == 0:
        return None, -1

    B = math.ceil(math.log2(graph.n_edges)) if graph.n_edges > 0 else 1

    if graph.n_edges < 4:  # Very small graph
        alpha = ALPHA_MIN
    else:
        raw_alpha = (phi**2) / (225 * math.log2(100 * math.sqrt(graph.n_edges)))
        alpha = max(raw_alpha, ALPHA_MIN)

    epsilon = 1 / (2**b * 48 * B)

    epsilon = max(epsilon, 1e-10)

    seed = {u: 1 if u == v else 0 for u in graph.neighbors}

    p, r = approximate_pagerank(graph, seed, alpha, epsilon)

    vertices = [u for u in p if u in graph.degrees and graph.degrees[u] > 0]
    vertices.sort(key=lambda u: p.get(u, 0) / graph.degrees[u], reverse=True)

    if not vertices:
        return None, -1

    target_vol_2b = 2**b
    target_vol_2b_minus_1 = 2**(b-1)
    threshold = 1 / (48 * B)
    total_volume = graph.volume()

    current_list = []
    current_set_set = set()
    current_volume = 0
    current_prob = 0.0
    current_boundary = 0
    p_val_2b = None
    p_val_2b_minus_1 = None

    for u in vertices:
        prev_volume = current_volume
        prev_prob = current_prob
        # update volume and prob
        du = graph.degrees.get(u, 0)
        pu = p.get(u, 0.0)
        current_volume += du
        current_prob += pu

        # compute number of neighbors of u already in current_set (inside neighbors)
        inside_neighbors = 0
        for nbr in graph.neighbors[u]:
            if nbr in current_set_set:
                inside_neighbors += 1

        # update boundary
        current_boundary += du - 2 * inside_neighbors

        # add u
        current_list.append(u)
        current_set_set.add(u)

        if p_val_2b_minus_1 is None and prev_volume < target_vol_2b_minus_1 <= current_volume:
            if du > 0:
                frac = (target_vol_2b_minus_1 - prev_volume) / du
                p_val_2b_minus_1 = prev_prob + frac * pu
            else:
                p_val_2b_minus_1 = current_prob

        if p_val_2b is None and prev_volume < target_vol_2b <= current_volume:
            if du > 0:
                frac = (target_vol_2b - prev_volume) / du
                p_val_2b = prev_prob + frac * pu
            else:
                p_val_2b = current_prob

        if p_val_2b_minus_1 is None and current_volume == target_vol_2b_minus_1:
            p_val_2b_minus_1 = current_prob
        if p_val_2b is None and current_volume == target_vol_2b:
            p_val_2b = current_prob

        # ---- compute incremental conductance for the current set ----
        if current_volume > 0 and current_volume < total_volume:
            conductance = current_boundary / min(current_volume, total_volume - current_volume)
        else:
            conductance = float('inf')

        # ---- evaluate conditions----
        prob_condition_satisfied = False
        if (p_val_2b is not None) and (p_val_2b_minus_1 is not None):
            prob_condition_satisfied = (p_val_2b - p_val_2b_minus_1) > threshold

        # Condition 1: volume strict bounds
        if current_volume > target_vol_2b_minus_1 and current_volume < (2/3) * total_volume:
            # Condition 2: conductance
            if conductance < phi:
                # Condition 3: probability condition must be known and satisfied
                if prob_condition_satisfied:
                    return current_list, conductance

    return None, -1

def find_community_for_vertex(graph, query_vertex, phi=None):
    if phi is None:
        phi = CONDUCTANCE_TARGET  
        
    if query_vertex not in graph.neighbors:
        return None, {"error": f"Vertex {query_vertex} not found in graph"}
    
    B = math.ceil(math.log2(graph.n_edges)) if graph.n_edges > 0 else 1
    
    for b in range(1, B + 1):
        community, conductance = pagerank_nibble(graph, query_vertex, phi, b)

        if community is not None and conductance != -1 and query_vertex in community:
            
            def analyze_community(vertices):
                internal_edges = 0
                external_edges = 0
                vertices_set = set(vertices)
                
                for v in vertices:
                    for u in graph.neighbors[v]:
                        if u in vertices_set:
                            internal_edges += 1
                        else:
                            external_edges += 1
                
                internal_edges //= 2 
                return internal_edges, external_edges
            
            internal, external = analyze_community(community)
            
            info = {
                "scale": b,
                "conductance": conductance,
                "size": len(community),
                "volume": graph.volume(community),
                "internal_edges": internal,
                "external_edges": external,
                "density": internal / (len(community) * (len(community) - 1) / 2) if len(community) > 1 else 0
            }
            
            print(f"Found community for vertex {query_vertex}:")
            print(f"  Scale parameter b: {b}")
            print(f"  Vertices: {sorted(community)}")
            print(f"  Size: {info['size']}")
            print(f"  Conductance: {info['conductance']:.4f}")
            print(f"  Density: {info['density']:.4f}")
            print(f"  Volume: {info['volume']}")
            
            return sorted(community), info
    
    print(f"No suitable community found for vertex {query_vertex}")
    return None, {"error": "No community found"}


def read_query_from_file(path: str) -> int:
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  
            parts = line.split()
            if len(parts) >= 1:
                return int(parts[0])  
    
    raise ValueError("No valid query node found in the file")


def write_results_to_file(graph, query_node, community, info, output_path='./output/pagerank.txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f'Result for PageRank community query with node {query_node}:\n')
        
        if community:
            f.write(f"# Community nodes: {sorted(community)}\n")
            f.write(f"# Community size: {len(community)}\n")
            f.write(f"# Conductance: {info['conductance']:.6f}\n")
            f.write(f"# Density: {info['density']:.4f}\n")
            f.write(f"# Internal edges: {info['internal_edges']}\n")
            f.write(f"# External edges: {info['external_edges']}\n")
            f.write(f"# Volume: {info['volume']}\n")
            
            community_list = sorted(list(community))
            for i in range(len(community_list)):
                for j in range(i + 1, len(community_list)):
                    u, v = community_list[i], community_list[j]
                    if v in graph.neighbors[u]:
                        f.write(f'{u} {v}\n')
        else:
            f.write("No community found containing the query node.\n")


def main():
    
    if len(sys.argv) != 3:
        print("Usage: python pr.py <graph_path> <query_path>")
        print("  graph_path: Path to the graph file (edge list format)")
        print("  query_path: Path to the query file (single node)")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    query_path = sys.argv[2]
    
    print("=== PageRank-Nibble Algorithm Parameters ===")
    print(f"  Target Conductance: {CONDUCTANCE_TARGET}")
    print(f"  Minimum Alpha: {ALPHA_MIN}")
    print(f"  Max Iterations (Approx PR): {MAX_ITERATIONS_APPROX_PR}")
    print(f"  Max Iterations (Partition): {MAX_ITERATIONS_PARTITION}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print("===========================================")
    print()
    
    random.seed(RANDOM_SEED)
    
    graph = Graph()
    graph.read_from_file(graph_path)
    
    try:
        query_node = read_query_from_file(query_path)
        print(f"Query node: {query_node}")
        print(f"Graph info: {graph.n_vertices} vertices, {graph.n_edges} edges")
    except Exception as e:
        print(f"Error reading query file: {e}")
        sys.exit(1)
    
    community, info = find_community_for_vertex(graph, query_node, CONDUCTANCE_TARGET)
    
    output_path = './output/pagerank.txt'
    write_results_to_file(graph, query_node, community, info, output_path)
    
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
