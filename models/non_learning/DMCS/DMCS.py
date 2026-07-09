import sys
import time
import heapq
import copy
from collections import defaultdict, deque

DEBUG = 0

class Edge:
    def __init__(self, u, v):
        if u > v:
            u, v = v, u
        self.u = u
        self.v = v

    def __lt__(self, other):
        if isinstance(other, Edge):
            return (self.u, self.v) < (other.u, other.v)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.u == other.u and self.v == other.v
        return NotImplemented

    def __hash__(self):
        return hash(self.u) ^ (hash(self.v) << 1)

def make_edge(u, v):
    return Edge(u, v)

class MyGraph:
    def __init__(self):
        self.umap_graph = defaultdict(dict)
        self.umap_degree = defaultdict(int)
        self.i_num_nodes = 0
        self.i_num_edges = 0
        
        self.i_num_nodes_subgraph = 0
        self.i_num_edges_subgraph = 0
        self.i_sum_degree_subgraph = 0
        self.umap_degree_subgraph = defaultdict(int)
        self.shortest_distance = dict()
        self.parent = dict()

    def get_connected_components(self, vi_query_nodes):
        umap_visited = defaultdict(bool)
        queue = deque()
        
        for node in vi_query_nodes:
            if not umap_visited[node]:
                queue.append(node)
                umap_visited[node] = True
                
        while queue:
            node = queue.popleft()
            for neighbor in self.umap_graph[node]:
                if not umap_visited[neighbor]:
                    umap_visited[neighbor] = True
                    queue.append(neighbor)
        
        for node in range(1, self.i_num_nodes + 1):
            if not umap_visited.get(node, False):
                self.remove_node(node)

    def dijkstra(self, i_start_node):
        self.shortest_distance.clear()
        self.parent.clear()
        self.parent[i_start_node] = i_start_node
        self.shortest_distance[i_start_node] = 0
        
        pq = []
        heapq.heappush(pq, (0, i_start_node))
        visited = set()
        
        while pq:
            dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            
            visited.add(node)
            
            for neighbor, weight in self.umap_graph[node].items():
                new_dist = dist + weight
                if (neighbor not in self.shortest_distance or 
                    new_dist < self.shortest_distance[neighbor]):
                    self.shortest_distance[neighbor] = new_dist
                    self.parent[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

    def dijkstra_multi_source(self, vi_query_nodes):
        self.shortest_distance.clear()
        self.parent.clear()
        pq = []
        
        for node in vi_query_nodes:
            self.parent[node] = node
            self.shortest_distance[node] = 0
            heapq.heappush(pq, (0, node))
            
        visited = set()
        
        while pq:
            dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            
            visited.add(node)
            
            for neighbor, weight in self.umap_graph[node].items():
                new_dist = dist + weight
                if (neighbor not in self.shortest_distance or 
                    new_dist < self.shortest_distance[neighbor]):
                    self.shortest_distance[neighbor] = new_dist
                    self.parent[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

    def get_shortest_path(self, i_start_node, i_end_node):
        path = []
        current = i_end_node
        
        while current != i_start_node:
            path.append(current)
            current = self.parent[current]
            if current is None:
                return []
        
        path.append(i_start_node)
        path.reverse()
        return path

    def calculate_num_nodesof_subgraph(self):
        # sum the nodes in the subgraph
        self.i_num_nodes_subgraph = 0
        for node in self.umap_graph:
            if len(self.umap_graph[node]) > 0:
                self.i_num_nodes_subgraph += 1
        return self.i_num_nodes_subgraph

    def calculate_num_edgesof_subgraph(self):
        edges = 0
        for node in self.umap_graph:
            edges += len(self.umap_graph[node])
        self.i_num_edges_subgraph = edges // 2
        return self.i_num_edges_subgraph

    def calculate_sum_of_degreesof_subgraph(self):
        total = 0
        for node in self.umap_degree_subgraph:
            total += self.umap_degree_subgraph[node]
        self.i_sum_degree_subgraph = total
        return self.i_sum_degree_subgraph

    def calculate_density_modularity(self):
        if self.i_num_nodes_subgraph == 0 or self.i_num_edges_subgraph == 0:
            return 0.0
        
        t1 = 1.0 / (2 * self.i_num_nodes_subgraph)
        t2 = 2 * self.i_num_edges_subgraph
        t3 = (self.i_sum_degree_subgraph**2) / (2 * self.i_num_edges)
        return t1 * (t2 - t3)

    def calculate_density_ratio(self, node):
        return self.umap_degree.get(node, 0) / self.umap_degree_subgraph.get(node, 0)

    def get_dis_group(self):
        dis_group = defaultdict(list)
        for node, dist in self.shortest_distance.items():
            dis_group[dist].append(node)
        return dis_group

    def get_neighbors(self, node):
        return list(self.umap_graph[node].keys())

    def print_graph(self):
        print(f"Number of subgraph nodes: {self.i_num_nodes_subgraph}")
        print(f"Number of subgraph edges: {self.i_num_edges_subgraph}")
        for node in self.umap_graph:
            print(f"{node}: {', '.join(map(str, self.umap_graph[node].keys()))}")

    def print_distance(self):
        print("shortest distance:")
        for node, dist in self.shortest_distance.items():
            print(f"{node}: {dist}")

    def print_dm_parameters(self):
        print(f"Number of nodes in subgraph: {self.i_num_nodes_subgraph}")
        print(f"Number of edges in subgraph: {self.i_num_edges_subgraph}")
        print(f"Sum of degrees in subgraph: {self.i_sum_degree_subgraph}")
        print(f"Density modularity: {self.calculate_density_modularity()}")

    def reset_dm_parameters(self):
        self.calculate_num_edgesof_subgraph()
        self.calculate_num_nodesof_subgraph()
        self.calculate_sum_of_degreesof_subgraph()

    def reduce_dm_parameters(self, i_num_nodes_subgraph_reduce, i_num_edges_subgraph_reduce):
        self.i_num_nodes_subgraph -= i_num_nodes_subgraph_reduce
        self.i_num_edges_subgraph -= i_num_edges_subgraph_reduce
        self.i_sum_degree_subgraph -= 2 * i_num_edges_subgraph_reduce

    def remove_node(self, i_node):
        if i_node not in self.umap_graph:
            return
        
        self.shortest_distance.pop(i_node, None)
        self.parent.pop(i_node, None)
        
        for neighbor in list(self.umap_graph[i_node].keys()):
            del self.umap_graph[neighbor][i_node]
            self.i_num_edges_subgraph -= 1
            self.umap_degree_subgraph[neighbor] -= 1
        
        self.i_sum_degree_subgraph -= self.umap_degree[i_node]
        self.umap_degree_subgraph.pop(i_node, None)
        self.i_num_nodes_subgraph -= 1
        # delete the node from the graph
        del self.umap_graph[i_node]
        self.umap_degree.pop(i_node, None)

    def read_graph_from_file(self, s_file_name):
        in_file = open(s_file_name, 'r')
        if not in_file:
            print("File does not exist.")
            return
        
        i_from, i_to = 0, 0
        while True:
            line = in_file.readline()
            if not line:
                break
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            i_from, i_to = parts[0], parts[1]
            if i_from == i_to:
                continue
            
            if i_to not in self.umap_graph[i_from]:
                self.umap_graph[i_from][i_to] = 1
                self.umap_degree[i_from] += 1
                self.i_num_edges += 1
            if i_from not in self.umap_graph[i_to]:
                self.umap_graph[i_to][i_from] = 1
                self.umap_degree[i_to] += 1
                self.i_num_edges += 1
            
            if i_from > self.i_num_nodes:
                self.i_num_nodes = i_from
            if i_to > self.i_num_nodes:
                self.i_num_nodes = i_to
        self.i_num_edges //= 2
        self.i_num_edges_subgraph = self.i_num_edges
        self.i_num_nodes_subgraph = self.i_num_nodes
        for node in self.umap_degree:
            self.umap_degree_subgraph[node] = self.umap_degree[node]
            self.i_sum_degree_subgraph += self.umap_degree[node]
        
        in_file.close()
    
    def get_graph(self):
        return dict(self.umap_graph)  

    def get_num_edges_of_whole_graph(self):
        return self.i_num_edges

class DMCS:
    def __init__(self, my_graph):
        self.my_graph = my_graph
        self.returned_community = {}
        self.start_time = None
        self.max_runtime = 1000.0

    def merge_query_nodes_into_a_whole_node(self, vi_query_nodes):
        if not vi_query_nodes:
            return
    
        start_node = vi_query_nodes[0]
        self.my_graph.dijkstra(start_node) 
        whole_nodes = self.my_graph.get_shortest_path(start_node, start_node)
        
        # Collect shortest paths between all query nodes
        for i in range(1, len(vi_query_nodes)):
            start = vi_query_nodes[0]
            end = vi_query_nodes[i]
            path = self.my_graph.get_shortest_path(start, end)
            if path:
                whole_nodes.extend(path)
        
        # Deduplicate and sort nodes
        whole_nodes = sorted(list(set(whole_nodes)))
        
        self.my_graph.dijkstra_multi_source(whole_nodes)

    def get_a_community_by_fpa(self, vi_query_nodes):
        self.start_time = time.time()
        
        # Step 1: Get connected components of query nodes
        self.my_graph.get_connected_components(vi_query_nodes)
        
        if time.time() - self.start_time > self.max_runtime:
            return
        
        if DEBUG:
            print(f"Connected components")
            self.my_graph.print_graph()
        
        # Step 2: Merge query nodes
        self.merge_query_nodes_into_a_whole_node(vi_query_nodes)
        
        if time.time() - self.start_time > self.max_runtime:
            return
        
        # Debug print
        if DEBUG:
            self.my_graph.print_distance()
        
        # Step 3: Prune based on layers
        self.prune_based_on_layer()
        
        if time.time() - self.start_time > self.max_runtime:
            return
        
        # Step 4: Remove nodes one by one
        self.remove_node_one_by_one()
        

    def prune_based_on_layer(self):
        dis_group = self.my_graph.get_dis_group()
        max_dm_layer = -1
        max_dm_value = self.my_graph.calculate_density_modularity()
        removed_edges = set()
        
        # Determine maximum layer
        max_layer = max(dis_group.keys()) if dis_group else 0
        
        # Process layers from top to bottom
        current_layer = max_layer
        while current_layer > 0:
            if self.start_time and time.time() - self.start_time > self.max_runtime:
                return
            
            layer_nodes = dis_group.get(current_layer, [])
            layer_edges = set()
            
            # Collect all edges in current layer
            for node in layer_nodes:
                if self.start_time and time.time() - self.start_time > self.max_runtime:
                    return
                
                neighbors = self.my_graph.get_neighbors(node)
                for neighbor in neighbors:
                    if (node, neighbor) not in removed_edges and (neighbor, node) not in removed_edges:
                        # order edges by node id to avoid duplicates
                        if node < neighbor:
                            layer_edges.add((node, neighbor))
                        else:
                            layer_edges.add((neighbor, node))
            
            # Update parameters and check modularity
            self.my_graph.reduce_dm_parameters(len(layer_nodes), len(layer_edges))
            current_dm = self.my_graph.calculate_density_modularity()
            
            if current_dm > max_dm_value:
                max_dm_value = current_dm
                max_dm_layer = current_layer
            
            removed_edges.update(layer_edges)
            current_layer -= 1
      
        # Reset parameters and remove nodes if needed
        self.my_graph.reset_dm_parameters()
        
        if max_dm_layer != -1:
            for layer in dis_group:
                if layer >= max_dm_layer:
                    for node in dis_group[layer]:
                        self.my_graph.remove_node(node)
        
        # Debug output
        if DEBUG:
            print(f"max_dm_layer: {max_dm_layer}")
            print(f"max_dm_value: {max_dm_value}")
            self.my_graph.print_graph()
            self.my_graph.print_dm_parameters()

    def remove_node_one_by_one(self):
        max_dm = self.my_graph.calculate_density_modularity()
        self.returned_community = copy.deepcopy(self.my_graph.get_graph())

        dis_group = self.my_graph.get_dis_group()
        max_layer = max(dis_group.keys()) if dis_group else 0

        while max_layer > 0:
            if self.start_time and time.time() - self.start_time > self.max_runtime:
                return
            
            layer_nodes = dis_group.get(max_layer, [])
            candidates = set(layer_nodes)
            if not candidates:
                max_layer -= 1
                continue

            current_ratio = {}
            heap = []
            for node in candidates:
                if self.start_time and time.time() - self.start_time > self.max_runtime:
                    return
                
                r = self.my_graph.calculate_density_ratio(node)
                current_ratio[node] = r
                heapq.heappush(heap, (-r, node))

            while candidates:
                if self.start_time and time.time() - self.start_time > self.max_runtime:
                    return
                
                while heap:
                    neg_r, node = heapq.heappop(heap)
                    r = -neg_r
                    if node in candidates and abs(current_ratio.get(node, -1) - r) < 1e-12:
                        best_node = node
                        break
                else:
                    break

                self.my_graph.remove_node(best_node)
                candidates.remove(best_node)

                for nei in self.my_graph.get_neighbors(best_node):
                    if nei in candidates:
                        new_r = self.my_graph.calculate_density_ratio(nei)
                        current_ratio[nei] = new_r
                        heapq.heappush(heap, (-new_r, nei))

                current_dm = self.my_graph.calculate_density_modularity()
                if current_dm > max_dm:
                    max_dm = current_dm
                    self.returned_community = self.my_graph.get_graph().copy()

            max_layer -= 1

    def print_returned_community(self, s_output_file_path):
        try:
            with open(s_output_file_path, 'w') as f:
                for node in self.returned_community:
                    for neighbor in self.returned_community[node]:
                        if node < neighbor:
                            f.write(f"{node} {neighbor}\n")
        except IOError as e:
            print(f"Error: Cannot open file {s_output_file_path} - {e}")

    def print_graph(self, graph):
        print(f"Num nodes: {len(graph)}")
        print(f"Num edges: {sum(len(neighbors) for neighbors in graph.values())}")
        for node in graph:
            print(f"Neighbors of {node}: {', '.join(map(str, graph[node]))}")
    
    def print_dis_group(self, dis_group):
        for layer in sorted(dis_group.keys()):
            print(f"Layer {layer}: {', '.join(map(str, dis_group[layer]))}")

# Utility functions for reading query nodes and main execution
def read_query_nodes(file_path: str) -> list:
    """Read query nodes from file"""
    try:
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: cannot open file {file_path}")
        return []

def main():
    """Main function that can be called directly or from command line"""
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} graph_file_path query_file_path output_file_path")
        return 1
    
    s_graph_path = sys.argv[1]
    s_query_path = sys.argv[2]
    s_output_path = sys.argv[3]

    # Create graph and read from file
    my_graph = MyGraph()
    my_graph.read_graph_from_file(s_graph_path)
 
    # Read query nodes
    query_nodes = read_query_nodes(s_query_path)
    if not query_nodes:
        return 1

    # Run DMCS algorithm
    dmcs = DMCS(my_graph)
    dmcs.get_a_community_by_fpa(query_nodes, s_output_path)
    dmcs.print_returned_community(s_output_path)

if __name__ == "__main__":
    main()