import sys
import heapq
from collections import defaultdict, deque

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
        # self.i_num_nodes_subgraph = len(self.umap_graph)
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
        self.i_num_edges -= len(self.umap_graph[i_node])

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