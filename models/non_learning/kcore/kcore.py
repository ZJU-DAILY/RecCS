from collections import deque
import sys

class Graph:
    def __init__(self, path=None):
        if path is not None:
            # Init
            self.adj = {}
            self.degrees = {}
            self.degrees_set = {}
            self.ordered_nodes = []
            self.degree_max = 0
            self.m = 0
            self.n = 0
            self.read_file(path)
            self.cores = {}
            self.cores_set = {}
            self.core_max = 0

    # read graph from file and compute basic infos
    def read_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue  # skip the empty line
                parts = line.strip().split()
                if len(parts) == 2:
                    # the input file save as src , dst
                    u, v = map(int, parts)
                    # read from file 
                    if u == v:
                        continue

                    if u not in self.adj:
                        self.adj[u] = set()
                    if v not in self.adj:
                        self.adj[v] = set()
                    self.adj[u].add(v)
                    self.adj[v].add(u)
                    self.m+=1
                    
        self.n = len(self.adj)
        # Update degrees
        for u in self.adj:
            self.degrees[u] = len(self.adj[u])
            self.degree_max = max(self.degree_max,self.degrees[u])
        
        # compute the degree-nodes dicts
        for node, degree in self.degrees.items():
            if degree not in self.degrees_set:
                self.degrees_set[degree] = set()
            self.degrees_set[degree].add(node)

    # prtint the basic graph infos
    def print_graph_info(self):
        # print the size of graph
        print(f"the node size is : {self.n}")
        print(f"the edge size is : {self.m}")

        print(f"the max degree is : {self.degree_max}")
        print((f"the degrees of the graph is : {len(self.degrees_set)}"))

        print(f"the max core is : {self.core_max}")
        print((f"the cores of the graph is : {len(self.cores_set)}"))

    # compute the k-core in graph
    def compute_core_groups(self):
        current_node = 0
        lowest_degree = 0
        neighbor_degree = 0

        while lowest_degree < self.degree_max:
            if lowest_degree not in self.degrees_set or not self.degrees_set[lowest_degree]:
                lowest_degree+=1

            else:

                current_node = self.degrees_set[lowest_degree].pop()
                self.cores[current_node] = lowest_degree

                for neighbor in self.adj[current_node]:
                    neighbor_degree = self.degrees[neighbor]
                    if neighbor_degree > lowest_degree:
                        self.degrees_set[neighbor_degree].remove(neighbor)
                        if neighbor_degree - 1 not in self.degrees_set:
                            self.degrees_set[neighbor_degree-1] = set()
                        self.degrees_set[neighbor_degree-1].add(neighbor)
                        self.degrees[neighbor] = neighbor_degree - 1
        
        # compute the degree-nodes dicts
        for node, core in self.cores.items():
            self.core_max = max(self.core_max,core)
            if core not in self.cores_set:
                self.cores_set[core] = set()
            self.cores_set[core].add(node)

class TreeIndex:
    def __init__(self, path=None):
        self.connected_component_nodes = {}
        self.graph = Graph(path)
        
        self.graph.compute_core_groups()

        self.graph.print_graph_info()

        ############################

        # dict : node_id - core_index  
        self.core_index = self.graph.cores.copy()
        
        # dict : core_index - core_number  
        self.core_minimum_degree = {}
        
        self.compute_core_index()
        
        ###########################
        
        # dict : shell_layer_index - component_id - node_ids
        self.layer_component_to_nodes = {}

        # dict : component_id - node_ids
        self.component_to_nodes = {}

        # dict : node_id - component_id
        self.node_to_component_id = {}

        self.identify_and_store_components()

        #############################

        # dict : k_shell_index - com_id - parent_com_id
        self.connected_component_parents = {}
        # dict : k_shell_index - com_id - child_com_id
        self.connected_component_children = {}
        # dict : com_id - parent_com_id
        self.component_parents_id = {}
        # dict : com_id - child_com_id
        self.component_children_id = {}

        self.build_parent_children_relationships()

        # optional ：rebuild the index from graph as tree 
        for com_id in self.component_parents_id:
            min_com_id = min(self.component_parents_id[com_id])
            self.component_parents_id[com_id].clear()
            self.component_parents_id[com_id].add(min_com_id)

        ############################
    
    # after this process,the vaules of core_index is the index instead of core number
    # and the core_minimum_degree stores the index-core_number
    def compute_core_index(self):
        core_number = set()
        # calculate the set of core
        for node,core in self.core_index.items():
            core_number.add(core)

        index = 0
        shell_count = 0

        # restore the index into core_index
        while core_number:
            self.core_minimum_degree [index] = core_number.pop()
            if self.core_minimum_degree[index] > shell_count :
                shell_count = self.core_minimum_degree[index]
            
            for node,core in self.core_index.items():
                if core == self.core_minimum_degree[index]:
                    self.core_index[node] = index
            index+=1

    # calculate the components in k-shell
    def identify_and_store_components(self):
        nodes_in_shell = {}
        next_component_id = 0

        # compute the shell
        for node,index in self.core_index.items():
            if index not in nodes_in_shell:
                nodes_in_shell[index] = set()
            nodes_in_shell[index].add(node)

        # calculate the components in every shell
        for index,nodes in nodes_in_shell.items():
            
            #BFS
            visited = set()
            for node in nodes :
                if node not in visited :
                    
                    # BFS Init
                    component = set()
                    queue = deque()
                    queue.append(node)
                    visited.add(node)

                    while queue:
                        current_node = queue.popleft()
                        component.add(current_node)

                        for neighbor in self.graph.adj[current_node]:
                            if neighbor in nodes and neighbor not in visited : 
                                queue.append(neighbor)
                                visited.add(neighbor)

                    next_component_id+=1
                    component_id = next_component_id
                    
                    if index not in self.layer_component_to_nodes:
                        self.layer_component_to_nodes [index] = {}
                    if component_id not in self.layer_component_to_nodes [index]:
                        self.layer_component_to_nodes [index][component_id] = set()
                    self.layer_component_to_nodes [index][component_id] = component
                    
                    self.component_to_nodes[component_id] = component
                    
                    for node_id in component:
                        self.node_to_component_id[node_id] = component_id
        
        #check 
        # print(len(self.layer_component_to_nodes))
        # sum=0
        # for index in self.layer_component_to_nodes:
        #     for com_id in self.layer_component_to_nodes[index]:
        #         sum+=len(self.layer_component_to_nodes [index][com_id])
        # print(sum)
        # print(len(self.component_to_nodes))
        # print(self.node_to_component_id)

    # find the relationships between components in different shell
    def build_parent_children_relationships(self):

        # set the parent component id
        for current_shell in self.layer_component_to_nodes:
            if current_shell not in self.connected_component_parents:
                self.connected_component_parents [current_shell] = {}

            for current_com_id  in self.layer_component_to_nodes[current_shell]:
                
                if current_com_id not in self.component_parents_id:
                    self.component_parents_id[current_com_id] = set()

                for node_id in self.layer_component_to_nodes[current_shell][current_com_id]:
                    for neighbor in self.graph.adj[node_id]:
                        if self.core_minimum_degree[self.core_index[node_id]] >= self.core_minimum_degree[self.core_index[neighbor]]:
                            continue;

                        if neighbor in self.node_to_component_id and self.node_to_component_id[neighbor]!=current_com_id:
                            
                            neighbor_com_id = self.node_to_component_id[neighbor]
                            
                            self.connected_component_parents [current_shell][current_com_id] = neighbor_com_id
                            
                            self.component_parents_id[current_com_id].add(neighbor_com_id)

                # set the top parent component id as -1
                if current_com_id not in self.connected_component_parents[current_shell]:
                                        
                    self.connected_component_parents[current_shell][current_com_id] = -1

                    self.component_parents_id[current_com_id].add(-1)
        
        # set the child component id
        for current_shell in sorted(self.layer_component_to_nodes.keys(), reverse=True):
            if current_shell not in self.connected_component_children:
                self.connected_component_children [current_shell] = {}

            for current_com_id  in self.layer_component_to_nodes[current_shell]:
                
                if current_com_id not in self.component_children_id:
                    self.component_children_id[current_com_id] = set()

                for node_id in self.layer_component_to_nodes[current_shell][current_com_id]:
                    for neighbor in self.graph.adj[node_id]:
                        if self.core_minimum_degree[self.core_index[node_id]] <= self.core_minimum_degree[self.core_index[neighbor]]:
                            continue;

                        if neighbor in self.node_to_component_id and self.node_to_component_id[neighbor]!=current_com_id:
                            
                            neighbor_com_id = self.node_to_component_id[neighbor]
                            
                            self.connected_component_children [current_shell][current_com_id] = neighbor_com_id
                            
                            self.component_children_id[current_com_id].add(neighbor_com_id)

                if current_com_id not in self.connected_component_children[current_shell]:
                                        
                    self.connected_component_children[current_shell][current_com_id] = -1

                    self.component_children_id[current_com_id].add(-1)

    # get the top parents as search tree roots
    def find_top_components(self,query,k):
        top_coms = set()
        query_coms = set()
        queue = deque()
        visited = set()

        #Init,deal with query
        for query_node in query:
            com_id = self.node_to_component_id[query_node]
            if com_id not in visited :
                query_coms.add(com_id)
                queue.append(com_id)
                visited.add(com_id)
        
        # BFS the parents&children' component network
        while queue:
            current_com_id = queue.popleft()
            # search all the parents' component
            for parent_id in self.component_parents_id[current_com_id]:
                if parent_id == -1:
                    top_coms.add(current_com_id)
                else:
                    if parent_id not in visited :
                        visited.add(parent_id)
                        queue.append(parent_id)
            
            # search children's
            for child_id in self.component_children_id[current_com_id]:
                if child_id not in visited and child_id != -1:
                    is_archieve_k = True
                    for child_node in self.component_to_nodes[child_id]:
                        if self.core_minimum_degree[self.core_index[child_node]]<k:
                            is_archieve_k = False
                            break
                    if is_archieve_k:
                        queue.append(child_id)
                        visited.add(child_id)

        return top_coms

    # Community Search with k_shell , but the result might be not connected 
    def search_with_shell(self,query):
        
        query_k = float('inf')
        com_result = set()
        com_queue = deque()
        node_result = set()

        for node in query:
            query_k = min(query_k,self.core_minimum_degree[self.core_index[node]])
        
        # print (f'the query k value is {query_k}')

        # get the top parents as search tree roots
        top_component_ids = self.find_top_components(query,query_k)

        for com_id in top_component_ids:
            com_queue.append(com_id)
        
        # BFS
        while com_queue:
            current_com_id = com_queue.popleft()
            if current_com_id in com_result:
                continue
            com_result.add(current_com_id)

            for child_com_id in self.component_children_id[current_com_id]:
                if child_com_id == -1:
                    continue
                child_node = 0
                for node_id in self.component_to_nodes[child_com_id]:
                    child_node = node_id
                    break
                if self.core_minimum_degree[self.core_index[child_node]]>=query_k:
                    com_queue.append(child_com_id)
        
        for com_id in com_result:
            for node_id in self.component_to_nodes[com_id]:
                node_result.add(node_id)

        print(f'the answer community size is : {len(node_result)}')
        edge_result = set()
        for u in node_result:
            if u in self.graph.adj:
                for v in self.graph.adj[u]:
                    if v in node_result and u < v:
                        edge_result.add((u, v))
        return node_result,edge_result

def main():
    if len(sys.argv) != 3:
        print("Usage: python TreeIndex.py <graph_path> <query_path>")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    
    treeindex = TreeIndex(graph_path)

    query_path = sys.argv[2]
    query_set = set()
    
    with open(query_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue  # skip the empty line
            parts = line.strip().split()
            if len(parts) == 1:
                # the input file save as src , dst
                q = int(parts[0])
                query_set.add(q)
    
    for query_node in query_set:
        nodes,edges = treeindex.search_with_shell([query_node])
        with open('./output/treeindex.txt', 'a') as f:
            f.write(f'Result for query node: {query_node}\n')
            for edge in edges:
                f.write(f'{edge[0]} {edge[1]}\n')
            f.write(f'\n')

if __name__ == "__main__":

    main()