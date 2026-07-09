from __future__ import annotations
from typing import List, Set, Tuple, Dict, Optional
import sys
from collections import defaultdict
import gc
import heapq


class Graph:
    def __init__(self, path=None):
        if path is not None:
            # Initialize
            self.adj = []  # Adjacency list
            self.n = 0     # Number of nodes
            self.m = 0     # Number of edges
            self.degree_max = 0  # Maximum degree
            self.read_file(path)
    
    def read_file(self, path):
        """Read a graph from a file."""
        edges = []
        max_node = -1
        
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue  # skip empty lines
                parts = line.strip().split()
                if len(parts) == 2:
                    u, v = map(int, parts)
                    if u == v:
                        continue  # skip self-loops
                    edges.append((u, v))
                    max_node = max(max_node, u, v)
                    self.m += 1
        
        # Create the adjacency list
        self.n = max_node + 1
        self.adj = [set() for _ in range(self.n)]
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        
        # Update degrees
        for u in range(self.n):
            degree = len(self.adj[u])
            self.degree_max = max(self.degree_max, degree)
    
    def print_graph_info(self):
        print(f"The node size is: {self.n}")
        print(f"The edge size is: {self.m // 2}") 
        print(f"The max degree is: {self.degree_max}")


class KCliqueIndex:
    def __init__(self, path=None):
        self.graph = Graph(path)
        if path is not None:
            self.graph.print_graph_info()
        self.all_cliques = self.find_maximal_cliques()
        print(f"The number of maximal cliques is: {len(self.all_cliques)}")
        self.clique_id_map = {idx: clique for idx, clique in enumerate(self.all_cliques)}
        self.node_to_clique_ids = self._build_node_to_clique_mapping()
        self.dcpc_index = None
        self._get_or_build_dcpc_index()
        
    def _get_or_build_dcpc_index(self):
        """Get cached DCPC index or build it if not exists"""
        if self.dcpc_index is None:
            self.dcpc_index = self.build_dcpc_index()
        return self.dcpc_index

    def degeneracy_ordering(self) -> Tuple[List[int], int]:
        """Return a degeneracy ordering and the degeneracy *d* (maximum core number)."""
        adj = self.graph.adj
        n = len(adj)
        degree = [len(nbs) for nbs in adj]
        if n == 0:
            return [], 0
    
        max_deg = max(degree)
        buckets: List[Set[int]] = [set() for _ in range(max_deg + 1)]
        for v, deg in enumerate(degree):
            buckets[deg].add(v)
    
        order: List[int] = []
        position = [-1] * n
        d = 0
        for _ in range(n):
            i = 0
            while i <= max_deg and not buckets[i]:
                i += 1
            if i > max_deg:
                break
            v = buckets[i].pop()
            order.append(v)
            position[v] = len(order) - 1
            d = max(d, i)
            for u in adj[v]:
                if position[u] == -1:
                    du = degree[u]
                    if du == 0:
                        continue
                    buckets[du].remove(u)
                    degree[u] -= 1
                    buckets[du - 1].add(u)
        return order, d
    
    def _bk_pivot(self, P: Set[int], R: Set[int], X: Set[int], cliques: List[Set[int]]) -> None:
        adj = self.graph.adj
        if not P and not X:
            cliques.append(R.copy())
            return
        if P or X:
            u = max(P.union(X), key=lambda v: len(P & adj[v]))
        else:
            u = None
        for v in list(P - adj[u] if u is not None else P):
            self._bk_pivot(P & adj[v], R | {v}, X & adj[v], cliques)
            P.remove(v)
            X.add(v)
    
    def find_maximal_cliques(self) -> List[Set[int]]:
        adj = self.graph.adj
        order, _ = self.degeneracy_ordering()
        pos = {v: i for i, v in enumerate(order)}
        cliques: List[Set[int]] = []
        for v in order:
            P = {u for u in adj[v] if pos[u] > pos[v]}
            X = {u for u in adj[v] if pos[u] < pos[v]}
            self._bk_pivot(P, {v}, X, cliques)
        return cliques
    
    def _build_node_to_clique_mapping(self) -> Dict[int, List[int]]:
        node_to_clique_ids = defaultdict(list)
        for clique_id, clique in enumerate(self.all_cliques):
            for node in clique:
                node_to_clique_ids[node].append(clique_id)
        return node_to_clique_ids
    
    def build_clique_adjacency_graph(self) -> List[Tuple[int, int, int]]:
        adjacency_counts = defaultdict(int)

        for node in self.node_to_clique_ids:
            clique_ids = self.node_to_clique_ids[node]
            for i in range(len(clique_ids)):
                for j in range(i+1, len(clique_ids)):
                    c1, c2 = clique_ids[i], clique_ids[j]
                    adjacency_counts[(min(c1, c2), max(c1, c2))] += 1
        
        edges = []
        for (c1, c2), count in adjacency_counts.items():
            if count > 0:
                edges.append((c1, c2, count))
        return edges
    
    def build_clique_adjacency_tree(self, edges: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        # Sort edges in descending order of weight
        edges_sorted = sorted(edges, key=lambda x: -x[2])

        parent = list(range(len(self.all_cliques)))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        mst_edges = []
        for c1, c2, weight in edges_sorted:
            root1 = find(c1)
            root2 = find(c2)
            if root1 != root2:
                mst_edges.append((c1, c2, weight))
                parent[root2] = root1 
        return mst_edges
    
    def build_ordered_adjacency_tree(self, mst_edges: List[Tuple[int, int, int]]) -> Dict:
        edges_sorted = sorted(mst_edges, key=lambda x: -x[2])
        
        # Initialize each clique as a leaf node
        tree_nodes = {}
        for clique_id in range(len(self.all_cliques)):
            tree_nodes[clique_id] = {
                'type': 'leaf',
                'weight': len(self.clique_id_map[clique_id]) - 1,  # w(v) = |C| - 1
                'children': [],
                'parent': None,
                'level': 0, 
                'id': clique_id 
            }
        
        root = {i: i for i in range(len(self.all_cliques) * 2 - 1)}
        
        def find(u):
            while root[u] != u:
                root[u] = root[root[u]]
                u = root[u]
            return u

        for c1, c2, weight in edges_sorted:
            root1 = find(c1)
            root2 = find(c2)
            
            child1_level = tree_nodes[root1]['level']
            child2_level = tree_nodes[root2]['level']
            new_level = max(child1_level, child2_level) + 1
            
            # Create a new internal node
            new_node_id = len(tree_nodes)
            tree_nodes[new_node_id] = {
                'type': 'internal',
                'weight': weight,
                'children': [root1, root2],
                'parent': None,
                'level': new_level
            }
            
            tree_nodes[root1]['parent'] = new_node_id
            tree_nodes[root2]['parent'] = new_node_id

            root[root1] = new_node_id
            root[root2] = new_node_id

        return tree_nodes

    def build_dcpc_index(self):
        
        # 1. Build Clique Adjacency Graph (G_f)
        edges_gf = self.build_clique_adjacency_graph()
        
        # 2. Build Clique Adjacency Tree (T_f)
        mst_edges = self.build_clique_adjacency_tree(edges_gf)
        
        # 3. Build Ordered Adjacency Tree (T_o)
        tree_to = self.build_ordered_adjacency_tree(mst_edges)
        
        # 4. Return index structure
        return {
            'T_o': tree_to
        }

    
    def write_results_to_file(self, query_results: Dict[int, List[Set[int]]], output_path='./output/kclique.txt'):
        """Write the query results to a file."""
        # Clear the file first
        with open(output_path, 'w') as f:
            f.write("")
        
        # Now append the results
        with open(output_path, 'a') as f:
            for query_node, cliques in query_results.items():
                f.write(f'Result for query node: {query_node}\n')
                # If there are no cliques containing this node
                if not cliques:
                    f.write("No cliques found containing this node.\n")
                    continue
                
                # For each clique, write all edges
                for clique in cliques:
                    f.write(f"# Clique: {sorted(clique)}\n")
                    clique_list = sorted(list(clique))
                    # Output each edge in the clique
                    for i in range(len(clique_list)):
                        for j in range(i + 1, len(clique_list)):
                            f.write(f'{clique_list[i]} {clique_list[j]}\n')
                f.write('\n')

    def dcpc_query(self, query_nodes: Set[int]) -> Set[int]:
        # Use local variables to avoid memory accumulation
        priority_queue = []
        bitmap_set = {}
        
        try:
            # initDCPC
            self._init_dcpc(query_nodes, priority_queue, bitmap_set)
            
            # computeDCPC
            dcpc_tree_node = self._compute_dcpc(priority_queue, bitmap_set)
            
            # retrieveDCPC
            if dcpc_tree_node:
                dcpc = self._retrieve_dcpc(dcpc_tree_node)
                print(f"Found DCPC with {len(dcpc)} nodes and k-value {dcpc_tree_node[1]['weight']+1}")
                return dcpc
            else:
                print("No DCPC found containing all query nodes")
                return set()
        finally:
            # Explicit cleanup
            priority_queue.clear()
            bitmap_set.clear()
            gc.collect()

    def _init_dcpc(self, query_nodes: Set[int], priority_queue: List, bitmap_set: Dict):
        to_tree = self._get_or_build_dcpc_index()['T_o']
        query_nodes_list = list(query_nodes)  
        
        # Use set to avoid duplicates in priority queue
        added_nodes = set()
        
        for i, vi in enumerate(query_nodes_list):
            if vi not in self.node_to_clique_ids:
                continue
                
            for clique_idx in self.node_to_clique_ids[vi]:
                leaf_node_id = self._find_leaf_node(clique_idx)
                if leaf_node_id is None or leaf_node_id in added_nodes:
                    continue
                
                # Initialize bitmap for this leaf node
                if leaf_node_id not in bitmap_set:
                    bitmap_set[leaf_node_id] = [0] * len(query_nodes_list)
                
                # Set bits for all query nodes contained in this clique
                for j, vj in enumerate(query_nodes_list):
                    if vj in self.all_cliques[clique_idx]:
                        bitmap_set[leaf_node_id][j] = 1
                
                # Add to priority queue only once
                priority_queue.append((leaf_node_id, to_tree[leaf_node_id]))
                added_nodes.add(leaf_node_id)
        
        # Sort once at initialization instead of repeatedly during computation
        priority_queue.sort(key=lambda item: item[1]['level'])
        added_nodes.clear()
    
    def _compute_dcpc(self, priority_queue: List, bitmap_set: Dict):
        to_tree = self._get_or_build_dcpc_index()['T_o']
        candidate_nodes = []
        processed_nodes = set()
        
        # Convert to heap for efficient operations
        heap = [(item[1]['level'], item[0], item[1]) for item in priority_queue]
        heapq.heapify(heap)
        priority_queue.clear()  # Free original list memory
        
        iteration_count = 0
        max_iterations = len(to_tree) * 2  # Safety limit
        
        while heap and iteration_count < max_iterations:
            iteration_count += 1
            level, node_id, node = heapq.heappop(heap)
            
            if node_id in processed_nodes:
                continue
                
            processed_nodes.add(node_id)
            parent_id = node.get('parent')
            
            if parent_id is None:
                continue
            
            parent_node = to_tree[parent_id]
            
            # Check if this is a valid candidate
            if node_id in bitmap_set and self._all_bits_set(bitmap_set[node_id]) and node['weight'] > parent_node['weight']:
                candidate_nodes.append((node_id, node))
            else:
                # Update parent bitmap
                if parent_id not in processed_nodes:
                    if parent_id not in bitmap_set and node_id in bitmap_set:
                        bitmap_set[parent_id] = [0] * len(bitmap_set[node_id])
                    
                    if node_id in bitmap_set and parent_id in bitmap_set:
                        for i in range(len(bitmap_set[node_id])):
                            bitmap_set[parent_id][i] |= bitmap_set[node_id][i]
                    
                    # Add parent to heap
                    heapq.heappush(heap, (parent_node['level'], parent_id, parent_node))
        
        if iteration_count >= max_iterations:
            print(f"[WARNING] DCPC computation reached maximum iterations ({max_iterations}), terminating")
            
        # Clean up
        processed_nodes.clear()
        
        if candidate_nodes:
            return max(candidate_nodes, key=lambda item: item[1]['weight'])
        return None

    def _retrieve_dcpc(self, tree_node_info):
        if tree_node_info is None:
            return set()
            
        node_id, node = tree_node_info
        to_tree = self._get_or_build_dcpc_index()['T_o']
        
        leaf_node_ids = []
        stack = [node_id]

        
        while stack:
            current_id = stack.pop()
            current_node = to_tree[current_id]
            
            if current_node['type'] == 'leaf':
                leaf_node_ids.append(current_id)
            else:
                stack.extend(current_node['children'])
        
        dcpc = set()
        for leaf_id in leaf_node_ids:
            dcpc.update(self.all_cliques[leaf_id])
        
        return dcpc
    
    def _all_bits_set(self, bitmap):
        """Check if all bits in the bitmap are set to 1."""
        return all(bit == 1 for bit in bitmap)
    
    def _find_leaf_node(self, clique_idx):
        """Find leaf node for given clique index with caching"""
        to_tree = self._get_or_build_dcpc_index()['T_o']
        
        if clique_idx in to_tree and to_tree[clique_idx].get('type') == 'leaf':
            return clique_idx
        
        for node_id, node in to_tree.items():
            if node.get('type') == 'leaf' and node_id == clique_idx:
                return node_id 
        
        return None

    def kcpc_query_with_k(self, query_nodes: Set[int], k_value: int) -> Set[int]:
        k_value -= 1
        # Use local variables to avoid memory accumulation
        priority_queue = []
        bitmap_set = {}
        
        try:
            # Initialize the DCPC search process
            self._init_dcpc(query_nodes, priority_queue, bitmap_set)
            
            # Compute the KCPC tree node with k >= k_value and maximum level
            kcpc_tree_node = self._compute_kcpc_with_k(priority_queue, bitmap_set, k_value)
            
            # Retrieve the KCPC
            if kcpc_tree_node:
                kcpc = self._retrieve_dcpc(kcpc_tree_node)
                return kcpc 
            else:
                return set()
        finally:
            # Explicit cleanup
            priority_queue.clear()
            bitmap_set.clear()
            gc.collect()
    
    def _compute_kcpc_with_k(self, priority_queue: List, bitmap_set: Dict, k_value: int):
        to_tree = self._get_or_build_dcpc_index()['T_o']
        candidate_nodes = []
        processed_nodes = set()
        
        # Use heap for efficient priority queue operations
        heap = [(item[1]['level'], item[0], item[1]) for item in priority_queue]
        heapq.heapify(heap)
        priority_queue.clear()  # Free original list memory
        
        iteration_count = 0
        max_iterations = len(to_tree) * 2  # Safety limit to prevent infinite loops
        
        while heap and iteration_count < max_iterations:
            iteration_count += 1
            level, node_id, node = heapq.heappop(heap)
            
            if node_id in processed_nodes:
                continue
                
            processed_nodes.add(node_id)
            
            # Check if this node satisfies the conditions
            if node_id in bitmap_set and self._all_bits_set(bitmap_set[node_id]) and node['weight'] >= k_value:
                candidate_nodes.append((node_id, node))
            
            parent_id = node.get('parent')
            if parent_id is not None and parent_id not in processed_nodes:
                parent_node = to_tree[parent_id]

                # Initialize parent bitmap if needed
                if parent_id not in bitmap_set and node_id in bitmap_set:
                    bitmap_set[parent_id] = [0] * len(bitmap_set[node_id])
                
                # Update parent bitmap
                if node_id in bitmap_set and parent_id in bitmap_set:
                    for i in range(len(bitmap_set[node_id])):
                        bitmap_set[parent_id][i] |= bitmap_set[node_id][i]
                
                # Add parent to heap if not already processed
                heapq.heappush(heap, (parent_node['level'], parent_id, parent_node))
        
        if iteration_count >= max_iterations:
            print(f"[WARNING] KCPC computation reached maximum iterations ({max_iterations}), terminating")

        # Clean up processed nodes set
        processed_nodes.clear()
        
        if candidate_nodes:
            # Sort by level (descending) to get the highest level candidate
            candidate_nodes.sort(key=lambda item: -item[1].get('level', 0))
            return candidate_nodes[0]
        
        return None

    def clear_cache(self):
        """Clear cached DCPC index to free memory"""
        if self.dcpc_index is not None:
            self.dcpc_index.clear()
            self.dcpc_index = None
            gc.collect()
            print("[KCLIQUE] Cleared cached DCPC index")
    
    def get_memory_info(self):
        """Get memory usage information for debugging"""
        info = {
            'num_cliques': len(self.all_cliques),
            'dcpc_index_cached': self.dcpc_index is not None
        }
        if self.dcpc_index:
            info['tree_nodes'] = len(self.dcpc_index.get('T_o', {}))
        return info


def read_query_from_file(path: str) -> Set[int]:
    """Read query nodes from a file."""
    query_set = set()
    
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue  # skip empty lines
            parts = line.strip().split()
            if len(parts) == 1:
                q = int(parts[0])
                query_set.add(q)
    
    return query_set


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python Index.py <graph_path> <query_path> [k_value]")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    query_path = sys.argv[2]
    
    # Initialize KCliqueIndex with the graph
    kclique_index = KCliqueIndex(graph_path)
    
    # Read the query nodes
    query_nodes = read_query_from_file(query_path)
    print(f"Query nodes: {query_nodes}")
    
    # Check if k_value is provided
    if len(sys.argv) == 4:
        k_value = int(sys.argv[3])
        print(f"Finding KCPC with k >= {k_value}...")
        result = kclique_index.kcpc_query_with_k(query_nodes, k_value)
        # Write results to output file
        with open('./output/kclique.txt', 'w') as f:
            f.write(f'Result for KCPC query with nodes {query_nodes} and k >= {k_value}:\n')
            if result:
                f.write(f"# KCPC nodes: {sorted(result)}\n")
                # Output edges in the KCPC
                result_list = sorted(list(result))
                for i in range(len(result_list)):
                    for j in range(i + 1, len(result_list)):
                        u, v = result_list[i], result_list[j]
                        if v in kclique_index.graph.adj[u]:
                            f.write(f'{u} {v}\n')
            else:
                f.write(f"No KCPC found containing all query nodes with k >= {k_value}.\n")
    else:
        print("Finding densest clique percolation community...")
        dcpc = kclique_index.dcpc_query(query_nodes)
        # Write results to output file
        with open('./output/kclique.txt', 'w') as f:
            f.write(f'Result for DCPC query with nodes {query_nodes}:\n')
            if dcpc:
                f.write(f"# DCPC nodes: {sorted(dcpc)}\n")
                # Output edges in the DCPC
                dcpc_list = sorted(list(dcpc))
                for i in range(len(dcpc_list)):
                    for j in range(i + 1, len(dcpc_list)):
                        u, v = dcpc_list[i], dcpc_list[j]
                        if v in kclique_index.graph.adj[u]:
                            f.write(f'{u} {v}\n')
            else:
                f.write("No DCPC found containing all query nodes.\n")
    
    print("Results written to ./output/kclique.txt")


if __name__ == "__main__":
    main()