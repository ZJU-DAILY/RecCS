import numpy as np
import networkx as nx
from collections import defaultdict, deque
import heapq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
import sys
import os
import time
from heapq import heappush, heappop
from networkx.algorithms.approximation.steinertree import steiner_tree as nx_steiner_tree
warnings.filterwarnings('ignore')


class FreeRiderDetector:
    def __init__(self, graph, decay_factor=0.9, max_iterations=100, tolerance=1e-6):
        self.graph = graph
        self.decay_factor = decay_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if graph.is_directed():
            self.graph = graph.to_undirected()
        self.nodes = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)
        self.degrees = {}
        for node in self.nodes:
            weighted_degree = sum(self.graph[node][neighbor].get('weight', 1.0)
                                  for neighbor in self.graph.neighbors(node))
            self.degrees[node] = weighted_degree
        self.max_degree = max(self.degrees.values()) if self.degrees else 1

        self._build_transition_matrix()

    def _build_transition_matrix(self):
        from scipy.sparse import csr_matrix
        edges_data = [(self.node_to_idx[u], self.node_to_idx[v], d.get('weight', 1.0))
                      for u, v, d in self.graph.edges(data=True)]

        if not edges_data:
            self.transition_matrix = csr_matrix((self.n_nodes, self.n_nodes))
            self.transition_matrix_dense = None
            return
        col_indices, row_indices, weights = zip(*edges_data)
        all_rows = list(row_indices) + list(col_indices)
        all_cols = list(col_indices) + list(row_indices)
        all_weights = list(weights) + list(weights)
        node_degrees = {}
        for i, node in enumerate(self.nodes):
            node_degrees[i] = sum(self.graph[node][neighbor].get('weight', 1.0)
                                  for neighbor in self.graph.neighbors(node))

        w_max = max(node_degrees.values()) if node_degrees else 1.0
        normalized_weights = [w / w_max for w in all_weights]

        self.transition_matrix = csr_matrix(
            (normalized_weights, (all_rows, all_cols)),
            shape=(self.n_nodes, self.n_nodes)
        )
        self.transition_matrix_dense = None

    def compute_node_weights(self, query_nodes):
        query_indices = [self.node_to_idx[node] for node in query_nodes if node in self.node_to_idx]

        if not query_indices:
            raise ValueError("No valid query nodes found in the graph")

        if self.n_nodes > 5000:
            print(f"[QDC] Using power iteration for large graph ({self.n_nodes} nodes)")
            proximity = self._power_iteration_solve_corrected(query_indices)
        else:
            proximity = self._direct_solve_linear_system(query_indices)

        eps = 1e-8
        proximity = np.maximum(proximity, eps)

        weights = {}
        for i, node in enumerate(self.nodes):
            weights[node] = 1.0 / proximity[i]

        return weights

    def _direct_solve_linear_system(self, query_indices):
        from scipy.sparse import eye
        from scipy.sparse.linalg import bicgstab

        proximity = np.zeros(self.n_nodes)

        query_mask = np.zeros(self.n_nodes, dtype=bool)
        query_mask[query_indices] = True
        non_query_mask = ~query_mask

        non_query_indices = np.where(non_query_mask)[0]
        n_reduced = len(non_query_indices)

        if n_reduced == 0:
            proximity[query_mask] = 1.0
        else:
            non_query_indices_array = np.array(non_query_indices)
            P_reduced = self.transition_matrix[non_query_indices_array][:, non_query_indices_array]
            P_to_query = self.transition_matrix[non_query_indices_array][:, query_indices]
            b_reduced = self.decay_factor * np.array(P_to_query.sum(axis=1)).flatten()
            I_reduced = eye(n_reduced, format='csr')
            A_sparse = I_reduced - self.decay_factor * P_reduced.T
            try:
                r_reduced, info = bicgstab(A_sparse, b_reduced, tol=self.tolerance)
                if info != 0:
                    r_reduced = spsolve(A_sparse, b_reduced)
            except:
                r_reduced = spsolve(A_sparse, b_reduced)
            proximity[non_query_indices] = r_reduced
        proximity[query_mask] = 1.0

        return proximity

    def _power_iteration_solve_corrected(self, query_indices):
        proximity = np.zeros(self.n_nodes)
        query_mask = np.zeros(self.n_nodes, dtype=bool)
        query_mask[query_indices] = True
        non_query_mask = ~query_mask
        proximity[query_mask] = 1.0
        for iteration in range(self.max_iterations):
            if self.transition_matrix_dense is not None:
                new_proximity = self.decay_factor * np.dot(self.transition_matrix_dense, proximity)
            else:
                new_proximity = self.decay_factor * self.transition_matrix.dot(proximity)
            old_non_query = proximity[non_query_mask].copy()
            proximity[non_query_mask] = new_proximity[non_query_mask]
            proximity[query_mask] = 1.0
            diff = proximity[non_query_mask] - old_non_query
            convergence_error = np.linalg.norm(diff)

            if convergence_error < self.tolerance:
                print(f"[QDC] Power iteration converged after {iteration + 1} iterations")
                break
        else:
            print(f"[QDC] Power iteration reached max iterations ({self.max_iterations})")

        return proximity

    def query_biased_density(self, subgraph_nodes, node_weights):
        if not subgraph_nodes:
            return 0.0
        if isinstance(subgraph_nodes, (list, tuple)):
            nodes_array = np.array(subgraph_nodes)
        else:
            nodes_array = np.array(list(subgraph_nodes))

        if len(nodes_array) == 0:
            return 0.0
        subgraph = self.graph.subgraph(nodes_array)
        edge_count = sum(d.get('weight', 1.0) for u, v, d in subgraph.edges(data=True))
        weight_sum = sum(node_weights[node] for node in nodes_array)
        if weight_sum == 0:
            return 0.0

        return edge_count / weight_sum

    def solve_qdcii(self, node_weights):
        nodes_remaining = set(self.nodes)
        best_subgraph = set()
        best_density = 0.0
        density_threshold = 0.0
        while nodes_remaining:
            qb_degrees = {}
            for node in nodes_remaining:
                degree = 0
                for neighbor in self.graph.neighbors(node):
                    if neighbor in nodes_remaining:
                        edge_weight = self.graph[node][neighbor].get('weight', 1.0)
                        degree += edge_weight
                qb_degrees[node] = degree / node_weights[node]

            current_density = self.query_biased_density(list(nodes_remaining), node_weights)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = nodes_remaining.copy()
                density_threshold = current_density

            if qb_degrees:
                min_node = min(qb_degrees.keys(), key=lambda x: qb_degrees[x])
                nodes_remaining.remove(min_node)
        if len(best_subgraph) > 10:
            core_nodes = self._compute_query_biased_core(best_subgraph, node_weights, density_threshold)
            if core_nodes:
                best_subgraph = core_nodes
        if len(best_subgraph) <= 100:
            try:
                exact_solution = self._solve_parametric_max_flow(best_subgraph, node_weights)
                if exact_solution and len(exact_solution) > 0:
                    exact_density = self.query_biased_density(list(exact_solution), node_weights)
                    if exact_density > best_density:
                        best_subgraph = exact_solution
                        best_density = exact_density
            except Exception as e:
                print(f"Warning: Parametric max-flow failed, using GND result: {e}")

        return best_subgraph

    def _solve_parametric_max_flow(self, candidate_nodes, node_weights):
        import networkx as nx

        if len(candidate_nodes) <= 1:
            return candidate_nodes

        candidate_list = list(candidate_nodes)
        best_density = 0.0
        best_subgraph = candidate_nodes
        node_degrees = {}
        for node in candidate_list:
            degree = sum(self.graph[node][neighbor].get('weight', 1.0)
                         for neighbor in self.graph.neighbors(node)
                         if neighbor in candidate_nodes)
            node_degrees[node] = degree
        density_candidates = []
        for node in candidate_list:
            if node_weights[node] > 0:
                density_candidates.append(node_degrees[node] / node_weights[node])

        density_candidates = sorted(set(density_candidates))
        density_candidates.insert(0, 0.0)
        for density_threshold in density_candidates:
            try:
                flow_net = nx.DiGraph()
                source, sink = 'S', 'T'
                flow_net.add_nodes_from([source, sink] + candidate_list)
                source_capacity = 0
                for node in candidate_list:
                    net_value = node_degrees[node] - density_threshold * node_weights[node]

                    if net_value > 1e-8:
                        flow_net.add_edge(source, node, capacity=net_value)
                        source_capacity += net_value
                    elif net_value < -1e-8:
                        flow_net.add_edge(node, sink, capacity=-net_value)
                for node in candidate_list:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in candidate_nodes and node < neighbor:
                            cap = max(1000, source_capacity * 2) if source_capacity > 0 else 1000
                            flow_net.add_edge(node, neighbor, capacity=cap)
                            flow_net.add_edge(neighbor, node, capacity=cap)

                if source_capacity > 1e-8:
                    flow_value, flow_dict = nx.maximum_flow(flow_net, source, sink)
                    residual_capacity = {}
                    for u in flow_net.nodes():
                        residual_capacity[u] = {}
                        for v in flow_net[u]:
                            residual_capacity[u][v] = (flow_net[u][v]['capacity'] -
                                                       flow_dict[u].get(v, 0))
                    reachable = {source}
                    queue = [source]
                    while queue:
                        u = queue.pop(0)
                        for v in flow_net[u]:
                            if v not in reachable and residual_capacity[u][v] > 1e-8:
                                reachable.add(v)
                                queue.append(v)
                    dense_nodes = reachable & set(candidate_list)

                    if len(dense_nodes) > 0:
                        actual_density = self.query_biased_density(list(dense_nodes), node_weights)
                        if actual_density > best_density:
                            best_density = actual_density
                            best_subgraph = dense_nodes

            except Exception:
                continue

        return best_subgraph

    def _compute_query_biased_core(self, nodes, node_weights, min_density):
        core_nodes = set(nodes)
        changed = True

        while changed:
            changed = False
            to_remove = set()

            for node in core_nodes:
                qb_degree = 0
                for neighbor in self.graph.neighbors(node):
                    if neighbor in core_nodes:
                        edge_weight = self.graph[node][neighbor].get('weight', 1.0)
                        qb_degree += edge_weight
                qb_degree /= node_weights[node]
                if qb_degree < min_density:
                    to_remove.add(node)
                    changed = True

            core_nodes -= to_remove

        return core_nodes

    def solve_qdci(self, query_nodes, node_weights):
        print(f"[QDC] Starting QDCI with {len(query_nodes)} query nodes")
        query_set = set(query_nodes)
        current_contracted_set = query_set.copy()

        iteration = 0
        max_iterations = self.n_nodes

        print(f"[QDC] QDCI max_iterations: {max_iterations}")

        while iteration < max_iterations:
            print(f"[QDC] QDCI iteration {iteration}, contracted_set size: {len(current_contracted_set)}")

            contracted_graph, supernode_id, node_mapping = self._create_contracted_graph(
                current_contracted_set, node_weights
            )

            if contracted_graph.number_of_nodes() <= 1:
                print(f"[QDC] QDCI early termination: contracted graph too small")
                break

            contracted_weights = self._compute_contracted_weights(
                contracted_graph, supernode_id, node_weights, node_mapping, current_contracted_set
            )

            densest_nodes = self._solve_qdcii_on_contracted(
                contracted_graph, contracted_weights
            )

            if supernode_id in densest_nodes:
                print(f"[QDC] QDCI found solution at iteration {iteration}")
                result = set()
                for node in densest_nodes:
                    if node == supernode_id:
                        result.update(current_contracted_set)
                    else:
                        original_node = node_mapping.get(node, node)
                        result.add(original_node)
                return result

            
            densest_original = set()
            for node in densest_nodes:
                if node != supernode_id:
                    original_node = node_mapping.get(node, node)
                    densest_original.add(original_node)

            old_size = len(current_contracted_set)
            current_contracted_set = current_contracted_set.union(densest_original)

            
            if len(current_contracted_set) == old_size:
                print(f"[QDC] QDCI early termination: no progress made")
                break

            
            if len(current_contracted_set) >= self.n_nodes:
                print(f"[QDC] QDCI termination: contracted all nodes")
                break

            iteration += 1

        return current_contracted_set

    def _create_contracted_graph(self, contracted_set, node_weights):
        contracted_graph = nx.Graph()
        supernode_id = 'supernode'
        node_mapping = {} 
        for node in self.graph.nodes():
            if node not in contracted_set:
                contracted_graph.add_node(node)
                node_mapping[node] = node
        contracted_graph.add_node(supernode_id)
        for u, v in self.graph.edges():
            u_in_contracted = u in contracted_set
            v_in_contracted = v in contracted_set

            if u_in_contracted and v_in_contracted:
                pass
            elif u_in_contracted:
                edge_weight = self.graph[u][v].get('weight', 1.0)
                if contracted_graph.has_edge(supernode_id, v):
                    contracted_graph[supernode_id][v]['weight'] += edge_weight
                else:
                    contracted_graph.add_edge(supernode_id, v, weight=edge_weight)

            elif v_in_contracted:
                edge_weight = self.graph[u][v].get('weight', 1.0)
                if contracted_graph.has_edge(u, supernode_id):
                    contracted_graph[u][supernode_id]['weight'] += edge_weight
                else:
                    contracted_graph.add_edge(u, supernode_id, weight=edge_weight)

            else:
                edge_weight = self.graph[u][v].get('weight', 1.0)
                contracted_graph.add_edge(u, v, weight=edge_weight)

        return contracted_graph, supernode_id, node_mapping

    def _compute_contracted_weights(self, contracted_graph, supernode_id, original_weights, node_mapping,
                                    current_contracted_set):
        contracted_weights = {}

        for node in contracted_graph.nodes():
            if node == supernode_id:
                contracted_weights[node] = sum(
                    original_weights[original_node] for original_node in current_contracted_set)
            else:
                original_node = node_mapping.get(node, node)
                contracted_weights[node] = original_weights[original_node]

        return contracted_weights

    def _solve_qdcii_on_contracted(self, contracted_graph, contracted_weights):
        nodes_remaining = set(contracted_graph.nodes())
        best_subgraph = set()
        best_density = 0.0

        while nodes_remaining:
            qb_degrees = {}
            for node in nodes_remaining:
                degree = 0
                for neighbor in contracted_graph.neighbors(node):
                    if neighbor in nodes_remaining:
                        edge_weight = contracted_graph[node][neighbor].get('weight', 1.0)
                        degree += edge_weight
                if contracted_weights[node] > 0:
                    qb_degrees[node] = degree / contracted_weights[node]
                else:
                    qb_degrees[node] = 0

            edge_count = 0
            weight_sum = 0
            for node in nodes_remaining:
                weight_sum += contracted_weights[node]
                for neighbor in contracted_graph.neighbors(node):
                    if neighbor in nodes_remaining:
                        if str(node) <= str(neighbor): 
                            edge_weight = contracted_graph[node][neighbor].get('weight', 1.0)
                            edge_count += edge_weight

            current_density = edge_count / weight_sum if weight_sum > 0 else 0
            if current_density > best_density:
                best_density = current_density
                best_subgraph = nodes_remaining.copy()

            if qb_degrees:
                min_node = min(qb_degrees.keys(), key=lambda x: qb_degrees[x])
                nodes_remaining.remove(min_node)

        return best_subgraph

    def is_connected(self, nodes):
        if len(nodes) <= 1:
            return True

        node_set = set(nodes)
        subgraph = self.graph.subgraph(nodes)
        return nx.is_connected(subgraph)

    def get_connected_components(self, nodes):
        if not nodes:
            return []
        subgraph = self.graph.subgraph(nodes)
        return [list(component) for component in nx.connected_components(subgraph)]
    def solve_qdc_heuristic_greedy_old(self, query_nodes, node_weights, alpha=2.0):
        import networkx as nx
        components = nx.connected_components(self.graph)
        target_component = None
        for comp in components:
            if all(q in comp for q in query_nodes):
                target_component = set(comp)
                break

        if target_component is None:
            return set(query_nodes)

        nodes_remaining = target_component
        best_subgraph = set(query_nodes)
        best_density = self.query_biased_density(list(best_subgraph), node_weights)

        while len(nodes_remaining) > len(query_nodes):
            subgraph = self.graph.subgraph(nodes_remaining)
            if not nx.is_connected(subgraph):
                break
            articulation_nodes = set(nx.articulation_points(subgraph))
            candidates = nodes_remaining - articulation_nodes - set(query_nodes)
            if not candidates:
                qb_degrees = {}
                for node in nodes_remaining - set(query_nodes):
                    degree = sum(self.graph[node][neighbor].get('weight', 1.0)
                                 for neighbor in self.graph.neighbors(node)
                                 if neighbor in nodes_remaining)
                    qb_degrees[node] = degree / node_weights[node]

                if qb_degrees:
                    min_node = min(qb_degrees.keys(), key=lambda x: qb_degrees[x])
                    candidates = {min_node}

            if not candidates:
                break
            current_density = self.query_biased_density(list(nodes_remaining), node_weights)
            threshold = alpha * current_density
            to_delete = set()
            for node in candidates:
                degree = sum(self.graph[node][neighbor].get('weight', 1.0)
                             for neighbor in self.graph.neighbors(node)
                             if neighbor in nodes_remaining)
                qb_degree = degree / node_weights[node]

                if qb_degree <= threshold:
                    to_delete.add(node)

            if not to_delete:
                qb_degrees = {}
                for node in candidates:
                    degree = sum(self.graph[node][neighbor].get('weight', 1.0)
                                 for neighbor in self.graph.neighbors(node)
                                 if neighbor in nodes_remaining)
                    qb_degrees[node] = degree / node_weights[node]

                min_node = min(qb_degrees.keys(), key=lambda x: qb_degrees[x])
                to_delete = {min_node}
            nodes_remaining -= to_delete
            current_density = self.query_biased_density(list(nodes_remaining), node_weights)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = nodes_remaining.copy()

        return best_subgraph

    def solve_qdc_heuristic_greedy(self, query_nodes, node_weights, alpha=2.0, time_limit = 10000):
        start = time.time()
        G = self.graph
        query_nodes = set(query_nodes)

        for comp in nx.connected_components(G):
            if query_nodes.issubset(comp):
                nodes_remaining = set(comp)
                break
        else:
            return set(query_nodes)

        best_subgraph = nodes_remaining.copy()
        best_density = self.query_biased_density(list(best_subgraph), node_weights)
        neighbors = {n: set(G[n]) for n in nodes_remaining}
        degree = {
            n: sum(G[n][v].get("weight", 1.0) for v in neighbors[n] if v in nodes_remaining)
            for n in nodes_remaining
        }
        qb_degree = {n: degree[n] / node_weights[n] for n in nodes_remaining}
        while len(nodes_remaining) > len(query_nodes):
            print(f"remaining nodes: {len(nodes_remaining)}")
            current_density = self.query_biased_density(list(nodes_remaining), node_weights)
            threshold = alpha * current_density
            subgraph = G.subgraph(nodes_remaining)
            bccs = list(nx.biconnected_components(subgraph))
            to_delete = set()
            flag =  False
            for bcc in bccs:
                bcc_nodes = set(bcc)

                if bcc_nodes <= query_nodes:
                    continue

                candidates = bcc_nodes - query_nodes
                if not candidates:
                    continue

                weak_nodes = [n for n in candidates if qb_degree.get(n, float('inf')) <= threshold]

                if weak_nodes:
                    min_node = min(weak_nodes, key=lambda n: qb_degree[n])
                    flag = True
                else:
                    min_node = min(candidates, key=lambda n: qb_degree[n])

                to_delete.add(min_node)

            if not to_delete:
                break
            if not flag and len(to_delete) > 1:
                min_degree_node = min(to_delete, key=lambda n: qb_degree[n])
                to_delete = {min_degree_node}

            for n in to_delete:
                if n not in nodes_remaining:
                    continue
                nodes_remaining.remove(n)
                for nb in neighbors[n]:
                    if nb in nodes_remaining:
                        degree[nb] -= G[nb][n].get("weight", 1.0)
                        qb_degree[nb] = degree[nb] / node_weights[nb]
                neighbors.pop(n, None)
                degree.pop(n, None)
                qb_degree.pop(n, None)

            current_density = self.query_biased_density(list(nodes_remaining), node_weights)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = nodes_remaining.copy()
            end = time.time()
            if end - start > time_limit:
                return best_subgraph
        return best_subgraph

    def solve_qdc_heuristic_steiner_old(self, query_nodes, node_weights, k=1000):
        import networkx as nx
        if len(query_nodes) == 1:
            steiner_nodes = set(query_nodes)
        else:
            steiner_nodes = set(query_nodes)
            for i, node1 in enumerate(query_nodes):
                for node2 in query_nodes[i + 1:]:
                    try:
                        path = nx.shortest_path(self.graph, node1, node2, weight='weight')
                        steiner_nodes.update(path)
                    except nx.NetworkXNoPath:
                        continue

        current_subgraph = steiner_nodes.copy()
        best_subgraph = current_subgraph.copy()
        best_density = self.query_biased_density(list(best_subgraph), node_weights)
        while len(current_subgraph) < k:
            boundary = set()
            for node in current_subgraph:
                boundary.update(self.graph.neighbors(node))
            boundary -= current_subgraph
            if not boundary:
                break
            best_candidate = None
            best_adjacency = 0

            for candidate in boundary:
                edge_weight_sum = sum(
                    self.graph[candidate][neighbor].get('weight', 1.0)
                    for neighbor in self.graph.neighbors(candidate)
                    if neighbor in current_subgraph
                )
                adjacency_value = edge_weight_sum / node_weights[candidate]

                if adjacency_value > best_adjacency:
                    best_adjacency = adjacency_value
                    best_candidate = candidate

            if best_candidate is None:
                break
            current_subgraph.add(best_candidate)
            current_density = self.query_biased_density(list(current_subgraph), node_weights)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = current_subgraph.copy()

        return best_subgraph

    def _approximate_steiner_tree_by_shortest_paths(self, G, query_nodes):
        steiner_nodes = set(query_nodes)
        query_list = list(query_nodes)
        for i, node1 in enumerate(query_list):
            for node2 in query_list[i + 1:]:
                path = nx.shortest_path(G, node1, node2)
                steiner_nodes.update(path)
        return steiner_nodes

    def solve_qdc_heuristic_steiner(self, query_nodes, node_weights, k=1000):
        print(f"[QDC] begin heuristic MAS algorithm")
        start_time = time.time()
        max_runtime = 1000.0
        G = self.graph
        query_nodes = set(query_nodes)
        if len(query_nodes) == 1:
            steiner_nodes = set(query_nodes)
        else:
            try:
                steiner_subgraph = nx_steiner_tree(G, query_nodes)
                steiner_nodes = set(steiner_subgraph.nodes)
            except (KeyError, ValueError, nx.NetworkXError, TypeError) as e:
                steiner_nodes = self._approximate_steiner_tree_by_shortest_paths(G, query_nodes)
        
        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime:
            return steiner_nodes

        current_subgraph = steiner_nodes
        best_subgraph = current_subgraph.copy()
        best_density = self.query_biased_density(list(current_subgraph), node_weights)
        boundary = set()
        adj_score = {}

        for node in current_subgraph:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime:
                return best_subgraph
            
            for nb in G.neighbors(node):
                if nb not in current_subgraph:
                    boundary.add(nb)
                    adj_score[nb] = adj_score.get(nb, 0) + G[node][nb].get('weight', 1.0)
        while len(current_subgraph) <= k and boundary:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime:
                return best_subgraph
            best_candidate = max(boundary, key=lambda n: adj_score[n] / node_weights[n])
            current_subgraph.add(best_candidate)
            boundary.remove(best_candidate)
            for nb in G.neighbors(best_candidate):
                w = G[best_candidate][nb].get('weight', 1.0)
                if nb in current_subgraph:
                    continue
                adj_score[nb] = adj_score.get(nb, 0) + w
                boundary.add(nb)
            current_density = self.query_biased_density(list(current_subgraph), node_weights)
            if current_density > best_density:
                best_density = current_density
                best_subgraph = current_subgraph.copy()

        return best_subgraph


    def detect_community(self, query_nodes, method='auto', **kwargs):
        query_nodes = [node for node in query_nodes if node in self.graph.nodes()]
        if not query_nodes:
            raise ValueError("No valid query nodes found in the graph")

        print(f"[QDC] Computing node weights...")
        node_weights = self.compute_node_weights(query_nodes)
        print(f"[QDC] Node weights computed.")

        if method == 'auto':
            qdci_solution = self.solve_qdci(query_nodes, node_weights)
            print(f"[QDC] QDCI completed, solution size: {len(qdci_solution)}")
            if self.is_connected(qdci_solution):
                community = qdci_solution
                method_used = 'qdci_optimal'
            else:
                components = self.get_connected_components(qdci_solution)
                valid_component = None

                for component in components:
                    if all(q in component for q in query_nodes) and len(component) > len(query_nodes):
                        valid_component = set(component)
                        break

                if valid_component:
                    community = valid_component
                    method_used = 'qdci_approximate'
                else:
                    community1 = self.solve_qdc_heuristic_greedy(query_nodes, node_weights,
                                                                 kwargs.get('alpha', 2.0))
                    community2 = self.solve_qdc_heuristic_steiner(query_nodes, node_weights,
                                                                kwargs.get('k', 1000))

                    density1 = self.query_biased_density(list(community1), node_weights)
                    density2 = self.query_biased_density(list(community2), node_weights)

                    if density1 >= density2:
                        community = community1
                        method_used = 'heuristic_greedy'
                    else:
                        community = community2
                        method_used = 'heuristic_steiner'

        elif method == 'qdci':
            qdci_solution = self.solve_qdci(query_nodes, node_weights)
            print(f"[QDC] QDCI completed, solution size: {len(qdci_solution)}")
            community = qdci_solution
            method_used = 'qdci'

        elif method == 'heuristic1':
            community = self.solve_qdc_heuristic_greedy(query_nodes, node_weights,
                                                        kwargs.get('alpha', 2.0))
            method_used = 'heuristic_greedy'

        elif method == 'heuristic2':
            community = self.solve_qdc_heuristic_steiner(query_nodes, node_weights,
                                                        kwargs.get('k', 1000))
            method_used = 'heuristic_steiner'

        else:
            raise ValueError(f"Unknown method: {method}")

        community = set(community) | set(query_nodes)
        final_density = self.query_biased_density(list(community), node_weights)

        return {
            'community': community,
            'density': final_density,
            'method_used': method_used,
            'node_weights': node_weights
        }

    def evaluate_free_rider_effect(self, community, node_weights):
        if len(community) <= 1:
            return {
                'has_free_riders': False,
                'main_component': None,
                'free_rider_components': []
            }

        components = self.get_connected_components(community)

        if len(components) <= 1:
            if components:
                main_density = self.query_biased_density(components[0], node_weights)
                main_component = {
                    'component_id': 0,
                    'nodes': set(components[0]),
                    'size': len(components[0]),
                    'density': main_density
                }
            else:
                main_component = None

            return {
                'has_free_riders': False,
                'main_component': main_component,
                'free_rider_components': []
            }

        component_info = []
        for i, component in enumerate(components):
            density = self.query_biased_density(component, node_weights)
            component_info.append({
                'component_id': i,
                'nodes': set(component),
                'size': len(component),
                'density': density
            })

        component_info.sort(key=lambda x: x['density'], reverse=True)

        main_component = component_info[0] if component_info else None
        potential_free_riders = component_info[1:] if len(component_info) > 1 else []

        return {
            'has_free_riders': len(potential_free_riders) > 0,
            'main_component': main_component,
            'free_rider_components': potential_free_riders
        }


def read_graph_from_file(graph_path):
    G = nx.Graph()
    edges = []

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u == v:
                        continue
                    edges.append((u, v))
                except ValueError:
                    continue

    G.add_edges_from(edges)

    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def read_query_from_file(query_path):
    with open(query_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 1:
                try:
                    return int(parts[0])
                except ValueError:
                    continue

    raise ValueError("No valid query node found in the file")


def write_results_to_file(graph, query_node, community, result_info, output_path='./output/freerider.txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f'Result for FreeRider community query with node {query_node}:\n')

        if community:
            community_list = sorted(list(community))

            internal_edges = 0
            external_edges = 0
            community_set = set(community)

            for node in community:
                for neighbor in graph.neighbors(node):
                    if neighbor in community_set:
                        if node < neighbor:
                            internal_edges += 1
                    else:
                        external_edges += 1

            n = len(community)
            max_edges = n * (n - 1) // 2 if n > 1 else 0
            density = internal_edges / max_edges if max_edges > 0 else 0

            volume_community = sum(graph.degree(node) for node in community)
            volume_graph = sum(dict(graph.degree()).values())
            conductance = external_edges / min(volume_community,
                                               volume_graph - volume_community) if volume_community > 0 else float(
                'inf')

            f.write(f"# Community nodes: {community_list}\n")
            f.write(f"# Community size: {len(community)}\n")
            f.write(f"# Query-biased density: {result_info['query_biased_density']:.6f}\n")
            f.write(f"# Traditional density: {density:.4f}\n")
            f.write(f"# Conductance: {conductance:.6f}\n")
            f.write(f"# Method used: {result_info['method_used']}\n")
            f.write(f"# Internal edges: {internal_edges}\n")
            f.write(f"# External edges: {external_edges}\n")
            f.write(f"# Volume: {volume_community}\n")

            for i in range(len(community_list)):
                for j in range(i + 1, len(community_list)):
                    u, v = community_list[i], community_list[j]
                    if graph.has_edge(u, v):
                        f.write(f'{u} {v}\n')
        else:
            f.write("No community found containing the query node.\n")


def find_community_for_vertex(graph, query_vertex, decay_factor=0.9, method='auto', **kwargs):
    if query_vertex not in graph.nodes():
        return None, {"error": f"Vertex {query_vertex} not found in graph"}

    try:
        detector = FreeRiderDetector(graph, decay_factor=decay_factor)
        result = detector.detect_community([query_vertex], method=method, **kwargs)
        community = result['community']

        free_rider_analysis = detector.evaluate_free_rider_effect(
            community, result['node_weights']
        )

        info = {
            "query_biased_density": result['density'],
            "method_used": result['method_used'],
            "size": len(community),
            "has_free_riders": free_rider_analysis['has_free_riders'],
            "main_component_size": len(free_rider_analysis['main_component']['nodes']) if free_rider_analysis[
                'main_component'] else 0,
            "free_rider_components": len(free_rider_analysis['free_rider_components'])
        }

        if community and query_vertex in community:
            print(f"Found community for vertex {query_vertex}:")
            print(f"  Vertices: {sorted(community)}")
            print(f"  Size: {info['size']}")
            print(f"  Query-biased density: {info['query_biased_density']:.4f}")
            print(f"  Method used: {info['method_used']}")
            print(f"  Free riders detected: {info['has_free_riders']}")
            return sorted(list(community)), info
        else:
            print(f"No suitable community found for vertex {query_vertex}")
            return None, {"error": "No community found or query vertex not in result"}

    except Exception as e:
        print(f"Error in FreeRider algorithm: {e}")
        return None, {"error": str(e)}


def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    graph_path = sys.argv[1]
    query_path = sys.argv[2]

    print("FreeRider Algorithm Parameters:")
    print(f"  Default Decay Factor: 0.85")
    print(f"  Default Method: auto")
    print(f"  Max Iterations: 100")
    print(f"  Tolerance: 1e-6")
    print()

    try:
        graph = read_graph_from_file(graph_path)
        query_node = read_query_from_file(query_path)
        print(f"Query node: {query_node}")

        community, info = find_community_for_vertex(graph, query_node)

        output_path = './output/freerider.txt'
        write_results_to_file(graph, query_node, community, info, output_path)

        print(f"\nResults written to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 