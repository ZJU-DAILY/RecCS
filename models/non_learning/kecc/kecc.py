import struct
import os
import sys
import tempfile
from typing import List, Optional, Tuple, Any

# Define constants for file names and paths
CG_BIN_FILENAME: str = "cg.bin"
MSPT_FILENAME: str = "mSPT.txt"
DEFAULT_QUERY_RESULTS_DIR: str = "./output"
DEFAULT_QUERY_RESULTS_FILENAME: str = "query_results.txt"

INT_SIZE: int = 4

class LinearHeap:
    def __init__(self, _n):
        self.n = _n
        self.po = -1

        self.head = [-1] * self.n
        self.pre = [-1] * self.n
        self.next = [-1] * self.n
        self.key = [0] * self.n

    def insert(self, id_val, value):
        self.key[id_val] = value
        self.pre[id_val] = -1
        self.next[id_val] = self.head[value]
        if self.head[value] != -1:
            self.pre[self.head[value]] = id_val
        self.head[value] = id_val

        if value > self.po:
            self.po = value

    def get_key(self, id_val):
        return self.key[id_val]

    def set_key(self, id_val, value):
        self.key[id_val] = value

    def exist(self, id_val):
        return self.key[id_val] != 0

    def get_max(self):
        while self.po >= 0 and self.head[self.po] == -1:
            self.po -= 1

        if self.po < 0:
            return None 

        id_res = self.head[self.po]
        value_res = self.key[id_res]
        return id_res, value_res

    def extract_max(self):
        while self.po >= 0 and self.head[self.po] == -1:
            self.po -= 1

        if self.po < 0:
            return None

        id_res = self.head[self.po]
        value_res = self.key[id_res]

        self.head[self.po] = self.next[id_res]
        if self.head[self.po] != -1:
            self.pre[self.head[self.po]] = -1
        
        return id_res, value_res

    def remove(self, id_val):
        if self.pre[id_val] == -1:
            self.head[self.key[id_val]] = self.next[id_val]
            if self.next[id_val] != -1:
                self.pre[self.next[id_val]] = -1
        else:
            pid = self.pre[id_val]
            self.next[pid] = self.next[id_val]
            if self.next[id_val] != -1:
                self.pre[self.next[id_val]] = pid

    def update(self, id_val, value):
        self.remove(id_val)
        self.insert(id_val, value)

    def clear(self):
        for i in range(self.po + 1):
            if i < self.n : 
                self.head[i] = -1
        self.po = -1

class Element:
    def __init__(self) -> None:
        self.next: Optional[Element] = None
        self.value: int = 0  # Or Any, depending on usage

class Edge:
    def __init__(self) -> None:
        self.pre: Optional[Edge] = None
        self.next: Optional[Edge] = None
        self.duplicate: Optional[Edge] = None
        self.nontree: Optional[Edge] = None
        self.node_id: int = 0
        self.sc: int = 0
        self.deleted: int = 0

class Node:
    def __init__(self) -> None:
        self.head: Optional[Element] = None
        self.tail: Optional[Element] = None
        self.first: Optional[Edge] = None
        self.last: Optional[Edge] = None

class Bin:
    def __init__(self) -> None:
        self.edge: Optional[Edge] = None
        self.pre: Optional[Bin] = None
        self.next: Optional[Bin] = None

class Graph:
    def __init__(self):
        self.output_file = ""
        self.dir = str("./")
        self.k_connectivity_parameter: int = -1

        self.n = 0
        self.m = 0

        self.elements = None
        self.edges = None
        self.nodes = None

        self.pnodes = None
        self.pedges = None

        self.inL = None
        self.computed = None
        self.height = None

        # Queue for BFS or other iterative processing 
        self.processing_queue = None

        self.degrees = None

        # Heap for priority queue operations
        self.heap = None

        self.levels = None


    def set_k(self, _K):
        self.k_connectivity_parameter = _K

    def add_edge(self, node_obj, edge_obj):
        edge_obj.next = None
        if node_obj.first is None:
            node_obj.first = edge_obj
            node_obj.last = edge_obj
            edge_obj.pre = None
        else:
            if node_obj.last is None:
                current = node_obj.first
                while current is not None and current.next is not None:
                    current = current.next
                node_obj.last = current
            
            if node_obj.last is not None:
                node_obj.last.next = edge_obj
                edge_obj.pre = node_obj.last
                node_obj.last = edge_obj
            else:
                node_obj.first = edge_obj
                node_obj.last = edge_obj
                edge_obj.pre = None

    def read_graph(self,file_path):
        mode = "r"

        f = open(file_path, mode)

        edge_pairs_temp = []
        max_node_id = -1

        for line_content in f:
            parts = line_content.split()
            if len(parts) >= 2: 
                try:
                    a = int(parts[0])
                    b = int(parts[1])
                    edge_pairs_temp.append((a, b))

                    if a > max_node_id:
                        max_node_id = a
                    if b > max_node_id:
                        max_node_id = b
                except ValueError:
                    pass 

        f.close() # Close the file after reading

        # 3. Calculate n and m
        if not edge_pairs_temp and max_node_id == -1: # Handle empty graph or no valid edges
            self.n = 0
            self.m = 0
        else:
            self.n = max_node_id + 1 # Assume node IDs start from 0
            self.m = len(edge_pairs_temp)

        # 4. Initialize graph structure using calculated n and m
        self.nodes = [Node() for _ in range(self.n)]
        self.edges = [Edge() for _ in range(2 * self.m)] # Each edge corresponds to two Edge objects
        self.processing_queue = [0] * self.n # Assume q is related to the number of nodes
        # --- Modification ends ---

        edge_c = 0 # Counter for self.edges

        # 5. Process temporarily stored edges
        for a, b in edge_pairs_temp:
            self.edges[edge_c].node_id = b
            self.edges[edge_c].sc = 1      # Default sc value
            self.edges[edge_c].deleted = 0
            self.edges[edge_c].duplicate = self.edges[edge_c + 1]
            self.add_edge(self.nodes[a], self.edges[edge_c])
            edge_c += 1

            # Create the second direction of the edge
            self.edges[edge_c].node_id = a
            self.edges[edge_c].sc = 1      # Default sc value
            self.edges[edge_c].deleted = 0
            self.edges[edge_c].duplicate = self.edges[edge_c - 1]
            self.add_edge(self.nodes[b], self.edges[edge_c])
            edge_c += 1

        # Initialize pnodes, pedges, elements
        self.pnodes = [Node() for _ in range(self.n)]
        self.pedges = [Edge() for _ in range(2 * self.m)] # Use calculated m

        self.elements = [Element() for _ in range(self.n)]
        for i in range(self.n):
            self.elements[i].value = i
        
        return True 
    
    def remove_inter_edges(self, cc_list, assign_sc):
        # Assign component id to each node in self.computed
        for j, component_head_element in enumerate(cc_list):
            e = component_head_element
            while e is not None:
                self.computed[e.value] = j + 1 # Mark node with its component id
                e = e.next

        q_len_for_kcore = 0 # Length of the queue for k-core decomposition

        # Iterate through each component and its nodes
        for j, component_head_element in enumerate(cc_list):
            e = component_head_element
            while e is not None:
                s_node_idx = e.value # Current source node index

                s_node_obj = self.nodes[s_node_idx] # Current source node object
                original_adj_list_head = s_node_obj.first # Head of the original adjacency list
                original_adj_list_last = s_node_obj.last  # Tail of the original adjacency list

                # Detach the inter-component edges from the original list
                if original_adj_list_last is not None:
                    deleted_inter_edges_head = original_adj_list_last.next
                else:
                    deleted_inter_edges_head = None

                s_node_obj.first = None # Reset intra-component adjacency list head
                s_node_obj.last = None  # Reset intra-component adjacency list tail

                intra_component_degree = 0 # Degree of the node within its component

                current_original_edge = original_adj_list_head
                # Iterate through the original adjacency list of s_node_idx
                while current_original_edge is not None and not current_original_edge.deleted:
                    next_original_edge = current_original_edge.next # Save next edge before modifying current

                    # Check if the target node is in the same component
                    if self.computed[current_original_edge.node_id] == self.computed[s_node_idx]:
                        # This is an intra-component edge, keep it in the node's main adjacency list
                        if s_node_obj.first is None:
                            s_node_obj.first = current_original_edge
                            s_node_obj.last = current_original_edge
                            current_original_edge.pre = None
                        else:
                            s_node_obj.last.next = current_original_edge
                            current_original_edge.pre = s_node_obj.last
                            s_node_obj.last = current_original_edge

                        intra_component_degree += 1
                    else:
                        # This is an inter-component edge, mark as deleted and move to a temporary list
                        current_original_edge.deleted = 1
                        if assign_sc:
                            current_original_edge.sc = self.k_connectivity_parameter - 1 # Assign support count if specified

                        # Add to the head of the deleted_inter_edges_list
                        current_original_edge.next = deleted_inter_edges_head
                        if deleted_inter_edges_head is not None:
                            deleted_inter_edges_head.pre = current_original_edge
                        deleted_inter_edges_head = current_original_edge
                        current_original_edge.pre = None # New head has no predecessor in this list

                    current_original_edge = next_original_edge

                self.degrees[s_node_idx] = intra_component_degree # Update degree to intra-component degree
                # If degree is less than K, add to queue for k-core processing
                if intra_component_degree < self.k_connectivity_parameter:
                    self.processing_queue[q_len_for_kcore] = s_node_idx
                    q_len_for_kcore += 1

                # Append the (now separated) inter-component edges back to the end of the node's adjacency list
                # These edges are marked as deleted but might be needed for other operations or reference
                if s_node_obj.first is None: # If no intra-component edges
                    s_node_obj.first = deleted_inter_edges_head
                    s_node_obj.last = deleted_inter_edges_head # This needs to find the actual last
                    if deleted_inter_edges_head is not None:
                        # Find the tail of the deleted_inter_edges_head list to set s_node_obj.last correctly
                        temp_tail = deleted_inter_edges_head
                        while temp_tail is not None and temp_tail.next is not None:
                            temp_tail = temp_tail.next
                        s_node_obj.last = temp_tail
                        deleted_inter_edges_head.pre = None # Head of list has no pre
                else: # If there are intra-component edges
                    s_node_obj.last.next = deleted_inter_edges_head
                    if deleted_inter_edges_head is not None:
                        deleted_inter_edges_head.pre = s_node_obj.last
                        # Find the tail of the deleted_inter_edges_head list to set s_node_obj.last correctly
                        temp_tail = deleted_inter_edges_head
                        while temp_tail is not None and temp_tail.next is not None:
                            temp_tail = temp_tail.next
                        if temp_tail is not None: # If deleted_inter_edges_head was not empty
                             s_node_obj.last = temp_tail

                e = e.next

        # Reset computed array for future use
        for component_head_element in cc_list:
            e = component_head_element
            while e is not None:
                self.computed[e.value] = 0 # Reset computed flag
                e = e.next

        self.kcore_optimization(q_len_for_kcore, assign_sc) # Perform k-core decomposition
    
    def k_steiner_connectivity_bottom_up(self, _K_val, id1_list, n_id1_len, id2_list): # k steiner connectivity Bottom-Up
        self.k_connectivity_parameter = _K_val # Set the current K value

        # Initialize auxiliary arrays if they don't exist
        if self.computed is None: self.computed = [0] * self.n
        if self.height is None: self.height = [0] * self.n
        if self.degrees is None: self.degrees = [0] * self.n

        # Reset computed and height for nodes in id1_list
        for i in range(n_id1_len):
            self.computed[id1_list[i]] = 0
            self.height[id1_list[i]] = 0

        q_core_len = 0 # Length of the queue for k-core decomposition
        # Calculate initial degrees and identify nodes for k-core removal
        for i in range(n_id1_len):
            node_idx = id1_list[i]

            degree_count = 0
            edge = self.nodes[node_idx].first
            # Count valid (not deleted) edges for the current node
            while edge is not None and not edge.deleted:
                degree_count += 1
                edge = edge.next

            self.degrees[node_idx] = degree_count
            # If degree is less than K, add to k-core removal queue
            if degree_count < self.k_connectivity_parameter:
                self.processing_queue[q_core_len] = node_idx
                q_core_len += 1

        self.kcore_optimization(q_core_len, 1) # Perform k-core decomposition, assign_sc=1

        max_l_val = 0 # Maximum l value found
        non_trivial_component_found = 0 # Flag for non-trivial components
        n_id2_len_current = 0 # Current length of id2_list (nodes passing the k-SC check)

        current_id1_list_idx = 0
        # Process nodes from id1_list
        while current_id1_list_idx < n_id1_len:
            node_to_process = id1_list[current_id1_list_idx]

            # Skip if already processed or removed by k-core
            if self.computed[node_to_process]:
                current_id1_list_idx += 1
                continue

            # Construct a partial graph (P-graph) and check if it's non-trivial
            if self.construct_pgraph(node_to_process, self.height) > 1: # If P-graph has more than 1 node
                non_trivial_component_found = 1

            cc_elements_list = [] # List to store connected components (as lists of elements)
            # Decompose the P-graph into support-connected components
            max_l_val = self.decomposition(node_to_process, cc_elements_list, max_l_val)

            if len(cc_elements_list) == 1: # If only one component results from decomposition
                element_item = cc_elements_list[0]
                # Add all nodes in this single component to id2_list
                while element_item is not None:
                    val = element_item.value
                    self.computed[val] = 1 # Mark as processed for this kSC iteration
                    id2_list[n_id2_len_current] = val
                    n_id2_len_current += 1
                    element_item = element_item.next
                current_id1_list_idx += 1 # Move to the next node in id1_list
            else:
                self.remove_inter_edges(cc_elements_list, 1) # assign_sc=1

        return non_trivial_component_found, n_id2_len_current

    def find_all_steiner_connectivity_bottom_up(self): # All steiner connectivity Bottom-Up
        id1_arr = [0] * self.n # Array for input nodes for kSC_BU
        id2_arr = [0] * self.n # Buffer for results from kSC_BU

        # Initialize id1_arr with all nodes
        for i in range(self.n):
            id1_arr[i] = i
        n_id1_current_len = self.n # Current number of nodes in id1_arr

        # Iterate K from 2 up to n
        for k_val_to_check in range(2, self.n + 1):
            # Call method to find nodes belonging to k-steiner_connectivity
            non_trivial, n_id2_result_len = self.k_steiner_connectivity_bottom_up(k_val_to_check, id1_arr, n_id1_current_len, id2_arr)

            if not non_trivial: # If no non-trivial components were found for this K
                print(f"Max K: {k_val_to_check}") # This k_val_to_check might be the K for which no SC exists or the one after the max
                break # Stop if no more non-trivial components

            n_id1_current_len = n_id2_result_len # Update length for the next iteration
            id1_arr, id2_arr = id2_arr, id1_arr # Swap lists: output of current becomes input for next

    def decomposition(self, ss_start_node, cc_list_ref, max_l_initial_val): # Decompose P-graph into support-connected components
        # Initialize heap and inL array if not already done
        if self.heap is None:
            self.heap = LinearHeap(self.n)

        if self.inL is None: # Tracks if a node is in the current component L being built
            self.inL = [0] * self.n

        cc_list_ref.clear() # Clear the list that will store component heads

        count_components_processed = 0 # Counts how many components are extracted from the P-graph
        current_max_l = max_l_initial_val # Tracks the maximum number of components found for any P-graph

        # Process while the starting node of the P-graph still has edges (i.e., P-graph is not fully decomposed)
        while self.pnodes[ss_start_node].first is not None:
            count_components_processed += 1

            # Insert the start node of the current component search into the heap with key 0
            self.heap.insert(ss_start_node, 0)

            q_component_nodes_idx = 0 # Index for self.processing_queue, storing nodes of the current component L

            while True: # Loop to extract nodes and build a component L
                heap_extraction_result = self.heap.extract_max() # Get node with max key from heap
                if heap_extraction_result is None: # Heap is empty, component L formation ends
                    break

                s_pivot, _ = heap_extraction_result # s_pivot is the node with max key, _ is its key

                self.inL[s_pivot] = 1 # Mark s_pivot as part of component L

                self.processing_queue[q_component_nodes_idx] = s_pivot # Add s_pivot to the component's node list
                q_component_nodes_idx += 1

                q_expansion_frontier_idx = q_component_nodes_idx # Frontier for BFS-like expansion within L

                current_expansion_node_ptr = q_component_nodes_idx - 1 # Start expansion from s_pivot

                # BFS-like expansion to find other nodes for L based on s_pivot
                while current_expansion_node_ptr < q_expansion_frontier_idx:
                    u_node = self.processing_queue[current_expansion_node_ptr] # Current node for expansion

                    edge_iterator = self.pnodes[u_node].first # Iterate neighbors of u_node in P-graph
                    while edge_iterator is not None:
                        v_neighbor = edge_iterator.node_id # Neighbor node
                        if not self.inL[v_neighbor]: # If neighbor is not yet in L
                            v_neighbor_current_key = self.heap.get_key(v_neighbor) # Current key in heap

                            if v_neighbor_current_key < self.k_connectivity_parameter: # If key < K, it might be added/updated
                                if v_neighbor_current_key > 0: # If already in heap with a positive key
                                    self.heap.remove(v_neighbor) # Remove to update its key

                                v_neighbor_tentative_key = v_neighbor_current_key + edge_iterator.sc # New potential key

                                if v_neighbor_tentative_key >= self.k_connectivity_parameter: # If new key >= K
                                    self.heap.set_key(v_neighbor, v_neighbor_tentative_key) # Update key
                                    self.processing_queue[q_expansion_frontier_idx] = v_neighbor # Add to expansion frontier
                                    q_expansion_frontier_idx += 1
                                else: # If new key < K
                                    self.heap.insert(v_neighbor, v_neighbor_tentative_key) # Insert with new key
                            else: # If key >= K, just increment its key
                                self.heap.set_key(v_neighbor, v_neighbor_current_key + edge_iterator.sc)

                        edge_iterator = edge_iterator.next

                    if u_node == s_pivot: # If u_node is the pivot itself, continue to next in expansion queue
                        current_expansion_node_ptr += 1
                        continue

                    # If u_node is not s_pivot, merge u_node into s_pivot
                    self.heap.set_key(s_pivot, self.heap.get_key(s_pivot) + self.heap.get_key(u_node)) # Add u's key to s_pivot
                    self.heap.set_key(u_node, 0) # Reset u's key
                    self.inL[u_node] = 0 # Mark u_node as no longer in L (it's merged)
                    self.merge(s_pivot, u_node, self.heap) # Merge P-graph node u_node into s_pivot

                    current_expansion_node_ptr += 1

            # Post-processing after a component L is formed (or fails to form fully)
            idx_for_cleanup = q_component_nodes_idx - 1 # Iterate backwards through nodes added to self.processing_queue

            # Finalize components that are "weak" (key < K)
            while idx_for_cleanup > 0: # Skip s_pivot (index 0) for now
                if self.heap.get_key(self.processing_queue[idx_for_cleanup]) < self.k_connectivity_parameter: # If node's key < K
                    t_node_to_finalize = self.processing_queue[idx_for_cleanup]

                    cc_list_ref.append(self.pnodes[t_node_to_finalize].head) # Add this as a finalized component

                    self.heap.set_key(t_node_to_finalize, 0) # Reset key
                    self.inL[t_node_to_finalize] = 0 # Reset inL status

                    # Remove edges of this finalized component from the P-graph
                    edge_to_remove_dup = self.pnodes[t_node_to_finalize].first
                    while edge_to_remove_dup is not None:
                        self.delete_edge(self.pnodes[edge_to_remove_dup.node_id], edge_to_remove_dup.duplicate)
                        edge_to_remove_dup = edge_to_remove_dup.next
                    self.pnodes[t_node_to_finalize].first = None # Clear adjacency list of finalized node

                    idx_for_cleanup -=1
                else: # Node's key >= K, part of the strong component around s_pivot
                    break

            # Reset heap keys and inL status for remaining nodes in self.processing_queue
            for i in range(idx_for_cleanup + 1):
                if idx_for_cleanup < 0: # Should not happen if q_component_nodes_idx was > 0
                    break
                node_to_reset_in_q = self.processing_queue[i]
                self.heap.set_key(node_to_reset_in_q, 0)
                self.inL[node_to_reset_in_q] = 0

        # Update max_l if current P-graph decomposition yielded more components
        if count_components_processed > current_max_l:
            current_max_l = count_components_processed

        # The remaining P-graph centered at ss_start_node is the last component
        cc_list_ref.append(self.pnodes[ss_start_node].head)

        return current_max_l

    def merge(self, s_idx, t_idx, heap_obj): # Merge P-graph node t_idx into P-graph node s_idx
        # Append t's element list to s's element list
        self.pnodes[s_idx].tail.next = self.pnodes[t_idx].head
        self.pnodes[s_idx].tail = self.pnodes[t_idx].tail

        current_edge = self.pnodes[t_idx].first # Iterate through edges of t_idx

        while current_edge is not None:
            next_edge_in_t_list = current_edge.next # Save next edge

            if current_edge.node_id == s_idx: # If edge connects t_idx to s_idx
                if heap_obj is not None: # Adjust s_idx's key in heap (remove contribution of t-s edge)
                    current_s_key = heap_obj.get_key(s_idx)
                    heap_obj.set_key(s_idx, current_s_key - current_edge.sc)
                # Delete the duplicate edge from s_idx's adjacency list
                self.delete_edge(self.pnodes[s_idx], current_edge.duplicate)
            else: # Edge connects t_idx to some other node v_neighbor
                # Update the duplicate edge (from v_neighbor to t_idx) to now point to s_idx
                current_edge.duplicate.node_id = s_idx
                # Add the current edge (from t_idx to v_neighbor, now effectively s_idx to v_neighbor) to s_idx's list
                self.add_edge(self.pnodes[s_idx], current_edge)

            current_edge = next_edge_in_t_list

        self.pnodes[t_idx].first = None # Clear t_idx's adjacency list as it's now merged
    
    def delete_edge(self, node_obj, edge_obj): # Deletes an edge object from a node's adjacency list
        if edge_obj.pre is None: # Edge is the first in the list
            node_obj.first = edge_obj.next
            if edge_obj.next is not None:
                edge_obj.next.pre = None
        else: # Edge is not the first
            if edge_obj is node_obj.last: # Edge is the last in the list
                node_obj.last = edge_obj.pre

            # Bypass the edge by linking its predecessor and successor
            edge_obj.pre.next = edge_obj.next
            if edge_obj.next is not None:
                edge_obj.next.pre = edge_obj.pre

    def delete_edge_to_last(self, node_obj, edge_obj): # Marks an edge as deleted and moves it to the end of the adjacency list
        edge_obj.deleted = 1 # Mark the edge as deleted
        if node_obj.first is node_obj.last: # If only one edge, no re-linking needed beyond marking
            return

        # Standard removal from its current position
        if edge_obj.pre is None: # If it's the first edge
            node_obj.first = edge_obj.next
            if edge_obj.next is not None:
                edge_obj.next.pre = None
        else: # If it's not the first edge
            if edge_obj is node_obj.last: # If it's the current last edge
                node_obj.last = edge_obj.pre

            tmp = edge_obj.pre
            tmp.next = edge_obj.next # Bypass edge_obj

            if edge_obj.next is not None:
                edge_obj.next.pre = tmp

        edge_obj.next = node_obj.last.next 
        if edge_obj.next is not None:
            edge_obj.next.pre = edge_obj
        node_obj.last.next = edge_obj # Current last points to edge_obj
        edge_obj.pre = node_obj.last # edge_obj's pre becomes the original last

    def kcore_optimization(self, q_c_initial_count, assign_sc): # Performs k-core decomposition
        current_q_idx = 0 # Current index for processing the queue
        q_dynamic_len = q_c_initial_count # Dynamic length of the queue

        # Process nodes with degree less than K
        while current_q_idx < q_dynamic_len:
            s = self.processing_queue[current_q_idx] # Get node from queue
            self.computed[s] = 1 # Mark node as processed/removed by k-core

            edge = self.nodes[s].first # Iterate through neighbors of s
            while edge is not None and not edge.deleted:
                t = edge.node_id # Neighbor node

                # For the neighbor t, "remove" the duplicate edge (s,t) from t's list
                self.delete_edge_to_last(self.nodes[t], edge.duplicate)

                self.degrees[t] -= 1 # Decrement degree of t

                # If t's degree drops to K-1, add t to the queue
                if self.degrees[t] == self.k_connectivity_parameter - 1:
                    self.processing_queue[q_dynamic_len] = t
                    q_dynamic_len += 1

                edge.deleted = 1 # Mark edge (s,t) as deleted

                if assign_sc: # If support count needs to be assigned
                    edge.sc = self.k_connectivity_parameter - 1
                    edge.duplicate.sc = self.k_connectivity_parameter - 1

                edge = edge.next

            self.nodes[s].last = self.nodes[s].first
            current_q_idx += 1    

    def construct_pgraph(self, s_start_node, height_list): # Constructs a partial graph (P-graph) via BFS
        pedge_c = 0 # Counter for P-graph edges
        q_len = 1   # Length of the BFS queue, initially with the start node

        self.computed[s_start_node] = 1 # Mark start node as visited for this BFS
        self.processing_queue[0] = s_start_node       # Add start node to BFS queue

        i = 0 # BFS queue pointer
        while i < q_len:
            s_current = self.processing_queue[i] # Current node being processed from BFS queue
            height_list[s_current] += 1 # Increment height (or visit count) for this node

            # Initialize P-graph node's element list
            self.pnodes[s_current].head = self.elements[s_current]
            self.pnodes[s_current].tail = self.elements[s_current]
            if self.elements[s_current] is not None: # Ensure element exists
                 self.elements[s_current].next = None

            edge = self.nodes[s_current].first # Iterate original graph edges of s_current
            while edge is not None and not edge.deleted:
                neighbor_node_id = edge.node_id
                if not self.computed[neighbor_node_id]: # If neighbor not visited in this BFS
                    self.computed[neighbor_node_id] = 1 # Mark as visited
                    self.processing_queue[q_len] = neighbor_node_id    # Add to BFS queue
                    q_len += 1

                # Add edge to P-graph if neighbor_node_id > s_current to avoid duplicates
                # Assumes P-graph edges are undirected but stored once
                if neighbor_node_id > s_current:
                    a = s_current
                    b = neighbor_node_id

                    # Add edge a-b to P-graph
                    self.pedges[pedge_c].node_id = b
                    self.pedges[pedge_c].sc = 1 # Default P-graph edge support count
                    self.pedges[pedge_c].deleted = 0
                    self.pedges[pedge_c].duplicate = self.pedges[pedge_c+1]
                    self.add_edge(self.pnodes[a], self.pedges[pedge_c])
                    pedge_c += 1

                    # Add edge b-a to P-graph
                    self.pedges[pedge_c].node_id = a
                    self.pedges[pedge_c].sc = 1
                    self.pedges[pedge_c].deleted = 0
                    self.pedges[pedge_c].duplicate = self.pedges[pedge_c-1]
                    self.add_edge(self.pnodes[b], self.pedges[pedge_c])
                    pedge_c += 1

                edge = edge.next
            i += 1

        # Reset computed flags for nodes processed in this P-graph construction
        for j in range(q_len):
            self.computed[self.processing_queue[j]] = 0

        return q_len # Return the number of nodes in the constructed P-graph

    def find_root(self, x, parent_list): # Finds the root of x in a disjoint set union (DSU) structure
        root = x
        # Traverse up to find the absolute root
        while parent_list[root] != root:
            root = parent_list[root]

        # Path compression: make all nodes on the path point directly to the root
        curr = x
        while parent_list[curr] != root:
            next_node = parent_list[curr]
            parent_list[curr] = root
            curr = next_node

        return root

    def max_spanning_tree(self): # Computes a maximum spanning tree/forest using Kruskal's algorithm variant
        # Initialize P-graph node adjacency lists (used here as buckets for edges by weight)
        for i in range(self.n):
            self.pnodes[i].first = None
            self.pnodes[i].last = None

        edge_c_for_pedges_phase1 = 0 # Counter for edges placed in buckets
        max_sc_val = 0 # Maximum support count (edge weight) found

        # Bucket sort edges by their support count (sc)
        for i in range(self.n):
            original_graph_edge = self.nodes[i].first
            while original_graph_edge is not None:
                if original_graph_edge.node_id > i: # Process each edge once
                    if original_graph_edge.sc > max_sc_val:
                        max_sc_val = original_graph_edge.sc

                    bucket_index = original_graph_edge.sc # sc is used as bucket index

                    # Store edge (i, original_graph_edge.node_id) in the bucket
                    # Using pedges as temporary storage for edge u, v and pnodes[bucket] as head of list
                    current_pedge_entry = self.pedges[edge_c_for_pedges_phase1]
                    current_pedge_entry.node_id = i # Store u
                    current_pedge_entry.sc = original_graph_edge.node_id # Store v in sc field temporarily
                    current_pedge_entry.next = None

                    # Add to the end of the bucket list
                    if self.pnodes[bucket_index].first is None:
                        self.pnodes[bucket_index].first = current_pedge_entry
                        self.pnodes[bucket_index].last = current_pedge_entry
                    else:
                        self.pnodes[bucket_index].last.next = current_pedge_entry
                        self.pnodes[bucket_index].last = current_pedge_entry

                    edge_c_for_pedges_phase1 += 1
                original_graph_edge = original_graph_edge.next

        # Initialize DSU structure
        parent_arr = [k for k in range(self.n)] # Each node is its own parent initially
        rank_arr = [0] * self.n # Rank for union by rank/size heuristic
        mst_edges_list = [] # List to store edges of the MST

        # Kruskal's: Iterate buckets from highest sc to lowest
        for current_sc_value in range(max_sc_val, 0, -1):
            pedge_in_bucket = self.pnodes[current_sc_value].first # Get edges with this sc
            while pedge_in_bucket is not None:
                u1 = pedge_in_bucket.node_id # Retrieve u
                v1 = pedge_in_bucket.sc      # Retrieve v (stored in sc field)

                root_u1 = self.find_root(u1, parent_arr)
                root_v1 = self.find_root(v1, parent_arr)

                if root_u1 != root_v1: # If u1 and v1 are in different sets, add edge to MST
                    mst_edges_list.append( ((u1, v1), current_sc_value) )

                    # Union by rank
                    if rank_arr[root_u1] < rank_arr[root_v1]:
                        parent_arr[root_u1] = root_v1
                    elif rank_arr[root_u1] > rank_arr[root_v1]:
                        parent_arr[root_v1] = root_u1
                    else:
                        parent_arr[root_u1] = root_v1 # Arbitrarily make root_v1 the new root
                        rank_arr[root_v1] += 1

                pedge_in_bucket = pedge_in_bucket.next

        # Clear pnodes (used as buckets) and rebuild adjacencies for the MST
        for i in range(self.n):
            self.pnodes[i].first = None
            self.pnodes[i].last = None # Though last is not strictly used for this new list

        edge_c_for_pedges_phase2 = 0 # Counter for pedges used for MST adjacency list
        for ((u, v), _) in mst_edges_list: # Rebuild MST graph structure in pnodes
            # Add edge u-v to u's list
            pedge_for_u_list = self.pedges[edge_c_for_pedges_phase2]
            pedge_for_u_list.node_id = v
            pedge_for_u_list.next = self.pnodes[u].first # Insert at head
            self.pnodes[u].first = pedge_for_u_list
            edge_c_for_pedges_phase2 += 1

            # Add edge v-u to v's list
            pedge_for_v_list = self.pedges[edge_c_for_pedges_phase2]
            pedge_for_v_list.node_id = u
            pedge_for_v_list.next = self.pnodes[v].first # Insert at head
            self.pnodes[v].first = pedge_for_v_list
            edge_c_for_pedges_phase2 += 1

        # Compute component IDs and levels in the MST/forest using BFS
        component_ids = [0] * self.n
        levels = [0] * self.n

        if self.computed is None: # Ensure 'computed' array (for visited status) exists
            self.computed = [0] * self.n
        else: # Reset if already exists
            for k in range(self.n):
                self.computed[k] = 0

        current_component_id_counter = 0
        for i in range(self.n): # For each potential start of a new component (forest)
            if self.computed[i]: # Skip if already visited
                continue

            q_bfs_len = 1 # BFS queue length
            self.processing_queue[0] = i # Start BFS from node i
            levels[i] = 0
            component_ids[i] = current_component_id_counter
            self.computed[i] = 1

            bfs_ptr = 0 # BFS queue pointer
            while bfs_ptr < q_bfs_len:
                current_bfs_node = self.processing_queue[bfs_ptr]
                mst_adjacency_edge = self.pnodes[current_bfs_node].first # Neighbors in MST
                while mst_adjacency_edge is not None:
                    neighbor = mst_adjacency_edge.node_id
                    if not self.computed[neighbor]:
                        self.computed[neighbor] = 1
                        self.processing_queue[q_bfs_len] = neighbor
                        q_bfs_len +=1
                        levels[neighbor] = levels[current_bfs_node] + 1
                        component_ids[neighbor] = current_component_id_counter
                    mst_adjacency_edge = mst_adjacency_edge.next
                bfs_ptr += 1
            current_component_id_counter += 1 # Move to next component ID

        dir_path = getattr(self, 'dir', '.') # Safely get self.dir, default to current directory
        output_file_path = f"{dir_path}/mSPT.txt" # Max Spanning Tree output file

        with open(output_file_path, "w") as fout_mspt:
            fout_mspt.write(f"{self.n} {len(mst_edges_list)}\n") # Num nodes, num MST edges

            for i in range(self.n): # Node component IDs and levels
                fout_mspt.write(f"{component_ids[i]} {levels[i]}\n")

            for ((u, v), sc_value) in mst_edges_list: # MST edges and their original sc
                fout_mspt.write(f"{u} {v} {sc_value}\n")

    def output_all_steiner_connectivity(self, fout_obj): # Outputs all support counts to a binary file and a text file
        vp_list = [] # List to store (sc, (u,v)) tuples for sorting

        # Path for the text file output of core graph (edges with support counts)
        cg_text_file_path = os.path.join(self.dir, "cg.txt") # Core Graph text file
        text_file_output = open(cg_text_file_path, "w")

        text_file_output.write(f"{self.n} {self.m}\n") # Num nodes, num edges

        # Write n and m to the binary file object
        fout_obj.write(struct.pack('i', self.n))
        fout_obj.write(struct.pack('i', self.m))

        # Collect all unique edges with their support counts
        for i in range(self.n):
            current_edge = self.nodes[i].first
            while current_edge is not None:
                if current_edge.node_id > i: # Process each edge once (u < v)
                    vp_list.append((current_edge.sc, (i, current_edge.node_id)))
                current_edge = current_edge.next

        vp_list.sort() # Sort by support count (ascending by default)

        buffer_for_binary_file = [] # Buffer to collect data before packing to binary

        # Iterate through the sorted list in reverse (largest sc first) for output
        for i in range(len(vp_list) - 1, -1, -1):
            score_value = vp_list[i][0] # Support count
            node_u = vp_list[i][1][0]
            node_v = vp_list[i][1][1]

            # Add to binary buffer list: u, v, sc
            buffer_for_binary_file.append(node_u)
            buffer_for_binary_file.append(node_v)
            buffer_for_binary_file.append(score_value)

            # Write to the text file: u, v, sc
            text_file_output.write(f"{node_u} {node_v} {score_value}\n")

        text_file_output.close() # Close the text file

        # Pack and write the buffered data to the binary file
        if buffer_for_binary_file:
            # Create format string like '15i' if buffer has 15 integers
            format_string = f'{len(buffer_for_binary_file)}i'
            packed_binary_data = struct.pack(format_string, *buffer_for_binary_file)
            fout_obj.write(packed_binary_data)

DEFAULT_MIN_QUERY_STRUCTURE_SIZE: int = 10
MIN_EDGES_POOL_SIZE: int = 1

class ConnGraph:
    MAX_UPDATE_COUNT: int = 1000  # Renamed from MAX_UPDATE for clarity

    def __init__(self, dir_path: str = "./") -> None:
        """
        Initializes the ConnGraph instance.

        Args:
            dir_path: The directory path for loading graph data files.
        """
        self.dir: str = str(dir_path)
        self.n: int = 0  # Number of nodes
        self.m: int = 0  # Number of edges in the original graph
        self.m_mst: int = 0  # Number of edges in the Maximum Spanning Tree

        # Core graph structure (likely related to MST or a tree decomposition)
        self.tnodes: Optional[List[Node]] = None  # Tree nodes
        self.levels: Optional[List[int]] = None
        self.component_ids: Optional[List[int]] = None
        
        # Attributes related to parent pointers, weights, or specific edge sets
        # Their exact usage would require more context or method implementations.
        self.pedges: Any = None # Optional[List[Edge]] or other type

        # Structures for query processing (e.g., BFS queue, visited arrays)
        self.Q: Optional[List[int]] = None # Query queue
        self.computed: Optional[List[int]] = None # Flags for computed nodes
        self.vis: Optional[List[int]] = None      # Visited array for traversals

        # Structures specifically for Steiner queries or similar advanced queries
        self.heads_steiner_query: Optional[List[Optional[Bin]]] = None
        self.ll_steiner_nodes: Optional[List[Bin]] = None # Linked list nodes (bins)

        # Temporary or auxiliary structures, likely used in algorithms not fully shown
        self.elements: Any = None
        self.edges: Any = None # Optional[List[Edge]]
        self.nodes: Any = None # Optional[List[Node]]
        self.heap: Any = None   # Heap structure

    def add_edge(self, node_obj: Node, edge_obj: Edge) -> None:
        edge_obj.next = None
        if node_obj.first is None:
            node_obj.first = edge_obj
            node_obj.last = edge_obj
            edge_obj.pre = None
        else:
            # We know node_obj.last is not None here due to node_obj.first not being None
            node_obj.last.next = edge_obj # type: ignore
            edge_obj.pre = node_obj.last
            node_obj.last = edge_obj

    def _initialize_query_structures(self) -> bool:
        if self.n == 0:
            # print("Warning: Cannot initialize query structures with n=0 nodes.") # Optional: for debugging
            return False

        # Initialize 'computed' array
        if self.computed is None or len(self.computed) != self.n:
            self.computed = [0] * self.n
        
        # Initialize 'Q' (queue)
        q_size: int = 2 * self.n if self.n > 0 else DEFAULT_MIN_QUERY_STRUCTURE_SIZE
        if self.Q is None or len(self.Q) < q_size:
            self.Q = [0] * q_size # Assuming int for node IDs

        # Initialize 'vis' (visited) array
        if self.vis is None or len(self.vis) != self.n:
            self.vis = [0] * self.n
            
        # Initialize 'heads_steiner_query'
        head_len: int = self.n if self.n > 0 else MIN_EDGES_POOL_SIZE # Ensure at least 1
        if self.heads_steiner_query is None or len(self.heads_steiner_query) < head_len:
            self.heads_steiner_query = [None] * head_len
        
        # Initialize 'll_steiner_nodes' (linked list of Bins)
        ll_size: int = 2 * self.n if self.n > 0 else DEFAULT_MIN_QUERY_STRUCTURE_SIZE
        if self.ll_steiner_nodes is None or len(self.ll_steiner_nodes) < ll_size:
            self.ll_steiner_nodes = [Bin() for _ in range(ll_size)]
        return True

    def _reset_query_structures_before_query(self) -> None:
        if self.n == 0:
            return

        if self.computed is not None:
            # Faster way to reset to all zeros if list is large
            # self.computed = [0] * self.n
            for i in range(self.n): # Current way is also fine and clear
                self.computed[i] = 0
        if self.vis is not None:
            # self.vis = [0] * self.n
            for i in range(self.n):
                self.vis[i] = 0
        
        if self.heads_steiner_query is not None:
            # self.heads_steiner_query = [None] * len(self.heads_steiner_query)
            for i in range(len(self.heads_steiner_query)):
                self.heads_steiner_query[i] = None

    def add_edge_to_tnodes(self, node_idx_from: int, node_idx_to: int,
                             sc_val: int, edge_pool: List[Edge],
                             edge_pool_idx: int) -> int:
        if self.tnodes is None:
            return edge_pool_idx 

        edge1: Edge = edge_pool[edge_pool_idx]
        edge2: Edge = edge_pool[edge_pool_idx + 1]

        # Configure edge from node_idx_from to node_idx_to
        edge1.node_id = node_idx_to
        edge1.sc = sc_val
        edge1.deleted = 0
        edge1.duplicate = edge2
        self.add_edge(self.tnodes[node_idx_from], edge1)

        # Configure corresponding edge from node_idx_to to node_idx_from
        edge2.node_id = node_idx_from
        edge2.sc = sc_val
        edge2.deleted = 0
        edge2.duplicate = edge1
        self.add_edge(self.tnodes[node_idx_to], edge2)
        
        return edge_pool_idx + 2

    def load_data(self, cg_bin_path: str = "cg.bin",
                  mspt_txt_path: str = "mSPT.txt") -> bool:
        actual_cg_bin_path = os.path.join(self.dir, cg_bin_path)
        try:
            with open(actual_cg_bin_path, "rb") as f_cg_bin:
                data_n_bytes = f_cg_bin.read(INT_SIZE)
                if not data_n_bytes or len(data_n_bytes) < INT_SIZE:
                    # print(f"Error: Could not read 'n' from {actual_cg_bin_path}") # Optional logging
                    self.n = 0
                    self.m = 0
                    return False
                self.n = struct.unpack('i', data_n_bytes)[0]

                data_m_bytes = f_cg_bin.read(INT_SIZE)
                if not data_m_bytes or len(data_m_bytes) < INT_SIZE:
                    # print(f"Error: Could not read 'm' from {actual_cg_bin_path}") # Optional logging
                    self.m = 0
                    return False
                self.m = struct.unpack('i', data_m_bytes)[0]

        except FileNotFoundError:
            # print(f"Error: File not found {actual_cg_bin_path}") # Optional logging
            self.n, self.m = 0, 0
            return False
        except (struct.error, IOError) as e:
            # print(f"Error reading or parsing {actual_cg_bin_path}: {e}") # Optional logging
            self.n, self.m = 0, 0
            return False

        if self.n <= 0:
            # print(f"Warning: Non-positive number of nodes ({self.n}) loaded.") # Optional logging
            return False # Or handle as an error depending on requirements
            
        actual_mspt_txt_path = os.path.join(self.dir, mspt_txt_path)
        try:
            with open(actual_mspt_txt_path, "r") as fin:
                first_line = fin.readline()
                if not first_line:
                    # print(f"Error: {actual_mspt_txt_path} is empty or first line missing.") # Optional logging
                    return False
                
                line1_parts = first_line.split()
                if len(line1_parts) < 2:
                    # print(f"Error: Invalid format in first line of {actual_mspt_txt_path}") # Optional logging
                    return False
                
                n_from_mst_file = int(line1_parts[0])
                self.m_mst = int(line1_parts[1])

                if n_from_mst_file != self.n:
                    print(f"Warning: Node count mismatch. cg.bin: {self.n}, "
                          f"mSPT.txt: {n_from_mst_file}. Using n from cg.bin.")
                
                self.levels = [0] * self.n
                self.component_ids = [0] * self.n 
                for i in range(self.n):
                    node_data_line = fin.readline()
                    if not node_data_line:
                        # print(f"Error: Unexpected end of file in {actual_mspt_txt_path} while reading node data.") # Optional logging
                        return False
                    line_node_data = node_data_line.split()
                    if len(line_node_data) < 2:
                        # print(f"Error: Invalid node data format in {actual_mspt_txt_path} at line {i+2}") # Optional logging
                        return False
                    self.component_ids[i] = int(line_node_data[0])
                    self.levels[i] = int(line_node_data[1])

                self.tnodes = [Node() for _ in range(self.n)]

                # Pool for edges in the MST (tnodes)
                tedges_pool_size = 2 * self.m_mst if self.m_mst > 0 else MIN_EDGES_POOL_SIZE
                tedges_pool: List[Edge] = [Edge() for _ in range(tedges_pool_size)]
                
                current_edge_pool_idx = 0
                for _ in range(self.m_mst):
                    edge_data_line = fin.readline()
                    if not edge_data_line:
                        # print(f"Error: Unexpected end of file in {actual_mspt_txt_path} while reading MST edges.") # Optional logging
                        return False
                    line_edge_data = edge_data_line.split()
                    if len(line_edge_data) < 3:
                        # print(f"Error: Invalid MST edge data format in {actual_mspt_txt_path}") # Optional logging
                        return False
                        
                    node_a = int(line_edge_data[0])
                    node_b = int(line_edge_data[1])
                    weight_c = int(line_edge_data[2]) # Assuming this is sc_val for MST edges
                    
                    if 0 <= node_a < self.n and 0 <= node_b < self.n:
                        # Ensure there's space for two edges in the pool
                        if current_edge_pool_idx + 1 < tedges_pool_size:
                            current_edge_pool_idx = self.add_edge_to_tnodes(
                                node_a, node_b, weight_c, tedges_pool, current_edge_pool_idx
                            )
                        else:
                            # print(f"Warning: Not enough space in _tedges_pool. Added {current_edge_pool_idx // 2} / {self.m_mst} MST edges.") # Optional logging
                            break 
                    else:
                        pass # Current behavior: skip this edge
        except FileNotFoundError:
            # print(f"Error: File not found {actual_mspt_txt_path}") # Optional logging
            return False
        except (ValueError, IOError) as e: # Catch parsing errors (int()) and other IO issues
            # print(f"Error processing file {actual_mspt_txt_path}: {e}") # Optional logging
            return False

        if not self._initialize_query_structures():
            # print("Warning: Failed to initialize query structures after loading data.") # Optional logging
            # Depending on requirements, this might be a critical failure.
            return False # Or True if partial success is acceptable
        return True

    def _clamp_key(self, key: int, max_len_exclusive: int) -> int:
        if max_len_exclusive <= 0: # Should not happen with heads_steiner_query
            return 0
        clamped = max(0, key)
        clamped = min(clamped, max_len_exclusive - 1)
        return clamped

    def query_smcc(self, query_nodes_list: List[int]) -> Tuple[List[int], int]:
        if not query_nodes_list:
            return [], 0 

        if self.n == 0 or self.tnodes is None or \
           self.computed is None or self.vis is None or \
           self.Q is None or self.heads_steiner_query is None or \
           self.ll_steiner_nodes is None:
            # print("Error: Query structures not properly initialized for query_smcc.") # Optional logging
            return [], 0

        self._reset_query_structures_before_query()
        
        valid_query_nodes: List[int] = []
        for node_idx in query_nodes_list:
            if 0 <= node_idx < self.n:
                self.computed[node_idx] = 1 # Mark as a target query node
                valid_query_nodes.append(node_idx)
            else:
                # print(f"Warning: Invalid node_idx {node_idx} in query_nodes_list skipped.") # Optional logging
                pass # Skip invalid nodes
        
        if not valid_query_nodes:
            return [], 0
        
        num_target_nodes: int = len(valid_query_nodes)
        first_query_node: int = valid_query_nodes[0]

        # Initial checks for the first query node's edges
        if not (0 <= first_query_node < len(self.tnodes)) or self.tnodes[first_query_node].first is None:
            # Cleanup computed flags for this query if it cannot proceed
            for node_idx_comp_clear in valid_query_nodes:
                if 0 <= node_idx_comp_clear < self.n: # Redundant if valid_query_nodes only has valid ones
                    self.computed[node_idx_comp_clear] = 0
            return [], 0

        # Initialize max_key_local with the score of the first edge of the first query node
        # This key seems to represent the current connectivity level being explored.
        max_key_local: int = self.tnodes[first_query_node].first.sc # type: ignore
        max_key_local = self._clamp_key(max_key_local, len(self.heads_steiner_query))

        # ll_steiner_nodes acts as a pool for Bin objects to build linked lists in heads_steiner_query
        edge_count_for_ll_pool: int = 0
        
        # Add first node's edges to the appropriate bin in heads_steiner_query
        if edge_count_for_ll_pool < len(self.ll_steiner_nodes):
            current_bin_obj: Bin = self.ll_steiner_nodes[edge_count_for_ll_pool]
            edge_count_for_ll_pool += 1

            current_bin_obj.edge = self.tnodes[first_query_node].first
            current_bin_obj.next = self.heads_steiner_query[max_key_local]
            self.heads_steiner_query[max_key_local] = current_bin_obj
        else:
            # print("Warning: ll_steiner_nodes pool exhausted at initialization.") # Optional logging
            # This might indicate an issue or insufficient pool size.
            pass # Continue, but some edges might not be processed.
        
        # BFS/Search initialization
        self.Q[0] = first_query_node
        self.vis[first_query_node] = 1 
        bfs_queue_idx: int = 1 

        # Track found target nodes and current connectivity
        # Connectivity level is initialized based on the first node if it's the only target
        count_target_nodes_found: int = 1 # Since computed[first_query_node] was set
        
        connectivity_level: int
        if num_target_nodes == 1: # Or more robustly: count_target_nodes_found == num_target_nodes
            connectivity_level = max_key_local
        else:
            connectivity_level = 0 # Or a very small number if scores can be negative
        
        # Main loop: Iteratively process edges from bins, starting from max_key_local downwards
        while max_key_local >= connectivity_level:
            # Find the highest score bin that is not empty
            while max_key_local >= connectivity_level and \
                  (max_key_local >= len(self.heads_steiner_query) or 
                   self.heads_steiner_query[max_key_local] is None):
                max_key_local -= 1
            
            if max_key_local < connectivity_level:
                break # All relevant bins processed or connectivity found

            # Pop the first Bin from the current active bin list
            current_active_bin: Bin = self.heads_steiner_query[max_key_local] # type: ignore
            edge_iterator: Optional[Edge] = current_active_bin.edge
            self.heads_steiner_query[max_key_local] = current_active_bin.next 

            # Iterate through edges in the popped Bin
            while edge_iterator is not None and edge_iterator.sc >= max_key_local:
                neighbor_node_id: int = edge_iterator.node_id
                if not self.vis[neighbor_node_id]: 
                    self.vis[neighbor_node_id] = 1
                    self.Q[bfs_queue_idx] = neighbor_node_id
                    bfs_queue_idx += 1
                    
                    # If ll_steiner_nodes pool is exhausted, cannot add more edges from neighbor
                    if edge_count_for_ll_pool >= len(self.ll_steiner_nodes):
                        # print("Warning: ll_steiner_nodes pool exhausted during neighbor processing.") # Optional logging
                        edge_iterator = edge_iterator.next 
                        continue 

                    # If neighbor has edges, add them to appropriate bins
                    if 0 <= neighbor_node_id < len(self.tnodes) and \
                       self.tnodes[neighbor_node_id].first is not None:
                        
                        new_bin_for_neighbor: Bin = self.ll_steiner_nodes[edge_count_for_ll_pool]
                        new_bin_for_neighbor.edge = self.tnodes[neighbor_node_id].first
                        
                        # Key for the new bin is the score of its first edge,
                        # capped by current max_key_local.
                        key_for_new_bin: int = new_bin_for_neighbor.edge.sc # type: ignore
                        key_for_new_bin = min(key_for_new_bin, max_key_local)
                        key_for_new_bin = self._clamp_key(key_for_new_bin, len(self.heads_steiner_query))
                            
                        new_bin_for_neighbor.next = self.heads_steiner_query[key_for_new_bin]
                        self.heads_steiner_query[key_for_new_bin] = new_bin_for_neighbor
                        edge_count_for_ll_pool += 1 # Consumed one Bin from pool
                    
                    # Check if this newly visited neighbor is one of the target query nodes
                    if self.computed[neighbor_node_id]: 
                        count_target_nodes_found += 1
                        if count_target_nodes_found == num_target_nodes:
                            # All target nodes are now connected at this max_key_local level
                            connectivity_level = max_key_local
                
                edge_iterator = edge_iterator.next 
            
            # If edge_iterator still has edges (meaning they had sc < max_key_local),
            # re-bin them with their actual (lower) sc value.
            if edge_iterator is not None: 
                if edge_count_for_ll_pool >= len(self.ll_steiner_nodes):
                    # print("Warning: ll_steiner_nodes pool exhausted when re-binning remaining edges.") # Optional logging
                    pass # Cannot re-bin
                else:
                    remaining_edge_bin: Bin = self.ll_steiner_nodes[edge_count_for_ll_pool]
                    remaining_edge_bin.edge = edge_iterator
                    
                    key_for_remaining_bin: int = remaining_edge_bin.edge.sc # type: ignore
                    key_for_remaining_bin = self._clamp_key(key_for_remaining_bin, len(self.heads_steiner_query))
                        
                    remaining_edge_bin.next = self.heads_steiner_query[key_for_remaining_bin]
                    self.heads_steiner_query[key_for_remaining_bin] = remaining_edge_bin
                    edge_count_for_ll_pool += 1
        
        # Collect all visited nodes in the Q up to bfs_queue_idx
        smcc_nodes_result: List[int] = [self.Q[i_res] for i_res in range(bfs_queue_idx)]
        
        # --- Cleanup phase for query structures ---
        # Clear remaining entries in heads_steiner_query
        # (The loop in the original code from max_key_local down to 0 is fine)
        idx_to_clear = len(self.heads_steiner_query) -1 # Start from the top
        while idx_to_clear >= 0:
            self.heads_steiner_query[idx_to_clear] = None # More direct than checking for None
            idx_to_clear -= 1

        # Reset visited flags for nodes included in the current query's BFS
        for i_vis_clear in range(bfs_queue_idx):
            node_to_clear_vis = self.Q[i_vis_clear]
            if 0 <= node_to_clear_vis < self.n : 
                 self.vis[node_to_clear_vis] = 0
        
        # Reset computed flags for the original query nodes
        for node_idx_comp_clear in valid_query_nodes: 
            # No need to check 0 <= node_idx_comp_clear < self.n if valid_query_nodes is already filtered
            self.computed[node_idx_comp_clear] = 0
            
        return smcc_nodes_result, connectivity_level