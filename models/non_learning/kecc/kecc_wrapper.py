import os
import tempfile
import networkx as nx
from typing import List, Set, Dict, Any, Optional, Tuple
import shutil
import signal
import time

from models.non_learning.kecc.kecc import Graph, ConnGraph
from models.non_learning.index_manager import IndexManager


class KECCWrapper:
    def __init__(self, graph: Optional[nx.Graph] = None, k: int = 2, **kwargs):
        self.graph = graph
        self.k = k
        self.silent_mode = kwargs.get('silent_mode', False)
        self.timeout = kwargs.get('timeout', 600) 
        
        self.method_name = kwargs.get('method_name', 'kecc')
        self.graph_path: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.n_nodes = 0
        
        # Index management
        self.index_manager = IndexManager()
        self.index_built = False
        self.force_rebuild_index = False  # Force rebuild index flag
        
        # KECC algorithm components
        self.kecc_graph: Optional[Graph] = None
        self.conn_graph: Optional[ConnGraph] = None
        self.node_to_id: Dict[int, int] = {}
        self.id_to_node: Dict[int, int] = {}
        
        # Index directory - now using persistent directory
        self.index_dir: Optional[str] = None
        self.build_start_time = None
        self.cg_bin_path = None
        self.mspt_txt_path = None
        self.index_base_name = None

    def build_index_only(self, graph_path: str, dataset_name: str):
        try:
            success, stats = self.index_manager.build_and_save_index('kecc', dataset_name, graph_path)
            return success
        except Exception as e:
            print(f"Error building K-ECC index: {e}")
            return False

    def load_graph(self, graph_path: str, dataset_name: Optional[str] = None) -> bool:
        try:
            self.graph_path = graph_path
            self.dataset_name = dataset_name or os.path.basename(os.path.dirname(graph_path))
            
            # Try to load existing index (unless force rebuild)
            if not self.force_rebuild_index and self.dataset_name and self.index_manager.index_exists('kecc', self.dataset_name):
                success, index_data = self.index_manager.load_index('kecc', self.dataset_name)
                if success and index_data:
                    if self._load_from_index_data(index_data):
                        self._log(f"Loaded pre-built KECC index for {self.dataset_name}")
                        return True
                    else:
                        self._log(f"Failed to load index data, will load from graph file")
            
            # If index doesn't exist, load graph from file
            self.graph = nx.Graph()
            with open(graph_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            u, v = int(parts[0]), int(parts[1])
                            if u != v:  # Skip self-loops
                                self.graph.add_edge(u, v)
            
            self.n_nodes = self.graph.number_of_nodes()
            self._log(f"Loaded graph from {graph_path}: {self.n_nodes} nodes, {self.graph.number_of_edges()} edges")
            
            return True
        except Exception as e:
            self._log(f"Error loading graph: {e}")
            return False

    def _load_from_index_data(self, index_data: Dict) -> bool:
        try:
            node_to_id = index_data.get('node_to_id') or index_data.get('node_mapping') or {}
            id_to_node = index_data.get('id_to_node') or index_data.get('reverse_mapping') or {}
            if not node_to_id and id_to_node:
                node_to_id = {v: k for k, v in id_to_node.items()}
            if not id_to_node and node_to_id:
                id_to_node = {v: k for k, v in node_to_id.items()}

            n_nodes = index_data.get('n_nodes') or index_data.get('stats', {}).get('n_nodes') or len(node_to_id)
            
            cg_bin_data = index_data.get('cg_bin_data')
            mspt_txt_data = index_data.get('mspt_txt_data')

            if not node_to_id or not id_to_node:
                raise KeyError("Missing node mappings in index_data")

            self.node_to_id = node_to_id
            self.id_to_node = id_to_node
            self.n_nodes = n_nodes

            # Rebuild graph with original node IDs
            self.graph = nx.Graph()
            for node_id in self.node_to_id.keys():
                self.graph.add_node(node_id)

            # # index directory
            # if self.index_dir is None:
            #     os.makedirs("output/model", exist_ok=True)
            #     if self.dataset_name:
            #         self.index_base_name = f"kecc_{self.dataset_name}_index_data"
            #     else:
            #         self.index_base_name = f"kecc_graph_index_data"
            #     self.index_dir = os.path.join("output/model", self.index_base_name)
            #     os.makedirs(self.index_dir, exist_ok=True)
            
            # self.cg_bin_path = os.path.join(self.index_dir, "cg.bin")
            # self.mspt_txt_path = os.path.join(self.index_dir, "mSPT.txt")

            with tempfile.NamedTemporaryFile(delete=False) as tmp_cg:
                tmp_cg.write(cg_bin_data)
                cg_tmp_path = tmp_cg.name
            print(f"[TEMP] Wrote cg.bin temp: {cg_tmp_path}, {len(cg_bin_data)} bytes")

            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_txt:
                tmp_txt.write(mspt_txt_data)
                txt_tmp_path = tmp_txt.name
            print(f"[TEMP] Wrote mSPT.txt temp: {txt_tmp_path}, {len(mspt_txt_data)} bytes")
        
            if cg_bin_data and mspt_txt_data:
                self.conn_graph = ConnGraph()
                if self.conn_graph.load_data(cg_tmp_path, txt_tmp_path):
                    self.index_built = True
                    self._log("Successfully loaded KECC index from pkl data")
                    os.remove(cg_tmp_path)
                    os.remove(txt_tmp_path)
                    self._log("Successfully remove tmp files")
                    return True
                else:
                    self._log("Failed to load index from restored files")
                    os.remove(cg_tmp_path)
                    os.remove(txt_tmp_path)
                    self._log("Successfully remove tmp files")
                    return False
                
        except Exception as e:
            self._log(f"Error loading from index data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def build_index(self) -> bool:
        if self.index_built:
            return True
            
        if self.graph is None:
            self._log("No graph loaded")
            return False
        
        try:
            # Check if we can load existing index
            if self.dataset_name and self.index_manager.index_exists('kecc', self.dataset_name):
                success, index_data = self.index_manager.load_index('kecc', self.dataset_name)
                if success and index_data:
                    self._load_from_index_data(index_data)
                    if self.index_built:
                        return True
            

            if self.index_dir is None:
                os.makedirs("output/model", exist_ok=True)
                if self.dataset_name:
                    self.index_base_name = f"kecc_{self.dataset_name}_index_data"
                else:
                    self.index_base_name = f"kecc_graph_index_data"
                self.index_dir = os.path.join("output/model", self.index_base_name)
                os.makedirs(self.index_dir, exist_ok=True)
            
            self.cg_bin_path = os.path.join(self.index_dir, "cg.bin")
            self.mspt_txt_path = os.path.join(self.index_dir, "mSPT.txt")


            # Build new index
            self.build_start_time = time.time()
            self._log(f"Building KECC index for graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
            # Convert graph to edge list format
            graph_file = os.path.join(self.index_dir, "graph.txt")
            self.node_to_id = self._convert_networkx_to_edge_list(graph_file)
            self.id_to_node = {v: k for k, v in self.node_to_id.items()}
            
            # Set up timeout for large graphs
            timeout_seconds = self.timeout
            if self.graph.number_of_nodes() > 5000:
                timeout_seconds = max(600, self.graph.number_of_nodes() // 10)  # At least 10 minutes for large graphs
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"KECC index building timed out after {timeout_seconds} seconds")
            
            import platform
            use_signal_timeout = platform.system() != "Windows" and hasattr(signal, 'SIGALRM')
            
            original_handler = None
            if use_signal_timeout:
                try:
                    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                except (AttributeError, OSError):
                    use_signal_timeout = False
            
            try:
                # Step 1: Initialize KECC Graph and read the graph
                self._log("Step 1: Reading graph structure...")
                self.kecc_graph = Graph()
                
                # Set the directory path for output files
                self.kecc_graph.dir = self.index_dir
                
                self.kecc_graph.read_graph(graph_file)
                
                # Step 2: Compute steiner connectivity  
                self._log("Step 2: Computing steiner connectivity...")
                self.kecc_graph.find_all_steiner_connectivity_bottom_up()
                
                # Step 2.5: Output steiner connectivity to binary file
                self._log("Step 2.5: Saving connectivity graph...")
                with open(self.cg_bin_path, "wb") as fout:
                    self.kecc_graph.output_all_steiner_connectivity(fout)
                
                # Step 3: Build MST (Maximum Spanning Tree)
                self._log("Step 3: Building maximum spanning tree...")
                self.kecc_graph.max_spanning_tree()
                
                # Step 4: Initialize ConnGraph for query processing
                self._log("Step 4: Initializing ConnGraph...")
                self.conn_graph = ConnGraph()
                
                # Step 5: Load the built index data
                self._log("Step 5: Loading index data...")
                if os.path.exists(self.mspt_txt_path):
                    if self.conn_graph.load_data(self.cg_bin_path, self.mspt_txt_path):
                        build_time = time.time() - self.build_start_time
                        self._log(f"KECC index built successfully in {build_time:.3f} seconds")
                        self.index_built = True
                        return True
                    else:
                        self._log("Failed to load KECC index data")
                        return False
                else:
                    self._log(f"MST file not found: {self.mspt_txt_path}")
                    return False
                    
            finally:
                if use_signal_timeout and original_handler is not None:
                    try:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)
                    except (AttributeError, OSError):
                        pass
                
        except TimeoutError as e:
            self._log(f"Index building timed out: {e}")
            return False
        except Exception as e:
            self._log(f"Error building index: {e}")
            return False
    
    def _save_index(self):
        try:
            if not self.dataset_name or not self.graph_path:
                self._log("Missing dataset_name or graph_path, cannot save index")
                return
                
            index_data = {
                'node_to_id': self.node_to_id,
                'id_to_node': self.id_to_node,
                'index_dir': self.index_dir,
                'n_nodes': self.n_nodes,
                'graph_path': self.graph_path
            }
            
            # Use IndexManager to save
            success, stats = self.index_manager.build_and_save_index('kecc', self.dataset_name, self.graph_path)
            if success:
                self._log("KECC index saved successfully using IndexManager")
            else:
                self._log("Failed to save KECC index using IndexManager")
                
        except Exception as e:
            self._log(f"Error saving index: {e}")

    def find_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        import threading
        if not hasattr(self, '_query_lock'):
            self._query_lock = threading.Lock()
        
        with self._query_lock:
            try:
                if not self.index_built or self.conn_graph is None:
                    self._log("Index not properly built, returning query nodes")
                    return set(query_nodes)
                
                kecc_query_nodes = []
                for node in query_nodes:
                    if node in self.node_to_id:
                        kecc_query_nodes.append(self.node_to_id[node])
                    else:
                        if not self.silent_mode:
                            self._log(f"Warning: Query node {node} not found in graph")
                        
                if not kecc_query_nodes:
                    if not self.silent_mode:
                        self._log("No valid query nodes found")
                    return set(query_nodes)
                
                smcc_nodes, connectivity = self.conn_graph.query_smcc(kecc_query_nodes)
                
                result_nodes = set()
                for kecc_id in smcc_nodes:
                    if kecc_id in self.id_to_node:
                        result_nodes.add(self.id_to_node[kecc_id])
                
                if not self.silent_mode and result_nodes:
                    self._log(f"Found SMCC with {len(result_nodes)} nodes, connectivity: {connectivity}")
                
                return result_nodes if result_nodes else set(query_nodes)
                
            except Exception as e:
                if not self.silent_mode:
                    self._log(f"Error in SMCC query: {e}")
                return set(query_nodes)
    
    def search_community(self, query_nodes: List[int], **kwargs) -> Set[int]:
        if 'silent_mode' in kwargs:
            self.silent_mode = kwargs['silent_mode']
        
        if not self.index_built:
            if not self.silent_mode:
                self._log("Index not built, cannot search community")
            return set(query_nodes)
            
        return self.find_community(query_nodes, **kwargs)
    
    def _convert_networkx_to_edge_list(self, output_path: str):
        if self.graph is None:
            raise ValueError("Graph not loaded")
        
        with open(output_path, 'w') as f:
            f.write(f"{self.graph.number_of_nodes()} {self.graph.number_of_edges()}\n")
            
            node_to_id = {node: i for i, node in enumerate(sorted(self.graph.nodes()))}
            for u, v in self.graph.edges():
                u_id = node_to_id[u]
                v_id = node_to_id[v]
                f.write(f"{u_id} {v_id}\n")
        
        return node_to_id
    
    def _log(self, message: str):
        if not self.silent_mode:
            print(f"[KECC] {message}")


class KECCDetector:
    
    def __init__(self):
        self.wrapper = KECCWrapper()
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.work_dir = None
        
    def load_from_file(self, graph_path: str) -> bool:
        try:
            # Extract dataset name from path
            dataset_name = os.path.basename(os.path.dirname(graph_path))
            
            # Load graph using wrapper
            success = self.wrapper.load_graph(graph_path, dataset_name)
            if not success:
                return False
            
            # Build index
            success = self.wrapper.build_index()
            if not success:
                return False
            
            self.node_mapping = self.wrapper.node_to_id.copy()
            self.reverse_mapping = self.wrapper.id_to_node.copy()
            self.work_dir = self.wrapper.index_dir
            
            return True
            
        except Exception as e:
            print(f"[KECC] Error in load_from_file: {e}")
            return False
    
    def query_community(self, query_nodes: List[int]) -> Set[int]:
        return self.wrapper.find_community(query_nodes) 