import os
import shutil
import sys
import pickle
import time
import psutil
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import threading

class SimpleLock:
    def __init__(self, lock_file: str, timeout: int = 300):
        self.lock_file = lock_file
        self.timeout = timeout
        
    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                if not os.path.exists(self.lock_file):
                    with open(self.lock_file, 'w') as f:
                        f.write(f"{os.getpid()}\n{time.time()}\n")
                    return self
                else:
                    try:
                        with open(self.lock_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                lock_time = float(lines[1].strip())
                                if time.time() - lock_time > self.timeout:
                                    print(f"[LOCK] Removing stale lock file: {self.lock_file}")
                                    os.remove(self.lock_file)
                                    continue
                    except (ValueError, FileNotFoundError):
                        try:
                            os.remove(self.lock_file)
                        except:
                            pass
                        continue
                
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Failed to acquire lock {self.lock_file} within {self.timeout}s")
                
                time.sleep(0.1)
                
            except Exception as e:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Failed to acquire lock {self.lock_file}: {e}")
                time.sleep(0.1)
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if os.path.exists(self.lock_file):
                with open(self.lock_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        lock_pid = int(lines[0].strip())
                        if lock_pid == os.getpid():
                            os.remove(self.lock_file)
        except:
            pass


class IndexManager:  
    def __init__(self, output_dir: str = "output/model", args = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.lock_dir = os.path.join(output_dir, "locks")
        os.makedirs(self.lock_dir, exist_ok=True)
        self.args = args

    def get_index_path(self, method_name: str, dataset_name: str, experiment_type: Optional[str] = None, **kwargs) -> str:
        if self.args is not None:
            parts = dataset_name.split('_')
            index_id = f"{parts[0]}_{method_name}"
        else:
            index_id = f"{dataset_name}_{method_name}"
        
        return os.path.join(self.output_dir, f"{index_id}_index.pkl")
    
    def get_stats_path(self, method_name: str, dataset_name: str, experiment_type: Optional[str] = None, **kwargs) -> str:
        index_id = f"{method_name}_{dataset_name}"

        return os.path.join(self.output_dir, f"{index_id}_index_stats.json")
    
    def get_lock_path(self, method_name: str, dataset_name: str, experiment_type: Optional[str] = None, **kwargs) -> str:
        index_id = f"{method_name}_{dataset_name}"
        if experiment_type:
            index_id += f"_{experiment_type}"
        else:
            index_id += "_standard"
        for key, value in kwargs.items():
            if value is not None:
                index_id += f"_{key}_{value}"
                
        return os.path.join(self.lock_dir, f"{index_id}_build.lock")
    
    def build_and_save_index(self, method_name: str, dataset_name: str, graph_path: str, 
                           experiment_type: Optional[str] = None, **kwargs) -> Tuple[bool, Dict]:
        if self.index_exists(method_name, dataset_name, experiment_type, **kwargs):
            print(f"[INDEX] Index already exists for {method_name} on {dataset_name}, loading existing index")
            return self.load_index(method_name, dataset_name, experiment_type, **kwargs)
        lock_path = self.get_lock_path(method_name, dataset_name, experiment_type, **kwargs)
        print(f"[INDEX] Building index for {method_name} on {dataset_name}...")
        print(f"[INDEX] Using lock file: {lock_path}")
        
        try:
            with SimpleLock(lock_path, timeout=1800): 
                if self.index_exists(method_name, dataset_name, experiment_type, **kwargs):
                    print(f"[INDEX] Index was built by another process, loading existing index")
                    return self.load_index(method_name, dataset_name, experiment_type, **kwargs)
                print(f"[INDEX] Process {os.getpid()} acquired lock, building index...")
                start_time = time.time()
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                try:
                    if method_name == 'kcore':
                        index_data = self._build_kcore_index(graph_path)
                    elif method_name == 'ktruss':
                        index_data = self._build_ktruss_index(graph_path)
                    elif method_name == 'kclique':
                        index_data = self._build_kclique_index(graph_path)
                    elif method_name == 'kecc':
                        index_data = self._build_kecc_index(graph_path)
                    else:
                        raise ValueError(f"Unsupported method: {method_name}")
                    
                    end_time = time.time()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    build_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    peak_memory = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else end_memory
                    
                    index_path = self.get_index_path(method_name, dataset_name, experiment_type, **kwargs)
                    with open(index_path, 'wb') as f:
                        pickle.dump(index_data, f)
                    
                    index_size = os.path.getsize(index_path) / 1024 / 1024  # MB
                    
                    stats = {
                        'method': method_name,
                        'dataset': dataset_name,
                        'graph_path': graph_path,
                        'experiment_type': experiment_type,
                        'parameters': kwargs,
                        'build_time': build_time,
                        'memory_usage': memory_usage,
                        'peak_memory': peak_memory,
                        'index_size': index_size,
                        'build_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'success': True,
                        'process_id': os.getpid()
                    }
                    
                    if 'stats' in index_data:
                        stats.update(index_data['stats'])
                    
                    
                    print(f"[INDEX] Process {os.getpid()}: {method_name} index built successfully!")
                    print(f"   Build time: {build_time:.2f}s")
                    print(f"   Memory usage: {memory_usage:.2f}MB")
                    print(f"   Index size: {index_size:.2f}MB")
                    print(f"   Index saved to: {index_path}")
                    
                    return True, stats
                    
                except Exception as e:
                    end_time = time.time()
                    build_time = end_time - start_time
                    
                    stats = {
                        'method': method_name,
                        'dataset': dataset_name,
                        'graph_path': graph_path,
                        'experiment_type': experiment_type,
                        'parameters': kwargs,
                        'build_time': build_time,
                        'error': str(e),
                        'build_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'success': False,
                        'process_id': os.getpid()
                    }
                    
                    print(f"[INDEX] Process {os.getpid()}: Failed to build {method_name} index: {e}")
                    return False, stats
                    
        except TimeoutError as e:
            print(f"[INDEX] Process {os.getpid()}: Failed to acquire lock for {method_name} index: {e}")
            print(f"[INDEX] Process {os.getpid()}: Another process may be building the same index")
            if self.index_exists(method_name, dataset_name, experiment_type, **kwargs):
                print(f"[INDEX] Process {os.getpid()}: Using existing index built by another process")
                return self.load_index(method_name, dataset_name, experiment_type, **kwargs)
            else:
                stats = {
                    'method': method_name,
                    'dataset': dataset_name,
                    'graph_path': graph_path,
                    'experiment_type': experiment_type,
                    'parameters': kwargs,
                    'error': f'Lock timeout: {e}',
                    'build_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'success': False,
                    'process_id': os.getpid()
                }
                return False, stats
    
    def load_index(self, method_name: str, dataset_name: str, experiment_type: Optional[str] = None, **kwargs) -> Tuple[bool, Any]:
        index_path = self.get_index_path(method_name, dataset_name, experiment_type, **kwargs)
        
        if not os.path.exists(index_path):
            print(f"[INDEX] Index file not found: {index_path}")
            return False, None
        
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            print(f"[INDEX] {method_name} index loaded from: {index_path}")
            return True, index_data
            
        except Exception as e:
            print(f"[INDEX] Failed to load {method_name} index: {e}")
            return False, None
    
    def index_exists(self, method_name: str, dataset_name: str, experiment_type: Optional[str] = None, **kwargs) -> bool:
        index_path = self.get_index_path(method_name, dataset_name, experiment_type, **kwargs)
        return os.path.exists(index_path)
    
    def get_index_stats(self, method_name: str, dataset_name: str, experiment_type: Optional[str] = None, **kwargs) -> Dict:
        stats_path = self.get_stats_path(method_name, dataset_name, experiment_type, **kwargs)
        
        if not os.path.exists(stats_path):
            return {}
        
        try:
            with open(stats_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _build_kcore_index(self, graph_path: str) -> Dict:
        sys.path.append(str(Path(__file__).parent / 'kcore'))
        from kcore import TreeIndex
        tree_index = TreeIndex(graph_path)
        stats = {
            'n_nodes': tree_index.graph.n,
            'max_core': tree_index.graph.core_max,
            'index_type': 'TreeIndex'
        }
        
        return {
            'tree_index': tree_index,
            'graph_path': graph_path,
            'stats': stats
        }
    
    def _build_ktruss_index(self, graph_path: str) -> Dict:
        sys.path.append(str(Path(__file__).parent / 'ktruss'))
        from ktruss import MyGraph, TecIndexSB
        
        mg = MyGraph()
        mg.read_graph_edgelist(graph_path)
        
        klistdict = {}
        trussd = {}
        klistdict = mg.compute_truss("", trussd)
        
        tec_index = TecIndexSB()
        tec_index.construct_index(klistdict, trussd, mg)

        stats = {
            'n_nodes': len(mg.g),
            'n_edges': mg.number_of_edge,
            'max_truss': max(klistdict.keys()) if klistdict else 2,
            'index_type': 'TecIndexSB'
        }
        
        return {
            'tec_index': tec_index,
            'my_graph': mg,
            'trussd': trussd,
            'klistdict': klistdict,
            'graph_path': graph_path,
            'stats': stats
        }
    
    def _build_kclique_index(self, graph_path: str) -> Dict:
        sys.path.append(str(Path(__file__).parent / 'kclique'))
        from Index import KCliqueIndex
        
        print(f"[INDEX] Building K-Clique index...")
        kclique_index = KCliqueIndex(graph_path)
        
        stats = {
            'n_nodes': kclique_index.graph.n,
            'n_edges': kclique_index.graph.m // 2,
            'n_cliques': len(kclique_index.all_cliques),
            'index_type': 'KCliqueIndex'
        }
        
        print(f"[INDEX] K-Clique index built: {stats['n_nodes']} nodes, {stats['n_cliques']} cliques")
        
        return {
            'kclique_index': kclique_index,
            'graph_path': graph_path,
            'stats': stats
        }
    
    def _build_kecc_index(self, graph_path: str) -> Dict:
        sys.path.append(str(Path(__file__).parent / 'kecc'))
        from kecc_wrapper import KECCDetector
        
        detector = KECCDetector()
        
        success = detector.load_from_file(graph_path)
        
        if not success:
            raise Exception("Failed to build KECC index using KECCDetector")
        
        index_dir = detector.work_dir
        cg_bin_path = os.path.join(index_dir, "cg.bin")
        mspt_txt_path = os.path.join(index_dir, "mSPT.txt")
        
        cg_bin_data = None
        mspt_txt_data = None
        
        if os.path.exists(cg_bin_path):
            with open(cg_bin_path, 'rb') as f:
                cg_bin_data = f.read()
            print(f"[INDEX] Read cg.bin: {len(cg_bin_data)} bytes")
        
        if os.path.exists(mspt_txt_path):
            with open(mspt_txt_path, 'r') as f:
                mspt_txt_data = f.read()
            print(f"[INDEX] Read mSPT.txt: {len(mspt_txt_data)} bytes")
        
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            
        stats = {
            'n_nodes': len(detector.node_mapping) if hasattr(detector, 'node_mapping') else 0,
            'cg_bin_size': len(cg_bin_data) if cg_bin_data else 0,
            'mspt_txt_size': len(mspt_txt_data) if mspt_txt_data else 0,
            'index_type': 'KECCIndex'
        }
        
        print(f"[INDEX] KECC index built: {stats['n_nodes']} nodes")
        
        return {
            'node_to_id': detector.node_mapping,
            'id_to_node': detector.reverse_mapping,
            'cg_bin_data': cg_bin_data,
            'mspt_txt_data': mspt_txt_data,
            'graph_path': graph_path,
            'stats': stats
        }