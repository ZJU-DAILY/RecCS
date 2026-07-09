import os
import sys
import time
import collections
from collections import defaultdict
from typing import Dict, Set, List, Any

class MyEdge:
    """Edge class representing a connection between two vertices"""
    def __init__(self, s: int, t: int):
        self.s = s  # Start vertex
        self.t = t  # End vertex
        # self.sp = -1  # Optional: support, can be uncommented if needed

class SGN:
    """Super Graph Node - represents a super node in the index structure"""
    def __init__(self, truss, tnid):
        self.truss = truss  # Value of the truss
        self.idd = tnid      # Node ID
        self.edgelist = []   # List to hold edges

    def add_edge(self, e):
        self.edgelist.append(e)  # Method to add an edge to the edge list

class MyGraph:
    """Graph class with truss decomposition and community search functionality"""
    def __init__(self):
        # Dictionary to hold the edge list: vertex -> {neighbor vertex -> MyEdge}
        self.g = {}
        self.number_of_edge = 0

    def compute_truss(self, path, trussd):
        """Compute truss decomposition of the graph"""
        klistdict = {}
        kedgelist = set()  # To store edges with the same truss value
        sp = {}
        kmax = self.compute_support(sp)
        k = 2

        sorted_elbys = [None] * len(sp)
        sorted_ep = {}
        svp = {}

        self.bucket_sort_edge_list(kmax, sp, sorted_elbys, svp, sorted_ep)

        for i in range(len(sorted_elbys)):
            e = sorted_elbys[i]
            val = sp[e]

            if val > (k - 2):
                klistdict[k] = kedgelist
                k = val + 2
                kedgelist = set()

            src = e.s
            dst = e.t
            nls = set(self.g[src].keys())

            if len(nls) > len(self.g[dst]):
                dst = e.s
                src = e.t
                nls = set(self.g[src].keys())

            for v in nls:
                if dst in self.g[v]:
                    e1 = self.get_edge(v, src)
                    e2 = self.g[v][dst]
                    if e1 not in trussd and e2 not in trussd:
                        if sp[e1] > (k - 2):
                            self.reorder_el(sorted_elbys, sorted_ep, sp, svp, e1)
                        if sp[e2] > (k - 2):
                            self.reorder_el(sorted_elbys, sorted_ep, sp, svp, e2)

            kedgelist.add(e)
            trussd[e] = k

        klistdict[k] = kedgelist
        return klistdict

    @staticmethod
    def reorder_el(sorted_elbys, sorted_ep, supd, svp, e1):
        """Reorder edge list during truss computation"""
        val = supd[e1]
        pos1 = sorted_ep[e1]
        cp = svp[val]

        if cp != pos1:
            tmp2 = sorted_elbys[cp]
            sorted_ep[e1] = cp
            sorted_ep[tmp2] = pos1
            sorted_elbys[pos1] = tmp2
            svp[val] = cp + 1
            sorted_elbys[cp] = e1
        else:
            if cp + 1 < len(sorted_elbys) and supd[sorted_elbys[cp + 1]] == val:
                svp[val] = cp + 1
            else:
                svp[val] = -1

        if val - 1 not in svp or svp[val - 1] == -1:
            svp[val - 1] = cp
        supd[e1] = val - 1

    @staticmethod
    def bucket_sort_edge_list(kmax, sp, sorted_elbys, svp, sorted_ep):
        """Bucket sort for edge list based on support values"""
        bucket = [0] * (kmax + 1)

        for e in sp.keys():
            bucket[sp[e]] += 1

        p = 0
        for j in range(kmax + 1):
            tmp = bucket[j]
            bucket[j] = p
            p += tmp

        for i in range(len(sorted_elbys)):
            sorted_elbys[i] = None

        for e in sp.items():
            sorted_elbys[bucket[e[1]]] = e[0]
            sorted_ep[e[0]] = bucket[e[1]]
            if e[1] not in svp:
                svp[e[1]] = bucket[e[1]]
            bucket[e[1]] += 1

    def write_support(self, filename, trussd):
        """Write truss values to file"""
        with open(filename, 'w') as bw:
            for e in trussd.keys():
                bw.write(f"{e.s},{e.t},{trussd[e]}\n")

    def compute_support(self, sp):
        """Compute support values for all edges"""
        s = 0
        maxs = 0

        for v in self.g:
            for v2 in self.g[v]:
                if self.g[v][v2] not in sp:
                    s = 0
                    for v3 in self.g[v]:
                        if v2 != v3 and v3 in self.g[v2]:
                            s += 1
                    if s > maxs:
                        maxs = s
                    sp[self.get_edge(v, v2)] = s
        return maxs

    def read_graph_edgelist(self, file_name):
        """Read graph from edge list file"""
        print(f"Reading graph from file: {file_name}")
        self.g = {}

        if not os.path.exists(file_name):
            print("File does not exist")
            exit(0)

        with open(file_name, 'r') as br:
            noe = 0
            for line in br:
                if self.process_line(line.strip()) == 1:
                    noe += 1

        self.number_of_edge = noe
        print(f"# of vertices: {len(self.g)}")
        print(f"# of edges: {self.number_of_edge}")

    def process_line(self, a_line):
        """Process a single line from the graph file"""
        id1, id2 = map(int, a_line.split())
        if id1 == id2:
            return 0

        me = MyEdge(id1, id2)

        if id1 not in self.g:
            self.g[id1] = {}
            self.g[id1][id2] = me
        else:
            if id2 in self.g[id1]:
                return 0
            self.g[id1][id2] = me

        if id2 not in self.g:
            self.g[id2] = {}
            self.g[id2][id1] = me
        else:
            self.g[id2][id1] = me

        return 1

    def get_edge(self, u, v):
        """Get edge between two vertices"""
        return self.g[u][v]

    def remove_edge(self, u, v):
        """Remove edge between two vertices"""
        if v in self.g[u]:
            re = self.g[u][v]
            del self.g[u][v]
            del self.g[v][u]
            return re
        return None

    def remove_edge_e(self, e):
        """Remove edge object from graph"""
        if e.s in self.g and e.t in self.g[e.s]:
            del self.g[e.s][e.t]
            del self.g[e.t][e.s]
            return 1
        return 0

    def add_edge(self, e):
        """Add edge object to graph"""
        if e.s in self.g:
            if e.t in self.g[e.s]:
                return 0
            self.g[e.s][e.t] = e
        else:
            self.g[e.s] = {e.t: e}

        if e.t in self.g:
            if e.s in self.g[e.t]:
                return 0
            self.g[e.t][e.s] = e
        else:
            self.g[e.t] = {e.s: e}

        return 1

    def add_edge_by_vals(self, x, y):
        """Add edge by vertex values"""
        me = MyEdge(x, y)
        if x not in self.g:
            self.g[x] = {y: me}
        else:
            if y in self.g[x]:
                return self.g[x][y]
            self.g[x][y] = me

        if y not in self.g:
            self.g[y] = {x: me}
        else:
            if x in self.g[y]:
                return self.g[y][x]
            self.g[y][x] = me

        return me

    def get_adj_list(self, x):
        """Get adjacency list for vertex x"""
        return self.g[x].keys()

    def contains_edge(self, u, v):
        """Check if edge exists between u and v"""
        return v in self.g[u]

    def read_graph_edgelist_wt(self, file_name, trussd):
        """Read graph with weights/truss values"""
        self.g = {int(i): {} for i in range(int(18483186.0 / 0.75 + 100))}

        with open(file_name, 'r') as br:
            noe = 0
            for line in br:
                a = self.process_line_wt(line.strip(), trussd)
                if a == 1:
                    noe += 1

        self.number_of_edge = noe
        print(f"Number of vertices: {len(self.g)}")
        print(f"Number of edges: {self.number_of_edge}")

    def process_line_wt(self, a_line, trussd):
        """Process line with weight/truss value"""
        id1, id2, t = map(int, a_line.split(','))

        if id1 == id2:
            return 0

        me = MyEdge(id1, id2)

        if id1 not in self.g:
            self.g[id1] = {}
            self.g[id1][id2] = me
        else:
            if id2 in self.g[id1]:
                return 0
            self.g[id1][id2] = me

        if id2 not in self.g:
            self.g[id2] = {}
            self.g[id2][id1] = me
        else:
            self.g[id2][id1] = me

        trussd[me] = t
        return 1

    def read_truss_file(self, path):
        """Read truss values from file"""
        trussd = {}
        with open(path + "trussd.txt", 'r') as br:
            for line in br:
                u, v, t = map(int, line.strip().split(','))
                trussd[self.get_edge(u, v)] = t
        return trussd

    @staticmethod
    def create_k_edge_list(trussd):
        """Create k-edge list from truss dictionary"""
        kmax = max(trussd.values())
        klistdict = {}
        
        for e in trussd.keys():
            key = trussd[e]
            if key not in klistdict:
                klistdict[key] = set()
            klistdict[key].add(e)

        return klistdict

class TecIndexG:
    """Base class for TEC index structures"""
    def __init__(self):
        # Dictionary for original graph vertices to summary graph nodes
        self.vtoSGN: Dict[int, Set[int]] = defaultdict(set)
        # Dictionary for super nodes, key is id and value is the super node object
        self.idSGN: Dict[int, SGN] = {}
        # Index summary graph
        self.SG: Dict[int, Set[int]] = defaultdict(set)
        self.size: float = 0.0

    def construct_index(self, klistdict: Dict[int, Set] , trussd: Dict[MyEdge, int], g: MyGraph):
        """Abstract method to construct index"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def find_k_community_for_query(self, query: int, k: int) -> List[List[MyEdge]]:
        """Abstract method to find k-community for query"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_size(self) -> float:
        """Get index size"""
        return self.size

    def compute_size(self) -> None:
        """Compute index size"""
        self.size = 0.0
        for v in self.vtoSGN.keys():
            self.size += (8 + len(self.vtoSGN[v]) * 4)

        for iv in self.idSGN.keys():
            self.size += 4  # for iv
            self.size += 4  # for truss value of index vertex iv
            self.size += 4  # for id of index vertex iv
            self.size += 4 * len(self.idSGN[iv].edgelist)

        self.size += 8 * len(self.SG)
        for v in self.SG.keys():
            self.size += 4 * len(self.SG[v])

    def write_index(self, path: str) -> None:
        """Write index to files"""
        with open(os.path.join(path, "superNodes.txt"), "w") as bw:
            for sid in self.idSGN.keys():
                sg = self.idSGN[sid]
                bw.write(f"id,{sid},truss,{sg.truss}\n")
                for e in sg.edgelist:
                    bw.write(f"{e.s},{e.t}\n")

        with open(os.path.join(path, "ogn_ign_dic.txt"), "w") as bw:
            bw.write("original_node_id index_graph_node_id\n")
            for k in self.vtoSGN.keys():
                for ign in self.vtoSGN[k]:
                    bw.write(f"{k} {ign}\n")

        with open(os.path.join(path, "summaryIndexGraph.txt"), "w") as bw:
            for kid in self.SG.keys():
                for nid in self.SG[kid]:
                    bw.write(f"{kid},{nid}\n")

    def read_index(self, g: MyGraph, path: str) -> None:
        """Read index from files"""
        with open(os.path.join(path, "superNodes.txt"), "r") as br:
            line = br.readline()
            sr = line.strip().split(",")
            id = int(sr[1])
            truss = int(sr[3])
            sg = SGN(truss, id)

            for line in br:
                sr = line.strip().split(",")
                if sr[0] == "id":
                    self.idSGN[id] = sg
                    nl = set()
                    self.SG[id] = nl
                    id = int(sr[1])
                    truss = int(sr[3])
                    sg = SGN(truss, id)
                else:
                    sg.edgelist.append(g.get_edge(int(sr[0]), int(sr[1])))

            self.idSGN[id] = sg
            nl = set()
            self.SG[id] = nl

        for ci in self.idSGN.keys():
            for e in self.idSGN[ci].edgelist:
                if e.s not in self.vtoSGN:
                    self.vtoSGN[e.s] = set()
                self.vtoSGN[e.s].add(ci)

                if e.t not in self.vtoSGN:
                    self.vtoSGN[e.t] = set()
                self.vtoSGN[e.t].add(ci)

        with open(os.path.join(path, "summaryIndexGraph.txt"), "r") as br:
            for line in br:
                if line.strip() == "vertex":
                    break
                sr = line.strip().split(",")
                self.SG[int(sr[0])].add(int(sr[1]))
                self.SG[int(sr[1])].add(int(sr[0]))

class TecIndexSB(TecIndexG):
    """TEC Index with Subgraph Building implementation"""
    def construct_index(self, klistdict: Dict[int, Set[MyEdge]], trussd: Dict[MyEdge, int], mg: MyGraph) -> None:
        """Construct the TEC index"""
        edgeigd = {}  # To store edge ID
        tnid = 0  # Tree node ID

        if 2 in klistdict:
            klistdict.pop(2)

        for t in klistdict.keys():
            Kedgelist = klistdict[t].copy()
            while Kedgelist:
                ek = Kedgelist.pop()  # Get and remove an edge from the list
                proes = set()
                Qk = collections.deque([ek])  # Queue of edges to process
                proes.add(ek)
                Vk = SGN(t, tnid)  # Create a new super node
                self.idSGN[tnid] = Vk
                nl = set()
                self.SG[tnid] = nl

                while Qk:
                    e = Qk.popleft()
                    x, y = e.s, e.t

                    # Ensure x is the smaller vertex (by degree)
                    if len(mg.g[x]) > len(mg.g[y]):
                        x, y = y, x

                    Vk.add_edge(e)
                    self.add_com_vertex(x, tnid)
                    self.add_com_vertex(y, tnid)

                    self.add_edge_to_truss_com(e, tnid, edgeigd)
                    mg.remove_edge(x, y)

                    for ne in mg.g[x].keys():
                        if ne in mg.g[y]:
                            e1 = mg.get_edge(x, ne)
                            t1 = trussd[e1]
                            e2 = mg.get_edge(y, ne)
                            t2 = trussd[e2]
                            self.process_triangle_edge(e1, t1, proes, Kedgelist, Qk, Vk, edgeigd)
                            self.process_triangle_edge(e2, t2, proes, Kedgelist, Qk, Vk, edgeigd)

                tnid += 1  # Increment tree node ID

    def add_com_vertex(self, x: int, tns: int) -> None:
        """Add vertex to community"""
        if x in self.vtoSGN:
            self.vtoSGN[x].add(tns)
        else:
            self.vtoSGN[x] = {tns}

    @staticmethod
    def process_triangle_edge(e1: MyEdge, t1: int, proes: Set[MyEdge],
                              kedgelist: Set[MyEdge], Qk: collections.deque,
                              Vk: SGN, edgeigd: Dict[MyEdge, Dict[int, int]]) -> None:
        """Process triangle edge during index construction"""
        if e1 not in proes:
            if t1 == Vk.truss:
                kedgelist.remove(e1)
                Qk.append(e1)
            else:
                TecIndexSB.add_edge_for_edge_spec(e1, Vk, edgeigd)
            proes.add(e1)

    @staticmethod
    def add_edge_for_edge_spec(e1: MyEdge, Vk: SGN, edgeigd: Dict[MyEdge, Dict[int, int]]) -> None:
        """Add edge for edge specification"""
        if e1 not in edgeigd:
            nl = {Vk.idd: Vk.truss}
            edgeigd[e1] = nl
        else:
            edgeigd[e1][Vk.idd] = Vk.truss

    def add_edge_to_truss_com(self, e: MyEdge, tns: int, edgeigd: Dict[MyEdge, Dict[int, int]]) -> None:
        """Add edge to truss community"""
        if e in edgeigd:
            for cm in edgeigd[e].keys():
                if cm not in self.SG[tns]:
                    self.SG[tns].add(cm)
                    self.SG[cm].add(tns)
            del edgeigd[e]  # Remove the edge from edgeigd after processing

    def find_k_community_for_query(self, query: int, k: int) -> List[List[MyEdge]]:
        """Find k-community for query vertex"""
        qIn = collections.deque(self.vtoSGN[query])
        cl = []
        ignidl = set()
        ignidq = collections.deque()
        community = []

        while qIn:
            qid = qIn.popleft()

            if self.idSGN[qid].truss >= k and qid not in ignidl:
                ignidq.append(qid)
                ignidl.add(qid)
                community.extend(self.idSGN[qid].edgelist)

                while ignidq:
                    ig = ignidq.popleft()
                    for nid in self.SG[ig]:
                        if self.idSGN[nid].truss >= k and nid not in ignidl:
                            ignidq.append(nid)
                            ignidl.add(nid)
                            community.extend(self.idSGN[nid].edgelist)

                cl.append(community)
                print("Number of edges in this community:", len(community))

        return cl

class MainF:
    """Main class for K-Truss execution"""
    def __init__(self):
        print("Welcome to Equitruss")
    
    @staticmethod
    def main(args):
        """Main execution function"""
        # Initialize parameters
        if len(args) == 3:
            c = int(args[0])
            file_name = args[1]
            pathtec = args[2]
        else:
            c = 1
            print("Path for graph file and path for index files is not given. Toy graph will be used")
            file_name = "data/toyg.txt"
            pathtec = "toy"

        tec = TecIndexSB()
        print(file_name)
        mg = MyGraph()

        # Read Graph
        print("start")
        start_time = time.time()
        mg.read_graph_edgelist(file_name)
        end_time = time.time()
        print(f"Graph read time: {end_time - start_time:.2f} seconds")
        
        if c == 1:
            klistdict = {}
            trussd = {}

            # Create directory for index files
            MainF.create_dir(pathtec)

            # Truss Decomposition
            start_time = time.time()
            klistdict = mg.compute_truss(pathtec, trussd)
            end_time = time.time()
            print(f"Truss computation time: {end_time - start_time:.2f} seconds")
            mg.write_support(f"{pathtec}/truss.txt", trussd)

            # Create Index
            start_time = time.time()
            tec.construct_index(klistdict, trussd, mg)
            end_time = time.time()
            print(f"Index for given graph is created. Index creation time: {end_time - start_time:.2f} seconds")
            tec.write_index(pathtec)
        else:
            if not os.path.exists(pathtec):
                print("Given path for index files does not exist")
                exit(0)
            tec.read_index(mg, pathtec)
            print("Index files are read and index is created")
            MainF.test(tec)

    @staticmethod
    def test(tec):
        """Interactive testing function"""
        print("Enter query node id and k truss value")
        query = int(input())
        while query != -1:
            k = int(input())
            start_time = time.time()
            com = tec.find_k_community_for_query(query, k)
            end_time = time.time()

            if not com:
                print("There is no community for this query with given truss value")
            else:
                for community in com:
                    for edge in community:
                        print(f"({edge.s}, {edge.t}), ", end="")
                    print()

            print(f"Query time: {end_time - start_time:.2f} seconds")
            print("\n\nEnter query node id and k truss value, -1 to exit")
            query = int(input())

    @staticmethod
    def create_dir(path):
        """Create directory for output files"""
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        except Exception as e:
            print("There is a problem with creating directory:", e)

if __name__ == "__main__":
    MainF.main(sys.argv[1:]) 