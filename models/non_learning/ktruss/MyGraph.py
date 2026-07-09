import os

class MyEdge:
    def __init__(self, s: int, t: int):
        self.s = s  # Start vertex
        self.t = t  # End vertex
        # self.sp = -1  # Optional: support, can be uncommented if needed
        
class MyGraph:
    def __init__(self):
        # Dictionary to hold the edge list: vertex -> {neighbor vertex -> MyEdge}
        self.g = {}
        self.number_of_edge = 0

    def compute_truss(self, path, trussd):
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
        with open(filename, 'w') as bw:
            for e in trussd.keys():
                bw.write(f"{e.s},{e.t},{trussd[e]}\n")

    def compute_support(self, sp):
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
        return self.g[u][v]

    def remove_edge(self, u, v):
        if v in self.g[u]:
            re = self.g[u][v]
            del self.g[u][v]
            del self.g[v][u]
            return re
        return None

    def remove_edge_e(self, e):
        if e.s in self.g and e.t in self.g[e.s]:
            del self.g[e.s][e.t]
            del self.g[e.t][e.s]
            return 1
        return 0

    def add_edge(self, e):
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
        return self.g[x].keys()

    def contains_edge(self, u, v):
        return v in self.g[u]

    def read_graph_edgelist_wt(self, file_name, trussd):
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
        trussd = {}
        with open(path + "trussd.txt", 'r') as br:
            for line in br:
                u, v, t = map(int, line.strip().split(','))
                trussd[self.get_edge(u, v)] = t
        return trussd

    @staticmethod
    def create_k_edge_list(trussd):
        kmax = max(trussd.values())
        klistdict = {}
        
        for e in trussd.keys():
            key = trussd[e]
            if key not in klistdict:
                klistdict[key] = set()
            klistdict[key].add(e)

        return klistdict
    

