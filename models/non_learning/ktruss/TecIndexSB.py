import collections
from typing import Dict, Set, List, Any
from TecIndexG import TecIndexG
from MyGraph import MyGraph,MyEdge
from SGN import SGN

class TecIndexSB(TecIndexG):
    def construct_index(self, klistdict: Dict[int, Set[MyEdge]], trussd: Dict[MyEdge, int], mg: MyGraph) -> None:
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
        if x in self.vtoSGN:
            self.vtoSGN[x].add(tns)
        else:
            self.vtoSGN[x] = {tns}

    @staticmethod
    def process_triangle_edge(e1: MyEdge, t1: int, proes: Set[MyEdge],
                              kedgelist: Set[MyEdge], Qk: collections.deque,
                              Vk: SGN, edgeigd: Dict[MyEdge, Dict[int, int]]) -> None:
        if e1 not in proes:
            if t1 == Vk.truss:
                kedgelist.remove(e1)
                Qk.append(e1)
            else:
                TecIndexSB.add_edge_for_edge_spec(e1, Vk, edgeigd)
            proes.add(e1)

    @staticmethod
    def add_edge_for_edge_spec(e1: MyEdge, Vk: SGN, edgeigd: Dict[MyEdge, Dict[int, int]]) -> None:
        if e1 not in edgeigd:
            nl = {Vk.idd: Vk.truss}
            edgeigd[e1] = nl
        else:
            edgeigd[e1][Vk.idd] = Vk.truss

    def add_edge_to_truss_com(self, e: MyEdge, tns: int, edgeigd: Dict[MyEdge, Dict[int, int]]) -> None:
        if e in edgeigd:
            for cm in edgeigd[e].keys():
                if cm not in self.SG[tns]:
                    self.SG[tns].add(cm)
                    self.SG[cm].add(tns)
            del edgeigd[e]  # Remove the edge from edgeigd after processing

    def find_k_community_for_query(self, query: int, k: int) -> List[List[MyEdge]]:
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
