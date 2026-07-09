import os
from collections import defaultdict
from typing import Dict, Set, List, Any
from MyGraph import MyEdge,MyGraph
from SGN import SGN

class TecIndexG:
    def __init__(self):
        # Dictionary for original graph vertices to summary graph nodes
        self.vtoSGN: Dict[int, Set[int]] = defaultdict(set)
        # Dictionary for super nodes, key is id and value is the super node object
        self.idSGN: Dict[int, SGN] = {}
        # Index summary graph
        self.SG: Dict[int, Set[int]] = defaultdict(set)
        self.size: float = 0.0

    def construct_index(self, klistdict: Dict[int, Set] , trussd: Dict[MyEdge, int], g: MyGraph):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def find_k_community_for_query(self, query: int, k: int) -> List[List[MyEdge]]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_size(self) -> float:
        return self.size

    def compute_size(self) -> None:
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
