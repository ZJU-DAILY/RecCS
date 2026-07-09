import torch
import random
import numpy as np
import networkx as nx
from .clustergcn import ClusterGCNTrainer
from .community import LocalCommunity
import time
import datetime


class SubGraph(object):
    def __init__(self, args, graph, features):
        self.args = args
        self.graph = graph
        self.features = features
        self._set_sizes()
        self.methods = {}
        self.time_map = {}
        self.rankloss = 0
        self.posforrank = []
        self.negforrank = []


    def _set_sizes(self):
        self.feature_count = self.features.shape[1]
        self.class_count = 2
        self.clusters = [0]
        self.cluster_membership = {node: 0 for node in self.graph.nodes()}


    def build_local_candidate(self, seed , gt):
        # STEP1: select vertices with BFS
        allNodes = []
        allNodes.append(seed)
        posNodes = set()
        negNodes = set()
        community = gt
        length = self.args.subgraph_size
        numLabel = int(length * self.args.ics_train_ratio / 2)
        pos = 0
        while pos < len(allNodes) and pos < length and len(allNodes) < length:
            cnode = allNodes[pos]
            for nb in self.graph.neighbors(cnode):
                if nb not in allNodes and len(allNodes) < length:
                    allNodes.append(nb)
                    if nb in community:
                        posNodes.add(nb)
                    else:
                        negNodes.add(nb)
            pos = pos + 1
        # if len(allNodes) < length:
        #     print(f"Fail to find a connected component containing seed with enough vertices. Number of vertices:{len(allNodes)}")
        # STEP2: process positive and negative samples
        posNodes = list(posNodes)
        negNodes = list(negNodes)
        # print("The size of subgraph is %d" % len(allNodes))
        posLabel = min(numLabel, len(posNodes))
        negLabel = min(numLabel, len(negNodes))
        # if posLabel < numLabel:
        #     print(f"positive sample are not enough: {posLabel}/{numLabel}")
        # if negLabel < numLabel:
        #     print(f"negative samples are not enough: {negLabel}/{numLabel}")
        posNodes.append(seed)
        random.shuffle(posNodes)
        random.shuffle(negNodes)
        if posLabel > 0:
            posNodes = [seed]+posNodes[:posLabel-1]
        else:
            posNodes = [seed]
        negNodes = negNodes[:negLabel]
        # print("Positive Nodes are ")
        # print(posNodes)
        # print("Negative Nodes are ")
        # print(negNodes)

        # STEP3: build candidate subgraph
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}

        self.subgraph = self.graph.subgraph(allNodes)

        # STEP4: build map index
        # print("size of nodes %d size of edges %d" % (len(self.subgraph.nodes), len(self.subgraph.edges)))
        self.sg_nodes[0] = [node for node in sorted(self.subgraph.nodes())]
        self.sg_predProbs = [0.0] * len(self.sg_nodes[0])
        self.sg_predLabels = [0] * len(self.sg_nodes[0])
        self.mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[0]))}
        self.rmapper = {i: node for i, node in enumerate(sorted(self.sg_nodes[0]))}
        self.sg_edges[0] = [[self.mapper[edge[0]], self.mapper[edge[1]]] for edge in self.subgraph.edges()] + [
            [self.mapper[edge[1]], self.mapper[edge[0]]] for edge in self.subgraph.edges()]
        self.sg_posNodes = [self.mapper[node] for node in posNodes]
        self.sg_negNodes = [self.mapper[node] for node in negNodes]

        # STEP5: split training and test data
        allNodes1 = [self.mapper[node] for node in allNodes]
        self.sg_train_nodes[0] = self.sg_posNodes + self.sg_negNodes
        self.sg_test_nodes[0] = list(set(allNodes1).difference(set(self.sg_train_nodes[0])))

        self.sg_test_nodes[0] = sorted(self.sg_test_nodes[0])
        self.sg_train_nodes[0] = sorted(self.sg_train_nodes[0])

        # STEP6: process features and obtain labels of subgraph
        self.sg_features[0] = self.features[self.sg_nodes[0], :]
        self.sg_targets[0] = [0] * len(allNodes1)
        for nd in community:
            if nd in self.mapper:
                self.sg_targets[0][self.mapper[nd]] = 1
        self.sg_targets[0] = np.array(self.sg_targets[0])
        self.sg_targets[0] = self.sg_targets[0][:, np.newaxis]
        self.sg_targets[0] = self.sg_targets[0].astype(int)

        for x in self.sg_posNodes:
            self.sg_predProbs[x] = 1.0
            self.sg_predLabels[x] = 1
            if self.sg_targets[0][x] != 1.0:
                print("wrong")
        for x in self.sg_negNodes:
            self.sg_predProbs[x] = 0.0
            self.sg_predLabels[x] = 0
            if self.sg_targets[0][x] != 0:
                print("wrong")

    def transfer_edges_and_nodes(self, device):
        '''
        Transfering the data to PyTorch format.
        '''
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster]).to(device)
            if len(self.sg_edges[cluster])==0:
                return False
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t().to(device)
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster]).to(device)
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster]).to(device)
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster]).to(device)
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster]).to(device)
        return True

    def community_search(self, seed, gt, device):
        '''
        GNN training subgraph, heuristic search community without/with rking loss
        '''

        # STEP1: build candidate subgraph
        begin_time = time.time()
        self.rankloss = 0
        self.build_local_candidate(seed, gt)
        judge = self.transfer_edges_and_nodes(device)
        if not judge:
            end_time = time.time()
            all_time = end_time - begin_time
            topk = [seed]
            return 0, all_time, 0, 0, topk, 0, 0

        # STEP2: train and predict community
        keepLayers = self.args.layers.copy()
        gcn_trainer = ClusterGCNTrainer(self, device)
        begin_time = time.time()
        nodeweight, predlabels, f1score, train_time, test_time, model_size, model_param, train_gpu_memory, train_cpu_memory = gcn_trainer.train_test_community()
        self.args.layers = keepLayers

        # STEP3: find community with different methods
        lc = LocalCommunity(self.args, self)

        for i in range(len(self.sg_test_nodes[0])):
            self.sg_predProbs[self.sg_test_nodes[0][i]] = nodeweight[i].item()
            self.sg_predLabels[self.sg_test_nodes[0][i]] = predlabels[i].item()

        # BFS Swap
        topk = lc.locate_community_BFS(seed)

        # # Greedy-G
        # topk = lc.locate_community_greedy_graph_prepath(seed)

        end_time = time.time()
        all_time = end_time - begin_time
        return train_time, all_time, model_size, model_param, topk, train_gpu_memory, train_cpu_memory
