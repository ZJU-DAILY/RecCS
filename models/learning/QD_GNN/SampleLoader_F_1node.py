import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp
from torch.utils.data.dataset import Dataset
import normalization as Norm

class Unified_Dataset(Dataset):
    def __init__(self, phase, data_dir=None, file=None, feats=None, adjs=None, Adj=None,
                 samples_in=None, samples_dis=None,
                 samples_out=None, samples_att=None):
        if phase == 'init':
            self.processData(data_dir, file)
        else:
            self.feats = feats
            self.adjs = adjs
            self.Adj = Adj
            self.samples_in = samples_in
            self.samples_dis = samples_dis
            self.samples_out = samples_out
            self.samples_att = samples_att

    def processData(self,data_dir,file):
        # load data
        data_file = data_dir + "/" + file

        # read feat
        feature = []
        feat_file = data_file + "/feat"
        with open(feat_file, 'r') as file:
            feature = np.array([[float(line.split()[1])] for line in file], dtype=np.float32)
        feature = self.fnormalize(feature)
        # read graph
        graph_file = data_file + "/graph"
        N = feature.shape[0]
        A = np.zeros((N, N), dtype=np.int32)
        G = nx.Graph()
        with open(graph_file, 'r') as file:
            for line in file:
                node1, node2 = map(int, line.split())
                A[node1, node2] = 1
                A[node2, node1] = 1
                G.add_edge(node1, node2)
        degree = np.zeros(A.shape[0], dtype=np.float32)
        for i in range(A.shape[0]):
            degree[i] = G.degree[i] / A.shape[0]
        Adj = A
        for i in range(A.shape[0]):
            A[i, i] = 1
            Adj[i][i] = 0
        A = Norm.normalized_adjacency(A)
        print("A", A.shape)

        # read query
        query_file = data_file + "/query"
        with open(query_file, "r") as f:
            queries = [list(map(int, line.strip().split())) for line in f.readlines()]
        query_onehot = torch.zeros((len(queries), N))
        for i, query_nodes in enumerate(queries):
            for node in query_nodes:
                query_onehot[i, node] = 1
        cur_in = query_onehot.numpy().astype(np.int32)

        # compute shortest distance
        print("begin shortes path computation")
        dis = nx.shortest_path(G)
        all_dis = np.zeros(A.shape, dtype=np.int32)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # all_dis[i,j] = len(dis[i][j])-1
                if i == j:
                    all_dis[i, j] = 0
                else:
                    if j in dis[i]:
                        assert len(dis[i][j]) > 1
                        all_dis[i, j] = len(dis[i][j]) - 1
                    else:
                        all_dis[i, j] = A.shape[0]

        cur_in_dis = np.zeros(cur_in.shape, dtype=np.float32)
        for i in range(cur_in.shape[0]):
            source = []
            for j in range(cur_in.shape[1]):
                if cur_in[i, j] == 1:
                    source.append(all_dis[j:j + 1])
            source = np.concatenate(source, axis=0)
            assert source.shape[0] == cur_in[i].sum()
            for j in range(cur_in.shape[1]):
                cur_in_dis[i, j] = source[:, j].min() # cur_in is the minimum distance between query vertices and other vertices
        print("end shortes path computation")

        # normalize distance
        for j in range(cur_in_dis.shape[0]):
            max_val = (cur_in_dis[j] * (cur_in_dis[j] < A.shape[0])).max()
            # print("max distance ", max_val)
            if (max_val == 0):
                print("max_val=0")
            for i in range(cur_in_dis[j].shape[0]):
                if cur_in_dis[j][i] == A.shape[0]:
                    cur_in_dis[j][i] = 0.
                else:
                    cur_in_dis[j][i] = 1. - cur_in_dis[j][i] / (max_val + 1)
            for k in range(cur_in.shape[1]):
                if cur_in[j][k] == 1:
                    cur_in_dis[j][k] = 1

        # load labels
        gt_file = data_file + "/gt"
        with open(gt_file, "r") as f:
            ground_truths = [list(map(int, line.strip().split())) for line in f.readlines()]
        gt_onehot = torch.zeros((len(ground_truths), N))
        for i, gt_nodes in enumerate(ground_truths):
            for node in gt_nodes:
                gt_onehot[i, node] = 1
        cur_out = gt_onehot.numpy().astype(np.int32)
        print("label shape", cur_out.shape)

        # set attribute as none
        attr = np.zeros((cur_out.shape[0], feature.shape[1]), dtype=np.int32)
        print("attr shape", attr.shape)

        # store data
        self.feats = feature # normalized feature
        self.adjs = A # normalized adjacency matrix after adding self loop
        self.Adj = Adj # origin adjacency matrix
        self.cur_in = cur_in # query one-hot
        self.cur_in_dis = cur_in_dis # the distance between each query and other vertices
        self.cur_out = cur_out # groundtruth of each query
        self.attr = attr # none

    def get_data(self, start_id, end_id):
        samples_in = []
        samples_out = []
        samples_dis = []
        samples_att = []
        for i in range(start_id, end_id):
            samples_in.append(self.cur_in[i:i + 1])
            samples_dis.append(self.cur_in_dis[i:i + 1])
            samples_out.append(self.cur_out[i:i + 1])
            samples_att.append(self.attr[i:i + 1])
        samples_in = np.concatenate(samples_in, 0)
        samples_dis = np.concatenate(samples_dis, 0)
        samples_out = np.concatenate(samples_out, 0)
        samples_att = np.concatenate(samples_att, 0)
        return Unified_Dataset(feats = self.feats, adjs = self.adjs, Adj = self.Adj, samples_in = samples_in, samples_dis = samples_dis, samples_out = samples_out, samples_att = samples_att)

    def __len__(self):
        return self.samples_in.shape[0]

    def __getitem__(self, item):

        cur_dis = self.samples_dis.copy()[item, :]
        cur_in = self.samples_in.copy()[item, :]
        cur_out = self.samples_out.copy()[item, :]
        cur_att = self.samples_att.copy()[item, :]

        # degree=self.degree[:,np.newaxis]
        # core = self.core[:, np.newaxis]
        # cluster = self.cluster[:, np.newaxis]
        # triangle=self.triangle[:, np.newaxis]

        cur_in = cur_in[:, np.newaxis]  # BN1
        cur_dis = cur_dis[:, np.newaxis]
        cur_out = cur_out[:, np.newaxis]  # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1
        cur_adj = self.adjs.copy()
        feats = self.feats.copy()
        # input=np.concatenate((self.feats[ego].copy(), cur_dis), axis=1)  # BN(D+2)

        # input = cur_in
        # input=np.concatenate((input, cur_in), axis=1) # BN(D+1)

        # input=cur_dis
        # input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)
        #
        input = cur_in
        input = np.concatenate((input, cur_dis), axis=1)  # BN(D+1)

        # return torch.FloatTensor(cur_feats), torch.FloatTensor(cur_out), torch.FloatTensor(cur_adj)
        return input, cur_att, cur_adj, feats, cur_out, self.Adj

    def fnormalize(self, mx):
        """Row-normalize sparse matrix"""

        mx = mx.transpose(0, 1)
        print("mx shape", mx.shape)
        rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        rowsum[rowsum == 0] = 1
        # print("rowsum shape", rowsum.shape)
        print("rowsum", rowsum[:24])
        mx = mx / rowsum[:, np.newaxis]
        mx = mx.transpose(0, 1)
        return mx

    def normalized_adjacency(self, adj):
        # adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(rowsum)
        r_inv = 1 / rowsum
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

        # rowsum = mx.sum(1)
        # rowsum = rowsum[:,np.newaxis]
        # rowsum[rowsum==0] = 1
        # mx = mx/rowsum
        # return mx