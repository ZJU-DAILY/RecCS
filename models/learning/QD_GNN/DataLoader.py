import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp
from torch.utils.data.dataset import Dataset

class Unified_Dataset(Dataset):
    def __init__(self, samples_in=None,
                 samples_out=None, samples_att=None):
        self.samples_in = samples_in
        self.samples_out = samples_out
        self.samples_att = samples_att

    def __len__(self):
        return self.samples_in.shape[0]

    def __getitem__(self, item):
        cur_in = self.samples_in[item, :]
        cur_out = self.samples_out[item, :]
        cur_att = self.samples_att[item, :]

        cur_in = cur_in[:, np.newaxis]  # BN1
        cur_out = cur_out[:, np.newaxis]  # BN1
        cur_att = cur_att[:, np.newaxis]  # BN1

        return cur_in, cur_out, cur_att

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