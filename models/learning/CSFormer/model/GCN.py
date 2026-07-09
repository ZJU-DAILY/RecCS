import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCNConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, args=None, use_bn=True):
        super(GCN, self).__init__()
        
        self.args = args
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        
        if args.task_type == "regression":
            self.fn1 = nn.Linear(hidden_channels, int(hidden_channels/2))
            self.fn2 = nn.Linear(int(hidden_channels/2), int(hidden_channels/4))
            self.fn3 = nn.Linear(int(hidden_channels/4), out_channels)
        else:
            pass
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, data):
        x = data.graph['node_feat']
        adj = data.graph['adj']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.args.task_type == "regression":
            x = self.fn1(x)
            x = self.fn2(x)
            x = self.fn3(x)
        else:    
            x = self.convs[-1](x, adj)
        return x
    