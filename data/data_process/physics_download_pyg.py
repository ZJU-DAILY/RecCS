import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse

dataset_str = 'Physics'

from torch_geometric.datasets import Coauthor
dataset = Coauthor(root='../datasets/', name='Physics')
graph = dataset[0]

# print(graph.x, graph.edge_index, graph.y)
print(graph.edge_index)

torch.save([edge_index_to_sparse_coo(graph.edge_index).type(torch.LongTensor), graph.x.type(torch.LongTensor), graph.y.type(torch.LongTensor)], "../datasets/" + dataset_str + "/" + "data.pt")

