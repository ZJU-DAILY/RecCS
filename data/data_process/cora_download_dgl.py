import dgl.data
import dgl
import scipy.sparse as sp
import networkx as nx
import torch
import os
dir_path = "../datasets/cora/"
os.makedirs(dir_path, exist_ok=True)
dataset = dgl.data.CoraGraphDataset(raw_dir= dir_path, force_reload = False)

graph = dataset[0]
labels = graph.ndata['label']

node_features = graph.ndata['feat']
print(node_features, node_features.shape)
print(node_features[0][0:20])
node_features = torch.sign(node_features)
print(node_features[0][0:20])
# node_features = node_features.type(torch.LongTensor)
# print(node_features[0][0:20])

node_labels = graph.ndata['label']

edges = graph.edges()


i = [edges[0].tolist(), edges[1].tolist()]
v = [1 for j in range(len(edges[0].tolist()))]
adj = torch.sparse.LongTensor(torch.LongTensor(i),
                              torch.LongTensor(v),
                              torch.Size([(node_labels.shape)[0], (node_labels.shape)[0]]))

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse

# print(adj)
# print(edges[0].tolist())
edge0 = edges[0].tolist() 
edge0 = [int(i) for i in edge0]
edge1 = edges[1].tolist() 
edge1 = [int(i) for i in edge1]
edges = torch.Tensor([edge0, edge1]).type(torch.int)
print(edges)
adj = edge_index_to_sparse_coo(edges)
dataset_str = "cora"
torch.save([adj.type(torch.LongTensor), node_features, node_labels.type(torch.LongTensor)], "../datasets/" + dataset_str + "/" + "data.pt")


