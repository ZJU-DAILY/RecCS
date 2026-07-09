from .transzero_utils import *
from utils import gt_to_onehot, query_to_onehot, read_Q_GT, read_feat_graph, split_data
def get_dataset(args):
    feature, G, node_id_map, N = read_feat_graph(args)
    row = []
    col = []
    for uid, vid in G.edges():
        row.extend([uid, vid])
        col.extend([vid, uid])
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.ones(len(row), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()
    adj_scipy = torch_adj_to_scipy(adj)
    graph = dgl.from_scipy(adj_scipy)
    lpe = laplacian_positional_encoding(graph, args.pe_dim)

    # remap feature according to new id
    features = torch.from_numpy(feature)
    features = torch.cat((features, lpe), dim=1)

    train_cur_in, train_cur_out, valid_cur_in, valid_cur_out, test_cur_in, test_cur_out = split_data(args,N,node_id_map, G, return_list = True)
    queries = query_to_onehot(test_cur_in, N)
    return adj, features, queries, test_cur_out, G
