from .dataset import get_dataset
from memory_count import FunctionPerformanceMonitor
import torch.nn as nn
from torch_geometric.utils import to_undirected
from .data_utils import re_features
import torch.utils.data as Data
from .learning_rate import PolynomialDecayLR
import time
from utils import *
from .model.transformer import TransformerModel
import math
def CSFormer_parameter(input_args):
    args = copy.deepcopy(input_args)
    args.batch_size = 2000
    args.lr = 0.001
    args.hidden_channels = 256
    args.outdim = 128
    args.dropout = 0.5
    args.use_bn = True
    args.num_laryers = 2 # number of layers for deep methods
    args.hops = 4 # Hop of neighbors to be calculated
    args.pe_dim = 3 # position embedding size
    args.hidden_dim = 512 # Hidden layer size
    args.ffn_dim = 64 # FFN layer size
    args.n_layers = 2 # Number of Transformer layers
    args.n_heads = 16 # Number of Transformer heads
    args.attention_dropout = 0.4 # Dropout in the attention layer

    args.min_thr = -1
    args.max_thr = -1
    args.thr_step = -1

    # Learning rate strategy
    args.tot_updates = 1000 # used for optimizer learning rate scheduling
    args.warmup_updates = 400 # warmup steps
    args.peak_lr = 0.001 # learning rate
    args.end_lr = 0.0001 # learning rate
    args.weight_decay = 0.00001 # weight decay

    # Community Search phase param
    args.search_hops = 2 # Community Search area
    args.h_index_order = 1 # order of h-index

    if args.exp_mod == 5 and args.fix:
        args.save_model = args.model_path + "/" + args.dataset + f"_CSFormer_1_0.pth"  # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_CSFormer_log_1_0"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_CSFormer_thr_1_0"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_CSFormer_{args.exp_mod}_{args.exp_param}_fix" # the file to save result
    else:
        args.save_model = args.model_path + "/" + args.dataset + f"_CSFormer_{args.exp_mod}_{args.exp_param}.pth" # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_CSFormer_log_{args.exp_mod}_{args.exp_param}"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_CSFormer_thr_{args.exp_mod}_{args.exp_param}"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_CSFormer_{args.exp_mod}_{args.exp_param}" # the file to save result
    args.save_processData = args.process_path + "/" + args.dataset + f"_CSFormer_{args.exp_mod}_{args.exp_param}.pth" # the file to save DataLoader of model
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_CSFormer_recLabel" # the file to save recommendation label
    return args

def find_neighbors(source_node, G, depth):
    neighbors = [source_node]
    next_visit = [source_node]
    for i in range(depth):
        next_hop_nodes = []
        for node in next_visit:
            neighbors.extend(list(G.neighbors(node)))
            next_hop_nodes.extend(list(G.neighbors(node)))
        next_visit = next_hop_nodes
        if len(next_visit) == 0:
            break
    neighbors = list(set(neighbors))
    return neighbors

def create_search_testdata(G, test_nodes_idx, depth=1, return_info = False):
    neighbors_list = []
    query_time = []
    query_cpu = []
    for i, source_node in enumerate(test_nodes_idx):
        t1 = time.time()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(find_neighbors, source_node, G, depth)
        subgraph = metrics['result']
        neighbors_list.append(subgraph)
        query_cpu.append(metrics['max_memory_mb'])
        t2 = time.time()
        query_time.append(t2- t1)
    if not return_info:
        return test_nodes_idx, neighbors_list
    else:
        return test_nodes_idx, neighbors_list, query_time, query_cpu

def get_max_hindex_nx(G: nx.Graph):
    def hIndex(citations):
        citations.sort(reverse=True)
        for i, j in enumerate(citations):
            if i + 1 > j:
                return i
        return len(citations)

    # Remove self-loops if any
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    degrees = dict(G.degree())
    nodes = list(G.nodes())
    hindex_dict = {}

    # Compute h-index for each node
    for node in nodes:
        neighbor_degrees = [degrees[nbr] for nbr in G.neighbors(node)]
        hindex_dict[node] = hIndex(neighbor_degrees)

    # Compute global max h-index (rounded up to multiple of 10)
    max_hindex = max(hindex_dict.values()) if hindex_dict else 0
    max_hindex = math.ceil(max_hindex / 10) * 10

    return max_hindex

def read_ori_coreness(args):
    # load data
    data_file = args.data_path + "/" + args.dataset

    # read feat
    if args.attr == 0:
        feature = []
        feat_file = data_file + "/feat"
        with open(feat_file, 'r') as file:
            feature = np.array([[float(line.split()[1])] for line in file],
                               dtype=np.float32)  # suppose that the feature has been normalized
    elif args.attr == 1:
        feat_file = data_file + "/feat_attr.npy"
        feature = np.load(feat_file)

    # read graph
    graph_file = data_file + "/graph"
    G = nx.Graph()
    ori_N = feature.shape[0]
    G.add_nodes_from(range(ori_N))
    with open(graph_file, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)

    max_hindex = get_max_hindex_nx(G)
    coreness_dict = nx.algorithms.core.core_number(G)
    for nd in G.nodes():
        if G.degree[nd] == 0:
            coreness_dict[nd] = 0
    coreness_values = list(set(list(coreness_dict.values())))
    coreness_values.sort()
    max_coreness = max(coreness_values)
    return max_hindex, max_coreness

def CSFormer_DataLoader(args, phase = 'all'):
    # read graph and feature
    feature, G, node_id_map, N = read_feat_graph(args)

    # self-loop judge
    assert len(list(nx.selfloop_edges(G))) == 0, "Graph has self-loop"
    # G.remove_edges_from(nx.selfloop_edges(G))

    # continuous id judge
    max_id = max(G.nodes)
    num_nodes = G.number_of_nodes()
    assert max_id == num_nodes - 1, f"The id of nodes in graph must be continuous"
    if args.fix and args.exp_mod == 5: # get the max_hindex and number of different coreness in the original graph
        max_hindex, max_coreness = read_ori_coreness(args)
        dataset = get_dataset(G, args, max_hindex, max_coreness)
    else:
        dataset = get_dataset(G, args)
    if args.attr == 1: # use attribute of graph and forbid h-index feature
        dataset.graph['node_feat'] = torch.from_numpy(feature)
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    processed_features = re_features(dataset.graph['adj'], dataset.graph['node_feat'], args.hops) # propagate feature
    dataset.graph['node_feat'] = processed_features

    processed_train, processed_valid, processed_test = None, None, None
    if phase == 'all' or phase == 'train':
        train_cur_in, train_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'train')
        valid_cur_in, valid_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'valid')
        train_queries = torch.tensor(single_query(train_cur_in))
        valid_queries = torch.tensor(single_query(valid_cur_in))
        # pack train data
        train_data_loader = None
        if args.train_size > 0:
            batch_data_train = Data.TensorDataset(dataset.graph['node_feat'][train_queries], dataset.label[train_queries], train_queries)
            train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True)
        # pack valid data
        batch_data_valid = Data.TensorDataset(dataset.graph['node_feat'][valid_queries], dataset.label[valid_queries],
                                            valid_queries)
        valid_data_loader = Data.DataLoader(batch_data_valid, batch_size=args.batch_size, shuffle=False)
        processed_train = (dataset, train_data_loader, valid_data_loader)
    if phase == 'all' or phase == 'test':
        test_cur_in, test_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'test')
        test_comms = test_cur_out
        test_queries = single_query(test_cur_in)
        # pack test data
        n = dataset.graph['num_nodes']
        idx = torch.Tensor(list(range(n)))
        batch_data_test = Data.TensorDataset(dataset.graph['node_feat'][:n], dataset.label[:n], idx[:n])
        test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False)
        processed_test = (G, test_queries, test_comms, test_data_loader)
    return processed_train, processed_valid, processed_test

def eval_loss(model, valid_data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for item in valid_data_loader:
            processed_features_i = item[0].to(device)
            labels = item[1].to(device)
            out = model(processed_features_i)
            loss = criterion(out, labels)
            total_loss += loss.item()
    return total_loss

def train_one_epoch(train_data_loader, optimizer, model, criterion, lr_scheduler, device):
    t_start = time.time()
    epoch_loss = 0.0
    model.train()
    for i, item in enumerate(train_data_loader):
        optimizer.zero_grad()
        processed_features_i = item[0].to(device)
        labels = item[1].to(device)
        out = model(processed_features_i).to(device)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
    epoch_time = time.time() - t_start
    return epoch_loss, epoch_time

def save_model_complete(model, args, trainloader=None):
    n_class = None
    input_dim = None
    if trainloader is not None:
        dataset, _, _ = trainloader
        n_class = dataset.graph["max_coreness"] + 1
        input_dim = dataset.graph['node_feat'].shape[2]
    
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'hops': args.hops,
            'n_class': n_class,
            'input_dim': input_dim,
            'pe_dim': args.pe_dim,
            'n_layers': args.n_layers,
            'num_heads': args.n_heads,
            'hidden_dim': args.hidden_dim,
            'ffn_dim': args.ffn_dim,
            'dropout_rate': args.dropout,
            'attention_dropout_rate': args.attention_dropout,
            'peak_lr': args.peak_lr,
            'end_lr': args.end_lr,
            'weight_decay': args.weight_decay,
            'warmup_updates': args.warmup_updates,
            'tot_updates': args.tot_updates
        }
    }
    torch.save(model_info, args.save_model)
    print(f"Model saved to {args.save_model}")

def load_model_complete(args, device):
    if not os.path.exists(args.save_model):
        print(f"Model file {args.save_model} not found")
        return None
    
    try:
        model_info = torch.load(args.save_model, map_location=device)
        model_config = model_info['model_config']
        
        model = TransformerModel(
            hops=model_config['hops'],
            n_class=model_config['n_class'],
            input_dim=model_config['input_dim'],
            pe_dim=model_config['pe_dim'],
            n_layers=model_config['n_layers'],
            num_heads=model_config['num_heads'],
            hidden_dim=model_config['hidden_dim'],
            ffn_dim=model_config['ffn_dim'],
            dropout_rate=model_config['dropout_rate'],
            attention_dropout_rate=model_config['attention_dropout_rate']
        )
        
        model.load_state_dict(model_info['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from {args.save_model}")
        print(f"Model config: {model_config}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_pretrained_model(args, device):
    print("Loading pre-trained model...")
    model = load_model_complete(args, device)
    if model is not None:
        print("Pre-trained model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"No pre-trained model found at {args.save_model}")

def CSFormer_train(args, trainloader, device):
    ini_time = time.time()
    
    if trainloader is None:
        print("Empty trainloader provided, attempting to load pre-trained model...")
        model = load_model_complete(args, device)
        if model is not None:
            print("Loaded pre-trained model successfully")
            return model, 0, 0
        else:
            raise ValueError("No pre-trained model found and trainloader is empty. Cannot proceed.")
    
    # set parameter
    dataset, train_data_loader, valid_data_loader = trainloader
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    d = dataset.graph['node_feat'].shape[2]
    c = dataset.graph["max_coreness"] + 1
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e//2} | num node feats {d} | num classes {c}")

    model = load_model_complete(args, device)
    if model is not None:
        print("Loaded pre-trained model, skipping training")
        return model, 0, 0
    
    # set model
    model = TransformerModel(hops=args.hops,
                             n_class=c,
                             input_dim=d,
                             pe_dim=args.pe_dim,
                             n_layers=args.n_layers,
                             num_heads=args.n_heads,
                             hidden_dim=args.hidden_dim,
                             ffn_dim=args.ffn_dim,
                             dropout_rate=args.dropout,
                             attention_dropout_rate=args.attention_dropout).to(device)
    max_cpu_memory = 0
    max_gpu_memory = 0
    
    save_model_complete(model, args, trainloader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epoch)
    early_stopping = EarlyStopping(model, **stopping_args)
    all_loss = []
    
    if train_data_loader is None:
        print("train set is null")
        model = load_model_complete(args, device)
        return model, max_cpu_memory, max_gpu_memory
    # train model
    print("Start Training:")
    best_loss = float('inf')
    best_model_state = None
    for epoch in range(args.epoch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(train_one_epoch, train_data_loader, optimizer, model, criterion, lr_scheduler, device)
        train_loss, epoch_time = metrics['result']
        valid_loss = eval_loss(model, valid_data_loader, criterion, device)
        all_loss.append(valid_loss)
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
        max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
        # save model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_state = model.state_dict().copy()
            save_model_complete(model, args, trainloader)
        avg_loss = float('inf') if len(train_data_loader) == 0 else train_loss / len(train_data_loader)
        avg_valid_loss = float('inf') if len(valid_data_loader) == 0 else valid_loss / len(valid_data_loader)
        sum_time = time.time() - ini_time
        write_train_info(args.save_trainInfo, epoch, epoch_time, sum_time, cur_cpu_memory, cur_gpu_memory, avg_loss, avg_valid_loss)
        if (epoch + 1) % args.save_per_epoch == 0:
            print(f"Epoch [{epoch + 1}/{args.epoch}], Time: {epoch_time:.6f}s, "
                  f"CPU Memory: {cur_cpu_memory:.6f} MB, "
                  f"GPU Memory: {cur_gpu_memory:.6f} MB, "
                  f"Train Loss: {avg_loss:.6f}, "
                  f"Valid Loss: {avg_valid_loss:.6f}")
        # early stop
        if args.early_stopping != 0 and early_stopping.simple_check(all_loss):
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_loss:.6f}")
    
    return model, max_cpu_memory, max_gpu_memory

def community_search(test_node, out, neighbors):
    node_label = out[test_node].item()
    y_pred = out[neighbors]
    y_pred = y_pred.squeeze(1) >= node_label
    y_pred_index = y_pred.nonzero().squeeze(1)
    y_pred = torch.tensor(neighbors)[y_pred_index]
    community = set(y_pred.tolist())
    assert test_node in community, "community must contain query node"
    return community

def CSFormer_RecGenerate(args, model_queries, model_gts, model, device):
    # load data
    feature, G, node_id_map, N = read_feat_graph(args)
    assert len(list(nx.selfloop_edges(G))) == 0, "Graph has self-loop"
    max_id = max(G.nodes)
    num_nodes = G.number_of_nodes()
    assert max_id == num_nodes - 1, f"The id of nodes in graph must be continuous"
    assert max_id == num_nodes - 1, f"The id of nodes in graph must be continuous"
    if args.fix and args.exp_mod == 5: # get the max_hindex and number of different coreness in the original graph
        max_hindex, max_coreness = read_ori_coreness(args)
        dataset = get_dataset(G, args, max_hindex, max_coreness)
    else:
        dataset = get_dataset(G, args)
    if args.attr == 1: # use attribute of graph and forbid h-index feature
        dataset.graph['node_feat'] = torch.from_numpy(feature)
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    processed_features = re_features(dataset.graph['adj'], dataset.graph['node_feat'], args.hops) # propagate feature
    dataset.graph['node_feat'] = processed_features
    test_comms = model_gts
    test_queries = single_query(model_queries)

    # pack test data
    n = dataset.graph['num_nodes']
    idx = torch.Tensor(list(range(n)))
    batch_data_test = Data.TensorDataset(dataset.graph['node_feat'][:n], dataset.label[:n], idx[:n])
    test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False)
    test_nodes_idx, neighbors_list, query_time, query_cpu = create_search_testdata(G, test_queries, depth=args.search_hops, return_info = True)

    # load model
    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()

    out = torch.tensor([])
    with torch.no_grad():
        for _, item in enumerate(test_data_loader):
            i_out = model(item[0].to(device))
            y_pred = i_out.argmax(dim=-1, keepdim=True).cpu()
            out = torch.cat([out, y_pred], dim=0)

    # search the community
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    for i, test_node in enumerate(test_nodes_idx):
        print(f"{i+1}/{len(test_nodes_idx)}")
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(community_search, test_node, out, neighbors_list[i])
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(max(query_cpu[i],cur_cpu_memory))
        gpu_memory.append(cur_gpu_memory)
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1 + query_time[i])

    str_queries = [str(G.nodes[node]['old_id']) for node in test_queries]
    model_thrs = [-1 for i in range(len(communities))]
    return G, communities, test_comms, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def CSFormer_RecQuery(args, model_queries, model_gts, model, device, best_thr = None):
    # load data
    feature, G, node_id_map, N = read_feat_graph(args)
    assert len(list(nx.selfloop_edges(G))) == 0, "Graph has self-loop"
    max_id = max(G.nodes)
    num_nodes = G.number_of_nodes()
    assert max_id == num_nodes - 1, f"The id of nodes in graph must be continuous"
    assert max_id == num_nodes - 1, f"The id of nodes in graph must be continuous"
    if args.fix and args.exp_mod == 5: # get the max_hindex and number of different coreness in the original graph
        max_hindex, max_coreness = read_ori_coreness(args)
        dataset = get_dataset(G, args, max_hindex, max_coreness)
    else:
        dataset = get_dataset(G, args)
    if args.attr == 1: # use attribute of graph and forbid h-index feature
        dataset.graph['node_feat'] = torch.from_numpy(feature)
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    processed_features = re_features(dataset.graph['adj'], dataset.graph['node_feat'], args.hops) # propagate feature
    dataset.graph['node_feat'] = processed_features
    test_queries = single_query(model_queries)

    # pack test data
    n = dataset.graph['num_nodes']
    idx = torch.Tensor(list(range(n)))
    batch_data_test = Data.TensorDataset(dataset.graph['node_feat'][:n], dataset.label[:n], idx[:n])
    test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False)
    test_nodes_idx, neighbors_list, query_time, query_cpu = create_search_testdata(G, test_queries, depth=args.search_hops, return_info = True)

    # load model
    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()

    out = torch.tensor([])
    with torch.no_grad():
        for _, item in enumerate(test_data_loader):
            i_out = model(item[0].to(device))
            y_pred = i_out.argmax(dim=-1, keepdim=True).cpu()
            out = torch.cat([out, y_pred], dim=0)

    # search the community
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    for i, test_node in enumerate(test_nodes_idx):
        print(f"{i+1}/{len(test_nodes_idx)}")
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(community_search, test_node, out, neighbors_list[i])
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(max(query_cpu[i],cur_cpu_memory))
        gpu_memory.append(cur_gpu_memory)
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1 + query_time[i])

    str_queries = [str(G.nodes[node]['old_id']) for node in test_queries]

    return communities, str_queries, all_t, cpu_memory, gpu_memory

def CSFormer_test(args, model, testloader, best_thr, device):
    G, queries, test_comms, test_data_loader = testloader
    test_nodes_idx, neighbors_list, query_time, query_cpu = create_search_testdata(G, queries, depth=args.search_hops, return_info = True)

    # predict coreness of all nodes
    model.eval()
    out = torch.tensor([])
    with torch.no_grad():
        for _, item in enumerate(test_data_loader):
            i_out = model(item[0].to(device))
            y_pred = i_out.argmax(dim=-1, keepdim=True).cpu()
            out = torch.cat([out, y_pred], dim=0)

    # search the community
    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    for i, test_node in enumerate(test_nodes_idx):
        print(f"{i+1}/{len(test_nodes_idx)}")
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(community_search, test_node, out, neighbors_list[i])
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(max(query_cpu[i],cur_cpu_memory))
        gpu_memory.append(cur_gpu_memory)
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1 + query_time[i])

    str_queries = [str(G.nodes[node]['old_id']) for node in queries]
    return G, communities, test_comms, str_queries, all_t, cpu_memory, gpu_memory