import time
from collections import deque

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .GFCN import QD_GCN,CS_GCN,SimpleCS_GCN,CS_GCN_NoF
from memory_count import FunctionPerformanceMonitor
from .DataLoader import Unified_Dataset
import tracemalloc
from utils import *

def eval_loss(validloader, feats, norm_adj, model):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for i_iter, batch in enumerate(validloader):
            input, label, attr = batch
            input = input.to(next(model.parameters()).device)
            label = label.to(next(model.parameters()).device)
            output, _ = model.forward(input, norm_adj.expand(input.shape[0], -1, -1), feats.expand(input.shape[0], -1, -1), training=False)
            criterion = torch.nn.BCELoss()
            loss = criterion(output, label)
            epoch_loss += loss.item()
    return epoch_loss

def train_one_epoch(trainloader, feats, norm_adj, model, optimizer, args, epoch):
    model.train()
    adjust_lr_poly(optimizer, args.learning_rate, epoch, args.epoch)
    epoch_loss = 0.0
    start = time.time()
    for i_iter, batch in enumerate(trainloader):
        input, label, attr = batch
        input = input.to(next(model.parameters()).device)
        label = label.to(next(model.parameters()).device)
        optimizer.zero_grad()
        output, _ = model.forward(input, norm_adj.expand(input.shape[0], -1, -1), feats.expand(input.shape[0], -1, -1), training=True)
        criterion = torch.nn.BCELoss()
        loss = criterion(output, label)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    end = time.time()
    epoch_time = end - start
    return epoch_loss, epoch_time

def train(train_data, valid_data, feats, norm_adj, model, args, trainloader=None):
    ini_time = time.time()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate, weight_decay=args.weight_decay)
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epoch)
    early_stopping = EarlyStopping(model, **stopping_args)
    all_loss = []
    device = next(model.parameters()).device

    if args.train_size == 0:
        print("train set is null")
        model = load_model_complete(args, device)
        return model, 0, 0
        
    feats = feats.to(device)
    norm_adj = norm_adj.to(device)
    best_loss = float('inf')
    best_model_state = None
    max_cpu_memory = 0
    max_gpu_memory = 0
    for epoch in range(args.epoch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(train_one_epoch,train_data, feats, norm_adj, model, optimizer, args, epoch)
        train_loss, epoch_time = metrics['result']
        valid_loss = eval_loss(valid_data, feats, norm_adj, model)
        all_loss.append(valid_loss)
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
        max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_state = model.state_dict().copy()
            save_model_complete(model, args, trainloader)
        avg_loss = float('inf') if len(train_data) == 0 else train_loss / len(train_data)
        avg_valid_loss = float('inf') if len(valid_data) == 0 else valid_loss / len(valid_data)
        sum_time = time.time() - ini_time
        write_train_info(args.save_trainInfo, epoch, epoch_time, sum_time, cur_cpu_memory, cur_gpu_memory, avg_loss, avg_valid_loss)
        if (epoch + 1) % args.save_per_epoch == 0:
            print(f"Epoch [{epoch + 1}/{args.epoch}], Time: {epoch_time:.6f}s, "
                  f"CPU Memory: {cur_cpu_memory:.6f} MB, "
                  f"GPU Memory: {cur_gpu_memory:.6f} MB, "
                  f"Train Loss: {avg_loss:.6f}, "
                  f"Valid Loss: {avg_valid_loss:.6f}")
        if args.early_stopping !=0 and early_stopping.simple_check(all_loss):
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_loss:.6f}")
    
    return model, max_cpu_memory, max_gpu_memory

def adjust_lr_poly(optimizer, base_lr, epoch, tot_epoch, power=0.9):
    lr = base_lr * (1-epoch*1./tot_epoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def community_BFS(queries, graph, output, thr):
    communities = []
    for idx, query in enumerate(queries):
        query_indices = np.nonzero(query)[0]
        pre_community = output[idx]
        community = set(bfs_connected_nodes(query_indices, graph, pre_community, thr))
        communities.append(community)
    return communities

# search the best parameter based on the BFS result
def _eval_BFS(args, model, validloader):
    valid_data, G, feats, norm_adj = validloader
    model.eval()
    all_preds = []
    all_labels = []
    all_queries = []
    # ensure the batch size is one
    device = next(model.parameters()).device
    feats = feats.to(device)
    norm_adj = norm_adj.to(device)
    for i_iter, batch in enumerate(valid_data):
        input, label, attr = batch
        input = input.to(device)
        output, _ = model.forward(input, norm_adj.expand(input.shape[0], -1, -1), feats.expand(input.shape[0], -1, -1), training=False)
        # output, _ = model.forward(input, attr, norm_adj, feats, feats, training=False)
        all_preds.append(output.detach().cpu().numpy().flatten().tolist())
        all_labels.append(label.numpy().flatten().tolist())
        all_queries.append(input.detach().cpu().numpy().flatten().tolist())
    labels = [set(np.nonzero(row)[0]) for row in all_labels]
    queries = [set(np.nonzero(row)[0]) for row in all_queries]

    # Determine optimal threshold
    best_thr = 0.0
    best_f1 = 0.0
    thresholds = np.arange(args.min_thr, args.max_thr + args.thr_step/2, args.thr_step)
    # Round to avoid floating point precision issues
    thresholds = np.round(thresholds, decimals=2)
    for thr in thresholds:
        # BFS based on threshold
        communities = community_BFS(all_queries, G, all_preds, thr)
        pre, rec, f1, jac = evaluate_gt_communities(communities, labels)
        avg_f1 = np.mean(f1)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thr = thr
    print(f"best thr {best_thr}")
    return best_thr

def bfs_connected_nodes(queries, graph, output, best_thr):
    visited = np.zeros(len(output), dtype=bool)
    queue = deque(queries)
    visited[queries] = True
    community = set()
    while queue:
        current_node = queue.popleft()
        community.add(current_node)
        for neighbor in graph.neighbors(current_node):
            if not visited[neighbor] and output[neighbor] >= best_thr:
                visited[neighbor] = True
                queue.append(neighbor)
    return community

def QD_GNN_parameter(input_args):
    args = copy.deepcopy(input_args)
    args.n_hid1 = 128 # first layer of GCN: number of hidden units, options [64, 128, 256]
    args.n_hid2 = 128 # second layer of GCN: number of hidden units, options [64, 128, 256]
    args.n_expert = 128 # attention layer: number of experts, options [16, 32, 64, 128]
    args.att_hid = 128 # attention layer: hidden units, options [64, 128, 256]
    args.dropout = 0.5 # dropout rate (1 - keep probability)
    args.batch_size = 4 # options: [32, 64, 128]
    args.learning_rate = 0.001 # options [1e-3, 1e-4]
    args.n_classes = 1 # the output classes number
    args.weight_decay = 5e-4 # weight decay (L2 loss on parameters)
    args.min_thr = 0
    args.max_thr = 0.95
    args.thr_step = 0.05
    if args.exp_mod == 5 and args.fix:
        args.save_model = args.model_path + "/" + args.dataset + f"_QD_GNN_1_0.pth"  # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_QD_GNN_log_1_0"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_QD_GNN_thr_1_0"  # the file to save QD-GNN best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_QD_GNN_{args.exp_mod}_{args.exp_param}_fix" # the file to save result
    else:
        args.save_model = args.model_path + "/" + args.dataset + f"_QD_GNN_{args.exp_mod}_{args.exp_param}.pth" # the file to save QD-GNN parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_QD_GNN_log_{args.exp_mod}_{args.exp_param}"  # the file to save log during training
        args.save_thr = args.model_path + "/" + args.dataset + f"_QD_GNN_thr_{args.exp_mod}_{args.exp_param}"  # the file to save QD-GNN best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_QD_GNN_{args.exp_mod}_{args.exp_param}" # the file to save result
    args.save_processData = args.process_path + "/" + args.dataset + f"_QD_GNN_{args.exp_mod}_{args.exp_param}.pth"  # the file to save DataLoader of model
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_QD_GNN_recLabel" # the file to save recommendation label
    return args

def save_model_complete(model, args, trainloader=None):
    feat_num = None
    if trainloader is not None:
        _, _, _, feats, _ = trainloader
        feat_num = feats.shape[1]
    
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'nfeat': feat_num,
            'nhid': args.n_hid1,
            'nclass': args.n_classes,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epoch': args.epoch
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
        
        model = CS_GCN(
            nfeat=model_config['nfeat'],
            nhid=model_config['nhid'],
            nclass=model_config['nclass'],
            dropout=model_config['dropout']
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

def QD_GNN_train(args, trainloader, device):
    if trainloader is None:
        print("Empty trainloader provided, attempting to load pre-trained model...")
        model = load_model_complete(args, device)
        if model is not None:
            print("Loaded pre-trained model successfully")
            return model, 0, 0
        else:
            raise ValueError("No pre-trained model found and trainloader is empty. Cannot proceed.")
    
    train_data, valid_data, G, feats, norm_adj = trainloader
    feat_num = feats.shape[1]
    
    model = load_model_complete(args, device)
    if model is not None:
        print("Loaded pre-trained model, skipping training")
        return model, 0, 0
    
    # set model
    model = CS_GCN(nfeat=feat_num, nhid=args.n_hid1, nclass=args.n_classes, dropout=args.dropout)
    # train model
    max_gpu_memory = 0
    max_cpu_memory = 0
    
    save_model_complete(model, args, trainloader)
    
    model = model.to(device)
    model, max_cpu_memory, max_gpu_memory = train(train_data, valid_data, feats, norm_adj, model, args, trainloader)
    
    return model, max_cpu_memory, max_gpu_memory

def QD_GNN_valid(args, model, validloader, device):
    if os.path.exists(args.save_thr):
        with open(args.save_thr, "r") as file:
            best_thr = float(file.read().strip())
    else:
        if next(model.parameters()).device != device:
            model = model.to(device)
        best_thr = _eval_BFS(args, model, validloader)
        with open(args.save_thr, "w") as file:
            file.write(f"{best_thr:.2f}")
    return best_thr

def community_search(model, graph, best_thr, input, attr, norm_adj, feats):
    device = next(model.parameters()).device
    input = input.to(device)
    with torch.no_grad():
        output, _ = model.forward(input, norm_adj.expand(input.shape[0], -1, -1), feats.expand(input.shape[0], -1, -1), training=False)
        #  output, _ = model.forward(input, attr, norm_adj, feats, feats, training=False)
        predict = output.detach().cpu().numpy().flatten().tolist()
        queries = input.detach().cpu().numpy().flatten().tolist()
    community = community_BFS([queries], graph, [predict], best_thr)[0]
    return queries, community

def QD_GNN_RecGenerate(args, model_queries, model_gts, model, device):
    feature, G, node_id_map, N = read_feat_graph(args)
    onehot_queries = query_to_onehot(model_queries, N)
    onehot_gts = gt_to_onehot(model_gts, N)
    A = np.zeros((N, N), dtype=np.float32)
    for uid, vid in G.edges():
        A[uid, vid] = 1
        A[vid, uid] = 1
    norm_adj = normalize_adj(A)
    test_cur_attr = np.zeros((len(onehot_queries), feature.shape[1]))
    process_test = Unified_Dataset(samples_in=onehot_queries,
                                samples_out=onehot_gts,
                                samples_att=test_cur_attr)
    test_data = DataLoader(process_test, batch_size=1, shuffle=False)
    feats = torch.from_numpy(feature)
    norm_adj = torch.from_numpy(norm_adj)

    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()
    all_labels = []
    all_queries = []
    all_communities = []
    cpu_memory = []
    gpu_memory = []
    all_t = []
    # ensure the batch size is one
    feats = feats.to(device)
    norm_adj = norm_adj.to(device)
    model_thrs = []
    # Generate threshold values using numpy to avoid floating point precision issues
    thr_values = np.arange(args.min_thr, args.max_thr + args.thr_step/2, args.thr_step)
    # Round to avoid floating point precision issues
    thr_values = np.round(thr_values, decimals=2)
    
    for i_iter, batch in enumerate(test_data):
        for thr in thr_values:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            monitor = FunctionPerformanceMonitor()
            input, label, attr = batch
            t1 = time.time()
            metrics = monitor.monitor_function(community_search,model, G, thr, input, attr, norm_adj, feats)
            queries, community = metrics['result']
            cur_cpu_memory = metrics['max_memory_mb']
            cur_gpu_memory = get_gpu_memory()
            cpu_memory.append(cur_cpu_memory)
            gpu_memory.append(cur_gpu_memory)
            t2 = time.time()
            all_queries.append(queries)
            all_labels.append(set(np.nonzero(label.numpy().flatten())[0]))
            all_communities.append(community)
            all_t.append(t2-t1)
            model_thrs.append(thr)
    queries = [set(np.nonzero(row)[0]) for row in all_queries]
    str_queries = ['_'.join(str(G.nodes[node]['old_id']) for node in query) for query in queries]
    return G, all_communities, all_labels, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def QD_GNN_RecQuery(args, model_queries, model_gts, model, device, best_thr = None):
    feature, G, node_id_map, N = read_feat_graph(args)
    onehot_queries = query_to_onehot(model_queries, N)
    onehot_gts = gt_to_onehot(model_gts, N)
    A = np.zeros((N, N), dtype=np.float32)
    for uid, vid in G.edges():
        A[uid, vid] = 1
        A[vid, uid] = 1
    norm_adj = normalize_adj(A)
    test_cur_attr = np.zeros((len(onehot_queries), feature.shape[1]))
    process_test = Unified_Dataset(samples_in=onehot_queries,
                                samples_out=onehot_gts,
                                samples_att=test_cur_attr)
    test_data = DataLoader(process_test, batch_size=1, shuffle=False)
    feats = torch.from_numpy(feature)
    norm_adj = torch.from_numpy(norm_adj)

    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()
    all_queries = []
    all_communities = []
    cpu_memory = []
    gpu_memory = []
    all_t = []
    # ensure the batch size is one
    feats = feats.to(device)
    norm_adj = norm_adj.to(device)
    for i_iter, batch in enumerate(test_data):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        input, label, attr = batch
        t1 = time.time()
        metrics = monitor.monitor_function(community_search,model, G, best_thr[i_iter], input, attr, norm_adj, feats)
        queries, community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        t2 = time.time()
        all_queries.append(queries)
        all_communities.append(community)
        all_t.append(t2-t1)
    queries = [set(np.nonzero(row)[0]) for row in all_queries]
    str_queries = ['_'.join(str(G.nodes[node]['old_id']) for node in query) for query in queries]
    return all_communities, str_queries, all_t, cpu_memory, gpu_memory

def QD_GNN_test(args, model, testloader, best_thr, device):
    test_data, G, feats, norm_adj= testloader
    if next(model.parameters()).device != device:
        model = model.to(device)
    model.eval()
    all_labels = []
    all_queries = []
    all_communities = []
    cpu_memory = []
    gpu_memory = []
    all_t = []

    # ensure the batch size is one
    feats = feats.to(device)
    norm_adj = norm_adj.to(device)
    for i_iter, batch in enumerate(test_data):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        input, label, attr = batch
        t1 = time.time()
        metrics = monitor.monitor_function(community_search,model, G, best_thr, input, attr, norm_adj, feats)
        queries, community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(cur_cpu_memory)
        gpu_memory.append(cur_gpu_memory)
        t2 = time.time()
        all_queries.append(queries)
        all_labels.append(label.numpy().flatten().tolist())
        all_communities.append(community)
        all_t.append(t2-t1)
    labels = [set(np.nonzero(row)[0]) for row in all_labels]
    queries = [set(np.nonzero(row)[0]) for row in all_queries]

    str_queries = ['_'.join(str(G.nodes[node]['old_id']) for node in query) for query in queries]
    return G, all_communities, labels, str_queries, all_t, cpu_memory, gpu_memory

def QD_GNN_DataLoader(args, phase = 'all'):
    feature, G, node_id_map, N = read_feat_graph(args)
    A = np.zeros((N, N), dtype=np.float32)
    for uid, vid in G.edges():
        A[uid, vid] = 1
        A[vid, uid] = 1
    norm_adj = normalize_adj(A)
    d = feature.shape[1]
    processed_train, processed_valid, processed_test = None, None, None
    feature = torch.from_numpy(feature)
    norm_adj = torch.from_numpy(norm_adj)
    if phase == 'all' or phase == 'train':
        valid_cur_in, valid_cur_out = split_data(args, N, node_id_map, G, return_query = 'valid')
        valid_cur_attr = np.zeros((len(valid_cur_in), d))
        valid_data = Unified_Dataset(samples_in=valid_cur_in,
                                        samples_out=valid_cur_out,
                                        samples_att=valid_cur_attr)
        validloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        processed_valid = (validloader, G, feature, norm_adj)
        
        if args.train_size > 0:
            train_cur_in, train_cur_out = split_data(args, N, node_id_map, G, return_query = 'train')
            train_cur_attr = np.zeros((len(train_cur_in), d))
            train_data = Unified_Dataset(samples_in=train_cur_in,
                                    samples_out=train_cur_out,
                                    samples_att=train_cur_attr)                  
            trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

            processed_train = (trainloader, validloader, G, feature, norm_adj)
        else:
            processed_train = (None, None, G, feature, norm_adj)

    if phase == 'all' or phase == 'test':
        test_cur_in, test_cur_out = split_data(args, N, node_id_map, G, return_query = 'test')
        test_cur_attr = np.zeros((len(test_cur_in), d))
        test_data = Unified_Dataset(samples_in=test_cur_in,
                                    samples_out=test_cur_out,
                                    samples_att=test_cur_attr)
        testloader = DataLoader(test_data, batch_size=1, shuffle=False)
        processed_test = (testloader, G, feature, norm_adj)
    return processed_train, processed_valid, processed_test