from dgl import label_informativeness
import psutil
from torch.utils.data import DataLoader
from .dataset import Mydata, mycollate
from tqdm import tqdm
import time
from memory_count import FunctionPerformanceMonitor
from utils import *
import torch.nn.functional as F
import networkx as nx
import torch
import os
from .diffusion_model import DenoisingDiffusion, my_one_hot, PredictComsize, to_dense
import copy
from .locator import Locator
import builtins

def tensor_normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def fast_info_nce_loss_emd(pred, labels, T, graph_indices, k, m, tau=0.1,selected_graphs_ratio=0.2):
    n = pred.shape[0]
    pred = pred.view(n, -1)
    labels = labels.view(n, -1)
    T = T.view(n, -1)
    allindex = torch.where(T<1)[0]
    sample_index = allindex

    cos_sim_matrix = torch.mm(tensor_normalize(pred), tensor_normalize(labels).T)
    sample_graph_indices = graph_indices[sample_index]
    same_graph_mask = (graph_indices.unsqueeze(0) == sample_graph_indices.unsqueeze(1)).float()
    other_graph_mask = (graph_indices.unsqueeze(0) != sample_graph_indices.unsqueeze(1)).float()
    rand_same = torch.rand((len(sample_index), n), device=pred.device)
    rand_other = torch.rand((len(sample_index), n), device=pred.device)
    same_graph_sorted = torch.argsort(rand_same * same_graph_mask + (1 - same_graph_mask) * float('1e9'), dim=1)
    other_graph_sorted = torch.argsort(rand_other * other_graph_mask + (1 - other_graph_mask) * float('1e9'), dim=1)

    # Select top k and m indices for negatives
    same_graph_negatives = same_graph_sorted[:, :k]
    other_graph_negatives = other_graph_sorted[:, :m]
    negative_indices = torch.cat((same_graph_negatives, other_graph_negatives), dim=1)
    negative_pairs = cos_sim_matrix[sample_index.unsqueeze(1), negative_indices]
    positive_pairs = cos_sim_matrix[sample_index, sample_index].unsqueeze(1)

    positive_exp = torch.exp(positive_pairs / tau)
    negative_exp = torch.exp(negative_pairs / tau)
    denominator = positive_exp + negative_exp.sum(dim=1, keepdim=True)
    pair_loss = -torch.log(positive_exp / denominator).squeeze(1)
    weighted_loss = T[sample_index].view(-1) * pair_loss
    loss = weighted_loss.sum() / len(sample_index)
    return loss

def train_one_epoch(model, train_data, device, optimizer, Discriminative_model=None, pre_model=None):
    start = time.time()
    epoch_loss = 0.0
    model.train()
    for i, batch in enumerate(train_data):
        batch['graph'].ndata['x'] = my_one_hot(batch['graph'].ndata['x'], 1).float()
        num_nodes_per_graph = batch['graph'].batch_num_nodes()
        graph_indices = torch.repeat_interleave(torch.arange(len(num_nodes_per_graph)), num_nodes_per_graph).to(device)
        batch['graph'] = batch['graph'].to(device)
        dense_data, node_mask = to_dense(batch, batch['graph'].ndata['x'])
        X = dense_data.X.to(device)
        node_mask = node_mask.to(device)
        if model.args.guided and model.args.competitor != 'cond':
            pg, _ = to_dense(batch, batch['graph'].ndata['pg'].reshape(-1, 1))
            pg = pg.X.cpu()
        else:
            pg = None
        with torch.no_grad():
            if Discriminative_model is not None: # coarse community
                Discriminative_dense_data, Discriminative_node_mask = to_dense(batch, batch['graph'].ndata['pg'])
                Discriminative_X = Discriminative_dense_data.X.to(device)
                Discriminative_node_mask = Discriminative_node_mask.to(device)
                Discriminative_inputX = Discriminative_X[Discriminative_node_mask].float().view(-1,1)
                Discriminative_inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
                Discriminative_inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
                Discriminative_inputD = (Discriminative_inputD - model.args.mean_triangles) / model.args.std_triangles
                Discriminative_inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)

                Discriminative_predX = Discriminative_model(batch['graph'], Discriminative_inputX, Discriminative_inputS,
                                                            Discriminative_inputD, Discriminative_inputC)
                # batch['graph'].ndata['pg'] = Discriminative_predX.reshape(-1)
                pg, _ = to_dense(batch, torch.sigmoid(Discriminative_predX))
                batch['graph'].ndata['pg'] = pg.X[Discriminative_node_mask].reshape(-1)
                pg = pg.X.cpu()

            if pre_model is not None:
                old_X, node_mask, _ = pre_model.sample_batch(batch, Discriminative_model)
                pg = old_X.cpu()

        noisy_data = model.apply_noise(X, node_mask, pg)
        inputT = noisy_data['t'].repeat(1, noisy_data['X_t'].shape[1])[node_mask].float().view(-1, 1)
        inputX = noisy_data['X_t'][node_mask].float().view(-1, 1)
        inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
        inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
        inputD = (inputD - model.args.mean_triangles) / model.args.std_triangles
        inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)
        predX = model(batch['graph'], inputX, inputS, inputT, inputD, inputC)
        # print(predX.dtype,noisy_data['epsX'].dtype)
        loss = F.mse_loss(predX, noisy_data['epsX'][node_mask].to(device).float())
        if model.args.contrast_loss:
            noisy_data_sample = model.apply_noise(X, node_mask, pg)
            sample_inputT = noisy_data_sample['t'].repeat(1, noisy_data_sample['X_t'].shape[1])[node_mask].float().view(-1, 1)
            sample_inputX = noisy_data_sample['X_t'][node_mask].float().view(-1, 1)
            sample_emd = model.get_emd(batch['graph'], sample_inputX, inputS, sample_inputT, inputD, inputC)
            pred_emd = model.get_emd(batch['graph'], inputX, inputS, inputT, inputD, inputC)
            closs =  model.args.contrast_alpah*fast_info_nce_loss_emd(pred_emd, sample_emd, (1-torch.abs(sample_inputT-inputT)), graph_indices,
                                                             model.args.contrast_negative_sample_in_graph, model.args.contrast_negative_sample_out_graph,
                                                             model.args.contrast_tau,1).view(loss.shape)
            print('Contrast_loss',closs)
            loss += closs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.item()

    end = time.time()
    epoch_time = end - start
    return epoch_loss, epoch_time

def CommunityDF_parameter(input_args):
    args = copy.deepcopy(input_args)
    args.dropout = 0.0
    args.sg_max_size = 100
    args.batch_size = 150
    args.diffusion_steps = 30
    args.weight_decay = 0
    args.lr = 1e-4
    args.optim = 'AdamW'
    args.gatlayers = [4, 4, 4]
    args.dim = 256
    args.pos_enc_dim = 200
    args.method_names = 'all'
    args.prob_normal = 'minmax'
    args.bfs_pagerank = True
    args.sample_pagerank = 'normal'
    args.attn_pool = True
    args.alpha = 0.85
    args.guided = True
    args.ori_xc_coefficient = False
    args.sample_place = 'all'
    args.contrast_loss = True
    args.contrast_negative_sample_out_graph = 5
    args.contrast_negative_sample_in_graph = 5
    args.contrast_alpah = 0.1
    args.contrast_tau = 1
    args.locator_train_size = -1
    args.locator_valid_size = -1
    args.locator_shuffle_time = 3
    args.locator_dropout = 0
    args.locator_hiddensize = 64
    args.test_batch_size = 1
    args.pos_test_batch_size = 1
    args.loc_model = True
    args.competitor = None

    args.min_thr = -1
    args.max_thr = -1
    args.thr_step = -1

    if args.exp_mod == 5 and args.fix:
        args.save_model = args.model_path + "/" + args.dataset + f"_CommunityDF_1_0.pth"  # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_CommunityDF_log_1_0"  # the file to save log during training
        args.save_model_loc = args.model_path + "/" + args.dataset + f"_CommunityDFloc_1_0.pth"  # the file to save the parameter of locator model
        args.save_trainInfo_loc = args.train_path + "/" + args.dataset + f"_CommunityDFloc_log_1_0"  # the file to save log during training of locator model
        args.save_thr = args.model_path + "/" + args.dataset + f"_CommunityDF_thr_1_0"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_CommunityDF_{args.exp_mod}_{args.exp_param}_fix" # the file to save result
    else:
        args.save_model = args.model_path + "/" + args.dataset + f"_CommunityDF_{args.exp_mod}_{args.exp_param}.pth" # the file to save the parameter
        args.save_trainInfo = args.train_path + "/" + args.dataset + f"_CommunityDF_log_{args.exp_mod}_{args.exp_param}"  # the file to save log during training
        args.save_model_loc = args.model_path + "/" + args.dataset + f"_CommunityDFloc_{args.exp_mod}_{args.exp_param}.pth"  # the file to save the parameter of locator model
        args.save_trainInfo_loc = args.train_path + "/" + args.dataset + f"_CommunityDFloc_log_{args.exp_mod}_{args.exp_param}"  # the file to save log during training of locator model
        args.save_thr = args.model_path + "/" + args.dataset + f"_CommunityDF_thr_{args.exp_mod}_{args.exp_param}"  # the file to save the best threshold
        args.save_result = args.result_path + "/" + args.dataset + f"_CommunityDF_{args.exp_mod}_{args.exp_param}" # the file to save result
    args.save_processData = args.process_path + "/" + args.dataset + f"_CommunityDF_{args.exp_mod}_{args.exp_param}.pth"  # the file to save DataLoader of model
    args.save_recLabel = args.process_path  + "/" + args.dataset + f"_CommunityDF_recLabel" # the file to save recommendation label
    return args

def CommunityDF_DataLoader(args, phase = 'all'):
    # read graph and feature
    feature, G, node_id_map, N = read_feat_graph(args)
    if feature is not None:
        args.feat_dim = feature.shape[1]
    else:
        args.feat_dim = 0

    processed_train, processed_valid, processed_test = None, None, None
    if phase == 'all' or phase == 'train':
        # split data
        train_cur_in, train_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'train', input_train_size = args.default_train_size)
        valid_cur_in, valid_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'valid')

        # pack train data
        train_comms = deduplicate_communities_list(train_cur_out)
        num_communities = int((args.train_size/args.default_train_size)*len(train_comms))
        train_comms = [list(com) for com in train_comms][:num_communities]

        print(f"The number of samples for training:{num_communities}")
        if args.locator_train_size == -1:
            args.locator_train_size = int(num_communities / 5) # split 1/5 samples to train the locator
        assert args.locator_train_size <= num_communities, "the number of samples for locator must be less than the total samples"

        trainloader = None
        args.mean_triangles = 0.0
        args.std_triangles = 1e-6
        if len(train_comms[args.locator_train_size:]) > 0:
            traindata = Mydata(args.dataset, G, args.sg_max_size, train_comms[args.locator_train_size:],
                            pos_enc_dim=args.pos_enc_dim,
                            shuffle=True,
                            sample_pagerank=args.sample_pagerank,
                            bfs_pagerank=args.bfs_pagerank,
                            features=feature)
            if len(traindata.data) > 0:
                triangles = []
                for data in traindata:
                    triangles.extend(list(nx.triangles(data['sg']).values()))
                triangles = torch.tensor(triangles)
                args.mean_triangles = float(sum(triangles) / len(triangles))
                args.std_triangles = float((sum((x - args.mean_triangles) ** 2 for x in triangles) / len(triangles)) ** 0.5)
                trainloader = DataLoader(traindata, collate_fn=mycollate, batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=0)
            else:
                print("Warning: traindata is empty after Mydata processing, trainloader will be None")
        gcntraindata = None
        if args.locator_train_size > 0:
            gcntraindata = Mydata(args.dataset, G, args.sg_max_size,
                                train_comms[:args.locator_train_size] * args.locator_shuffle_time,
                                pos_enc_dim=args.pos_enc_dim,
                                shuffle=True,
                                sample_pagerank=args.sample_pagerank,
                                bfs_pagerank=args.bfs_pagerank,
                                features=feature, gcn_use=True)
            if len(gcntraindata.data) == 0:
                print("Warning: gcntraindata is empty after Mydata processing, gcntraindata will be None")
                gcntraindata = None

        # pack valid data
        valid_comms = deduplicate_communities_list(valid_cur_out)
        valid_comms = [list(com) for com in valid_comms]
        valid_num = len(valid_comms)
        if args.locator_valid_size == -1:
            args.locator_valid_size = max(int(valid_num / 5), 1) # split 1/5 samples to valid the locator
        assert args.locator_valid_size <= valid_num, "the number of samples for locator must be less than the total samples"
        validloader = None
        if len(valid_comms[args.locator_valid_size:]) > 0:
            validdata = Mydata(args.dataset, G, args.sg_max_size, valid_comms[args.locator_valid_size:],
                            pos_enc_dim=args.pos_enc_dim,
                            shuffle=True,
                            sample_pagerank=args.sample_pagerank,
                            bfs_pagerank=args.bfs_pagerank,
                            features=feature)
            if len(validdata.data) > 0:
                validloader = DataLoader(validdata, collate_fn=mycollate, batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=0)
            else:
                print("Warning: validdata is empty after Mydata processing, validloader will be None")
        gcnvaliddata = None
        if args.locator_valid_size > 0:
            gcnvaliddata = Mydata(args.dataset, G, args.sg_max_size,
                                valid_comms[:args.locator_valid_size] * args.locator_shuffle_time,
                                pos_enc_dim=args.pos_enc_dim,
                                shuffle=False,
                                sample_pagerank=args.sample_pagerank,
                                bfs_pagerank=args.bfs_pagerank,
                                features=feature, gcn_use=True)
            if len(gcnvaliddata.data) == 0:
                print("Warning: gcnvaliddata is empty after Mydata processing, gcnvaliddata will be None")
                gcnvaliddata = None

        processed_train = (trainloader, gcntraindata, validloader, gcnvaliddata)

    if phase == 'all' or phase == 'test':
        test_cur_in, test_cur_out = split_data(args, N, node_id_map, G, return_list = True, return_query = 'test')
        processed_test = (test_cur_in, test_cur_out, G, feature)

    return processed_train, processed_valid, processed_test

def eval_loss(model, valid_data, device, Discriminative_model=None, pre_model=None):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_data):
            batch['graph'].ndata['x'] = my_one_hot(batch['graph'].ndata['x'], 1).float()
            num_nodes_per_graph = batch['graph'].batch_num_nodes()
            graph_indices = torch.repeat_interleave(torch.arange(len(num_nodes_per_graph)), num_nodes_per_graph).to(device)
            batch['graph'] = batch['graph'].to(device)
            dense_data, node_mask = to_dense(batch, batch['graph'].ndata['x'])
            X = dense_data.X.to(device)
            node_mask = node_mask.to(device)
            if model.args.guided and model.args.competitor != 'cond':
                pg, _ = to_dense(batch, batch['graph'].ndata['pg'].reshape(-1, 1))
                pg = pg.X.cpu()
            else:
                pg = None
            with torch.no_grad():
                if Discriminative_model is not None: # coarse community
                    Discriminative_dense_data, Discriminative_node_mask = to_dense(batch, batch['graph'].ndata['pg'])
                    Discriminative_X = Discriminative_dense_data.X.to(device)
                    Discriminative_node_mask = Discriminative_node_mask.to(device)
                    Discriminative_inputX = Discriminative_X[Discriminative_node_mask].float().view(-1,1)
                    Discriminative_inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
                    Discriminative_inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
                    Discriminative_inputD = (Discriminative_inputD - model.args.mean_triangles) / model.args.std_triangles
                    Discriminative_inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)

                    Discriminative_predX = Discriminative_model(batch['graph'], Discriminative_inputX, Discriminative_inputS,
                                                                Discriminative_inputD, Discriminative_inputC)
                    # batch['graph'].ndata['pg'] = Discriminative_predX.reshape(-1)
                    pg, _ = to_dense(batch, torch.sigmoid(Discriminative_predX))
                    batch['graph'].ndata['pg'] = pg.X[Discriminative_node_mask].reshape(-1)
                    pg = pg.X.cpu()

                if pre_model is not None:
                    old_X, node_mask, _ = pre_model.sample_batch(batch, Discriminative_model)
                    pg = old_X.cpu()

            noisy_data = model.apply_noise(X, node_mask, pg)
            inputT = noisy_data['t'].repeat(1, noisy_data['X_t'].shape[1])[node_mask].float().view(-1, 1)
            inputX = noisy_data['X_t'][node_mask].float().view(-1, 1)
            inputS = batch['graph'].ndata['seed'].float().view(-1, 1)
            inputD = batch['graph'].ndata['triangles'].float().view(-1, 1)
            inputD = (inputD - model.args.mean_triangles) / model.args.std_triangles
            inputC = batch['graph'].ndata['clustering_coefficients'].float().view(-1, 1)
            predX = model(batch['graph'], inputX, inputS, inputT, inputD, inputC)
            # print(predX.dtype,noisy_data['epsX'].dtype)
            loss = F.mse_loss(predX, noisy_data['epsX'][node_mask].to(device).float())
            if model.args.contrast_loss:
                noisy_data_sample = model.apply_noise(X, node_mask, pg)
                sample_inputT = noisy_data_sample['t'].repeat(1, noisy_data_sample['X_t'].shape[1])[node_mask].float().view(-1, 1)
                sample_inputX = noisy_data_sample['X_t'][node_mask].float().view(-1, 1)
                sample_emd = model.get_emd(batch['graph'], sample_inputX, inputS, sample_inputT, inputD, inputC)
                pred_emd = model.get_emd(batch['graph'], inputX, inputS, inputT, inputD, inputC)
                closs =  model.args.contrast_alpah*fast_info_nce_loss_emd(pred_emd, sample_emd, (1-torch.abs(sample_inputT-inputT)), graph_indices,
                                                                 model.args.contrast_negative_sample_in_graph, model.args.contrast_negative_sample_out_graph,
                                                                 model.args.contrast_tau,1).view(loss.shape)
                print('Contrast_loss',closs)
                loss += closs
            epoch_loss += loss.data.item()
    return epoch_loss

def save_model_complete(locator, args, trainloader=None):
    model_info = {
        'locator_config': {
            'post_batch_size': getattr(args, 'post_batch_size', 30),
            'post_test_batch_size': getattr(args, 'post_test_batch_size', 1),
            'gcnlr': getattr(args, 'gcnlr', 1e-2),
            'gcn_hidden_size': locator.gcn_hidden_size,
            'gcn_dropout': locator.gcn_dropout
        },
        'locator_gcn_state_dict': locator.gcnmodel.state_dict(),
        'diffusion_model_state_dict': locator.pre_model.state_dict(),
        'model_config': {
            'dim': args.dim,
            'gatlayers': args.gatlayers,
            'dropout': args.dropout,
            'diffusion_steps': args.diffusion_steps,
            'feat_dim': args.feat_dim,
            'pos_enc_dim': args.pos_enc_dim,
            'competitor': args.competitor,
            'optim': args.optim,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'mean_triangles': args.mean_triangles,
            'std_triangles': args.std_triangles
        }
    }
    
    torch.save(model_info, args.save_model)
    print(f"Locator model (with diffusion model) saved to {args.save_model}")

def load_model_complete(args, device):
    if not os.path.exists(args.save_model):
        print(f"Model file {args.save_model} not found")
        return None
    
    try:
        model_info = torch.load(args.save_model, map_location=device)
        model_config = model_info['model_config']
        locator_config = model_info['locator_config']
        args.dim = model_config['dim']
        args.gatlayers = model_config['gatlayers']
        args.dropout = model_config['dropout']
        args.diffusion_steps = model_config['diffusion_steps']
        args.feat_dim = model_config['feat_dim']
        args.pos_enc_dim = model_config['pos_enc_dim']
        args.competitor = model_config['competitor']
        args.optim = model_config['optim']
        args.lr = model_config['lr']
        args.weight_decay = model_config['weight_decay']
        args.mean_triangles = model_config['mean_triangles']
        args.std_triangles = model_config['std_triangles']
        diffusion_model = DenoisingDiffusion(args, args.dim, args.dim, args.gatlayers, 
                                           device=device, dropout=args.dropout,
                                           diffusion_steps=args.diffusion_steps)
        
        if 'diffusion_model_state_dict' in model_info:
            diffusion_model.load_state_dict(model_info['diffusion_model_state_dict'])
        else:
            diffusion_model.load_state_dict(model_info['model_state_dict'])
        
        diffusion_model.to(device)
        
        from .locator import Locator
        locator = Locator(
            args=args,
            device=device,
            pre_model=diffusion_model,
            ics_model=None, 
            post_batch_size=locator_config['post_batch_size'],
            post_test_batch_size=locator_config['post_test_batch_size'],
            gcnlr=locator_config['gcnlr'],
            gcn_hidden_size=locator_config['gcn_hidden_size'],
            gcn_dropout=locator_config['gcn_dropout']
        )
        
        locator.gcnmodel.load_state_dict(model_info['locator_gcn_state_dict'])
        locator.gcnmodel.to(device)
        
        print(f"Locator model loaded from {args.save_model}")
        print(f"Diffusion model config: {model_config}")
        print(f"Locator config: {locator_config}")
        
        return locator
        
    except Exception as e:
        print(f"Error loading locator model: {e}")
        return None

def save_locator_complete(locator, args, trainloader=None):
    save_model_complete(locator, args, trainloader)
    print("Complete locator model saved successfully")

def load_locator_complete(args, device):
    locator = load_model_complete(args, device)
    if locator is not None:
        print("Complete locator model loaded successfully")
        return locator
    else:
        print("Failed to load complete locator model")
        return None

def CommunityDF_train(args, trainloader, device):
    ini_time = time.time()
    if trainloader is None:
        print("Empty trainloader provided, attempting to load pre-trained locator model...")
        locator = load_model_complete(args, device)
        if locator is not None:
            print("Loaded pre-trained locator model successfully")
            return locator, 0, 0
        else:
            raise ValueError("No pre-trained locator model found and trainloader is empty. Cannot proceed.")
    
    train_data, gcntraindata, valid_data, gcnvaliddata = trainloader

    locator = load_model_complete(args, device)
    if locator is not None:
        print("Loaded pre-trained locator model, skipping training")
        return locator, 0, 0
    
    model = DenoisingDiffusion(args, args.dim, args.dim, args.gatlayers, device=device, dropout=args.dropout,
                               diffusion_steps=args.diffusion_steps).to(device)
    max_cpu_memory = 0
    max_gpu_memory = 0
    # use pagerank score for coarse community
    Discriminative_model = None
    pre_model = None

    
    if not os.path.exists(args.save_model):
        if args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True,
                                          weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        print("starting training...")
        stopping_args = Stop_args(patience=args.patience, max_epochs=args.epoch)
        early_stopping = EarlyStopping(model, **stopping_args)
        all_loss = []
        
        if train_data is not None and valid_data is not None:
            best_loss = float('inf')
            best_model_state = None
            for epoch in range(args.epoch):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                monitor = FunctionPerformanceMonitor()
                metrics = monitor.monitor_function(train_one_epoch, model, train_data, device, optimizer, Discriminative_model, pre_model)
                train_loss, epoch_time = metrics['result']
                cur_cpu_memory = metrics['max_memory_mb']
                cur_gpu_memory = get_gpu_memory()
                max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
                max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
                valid_loss = eval_loss(model, valid_data, device, Discriminative_model, pre_model)
                all_loss.append(valid_loss)
                # save model
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model_state = model.state_dict().copy()
                avg_loss = float('inf') if len(train_data) == 0 else train_loss / len(train_data)
                avg_valid_loss = float('inf') if len(valid_data) == 0 else valid_loss / len(valid_data)
                # record information
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
                print(f"Loaded best diffusion model with validation loss: {best_loss:.6f}")
        else:
            print("train set for diffusion model is null")

    # Train locator model
    print('train locator')
    locator = Locator(args, device, model, None, post_test_batch_size= args.pos_test_batch_size, gcn_dropout=args.locator_dropout,
                      gcn_hidden_size=args.locator_hiddensize)
    loc_cpu_memory, loc_gpu_memory = locator.train(gcntraindata, gcnvaliddata)
    print(f"difussion gpu memory:{max_gpu_memory}, locator gpu memory:{loc_gpu_memory}")
    print(f"difussion cpu memory:{max_cpu_memory}, locator cpu memory:{loc_cpu_memory}")
    max_cpu_memory = max(loc_cpu_memory, max_cpu_memory)
    max_gpu_memory = max(loc_gpu_memory, max_gpu_memory)
    save_model_complete(locator, args, trainloader)
    print("Complete locator model saved after training")
    
    return locator, max_cpu_memory, max_gpu_memory

def CommunityDF_test(args, model, testloader, best_thr, device):
    # process test data
    test_cur_in, test_cur_out, G, feature = testloader
    test_comms = [list(com) for com in test_cur_out]
    labels = test_cur_out
    # labels = test_cur_out
    test_queries = single_query(test_cur_in)
    testdata = Mydata(args.dataset, G, args.sg_max_size, test_comms, mode='test',
                      pos_enc_dim=args.pos_enc_dim,
                      sample_pagerank=args.sample_pagerank, bfs_pagerank=args.bfs_pagerank,
                      features=feature, input_queries = test_queries)
    fail_Index = testdata.fail_Index
    test_dataloader = DataLoader(model.update(testdata), collate_fn=mycollate, batch_size=model.args.post_test_batch_size,
                                     shuffle=False,
                                     num_workers=0)
    # load model
    if next(model.gcnmodel.parameters()).device != device:
        model.gcnmodel = model.gcnmodel.to(device)
    if next(model.pre_model.parameters()).device != device:
        model.pre_model = model.pre_model.to(device)
    model.pre_model.eval()
    model.gcnmodel.eval()

    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    count = 0
    for batch in tqdm(test_dataloader):
        seed_new_id = batch['seed'][0]
        q = batch['rmapper'][0][seed_new_id]
        assert testdata.safe_query[count] == q
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(model.test, batch)
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(max(testdata.query_cpu[count],cur_cpu_memory))
        gpu_memory.append(cur_gpu_memory)
        if q not in community:
            community.add(q)
            print(f"[CommunityDF]query {G.nodes[q]['old_id']} is not in the community")
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1 + testdata.query_time[count])
        count += 1
    for i in fail_Index:
        communities.insert(i, {test_queries[i]})
        all_t.insert(i, 0.0)
        cpu_memory.insert(i, 0.0)
        gpu_memory.insert(i, 0.0)
    
    assert len(communities) == len(labels), f"the length of community and label is different"
    str_queries = [str(G.nodes[query]['old_id']) for query in test_queries]
    return G, communities, labels, str_queries, all_t, cpu_memory, gpu_memory

def CommunityDF_RecGenerate(args, model_queries, model_gts, model, device):
    # read graph and feature
    feature, G, node_id_map, N = read_feat_graph(args)
    # process test data
    test_cur_in = model_queries
    test_cur_out = model_gts
    labels = test_cur_out
    test_comms = [list(com) for com in test_cur_out]
    test_queries = single_query(test_cur_in)
    testdata = Mydata(args.dataset, G, args.sg_max_size, test_comms,  mode='test',
                      pos_enc_dim=args.pos_enc_dim,
                      sample_pagerank=args.sample_pagerank, bfs_pagerank=args.bfs_pagerank,
                      features=feature, input_queries = test_queries)
    fail_Index = testdata.fail_Index
    test_dataloader = DataLoader(model.update(testdata), collate_fn=mycollate, batch_size=model.args.post_test_batch_size,
                                     shuffle=False,
                                     num_workers=0)

    # load model
    if next(model.gcnmodel.parameters()).device != device:
        model.gcnmodel = model.gcnmodel.to(device)
    if next(model.pre_model.parameters()).device != device:
        model.pre_model = model.pre_model.to(device)
    model.pre_model.eval()
    model.gcnmodel.eval()

    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    count = 0
    for batch in tqdm(test_dataloader):
        seed_new_id = batch['seed'][0]
        q = batch['rmapper'][0][seed_new_id]
        assert testdata.input_queries[count] == q
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(model.test, batch)
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(max(testdata.query_cpu[count],cur_cpu_memory))
        gpu_memory.append(cur_gpu_memory)
        if q not in community:
            community.add(q)
            print(f"[CommunityDF]query {G.nodes[q]['old_id']} is not in the community")
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1 + testdata.query_time[count])
        count += 1
    for i in fail_Index:
        communities.insert(i, {test_queries[i]})
        all_t.insert(i, 0.0)
        cpu_memory.insert(i, 0.0)
        gpu_memory.insert(i, 0.0)
    
    assert len(communities) == len(labels), f"the length of community and label is different"
    str_queries = [str(G.nodes[query]['old_id']) for query in test_queries]
    model_thrs = [-1 for i in range(len(communities))]
    return G, communities, labels, str_queries, all_t, cpu_memory, gpu_memory, model_thrs

def CommunityDF_RecQuery(args, model_queries, model_gts, model, device, best_thr = None):
    # read graph and feature
    feature, G, node_id_map, N = read_feat_graph(args)
    # process test data
    test_cur_in = model_queries
    test_cur_out = model_gts
    test_comms = [list(com) for com in test_cur_out]
    labels = test_cur_out
    test_queries = single_query(test_cur_in)
    testdata = Mydata(args.dataset, G, args.sg_max_size, test_comms,  mode='test',
                      pos_enc_dim=args.pos_enc_dim,
                      sample_pagerank=args.sample_pagerank, bfs_pagerank=args.bfs_pagerank,
                      features=feature, input_queries = test_queries)
    fail_Index = testdata.fail_Index
    test_dataloader = DataLoader(model.update(testdata), collate_fn=mycollate, batch_size=model.args.post_test_batch_size,
                                     shuffle=False,
                                     num_workers=0)

    # load model
    if next(model.gcnmodel.parameters()).device != device:
        model.gcnmodel = model.gcnmodel.to(device)
    if next(model.pre_model.parameters()).device != device:
        model.pre_model = model.pre_model.to(device)
    model.pre_model.eval()
    model.gcnmodel.eval()

    communities = []
    all_t = []
    cpu_memory = []
    gpu_memory = []
    count = 0
    for batch in tqdm(test_dataloader):
        seed_new_id = batch['seed'][0]
        q = batch['rmapper'][0][seed_new_id]
        assert testdata.input_queries[count] == q
        t1 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        monitor = FunctionPerformanceMonitor()
        metrics = monitor.monitor_function(model.test, batch)
        community = metrics['result']
        cur_cpu_memory = metrics['max_memory_mb']
        cur_gpu_memory = get_gpu_memory()
        cpu_memory.append(max(testdata.query_cpu[count],cur_cpu_memory))
        gpu_memory.append(cur_gpu_memory)
        if q not in community:
            community.add(q)
            print(f"[CommunityDF]query {G.nodes[q]['old_id']} is not in the community")
        communities.append(community)
        t2 = time.time()
        all_t.append(t2 - t1 + testdata.query_time[count])
        count += 1
    for i in fail_Index:
        communities.insert(i, {test_queries[i]})
        all_t.insert(i, 0.0)
        cpu_memory.insert(i, 0.0)
        gpu_memory.insert(i, 0.0)
    
    assert len(communities) == len(labels), f"the length of community and label is different"
    str_queries = [str(G.nodes[query]['old_id']) for query in test_queries]
    return communities, str_queries, all_t, cpu_memory, gpu_memory