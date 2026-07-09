from torch.utils.data import DataLoader
from memory_count import FunctionPerformanceMonitor
from .dataset import mycollate
from tqdm import tqdm
import dgl
from .CommunityDF_utils import f1_score_,eval_bimatching_f1, set_seed
from .gat import GCN
import torch
import torch.nn.functional as F
from .getcomm import heap_com_threshold,heap_com_threshold_nonc,heap_com_topk,heap_com
from utils import *
import time

def get_com(args,seeds,sg,probx,batch,method_name,predsize):
    comms = []
    com_method, threshold = method_name.split('+')
    threshold = float(threshold)
    # print(com_method,threshold)
    if com_method == 'heap_com_threshold':
        sg_coms = [heap_com_threshold(int(seed), sg, probx, size)for seed,size in zip(seeds,predsize)]
    elif com_method == 'heap_topk':
        sg_coms = [heap_com_topk(int(seed), sg, probx, size) for seed,size in zip(seeds,predsize)]
    elif com_method == 'heap':
        sg_coms = [heap_com(int(seed), sg, probx) for seed in seeds]
    elif com_method=='heap_com_threshold_nonc':
        sg_coms = [heap_com_threshold_nonc(int(seed), sg, probx, threshold,list(batch['rmapper'][index].keys()),int(batch['sg_size'][index])) for index,seed in enumerate(seeds) ]
    assert len(sg_coms) == len(batch['rmapper'])
    coms = []
    for index in range(len(sg_coms)):
        com = [batch['rmapper'][index][int(node - batch['sg_size'][index])] for node in sg_coms[index]]
        coms.append(com)
    for com in coms:
        if len(com):
            comms.append(com)
    return comms

class Locator():
    def __init__(self,args,device,pre_model,ics_model,post_batch_size=30,post_test_batch_size=1,gcnlr=1e-2,gcn_hidden_size=16,gcn_dropout=0.0,log=""):
        self.args = args
        args.post_batch_size = post_batch_size
        args.post_test_batch_size = post_test_batch_size
        self.pre_model = pre_model
        self.pre_model.eval()
        self.ics_model = ics_model
        if self.ics_model is not None:
            self.ics_model.eval()
        self.loader = {}
        self.gcn_dropout =gcn_dropout
        self.gcn_hidden_size = gcn_hidden_size
        self.log = log
        self.train_time = 0
        self.traindata = None
        self.validdata = None
        self.testdata = None
        self.args.gcnlr =gcnlr
        self.gcnmodel = GCN(2, gcn_hidden_size,1, gcn_dropout).to(device)
        self.device = device

    def eval_loss(self):
        self.gcnmodel.eval()
        epoch_loss = 0.0
        for batch in (self.loader['valid']):
            batch['graph'] = batch['graph'].to(self.device)
            predX = batch['graph'].ndata['probx']
            predX = torch.hstack([predX, batch['graph'].ndata['seed'].view(-1, 1)])
            predC = self.gcnmodel(batch['graph'], predX, batch['graph'].ndata['seed'] == 1)
            label = torch.tensor(batch['best_value']).view(-1, 1).to(self.device).float()
            loss = F.mse_loss(predC, label)
            epoch_loss += loss.item()
        return epoch_loss

    def train_one_epoch(self, optimizer):
        self.gcnmodel.train()
        start = time.time()
        epoch_loss = 0.0
        for batch in (self.loader['train']):
            optimizer.zero_grad()
            batch['graph'] = batch['graph'].to(self.device)
            predX = batch['graph'].ndata['probx']
            predX = torch.hstack([predX, batch['graph'].ndata['seed'].view(-1, 1)])
            predC = self.gcnmodel(batch['graph'], predX, batch['graph'].ndata['seed'] == 1)
            label = torch.tensor(batch['best_value']).view(-1, 1).to(self.device).float()
            loss = F.mse_loss(predC, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        end = time.time()
        epoch_time = end - start
        return epoch_loss, epoch_time

    def train(self, traindata, validdata):
        max_cpu_memory = 0
        max_gpu_memory = 0
        ini_time = time.time()
        optimizer = torch.optim.AdamW(self.gcnmodel.parameters(), lr=self.args.gcnlr)
        stopping_args = Stop_args(patience=self.args.patience, max_epochs=self.args.epoch)
        early_stopping = EarlyStopping(self.gcnmodel, **stopping_args)
        all_loss = []
        if traindata is not None and validdata is not None:
            self.traindata = self.update(traindata) # generate labels for training
            self.loader['train'] = DataLoader(traindata, collate_fn=mycollate, batch_size=min(len(self.traindata), self.args.post_batch_size), shuffle=True,
                                         num_workers=0, drop_last=False)
            self.validdata = self.update(validdata)
            self.loader['valid'] = DataLoader(validdata, collate_fn=mycollate,
                                              batch_size=min(len(self.validdata), self.args.post_batch_size),
                                              shuffle=False,
                                              num_workers=0, drop_last=False)
            best_loss = float('inf')
            best_model_state = None
            for epoch in tqdm(range(self.args.epoch)):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                monitor = FunctionPerformanceMonitor()
                metrics = monitor.monitor_function(self.train_one_epoch, optimizer)
                train_loss, epoch_time = metrics['result']
                cur_cpu_memory = metrics['max_memory_mb']
                cur_gpu_memory = get_gpu_memory()
                max_gpu_memory = max(max_gpu_memory, cur_gpu_memory)
                max_cpu_memory = max(max_cpu_memory, cur_cpu_memory)
                valid_loss = self.eval_loss()
                all_loss.append(valid_loss)
                # save model
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model_state = self.gcnmodel.state_dict().copy()
                avg_loss = float('inf') if len(self.loader['train']) == 0 else train_loss / len(self.loader['train'])
                avg_valid_loss = float('inf') if len(self.loader['valid']) == 0 else valid_loss / len(self.loader['valid'])
                # record information
                sum_time = time.time() - ini_time
                write_train_info(self.args.save_trainInfo_loc, epoch, epoch_time, sum_time, cur_cpu_memory, cur_gpu_memory, avg_loss, avg_valid_loss)
                if (epoch + 1) % self.args.save_per_epoch == 0:
                    print(f"Epoch [{epoch + 1}/{self.args.epoch}], Time: {epoch_time:.6f}s, "
                          f"CPU Memory: {cur_cpu_memory:.6f} MB, "
                          f"GPU Memory: {cur_gpu_memory:.6f} MB, "
                          f"Train Loss: {avg_loss:.6f}, "
                          f"Valid Loss: {avg_valid_loss:.6f}")
                # early stop
                if self.args.early_stopping != 0 and early_stopping.simple_check(all_loss):
                    break
            
            if best_model_state is not None:
                self.gcnmodel.load_state_dict(best_model_state)
                print(f"Loaded best locator model with validation loss: {best_loss:.6f}")
        else:
            print("train set for locator model is null")
        return max_cpu_memory, max_gpu_memory

    def test(self, batch):
        with torch.no_grad():
            batch['graph'] = batch['graph'].to(self.device)
            predX = batch['graph'].ndata['probx']
            predX = torch.hstack([predX,batch['graph'].ndata['seed'].view(-1,1)])
            predC = self.gcnmodel(batch['graph'], predX, batch['graph'].ndata['seed']==1)
            presize = predC.cpu()
            sg = batch['graph'].cpu().to_networkx()
            seeds = torch.tensor(batch['seed']) + batch['sg_size'][:-1]
            com = get_com(self.args, seeds, sg, batch['graph'].ndata['probx'].cpu(), batch, 'heap_com_threshold+0', presize)
        return set(com[0])

    def update(self,dataset):
        ics_train_loader = DataLoader(dataset, collate_fn=mycollate, batch_size=self.args.post_batch_size, shuffle=False,
                                      num_workers=0)
        idx = 0
        for batch in ics_train_loader:
            probx = self.get_df_prob(batch)
            batch['graph'].ndata['probx'] = probx
            bg = dgl.unbatch(batch['graph'])
            for g in bg:
                # print(g)
                dataset[idx]['dgl_graph'].ndata['probx'] = g.ndata['probx'].cpu()
                idx += 1
        for idx, tmpdata in tqdm(enumerate(dataset)):
            seed = tmpdata['seed']
            sg = tmpdata['sg']
            probx = tmpdata['dgl_graph'].ndata['probx']
            size = sum(tmpdata['labels'])
            graph_size = len(tmpdata['labels'])
            # sg_coms = [heap_com_threshold(int(seed), sg, probx, threshold) for seed in seeds]
            best_k, best_f1, best_value = 0, 0, 0
            best_new_labels = []
            for size_bound in range(-10, 10):
                if size + size_bound >= 3:
                    sg_com = heap_com_topk(int(seed), sg, probx, size + size_bound)
                    new_labels = [1 if i in sg_com else 0 for i in range(graph_size)]
                    new_sg_com = [tmpdata['rmapper'][n] for n in sg_com]
                    f1, _, _, k = f1_score_([new_sg_com], [tmpdata['com']])
                    minvalue = min(probx[sg_com])
                    # print(f1,k,minvalue)
                    if best_f1 < f1:
                        best_f1 = f1
                        best_k = k
                        best_value = minvalue
                        best_new_labels = new_labels
            dataset[idx]['best_k'] = int(best_k)
            dataset[idx]['best_value'] = best_value
            dataset[idx]['best_new_labels'] = best_new_labels
        return dataset

    def get_df_prob(self,batch):
        self.pre_model.eval()
        with torch.no_grad():
            X, node_mask, _ = self.pre_model.sample_batch(batch, self.ics_model)
            #print(ics_inputX.shape)
            if self.args.prob_normal == 'softmax':
                X = X[node_mask]
                X = torch.sigmoid(X)
            elif self.args.prob_normal == 'minmax':
                for i in range(len(X)):
                    X[i] = (X[i] - min(X[i])) / (max(X[i]) - min(X[i]))
                X = X[node_mask]
        return X


