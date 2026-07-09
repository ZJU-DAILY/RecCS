#! usr/bin/python

import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
from collections import deque
import os
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
os.environ["OMP_NUM_THREADS"]="10"
os.environ["MKL_NUM_THREADS"]="10"
torch.set_num_threads(10) 

# import kcore
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from GFCN import QD_GCN,CS_GCN,SimpleCS_GCN,CS_GCN_NoF
# from pygcn import GFCN, GCNAtt, GCNAtt_my,ResDeepGCN,DenseDeepGCN,GCN_BN,ResDeepGCN_BN8,\
#     ResDeepGCN_BN16, DenseDeepGCN_BN16, ResDeepGCN_BN16_ATN,ResGCN_BN,Self_ResGCN_BN,ResGCN_BN_ADD,ResGCN_BN_Update
from earlystopping import EarlyStopping
# from SampleLoader_F import EmailDataset,SPcoraDataset,PhilDataset
from SampleLoader_F_1node import GraphDataset,WebKBDataset, Unified_Dataset
from torch.utils.data.dataset import Dataset
import numpy as np
import numpy.random as npr
import networkx as nx
# import metric
from metric import *



def train(args):
    # STEP1: read parameters
    n_hid1 = args.n_hid1
    n_hid2 = args.n_hid2
    n_expert = args.n_expert
    att_hid = args.att_hid
    dropout = args.dropout
    input_folder = args.input_data_folder
    model_dir = args.model_dir + "QD-GNN.pth"
    steps = args.steps
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    lr = args.learning_rate
    normalization = args.normalization
    ef = args.extra_feats
    verbose = args.verbose
    set_size = args.b
    k = args.k
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    lossCE = True
    method="QD-GCN"

    # STEP2: load data
    trainloader = DataLoader(Unified_Dataset(phase='train',file=args.dataset_name,method=method),
                             batch_size=batch_size, shuffle=True, sampler=None)
    evalloader = DataLoader(Unified_Dataset(phase='eval',file=args.dataset_name,method=method),
                            batch_size=1, shuffle=False, sampler=None)
    testloader = DataLoader(Unified_Dataset(phase='test',file=args.dataset_name,method=method),
                            batch_size=1, shuffle=False, sampler=None)

    # STEP3: set model
    for i_iter, batch in enumerate(trainloader):
        input, attr, adj, feat, label , Adj = batch
        max_dim=feat.shape[2]
        print(feat.shape)
        print("max_dim: " + str(max_dim))
        break

    model = QD_GCN(nfeat = max_dim, nhid = n_hid1, nclass= args.n_classes, dropout = dropout)
    print("QD_GCN")
    model.cuda()
    optimizer = optim.Adam(model.parameters(),
                            lr=lr,  weight_decay=args.weight_decay)


    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.8)

    #criterion = torch.nn.MultiLabelMarginLoss()
    #criterion = torch.nn.MSELoss()

    # if model exists, reload the model
    model_exists = os.path.isfile(model_dir)
    if model_exists:
        print(model_dir)
        checkpoint=torch.load(model_dir)['model_state_dict']
        state = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in state}
        state.update(checkpoint)
        model.load_state_dict(state)
    if verbose:
        print("training...")


    # STEP4: train  model
    ### train ####################################################################################################################################################
    t = time.time()

    if args.earlystopping > 0:
        early_stopping = EarlyStopping(patience=args.earlystopping, verbose=False)
    print('len of trainloader', len(trainloader))
    # assert 1<0
    # model.train()
    best_val_score = [0,0,0]
    test_score = [0,0,0]

    saveall=[]
    for epoch in range(args.epoch):
        save = []
        ret=False
        if (epoch % 10 == 9):
        # if (epoch % 10 == 2):
            # print("train")
            # _eval(model, trainloader,best_val_score)
            print("\nvalidation")
            save.append(epoch)
            ret=eval_and_test(model, model_dir, evalloader, testloader, best_val_score, test_score,save)
        model.train()
        adjust_lr_poly(optimizer, args.learning_rate, epoch, args.epoch)
        savet = []
        savelb = []
        # savelable = {}
        count = 0
        for i_iter, batch in enumerate(trainloader):
            input,attr, adj,feat, label ,Adj= batch
            input = input.float()#.unsqueeze(-1)
            attr = attr.float()  # .unsqueeze(-1)
            feat = feat.float()
            adj = adj.float()
            label = label.float()#.unsqueeze(-1)
            optimizer.zero_grad()
            input=input.cuda()
            attr=attr.cuda()
            adj=adj.cuda()
            feat=feat.cuda()
            label = label.cuda()
            output,xsave = model.forward(input,attr, adj,feat,feat, training=True)

            if lossCE:
                pos_lab = torch.zeros(label.shape).float()
                pos_lab=pos_lab.cuda()
                pos_lab[label>0.6] = 1
                pos_lab[label<0.6] = 1
                criterion = torch.nn.BCELoss(weight=pos_lab)
                loss = criterion(output, label)
            else:
                loss = iou_loss(output, label)
                # loss = iou_loss_merge(output, label)
                # loss = f1_loss(output, label)

            loss.backward()
            optimizer.step()

            if args.verbose and i_iter % 10 == 0:
                print('i_iter: {:04d}'.format(i_iter),
                    'epoch: {:04d}'.format(epoch),
                    'loss: {:^10}'.format(loss.item()),
                    # 'val loss: {:^10}'.format(val_loss.item()),
                    'cur_lr: {:^10}'.format(get_lr(optimizer)),
                    'time: {:.4f}s'.format(time.time() - t))

        if (epoch % 10 == 0):
            save.append(loss.item())
            save.append(time.time() - t)
            saveall.append(save)


    # STEP5: test model
    save=[]
    save.append(epoch)
    eval_and_test(model,model_dir, evalloader, testloader, best_val_score, test_score,save)

    save.append(loss.item())
    save.append(time.time() - t)
    saveall.append(save)
    # print(saveall)
    # np.savetxt('./data/facebook/Log/log{}_batch{}_hid{}_lr{}.txt'.format(ego, batch_size, n_hid1, lr), saveall, delimiter='\t', fmt='%f')
    # np.savetxt('./data/WebKB/Log/log{}_batch{}_hid{}_lr{}.txt'.format(file, batch_size, n_hid1, lr), saveall, delimiter='\t', fmt='%f')

    print(model_dir)
    print('Final Results')
    print('best val score (precision, recall, f1score) :', best_val_score[0], best_val_score[1], best_val_score[2])
    print('test score (precision, recall, f1score) :', test_score[0], test_score[1], test_score[2])


def search(args):
    # STEP1: set model
    testloader = DataLoader(Unified_Dataset(phase='test',file=args.dataset_name,method=args.method),
                            batch_size=1, shuffle=False, sampler=None)
    for i_iter, batch in enumerate(testloader):
        input, attr, adj, feat, label , Adj = batch
        max_dim=feat.shape[2]
        print(feat.shape)
        print("max_dim: " + str(max_dim))
        break
    model = QD_GCN(nfeat=max_dim, nhid=args.n_hid1, nclass=args.n_classes, dropout=args.dropout)

    # STEP2: load pretrained model
    model_dir = args.model_dir + "QD-GNN.pth"
    model_exists = os.path.isfile(model_dir)
    if model_exists:
        print(model_dir)
        checkpoint=torch.load(model_dir)['model_state_dict']
        state = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in state}
        state.update(checkpoint)
        model.load_state_dict(state)
    model.cuda()

    # STEP3: community search
    best_thr = 0.5
    for i_iter, batch in enumerate(testloader):
        input, attr, adj, feat, label , Adj = batch
        flag,community = communitySearch(model, input, attr, adj, feat, best_thr)
        if not flag:
            continue
        community_list = list(community)
        label = label[0]
        predicted_labels = np.zeros_like(label)
        for idx in community_list:
            predicted_labels[idx] = 1
        precision = precision_score(label, predicted_labels,
                                    average='macro')
        recall = recall_score(label, predicted_labels, average='macro')
        f1 = f1_score(label, predicted_labels, average='macro')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

def adjust_lr_poly(optimizer, base_lr, epoch, tot_epoch, power=0.9):
    lr = base_lr * (1-epoch*1./tot_epoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_exp(optimizer, base_lr, epoch, tot_epoch, power=0.1):
    lr = base_lr * (power**(epoch*1./tot_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _eval(model, evalloader):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    all_output_bin = []
    all_label_bin = []
    all_output_bin_04 = []
    all_label_bin_04 = []
    iou_thr = 0.35
    for i_iter, batch in enumerate(evalloader):
        if i_iter > 99:
            break
        input,attr, adj,feat, label,Adj= batch
        input = input.float()#.unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()#.unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        input=input.cuda()
        attr=attr.cuda()
        adj=adj.cuda()
        feat=feat.cuda()
        label = label
        output, xsave= model.forward(input,attr, adj,feat,feat, training=False)
        output = output.cpu()
        all_output = all_output + (output).view(-1).detach().numpy().tolist()
        all_label = all_label + (label).view(-1).detach().numpy().tolist()
        all_output_bin = all_output_bin + (output>0.5).view(-1).detach().numpy().tolist()
        all_label_bin = all_label_bin + (label>0.5).view(-1).detach().numpy().tolist()
        all_output_bin_04 = all_output_bin_04 + (output > iou_thr).view(-1).detach().numpy().tolist()
        all_label_bin_04 = all_label_bin_04 + (label > iou_thr).view(-1).detach().numpy().tolist()
        # print("output  label", output[:5], label[:5])
        # assert 1<0


    # print(all_output[:10])
    # print(all_label[:10])
    print('num  all, output label ', len(all_output), sum(all_output), sum(all_label))
    print('num  all, output_bin label_bin ', len(all_output), sum(all_output_bin), sum(all_label_bin))
    print('num  all, output_bin_0.4 label_bin_0.4 ', len(all_output), sum(all_output_bin_04), sum(all_label_bin_04))
    best_thr=0
    best_f1=0
    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin/100.

        precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                precision, recall, f1_score))
        if f1_score > best_f1:
            best_f1 = f1_score
            best_thr=iou_thr

            #
            # tem_dir = model_dir + ".tmp"
            # print('save weight to', tem_dir)
            # torch.save({
            # 	# 'step': ini_step + args.steps,
            # 	'model_state_dict': model.state_dict(),
            # 	# 'loss': loss.item()
            # }, tem_dir)

    if best_thr == 0.:
        best_thr=0.5
    precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label,best_thr)

    return precision, recall, f1_score,best_thr

    # print('score on val set:')
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1_score:', f1_score)
    # if better:
    # 	# 	save weight
    # 	tem_dir=model_dir+".tmp"
    # 	print('save weight to', tem_dir)
    # 	torch.save({
    # 		# 'step': ini_step + args.steps,
    # 		'model_state_dict': model.state_dict(),
    # 		# 'loss': loss.item()
    # 	}, tem_dir)

def _eval_save(model, evalloader):
    ### eval ###
    model.eval()
    save = []
    savelb = []
    # savelable = {}
    count = 0
    for i_iter, batch in enumerate(evalloader):
        # if i_iter > 99:
        #     break
        input,attr, adj,feat, label, Adj = batch
        input = input.float()#.unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()#.unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        input=input.cuda()
        attr=attr.cuda()
        adj=adj.cuda()
        feat=feat.cuda()
        # label = label.cuda()
        output,xsave = model.forward(input,attr, adj,feat,feat, training=False)

        save.append(xsave.cpu().data.numpy()[:, :, 0])
        key_label = tuple(label.data.numpy()[0, :, 0])

        # if key_label not in savelable:
        #     savelable[key_label] = count
        #     count += 1

        # savelb.append(savelable[key_label])

    save = np.concatenate(save, axis=0)
    savelb = np.array(savelb)[np.newaxis, :].astype(np.int32)
    print("save size", save.shape)
    print("savelb size", savelb.shape)
    np.savez("./citeseer_valid.npz", savelb, save)

def _test(model, testloader,best_thr):
    ### eval ###
    model.eval()
    all_output = []
    all_label = []
    count=0
    totaltime=0
    for i_iter, batch in enumerate(testloader):
        input,attr, adj,feat, label, Adj = batch
        input = input.float()#.unsqueeze(-1)
        attr = attr.float()  # .unsqueeze(-1)
        feat = feat.float()
        adj = adj.float()
        label = label.float()#.unsqueeze(-1)
        # assert 1<0, (input.shape, adj.shape)
        test1 = time.time()
        input=input.cuda()
        attr=attr.cuda()
        adj=adj.cuda()
        feat=feat.cuda()
        # label = label.cuda()
        output,xsave = model.forward(input,attr, adj,feat,feat, training=False)
        output = output.cpu()
        test2 = time.time()
        totaltime=totaltime+test2-test1
        print("test time, test num",test2-test1)
        all_output = all_output + (output).view(-1).detach().numpy().tolist()
        all_label = all_label + (label).view(-1).detach().numpy().tolist()
        count=count+1

    for iou_thr_bin in range(5, 96, 5):
        iou_thr = iou_thr_bin/100.
        precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label, iou_thr)
        print('iou_thr: %.2f, precision, recall, f1_score : %.3f %.3f %.3f' % (iou_thr,
                precision, recall, f1_score))

    print("test time, avg time",totaltime, 1000*totaltime/len(testloader))
    precision, recall, f1_score = calc_f1_score_iouthr(all_output, all_label,best_thr)

    return precision, recall, f1_score

def bfs_connected_nodes(start_idx, adj, output, best_thr):
    # shape of adj is 1*N*D
    visited = np.zeros(adj.shape[1], dtype=bool)
    queue = deque([start_idx])
    visited[start_idx] = True
    connected_nodes = []

    while queue:
        current_node = queue.popleft()
        if output[current_node] > best_thr:
            connected_nodes.append(current_node)

        for neighbor in range(adj.shape[1]):
            if adj[current_node, neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return connected_nodes

def communitySearch(model, input, attr, adj, feat, best_thr):
    model.eval()
    input = input.float()
    attr = attr.float()
    feat = feat.float()
    adj = adj.float()
    test1 = time.time()
    input=input.cuda()
    attr=attr.cuda()
    adj=adj.cuda()
    feat=feat.cuda()
    output,xsave = model.forward(input,attr, adj,feat,feat, training=False)
    output = output.cpu()
    query_indices = np.where(input.cpu().numpy() == 1)[1]
    id = query_indices[0]
    adj = adj.cpu().numpy()[0]
    output = output.detach().cpu().numpy()[0]
    print(output.shape)
    print(adj.shape)
    connected_nodes = set(bfs_connected_nodes(id, adj, output, best_thr))
    community = set()
    for idx in query_indices:
        if idx not in connected_nodes:
            return False, community
    test2 = time.time()
    print("query time", test2-test1)
    community = connected_nodes
    return True, community

def eval_and_test(model,model_dir, evalloader, testloader, best_val_score, test_score,save):
    # STEP1: evaluate quality and find the best thr
    precision, recall, f1_score,best_thr = _eval(model, evalloader)
    print("current best thr ",best_thr)
    print('current val score (precision, recall, f1score) :', precision, recall, f1_score)
    save.append(f1_score)
    ret=False

    # STEP2: store the best model
    if f1_score > best_val_score[2]:
        best_val_score[0] = precision
        best_val_score[1] = recall
        best_val_score[2] = f1_score
        ret=True

        # STEP3: save model
        # _eval_save(model, evalloader)
        # 	save weight
        tem_dir = model_dir
        print('save weight to', tem_dir)
        torch.save({
            # 'step': ini_step + args.steps,
            'model_state_dict': model.state_dict(),
            # 'loss': loss.item()
        }, tem_dir)
        test1=time.time()
        precision, recall, f1_score = _test(model, testloader, best_thr)
        test2=time.time()
        print("test time, test num",test2-test1,len(testloader))
        test_score[0] = precision
        test_score[1] = recall
        test_score[2] = f1_score


    print('current best val score (precision, recall, f1score) :', best_val_score[0], best_val_score[1],
          best_val_score[2])
    print('corresponding test score (precision, recall, f1score) :', test_score[0], test_score[1], test_score[2])

    return ret


if __name__ == "__main__":
    parser = ArgumentParser("gcn", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    # Model settings
    parser.add_argument('--method', type=str, default="QD-GCN",
                        help='selected method')
    parser.add_argument("--n_hid1", default=256, type=int, help ="first layer of GCN: number of hidden units") # options [64, 128, 256]
    parser.add_argument("--n_hid2", default=256, type=int, help ="second layer of GCN: number of hidden units") # options [64, 128, 256]
    parser.add_argument("--n_expert", default=256, type=int, help ="attention layer: number of experts") # options [16, 32, 64, 128]
    parser.add_argument("--att_hid", default=256, type=int, help ="attention layer: hidden units") # options [64, 128, 256]
    parser.add_argument("--model_dir", type=str, default="./GCN_model.pt")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--normalization", default="AugNormAdj",
        help="The normalization on the adj matrix.")

    # Training settings
    parser.add_argument("--batch_size", default=128, type=int) # options: [32, 64, 128]
    parser.add_argument("--steps", default=10000, type=int)  # options:  (1000, 2000, ... 40000)
    parser.add_argument("--learning_rate", default = 0.001, type=float) #options [1e-3, 1e-4]
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--earlystopping", type=int, default=0,
        help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")

    #Others
    parser.add_argument("--extra_feats", default=0, type=int,
        help="whether or not enable extra feats (e.g.,core num, etc.) 0 Disables/1 Enable")
    parser.add_argument("--input_data_folder", default="data/wv", help="Input data folder")
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--k", default=100, type=int, help = "the k core to be collesped") # options [20, 30, 40]
    parser.add_argument("--b", default=100, type=int, help = "the result set size")
    parser.add_argument("--n_classes", default=1, type=int, help = "the output classes number")
    parser.add_argument("--epoch", default=100, type=int, help = "training epoch")

    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--dim", default=1703, type=int, help="attribute dim")
    parser.add_argument('--model_dir', type=str, default='./pretrain_model/',
                        help='path to save model')
    parser.add_argument('--result_dir', type=str, default='./pretrain_result/',
                        help='path to save result')
    parser.add_argument('--query_dir', type=str, default='./query/',
                        help='path to save query')
    parser.add_argument('--data_path', type=str, default='./',
                        help='path to load data')

    # unused parameters
    '''
    parser.add_argument("--dev_data_file", default = "")
    parser.add_argument("--n_eval_data", default = 1000, type = int) # number of eval data to generate/load
    parser.add_argument('--lradjust',action='store_true', default=False, 
        help = 'Enable leraning rate adjust.(ReduceLROnPlateau)')
    parser.add_argument("--debug_samplingpercent", type=float, default=1.0, 
        help="The percent of the preserve edges (debug only)")
    '''
    args = parser.parse_args()
    # path of this model
    work_path = os.path.dirname(sys.argv[0])
    args.work_path = work_path
    print("args.work_path_path: " + args.work_path)

    # root path of this project
    args.data_path = os.path.dirname(os.path.dirname(os.path.dirname(work_path))) + "/data/datasets"
    print("arg.data_path: " + args.data_path)

    # path of some pretrained or sampled results
    args.model_dir = work_path + '/' + 'pretrain_model/'
    os.makedirs(args.model_dir, exist_ok=True)
    print("args.model_dir: " + args.model_dir)
    args.result_dir = work_path + '/' + 'pretrain_result/'
    os.makedirs(args.result_dir, exist_ok=True)
    print("args.result_dir: " + args.result_dir)
    args.query_dir = work_path + '/' + 'query/'
    os.makedirs(args.query_dir, exist_ok=True)
    print("args.query_dir: " + args.query_dir)

    # train(args)
    search(args)

