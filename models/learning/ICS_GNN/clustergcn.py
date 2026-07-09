import torch
import random
import numpy as np
from tqdm import trange

from memory_count import FunctionPerformanceMonitor
from .layers import StackedGCN
from sklearn import metrics
import time
import os
from utils import Stop_args, EarlyStopping, get_gpu_memory
import tracemalloc
import builtins
class ClusterGCNTrainer(object):
    def __init__(self, clustering_machine, device):
        self.args =  clustering_machine.args
        self.clustering_machine = clustering_machine
        self.device = device
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        predictions = self.model(edges, features)
        #print(self.model.embeddings.size())
        average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        #average_loss = torch.nn.functional.cross_entropy(predictions[train_nodes], target[train_nodes])
        if self.clustering_machine.rankloss ==1:
            softmax_prediction = torch.nn.functional.softmax(predictions, dim=1)
            pos = softmax_prediction[self.clustering_machine.posforrank][:,1]
            neg = softmax_prediction[self.clustering_machine.negforrank][:,1]
            target = [1] * len(self.clustering_machine.posforrank)
            target = torch.LongTensor(target).to(self.device)
            rank_loss = torch.nn.functional.margin_ranking_loss(pos, neg, target)
            average_loss = average_loss + rank_loss
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        with torch.no_grad():
            prediction = self.model(edges, features)
        prediction = prediction[test_nodes,:]
        return prediction, target
    
    def train_one_epoch(self):
        start_time = time.time()
        self.model.train()
        epoch_loss = 0.0
        random.shuffle(self.clustering_machine.clusters)
        self.node_count_seen = 0
        self.accumulated_training_loss = 0
        for cluster in self.clustering_machine.clusters:
            self.optimizer.zero_grad()
            batch_average_loss, node_count = self.do_forward_pass(cluster)
            batch_average_loss.backward()
            self.optimizer.step()
            average_loss = self.update_average_loss(batch_average_loss, node_count)
            epoch_loss = epoch_loss + average_loss
        end_time = time.time()
        epoch_time = end_time - start_time
        return epoch_loss, epoch_time

    def train_test_community(self):
        """
        Training a model.
        """
        ### Train Phrase
        train_gpu_memory = 0
        train_cpu_memory = 0
        train_start = time.time()
        # print("Training started")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        stopping_args = Stop_args(patience=self.args.patience, max_epochs=self.args.epoch)
        early_stopping = EarlyStopping(self.model, **stopping_args)
        all_loss = []
        if self.args.ics_train_ratio > 0:
            for epoch in range(self.args.epoch):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                monitor = FunctionPerformanceMonitor()
                rsl = monitor.monitor_function(self.train_one_epoch)
                train_loss, epoch_time = rsl['result']
                all_loss.append(train_loss)
                cur_cpu_memory = rsl['max_memory_mb']
                cur_gpu_memory = get_gpu_memory()
                train_gpu_memory = max(train_gpu_memory, cur_gpu_memory)
                train_cpu_memory = max(train_cpu_memory, cur_cpu_memory)
                # early stop
                avg_loss = float('inf') if len(self.clustering_machine.clusters) == 0 else train_loss / len(self.clustering_machine.clusters)
                if self.args.early_stopping != 0 and early_stopping.simple_check(all_loss):
                    break
        train_end = time.time()
        train_time = train_end - train_start
        model_param = builtins.sum(p.numel() for p in self.model.parameters())
        torch.save(self.model.state_dict(), self.args.save_model)
        model_size = os.path.getsize(self.args.save_model) / 1024 / 1024
        os.remove(self.args.save_model)

        ### Test Phrase
        test_begin = time.time()
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)

        self.predictions = np.concatenate(self.predictions).argmax(1)
        score = metrics.f1_score(self.targets, self.predictions, average="micro")

        softmax_prediction=torch.nn.functional.softmax(prediction, dim =1)
        softmax_prediction = softmax_prediction.data[:,1]

        test_end = time.time()
        test_time = test_end - test_begin

        return softmax_prediction, self.predictions, score, train_time, test_time, model_size, model_param, train_gpu_memory, train_cpu_memory






