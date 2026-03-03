import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, CosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import time
import copy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import math

T = 2
EPSILON = 1e-8

class Dual_E(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if self.args["cosine"]:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = CosineIncrementalNet(args, True)
            else:
                self._network = CosineIncrementalNet(args, False)
        else:
            if self.args["dataset"] == "cub200" or self.args["dataset"] == "cars":
                self._network = IncrementalNet(args, True)
            else:
                self._network = IncrementalNet(args, False)

        self._protos = []
        self._covs = []
        self._projectors = []
        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_milestones =args['init_milestones']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.milestones = args['milestones']
        self.lrate_decay = args['lrate_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']
        self.w_kd = args['w_kd']
        self.use_past_model = args['use_past_model']
        self.save_model = args['save_model']
        self.model_dir = args['model_dir']
        self.dataset = args['dataset']
        self.init_cls = args['init_cls']
        self.increment = args['increment']
        self._process_id = args['process_id']
        ## 
        self.rank_prop =  args.get('rank_prop', 0.6)
        self.w_expert = args.get('w_expert', 'adapt')
        self.expert_num = args.get('expert_num', 5)
        self.w_proto = args.get('w_proto', 1)
        self.rout_T = args.get('rout_T', 1)
        self.eps = args.get('eps', 0.5)
        self.noise_std = args.get('noise_std', 0.02)
        self.epoch_drift = args.get('epoch_drift', 10)
        self.use_drift_compensation = args.get('use_drift_compensation', True)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        if self.save_model:
            path = self.model_dir + "{}/{}".format(self.dataset, self.args['seed'])
            if not os.path.exists(path):
                os.makedirs(path)
            self.save_checkpoint("{}/{}_{}".format(path, self.init_cls,self.increment))


    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args["cosine"]:
            self._network.update_fc(self._total_classes, self._cur_task)
        else:
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        self.shot = None
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train",mode="train",)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        if self._old_network is not None:
            self._old_network.to(self._device)
        model_dir = "{}/{}/{}/{}_{}_{}.pkl".format(self.args["model_dir"],self.args["dataset"], self.args['seed'], self.args["init_cls"],self.args["increment"],self._cur_task)
        if self._cur_task == 0:
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.init_milestones, gamma=self.init_lr_decay)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            if self.use_past_model and os.path.exists(model_dir):
                self._network.load_state_dict(torch.load(model_dir)["model_state_dict"], strict=True)
                self._network.to(self._device)
            else:
                self._network.to(self._device)
                optimizer = optim.SGD(self._network.parameters(), lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)
                self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if self.use_drift_compensation:
                self._update_memory(train_loader)
        self._build_protos()
    

    def _train_dual_estimator(self, train_loader, k=5):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network

        class LowRank_DriftEstimator(nn.Module):
            def __init__(self, in_features, out_features, rank=256):
                super().__init__()
                self.U = nn.Parameter(torch.empty(in_features, rank))
                self.V = nn.Parameter(torch.empty(rank, out_features))
                nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
                # self.bias = nn.Parameter(torch.zeros(out_features))

            def forward(self, x):
                weight  = (self.U @ self.V).T  # [out, in]
                return F.linear(x, weight)  # , self.bias
            
            def get_weight(self):
                return self.U.detach(), self.V.detach()

        MoE_T = self.rout_T
        class MoE_DriftEstimator(nn.Module):
            def __init__(self, in_features, out_features, k):
                super().__init__()
                self.Q_list = nn.ParameterList([
                    nn.Parameter(torch.empty(in_features, out_features)) for _ in range(k)
                ])
                for mat in self.Q_list:
                    nn.init.kaiming_uniform_(mat, a=math.sqrt(5))
                self.keys = None
                self.temperature = MoE_T

            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # [1, D]
                # dists = torch.cdist(F.normalize(x, dim=-1), F.normalize(self.keys, dim=-1))  # [N,k]
                # weights = F.softmax(-dists / self.temperature, dim=1)  # [N, k]
                x_norm = F.normalize(x, dim=-1)              # [N, d]
                keys_norm = F.normalize(self.keys, dim=-1)   # [k, d]
                cos_sim = torch.matmul(x_norm, keys_norm.T)  # [N, k]
                weights = F.softmax(cos_sim / self.temperature, dim=1)  # [N, k]
                outputs = []
                for i, Q in enumerate(self.Q_list):
                    # [N,D] @ [D,D] -> [N,D]
                    expert_out = x @ Q
                    outputs.append(expert_out.unsqueeze(2))  # [N,D,1]

                expert_outputs = torch.cat(outputs, dim=2)
                out = torch.sum(expert_outputs * weights.unsqueeze(1), dim=2)

                if out.size(0) == 1:
                    return out.squeeze(0), cos_sim.squeeze(0)
                return out, cos_sim
            
            def get_weight(self):
                return self.Q.detach()
            
        MoE_estimator = MoE_DriftEstimator(self.feature_dim, self.feature_dim, k).to(self._device)
        LowRank_estimator = LowRank_DriftEstimator(self.feature_dim, self.feature_dim, max(1,int(self.feature_dim * self.rank_prop))).to(self._device)
        for name, param in LowRank_estimator.named_parameters():
            param.requires_grad = False
        for name, param in MoE_estimator.named_parameters():
            param.requires_grad = False
        MoE_estimator.eval()
        LowRank_estimator.eval()
        _network.eval()
        self._old_network.eval()
        self._init_keys_with_kmeans(MoE_estimator, train_loader, self._device, k)

        prog_bar = tqdm(range(self.epoch_drift), colour='#CADDC6', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        
        orig_proto = []
        for cls_index in range(0, self._known_classes):
            orig_proto.append(torch.tensor(self._protos[cls_index]).to(self._device).to(torch.float32))
        P = torch.stack(orig_proto).to(self._device) # orig_proto
        train_MoE = True
        with torch.no_grad():
            # ==============================Train MoE_DriftEstimator ========================
            X1TX1_list = [torch.eye(self.feature_dim).to(self._device) * 1e-9 for _ in range(k)]
            X1TX2_list = [0 for _ in range(k)]
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                inputs = self.pixel_fusion_augment(inputs, 4)
                X1 = self._old_network(inputs)["features"].detach()
                X2 = _network(inputs)["features"].detach()
                dists = torch.cdist(F.normalize(X1, dim=-1), F.normalize(MoE_estimator.keys, dim=-1))  # [N,k]
                cluster_ids = torch.argmin(dists, dim=1)
                masks = [(cluster_ids == i) for i in range(k)]
                X1_list = [X1[masks[i]] for i in range(k)]
                X2_list = [X2[masks[i]] for i in range(k)]
                for idx, item in enumerate(X1TX1_list):
                    X1TX1_list[idx] += X1_list[idx].T @ X1_list[idx]
                    X1TX2_list[idx] += X1_list[idx].T @ X2_list[idx]
            Q_list = [torch.inverse(X1TX1_list[i]) @ (X1TX2_list[i]) for i in range(k)]
            for i in range(k):
                MoE_estimator.Q_list[i].data.copy_(Q_list[i].float())

            # ==============================Train LowRank_estimator ========================
            for epoch in prog_bar:
                X1TX1 = torch.eye(self.feature_dim).to(self._device) * 1e-9
                X1TX2VT = 0
                U, V = LowRank_estimator.get_weight()
                N=0
                for _, inputs, targets in train_loader:
                    N += inputs.shape[0]
                    inputs = inputs.to(self._device)
                    X1 = self._old_network(inputs)["features"].detach() # X_old
                    X2 = _network(inputs)["features"].detach()  #X_new
                    X1TX1 += X1.T @ X1
                    X1TX2VT += X1.T @ X2 @ V.T

                inv1 = torch.inverse(X1TX1 + self.w_proto * P.T @ P)
                inv2 = torch.inverse(V @ V.T)
                U = inv1 @ (X1TX2VT  + self.w_proto * P.T @ P @ V.T) @ inv2
                LowRank_estimator.U = torch.nn.parameter.Parameter(U.float())
                UTX1TX1U = torch.eye(U.T.shape[0]).to(self._device) * 1e-9
                UTX1TX2 = 0
                UTPTPU = 0
                UTPTP = 0
                for _, inputs, targets in train_loader:
                    inputs = inputs.to(self._device)
                    X1 = self._old_network(inputs)["features"].detach() # X_old
                    X2 = _network(inputs)["features"].detach()  #X_new
                    UTX1TX1U += U.T @ X1.T @ X1 @ U
                    UTX1TX2 += U.T @ X1.T @ X2
                    UTPTPU += U.T @ P.T @ P @ U
                    UTPTP += U.T @ P.T @P
                inv = torch.inverse(UTX1TX1U + self.w_proto * UTPTPU)
                V = inv @ (UTX1TX2 + self.w_proto * UTPTP)
                LowRank_estimator.V = torch.nn.parameter.Parameter(V.float())
                
                info = "P{}: Task {}, Epoch {}/{} => ".format(
                    self._process_id,
                    self._cur_task, 
                    epoch + 1, 
                    self.epoch_drift,
                )
                prog_bar.set_description(info)
        return MoE_estimator, LowRank_estimator


    def _update_memory(self, train_loader):
        MoE_estimator, LowRank_estimator = self._train_dual_estimator(train_loader, int(self.expert_num))
        MoE_estimator.eval()
        LowRank_estimator.eval()
        proto_current_task = []
        cov_prime = torch.zeros(self.feature_dim, self.feature_dim).to(self._device)

        for cls_index in range(0, self._known_classes):
            lowR_update = LowRank_estimator(torch.tensor(self._protos[cls_index]).to(self._device).to(torch.float32))
            upd_MoE, cos_sim = MoE_estimator(torch.tensor(self._protos[cls_index]).to(self._device).to(torch.float32))
            max_sim, max_idx = torch.max(cos_sim, dim=0)
            if self.w_expert == 'adapt':
                adapt_expert = max_sim.item()
                # print(adapt_expert)
                self._protos[cls_index] = lowR_update.detach().cpu().numpy() *(1-adapt_expert) + upd_MoE.detach().cpu().numpy() * adapt_expert
            else:
                self._protos[cls_index] = lowR_update.detach().cpu().numpy() *(1-self.w_expert) + upd_MoE.detach().cpu().numpy() * self.w_expert



    def _init_keys_with_kmeans(self, drift_estimator, train_loader, device, k):      
        all_features = []
        with torch.no_grad():
            for _, inputs, _ in train_loader:
                inputs = inputs.to(device)
                feats_old = self._old_network(inputs)["features"]
                all_features.append(feats_old.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(all_features)
        
        with torch.no_grad():
            drift_estimator.keys = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    def _build_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',mode='test', shot=self.shot, ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0) 
            self._protos.append(class_mean) 
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_cont = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                features = outputs["features"]
                logits = self._network.fc(features)['logits']
                
                loss_new_cls = F.cross_entropy(logits, targets) 
                L_new_cls += loss_new_cls.item()
                loss =  loss_new_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_cont / len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_cont {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.init_epoch,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_cont / len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        if hasattr(self._network, "module"):
            _network = self._network.module
        else:
            _network = self._network

        prog_bar = tqdm(range(self.epochs), colour='red', position=self._process_id, dynamic_ncols=True, ascii=" =", leave=True)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            L_all = 0.0
            L_new_cls = 0.0
            L_kd = 0.0
            L_trsf = 0.0
            correct, total = 0, 0
            

            for i, (_, inputs, targets) in enumerate(train_loader):
                loss_clf, loss_kd, loss_transfer = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs= _network(inputs)
                features = outputs["features"]
                logits = _network.fc(features)["logits"]
                
                outputs_old = self._old_network(inputs)
                fea_old = outputs_old["features"]
                logits_old = self._old_network.fc(fea_old)["logits"]

                # ---------------- loss1: new sample classification ---------------
                fake_targets = targets - self._known_classes 
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                L_new_cls += loss_clf.item()

                # ---------------- loss2: kd -------------------------

                loss_kd = _KD_loss(logits[:, : self._known_classes], logits_old, T) * self.w_kd

                L_kd += loss_kd.item()

                loss =  loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                L_all += loss.item()

                with torch.no_grad():
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_kd {:.3f}, L_trsf {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_kd  / len(train_loader), 
                    L_trsf  / len(train_loader), 
                    train_acc,
                    test_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "P{}: Task {}, Epoch {}/{} => L_all {:.3f}, L_new_cls {:.3f}, L_kd {:.3f}, L_trsf {:.3f}, Train_accy {:.2f}".format(
                    self._process_id,
                    self._cur_task, epoch + 1, self.epochs,
                    L_all / len(train_loader), 
                    L_new_cls  / len(train_loader), 
                    L_kd  / len(train_loader), 
                    L_trsf  / len(train_loader), 
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def pixel_fusion_augment(self, inputs, augment_factor=1):
        if augment_factor == 0:
            return inputs
        B, C, H, W = inputs.shape
        aug_inputs_list = [inputs]
        for _ in range(augment_factor):
            index = torch.randperm(B).to(self._device)
            shuffled_inputs = inputs[index].clone()
            for i in range(B):
                k = torch.randint(0, 4, (1,)).item()  # 0=0° 1=90°,2=180°,3=270°
                shuffled_inputs[i] = torch.rot90(shuffled_inputs[i], k, dims=[1, 2])
            mixed_inputs = 0.5 * inputs + 0.5 * shuffled_inputs
            noise = torch.randn_like(mixed_inputs) * self.noise_std
            mixed_inputs = mixed_inputs + noise
            aug_inputs_list.append(mixed_inputs)

        aug_inputs = torch.cat(aug_inputs_list, dim=0)
        return aug_inputs
    

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]





