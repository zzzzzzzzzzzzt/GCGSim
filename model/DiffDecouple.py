from distutils.command.config import config
from platform import node
from turtle import forward
from gpustat import print_gpustat
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool, global_mean_pool
from model.layers import AttentionModule, MLPLayers, TensorNetworkModule, FF, GlobalContextAware
from utils.gan_losses import get_negative_expectation, get_positive_expectation
from collections import OrderedDict, defaultdict
import numpy as np
from torchmetrics.regression import PearsonCorrCoef

class DiffDecouple(nn.Module):
    def __init__(self, config, n_feat):
        super(DiffDecouple, self).__init__()
        self.config                 = config
        self.batchsize              = self.config['batch_size']
        self.n_feat                 = n_feat
        self.pearsoncorrcoef        = PearsonCorrCoef(self.config['batch_size'])
        self.setup_layers()
    
    def setup_layers(self):
        self.gnn_enc                = self.config['gnn_encoder']
        self.filters                = self.config['gnn_filters']
        self.num_filter             = len(self.filters)
        self.use_ssl                = self.config.get('use_ssl', False)

        if self.config['fuse_type'] == 'stack':
            filters                 = []
            for i in range(self.num_filter):
                filters.append(self.filters[0])
            self.filters            = filters
        self.gnn_list               = nn.ModuleList()  
        self.NTN_list               = nn.ModuleList()
        if self.config['graph_encoder'] == 'GCA':
            self.com_list           = nn.ModuleList()  
            self.pri_list           = nn.ModuleList()
        elif self.config['graph_encoder'] == 'deepset':
            self.deepset_inner      = nn.ModuleList()  
            self.c_deepset_outer    = nn.ModuleList()
            self.p_deepset_outer    = nn.ModuleList()
            self.act_inner          = getattr(F, self.config.get('deepsets_inner_act', 'relu'))
            self.act_outer          = getattr(F, self.config.get('deepsets_outer_act', 'relu'))

        self.setup_backbone()
        if self.config.get('setup_disentangle', True):
            self.setup_disentangle()

    def setup_backbone(self):
        if self.gnn_enc             == 'GCN':  # append
            self.gnn_list.append(GCNConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GCNConv(self.filters[i],self.filters[i+1]))
        elif self.gnn_enc           == 'GAT':
            self.gnn_list.append(GATConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GATConv(self.filters[i],self.filters[i+1]))  
        elif self.gnn_enc           == 'GIN':
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.n_feat, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ),eps=True))

            for i in range(self.num_filter-1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.filters[i],self.filters[i+1]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[i+1], self.filters[i+1]),
                torch.nn.BatchNorm1d(self.filters[i+1]),
            ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

        if self.config['NTN_layers'] == 1:
            self.NTN_list.append(TensorNetworkModule(self.config, 2*self.filters[-1]))
        elif self.config['NTN_layers'] == self.num_filter:
            for i in range(self.num_filter):
                self.NTN_list.append(TensorNetworkModule(self.config, 2*self.filters[i]))
        else:
            raise NotImplementedError("Error NTN_layer number.")
         
        self.score_sim_layer        = nn.Sequential(nn.Linear(self.config['tensor_neurons']*self.config['NTN_layers'], self.config['tensor_neurons']),
                                                    nn.ReLU(),
                                                    nn.Linear(self.config['tensor_neurons'] , 1))
        
    def setup_disentangle(self):
        if self.config['graph_encoder'] == 'GCA':
            for i in range(self.num_filter):
                self.com_list.append(GlobalContextAware(self.config, self.filters[i]))
                self.pri_list.append(GlobalContextAware(self.config, self.filters[i]))
        elif self.config['graph_encoder'] == 'deepset':
            for i in range(self.num_filter):
                if self.config['inner_mlp']:
                    if self.config.get('inner_mlp_layers', 1) == 1:
                        self.deepset_inner.append(MLPLayers(self.filters[i], 
                                                             self.filters[i], 
                                                             None, 
                                                             num_layers=1, 
                                                             use_bn=False))
                    else:
                        self.deepset_inner.append(MLPLayers(self.filters[i], 
                                                             self.filters[i], 
                                                             self.filters[i], 
                                                             num_layers=self.config['inner_mlp_layers'], 
                                                             use_bn=False))
                if self.config.get('outer_mlp_layers', 1) == 1:
                    self.c_deepset_outer.append(MLPLayers(self.filters[i], 
                                                          self.filters[i], 
                                                          None, 
                                                          num_layers=1, 
                                                          use_bn=False))
                    self.p_deepset_outer.append(MLPLayers(self.filters[i], 
                                                          self.filters[i], 
                                                          None, 
                                                          num_layers=1, 
                                                          use_bn=False))
                else:
                    self.c_deepset_outer.append(MLPLayers(self.filters[i], 
                                                          self.filters[i], 
                                                          self.filters[i], 
                                                          num_layers=self.config['outer_mlp_layers'], 
                                                          use_bn=False))
                    self.p_deepset_outer.append(MLPLayers(self.filters[i], 
                                                          self.filters[i], 
                                                          self.filters[i], 
                                                          num_layers=self.config['outer_mlp_layers'], 
                                                          use_bn=False))

    def forward(self, data):
        edge_index_1                = data['g1'].edge_index.cuda()
        edge_index_2                = data['g2'].edge_index.cuda()
        features_1                  = data["g1"].x.cuda()
        features_2                  = data["g2"].x.cuda()
        batch_1                     = (
                                        data["g1"].batch.cuda()
                                        if hasattr(data["g1"], "batch")
                                        else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).cuda()
                                        )
        batch_2                     = (
                                        data["g2"].batch.cuda()
                                        if hasattr(data["g2"], "batch")
                                        else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).cuda()
                                        )
        
        conv_source_1               = torch.clone(features_1)
        conv_source_2               = torch.clone(features_2)
        
        
        common_feature_1            = list()
        common_feature_2            = list()
        private_feature_1           = list()
        private_feature_2           = list()
        for i in range(self.num_filter):
            if self.config.get('convolpass', True):
                conv_source_1       = self.convolutional_pass(self.gnn_list[i], edge_index_1, conv_source_1)
                conv_source_2       = self.convolutional_pass(self.gnn_list[i], edge_index_2, conv_source_2)
            else:
                conv_source_1       = self.gnn_list[i](conv_source_1, edge_index_1)
                conv_source_2       = self.gnn_list[i](conv_source_2, edge_index_2)

            if self.config['graph_encoder'] == 'GCA': 
                # generate common feature
                common_feature_1    .append(self.com_list[i](conv_source_1, batch_1))
                common_feature_2    .append(self.com_list[i](conv_source_2, batch_2))

                # generate private feature
                private_feature_1   .append(self.pri_list[i](conv_source_1, batch_1))
                private_feature_2   .append(self.pri_list[i](conv_source_2, batch_2))
            elif self.config['graph_encoder'] == 'deepset':
            
        # computer score and loss
        ntn_score                   = self.compute_ntn_score(common_feature_1, common_feature_2, private_feature_1, private_feature_2)
        decouple_loss               = self.compute_decouple_loss(common_feature_1, common_feature_2, private_feature_1, private_feature_2)

        
        return ntn_score, decouple_loss
    
    def compute_decouple_loss(self, common_feature_1,
                            common_feature_2, 
                            private_feature_1, 
                            private_feature_2):
        
        f                           = lambda x: torch.exp(x / self.config.get('tau', 1))
        for i in range(self.num_filter):
            # compute correlation coefficient between common feature and private feature

            # _common_feature_1, _private_feature_1 = self.shape_for_corr(common_feature_1[i], private_feature_1[i])
            # _common_feature_2, _private_feature_2 = self.shape_for_corr(common_feature_2[i], private_feature_2[i])

            cor_loss_1              = (
                                        self.cal_corr(common_feature_1[i], private_feature_1[i])
                                        if i == 0
                                        else torch.cat((cor_loss_1, self.cal_corr(common_feature_1[i], private_feature_1[i])), dim=0)
                                        )
            cor_loss_2              = (
                                        self.cal_corr(common_feature_2[i], private_feature_2[i])
                                        if i == 0
                                        else torch.cat((cor_loss_2, self.cal_corr(common_feature_2[i], private_feature_2[i])), dim=0)
                                        )
            
            # compute similarity between features
            # if self.config.get('norm', False):
            #     common_feature_1        = F.normalize(common_feature_1, dim=2)
            #     common_feature_2        = F.normalize(common_feature_2, dim=2)
            #     private_feature_1       = F.normalize(private_feature_1, dim=2)
            #     private_feature_2       = F.normalize(private_feature_2, dim=2)

            # com_dot                     = torch.mm(common_feature_1, torch.transpose(common_feature_2, dim0=1, dim1=2))  
            # pri_dot                     = torch.mm(private_feature_1, torch.transpose(private_feature_2, dim0=1, dim1=2))

            sim_com                 = (
                                        f(F.cosine_similarity(common_feature_1[i], common_feature_2[i], dim=-1))
                                        if i == 0
                                        else torch.cat((sim_com, f(F.cosine_similarity(common_feature_1[i], common_feature_2[i], dim=-1))), dim=0)
                                        )

            sim_pri                 = (
                                        f(F.cosine_similarity(common_feature_2[i], private_feature_2[i], dim=-1))
                                        if i == 0
                                        else torch.cat((sim_pri, f(F.cosine_similarity(common_feature_2[i], private_feature_2[i], dim=-1))), dim=0)
                                        )
        
        return -torch.log(sim_com/(sim_com + sim_pri)).mean() + cor_loss_1.mean() + cor_loss_2.mean()

    def compute_ntn_score(self, common_feature_1, 
                        common_feature_2,
                        private_feature_1, 
                        private_feature_2):
        if self.config['NTN_layers'] != 1:
            feature_1 = [torch.cat((common_feature_1[i], private_feature_1[i]), dim=-1) for i in range(self.num_filter)]
            feature_2 = [torch.cat((common_feature_2[i], private_feature_2[i]), dim=-1) for i in range(self.num_filter)]
        else:
            feature_1 = [torch.cat((common_feature_1[-1], private_feature_1[-1]), dim=-1)]
            feature_2 = [torch.cat((common_feature_2[-1], private_feature_2[-1]), dim=-1)]
            
        for i in range(self.config['NTN_layers']):
            ntn_score               = (
                                        self.NTN_list[i](feature_1[i], feature_2[i])
                                        if i == 0
                                        else torch.cat((ntn_score, self.NTN_list[i](feature_1[i], feature_2[i])), dim=-1)
                                        )
            
        return torch.sigmoid(self.score_sim_layer(ntn_score).squeeze())
    
    def convolutional_pass(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p = self.config['dropout'], training=self.training)
        return feat

    def deepset_output(self, x1, x2, batch1, batch2, filter_idx):
        # deepset inner pass
        deepsets_inner_1 = self.act_inner(self.deepset_inner[filter_idx](x1))
        deepsets_inner_2 = self.act_inner(self.deepset_inner[filter_idx](x2))

        pool_1 = self._pool(deepsets_inner_1, batch1)
        pool_2 = self._pool(deepsets_inner_2, batch2)

        

    def _pool(self, feat, batch, size = None):
        size = (batch[-1].item() + 1 if size is None else size)   # 一个batch中的图数
        pool = global_add_pool(feat, batch, size=size) if self.config['pooling']=='add' else global_mean_pool(feat, batch, size=size) 
        return pool
        
    def compute_corr(self, x1, x2):
        # Subtract the mean
        x1_mean = torch.mean(x1, dim=-1, keepdim=True)
        x1 = x1 - x1_mean
        x2_mean = torch.mean(x2, dim=-1, keepdim=True)
        x2 = x2 - x2_mean


        # Compute the cross correlation
        sigma1 = torch.sqrt(torch.mean(x1.pow(2), dim=-1))
        sigma2 = torch.sqrt(torch.mean(x2.pow(2), dim=-1))
        corr = torch.abs(torch.mean(x1*x2, dim=-1))/(sigma1*sigma2)

        return corr

    def cal_corr(self, x1, x2, eps = 1e-7):
        vx = x1 - torch.mean(x1, dim=-1, keepdim=True)
        vy = x2 - torch.mean(x2, dim=-1, keepdim=True)
        
        corr = torch.abs(torch.sum(vx * vy, dim=-1))/(torch.norm(vx, dim=-1) * torch.norm(vy, dim=-1) + eps)  # use Pearson correlation

        return corr
    
    def shape_for_corr(self, x1, x2):
        if x1.size()[0] == self.config['batch_size']:
            out_x1 = x1.T
            out_x2 = x2.T
        else:
            pad_num = self.config['batch_size'] - x1.size()[0]
            out_x1 = F.pad(x1, 
                           (0, 0, 0, pad_num), 
                           mode='constant', 
                           value=0).T

            out_x2 = F.pad(x2, 
                           (0, 0, 0, pad_num), 
                           mode='constant', 
                           value=0).T
            
        return out_x1, out_x2
    
    def log_param(self, writer, log_i):
        for name, param in self.named_parameters():
            if 'gnn_list' in name:
                writer.add_histogram('param/{}'.format(name), param, log_i)
                writer.add_histogram('param_grad/{}'.format(name), param.grad, log_i)

    