from distutils.command.config import config
from platform import node
from turtle import forward
from gpustat import print_gpustat
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool, global_mean_pool
from model.layers import AttentionModule, MLPLayers, TensorNetworkModule, FF, GlobalContextAware, Node2GraphAttention, MLP
from utils.gan_losses import get_negative_expectation, get_positive_expectation
from collections import OrderedDict, defaultdict
import numpy as np
from torchmetrics.regression import PearsonCorrCoef
from functools import partial
from typing import Any, Optional, Tuple
import random
from torch_geometric.utils import to_dense_batch

class DiffDecouple(nn.Module):
    def __init__(self, config, n_feat):
        super(DiffDecouple, self).__init__()
        self.config                 = config
        self.batchsize              = self.config['batch_size']
        self.n_feat                 = n_feat
        self.pearsoncorrcoef        = PearsonCorrCoef(self.config['batch_size'])
        self.emb_log                = False
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
        self.NTN_ged_list           = nn.ModuleList()
        # if self.config['graph_encoder'] == 'GCA':
        #     self.com_list           = nn.ModuleList()  
        #     self.pri_list           = nn.ModuleList()
        # elif self.config['graph_encoder'] == 'deepset':
        self.deepset_inner      = nn.ModuleList()  
        self.c_deepset_outer    = nn.ModuleList()
        self.p_deepset_outer    = nn.ModuleList()
        self.comatt_MLP         = nn.ModuleList()
        self.n2gatt             = Node2GraphAttention(self.config, 'cosine_similarity')
        self.negative_slope     = 0.01
        if self.config['deepsets_inner_act'] == 'relu':
            self.act_inner      = F.relu
        elif self.config['deepsets_inner_act'] == 'leaky_relu':
            self.act_inner      = partial(F.leaky_relu, negative_slope=self.negative_slope)
        if self.config['deepsets_outer_act'] == 'relu':
            self.act_outer      = F.relu
        elif self.config['deepsets_outer_act'] == 'leaky_relu':
            self.act_outer      = partial(F.leaky_relu, negative_slope=self.negative_slope)
        # self.act_inner          = getattr(F, self.config.get('deepsets_inner_act', 'relu'))
        # self.act_outer          = getattr(F, self.config.get('deepsets_outer_act', 'relu'))

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
                if self.config['cat_NTN']:
                    self.NTN_list.append(TensorNetworkModule(self.config, 2*self.filters[i]))
                else:
                    self.NTN_list.append(TensorNetworkModule(self.config, self.filters[i]))
                self.NTN_ged_list.append(TensorNetworkModule(self.config, self.filters[i]))
        else:
            raise NotImplementedError("Error NTN_layer number.")
         
        self.score_sim_layer        = nn.Sequential(nn.Linear(self.config['tensor_neurons']*self.config['NTN_layers'], self.config['tensor_neurons']),
                                                    nn.ReLU(),
                                                    nn.Linear(self.config['tensor_neurons'] , 1))
        self.ged_sim_layer          = nn.Sequential(nn.Linear(self.config['tensor_neurons']*self.config['NTN_layers'], self.config['tensor_neurons']),
                                                    nn.ReLU(),
                                                    nn.Linear(self.config['tensor_neurons'] , 1))
        if  self.config['reconstruction'] == True:
            self.rec_MLP            = nn.Sequential(nn.Linear(2*self.filters[-1],2*self.filters[-1]),
                                                    nn.ReLU(),
                                                    nn.Linear(2*self.filters[-1],self.filters[-1]),
                                                    nn.ReLU())
            
    def setup_disentangle(self):
        # if self.config['graph_encoder'] == 'GCA':
        #     for i in range(self.num_filter):
        #         self.com_list.append(GlobalContextAware(self.config, self.filters[i]))
        #         self.pri_list.append(GlobalContextAware(self.config, self.filters[i]))
        # elif self.config['graph_encoder'] == 'deepset':
        for i in range(self.num_filter):
            if self.config['graph_encoder'] == 'deepset':
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
            elif self.config['graph_encoder'] == 'GCA':
                self.deepset_inner.append(GlobalContextAware(self.config, self.filters[i]))
            elif self.config['graph_encoder'] == 'None':
                pass

            if self.config.get('outer_mlp_layers', 1) == 1:
                self.c_deepset_outer.append(MLPLayers(2*self.filters[i], 
                                                        self.filters[i], 
                                                        None, 
                                                        num_layers=1, 
                                                        use_bn=False))
                self.p_deepset_outer.append(MLPLayers(2*self.filters[i], 
                                                        self.filters[i], 
                                                        None, 
                                                        num_layers=1, 
                                                        use_bn=False))
            else:
                self.c_deepset_outer.append(MLP(2*self.filters[i], 
                                                        self.filters[i], 
                                                        self.filters[i], 
                                                        num_layers=self.config['outer_mlp_layers'], 
                                                        use_bn=True))
                self.p_deepset_outer.append(MLP(2*self.filters[i], 
                                                        self.filters[i], 
                                                        self.filters[i], 
                                                        num_layers=self.config['outer_mlp_layers'], 
                                                        use_bn=True))
            
            if self.config['com_att']:
                self.comatt_MLP.append(nn.Sequential(nn.Linear(self.filters[i],self.filters[i]), nn.Sigmoid()))

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
        
        node_feature_1              = list()
        node_feature_2              = list()
        common_feature_1            = list()
        common_feature_2            = list()
        private_feature_1           = list()
        private_feature_2           = list()
        g1_pool                     = list()
        g2_pool                     = list()

        for i in range(self.num_filter):
            if self.config.get('convolpass', True):
                conv_source_1       = self.convolutional_pass(self.gnn_list[i], edge_index_1, conv_source_1)
                conv_source_2       = self.convolutional_pass(self.gnn_list[i], edge_index_2, conv_source_2)
            else:
                conv_source_1       = self.gnn_list[i](conv_source_1, edge_index_1)
                conv_source_2       = self.gnn_list[i](conv_source_2, edge_index_2)

            # if self.config['graph_encoder'] == 'GCA': 
            #     # generate common feature
            #     common_feature_1    .append(self.com_list[i](conv_source_1, batch_1))
            #     common_feature_2    .append(self.com_list[i](conv_source_2, batch_2))

            #     # generate private feature
            #     private_feature_1   .append(self.pri_list[i](conv_source_1, batch_1))
            #     private_feature_2   .append(self.pri_list[i](conv_source_2, batch_2))
            # elif self.config['graph_encoder'] == 'deepset':
            _common_feature_1,  \
            _common_feature_2,  \
            _private_feature_1, \
            _private_feature_2, \
            _g1_pool,           \
            _g2_pool            = self.deepset_output(conv_source_1, conv_source_2, batch_1, batch_2, i, True)

            node_feature_1      .append(conv_source_1)
            node_feature_2      .append(conv_source_2)
            common_feature_1    .append(_common_feature_1)
            common_feature_2    .append(_common_feature_2)
            private_feature_1   .append(_private_feature_1)
            private_feature_2   .append(_private_feature_2)
            g1_pool             .append(_g1_pool)
            g2_pool             .append(_g2_pool)

        if self.emb_log:
            self.com1_list          = common_feature_1
            self.com2_list          = common_feature_2
            self.pri1_list          = private_feature_1
            self.pri2_list          = private_feature_2
            self.nod1_list          = self.dense_batch(node_feature_1, batch_1)
            self.nod2_list          = self.dense_batch(node_feature_2, batch_2)

        ged_com, ged_pri            = self.compute_ged_score(common_feature_1, common_feature_2, private_feature_1, private_feature_2)
        if self.config['sim_rat']:
            com_Di, pri_Di          = self.get_sim_rat(g1_pool, g2_pool)
            com_distri, pri_distri  = com_Di, pri_Di
            common_feature_1, \
            common_feature_2, \
            private_feature_1, \
            private_feature_2       = self.feature_distri(common_feature_1, common_feature_2, private_feature_1, private_feature_2, com_distri, pri_distri)

        # computer score and loss
        ntn_score                   = self.compute_ntn_score(common_feature_1, common_feature_2, private_feature_1, private_feature_2, g1_pool, g2_pool)
        # ged_com, ged_pri            = self.compute_ged_score(common_feature_1, common_feature_2, private_feature_1, private_feature_2)
        dis_loss                    = self.compute_distance_loss(common_feature_1, common_feature_2, private_feature_1, private_feature_2, g1_pool, g2_pool, self.config['cat_disloss'])

        # log loss 
        self.dis_loss_log = dis_loss
        # self.cor_loss_log = cor_loss

        reg_dict = {'ged_com': ged_com, 
                    'ged_pri': ged_pri, 
                    'reg_loss': dis_loss}
        
        return ntn_score, reg_dict

    def collect_embeddings(self, all_graphs):
        node_embs_dict = dict()  
        for g in all_graphs:
            feat = g.x.cuda()
            edge_index = g.edge_index.cuda()
            for i, gnn in enumerate(self.gnn_list):
                # add gnn_list as key
                if i not in node_embs_dict.keys():
                    node_embs_dict[i] = dict()

                feat = gnn(feat, edge_index)
                if self.config.get('convolpass', True):
                    feat = F.relu(feat)

                node_embs_dict[i][int(g['i'])] = feat
        return node_embs_dict
    
    def collect_comandpri_embeddings(self, g_1, g_2, node_embs_dict, graph_embs_dicts):
        com_pri = list()
        for i in node_embs_dict.keys():
            n_1_i = node_embs_dict[i][int(g_1['i'])]
            n_2_i = node_embs_dict[i][int(g_2['i'])]

            com_1_i, \
            com_2_i, \
            pri_1_i, \
            pri_2_i, \
            _, _, _, _ = self.deepset_output_for1(n_1_i, n_2_i, i)

            com_pri.append({'com_1_i': com_1_i, 
                            'com_2_i': com_2_i, 
                            'pri_1_i': pri_1_i,
                            'pri_2_i': pri_2_i})
            
        graph_embs_dicts[int(g_1['i'])][int(g_2['i'])] = com_pri
        return graph_embs_dicts
    
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

            sim_pri1                = (
                                        f(F.cosine_similarity(common_feature_1[i], private_feature_1[i], dim=-1))
                                        if i == 0
                                        else torch.cat((sim_pri1, f(F.cosine_similarity(common_feature_1[i], private_feature_1[i], dim=-1))), dim=0)
                                        )
            
            sim_pri2                = (
                                        f(F.cosine_similarity(common_feature_2[i], private_feature_2[i], dim=-1))
                                        if i == 0
                                        else torch.cat((sim_pri2, f(F.cosine_similarity(common_feature_2[i], private_feature_2[i], dim=-1))), dim=0)
                                        )
        
        self.sim_com_log            = sim_com.mean()
        self.sim_pri1_log           = sim_pri1.mean()
        self.sim_pri2_log           = sim_pri2.mean()
        return -torch.log(sim_com/(sim_com + sim_pri1 + sim_pri2)).mean(), cor_loss_1.mean() + cor_loss_2.mean()

    def compute_distance_loss(self, common_feature_1, common_feature_2, private_feature_1, private_feature_2, g1_pool, g2_pool, cat = False):
        f = lambda x: torch.exp(x / self.config.get('tau', 1))

        if cat:
            common_feature_1 = [torch.cat(common_feature_1, dim=-1)]
            common_feature_2 = [torch.cat(common_feature_2, dim=-1)]
            private_feature_1 = [torch.cat(private_feature_1, dim=-1)]
            private_feature_2 = [torch.cat(private_feature_2, dim=-1)]

        len_list = len(common_feature_1)
        if self.config['noise']:
            common_feature_1 = self.add_noise(common_feature_1)
            common_feature_2 = self.add_noise(common_feature_2)
            private_feature_1 = self.add_noise(private_feature_1)
            private_feature_2 = self.add_noise(private_feature_2)

        dis_com =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_1[i], common_feature_2[i], dim=-1))) for i in range(len_list)], dim=0)
        dis_pri =  torch.cat([f(torch.abs(F.cosine_similarity(private_feature_1[i], private_feature_2[i], dim=-1))) for i in range(len_list)], dim=0)
        if self.config['detach']:
            dis_cp1 =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_1[i].detach(), private_feature_1[i], dim=-1))) for i in range(len_list)], dim=0)
            dis_cp2 =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_2[i].detach(), private_feature_2[i], dim=-1))) for i in range(len_list)], dim=0)
        else:
            dis_cp1 =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_1[i], private_feature_1[i], dim=-1))) for i in range(len_list)], dim=0)
            dis_cp2 =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_2[i], private_feature_2[i], dim=-1))) for i in range(len_list)], dim=0)
        dis_cg1 =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_1[i], g1_pool[i].detach(), dim=-1))) for i in range(len_list)], dim=0)
        dis_cg2 =  torch.cat([f(torch.abs(F.cosine_similarity(common_feature_2[i], g2_pool[i].detach(), dim=-1))) for i in range(len_list)], dim=0)

        center_common_1 = self.mean_centering(common_feature_1)
        center_common_2 = self.mean_centering(common_feature_2)
        center_private_1 = self.mean_centering(private_feature_1)
        center_private_2 = self.mean_centering(private_feature_2)

        dis_mean_cp1 =  torch.cat([f(torch.abs(F.cosine_similarity(center_common_1[i], center_private_1[i], dim=-1))) for i in range(len_list)], dim=0)
        dis_mean_cp2 =  torch.cat([f(torch.abs(F.cosine_similarity(center_common_2[i], center_private_2[i], dim=-1))) for i in range(len_list)], dim=0)

        self.sim_com_log = dis_com.mean()
        self.dis_pri_log = dis_pri.mean()
        self.sim_pri1_log = dis_cp1.mean()
        self.sim_pri2_log = dis_cp2.mean()
        self.dis_cg1_log = dis_cg1.mean()
        self.dis_cg2_log = dis_cg2.mean()
        self.dis_mean_cp1_log = dis_mean_cp1.mean()
        self.dis_mean_cp2_log = dis_mean_cp2.mean()

        dis_cp = self.config['alpha_weight']*(dis_cp1+dis_cp2)
        dis_mean_cp = self.config['beta_weight']*(dis_mean_cp1+dis_mean_cp2) 
        dis_pri = self.config['mu_weight']*dis_pri
        return ((dis_cp+dis_mean_cp+dis_pri)/dis_com).mean()

    def compute_ntn_score(self, common_feature_1, common_feature_2, private_feature_1, private_feature_2, g1_pool, g2_pool):
        if self.config['NTN_layers'] != 1:
            feature_1 = [torch.cat((common_feature_1[i], private_feature_1[i]), dim=-1) for i in range(self.num_filter)]
            feature_2 = [torch.cat((common_feature_2[i], private_feature_2[i]), dim=-1) for i in range(self.num_filter)]
        else:
            feature_1 = [torch.cat((common_feature_1[-1], private_feature_1[-1]), dim=-1)]
            feature_2 = [torch.cat((common_feature_2[-1], private_feature_2[-1]), dim=-1)]
            
        if self.config['cat_NTN']:    
            for i in range(self.config['NTN_layers']):
                ntn_score               = (
                                            self.NTN_list[i](feature_1[i], feature_2[i])
                                            if i == 0
                                            else torch.cat((ntn_score, self.NTN_list[i](feature_1[i], feature_2[i])), dim=-1)
                                            )
        else:
            ntn_score = torch.cat([self.NTN_list[i](g1_pool[i], g2_pool[i]) for i in range(self.config['NTN_layers'])], dim=-1)
        return torch.sigmoid(self.score_sim_layer(ntn_score).squeeze())
    
    def compute_ged_score(self, common_feature_1, common_feature_2, private_feature_1, private_feature_2):
        ged_c = torch.cat([self.NTN_ged_list[i](common_feature_1[i], common_feature_2[i]) for i in range(self.config['NTN_layers'])], dim=-1)
        ged_p = torch.cat([self.NTN_ged_list[i](private_feature_1[i], private_feature_2[i]) for i in range(self.config['NTN_layers'])], dim=-1)
        
        ged_com = torch.sigmoid(self.ged_sim_layer(ged_c).squeeze())
        ged_pri = torch.sigmoid(self.ged_sim_layer(ged_p).squeeze())

        return ged_com, ged_pri
    
    def reconstruction_loss(self, com_1, com_2, pri_1, pri_2, g_1, g_2):
        loss_fun = nn.MSELoss(reduction='sum')
        loss_1 = self._rec_loss(com_1, pri_1, g_1, loss_fun)
        loss_2 = self._rec_loss(com_2, pri_2, g_2, loss_fun)
        return loss_1 + loss_2
    
    def _rec_loss(self, common_feature, private_feature, grap_embedding, loss_fun):
        reconstruction = self.rec_MLP(torch.cat((common_feature,private_feature), dim=-1))
        return loss_fun(reconstruction, grap_embedding)
    
    def convolutional_pass(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p = self.config['dropout'], training=self.training)
        return feat

    def deepset_output(self, x1, x2, batch1, batch2, filter_idx, out=False):
        if self.config['graph_encoder'] == 'deepset':
            # deepset inner pass
            deepsets_inner_1 = self.act_inner(self.deepset_inner[filter_idx](x1))
            deepsets_inner_2 = self.act_inner(self.deepset_inner[filter_idx](x2))

            pool_1 = self._pool(deepsets_inner_1, batch1)
            pool_2 = self._pool(deepsets_inner_2, batch2)

            att_1with_2 = self.n2gatt(deepsets_inner_1, pool_2, batch1)
            att_2with_1 = self.n2gatt(deepsets_inner_2, pool_1, batch2)
        elif self.config['graph_encoder'] == 'GCA':
            pool_1 = self.deepset_inner[filter_idx](x1, batch1)
            pool_2 = self.deepset_inner[filter_idx](x2, batch2)

            att_1with_2 = self.n2gatt(x1, pool_2, batch1)
            att_2with_1 = self.n2gatt(x2, pool_1, batch2)
        elif self.config['graph_encoder'] == 'None':
            pool_1 = self._pool(x1, batch1)
            pool_2 = self._pool(x2, batch2)

            att_1with_2 = self.n2gatt(x1, pool_2, batch1)
            att_2with_1 = self.n2gatt(x2, pool_1, batch2)

        if self.config['com_att']:
            c1_coef = self.comatt_MLP[filter_idx](self.n2gatt(x1, pool_2, batch1))
            c2_coef = self.comatt_MLP[filter_idx](self.n2gatt(x2, pool_1, batch2))

            common_feature_1 = c1_coef * pool_1
            common_feature_2 = c2_coef * pool_2

            private_feature_1 = pool_1 - common_feature_1
            private_feature_2 = pool_2 - common_feature_2
        else:
            g1_embedding_att = torch.cat((pool_1, att_1with_2), dim=-1)
            g2_embedding_att = torch.cat((pool_2, att_2with_1), dim=-1)

            common_feature_1 = self.act_outer(self.c_deepset_outer[filter_idx](g1_embedding_att))
            common_feature_2 = self.act_outer(self.c_deepset_outer[filter_idx](g2_embedding_att))

            private_feature_1 = self.act_outer(self.p_deepset_outer[filter_idx](g1_embedding_att))
            private_feature_2 = self.act_outer(self.p_deepset_outer[filter_idx](g2_embedding_att))

        out_1, out_2 = None, None
        if out:
            out_1, out_2 = pool_1, pool_2
        return common_feature_1, common_feature_2, private_feature_1, private_feature_2, out_1, out_2
    
    def deepset_output_for1(self, x1, x2, filter_idx, out=False):
        deepsets_inner_1 = self.act_inner(self.deepset_inner[filter_idx](x1))
        deepsets_inner_2 = self.act_inner(self.deepset_inner[filter_idx](x2))

        pool_1 = torch.sum(deepsets_inner_1, dim=0)
        pool_2 = torch.sum(deepsets_inner_2, dim=0)

        coefs_12_i = self.n2gatt.get_coefs(deepsets_inner_1, pool_2)
        coefs_21_i = self.n2gatt.get_coefs(deepsets_inner_2, pool_1)

        att_1with_2 = torch.sum(coefs_12_i.unsqueeze(-1)*deepsets_inner_1, dim=0)
        att_2with_1 = torch.sum(coefs_21_i.unsqueeze(-1)*deepsets_inner_2, dim=0)

        g1_embedding_att = torch.cat((pool_1, att_1with_2), dim=-1)
        g2_embedding_att = torch.cat((pool_2, att_2with_1), dim=-1)

        common_feature_1 = self.act_outer(self.c_deepset_outer[filter_idx](g1_embedding_att))
        common_feature_2 = self.act_outer(self.c_deepset_outer[filter_idx](g2_embedding_att))

        private_feature_1 = self.act_outer(self.p_deepset_outer[filter_idx](g1_embedding_att))
        private_feature_2 = self.act_outer(self.p_deepset_outer[filter_idx](g2_embedding_att))

        out_1, out_2 = None, None
        if out:
            out_1, out_2 = pool_1, pool_2
        return common_feature_1, common_feature_2, private_feature_1, private_feature_2, out_1, out_2, coefs_12_i, coefs_21_i

    def _pool(self, feat, batch, size = None):
        size = (batch[-1].item() + 1 if size is None else size)   # 一个batch中的图数
        pool = global_add_pool(feat, batch, size=size) if self.config['pooling']=='add' else global_mean_pool(feat, batch, size=size) 
        return pool
    
    def get_sim_rat(self, pool_1, pool_2):
        com_distri = [F.cosine_similarity(pool_1[i], pool_2[i], dim=-1).unsqueeze(-1) for i in range(self.num_filter)]
        pri_distri = [1 - com_distri[i] for i in range(self.num_filter)]
        return com_distri, pri_distri

    def feature_distri(self, common_feature_1, common_feature_2, private_feature_1, private_feature_2, com_distri, pri_distri):
        if type(com_distri) is list:
            com_1 = [com_distri[i]*common_feature_1[i] for i in range(self.num_filter)]
            com_2 = [com_distri[i]*common_feature_2[i] for i in range(self.num_filter)]
            pri_1 = [pri_distri[i]*private_feature_1[i] for i in range(self.num_filter)]
            pri_2 = [pri_distri[i]*private_feature_2[i] for i in range(self.num_filter)]
        else:
            com_1 = [com_distri*common_feature_1[i] for i in range(self.num_filter)]
            com_2 = [com_distri*common_feature_2[i] for i in range(self.num_filter)]
            pri_1 = [pri_distri*private_feature_1[i] for i in range(self.num_filter)]
            pri_2 = [pri_distri*private_feature_2[i] for i in range(self.num_filter)]
        return com_1, com_2, pri_1, pri_2
     
    
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
    
    def mean_centering(self, x):
        return [x[i] - torch.mean(x[i], dim=-1, keepdim=True) for i in range(len(x))]
    
    def add_noise(self, x):
        return [x[i] + torch.normal(mean=0.0, std=1e-5, size=x[i].shape).cuda() for i in range(len(x))]
    
    def graphcom_shuffle(self, g1, g2, c1, c2):
        batchlen = g1.shape[0]
        ex_index = torch.LongTensor(random.sample(range(batchlen), batchlen//2))
        
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
    
    def dense_batch(self, n, b):
        return [to_dense_batch(n[i], b) for i in range(self.num_filter)]

    def log_param(self, writer, log_i):
        for name, param in self.named_parameters():
            if 'gnn_list' in name:
                writer.add_histogram('param/{}'.format(name), param, log_i)
                writer.add_histogram('param_grad/{}'.format(name), param.grad, log_i)

class GraphCommonPrediction(nn.Module):
    def __init__(self, n_feat):
        super(GraphCommonPrediction, self).__init__()
        self.f = nn.Sequential(nn.Linear(n_feat, n_feat//2),
                                nn.ReLU(),
                                nn.Linear(n_feat//2 , 1))
        
    def forward(self, x, p=1.0):
        return self.f(GradientReversalLayer.apply(x,p))
    
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None