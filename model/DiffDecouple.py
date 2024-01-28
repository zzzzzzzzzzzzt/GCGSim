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


class DiffDecouple(nn.Module):
    def __init__(self, config, n_feat):
        super(DiffDecouple, self).__init__()
        self.config                     = config
        self.n_feat                     = n_feat
        self.setup_layers()
        self.setup_score_layer()
        if config['dataset_name']== 'IMDBMulti':
            self.scale_init()
    
    def setup_layers(self):
        self.gnn_enc                         = self.config['gnn_encoder']
        self.filters                    = self.config['gnn_filters']
        self.num_filter                 = len(self.filters)
        self.use_ssl                    = self.config.get('use_ssl', False)

        if self.config['fuse_type']     == 'stack':
            filters                     = []
            for i in range(self.num_filter):
                filters.append(self.filters[0])
            self.filters                = filters
        self.gnn_list                   = nn.ModuleList()
        self.com_list                   = nn.ModuleList()  
        self.pri_list                   = nn.ModuleList()  
        self.NTN_list                   = nn.ModuleList()

        self.setup_backbone()
        if self.config['setup_disentangle']:
            self.setup_disentangle()

    def setup_backbone(self):
        if self.gnn_enc                      == 'GCN':  # append
            self.gnn_list.append(GCNConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GCNConv(self.filters[i],self.filters[i+1]))
        elif self.gnn_enc                    == 'GAT':
            self.gnn_list.append(GATConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GATConv(self.filters[i],self.filters[i+1]))  
        elif self.gnn_enc                    == 'GIN':
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
        
    def setup_disentangle(self):
        for i in range(self.num_filter):
            self.com_list.append(GlobalContextAware(self.config, self.filters[i]))
            self.pri_list.append(GlobalContextAware(self.config, self.filters[i]))
            self.NTN_list.append(TensorNetworkModule(self.config, self.filters[i]))