import torch.nn.functional as F
import torch
import torch.nn as nn 


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers = 2 ,use_bn=True):
        super(MLP, self).__init__()

        modules = []
        modules.append(nn.Linear(nfeat, nhid, bias=True))
        if use_bn:
            modules.append(nn.BatchNorm1d(nhid))
        modules.append(nn.ReLU())
        for i in range(num_layers-2):
            modules.append(nn.Linear(nhid, nhid, bias=True))
            if use_bn:
               modules.append(nn.BatchNorm1d(nhid)) 
            modules.append(nn.ReLU())


        modules.append(nn.Linear(nhid, nclass, bias=True))
        self.mlp_list = nn.Sequential(*modules)

    def forward(self, x):
        x = self.mlp_list(x)
        return x

class MLPLayers(nn.Module):
    def __init__(self, n_in, n_hid, n_out, num_layers = 2 ,use_bn=True):
        super(MLPLayers, self).__init__()
        modules = []
        modules.append(nn.Linear(n_in, n_hid))
        out = n_hid
        use_act = True
        for i in range(num_layers-1):  # num_layers = 3  i=0,1
            if i == num_layers-2:
                use_bn = False
                use_act = False
                out = n_out
            modules.append(nn.Linear(n_hid, out))
            if use_bn:
                modules.append(nn.BatchNorm1d(out)) 
            if use_act:
                modules.append(nn.ReLU())
        self.mlp_list = nn.Sequential(*modules)
    def forward(self,x):
        x = self.mlp_list(x)
        return x


class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.fc1 = MLPLayers(2*dim, dim, 1, num_layers=4, use_bn=False)
        self.init_parameters()

    def forward(self,x,y):
        h1 = torch.cat([x, y], dim=1)
        h2 = self.fc1(h1)
        return h2
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)