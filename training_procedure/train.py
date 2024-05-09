
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
def train(self, graph_batch, model, loss_func, optimizer, target, dataset = None):
    model.train(True)

    
    config                        = self.config   
    use_ssl                       = config.get('use_ssl', False) 
    use_compre                    = config.get('use_compre', False)
    use_pripre                    = config.get('use_pripre', False)
    optimizer.zero_grad()

    if config['model_name'] in ['GSC_GNN']: 
        # if not config['use_sim']:
        prediction, reg_dict      = model(graph_batch)

        # from torchviz import make_dot
        # graph_forward = make_dot(model(graph_batch))
        # graph_forward.render(filename='graph/DiffDecouple163', view=False, format='pdf')

        loss                      = loss_func(prediction, target['target']) 
        loss_cl                   = 0
        loss_compre               = 0
        loss_pripre               = 0

        if use_ssl:
            loss_cl = reg_dict['reg_loss']
            loss += loss_cl
        if use_compre:
            com_lable = torch.abs(torch.normal(mean=0.0, std=0.1, size=(reg_dict['ged_com'].shape[0],))).cuda()
            loss_compre = loss_func(reg_dict['ged_com'], com_lable)
            loss += loss_compre
        if use_pripre:
            pri_lable = target['target_scaler']
            loss_pripre = loss_func(reg_dict['ged_pri'], pri_lable)
            loss += loss_pripre
            
        loss.backward()
        if self.config.get('clip_grad', False):
            nn.utils.clip_grad_norm_(model.parameters(), 1)
        # else:
        #     prediction_diff, prediction_sim = model(graph_batch)
        #     loss = (loss_func(prediction_diff, target) + loss_func(prediction_sim, target))/2
        # prediction = classifier(score_vec)
    elif config['model_name'] in ['SimGNN']:
        prediction                = model(graph_batch)
        loss                      = loss_func(prediction, target)
        loss.backward()

    optimizer.step()
    
    return model, float(loss), float(loss_cl), float(loss_compre), float(loss_pripre)
