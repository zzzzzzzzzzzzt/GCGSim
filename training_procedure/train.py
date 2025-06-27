
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
def train(self, graph_batch, model, loss_func, optimizer, target, epoch):
    model.train(True)

    
    config                        = self.config   
    use_ssl                       = config.get('use_ssl', False) 
    use_mutualloss                = config.get('use_mutualloss', False)
    use_comloss                   = config.get('use_comloss', False)
    use_swap                      = config.get('use_swap', False)
    ppre_rate                     = config.get('ppre_rate', 0.0)
    optimizer.zero_grad()

    if config['model_name'] in ['CPRGsim']: 
        # if not config['use_sim']:
        prediction, reg_dict      = model(graph_batch)

        # from torchviz import make_dot
        # graph_forward = make_dot(model(graph_batch))
        # graph_forward.render(filename='graph/DiffDecouple163', view=False, format='pdf')
        _target = target['target']
        if reg_dict['prep_num']:
            _target[reg_dict['prep_num']:] = torch.ones_like(target['target'][reg_dict['prep_num']:])
        loss                      = loss_func(prediction, _target)
        com_loss, mutual_loss, swap_loss, ppre_loss = 0, 0, 0, 0

        if use_comloss:
            com_loss = config['alpha_weight']*reg_dict['com_loss']
            loss += com_loss
        # if use_compre:
        #     com_lable = torch.abs(torch.normal(mean=0.0, std=0.1, size=(reg_dict['ged_com'].shape[0],))).cuda()
        #     loss_compre = config['lambda_weight']*loss_func(reg_dict['ged_com'], com_lable)
        #     loss += loss_compre
        if use_mutualloss:
            mutual_loss = config['beta_weight']*reg_dict['mutual_loss']
            loss += mutual_loss
        if use_swap:
            swap_loss = config['mu_weight']*loss_func(reg_dict['swap_score'], target['target'])
            loss += swap_loss
        if ppre_rate > 0:
            c_loss = ppre_rate*loss_func(reg_dict['c_ged'], torch.zeros_like(target['target_ged']).cuda())
            p_loss = ppre_rate*loss_func(reg_dict['p_ged'], target['target_ged'])
            ppre_loss = 0.5*(c_loss+p_loss)
            loss += ppre_loss

        loss.backward()
        if self.config.get('clip_grad', False):
            nn.utils.clip_grad_norm_(model.parameters(), 1)
        # else:
        #     prediction_diff, prediction_sim = model(graph_batch)
        #     loss = (loss_func(prediction_diff, target) + loss_func(prediction_sim, target))/2
        # prediction = classifier(score_vec)
    elif config['model_name'] in ['GSC_GNN']: 
        # if not config['use_sim']:
        prediction, loss_cl       = model(graph_batch)
        loss                      = loss_func(prediction, target) if not use_ssl else loss_func(prediction, target)+loss_cl
        loss.backward()
        if self.config.get('clip_grad', False):
            nn.utils.clip_grad_norm_(model.parameters(), 1)
    elif config['model_name'] in ['SimGNN']:
        prediction                = model(graph_batch)
        loss                      = loss_func(prediction, target)
        loss.backward()

    optimizer.step()
    
    return model, float(loss), float(com_loss), float(mutual_loss), float(swap_loss), float(ppre_loss)
