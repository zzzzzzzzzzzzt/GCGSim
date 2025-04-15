from model.CPRGsim import CPRGsim
from argparse import ArgumentParser
import os.path as osp
from utils.utils import *
import torch
import torch.nn.functional as F
import random
from DataHelper.data_utils import *
from torch_geometric.utils import to_undirected
import numpy as np
from torch_geometric.data import Batch

def gen_subgraph(n_node, n_feature, n_link):
    random_nodes = torch.randint(0, n_feature, (n_node,))
    one_hot_labels = F.one_hot(random_nodes, num_classes=n_feature)
    link_node = torch.randperm(n_node)[:n_link]

    adj = torch.ones((n_node, n_node), dtype=torch.uint8)
    non_edge_index = adj.nonzero().t()
    directed_non_edge_index = to_directed(non_edge_index)
    # num_edges = directed_non_edge_index.size()[1]
    # to_add = random.randint(n_node-1, n_node*(n_node-1)//2+1)

    # edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]
    # if edge_index_p.size(1):
    #     edge_index_p = to_undirected(edge_index_p)
    
    return one_hot_labels, directed_non_edge_index, link_node

def joint(g, one_hot_labels, edge_index_p, link_node, id):
    edge_index_p = edge_index_p + g.num_nodes
    link_node = link_node + g.num_nodes

    n_link= link_node.size()[0]
    tolink_node = torch.randperm(g.num_nodes)[:n_link]
    link_edge = torch.stack((link_node,tolink_node), 0)
    edge = torch.cat((to_directed(g.edge_index), edge_index_p, link_edge), 1)
    node = torch.cat((g.x, one_hot_labels), 0)

    if edge.size(1):
        edge = to_undirected(edge)
    G = Data(x=node, edge_index=edge, i=id)

    return G
def gen_pair(g1, g2, n_node, n_feature, n_link):
    one_hot_labels, edge_index_p, link_node = gen_subgraph(n_node, n_feature, n_link)

    g1_added = joint(g1, one_hot_labels, edge_index_p, link_node)
    g2_added = joint(g2, one_hot_labels, edge_index_p, link_node)

    GED = n_node + edge_index_p.size()[1] + n_link
    return (g1_added, g2_added), GED

def gen_samepri(data_graph, n_feature, n_node=4, n_link=2):
    count = len(data_graph)
    mat = torch.full((count, count), float("inf")) 
    norm_mat = torch.full((count, count), float("inf"))

    synth_graph = []
    for i in range(count//2):
        one_hot_labels, edge_index_p, link_node = gen_subgraph(n_node, n_feature, n_link)
        g1_added = joint(data_graph[2*i], one_hot_labels, edge_index_p, link_node, 2*i)
        g2_added = joint(data_graph[2*i+1], one_hot_labels, edge_index_p, link_node, 2*i+1)
        ged_1 = n_node + edge_index_p.size()[1] + n_link
        ged_2 = ged_1

        mat[2*i,2*i], mat[2*i+1,2*i+1] = ged_1, ged_2
        norm_mat[2*i,2*i] = ged_1 / (0.5 * (g1_added.num_nodes + data_graph[2*i].num_nodes))
        norm_mat[2*i+1,2*i+1] = ged_2 / (0.5 * (g2_added.num_nodes + data_graph[2*i+1].num_nodes))

        synth_graph.append(g1_added)
        synth_graph.append(g2_added)
    
    return data_graph, synth_graph, mat, norm_mat

def gen_samecom(data_graph, n_feature, n_node=1, n_link=1, same=True):
    count = len(data_graph)
    mat = torch.full((count, 2*count), float("inf")) 
    norm_mat = torch.full((count, 2*count), float("inf"))

    synth_graph = []
    for i in range(count):
        one_hot_labels, edge_index_p, link_node = gen_subgraph(n_node, n_feature, n_link)
        g1_added = joint(data_graph[i], one_hot_labels, edge_index_p, link_node, 2*i)
        ged_1 = n_node + edge_index_p.size()[1] + n_link

        one_hot_labels, edge_index_p, link_node = gen_subgraph(n_node+1, n_feature, n_link+1)
        g2_added = joint(data_graph[i], one_hot_labels, edge_index_p, link_node, 2*i+1)
        ged_2 = n_node+1 + edge_index_p.size()[1] + n_link+1

        mat[i,2*i], mat[i,2*i+1] = ged_1, ged_2
        norm_mat[i,2*i] = ged_1 / (0.5 * (g1_added.num_nodes + data_graph[i].num_nodes))
        norm_mat[i,2*i+1] = ged_2 / (0.5 * (g2_added.num_nodes + data_graph[i].num_nodes))

        synth_graph.append(g1_added)
        synth_graph.append(g2_added)
    
    return data_graph, synth_graph, mat, norm_mat

def transform_batch(batch, mat, norm_mat):
    new_data = dict()

    new_data["g1"] = batch[0]
    new_data["g2"] = batch[1] 

    normalized_ged = norm_mat[
        batch[0]["i"].reshape(-1).tolist(), batch[1]["i"].reshape(-1).tolist()
    ].tolist()
    new_data["target"] = (
        torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()
    )
    new_data['norm_ged'] = (
        torch.from_numpy(np.array([(el) for el in normalized_ged])).view(-1).float()    # nged
    )
    ged = mat[
        batch[0]["i"].reshape(-1).tolist(), batch[1]["i"].reshape(-1).tolist()
    ].tolist()

    new_data["target_ged"] = (
        torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()   
    )

    return new_data

@torch.no_grad()
def evaluate(model, dataset: DatasetLocal, synth_type):
    model.eval()

    training_graphs = dataset.training_graphs
    mapsize = len(training_graphs)
    if synth_type == 'samecom':
        scores = np.empty((mapsize,2))
        prediction_mat = np.empty((mapsize,2))
        ground_truth = np.empty((mapsize,2))

        source_graph, synth_graph, mat, norm_mat = gen_samecom(training_graphs, dataset.input_dim)

        for i in range(mapsize):
            source_batch = Batch.from_data_list([source_graph[i], source_graph[i]])
            target_batch = Batch.from_data_list([synth_graph[2*i], synth_graph[2*i+1]])

            data = transform_batch((source_batch, target_batch), mat, norm_mat)
            target = data["target"]
            ground_truth[i] = target

            prediction, loss_cl = model(data)

            prediction_mat[i] = prediction.cpu().detach().numpy()
            scores[i]  = ( F.mse_loss(prediction.cpu().detach(), target, reduction="none").numpy())
        
        return 0
    if synth_type == 'samepri':
        scores = np.empty((mapsize//2,2))
        prediction_mat = np.empty((mapsize//2,2))
        ground_truth = np.empty((mapsize//2,2))
        
        source_graph, synth_graph, mat, norm_mat = gen_samepri(training_graphs, dataset.input_dim)

        for i in range(mapsize//2):
            source_batch = Batch.from_data_list([source_graph[2*i], source_graph[2*i+1]])
            target_batch = Batch.from_data_list([synth_graph[2*i], synth_graph[2*i+1]])            
            data = transform_batch((source_batch, target_batch), mat, norm_mat)
            target = data["target"]
            ground_truth[i] = target

            prediction, loss_cl = model(data)

            prediction_mat[i] = prediction.cpu().detach().numpy()
            scores[i]  = ( F.mse_loss(prediction.cpu().detach(), target, reduction="none").numpy())

        return 0
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset',           type = str,              default = 'AIDS700nef') 
    parser.add_argument('--data_dir',          type = str,              default = 'datasets/')
    parser.add_argument('--extra_dir',         type = str,              default = 'exp/')    
    parser.add_argument('--gpu_id',            type = int  ,            default = 0)
    parser.add_argument('--model',             type = str,              default = 'CPRGsim')  # GCN, GAT or other
    parser.add_argument('--recache',         action = "store_true",        help = "clean up the old adj data", default=True)
    parser.add_argument('--pretrain_path',     type = str,              default = 'model_saved/AIDS700nef/2024-07-21/CPRGsim_AIDS700nef_tensorneurons_0')
    args = parser.parse_args()
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # torch.cuda.set_device("cuda:0")

    config_path                 = osp.join(args.pretrain_path, 'config' + '.yml')
    config                      = get_config(config_path)
    # config                      = config[args.model] 
    config['dataset_name']      = args.dataset
    print(config)

    dataset                     = load_data(args, False)
    dataset                     . load(config)
    model                       = CPRGsim(config, dataset.input_dim, True).cuda()
    para                        = osp.join(args.pretrain_path, 'CPRGsim_{}_checkpoint.pth'.format(args.dataset))
    model                       . load_state_dict(torch.load(para))
    model                       . eval()

    evaluate(model, dataset, 'samecom')