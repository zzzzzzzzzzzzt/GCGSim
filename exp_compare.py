from model.CPRGsim import CPRGsim
from model.GSC import GSC
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
from tqdm import tqdm
import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate(testing_graphs, training_graphs, model_1, model_2, dataset: DatasetLocal, config):
    model_1.eval()
    model_2.eval()

    scores_1 = np.empty((len(testing_graphs), len(training_graphs)))
    scores_2 = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth_ged = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth_nged = np.empty((len(testing_graphs), len(training_graphs)))
    prediction_mat_1 = np.empty((len(testing_graphs), len(training_graphs)))
    prediction_mat_2 = np.empty((len(testing_graphs), len(training_graphs)))

    num_test_pairs = len(testing_graphs) * len(training_graphs)
    t = tqdm(total=num_test_pairs)

    for i,g in enumerate(testing_graphs):
        source_batch = Batch.from_data_list([g] * len(training_graphs))
        target_batch = Batch.from_data_list(training_graphs)
        data = dataset.transform_batch((source_batch, target_batch), config)
            
        ground_truth[i] = data["target"]    
        ground_truth_ged[i] = data["target_ged"]
        ground_truth_nged[i] = data["norm_ged"]

        prediction, _ = model_1(data)
        prediction_mat_1[i] = prediction.cpu().detach().numpy()
        scores_1[i] = (F.mse_loss(prediction.cpu().detach(), data["target"], reduction="none").numpy())

        prediction, _ = model_2(data)
        prediction_mat_2[i] = prediction.cpu().detach().numpy()
        scores_2[i] = (F.mse_loss(prediction.cpu().detach(), data["target"], reduction="none").numpy())
        t.update(len(training_graphs))

    return scores_1, scores_2, ground_truth, ground_truth_ged

def loss_distribution(scores_1, scores_2, *args):
    if len(args)==2:
        return loss_distribution_similarity(scores_1, scores_2, args[0], args[1])
    elif len(args)==1:
        return loss_distribution_GED(scores_1, scores_2, args[0])

def loss_distribution_similarity(scores_1, scores_2, gt, n=25):
    step = 1/n
    x = []
    y_1 = []
    y_2 = []
    for i in range(n):
        down = step*i
        up = step*(i+1)
        x.append((down+up)/2)
        y_1.append(f_average(gt, scores_1, down, up))
        y_2.append(f_average(gt, scores_2, down, up))

        fig, ax = plt.subplots()
        ax.plot(x, y_1, x, y_2)

    _, mode_dir = osp.split(args.pretrain_path_1)
    exp_figure_name = 'loss_distribution_similarity'
    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), 'compare_loss_distribution_similarity_{}'.format(args.dataset))
    plt.close

def loss_distribution_GED(scores_1, scores_2, gt_ged):
    ged_max = int(np.max(gt_ged))
    ged_min = int(np.min(gt_ged))
    x = []
    y_1 = []
    y_2 = []
    for i in range(ged_min, 20):
        x.append(i)
        y_1.append(f_average(gt_ged, scores_1, i))
        y_2.append(f_average(gt_ged, scores_2, i))

    fig, ax = plt.subplots()
    ax.plot(x, y_1, x, y_2)

    _, mode_dir = osp.split(args.pretrain_path_1)
    exp_figure_name = 'loss_distribution_GED'
    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), 'compare_loss_distribution_GED_{}'.format(args.dataset))
    plt.close

def f_average(gt, s, *args):
    if len(args)==2:
        f = np.where((gt>args[0]) & (gt<=args[1]), s, 0.0)
    elif len(args)==1:
        f = np.where(gt==args[0], s, 0.0)
    numf = len(np.nonzero(f)[0])
    return f.sum()/numf

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset',           type = str,              default = 'IMDBMulti') 
    parser.add_argument('--data_dir',          type = str,              default = 'datasets/')
    parser.add_argument('--extra_dir',         type = str,              default = 'exp/')    
    parser.add_argument('--gpu_id',            type = int  ,            default = 0)
    parser.add_argument('--recache',         action = "store_true",        help = "clean up the old adj data", default=True)
    parser.add_argument('--pretrain_path_1',     type = str,              default = 'model_saved/AIDS700nef/2024-07-21/CPRGsim_AIDS700nef_tensorneurons_1')
    parser.add_argument('--pretrain_path_2',     type = str,              default = 'model_saved/AIDS700nef/2024-07-21/CPRGsim_AIDS700nef_tensorneurons_1')
    args = parser.parse_args()
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # torch.cuda.set_device("cuda:0")

    config_path_1 = osp.join(args.pretrain_path_1, 'config' + '.yml')
    config_1 = get_config(config_path_1)
    config_1['dataset_name'] = args.dataset
    print(config_1)

    config_path_2 = osp.join(args.pretrain_path_2, 'config' + '.yml')
    config_2 = get_config(config_path_2)
    config_2['dataset_name'] = args.dataset
    print(config_2)

    dataset = load_data(args, False)
    dataset.load(config_1)

    model_1 = CPRGsim(config_1, dataset.input_dim).cuda()
    para = osp.join(args.pretrain_path_1, 'CPRGsim_{}_checkpoint_mse.pth'.format(args.dataset))
    model_1.load_state_dict(torch.load(para, map_location='cuda:0'))
    model_1.eval()

    model_2 = CPRGsim(config_2, dataset.input_dim, True).cuda()
    para = osp.join(args.pretrain_path_2, 'CPRGsim_{}_checkpoint_mse.pth'.format(args.dataset))
    model_2.load_state_dict(torch.load(para, map_location='cuda:0'))
    model_2.eval()

    s_1, s_2, gt, gt_ged = evaluate(dataset.testing_graphs, dataset.trainval_graphs, model_1, model_2, dataset, config_1)
    loss_distribution(s_1, s_2, gt_ged)