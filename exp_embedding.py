from re import S
from statistics import mode
import numpy as np
# from rich import print
from model.GSC import GSC
from model.CPRGsim import CPRGsim
from model.MLP import Net
from argparse import ArgumentParser
from utils.utils import *
from utils.vis import vis_small, vis_graph_pair
import seaborn as sb
import matplotlib
import matplotlib.colors as mcolors
from torch_geometric.data import Batch
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from utils.color import *
# matplotlib.use('Agg')
# matplotlib.rc('font', **{'family': 'serif', 'size': 22})
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
matplotlib.rcParams['lines.linewidth'] = 0.6
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import os.path as osp
TRUE_MODEL = 'astar'
from scipy.stats import spearmanr, kendalltau
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

@torch.no_grad()
def evaluate(testing_graphs, training_graphs, model, dataset: DatasetLocal, config, emb_log=True):
    model.eval()

    scores                         = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth                   = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth_ged               = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth_nged              = np.empty((len(testing_graphs), len(training_graphs)))
    prediction_mat                 = np.empty((len(testing_graphs), len(training_graphs)))
    graph_embs                     = dict()
    node_embs                      = dict()
    for i_filter in range(model.num_filter):
        graph_embs[i_filter] = dict()
        for n in ['com_1', 'com_2', 'pri_1', 'pri_2', 'g1', 'g2', 'i_as', 'i_us']:
            graph_embs[i_filter][n] = list()
        node_embs[i_filter]  = dict()
        for n in ['g1', 'g2']:
            node_embs[i_filter][n] = list()

    rho_list                       = []
    tau_list                       = []
    prec_at_10_list                = []
    prec_at_20_list                = []

    num_test_pairs                 = len(testing_graphs) * len(training_graphs)
    t                              = tqdm(total=num_test_pairs)

    for i,g in enumerate(testing_graphs):
        source_batch               = Batch.from_data_list([g] * len(training_graphs))
        target_batch               = Batch.from_data_list(training_graphs)
        data                       = dataset.transform_batch((source_batch, target_batch), config)

        target                     = data["target"]
        ground_truth[i]            = target
        target_ged                 = data["target_ged"]
        ground_truth_ged[i]        = target_ged
        target_nged                = data["norm_ged"]
        ground_truth_nged[i]       = target_nged

        model.cp_generator.emb_log = emb_log
        prediction, loss_cl        = model(data)
        prediction_mat[i]          = prediction.cpu().detach().numpy()
        scores[i]                  = ( F.mse_loss(prediction.cpu().detach(), target, reduction="none").numpy())

        rho_list.append(
            calculate_ranking_correlation(
                spearmanr, prediction_mat[i], ground_truth[i]
            )
        )
        tau_list.append(
            calculate_ranking_correlation(
                kendalltau, prediction_mat[i], ground_truth[i]
            )
        )
        prec_at_10_list.append(
            calculate_prec_at_k(
                10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]
            )
        )
        prec_at_20_list.append(
            calculate_prec_at_k(
                20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]
            )
        )
        # if type(model.c_distri_list) is list:
        #     c_distri_list = [model.c_distri_list[i].cpu().detach().numpy() for i in range(model.num_filter)]
        # else:
        #     c_distri_list = model.c_distri_list.cpu().detach().numpy()
        # graph_cdistri_dicts.append(c_distri_list)

        if emb_log:
            for i_filter in range(model.num_filter):
                graph_embs[i_filter]['com_1'].append(model.com_1[i_filter].cpu().detach().numpy())
                graph_embs[i_filter]['com_2'].append(model.com_2[i_filter].cpu().detach().numpy())             
                graph_embs[i_filter]['pri_1'].append(model.pri_1[i_filter].cpu().detach().numpy())                  
                graph_embs[i_filter]['pri_2'].append(model.pri_2[i_filter].cpu().detach().numpy())
                graph_embs[i_filter]['i_as'].append(model.discriminator._ntn_score_onlyc[i_filter].cpu().detach().numpy())
                graph_embs[i_filter]['i_us'].append(model.discriminator._ntn_score_onlyp[i_filter].cpu().detach().numpy())
                graph_embs[i_filter]['g1'].append(model.cp_generator.g1_list[i_filter][0].cpu().detach().numpy())
                node_embs[i_filter]['g1'].append(model.cp_generator.nod1_list[i_filter][0][0].cpu().detach().numpy())
        t.update(len(training_graphs))

    if emb_log:
        for i_filter in range(model.num_filter):    
            embs = model.cp_generator.nod2_list[i_filter][0].cpu().detach().numpy()
            masks = model.cp_generator.nod2_list[i_filter][1].cpu().detach().numpy()
            new_embs = []
            for emb, mask in zip(embs, masks):
                new_embs.append(emb[mask])
            graph_embs[i_filter]['g2'] = model.cp_generator.g2_list[i_filter].cpu().detach().numpy()
            node_embs[i_filter]['g2'] = new_embs

    rho                            = np.mean(rho_list).item()
    tau                            = np.mean(tau_list).item()
    prec_at_10                     = np.mean(prec_at_10_list).item()
    prec_at_20                     = np.mean(prec_at_20_list).item()
    model_mse_error                = np.mean(scores).item()
    def print_evaluation(model_mse,test_rho,test_tau,test_prec_at_10,test_prec_at_20):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): "   + str(round(model_mse * 1000, 5)) + ".")
        print("Spearman's rho: " + str(round(test_rho, 5)) + ".")
        print("Kendall's tau: "  + str(round(test_tau, 5)) + ".")
        print("p@10: "           + str(round(test_prec_at_10, 5)) + ".")
        print("p@20: "           + str(round(test_prec_at_20, 5)) + ".")
    print_evaluation(model_mse_error, rho, tau, prec_at_10, prec_at_20)   

    return scores, ground_truth, ground_truth_ged, ground_truth_nged, prediction_mat, graph_embs, node_embs
    
def loss_sim_distribution(scores, ground_truth_ged, ground_truth, prediction_mat):
    ged_max = int(np.max(ground_truth_ged))
    ged_min = int(np.min(ground_truth_ged))
    ged = []
    loss = []
    for i in range(ged_min, ged_max+1):
        find = np.where(ground_truth_ged == i, scores, 0.0)
        find_num = len(np.nonzero(find)[0])
        find_average = find.sum()/find_num
        ged.append('{}/{}'.format(i, find_num))
        loss.append(find_average)

    nged_max = 1
    nged_min = 0
    nged = []
    nloss = []
    nstep = 20
    findnumlist = []
    step = (nged_max - nged_min)/nstep

    for i in range(nstep):
        find = np.where((ground_truth>=i*step) & (ground_truth<(i+1)*step), scores, 0.0)
        find_num = len(np.nonzero(find)[0])
        findnumlist.append(find_num)
        find_average = find.sum()/find_num
        nged.append('{:.1f}-{:.1f}/{}'.format(i*step, (i+1)*step, find_num))
        nloss.append(find_average)

    nged = []
    ngt = []
    for i in range(nstep):
        find = np.where((ground_truth>=i*step) & (ground_truth<(i+1)*step), prediction_mat, 0.0)
        find_num = len(np.nonzero(find)[0])
        find_average = find.sum()/find_num
        nged.append('{:.1f}-{:.1f}/{}'.format(i*step, (i+1)*step, find_num))
        ngt.append(find_average)

    fig, (ax_ged, ax_nged) = plt.subplots(1, 2, figsize=(14.4, 4.8))

    ax_ged.bar(ged, loss, width=0.5)
    ax_ged.tick_params(axis='x', labelrotation=90)
    ax_ged.set_ylabel('loss average')
    ax_ged.set_xlabel('ged/num')
    ax_ged.set_title('{:.5f} Loss related to the distribution of GED'.format(scores.mean()))

    ax_nged.bar(nged, nloss, width=0.5)
    ax_nged.tick_params(axis='x', labelrotation=90)
    ax_nged.set_ylabel('loss average')
    ax_nged.set_xlabel('nged/num')
    ax_nged.set_title('{:.5f} Loss related to the distribution of truth'.format(scores.mean()))
    ax_nged.set_ylim(0, 0.014)
    
    _, mode_dir = osp.split(args.pretrain_path)
    exp_figure_name = 'loss_distribution'

    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), exp_figure_name)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(nged, ngt, width=0.5)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_ylabel('pre average')
    ax.set_xlabel('nged/num')
    ax.set_title('{:.5f} gt related to the distribution of GED'.format(scores.mean()))
    
    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), 'pre_distribution')
    plt.close()

    fig, (ax_ged, ax_nged) = plt.subplots(1, 2, figsize=(14.4, 4.8))

    ax_ged.bar(nged, findnumlist, width=0.5)
    ax_ged.tick_params(axis='x', labelrotation=90)
    ax_ged.set_ylabel('GED average')
    ax_ged.set_xlabel('ged/num')
    ax_ged.set_title('GED distribution')

    ax_nged.bar(nged, nloss, width=0.5)
    ax_nged.tick_params(axis='x', labelrotation=90)
    ax_nged.set_ylabel('loss average')
    ax_nged.set_xlabel('nged/num')
    ax_nged.set_title('loss distribution')

    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), 'GED_loss_distribution')
    plt.close()

def GED_distribution(ground_truth):
    fig, (ax_1, ax_2) = plt.subplots(1, 2)

    ax_1.hist(ground_truth.flatten(), 50)
    ax_1.set_xlabel('similarity')
    ax_1.set_title('testdatabase similarity distribution')

    train_nged_list = dataset.trainval_nged_matrix[0:len(dataset.training_graphs), 0:len(dataset.training_graphs)].cpu().detach().numpy().flatten()
    ax_2.hist(np.exp(-train_nged_list), 50)
    ax_2.set_title('traindatabase similarity distribution')
    _, mode_dir = osp.split(args.pretrain_path)
    exp_figure_name = 'sim_distribution'

    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), 'sim_distribution')
    plt.tight_layout()
    plt.close()
    
def compri_dist_l2(ground_truth_ged, graph_embs_dicts, dataset):
    len_trival                     = len(dataset.trainval_graphs)
    sort_id_mat                    = np.argsort(ground_truth_ged,  kind = 'mergesort')
    test_gidlist                   = [10, 20, 30, 40, 50, 60]
    filter_list                    = [0,1,2,3]
    graph_embs_dist                = dict()
    for i_filter in range(model.num_filter):
        graph_embs_dist[i_filter]  = dict()
        for name in ['com_1', 'com_2', 'pri_1', 'pri_2']:
             graph_embs_dist[i_filter][name] \
                                   = dict()

    for i_filter in filter_list:
        for test_id in test_gidlist:
            com_dist_1             = list()
            com_dist_2             = list()
            pri_dist_1             = list()
            pri_dist_2             = list()
            for traval_id in sort_id_mat[test_id]:
                com_dist_1         . append(np.linalg.norm(graph_embs_dicts[i_filter]['com_1'][test_id][traval_id]))
                com_dist_2         . append(np.linalg.norm(graph_embs_dicts[i_filter]['com_2'][test_id][traval_id]))
                pri_dist_1         . append(np.linalg.norm(graph_embs_dicts[i_filter]['pri_1'][test_id][traval_id]))
                pri_dist_2         . append(np.linalg.norm(graph_embs_dicts[i_filter]['pri_2'][test_id][traval_id]))

            graph_embs_dist[i_filter]['com_1'][test_id] \
                                   = com_dist_1
            graph_embs_dist[i_filter]['com_2'][test_id] \
                                   = com_dist_2                                    
            graph_embs_dist[i_filter]['pri_1'][test_id] \
                                   = pri_dist_1 
            graph_embs_dist[i_filter]['pri_2'][test_id] \
                                   = pri_dist_2
    
    for i_filter in filter_list:
        for test_id in test_gidlist:
            fig, ([ax_c1, ax_c2], [ax_p1, ax_p2]) \
                                   = plt.subplots(2, 2, figsize=(14.4, 9.6))

            ax_c1.fill_between(list(range(len_trival)), \
                               min(graph_embs_dist[i_filter]['com_1'][test_id]), \
                               graph_embs_dist[i_filter]['com_1'][test_id], \
                               alpha=0.7)
            ax_c2.fill_between(list(range(len_trival)), \
                               min(graph_embs_dist[i_filter]['com_2'][test_id]), \
                               graph_embs_dist[i_filter]['com_2'][test_id], \
                               alpha=0.7)
            ax_p1.fill_between(list(range(len_trival)), \
                               min(graph_embs_dist[i_filter]['pri_1'][test_id]), \
                               graph_embs_dist[i_filter]['pri_1'][test_id], \
                               alpha=0.7)
            ax_p2.fill_between(list(range(len_trival)), \
                               min(graph_embs_dist[i_filter]['pri_2'][test_id]), \
                               graph_embs_dist[i_filter]['pri_2'][test_id], \
                               alpha=0.7)
            for ax_i, name in zip([ax_c1, ax_c2, ax_p1, ax_p2], ['com_1', 'com_2', 'pri_1', 'pri_2']):
                ax_i.set_ylabel('emb_dist_l2')
                ax_i.set_title(name)
            fig.suptitle('test_{}\' com and pri dist_l2 to all traval'.format(test_id), fontsize=20)

            _, mode_dir = osp.split(args.pretrain_path)
            exp_figure_name = 'compri_dist_l2'
            dir_ = osp.join('img', mode_dir, exp_figure_name, 'filter_{}'.format(i_filter))
            img_name = 'test_{}_compri_dist_l2'.format(test_id)

            save_fig(plt, dir_, img_name)
            plt.close()
def emb_hist(ground_truth_ged, graph_embs_dicts):
    nbins                          = 30
    sort_id_mat                    = np.argsort(ground_truth_ged,  kind = 'mergesort')
    gidraw                         = [30, 60, 90]
    rankcol                        = [0, 100, 200, 300,]  
    filter_list                    = [0,1,2,3]

    for i_filter in filter_list:
        for test_id in gidraw:
            for traval_id in sort_id_mat[test_id][rankcol]:
                fig, ([ax_c1, ax_c2], [ax_p1, ax_p2]) = plt.subplots(2, 2, figsize=(14.4, 9.6), tight_layout=True)
                ax_c1.hist(graph_embs_dicts[i_filter]['com_1'][test_id][traval_id], bins=nbins)
                ax_c2.hist(graph_embs_dicts[i_filter]['com_2'][test_id][traval_id], bins=nbins)
                ax_p1.hist(graph_embs_dicts[i_filter]['pri_1'][test_id][traval_id], bins=nbins)
                ax_p2.hist(graph_embs_dicts[i_filter]['pri_2'][test_id][traval_id], bins=nbins)
                
                fig.suptitle('{}_{}_{}\' com and pri emb hist'.format(test_id, traval_id, i_filter), fontsize=20)
                _, mode_dir = osp.split(args.pretrain_path)
                exp_figure_name = 'emb_hist'
                img_name = '{}_{}_{}'.format(test_id, traval_id, i_filter)
                save_fig(plt, osp.join('img', mode_dir, exp_figure_name), img_name)
                plt.close()

def compri_distri_distribution(scores, ground_truth, graph_cdistri_dicts):
    truth_max = 1
    truth_min = 0
    x_truth = []
    nstep = 20
    step = (truth_max - truth_min)/nstep
    graph_cdistri_dict = dict()
    graph_cdistri_var_dict = dict()
    for i_filter in range(model.num_filter):
        graph_cdistri_dict[i_filter] = list()
        graph_cdistri_var_dict[i_filter] = list()

    if type(graph_cdistri_dicts[0]) is list:
        filter_list = [0,1,2,3]
    else:
        filter_list = [0]

    for i_filter in filter_list:
        c_distri = list()
        for i in range(nstep):
            index = np.where((ground_truth>=i*step) & (ground_truth<(i+1)*step))
            for index_x, index_y in zip(index[0], index[1]):
                if type(graph_cdistri_dicts[0]) is list:
                    c_distri.append(graph_cdistri_dicts[index_x][i_filter][index_y])
                else:
                    c_distri.append(graph_cdistri_dicts[index_x][index_y])
            if i_filter == 0:
                x_truth.append('{:.2f}-{:.2f}/{}'.format(i*step, (i+1)*step, len(c_distri)))
            graph_cdistri_dict[i_filter].append(np.mean(c_distri))
            graph_cdistri_var_dict[i_filter].append(np.var(c_distri))
    
    colorlist = ['#00a8e1', '#99cc00', '#e30039', '#fcd300']
    fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=(14.4, 4.8))
    for i_filter in filter_list:
        if type(graph_cdistri_dicts[0]) is list:
            label_ = 'filter_{}'.format(str(i_filter))
        else:
            label_ = None
        ax_mean.plot(x_truth,graph_cdistri_dict[i_filter],color=colorlist[i_filter],label=label_)
        ax_var.plot(x_truth,graph_cdistri_var_dict[i_filter],color=colorlist[i_filter],label=label_)
    ax_mean.tick_params(axis='x', labelrotation=90)
    ax_mean.set_ylabel('c_distri mean')
    ax_mean.set_xlabel('truth/num')
    ax_mean.set_title('{:.5f} the distri of c_distri mean related to truth'.format(scores.mean()))
    ax_mean.legend()
    ax_var.tick_params(axis='x', labelrotation=90)
    ax_var.set_ylabel('c_distri var')
    ax_var.set_xlabel('truth/num')
    ax_var.set_title('{:.5f} the distri of c_distri var related to truth'.format(scores.mean()))
    ax_var.legend()
    
    _, mode_dir = osp.split(args.pretrain_path)
    exp_figure_name = 'c_distri_distribution'

    save_fig(plt, osp.join('img', mode_dir, exp_figure_name), exp_figure_name)

def compri_sim(ground_truth_ged, graph_embs_dicts):
    len_trival                     = len(dataset.trainval_graphs)
    sort_id_mat                    = np.argsort(ground_truth_ged,  kind = 'mergesort')
    test_gidlist                   = [10, 20, 30, 40, 50, 60]
    filter_list                    = [0,1,2,3]
    graph_embs_dist                = dict()
    for i_filter in range(model.num_filter):
        graph_embs_dist[i_filter]  = dict()
        for name in ['com_sim', 'selfcp1_sim', 'selfcp2_sim', 'eachpri_sim', 'eachcp12_sim', 'eachcp21_sim']:
             graph_embs_dist[i_filter][name] \
                                   = dict()
    for i_filter in filter_list:
        for test_id in test_gidlist:
            com_sim                = list()
            selfcp1_sim            = list()
            selfcp2_sim            = list()
            eachpri_sim            = list()
            eachcp12_sim           = list()
            eachcp21_sim           = list() 
            for traval_id in sort_id_mat[test_id]:
                com_sim            . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_1'][test_id][traval_id]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['com_2'][test_id][traval_id]), dim=-1))
                
                selfcp1_sim        . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_1'][test_id][traval_id]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_1'][test_id][traval_id]), dim=-1))
                
                selfcp2_sim        . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_2'][test_id][traval_id]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_2'][test_id][traval_id]), dim=-1))
                
                eachpri_sim        . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['pri_1'][test_id][traval_id]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_2'][test_id][traval_id]), dim=-1))
                
                eachcp12_sim       . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_1'][test_id][traval_id]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_2'][test_id][traval_id]), dim=-1))
                
                eachcp21_sim       . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_2'][test_id][traval_id]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_1'][test_id][traval_id]), dim=-1))
                                
            graph_embs_dist[i_filter]['com_sim'][test_id] \
                                   = com_sim
            graph_embs_dist[i_filter]['selfcp1_sim'][test_id] \
                                   = selfcp1_sim
            graph_embs_dist[i_filter]['selfcp2_sim'][test_id] \
                                   = selfcp2_sim
            graph_embs_dist[i_filter]['eachpri_sim'][test_id] \
                                   = eachpri_sim
            graph_embs_dist[i_filter]['eachcp12_sim'][test_id] \
                                   = eachcp12_sim
            graph_embs_dist[i_filter]['eachcp21_sim'][test_id] \
                                   = eachcp21_sim
            
    for i_filter in filter_list:
        for test_id in test_gidlist:
            fig, ax                = plt.subplots(figsize=(7.2, 4.8))
            ax.plot(list(range(len_trival)),
                    graph_embs_dist[i_filter]['com_sim'][test_id], 
                    color='#00a8e1',
                    label="com_sim")
            ax.plot(list(range(len_trival)),
                    graph_embs_dist[i_filter]['selfcp1_sim'][test_id], 
                    color='#99cc00',
                    label="selfcp1_sim")
            ax.plot(list(range(len_trival)),
                    graph_embs_dist[i_filter]['selfcp2_sim'][test_id], 
                    color='#e30039',
                    label="selfcp2_sim")
            ax.plot(list(range(len_trival)),
                    graph_embs_dist[i_filter]['eachpri_sim'][test_id], 
                    color='#fcd300',
                    label="eachpri_sim")
            ax.plot(list(range(len_trival)),
                    graph_embs_dist[i_filter]['eachcp12_sim'][test_id], 
                    color='#800080',
                    label="eachcp12_sim")
            ax.plot(list(range(len_trival)),
                    graph_embs_dist[i_filter]['eachcp21_sim'][test_id], 
                    color='#00994e',
                    label="eachcp21_sim")
            ax.legend()

            ax.set_title('test_{}_filter_{}\' emb sim to all traval'.format(test_id, i_filter))
            _, mode_dir = osp.split(args.pretrain_path)
            exp_figure_name = 'emb_sim'
            dir_ = osp.join('img', mode_dir, exp_figure_name, 'filter_{}'.format(i_filter))
            img_name = 'test_{}_emb_sim'.format(test_id)

            save_fig(plt, dir_, img_name)
            plt.close()
def compri_sim_distribution(ground_truth_ged, graph_embs_dicts):
    ged_max = int(np.max(ground_truth_ged))
    ged_min = int(np.min(ground_truth_ged))
    filter_list                    = [0,1,2,3]
    graph_embs_dist                = dict()
    for i_filter in range(model.num_filter):
        graph_embs_dist[i_filter]  = dict()
        for name in ['com_sim', 'selfcp1_sim', 'selfcp2_sim', 'eachpri_sim', 'eachcp12_sim', 'eachcp21_sim']:
             graph_embs_dist[i_filter][name] \
                                   = list()
             
    for i_filter in filter_list:
        for i in range(ged_min, ged_max+1):
            com_sim                = list()
            selfcp1_sim            = list()
            selfcp2_sim            = list()
            eachpri_sim            = list()
            eachcp12_sim           = list()
            eachcp21_sim           = list()
            for index in np.argwhere(ground_truth_ged == i):
                com_sim            . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_1'][index[0]][index[1]]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['com_2'][index[0]][index[1]]), dim=-1))

                selfcp1_sim        . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_1'][index[0]][index[1]]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_1'][index[0]][index[1]]), dim=-1))

                selfcp2_sim        . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_2'][index[0]][index[1]]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_2'][index[0]][index[1]]), dim=-1))
                                                                    
                eachpri_sim        . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['pri_1'][index[0]][index[1]]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_2'][index[0]][index[1]]), dim=-1))

                eachcp12_sim       . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_1'][index[0]][index[1]]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_2'][index[0]][index[1]]), dim=-1))

                eachcp21_sim       . append(F.cosine_similarity(torch.Tensor(graph_embs_dicts[i_filter]['com_2'][index[0]][index[1]]),
                                                                torch.Tensor(graph_embs_dicts[i_filter]['pri_1'][index[0]][index[1]]), dim=-1))

            graph_embs_dist[i_filter]['com_sim'] \
                                   . append(np.mean(com_sim))
            graph_embs_dist[i_filter]['selfcp1_sim'] \
                                   . append(np.mean(selfcp1_sim))
            graph_embs_dist[i_filter]['selfcp2_sim'] \
                                   . append(np.mean(selfcp2_sim))
            graph_embs_dist[i_filter]['eachpri_sim'] \
                                   . append(np.mean(eachpri_sim))
            graph_embs_dist[i_filter]['eachcp12_sim'] \
                                   . append(np.mean(eachcp12_sim))
            graph_embs_dist[i_filter]['eachcp21_sim'] \
                                   . append(np.mean(eachcp21_sim))
              
    colorlist                      = ['#00a8e1', '#99cc00', '#e30039', '#fcd300']
    for name in ['com_sim', 'selfcp1_sim', 'selfcp2_sim', 'eachpri_sim', 'eachcp12_sim', 'eachcp21_sim']:
        fig, ax                    = plt.subplots(figsize=(7.2, 4.8))
        for i_filter in filter_list:
            ax.plot(list(range(ged_min, ged_max+1)),
                    graph_embs_dist[i_filter][name], 
                    color=colorlist[i_filter],
                    label='filter_{}'.format(str(i_filter)))
        ax.legend()
        ax.set_ylabel('emb_cossim')
        ax.set_xlabel('ged')
        ax.set_title('{} emb cossim related to the distribution of GED'.format(name))
        _, mode_dir = osp.split(args.pretrain_path)
        exp_figure_name = 'sim_distribution'
        dir_ = osp.join('img', mode_dir, exp_figure_name)
        img_name = '{}_sim_distribution'.format(name)

        save_fig(plt, dir_, img_name)
        plt.close()

def nodecp_sim_matrix_hist_heat(prediction_mat, graph_embs_dicts, node_embs_dicts):
    sort_id_mat_pre = np.argsort(prediction_mat,  kind = 'mergesort')[:, ::-1]
    gidraw = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 , 10 ,11,22, 21, 76,64]
    rankcol = [1,]  
    filter_list = [0,1,2,3]

    for i_filter in filter_list:
        for test_id in gidraw:
            for traval_id in sort_id_mat_pre[test_id][rankcol]:
                node1_emb = node_embs_dicts[i_filter]['g1'][test_id][0][traval_id][node_embs_dicts[i_filter]['g1'][test_id][1][traval_id]]
                node2_emb = node_embs_dicts[i_filter]['g2'][test_id][0][traval_id][node_embs_dicts[i_filter]['g2'][test_id][1][traval_id]]
                cp1_emb = np.array([graph_embs_dicts[i_filter]['com_1'][test_id][traval_id], 
                                    graph_embs_dicts[i_filter]['pri_1'][test_id][traval_id]])
                cp2_emb = np.array([graph_embs_dicts[i_filter]['com_2'][test_id][traval_id], 
                                    graph_embs_dicts[i_filter]['pri_2'][test_id][traval_id]])
                n1n2_cos_matrix = cosine_similarity(node1_emb, node2_emb)
                n1g2_cos_matrix = cosine_similarity(node1_emb, cp2_emb)
                g1n2_cos_matrix = cosine_similarity(cp1_emb, node2_emb)

                fig = plt.figure(figsize=(6, 6))
                # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
                # the size of the marginal Axes and the main Axes in both directions.
                # Also adjust the subplot parameters for a square plot.
                gs = fig.add_gridspec(2, 2,  width_ratios=(len(node1_emb), 2), height_ratios=(2, len(node2_emb)),
                                      left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
                # Create the Axes.
                n1n2_ax = fig.add_subplot(gs[1, 0])
                n1g2_ax = fig.add_subplot(gs[1, 1], sharey=n1n2_ax)
                g1n2_ax = fig.add_subplot(gs[0, 0], sharex=n1n2_ax)
                
                for ax in [n1n2_ax, n1g2_ax, g1n2_ax]:
                    ax.label_outer()
                heatmap(n1g2_cos_matrix, None, ['C2', 'P2'], ax=n1g2_ax, cmap="YlGn")
                heatmap(g1n2_cos_matrix, ['C1', 'P1'], None, ax=g1n2_ax, cmap="YlGn")
                heatmap(n1n2_cos_matrix, ["G%i"% i for i in range(len(node1_emb))], ["g%i"% i for i in range(len(node2_emb))], ax=n1n2_ax, cmap="YlGn",)
                # fig.tight_layout()
                _, mode_dir = osp.split(args.pretrain_path)
                exp_figure_name = 'sim_matrix_hist_heat'
                dir_ = osp.join('img', mode_dir, exp_figure_name)
                img_name = '{}_{}_filter_{}'.format(test_id, traval_id, i_filter)

                save_fig(plt, dir_, img_name)
                plt.close()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    if row_labels:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticks([])
    if col_labels:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
    else:
        ax.set_xticks([])
        
    ax.spines[:].set_visible(False)
    # ax.set(aspect=1)
    annotate_heatmap(im)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
def cpembedding_singular(graph_embs_dicts):
    for i_filter in range(model.num_filter):
        for i in range(len(graph_embs_dicts[i_filter]['com_1'])):
            c = _singular(graph_embs_dicts[i_filter]['com_1'][i])
    return c

def _singular(feature):
    # feature = torch.cat(feature, dim=0)

    # z = torch.nn.functional.normalize(feature, dim=1)

    # # calculate covariance
    # z = z.cpu().detach().numpy()
    z = np.transpose(feature)
    c = np.cov(z)
    _, d, _ = np.linalg.svd(c)
    return d

def get_true_result(all_graphs, testing_graphs, trainval_graphs, sim_or_dist = 'dist'):
    ged_matrix                                  = trainval_graphs.ged
    nged_matrix                                 = trainval_graphs.norm_ged
    ground_truth_ged                            = np.empty((len(testing_graphs), len(trainval_graphs)))   # test graph 和 train-val graph 之间的距离
    ground_truth_nged                           = np.empty((len(testing_graphs), len(trainval_graphs))) 
    for test_id, test_g in  enumerate(testing_graphs):
        for tv_id, tv_g in  enumerate(trainval_graphs):
            ground_truth_ged[test_id][tv_id]    = ged_matrix[test_g['i'], tv_g['i']]
            ground_truth_nged[test_id][tv_id]   = nged_matrix[test_g['i'], tv_g['i']]
    print(ground_truth_ged.shape)
    sort_id_mat                                 = np.argsort(ground_truth_ged,  kind = 'mergesort')
    sort_id_mat_normed                          = np.argsort(ground_truth_nged, kind = 'mergesort')
    if sim_or_dist == 'sim':
        sort_id_mat                             = sort_id_mat[:, ::-1]
        sort_id_mat_normed                      = sort_id_mat_normed[:, ::-1]
    return ground_truth_ged, ground_truth_nged, sort_id_mat, sort_id_mat_normed

def get_pred_result(pred_mat, sim_or_dist = 'sim'):
    # pred_mat 为相似度矩阵   (140, 560)  = sim_mat
    sort_id_mat = np.argsort(pred_mat, kind='mergesort')[:, ::-1]   # 每行按照相似度从大到小排序
    return sort_id_mat


def copy_unnormalized_sim_mat(sim_mat, testing_graphs, trainval_graphs):
    rtn = np.copy(sim_mat)
    m, n = sim_mat.shape
    for i in range(m):
        for j in range(n):
            rtn[i][j] = unnormalized_dist_sim(rtn[i][j], testing_graphs[i], trainval_graphs[j])
    return rtn

def unnormalized_dist_sim(d, g1, g2):
    g1_size = g1.num_nodes
    g2_size = g2.num_nodes

    return torch.log(d) * (g1_size + g2_size) / 2

def draw_emb_hist_heat(args, testing_graphs, trainval_graphs, model: GSC, extra_dir=None, plot_max_num=np.inf):
    extra_dir = osp.join(args.extra_dir, args.dataset, 'mne')
    all_graphs = testing_graphs + trainval_graphs
    emb_layers_dicts = model.collect_embeddings(all_graphs)
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    colors = ['Oranges', 'Blues', 'Greens']
    for gnn_id in sorted(emb_layers_dicts.keys()):
        nel = emb_layers_dicts[gnn_id]
        cmap_color = colors[gnn_id % len(colors)]
        draw_emb_hist_heat_helper(gnn_id, nel, cmap_color, args.dataset, testing_graphs, trainval_graphs, sort_id_mat_normed, ground_truth_nged, True, plot_max_num, extra_dir)


def draw_emb_hist_heat_helper(gnn_id, nel, cmap_color, dataset_name,
                              test_graphs, trainval_graphs, ids, ground_truth_nged, ds_norm,
                              plot_max_num, extra_dir):

    plt_cnt = 0
    for i, tv in tqdm(enumerate(test_graphs)):
        sort = [1, 10, 50, 100, len(trainval_graphs)]
        gids = [int(ids[i][:sort[0]]), int(ids[i][sort[1]]), int(ids[i][sort[2]]), int(ids[i][sort[3]]) ,int(ids[i][-1:])]
        for s, j in enumerate(gids):
            d = ground_truth_nged[i][j]
            query_nel_idx = int(tv['i'])                   
            match_nel_idx = int(j)        
            result = cosine_similarity(nel[int(query_nel_idx)].cpu().detach().numpy(),  nel[int(match_nel_idx)].cpu().detach().numpy())
            if s == 0:
                vmax = result.max()
            plt.figure()
            sb_plot = sb.heatmap(result, fmt='d', cmap=cmap_color, vmax = vmax)
            fig = sb_plot.get_figure()
            dir = '{}/{}'.format(extra_dir, 'heatmap')
            fn  = '{}_{}_{}_sort{}_gcn{}'.format(i, j, d, sort[s],gnn_id)
            plt_cnt += save_fig(fig, dir, fn, print_path=False)
            if extra_dir:
                plt_cnt += save_fig(fig, extra_dir + '/heatmap', fn, print_path=False)
            plt.close()
            result_array = []
            for m in range(len(result)):  
                for n in range(len(result[m])):
                    result_array.append(result[m][n])
            plt.figure()
            sb_plot = sb.distplot(result_array, bins=16, color='r',
                                                kde=False, rug=False, hist=True, )
            sb_plot.set(xlabel='Similarity', ylabel='Num of Nodes')
            fig = sb_plot.get_figure()
            dir = '{}/{}'.format(extra_dir, 'histogram')
            fn = '{}_{}_{}_gcn{}'.format(i, j, d, gnn_id)                                    
            plt_cnt += save_fig(fig, dir, fn, print_path=False)
            if extra_dir:
                plt_cnt += save_fig(fig, extra_dir + '/histogram', fn, print_path=False)
            plt.close()
        if plt_cnt > plot_max_num:
            print('Saved {} node embeddings mne plots for gcn{}'.format(plt_cnt, gnn_id))
            return
    print('Saved {} node embeddings mne plots for gcn{}'.format(plt_cnt, gnn_id))            


def visualize_embeddings_gradual(args, testing_graphs, trainval_graphs, model: GSC, plot_max_num=np.inf,  concise=False , sort = True, perplexity = None):
    extra_dir = osp.join(args.extra_dir, args.dataset, 'emb_vis_gradual')
    all_graphs = testing_graphs + trainval_graphs
    graph_emb_layers = model.collect_graph_embeddings(all_graphs)
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    for gnn_id in graph_emb_layers.keys():
        if gnn_id == 3:
            print(gnn_id)
        if sort:
            tsne = TSNE(n_components=2, perplexity=10)  # , method='barnes_hut'
        else:
            if perplexity is None:
                tsne = TSNE(n_components=2)
            else:
                tsne = TSNE(n_components=2, perplexity=perplexity)
        g_emb_gnn_i = graph_emb_layers[gnn_id]
        vec = []
        ids = []
        for i, v in g_emb_gnn_i.items():   
            vec.append(v.cpu().detach().numpy())
            ids.append(i)
        vec = np.array(vec)
        embs = tsne.fit_transform(vec)
        plt_cnt = 0
        if not concise:
            print('TSNE embeddings: {} --> {} to plot'.format(
                vec.shape, embs.shape))
        # plot_embeddings_for_gs_by_glabel(embs, testing_graphs, trainval_graphs, extra_dir)        
        plot_embeddings_for_each_query(embs, ids, plt_cnt, ground_truth_nged, sort_id_mat_normed,extra_dir,
                                    plot_max_num, gnn_id ,trainval_graphs, testing_graphs,concise, sort, perplexity)

def plot_cp_embeddings(ground_truth, graph_embs_dicts, dataset):
    len_trival                     = len(dataset.trainval_graphs)
    len_test                       = len(dataset.testing_graphs)
    sort_id_mat                    = np.argsort(-ground_truth,  kind = 'mergesort')             
    filter_list                    = [0,1,2,3]
    tsne = TSNE(n_components=2, perplexity=30, metric='cosine', learning_rate = 400)
    extra_dir = osp.join('img', osp.split(args.pretrain_path)[1], 'plot_cp_emb')
    slice_percent = 0.1  # 定义我们要取的切片占总数的百分比 (5%)
    slice_size = int(len_trival * slice_percent)
    middle_slice_start_index = int(len_trival * 0.5)
    

    for qid in range(len_test):
        for gnn_id in filter_list:
            # cg, mg, fg = [], [], []
            sorted_pids = sort_id_mat[qid]

            cg = list(sorted_pids[:slice_size])
            mg = list(sorted_pids[middle_slice_start_index : middle_slice_start_index + slice_size])
            fg = list(sorted_pids[-slice_size:])

            _plot_cp_embeddings(qid, cg, mg, fg, graph_embs_dicts, gnn_id, tsne, extra_dir, sort_id_mat, ground_truth)

# def _plot_cp_embeddings(qid, cg, mg, fg, embs, gnn_id, tsne, extra_dir, sort_id_mat, s_size=10):
#     emb = np.array([embs[gnn_id]['com_1'][qid],
#                     embs[gnn_id]['pri_1'][qid],
#                     embs[gnn_id]['com_2'][qid],
#                     embs[gnn_id]['pri_2'][qid]])
#     emb = tsne.fit_transform(emb.reshape(-1, emb.shape[-1]))
#     emb = emb.reshape(4, -1, emb.shape[-1])
#     c1, p1, c2, p2 = emb[0][sort_id_mat[qid]], emb[1][sort_id_mat[qid]], emb[2][sort_id_mat[qid]], emb[3][sort_id_mat[qid]]
#     plt.figure()
#     plt.scatter(c1[:,0], c1[:,1], s=s_size,
#                 c=sorted(range(len(c1)), reverse=False),
#                 marker='s', cmap=plt.cm.get_cmap('Blues'))
#     plt.scatter(c2[:,0], c2[:,1], s=s_size,
#                 c=sorted(range(len(c2)), reverse=False),
#                 marker='o', cmap=plt.cm.get_cmap('Reds'))
#     plt.scatter(p1[:,0], c1[:,1], s=s_size,
#                 c=sorted(range(len(p1)), reverse=False),
#                 marker='s', cmap=plt.cm.get_cmap('Greens'))
#     plt.scatter(p1[:,0], p2[:,1], s=s_size,
#                 c=sorted(range(len(p2)), reverse=False),
#                 marker='o', cmap=plt.cm.get_cmap('Oranges'))
#     # plt.scatter(p1[cg][0], p1[cg][1], s=20,
#     #             color = 'red',
#     #             marker='s')
#     cur_axes = plt.gca()
#     cur_axes.axes.get_xaxis().set_visible(False)
#     cur_axes.axes.get_yaxis().set_visible(False)

#     exp_figure_name = '{}_gcn{}'.format(qid, gnn_id)

#     save_fig(plt, extra_dir, exp_figure_name)

#     plt.close()
def _plot_cp_embeddings(qid, cg, mg, fg, embs, gnn_id, tsne, extra_dir, sort_id_mat, gt, s_size=5):
    # --------------------------
    # 1. 提取嵌入并进行t-SNE降维
    # --------------------------
    def _concatenate(embs, sub, qid):
        arrays = [embs[i][sub][qid] for i in range(2)]
        return np.concatenate(arrays, axis=1)

    emb = np.array([
        embs[gnn_id]['com_1'][qid],
        embs[gnn_id]['pri_1'][qid],
        embs[gnn_id]['com_2'][qid],
        embs[gnn_id]['pri_2'][qid]
    ])
    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(emb)
    # emb = np.array([
    #     _concatenate(embs, 'com_1', qid),
    #     _concatenate(embs, 'pri_1', qid),
    #     _concatenate(embs, 'com_2', qid),
    #     _concatenate(embs, 'pri_2', qid)
    # ])
    # 展平后降维，再恢复原结构（4类嵌入，每类形状为[num_samples, 2]）
    emb_flat = emb.reshape(-1, emb.shape[-1])  # 展平为[4*N, D]
    emb_tsne = tsne.fit_transform(emb_flat)    # t-SNE降维到2D
    emb_2d = emb_tsne.reshape(4, -1, 2)        # 恢复为[4, N, 2]

    # 提取四类嵌入的t-SNE结果（按原逻辑）
    c1, p1, c2, p2 = emb_2d[0][sort_id_mat[qid]], emb_2d[1][sort_id_mat[qid]], \
                     emb_2d[2][sort_id_mat[qid]], emb_2d[3][sort_id_mat[qid]]

    # --------------------------
    # 2. 计算cg/mg/fg的聚类中心和半径
    # --------------------------
    def get_cluster_stats_robust(points):
        if len(points) == 0:
            return None, None
        
        # --- 1. 确定中心点 (Medoid) ---
        # 计算点集中任意两点之间的欧几里得距离矩阵
        distance_matrix = squareform(pdist(points, 'euclidean'))
        
        # 计算每个点到其他所有点的距离总和
        sum_of_distances = np.sum(distance_matrix, axis=1)
        
        # 找到距离总和最小的点的索引，这个点就是Medoid
        medoid_index = np.argmin(sum_of_distances)
        center = points[medoid_index]
        
        # --- 2. 确定聚类半径 (Average Distance) ---
        # 计算所有点到 Medoid 的距离
        # 我们可以直接从距离矩阵中提取这一列/行的数据，无需重复计算
        distances_to_medoid = distance_matrix[:, medoid_index]
        
        # 计算这些距离的平均值
        radius = np.mean(distances_to_medoid)
        
        return center, radius
    def get_cluster_stats_density(points, eps=4.5, min_samples=5):
        """
        基于DBSCAN密度聚类，仅用核心样本计算统计量
        :param eps: DBSCAN邻域半径
        :param min_samples: 核心点最小邻居数
        :return: 核心样本的中心点，半径
        """
        if len(points) < min_samples:
            return get_cluster_stats_robust(points)
        
        # 1. DBSCAN聚类（-1为噪声点）
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        
        # 2. 筛选核心样本（非噪声点）
        core_mask = labels != -1
        core_points = points[core_mask]
        
        # 3. 若核心样本太少，回退到稳健方法
        if len(core_points) < 3:
            return None, None
        
        # 4. 用核心样本计算稳健统计
        return get_cluster_stats_robust(core_points)
    def get_cluster_stats_with_outlier_removal(points, z_threshold=1.5):
        """
        先剔除异常点，再计算核心样本的聚类统计
        :param points: 聚类样本点，形状为[N, 2]
        :param z_threshold: Z-score阈值（>阈值视为异常点）
        :return: 核心样本的中心点，半径
        """
        if len(points) < 3:  # 样本太少时不剔除异常点
            return get_cluster_stats_robust(points)  # 直接用稳健方法
        
        # 1. 计算每个样本到均值中心的距离（临时中心，用于初步筛选）
        temp_center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - temp_center, axis=1)
        
        # 2. Z-score识别异常点（距离偏离均值太远的为异常点）
        z_scores = (distances - np.mean(distances)) / np.std(distances)
        core_mask = z_scores <= z_threshold  # 核心样本掩码（非异常点）
        core_points = points[core_mask]
        
        # 3. 若过滤后样本太少，回退到原始样本
        if len(core_points) < 3:
            core_points = points
        
        # 4. 用核心样本计算稳健统计（中位数+分位数）
        return get_cluster_stats_robust(core_points)        
        # 以c1为例计算三类样本的聚类 stats（如需其他嵌入，可替换c1为p1/c2/p2）
    c1_all = p1  # c1的所有样本点
    # 提取cg/mg/fg在c1中的对应点（cg/mg/fg是样本索引）
    c1_cg = c1_all[cg] if cg else np.array([])
    c1_mg = c1_all[mg] if mg else np.array([])
    c1_fg = np.array([])

    # 计算三类的中心和半径
    center_cg, radius_cg = get_cluster_stats_robust(c1_cg)
    center_mg, radius_mg = get_cluster_stats_robust(c1_mg)
    center_fg, radius_fg = get_cluster_stats_robust(c1_fg)

    # --------------------------
    # 3. 绘制散点图和聚类半径
    # --------------------------
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # 绘制原始散点（保持原逻辑）
    plt.scatter(c1[:,0], c1[:,1], s=s_size,
                c=gt[qid][sort_id_mat[qid]],
                marker='s', cmap=plt.cm.get_cmap('Blues'), alpha=0.6, label='Has_1')
    plt.scatter(c2[:,0], c2[:,1], s=s_size,
                c=gt[qid][sort_id_mat[qid]],
                marker='o', cmap=plt.cm.get_cmap('Reds'), alpha=0.6, label='Has_2')
    plt.scatter(p1[:,0], p1[:,1], s=s_size,  # 原代码此处笔误：c1[:,1]→p1[:,1]
                c=gt[qid][sort_id_mat[qid]],
                marker='s', cmap=plt.cm.get_cmap('Greens'), alpha=0.6, label='Hus_1')
    plt.scatter(p2[:,0], p2[:,1], s=s_size,  # 原代码此处笔误：p1[:,0]→p2[:,0]
                c=gt[qid][sort_id_mat[qid]],
                marker='o', cmap=plt.cm.get_cmap('Oranges'), alpha=0.6, label='Hus_2')

    # # 绘制cg/mg/fg的聚类半径（用不同颜色的圆圈）
    # if center_cg is not None:
    #     circle_cg = Circle(center_cg, radius_cg, fill=False, edgecolor='blue', 
    #                       linestyle='-', linewidth=2, label='CG Radius')
    #     ax.add_patch(circle_cg)
    # if center_mg is not None:
    #     circle_mg = Circle(center_mg, radius_mg, fill=False, edgecolor='purple', 
    #                       linestyle='--', linewidth=2, label='MG Radius')
    #     ax.add_patch(circle_mg)
    # if center_fg is not None:
    #     circle_fg = Circle(center_fg, radius_fg, fill=False, edgecolor='red', 
    #                       linestyle='-.', linewidth=2, label='FG Radius')
    #     ax.add_patch(circle_fg)
    # 绘制cg/mg/fg的聚类半径（圆圈）
    if center_cg is not None:
        # 绘制圆圈
        circle_cg = Circle(center_cg, radius_cg, fill=False, edgecolor='blue', 
                          linestyle='-', linewidth=2, label='Similar Graphs')
        ax.add_patch(circle_cg)

    if center_mg is not None:
        # 绘制圆圈
        circle_mg = Circle(center_mg, radius_mg, fill=False, edgecolor='purple', 
                          linestyle='--', linewidth=2, label='MG Radius')
        ax.add_patch(circle_mg)

    if center_fg is not None:
        # 绘制圆圈
        circle_fg = Circle(center_fg, radius_fg, fill=False, edgecolor='red', 
                          linestyle='-.', linewidth=2, label='Dissimilar Graphs')
        ax.add_patch(circle_fg)

    # --------------------------
    # 4. 美化与保存
    # --------------------------
    plt.legend(loc='best', fontsize=10)
    ax.axes.get_xaxis().set_visible(False)  # 隐藏坐标轴
    ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

    # 创建保存目录并保存图片
    os.makedirs(extra_dir, exist_ok=True)
    save_path = osp.join(extra_dir, f'qid_{qid}_gnn_{gnn_id}_with_radius.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved to {}'.format(save_path))    

def plot_embeddings_for_each_query(embs, ids, plt_cnt,ground_truth_nged, sort_id_mat_normed, extra_dir,
                                   plot_max_num, gnn_id, trainval_graphs, testing_graphs, concise, sort, perplexity):
    emb_dict = dict()
    for k in range(embs.shape[0]):
        emb_dict[ids[k]] = embs[k]
    m = np.shape(sort_id_mat_normed)[0]
    n = np.shape(sort_id_mat_normed)[1]
    if sort:
        plot_what = 'emb_vis_query'
    else:
        if perplexity is None:
            plot_what = 'emb_vis_query_unsort'
        else:
            plot_what = 'emb_vis_query_unsort_{}'.format(perplexity)
    
    for i in tqdm(range(m)):  
        axis_x_red = []
        axis_y_red = []
        axis_x_blue = []
        axis_y_blue = []
        axis_x_query = []
        axis_y_query = []
        red_number = []
        blue_number = []
        for j in range(n):   
            if ranking(i, j, sort_id_mat_normed, ground_truth_nged) < n / 2:  
                red_number.append((j, ground_truth_nged[i][j]))   
            else:
                blue_number.append((j, ground_truth_nged[i][j]))
        sorted(red_number, key=lambda x: x[1])
        sorted(blue_number, key=lambda x: x[1], reverse=True)
        for j in range(len(red_number)):  
            axis_x_red.append(emb_dict[int(trainval_graphs[red_number[j][0]]['i'])][0])
            axis_y_red.append(emb_dict[int(trainval_graphs[red_number[j][0]]['i'])][1])
        for j in range(len(blue_number)):
            axis_x_blue.append(emb_dict[int(trainval_graphs[blue_number[j][0]]['i'])][0])
            axis_y_blue.append(emb_dict[int(trainval_graphs[blue_number[j][0]]['i'])][1])
        axis_x_query.append(emb_dict[int(testing_graphs[i]['i'])][0])
        axis_y_query.append(emb_dict[int(testing_graphs[i]['i'])][1])
        # Plot.
        plt.figure()
        if sort:
            plt.scatter(axis_x_blue, axis_y_blue, s=20,
                        c=sorted(range(len(axis_x_blue)), reverse=False),
                        marker='s', cmap=plt.cm.get_cmap('Blues'))
            plt.scatter(axis_x_red, axis_y_red, s=20,
                        c=sorted(range(len(axis_x_red)), reverse=False),
                        marker='o', cmap=plt.cm.get_cmap('Reds'))
            plt.scatter(axis_x_query, axis_y_query, s=400, c='green',
                        marker='P', alpha=0.8)
        else:
            plt.scatter(axis_x_blue, axis_y_blue, s=20,
                        color = 'blue',
                        marker='s')
            plt.scatter(axis_x_red, axis_y_red, s=20,
                        c='red',
                        marker='o')
            plt.scatter(axis_x_query, axis_y_query, s=400, c='green',
                        marker='P')
        # plt.axis('on')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        fn = str(i)
        plt_cnt += save_fig(plt, '{}/{}/gnn_{}'.format(extra_dir, plot_what, gnn_id), fn)
        # if extra_dir:
        #     plt_cnt += save_fig(plt, '{}/{}/gnn_{}/query_'.format(extra_dir, plot_what, gnn_id), fn)
        plt.close()
        if plt_cnt > plot_max_num:
            if not concise:
                print('Saved {} embedding visualization plots'.format(plt_cnt))
            return
    if not concise:
        print('Saved {} embedding visualization plots'.format(plt_cnt))


def ranking(qid, gid, sort_id_mat_normed, ground_truth_nged, one_based=True):

    # Assume self is ground truth.
    sort_id_mat = sort_id_mat_normed

    finds = np.where(sort_id_mat[qid] == gid)  
    assert (len(finds) == 1 and len(finds[0]) == 1)
    fid = finds[0][0]   
    dist_sim_mat = ground_truth_nged
    while fid > 0:
        cid = sort_id_mat[qid][fid]      
        pid = sort_id_mat[qid][fid - 1]  
        if dist_sim_mat[qid][pid] == dist_sim_mat[qid][cid]: 
            fid -= 1
        else:
            break
    if one_based:
        fid += 1
    return fid

def plot_embeddings_for_gs_by_glabel(embs, test_gs, train_gs, extra_dir):
    points = []
    for i, g in enumerate(train_gs + test_gs):
        emb = embs[i]  
        assert (emb.shape == (2,))
        points.append({'x': emb[0], 'y': emb[1], 'glabel': g.graph['glabel']})
    plot_embeddings_for_points(points, 'all_gs', dir, extra_dir)
    plot_embeddings_for_points(points[0:len(train_gs)], 'train_gs', dir, extra_dir)
    plot_embeddings_for_points(points[len(train_gs):], 'test_gs', dir, extra_dir)


def plot_embeddings_for_points(points, what_gs,extra_dir):
    plot_what = 'emb_vis_glabel'
    markers = ('D', 's', 'P', 'v', '^', '*', '<', '>', 'p', 'o')
    colors = ('red', 'blue', 'green')
    cmaps = plt.cm.get_cmap('tab20')
    plt.figure()
    for p in points:
        x = p['x']
        y = p['y']
        glabel = p['glabel']
        # glabel = np.random.randint(1, 16)
        assert (type(glabel) is int and glabel >= 0)
        if glabel < 3:
            c = colors[glabel]
        else:
            c = cmaps(glabel)
        m = markers[glabel % len(markers)]
        plt.scatter(x, y, c=c, marker=m)
    plt.axis('off')
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    fn = what_gs
    save_fig(plt, '{}/{}'.format(dir, plot_what), fn)
    if extra_dir:
        save_fig(plt, '{}/{}'.format(extra_dir, plot_what), fn)
    plt.close()





def classification_mat(thresh_pos, thresh_neg,ground_truth_nged, norm = True):  
    snorm = ('norm' if norm else 'unnorm')
    vname = 'classif_mat_{}_{}_{}'.format(thresh_pos, thresh_neg, snorm)
    dist_mat = ground_truth_nged
    label_mat, num_poses, num_negs, _, _ = get_classification_labels_from_dist_mat(dist_mat, thresh_pos, thresh_neg)
    rtn = (label_mat, num_poses, num_negs)
    return rtn

def visualize_embeddings_binary(args, testing_graphs, trainval_graphs, model: GSC, norm = True, perplexity = None):
    extra_dir = osp.join(args.extra_dir, args.dataset, 'emb_vis_binary')
    if args.dataset == 'AIDS700nef' or args.dataset == 'LINUX' or args.dataset == 'IMDBMulti':
        thresh_pos = 0.95
        thresh_neg = 0.95
        thresh_pos_sim = 0.5
        thresh_neg_sim = 0.5
    all_graphs = testing_graphs + trainval_graphs
    graph_emb_layers = model.collect_graph_embeddings(all_graphs)
    ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    label_mat, _, _ = classification_mat(thresh_pos, thresh_neg, ground_truth_nged, norm)
    for gnn_id in graph_emb_layers.keys():
        if gnn_id == 3:
            print(gnn_id)

        if perplexity is None:
            tsne = TSNE(n_components=2)
        else:
            tsne = TSNE(n_components=2, perplexity=perplexity)
        g_emb_gnn_i = graph_emb_layers[gnn_id]
        vec = []
        ids = []
        for i, v in g_emb_gnn_i.items():   
            vec.append(v.cpu().detach().numpy())
            ids.append(i)
        vec = np.array(vec)
        embs = tsne.fit_transform(vec)
        create_dir_if_not_exists(extra_dir)
        m = np.shape(label_mat)[0]
        n = np.shape(label_mat)[1]
        plt_cnt = 0
        print('TSNE embeddings: {} --> {} to plot'.format(
            vec.shape, embs.shape))


        if perplexity is None:
            plot_what = 'emb_vis_binary'
        else:
            plot_what = 'emb_vis_binary_{}'.format(perplexity)
    
        emb_dict = dict()
        for k in range(embs.shape[0]):
            emb_dict[ids[k]] = embs[k]

        for j in range(m):
            axis_x_red = []
            axis_y_red = []
            axis_x_blue = []
            axis_y_blue = []
            axis_x_query = []
            axis_y_query = []
            for i in range(n):
                if label_mat[j][i] == -1:
                    axis_x_blue.append(emb_dict[int(trainval_graphs[i]['i'])][0])
                    axis_y_blue.append(emb_dict[int(trainval_graphs[i]['i'])][1])
                else:
                    axis_x_red.append(emb_dict[int(trainval_graphs[i]['i'])][0])
                    axis_y_red.append(emb_dict[int(trainval_graphs[i]['i'])][1])
            axis_x_query.append(emb_dict[int(testing_graphs[j]['i'])][0])
            axis_y_query.append(emb_dict[int(testing_graphs[j]['i'])][1])

            plt.figure()
            plt.scatter(axis_x_red, axis_y_red, s=20, c='red', marker='o')
            plt.scatter(axis_x_blue, axis_y_blue, s=20, c='blue',
                        marker='o')
            plt.scatter(axis_x_query, axis_y_query, s=300, c='green', marker='X', alpha=0.7)
            plt.axis('off')
            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_visible(False)
            cur_axes.axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            fn = str(j)
            # plt.savefig(dir + '/' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
            plt_cnt += save_fig(plt, '{}/{}/gnn_{}'.format(extra_dir, plot_what, gnn_id), fn)
            plt_cnt += 1
            plt.close()
        print('Saved {} embedding visualization plots'.format(plt_cnt))


def get_text_label_for_ranking(ds_metric, qid, j, norm, is_query, dataset, ranktype, 
                               rank, score_matrix, unnormed_matrix=None, normed_matrix=None, type='metric', selected_ids=None):
    # norm = True
    rtn = ''
    gid = rank[j]
    if type == 'metric':
        if ds_metric == 'ged':  # here
            if norm:
                ds_label = 'nGED'
            else:
                ds_label = 'GED'
        elif ds_metric == 'glet':
            ds_label = 'glet'
        elif ds_metric == 'mcs':
            if norm:
                ds_label = 'nMCS'
            else:
                ds_label = 'MCS'
        else:
            raise NotImplementedError()
        if is_query:  
            if ds_metric == 'mcs':
                rtn += '{} by\n'.format(ds_label)  # nGED by \n A*
            else:
                if 'AIDS' in dataset or dataset == 'LINUX':
                    rtn += '{} by\nA*'.format(ds_label)
                else:
                    # rtn += '{} by Beam-\nHungarian-VJ'.format(ds_label)
                    rtn += '{} by \nB-H-V'.format(ds_label)
        else:  # 
            if ranktype== 'gt':
                if norm:
                    ds = normed_matrix[qid][gid]
                else:
                    ds = unnormed_matrix[qid][gid]
            elif ranktype== 'pred':
                if norm:
                    ds = -1 * np.log(score_matrix[qid][gid]-1e-10)
                else:
                    raise NotImplementedError()
            rtn = '{:.2f}'.format(ds)
    elif type == 'rank':
        text = selected_ids[j]+1
        if j == len(rank) - 3:
            rtn = '... rank {} ...'.format(text)
        else:
            rtn = 'rank {}'.format(text)
    return rtn

def _get_rank_text(rank, rank_list, g2_len):
    if rank == len(rank_list) - 3:
        rtn = '{}'.format(int(g2_len / 2))
    elif rank == len(rank_list) - 2:
        rtn = '{}'.format(int(g2_len-1))
    elif rank == len(rank_list) - 1:
        rtn = '{}'.format(int(g2_len))
    else:
        rtn = '{}'.format(str(rank + 1))
    return rtn

def get_text_metric_from(ds_metric, norm):
    if ds_metric == 'ged':
        if norm:
            ds_label = 'nGED'
        else:
            ds_label = 'GED'
    elif ds_metric == 'glet':
        ds_label = 'glet'
    elif ds_metric == 'mcs':
        if norm:
            ds_label = 'nMCS'
        else:
            ds_label = 'MCS'
    else:
        raise NotImplementedError()
    return ds_label

def get_ged_select_norm_str(unnormed_matrix,normed_matrix, qid, gid, norm):
    # ged = r.dist_sim(qid, gid, norm=False)[1]
    ged = unnormed_matrix[qid][gid]   
    # norm_ged = r.dist_sim(qid, gid, norm=True)[1]
    norm_ged  =  normed_matrix[qid][gid]
    if norm:
        return '{:.2f}({})'.format(norm_ged, int(ged))
    else:
        return '{}({:.2f})'.format(int(ged), norm_ged)

def format_ds(ds):
    return '{:.2f}'.format(ds)

def get_norm_str(norm):
    if norm is None:
        return ''
    elif norm:
        return '_norm'
    else:
        return '_nonorm'

def set_save_paths_for_vis(info_dict, extra_dir, fn, plt_cnt):
    info_dict['plot_save_path_pdf'] = '{}/{}.{}'.format(extra_dir, fn, 'png')
    plt_cnt += 1
    return info_dict, plt_cnt

def draw_ranking(args, testing_graphs, trainval_graphs,
                 gt, gt_GED, gt_nGED, pred_mat, existing_mappings = None, plot_gids=False, verbose=True, plot_max_num=np.inf, model_path = None, color = True, plot_node_ids = True):
    plot_what = 'ranking'
    concise = True
    c = None
    if color:
        c = 'color'
    else:
        c = 'gray'
    extra_dir = osp.join(args.extra_dir, args.dataset, plot_what, model_path, c)
    # em = self._get_node_mappings(data)
    # plot_node_ids= args.dataset != 'webeasy' and em
    types = trainval_graphs.types if args.dataset == 'AIDS700nef' else None
    if args.dataset == 'AIDS700nef':
        if color:
            color_map = get_color_map(trainval_graphs + testing_graphs, trainval_graphs)
        else:
            color_map = get_color_map(trainval_graphs + testing_graphs, trainval_graphs, use_color=color)
    elif args.dataset == 'LINUX':
        color_map = COLOR1
    elif args.dataset == 'IMDBMulti':
        color_map = COLOR2

    info_dict = {
        # draw node config
        'draw_node_size': 20,
        'draw_node_label_enable': True,
        'show_labels': plot_node_ids,
        'node_label_type': 'label' if plot_node_ids else 'type',
        'node_label_name': 'type' if args.dataset == 'AIDS700nef' else None,
        'draw_node_label_font_size': 6,
        'draw_node_color_map': color_map ,
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_title_list': [],
        'each_graph_text_font_size': 10,
        'each_graph_title_pos': [0.5, 1.1],
        'each_graph_text_pos': [0.5, -0.2],
        'each_graph_text_from_pos': [0.5, -0.4],
        # graph padding: value range: [0, 1]
        'top_space': 0.1 if concise else 0.26,  # out of whole graph
        'bottom_space': 0.15,
        'hbetween_space': 0.5 if concise else 1,  # out of the subgraph
        'wbetween_space': 0.01,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': '',
        'plot_save_path_pdf': ''
    }
    plt_cnt = 0
    all_graphs = testing_graphs + trainval_graphs
    # ground_truth_ged , ground_truth_nged, sort_id_mat, sort_id_mat_normed = get_true_result(all_graphs, testing_graphs, trainval_graphs)
    pred_ids_rank = get_pred_result(pred_mat)  # sim from big to small 
    for i in range(len(testing_graphs)):
        q = testing_graphs[i]
        middle_idx = len(trainval_graphs) // 2
        if len(trainval_graphs) < 5:
            print('Too few train gs {}'.format(len(trainval_graphs)))
            return        
        # Choose the top 6 matches, the overall middle match, and the worst match.
        selected_ids = list(range(5))  
        selected_ids.extend([middle_idx, len(trainval_graphs)-2, len(trainval_graphs)-1])   
        # Get the selected graphs from the groundtruth and the model.
        sort_id_mat_normed = np.argsort(gt_nGED, kind = 'mergesort')
        gids_groundtruth = np.array(sort_id_mat_normed[i][selected_ids])   
        gids_rank = np.array(pred_ids_rank[i][selected_ids])              
        # Top row graphs are only the groundtruth outputs.
        gs_groundtruth = [trainval_graphs[j] for j in gids_groundtruth]  # 
        # Bottom row graphs are the query graph + model ranking.
        gs_rank = [] 
        gs_rank = gs_rank + [trainval_graphs[j] for j in gids_rank]  
        gs = gs_groundtruth + gs_rank  

        # Create the plot labels.
        text = []
        text += [get_text_label_for_ranking(
                            'ged', i, j, True, False, args.dataset, 'gt', gids_groundtruth, gt, gt_GED, gt_nGED, type='metric')
                            for j in range(len(gids_groundtruth))]
        text += [get_text_label_for_ranking(
                            'ged', i, j, True, False, args.dataset, 'pred', gids_rank, pred_mat, type='metric')
                            for j in range(len(gids_rank))]
        info_dict['each_graph_text_list'] = text

        text = []
        text.append('Query')
        text += [get_text_label_for_ranking(
                    'ged', i, j, True, False, args.dataset, None, 
                    gids_groundtruth, gt, gt_GED, gt_nGED, type='rank', selected_ids=selected_ids)
                    for j in range(len(gids_rank))]
        info_dict['each_graph_title_list'] = text

        text_metric = get_text_metric_from('ged', True)
        info_dict['each_graph_text_from_gt'] = 'Ground-truth {}'.format(text_metric)
        info_dict['each_graph_text_from_pred'] = 'Predicted {} by {}'.format(text_metric, args.model_name)

        fn = '{}_{}_{}{}'.format(
            plot_what, 'astar', int(testing_graphs[i]['i']), get_norm_str(True))  # ranking_astar_i_norm
        info_dict, plt_cnt = set_save_paths_for_vis(
            info_dict, extra_dir, fn, plt_cnt)    

        vis_small(q, gs, info_dict, types)
        if plt_cnt > plot_max_num:
            print('Saved {} query demo plots'.format(plt_cnt))
            return
    print('Saved {} query demo plots'.format(plt_cnt))


def get_color_map(gs, trainval_graphs, use_color = True):
    fl = len(SET_COLORS)
    rtn = {}
    ntypes = defaultdict(int) 
    types = trainval_graphs.types
    for g in gs:  
        for node in range(g.num_nodes):   
            node_type_idx = int(np.where(g.x[node].numpy() != 0)[0])
            node_type_name = types[node_type_idx]
            ntypes[node_type_name] += 1

    secondary = {}
    for i, (ntype, cnt) in enumerate(sorted(ntypes.items(), key=lambda x: x[1],
                                            reverse=True)):
        if ntype is None:
            color = None
            rtn[ntype] = color
        elif i >= fl:
            cmaps = plt.cm.get_cmap('hsv')
            color = cmaps((i - fl) / (len(ntypes) - fl))
            secondary[ntype] = color if use_color else mcolors.to_rgba(SET_COLORS[0])
        else:
            color = mcolors.to_rgba(SET_COLORS[i])[:3]
            rtn[ntype] = color if use_color else mcolors.to_rgba(SET_COLORS[0])
    if secondary:
        rtn.update((secondary))
    return rtn

def _get_sim_score_for_embsandemb(embs, emb, method='cosine'):
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    dot_product = np.dot(embs, emb.T)  # (n,m) × (m,k) → (n,k)
    if method == 'dot':
        return dot_product
    elif method == 'cosine':
        norm_embs = np.linalg.norm(embs, axis=1, keepdims=True)  # (n,1)
        norm_emb = np.linalg.norm(emb, axis=1, keepdims=True)  # (k,1)
        cosine_similarity = dot_product / (norm_embs * norm_emb.T + 1e-10)
        return cosine_similarity
    elif method == 'l2':
        return -np.linalg.norm(embs-emb, axis=1, keepdims=True)
     
def row_averages_keep_shape(matrix):
    row_means = np.mean(matrix, axis=1, keepdims=True)
    
    result = np.broadcast_to(row_means, matrix.shape)
    
    return result
    
def _norm_map(list1, list2=None):
    if list2 == None:
        for i in range(len(list1)):
            x = list1[i]
            x_nor = (x - x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)+1e-10)
            list1[i] = x_nor
        return list1
    else:
        for i in range(len(list1)): 
            x = np.concatenate([list1[i], list2[i]])
            x_nor = (x - x.min(axis=0))/(x.max(axis=0)-x.min(axis=0)+1e-10)
            x_nor = x_nor.reshape(-1, 2)
            list1[i], list2[i] = x_nor[:,0], x_nor[:,1]
        return list1, list2

def draw_mapping(args, testing_graphs, trainval_graphs, 
                gt_nGED, g_embs, n_embs, map_type,
                plot_max_num=np.inf, color = True, plot_node_ids = True):
    if map_type == 'gncm':
        plot_what = 'gncmmap'
    elif map_type == 'psgd':
        plot_what = 'psgdmap'

    types = trainval_graphs.types if args.dataset == 'AIDS700nef' else None
    if args.dataset == 'AIDS700nef':
        if color:
            color_map = get_color_map(trainval_graphs + testing_graphs, trainval_graphs)
        else:
            color_map = get_color_map(trainval_graphs + testing_graphs, trainval_graphs, use_color=color)
    elif args.dataset == 'LINUX':
        color_map = COLOR1
    elif args.dataset == 'IMDBMulti':
        color_map = COLOR2

    node_mapdcit = LinearSegmentedColormap('mapdcit', MAPDICT)
    info_dict = {
        # draw node config
        'draw_node_size': 20,
        'draw_node_label_enable': True,
        'show_labels': plot_node_ids,
        'node_label_type': 'label' if plot_node_ids else 'type',
        'node_label_name': 'type' if args.dataset == 'AIDS700nef' else None,
        'draw_node_label_font_size': 6,
        'draw_node_color_map': color_map ,
        'draw_node_color_mapdcit': node_mapdcit,
        'get_map_mothed': 'l2',
        'map_norm': True,
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_title_list': [],
        'each_graph_text_font_size': 10,
        'each_graph_title_pos': [0.5, 1.05],
        'each_graph_text_pos': [0.5, -1.36],
        'bar_text': 'Similarity Value',
        'bar_text_font_size': 8,
        # graph padding: value range: [0, 1]
        'left_space': 0.01,
        'right_space': 0.909,
        'top_space': 0.08,
        'bottom_space': 0.09,
        'hbetween_space': 0.05,
        'wbetween_space': 0.05,
        'subplot_size':0.9,
        'bar_sie':0.05,
        'curlyBrace_size':0.12,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': '',
        'plot_save_path_pdf': ''
    }
    mothed = info_dict['get_map_mothed']
    extra_dir = osp.join(args.extra_dir, args.dataset, plot_what, mothed)
    text = []
    for i in range(len(g_embs)):
        text.append('Layer {}'.format(i+1))
    info_dict['each_graph_title_list'] = text
    info_dict['each_graph_text_list'] = ['Original\nGraphs', 'Similarity Heatmap']
    plt_cnt = 0
    selected_ids = [0, 100, 200, 400, 400, 500]
    
    for g1_id in range(len(testing_graphs)):
        g1 = testing_graphs[g1_id]
        middle_idx = len(trainval_graphs) // 2
        if len(trainval_graphs) < 5:
            print('Too few train gs {}'.format(len(trainval_graphs)))
            return        
        sort_id_mat_normed = np.argsort(gt_nGED, kind = 'mergesort')
        gids_groundtruth = np.array(sort_id_mat_normed[g1_id][selected_ids])

        for i, g2_id in enumerate(gids_groundtruth):
            g2 = trainval_graphs[g2_id]

            rank_text = 'rank{}'.format(selected_ids[i]+1)
            fn = '{}_{}_{}_{}_{}'.format(plot_what, info_dict['get_map_mothed'],
                                        'norm' if info_dict['map_norm'] else '',
                                        int(testing_graphs[g1_id]['i']),
                                        rank_text, 
                                        int(trainval_graphs[g2_id]['i']))
            info_dict, plt_cnt = set_save_paths_for_vis(info_dict, extra_dir, fn, plt_cnt)

            if map_type == 'gncm':
                node1_maplist = []
                node2_maplist = []
                for layer_i in range(len(g_embs)):
                    g1_graph_emb = g_embs[layer_i]['g1'][g1_id]
                    g2_graph_emb = g_embs[layer_i]['g2'][g2_id]
                    g1_node_emb = n_embs[layer_i]['g1'][g1_id]
                    g2_node_emb = n_embs[layer_i]['g2'][g2_id]
                    if len(g1_node_emb) != g1.num_nodes or len(g2_node_emb) != g2.num_nodes:
                        raise NotImplementedError()
                    node1_maplist.append(_get_sim_score_for_embsandemb(g1_node_emb, g2_graph_emb, mothed))
                    node2_maplist.append(_get_sim_score_for_embsandemb(g2_node_emb, g1_graph_emb, mothed))
                vis_graph_pair(g1, g2, info_dict, types, node1_maplist, node2_maplist)

            elif map_type == 'psgd':
                node1_c2_maplist = []
                node1_p2_maplist = []
                node2_c1_maplist = []
                node2_p1_maplist = []

                for layer_i in range(len(g_embs)):
                    c1_emb = g_embs[layer_i]['com_1'][g1_id][g2_id]
                    p1_emb = g_embs[layer_i]['pri_1'][g1_id][g2_id]
                    c2_emb = g_embs[layer_i]['com_2'][g1_id][g2_id]
                    p2_emb = g_embs[layer_i]['pri_2'][g1_id][g2_id]
                    g1_node_emb = n_embs[layer_i]['g1'][g1_id]
                    g2_node_emb = n_embs[layer_i]['g2'][g2_id]
                    # g1_node_emb = row_averages_keep_shape(g1_node_emb)
                    # g2_node_emb = row_averages_keep_shape(g2_node_emb)
                    if len(g1_node_emb) != g1.num_nodes or len(g2_node_emb) != g2.num_nodes:
                        raise NotImplementedError()
                    node1_c2_maplist.append(_get_sim_score_for_embsandemb(g1_node_emb, c2_emb, mothed))
                    node1_p2_maplist.append(_get_sim_score_for_embsandemb(g1_node_emb, p2_emb, mothed))
                    node2_c1_maplist.append(_get_sim_score_for_embsandemb(g2_node_emb, c1_emb, mothed))
                    node2_p1_maplist.append(_get_sim_score_for_embsandemb(g2_node_emb, p1_emb, mothed))


                # node1_c2_maplist = [np.mean(np.array(node1_c2_maplist), axis=0) for _ in node1_c2_maplist]
                # node1_p2_maplist = [np.mean(np.array(node1_p2_maplist), axis=0) for _ in node1_p2_maplist]
                # node2_c1_maplist = [np.mean(np.array(node2_c1_maplist), axis=0) for _ in node2_c1_maplist]
                # node2_p1_maplist = [np.mean(np.array(node2_p1_maplist), axis=0) for _ in node2_p1_maplist]

                if info_dict['map_norm']:
                    node1_c2_maplist = _norm_map(node1_c2_maplist)
                    node1_p2_maplist = _norm_map(node1_p2_maplist)
                    node2_c1_maplist = _norm_map(node2_c1_maplist)
                    node2_p1_maplist = _norm_map(node2_p1_maplist)
                vis_graph_pair(g1, g2, info_dict, types, 
                               node1_c2_maplist+node1_p2_maplist,
                               node2_c1_maplist+node2_p1_maplist)               

            if plt_cnt > plot_max_num:
                print('Saved {} query demo plots'.format(plt_cnt))
                return 
    pass

def MINE(args, config, graph_embs):
    ws = 50
    last_n = 400
    _, mode_dir = osp.split(args.pretrain_path)

    def gen_x_tensor(x):
        return torch.Tensor(x)
    def gen_y_tensor(y):
        return torch.Tensor(y)
    def moving_average(losses, window_size=5):
        return np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    model = torch.nn.ModuleList()
    for i in config['gnn_filters']:
        model.append(Net(dim=i))# 实例化模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # 使用Adam优化器并设置学习率为0.01
    
    n_epoch = 1500
    for layer_i in range(len(graph_embs)):
        mine_list = []
        exp_figure_name = 'MINE_layer_{}'.format(layer_i)
        for g1_id in range(len(graph_embs[layer_i]['com_1'])): # 遍历所有图对
            plot_loss = []
            model[layer_i].init_parameters()
            for _ in range(n_epoch):
                x_sample = gen_x_tensor(graph_embs[layer_i]['com_1'][g1_id])
                y_sample = gen_y_tensor(graph_embs[layer_i]['pri_1'][g1_id])
                y_shuffle = y_sample[torch.randperm(y_sample.size(0))]#将 y_sample按照批次维度打乱顺序得到y_shuffle
        
                model[layer_i].zero_grad()
                pred_xy = model[layer_i](x_sample, y_sample)  # 式(8-49）中的第一项联合分布的期望:将x_sample和y_sample放到模型中，得到联合概率（P(X,Y)=P(Y|X)P(X)）关于神经网络的期望值pred_xy。
                pred_x_y = model[layer_i](x_sample, y_shuffle)  # 式(8-49)中的第二项边缘分布的期望:将x_sample和y_shuffle放到模型中，得到边缘概率关于神经网络的期望值pred_x_y 。
        
                ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))) # 将pred_xy和pred_x_y代入式（8-49）中，得到互信息ret。
                loss = - ret  # 最大化互信息：在训练过程中，因为需要将模型权重向着互信息最大的方向优化，所以对互信息取反，得到最终的loss值。
                plot_loss.append(loss.data)  # 收集损失值
                loss.backward()  # 反向传播：在得到loss值之后，便可以进行反向传播并调用优化器进行模型优化。
                optimizer.step()  # 调用优化器            
            plot_y = np.array(plot_loss).reshape(-1, )
            plot_y_average = moving_average(-plot_y, window_size=ws)
            mine_list.append(np.mean(plot_y_average[-last_n:]))
            plt.plot(np.arange(len(plot_loss)), -plot_y, 'r')
            plt.plot(np.arange(len(plot_loss)-ws+1), plot_y_average, 'b')
            save_fig(plt, osp.join('img', mode_dir, exp_figure_name), '{}'.format(g1_id))
            plt.close()
        mine_arr = np.array(mine_list)
        mine_arr[mine_arr < 0] = 0
        with open(osp.join(osp.join('img', mode_dir), 'MINE.txt'), 'a') as f:
            f.write("Layer_{} {:.2f}±{:.2f}\n".format(layer_i, mine_arr.mean(), mine_arr.std()))
    # plt.plot(np.arange(len(mine_list)), np.array(mine_list), 'b')
    # plt.savefig('MINE.PNG')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset',           type = str,              default = 'IMDBMulti') 
    parser.add_argument('--data_dir',          type = str,              default = 'datasets/')
    parser.add_argument('--extra_dir',         type = str,              default = 'exp/')    
    parser.add_argument('--gpu_id',            type = int  ,            default = 0)
    parser.add_argument('--model_name',             type = str,              default = 'GSC_GNN')  # GCN, GAT or other
    parser.add_argument('--recache',         action = "store_true",        help = "clean up the old adj data", default=True)
    parser.add_argument('--pretrain_path',     type = str,              default = 'model_saved/IMDBMulti/2024-10-30/CPRGsim_IMDBMulti_woNGSA_0')
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
    model                       = CPRGsim(config, dataset.input_dim).cuda()
    para                        = osp.join(args.pretrain_path, 'CPRGsim_{}_checkpoint.pth'.format(args.dataset))
    model                       . load_state_dict(torch.load(para))
    model                       . eval()

    # draw_emb_hist_heat(args, dataset.testing_graphs, dataset.trainval_graphs, model)
    scores,                     \
    ground_truth,               \
    ground_truth_ged,           \
    ground_truth_nged,          \
    prediction_mat,             \
    graph_embs,                 \
    node_embs,                  = evaluate(dataset.testing_graphs, dataset.trainval_graphs, model, dataset, config, True)
    # MINE(args, config, graph_embs)
    plot_cp_embeddings(ground_truth, graph_embs, dataset)
    # compri_sim(ground_truth_ged, graph_embs_dicts)
    # compri_dist_l2(ground_truth_ged, graph_embs_dicts, dataset)
    # nodecp_sim_matrix_hist_heat(prediction_mat, graph_embs_dicts, node_embs_dicts)
    # draw_ranking(args, dataset.testing_graphs, dataset.trainval_graphs, 
    #             ground_truth, ground_truth_ged, ground_truth_nged,
    #             prediction_mat, None, model_path='', plot_node_ids=args.dataset=='AIDS700nef')
    # draw_mapping(args, dataset.testing_graphs, dataset.trainval_graphs, 
    #             ground_truth_nged, graph_embs, node_embs, 'psgd',
    #             plot_node_ids=args.dataset=='AIDS700nef')
    # cpembedding_singular(graph_embs_dicts)