import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict
from os.path import dirname
import math
from colour import Color
import numpy as np
from torch_geometric.utils import to_networkx
import os
from matplotlib.gridspec import GridSpec
from curlyBrace.curlyBrace import curlyBrace

def set_node_attr(g, types):
    node_attr = dict()
    if not types is None:
        for i in range(g.num_nodes):
            node_attr[i] = dict()
            nlabel = int(np.where(g.x[i].numpy() != 0)[0])
            ntype = types[nlabel]
            node_attr[i]['label'] = nlabel
            node_attr[i]['type'] = ntype
    else:
        for i in range(g.num_nodes):
            node_attr[i] = dict()
            node_attr[i]['label'] = None
            node_attr[i]['type'] = None
    return node_attr

def vis_graph_pair(g1, g2, info_dict, types, node1_maplist=None, node2_maplist=None):
    nx_g1 = to_networkx(g1, to_undirected=True)
    nx_g2 = to_networkx(g2, to_undirected=True)
    nx.set_node_attributes(nx_g1, set_node_attr(g1, types))
    nx.set_node_attributes(nx_g2, set_node_attr(g2, types))

    n = 1 + len(node1_maplist) if node1_maplist is not None else 1
    subplot_size = info_dict['subplot_size']
    wsize = info_dict['wbetween_space']*subplot_size
    hszie = info_dict['hbetween_space']*subplot_size
    fig_width = (subplot_size * n + wsize * n + subplot_size*info_dict['bar_sie'])/(info_dict['right_space'] - info_dict['left_space'])
    fig_hight = (subplot_size * 2 + hszie *2 + subplot_size*info_dict['curlyBrace_size'])/(1 - info_dict['top_space'] - info_dict['bottom_space'])

    wspace = wsize *(n+1) / (subplot_size * (n+info_dict['bar_sie']))
    hspace = hszie *(2+1) / (subplot_size * (2+info_dict['curlyBrace_size']))
    
    gs = GridSpec(3, n+1, width_ratios=[1]*n + [info_dict['bar_sie']], height_ratios=[1, 1, info_dict['curlyBrace_size']],
                  wspace=wspace, hspace=hspace,
                  bottom=info_dict['bottom_space'], top=1-info_dict['top_space'],
                  left=info_dict['left_space'], right=info_dict['right_space'])
    fig = plt.figure(figsize=(fig_width, fig_hight))

    for i in range(n):
        ax_up = fig.add_subplot(gs[0, i])
        ax_down = fig.add_subplot(gs[1, i])
        ax_up.axis('off')
        ax_down.axis('off')
        if i == 0:
            ifmap=False
        else:
            ifmap=True
        draw_graph_small(nx_g1, info_dict, ax_up, ifmap, node1_maplist[i-1])
        draw_graph_small(nx_g2, info_dict, ax_down, ifmap, node2_maplist[i-1])
        if i > 0:
            draw_extra(i, ax_up, info_dict,
                        _list_safe_get(info_dict['each_graph_title_list'], i-1 , ""), 'title')

    
    
    brace_ax = fig.add_subplot(gs[2, 1:5])
    origraph_ax = fig.add_subplot(gs[2, 0])
    brace_ax.axis('off')
    origraph_ax.axis('off')

    add_colorbar_to_fig(fig, gs, info_dict['draw_node_color_mapdcit'], 
                        label_default=info_dict['bar_text'], 
                        tick_label_size=info_dict['bar_text_font_size'])
    pos = brace_ax.get_position()
    # right_pos = brace_ax.get_position()
    brace_ax.set_xlim(pos.x1, pos.x0)
    curlyBrace(fig, brace_ax, 
                (pos.x1, pos.y1),
                (pos.x0, pos.y1), 
               k_r=0.2, color='black', linewidth=1)
    
    x = _list_safe_get(info_dict['each_graph_text_pos'], 0, 0.5)
    y = _list_safe_get(info_dict['each_graph_text_pos'], 1, 0.8)  
    origraph_ax.text(x, y, 
            info_dict['each_graph_text_list'][0], 
            fontsize=info_dict['each_graph_text_font_size'], 
            ha='center', transform=origraph_ax.transAxes)
    brace_ax.text(x, y, 
            info_dict['each_graph_text_list'][1], 
            fontsize=info_dict['each_graph_text_font_size'], 
            ha='center', transform=brace_ax.transAxes)    
    _save_figs(info_dict)
    pass      

def vis_small(q=None, gs=None, info_dict=None, types = None):
    n = int(len(gs)//2) + 1
    subplot_size = info_dict['subplot_size']
    wsize = info_dict['wbetween_space']*subplot_size
    hszie = info_dict['hbetween_space']*subplot_size
    wspace = wsize / subplot_size 
    hspace = hszie / subplot_size 
    fig_width = (subplot_size * n + wsize * (n-1))/(info_dict['right_space'] - info_dict['left_space'])
    fig_hight = (subplot_size * 2 + hszie *1)/(1 - info_dict['top_space'] - info_dict['bottom_space'])
    fig = plt.figure(figsize=(fig_width, fig_hight))
    _info_dict_preprocess(info_dict)
    nx_q = to_networkx(q, to_undirected=True)
    
    nx.set_node_attributes(nx_q, set_node_attr(q, types))
    nx_gs = []
    for g in gs:
        nx_g = to_networkx(g, to_undirected=True)
        nx.set_node_attributes(nx_g, set_node_attr(g, types))
        nx_gs.append(nx_g)

    # get num
    graph_num = 2 + len(nx_gs)   
    plot_m, plot_n = _calc_subplot_size_small(graph_num)  

    # draw query graph
    # info_dict['each_graph_text_font_size'] = 9
    ax_q, ax_p, grid= set_ax(plot_m, plot_n, fig) # ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph_small(nx_q, info_dict, ax_q,)
    draw_extra(0, ax_q, info_dict,
               _list_safe_get(info_dict['each_graph_title_list'], 0, ""), 'title')

    # draw graph candidates
    # info_dict['each_graph_text_font_size'] = 12
    for i in range(len(nx_gs)):
        # ax = plt.subplot(plot_m, plot_n, i + 2)
        m = i//(plot_n-1)
        n = i%(plot_n-1)
        draw_graph_small(nx_gs[i], info_dict, ax_p[m][n])
        draw_extra(i, ax_p[m][n], info_dict,
                _list_safe_get(info_dict['each_graph_text_list'], i , ""), 'text')
        if m == 0:
            draw_extra(i, ax_p[m][n], info_dict,
                    _list_safe_get(info_dict['each_graph_title_list'], i+1 , ""), 'title')


    # # plot setting
    # # plt.tight_layout()
    # left = 0.01  # the left side of the subplots of the figure
    # right = 0.99  # the right side of the subplots of the figure
    # top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    # bottom = \
    #     info_dict['bottom_space']  # the bottom of the subplots of the figure
    # wspace = \
    #     info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    # hspace = \
    #     info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=info_dict['left_space'], bottom=info_dict['bottom_space'], 
                        right=info_dict['right_space'], top=1-info_dict['top_space'],
                        wspace=wspace, hspace=hspace)
    draw_bottom_title(fig, grid, info_dict)
    axq_adjust(ax_q)
    # save / display
    _save_figs(info_dict)
    pass



def axq_adjust(ax_q):
    l, b, w, h = ax_q.get_position().bounds
    ax_q.set_position((l, b+h*0.25, w, h*0.4))

def _get_line_width(g):
    lw = 5.0 * np.exp(-0.05 * g.number_of_edges())
    return lw


def _get_edge_width(g, info_dict):
    ew = info_dict.get('edge_weight_default', 1.0)
    ew = ew * np.exp(-0.0015 * g.number_of_edges())
    return info_dict.get('edge_weights', [ew] * len(g.edges()))

def set_ax(m, n, fig):
    grid = GridSpec(m, n, figure=fig)
    ax_q = fig.add_subplot(grid[:, 0])
    ax_q.axis("off")

    ax_p = [['' for _ in range(n-1)] for _ in range(m)]
    for i in range(m):
        for j in range(n-1):
            ax_p[i][j] = fig.add_subplot(grid[i, j+1])
            ax_p[i][j].axis("off")

    return ax_q, ax_p, grid

def draw_graph_small(g, info_dict, ax, ifmap=None, maplist=None):
    if g is None:
        return
    if g.number_of_nodes() > 1000:
        print('Graph to plot too large with {} nodes! skip...'.format(
            g.number_of_nodes()))
        return
    pos = _sorted_dict(graphviz_layout(g))
    if not ifmap:
        color_values = _get_node_colors(g, info_dict)
    else:
        color_values = _get_node_map_colors(g, info_dict,maplist)
    node_labels = _sorted_dict(nx.get_node_attributes(g, info_dict['node_label_type']))

    if not info_dict['show_labels'] or ifmap:
        for key, value in node_labels.items():
            node_labels[key] = ''
    # print(pos)
    nx.draw_networkx(g, pos, ax= ax, nodelist=pos.keys(),
                     node_color=color_values, with_labels=True,
                     node_size=_get_node_size(g, info_dict),
                     labels=node_labels,
                     font_size=info_dict['draw_node_label_font_size'],
                     linewidths=_get_line_width(g), width=_get_edge_width(g, info_dict))
    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])



def _get_node_size(g, info_dict):
    ns = info_dict['draw_node_size']
    return ns * np.exp(-0.02 * g.number_of_nodes())

def _sorted_dict(d):
    rtn = OrderedDict()
    for k in sorted(d.keys()):
        rtn[k] = d[k]
    return rtn

def _get_node_colors(g, info_dict):
    if info_dict['node_label_name'] is not None:
        color_values = []
        node_color_labels = _sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        for node_label in node_color_labels.values():
            color = info_dict['draw_node_color_map'].get(node_label, None)
            color_values.append(color)
    else:
        # color_values = ['lightskyblue'] * g.number_of_nodes()
        color_values = info_dict['draw_node_color_map'] * g.number_of_nodes()
    # print(color_values)
    return color_values

def add_colorbar_to_fig(fig, gs, cmap, label_default, tick_label_size):
    if cmap is not None:
        # 创建颜色条轴（横跨两行）
        cbar_ax = fig.add_subplot(gs[0:2, -1])
        
        # 创建ScalarMappable对象关联颜色映射
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])  # 只需要颜色映射，不需要实际数据
        
        # 添加颜色条
        cbar = fig.colorbar(
            sm,
            cax=cbar_ax,
            label=label_default
        )

        cbar.ax.tick_params(labelsize=tick_label_size)
    return fig

def _get_node_map_colors(g, info_dict, map):
    # if info_dict['get_map_mothed'] == 'cosine':
    #     norm = False
    # elif info_dict['get_map_mothed'] == 'dot' or 'l2':
    #     norm = False
    # if not norm:
    #     max = np.max(map)
    #     min = np.min(map)
    #     map = [(i - min) / (max - min + 1e-10) for i in map]
    
    node_colors = [info_dict['draw_node_color_mapdcit'](val) for val in map]
    return node_colors


def _info_dict_preprocess(info_dict):
    info_dict.setdefault('draw_node_size', 10)
    info_dict.setdefault('draw_node_label_enable', True)
    info_dict.setdefault('node_label_name', '')
    info_dict.setdefault('draw_node_label_font_size', 6)

    info_dict.setdefault('draw_edge_label_enable', False)
    info_dict.setdefault('edge_label_name', '')
    info_dict.setdefault('draw_edge_label_font_size', 6)

    info_dict.setdefault('each_graph_text_font_size', "")
    info_dict.setdefault('each_graph_text_pos', [0.5, 0.8])

    info_dict.setdefault('plot_dpi', 200)
    info_dict.setdefault('plot_save_path', "")

    info_dict.setdefault('top_space', 0.08)
    info_dict.setdefault('bottom_space', 0)
    info_dict.setdefault('hbetween_space', 0.5)
    info_dict.setdefault('wbetween_space', 0.01)

def _calc_subplot_size_small(area):
    w = int(area)
    return [2, math.ceil(w / 2)]

def draw_extra(i, ax, info_dict, text, text_or_title):
    if text_or_title == 'text':
        x = _list_safe_get(info_dict['each_graph_text_pos'], 0, 0.5)
        y = _list_safe_get(info_dict['each_graph_text_pos'], 1, 0.8)
    elif text_or_title == 'title':
        x = _list_safe_get(info_dict['each_graph_title_pos'], 0, 0.5)
        y = _list_safe_get(info_dict['each_graph_title_pos'], 1, 0.8)
    # print(left, bottom)
    ax.text(x, y, text, fontsize=info_dict['each_graph_text_font_size'], ha='center', transform=ax.transAxes)
    plt.axis('off')

def draw_bottom_title(fig, gs, info_dict):
    x = _list_safe_get(info_dict['each_graph_text_from_pos'], 0, 0.5)
    y = _list_safe_get(info_dict['each_graph_text_from_pos'], 1, 0.8)    
    ax_gt = fig.add_subplot(gs[0, 1:-1])
    ax_gt.axis("off")  # 隐藏轴的边框和刻度
    ax_gt.text(x, y, 
            info_dict['each_graph_text_from_gt'], 
            fontsize=info_dict['each_graph_text_font_size'], 
            ha='center', transform=ax_gt.transAxes)
    ax_pred = fig.add_subplot(gs[1, 1:-1])
    ax_pred.axis("off")  # 隐藏轴的边框和刻度
    ax_pred.text(x, y, 
            info_dict['each_graph_text_from_pred'], 
            fontsize=info_dict['each_graph_text_font_size'], 
            ha='center', transform=ax_pred.transAxes)
def _list_safe_get(l, index, default):
    try:
        return l[index]
    except IndexError:
        return default


def _save_figs(info_dict):
    save_path = info_dict['plot_save_path_pdf']
    print(save_path)
    if not save_path:
        # print('plt.show')
        plt.show()
    else:
        for full_path in [info_dict['plot_save_path_pdf']]:
            if not full_path:
                continue
            # print('Saving query vis plot to {}'.format(sp))
            if not os.path.exists(dirname(full_path)):
                os.makedirs(dirname(full_path))
            plt.savefig(full_path, dpi=info_dict['plot_dpi'])
    plt.close()

