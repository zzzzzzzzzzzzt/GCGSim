import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_undirected, to_networkx
import random
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)

def gen_pair(g, kl=None, ku=2):
    if kl is None:
        kl = ku

    directed_edge_index = to_directed(g.edge_index)

    n = g.num_nodes
    num_edges = directed_edge_index.size()[1]
    to_remove = random.randint(kl, ku) 

    edge_index_n = directed_edge_index[:, torch.randperm(num_edges)[to_remove:]]   
    if edge_index_n.size(1) != 0:  
        edge_index_n = to_undirected(edge_index_n)

    row, col = g.edge_index
    adj = torch.ones((n, n), dtype=torch.uint8)  
    adj[row, col] = 0  
    non_edge_index = adj.nonzero().t()  # 

    directed_non_edge_index = to_directed(non_edge_index)
    num_edges = directed_non_edge_index.size()[1]

    to_add = random.randint(kl, ku)

    edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]  
    if edge_index_p.size(1):
        edge_index_p = to_undirected(edge_index_p)
    edge_index_p = torch.cat((edge_index_n, edge_index_p), 1)

    if hasattr(g, "i"):
        g2 = Data(x=g.x, edge_index=edge_index_p, i=g.i)
    else:
        g2 = Data(x=g.x, edge_index=edge_index_p)

    g2.num_nodes = g.num_nodes
    return g2, to_remove + to_add


def gen_pairs(graphs, kl=None, ku=2):

    gen_graphs_1 = []  
    gen_graphs_2 = []  

    count = len(graphs)
    mat = torch.full((count, count), float("inf")) 
    norm_mat = torch.full((count, count), float("inf"))

    for i, g in enumerate(graphs):
        g = g.clone()
        g.i = torch.tensor([i])  # 图的id
        g2, ged = gen_pair(g, kl, ku)
        gen_graphs_1.append(g)
        gen_graphs_2.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g.num_nodes + g2.num_nodes))

    return gen_graphs_1, gen_graphs_2, mat, norm_mat

def random_walk_positional_encoding(g, walk_length):
    num_nodes = g.num_nodes
    edge_index = g.edge_index

    adj = SparseTensor.from_edge_index(edge_index, None, sparse_sizes=(num_nodes, num_nodes))

    # Compute D^{-1} A:
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)

    out = adj
    row, col, value = out.coo()
    pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
    for _ in range(walk_length - 1):
        out = out @ adj
        row, col, value = out.coo()
        pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
    g['pos_enc'] = torch.stack(pe_list, dim=-1)
    return g
    
def get_self_loop_attr(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones(loop_index.numel(), device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr