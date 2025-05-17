import torch
from torch import Tensor, LongTensor
from typing import Tuple


def to_dense_x(batch: Tensor, cum_nodes: Tensor, batch_size: int):
    '''
    return [batch_size, max_num_nodes], Tuple(idx0, idx1)
    '''
    num_nodes = cum_nodes[1:] - cum_nodes[:-1]
    max_num_nodes = num_nodes.max()

    idx = torch.arange(batch.size(0), dtype=torch.long, device=batch.device)
    idx0 = batch
    idx1 = idx - cum_nodes[batch]

    mask = torch.zeros((batch_size, max_num_nodes), device=batch.device)
    mask[idx0, idx1] = 1

    return mask, (idx0, idx1)


def to_dense_adj(edge_index: Tensor, batch: Tensor, cum_nodes: Tensor, batch_size: int):
    '''
    return [batch_size, max_num_nodes, max_num_nodes], Tuple(idx0, idx1, idx2)
    '''
    i, j = (1, 0)  # if self.flow == 'source_to_target' else (0, 1)

    num_nodes = cum_nodes[1:] - cum_nodes[:-1]
    max_num_nodes = num_nodes.max()

    idx0 = batch[edge_index[i]]
    idx1 = edge_index[i] - cum_nodes[batch][edge_index[i]]
    idx2 = edge_index[j] - cum_nodes[batch][edge_index[j]]

    adj = torch.zeros([batch_size, max_num_nodes, max_num_nodes], device=batch.device)
    adj[idx0, idx1, idx2] = 1

    return adj, (idx0, idx1, idx2)


def get_dense_x(x: Tensor, d_x_mask: Tensor, d_x_idx: Tuple[LongTensor, LongTensor]):
    dense_x = x.new_zeros(list(d_x_mask.size()) + list(x.size())[1:])
    dense_x[d_x_idx] = x
    return dense_x


def get_dense_e(e: Tensor, adj: Tensor, e_idx: Tuple[LongTensor, LongTensor, LongTensor]):
    dense_up_attr = e.new_zeros(list(adj.size()) + list(e.size())[1:])
    dense_up_attr[e_idx] = e
    return dense_up_attr


def to_dense_x_0(batch, num_nodes, batch_size):
    '''
    return [batch_size, max_num_nodes, 1], Tuple(idx0, idx1)
    '''
    num_nodes = torch.tensor(num_nodes, device=batch.device)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=batch.device)
    idx0 = batch
    idx1 = idx - cum_nodes[batch]

    mask = torch.zeros((batch_size, max_num_nodes), device=batch.device)
    mask[idx0, idx1] = 1

    return mask, (idx0, idx1)


def to_dense_adj_0(edge_index, batch_i, batch_j, num_nodes_i, num_nodes_j, batch_size):
    '''
    return [batch_size, source_dim, target_dim], Tuple(idx0, idx1, idx2)
    '''
    i, j = (1, 0)  # if self.flow == 'source_to_target' else (0, 1)

    num_nodes_i = torch.tensor(num_nodes_i, device=batch_i.device)
    num_nodes_j = torch.tensor(num_nodes_j, device=batch_j.device)
    cum_nodes_i = torch.cat([edge_index.new_zeros(1), num_nodes_i.cumsum(dim=0)])
    cum_nodes_j = torch.cat([edge_index.new_zeros(1), num_nodes_j.cumsum(dim=0)])
    max_num_nodes_i = int(num_nodes_i.max())
    max_num_nodes_j = int(num_nodes_j.max())

    idx0 = batch_i[edge_index[i]]
    # idx0 = batch_j[edge_index[j]]
    idx1 = edge_index[i] - cum_nodes_i[batch_i][edge_index[i]]
    idx2 = edge_index[j] - cum_nodes_j[batch_j][edge_index[j]]

    adj = torch.zeros([batch_size, max_num_nodes_i, max_num_nodes_j], device=edge_index.device)
    adj[idx0, idx1, idx2] = 1

    return adj, (idx0, idx1, idx2)
