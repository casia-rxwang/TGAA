import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter
from typing import List
from mp.weight_mp import dot_product


class Origin(torch.nn.Module):
    def __init__(self):
        super(Origin, self).__init__()

    def forward(self, x, ref):
        return x.new_ones(x.size()[:-1] + (1, ))


class Cosine_Similarity(torch.nn.Module):
    def __init__(self, in_size):
        super(Cosine_Similarity, self).__init__()
        self.lin = dot_product(in_size)

    def forward(self, x, ref):
        w = F.cosine_similarity(self.lin(x), ref).view(-1, 1)
        return w


class Gate(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Gate, self).__init__()
        self.lin = Linear(in_size, out_size, bias=True)

    def forward(self, x, ref):
        return torch.tanh(self.lin(torch.cat([x, ref], dim=-1)))


class Gate_Diff(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Gate_Diff, self).__init__()
        self.lin = Linear(in_size, out_size, bias=True)

    def forward(self, x, ref):
        return torch.tanh(self.lin(x - ref))


class Multiply(torch.nn.Module):
    def __init__(self, in_size, embed_dim):
        super(Multiply, self).__init__()
        self.lin = Linear(in_size, embed_dim, bias=False)

    def forward(self, x, ref):
        w = torch.mul(self.lin(x), ref).sum(dim=1, keepdim=True)
        return torch.tanh(w)


class Multiply_C(torch.nn.Module):
    def __init__(self, in_size, embed_dim):
        super(Multiply_C, self).__init__()
        self.lin = Linear(in_size, embed_dim, bias=False)

    def forward(self, x, ref):
        w = torch.mul(self.lin(x), ref)
        return torch.tanh(w)


class Attention(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Attention, self).__init__()
        self.lin = Linear(in_size, out_size, bias=True)

    def forward(self, x, ref, mask, x_idx):
        w = torch.tanh(self.lin(torch.cat([x, ref], dim=-1)))
        dense_att = w.new_full(mask.size() + w.size()[-1:], fill_value=-9e15)  # b,n,f
        dense_att[x_idx] = w
        dense_att = F.softmax(dense_att, dim=-2)
        w = dense_att[x_idx]

        return w


def create_pool_weight(embed_dim, hidden, args):
    in_size = hidden * args.num_layers if args.jump_mode == 'cat' else hidden
    out_size = hidden if args.pool_channel else 1

    pool_cal = args.pool_cal
    if pool_cal == 'cosine':
        return Cosine_Similarity(in_size)
    elif pool_cal == 'mlp':
        return Gate(embed_dim + in_size, out_size)
    elif pool_cal == 'diff':
        return Gate_Diff(in_size, out_size)
    elif pool_cal == 'mul':
        if args.pool_channel:
            return Multiply_C(in_size, embed_dim)
        else:
            return Multiply(in_size, embed_dim)
    else:
        return Origin()


def sum_final_readout(xs: List[Tensor]):
    x = torch.stack(xs, dim=0)
    return x.sum(0)


def mean_final_readout(xs: List[Tensor]):
    x = torch.stack(xs, dim=0)
    return x.mean(0)


def cat_final_readout(xs: List[Tensor]):
    return torch.cat(xs, dim=-1)


def crate_final_readout(method):
    if method == 'sum':
        return sum_final_readout
    elif method == 'mean':
        return mean_final_readout
    else:
        return cat_final_readout
