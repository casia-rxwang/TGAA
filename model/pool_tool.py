import torch
from torch import Tensor
from torch.nn import Linear
from typing import List


class Origin(torch.nn.Module):
    def __init__(self):
        super(Origin, self).__init__()

    def forward(self, x, ref):
        return x.new_ones(x.size(0), 1)


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
    def __init__(self, in_size, hidden):
        super(Multiply, self).__init__()
        self.lin = Linear(in_size, hidden, bias=False)

    def forward(self, x, ref):
        w = torch.mul(self.lin(ref), x).sum(dim=1, keepdim=True)
        w = torch.tanh(w)
        return w


class Multiply_C(torch.nn.Module):
    def __init__(self, in_size, hidden):
        super(Multiply_C, self).__init__()
        self.lin = Linear(in_size, hidden, bias=False)

    def forward(self, x, ref):
        w = torch.mul(self.lin(ref), x)
        w = torch.tanh(w)
        return w


def create_pool_weight(embed_dim, hidden, args):
    pool_cal = args.pool_cal
    pool_channel = args.pool_channel
    in_size = hidden * args.num_layers if args.jump_mode == 'cat' else hidden
    out_size = hidden if args.pool_channel else 1

    if pool_cal == 'mlp':
        return Gate(embed_dim + in_size, out_size)
    elif pool_cal == 'diff':
        return Gate_Diff(hidden, out_size)
    elif pool_cal == 'mul':
        if pool_channel:
            return Multiply_C(in_size, hidden)
        else:
            return Multiply(in_size, hidden)
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
