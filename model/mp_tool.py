import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


class Origin(torch.nn.Module):
    def __init__(self):
        super(Origin, self).__init__()

    def forward(self, x_i, x_j, e_ij, v):
        return x_i.new_ones(x_i.size(0), 1)


class Gate(torch.nn.Module):
    def __init__(self, in_size, k_s, agg_k, agg_q, out_size):
        super(Gate, self).__init__()
        self.lin = Linear(in_size * k_s, out_size, bias=True)
        self.agg_k = agg_k
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij, v):
        k = self.agg_k((x_j, e_ij, v))
        q = self.agg_q((x_i, x_i))
        w = torch.tanh(self.lin(F.leaky_relu(k + q)))
        return w


class Gate_Diff(torch.nn.Module):
    def __init__(self, in_size, k_s, agg_k, agg_q, out_size):
        super(Gate_Diff, self).__init__()
        self.lin = Linear(in_size * k_s, out_size, bias=True)
        self.agg_k = agg_k
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij, v):
        k = self.agg_k((x_j, e_ij, v))
        q = self.agg_q((x_i, x_i))
        w = torch.tanh(self.lin(F.leaky_relu(k - q)))
        return w


class GateV2(torch.nn.Module):
    def __init__(self, in_size, k_s, agg_k, out_size, hidden):
        super(GateV2, self).__init__()
        self.lin1 = Linear(in_size * (k_s + 1), hidden, bias=True)
        self.lin2 = Linear(hidden, out_size, bias=True)
        self.agg_k = agg_k
        # self.agg_q = lambda xs: xs[0]

    def forward(self, x_i, x_j, e_ij, v):
        k = self.agg_k((x_j, e_ij, v))
        # q = self.agg_q((x_i, x_i))
        w = F.leaky_relu(self.lin1(torch.cat([k, x_i], dim=-1)))
        w = torch.tanh(self.lin2(w))
        return w


class Multiply(torch.nn.Module):
    def __init__(self, in_size, k_s, agg_k):
        super(Multiply, self).__init__()
        self.lin = Linear(in_size * k_s, in_size, bias=False)
        self.agg_k = agg_k
        # self.agg_q = lambda xs: xs[0]

    def forward(self, x_i, x_j, e_ij, v):
        k = self.agg_k((x_j, e_ij, v))
        # q = self.agg_q((x_i, x_i))
        w = torch.mul(self.lin(k), x_i).sum(dim=1, keepdim=True)
        w = torch.tanh(w)
        return w


class Multiply_C(torch.nn.Module):
    def __init__(self, in_size, k_s, agg_k):
        super(Multiply_C, self).__init__()
        self.lin = Linear(in_size * k_s, in_size, bias=False)
        self.agg_k = agg_k
        # self.agg_q = lambda xs: xs[0]

    def forward(self, x_i, x_j, e_ij, v):
        k = self.agg_k((x_j, e_ij, v))
        # q = self.agg_q((x_i, x_i))
        w = torch.mul(self.lin(k), x_i)
        w = torch.tanh(w)
        return w


def create_fb_weight(in_size, args):
    agg_kq = args.agg_kq if args.use_coboundaries.lower() == 'true' else min(1, args.agg_kq)
    if agg_kq == 2:
        agg_k = lambda xs: torch.cat(xs[:2], dim=-1)
        agg_q = lambda xs: torch.cat(xs[:2], dim=-1)
    elif agg_kq == 1:
        agg_k = lambda xs: xs[0]
        agg_q = lambda xs: xs[0]
    else:
        agg_k = lambda xs: xs[2]
        agg_q = lambda xs: xs[0]

    k_s = max(1, agg_kq)
    mp_cal = args.mp_cal
    mp_channel = args.mp_channel
    out_size = in_size if args.mp_channel else 1

    if mp_cal == 'mlp':
        return Gate(in_size, k_s, agg_k, agg_q, out_size)
    elif mp_cal == 'mlpv2':
        return GateV2(in_size, k_s, agg_k, out_size, args.mlpv2_hidden)
    elif mp_cal == 'diff':
        return Gate_Diff(in_size, k_s, agg_k, agg_q, out_size)
    elif mp_cal == 'mul':
        if mp_channel:
            return Multiply_C(in_size, k_s, agg_k)
        else:
            return Multiply(in_size, k_s, agg_k)
    else:
        return Origin()


def mean_d_i(w: Tensor):
    '''GraphSage'''
    return w / (w.sum(dim=-1, keepdim=True) + 1e-6)


def sqrt_d_ij(w: Tensor):
    '''GCN'''
    d_i = w.sum(dim=-1, keepdim=True)
    d_j = w.sum(dim=-2, keepdim=True)
    return w / (torch.sqrt(torch.mul(d_i, d_j)) + 1e-6)


def create_sb_weight(method):
    if method == 'mean':
        return mean_d_i
    elif method == 'degree':
        return sqrt_d_ij
    else:
        return lambda x: x
