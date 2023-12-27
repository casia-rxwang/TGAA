import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.inits import glorot


class Origin(torch.nn.Module):
    def __init__(self):
        super(Origin, self).__init__()

    def forward(self, x_i, x_j, e_ij):
        return x_i.new_ones(x_i.size()[:-1] + (1, ))


class Cosine_Similarity(torch.nn.Module):
    def __init__(self, in_size, dim_coff, agg_q, agg_k):
        super(Cosine_Similarity, self).__init__()
        self.lin = dot_product(in_size * dim_coff)
        self.agg_q = agg_q
        self.agg_k = agg_k

    def forward(self, x_i, x_j, e_ij):
        q = self.agg_q((x_j, e_ij))
        k = self.agg_k((x_i, x_i))
        w = F.cosine_similarity(self.lin(q), k).view(-1, 1)
        return w


class Gate(torch.nn.Module):
    def __init__(self, in_size, dim_coff, agg_q, out_size):
        super(Gate, self).__init__()
        self.lin = Linear(in_size * dim_coff, out_size, bias=True)
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij):
        q = self.agg_q((x_j, e_ij))
        w = torch.tanh(self.lin(torch.cat([q, x_i], dim=-1)))
        return w


class GateV2(torch.nn.Module):
    def __init__(self, in_size, dim_coff, agg_q, out_size, hidden):
        super(GateV2, self).__init__()
        self.lin1 = Linear(in_size * dim_coff, hidden, bias=True)
        self.lin2 = Linear(hidden, out_size, bias=True)
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij):
        q = self.agg_q((x_j, e_ij))
        w = torch.tanh(self.lin2(F.leaky_relu(self.lin1(torch.cat([q, x_i], dim=-1)))))
        return w


class Multiply(torch.nn.Module):
    def __init__(self, in_size, dim_coff, agg_q):
        super(Multiply, self).__init__()
        self.lin = Linear(in_size * dim_coff, in_size, bias=False)
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij):
        q = self.agg_q((x_j, e_ij))
        w = torch.mul(self.lin(q), x_i).sum(dim=1, keepdim=True)
        w = torch.tanh(w)
        return w


class Multiply_C(torch.nn.Module):
    def __init__(self, in_size, dim_coff, agg_q):
        super(Multiply_C, self).__init__()
        self.lin = Linear(in_size * dim_coff, in_size, bias=False)
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij):
        q = self.agg_q((x_j, e_ij))
        w = torch.mul(self.lin(q), x_i)
        w = torch.tanh(w)
        return w


class Attention(torch.nn.Module):
    '''GAT'''
    def __init__(self, in_size, dim_coff, agg_q, out_size):
        super(Attention, self).__init__()
        self.lin = Linear(in_size * dim_coff, out_size, bias=True)
        self.agg_q = agg_q

    def forward(self, x_i, x_j, e_ij, adj, e_idx):
        '''
        adj: Tensor [N, N]
        e_idx: Tuple (Tensor, Tensor)
        '''
        q = self.agg_q((x_j, e_ij))

        w = torch.tanh(self.lin(torch.cat([q, x_i], dim=-1)))
        dense_att = w.new_full(adj.size() + w.size()[-1:], fill_value=-9e15)  # b,n_target,n_source,f
        dense_att[e_idx] = w
        dense_att = F.softmax(dense_att, dim=-2)
        w = dense_att[e_idx]

        return w


def create_fb_weight(in_size, args, agg_qk=-1):
    # 0: q=x_j, k=x; 1: q=e_ij, k=x; 2: q=x_j||e_ij, k=x||x
    if agg_qk == -1:
        agg_qk = args.agg_qk
    if agg_qk == 2:
        agg_k = lambda xs: torch.cat(xs[:2], dim=-1)
        agg_q = lambda xs: torch.cat(xs[:2], dim=-1)
    elif agg_qk == 1:
        agg_k = lambda xs: xs[0]
        agg_q = lambda xs: xs[1]
    else:
        agg_k = lambda xs: xs[0]
        agg_q = lambda xs: xs[0]

    dim_coff = max(1, agg_qk)
    out_size = in_size if args.mp_channel else 1

    mp_cal = args.mp_cal
    if mp_cal == 'cosine':
        return Cosine_Similarity(in_size, dim_coff, agg_k, agg_q)
    elif mp_cal == 'mlp':
        return Gate(in_size, dim_coff + 1, agg_q, out_size)
    elif mp_cal == 'mlpv2':
        return GateV2(in_size, dim_coff + 1, agg_q, out_size, args.mlpv2_hidden)
    elif mp_cal == 'mul':
        if args.mp_channel:
            return Multiply_C(in_size, dim_coff, agg_q)
        else:
            return Multiply(in_size, dim_coff, agg_q)
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


def create_sb_weight(sb_cal):
    if sb_cal == 'mean':
        return mean_d_i
    elif sb_cal == 'degree':
        return sqrt_d_ij
    else:
        return lambda x: x


class chebyshev_ploy(torch.nn.Module):
    def __init__(self, K, channels=1):
        super(chebyshev_ploy, self).__init__()

        assert K > 1, "\n!! K < 2 in chebyshev_ploy."
        self.lins = torch.nn.ModuleList([dot_product(channels) for _ in range(K)])
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor):
        '''
        x: (*, in_channels)
        out: (*, out_channels)
        '''
        Tx_0 = x * 0 + 1.0
        Tx_1 = x
        out = self.lins[0](Tx_0) + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = 2. * torch.mul(Tx_1, x) - Tx_0
            out = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out


class dot_product(torch.nn.Module):
    def __init__(self, channels):
        super(dot_product, self).__init__()
        self.coffs = torch.nn.Parameter(torch.ones(1, channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.coffs)

    def forward(self, x: Tensor):
        '''
        x: (*, channels)
        out: (*, channels)
        '''
        return torch.mul(self.coffs, x)
