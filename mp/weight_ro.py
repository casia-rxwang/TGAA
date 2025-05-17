import torch
from torch import Tensor
from torch.nn import Linear, Sequential, Parameter
from typing import List
from torch_scatter import scatter


class Origin(torch.nn.Module):
    def __init__(self, ro_agg='sum'):
        super(Origin, self).__init__()
        self.reduce = ro_agg if ro_agg in ['sum', 'mean', 'max'] else 'sum'

    def forward(self, x, ref, index, batch_size, dim=0):
        return scatter(x, index, dim, dim_size=batch_size, reduce=self.reduce)


class Gate(torch.nn.Module):
    def __init__(self, in_size):
        super(Gate, self).__init__()
        self.lin = Linear(in_size, 1, bias=True)

    def forward(self, x, ref, index, batch_size, dim=0):
        w = torch.tanh(self.lin(torch.cat([x, ref], dim=-1)))

        return scatter(torch.mul(w, x), index, dim, dim_size=batch_size, reduce='sum')


class Attention(torch.nn.Module):
    def __init__(self, in_size):
        super(Attention, self).__init__()
        self.lin = Linear(in_size, 1, bias=True)

    def forward(self, x, ref, index, batch_size, dim=0):
        # torch_geometric.utils.softmax
        # index: target batch idx
        src = torch.tanh(self.lin(torch.cat([x, ref], dim=-1)))

        src_max = scatter(src.detach(), index, dim, dim_size=batch_size, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=batch_size, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)

        w = out / out_sum
        return scatter(torch.mul(w, x), index, dim, dim_size=batch_size, reduce='sum')


class Label_Histogram(torch.nn.Module):
    def __init__(self, in_size: int, num_cluster: int, heads: int = 1, gamma: float = 1., tau: float = 10.):
        super(Label_Histogram, self).__init__()
        self.in_size = in_size
        self.heads = heads
        self.num_cluster = num_cluster
        self.gamma = gamma
        self.tau = tau

        self.k = Parameter(torch.empty(heads, num_cluster, in_size))
        # self.lin = Linear(heads, 1, bias=True)
        self.trans = Sequential(Linear(num_cluster * in_size, in_size), torch.nn.LeakyReLU())

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.k.data)
        # self.lin.reset_parameters()

    def forward(self, x, ref, index, batch_size, dim=0):
        N, H, K, D = x.size(0), self.heads, self.num_cluster, self.in_size

        dist = torch.cdist(self.k.view(H * K, D), x, p=2)**2
        dist = (1. + dist / self.gamma).pow(-(self.gamma + 1.0) / 2.0)

        dist = dist.view(H, K, N).permute(2, 1, 0)  # [N, K, H]

        # w = dist / dist.sum(dim=-2, keepdim=True)
        # w = self.lin(w).squeeze(dim=-1).softmax(dim=-1)
        w = dist / dist.sum(dim=-2, keepdim=True) * self.tau
        w = w.mean(dim=-1, keepdim=False).softmax(dim=-1)  # [N, K]

        new_x = torch.mul(w.unsqueeze(dim=-1), x.unsqueeze(dim=-2)).view(N, K * D)  # [N, KD]
        agg_x = scatter(new_x, index, dim, dim_size=batch_size, reduce='sum')  # [B, KD]
        return self.trans(agg_x)  # [B, D]


class MLP(torch.nn.Module):
    def __init__(self, in_size: int, num_cluster: int):
        super(MLP, self).__init__()
        self.lin = Linear(in_size, num_cluster * in_size)
        self.trans = Sequential(Linear(num_cluster * in_size, in_size), torch.nn.LeakyReLU())

    def forward(self, x, ref, index, batch_size, dim=0):
        new_x = self.lin(x)
        agg_x = scatter(new_x, index, dim, dim_size=batch_size, reduce='sum')
        return self.trans(agg_x)


def create_ro_weight(embed_dim, hidden, args, dim):
    ro_agg = args.ro_agg
    K = args.ro_clusters[dim]

    if ro_agg == 'gate':
        return Gate(embed_dim + hidden)
    elif ro_agg == 'att':
        return Attention(embed_dim + hidden)
    elif ro_agg == 'lha' and K > 1:
        return Label_Histogram(hidden, K, gamma=args.gamma, tau=args.tau)
    elif ro_agg == 'mlp' and K > 1:
        return MLP(hidden, K)
    else:
        return Origin(ro_agg=ro_agg)


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
