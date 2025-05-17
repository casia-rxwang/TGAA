import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Parameter
from typing import Callable
from torch_scatter import scatter


class Origin(torch.nn.Module):
    def __init__(self, mp_agg='sum'):
        super(Origin, self).__init__()
        self.reduce = mp_agg if mp_agg in ['sum', 'mean', 'max'] else 'sum'

    def forward(self, msg, x_i, x_j, e_ij, index, num_nodes, dim=0):
        return scatter(msg, index, dim, dim_size=num_nodes, reduce=self.reduce)


class Gate(torch.nn.Module):
    def __init__(self, in_size, agg_q):
        super(Gate, self).__init__()
        self.lin = Linear(in_size, 1, bias=True)
        self.agg_q = agg_q

    def forward(self, msg, x_i, x_j, e_ij, index, num_nodes, dim=0):
        q = self.agg_q((x_j, e_ij))
        w = torch.tanh(self.lin(torch.cat([q, x_i], dim=-1)))

        return scatter(torch.mul(w, msg), index, dim, dim_size=num_nodes, reduce='sum')


class GateV2(torch.nn.Module):
    def __init__(self, in_size, hidden, agg_q):
        super(GateV2, self).__init__()
        self.lin1 = Linear(in_size, hidden, bias=True)
        self.lin2 = Linear(hidden, 1, bias=True)
        self.agg_q = agg_q

    def forward(self, msg, x_i, x_j, e_ij, index, num_nodes, dim=0):
        q = self.agg_q((x_j, e_ij))
        w = torch.tanh(self.lin2(F.leaky_relu(self.lin1(torch.cat([q, x_i], dim=-1)))))

        return scatter(torch.mul(w, msg), index, dim, dim_size=num_nodes, reduce='sum')


class Attention(torch.nn.Module):
    def __init__(self, in_size, agg_q):
        super(Attention, self).__init__()
        self.lin = Linear(in_size, 1, bias=True)
        self.agg_q = agg_q

    def forward(self, msg, x_i, x_j, e_ij, index, num_nodes, dim=0):
        # torch_geometric.utils.softmax
        # index: target node idx
        q = self.agg_q((x_j, e_ij))
        src = torch.tanh(self.lin(torch.cat([q, x_i], dim=-1)))

        src_max = scatter(src.detach(), index, dim, dim_size=num_nodes, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=num_nodes, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)

        w = out / out_sum
        return scatter(torch.mul(w, msg), index, dim, dim_size=num_nodes, reduce='sum')


class Label_Histogram(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, num_cluster: int, agg_q: Callable, heads: int = 1, gamma: float = 1., tau: float = 10.):
        super(Label_Histogram, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_cluster = num_cluster
        self.heads = heads
        self.gamma = gamma
        self.tau = tau

        self.k = Parameter(torch.empty(heads, num_cluster, in_size))
        # self.lin = Linear(heads, 1, bias=True)
        self.trans = Sequential(Linear(num_cluster * out_size, out_size), torch.nn.LeakyReLU())
        self.agg_q = agg_q

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.k.data)
        # self.lin.reset_parameters()

    def forward(self, msg, x_i, x_j, e_ij, index, num_nodes, dim=0):
        x = self.agg_q((x_j, e_ij))
        D_ = self.out_size
        N, H, K, D = x.size(0), self.heads, self.num_cluster, self.in_size

        dist = torch.cdist(self.k.view(H * K, D), x, p=2)**2
        dist = (1. + dist / self.gamma).pow(-(self.gamma + 1.0) / 2.0)

        dist = dist.view(H, K, N).permute(2, 1, 0)  # [N, K, H]

        # w = dist / dist.sum(dim=-2, keepdim=True)
        # w = self.lin(w).squeeze(dim=-1).softmax(dim=-1)
        w = dist / dist.sum(dim=-2, keepdim=True) * self.tau
        w = w.mean(dim=-1, keepdim=False).softmax(dim=-1)  # [N, K]

        new_msg = torch.mul(w.unsqueeze(dim=-1), msg.unsqueeze(dim=-2)).view(N, K * D_)  # [N, KD]
        agg_msg = scatter(new_msg, index, dim, dim_size=num_nodes, reduce='sum')  # [B, KD]
        return self.trans(agg_msg)  # [B, D]


class MLP(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, num_cluster: int, agg_q: Callable):
        super(MLP, self).__init__()
        self.lin = Linear(out_size, num_cluster * out_size)
        self.trans = Sequential(Linear(num_cluster * out_size, out_size), torch.nn.LeakyReLU())
        self.agg_q = agg_q

    def forward(self, msg, x_i, x_j, e_ij, index, num_nodes, dim=0):
        new_msg = self.lin(msg)
        agg_msg = scatter(new_msg, index, dim, dim_size=num_nodes, reduce='sum')
        return self.trans(agg_msg)


def create_mp_weight(hidden, args, depth):
    mp_agg = args.mp_agg
    K = args.mp_clusters

    # 0: q=x_j; 1: q=e_ij; 2: q=x_j||e_ij
    if args.agg_qk in [0, 1]:
        in_size = hidden
        def agg_q(xs): return xs[args.agg_qk]
    else:
        in_size = hidden * 2
        def agg_q(xs): return torch.cat(xs[:2], dim=-1)

    if depth < args.mp_agg_depth:
        return Origin(mp_agg=mp_agg)  # extra switch

    if mp_agg == 'gate':
        return Gate(hidden + in_size, agg_q)
    elif mp_agg == 'gatev2':
        return GateV2(hidden + in_size, args.gatev2_hidden, agg_q)
    elif mp_agg == 'att':
        return Attention(hidden + in_size, agg_q)
    elif mp_agg == 'lha' and K > 1 and depth == args.num_layers:
        return Label_Histogram(in_size, hidden, K, agg_q, gamma=args.gamma, tau=args.tau)
    elif mp_agg == 'mlp' and K > 1 and depth == args.num_layers:
        return MLP(in_size, hidden, K, agg_q)
    else:
        return Origin(mp_agg=mp_agg)
