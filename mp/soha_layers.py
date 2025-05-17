import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Sequential, BatchNorm1d, Conv2d, KLDivLoss, Parameter
from torch_scatter import scatter
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from typing import Callable, List

from mp.graph_utils import to_dense_x, to_dense_adj

EPS = 1e-15


class MPParam():
    def __init__(self, data: Data):
        mask, x_idx = to_dense_x(batch=data.batch, cum_nodes=data.ptr, batch_size=data._num_graphs)
        adj, e_idx = to_dense_adj(edge_index=data.edge_index, batch=data.batch, cum_nodes=data.ptr, batch_size=data._num_graphs)

        # fixed
        self.batch_size = data._num_graphs
        self.batch = data.batch
        self.edge_index = data.edge_index
        self.mask = mask  # B,N
        self.x_idx = x_idx
        self.adj = adj  # B,N,N
        self.e_idx = e_idx

        self.x0 = data.x
        if data.edge_attr is None:
            self.e0 = (data.x[data.edge_index[0]] + data.x[data.edge_index[1]]) / 2.0
        else:
            self.e0 = data.edge_attr

        # will be init after
        self.x = None  # sparse
        self.e = None  # sparse
        self.attr_list = list()  # B,N,F
        self.adj_list = list()  # B,C,C
        self.nc_list = list()  # B,N,C; membership of node to cluster
        self.cn_list = list()  # B,C,N; membership of cluster to node

    def init_attr(self, x: Tensor, e: Tensor, attr_list: List[Tensor], adj_list: List[Tensor], nc_list: List[Tensor],
                  cn_list: List[Tensor]):
        self.x = x
        self.e = e
        for i in range(len(attr_list)):
            self.attr_list.append(attr_list[i])
            self.adj_list.append(adj_list[i])
            self.nc_list.append(nc_list[i])
            self.cn_list.append(cn_list[i])

    def update_attr(self, x: Tensor, e: Tensor, sub_attrs: List[Tensor]):
        # x,e
        self.x = x
        self.e = e
        dense_x = x.new_zeros(list(self.mask.size()) + list(x.size())[1:])
        dense_x[self.x_idx] = x
        self.attr_list[0] = dense_x
        # sub-structure
        for i in range(len(sub_attrs)):
            self.attr_list[i + 1] = sub_attrs[i]
        # graph
        self.attr_list[-1] = torch.matmul(self.cn_list[-1], self.attr_list[-2])


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, xs):
        return torch.cat(xs, dim=-1)


class AHGConv(torch.nn.Module):
    def __init__(self,
                 cluster_dims: int = 1,
                 layer_dim: int = 32,
                 hidden: int = 32,
                 act_module=torch.nn.ReLU,
                 graph_norm=BatchNorm1d,
                 args=None):
        super(AHGConv, self).__init__()

        def get_lambda_x2x():
            return lambda x: x

        def get_msg_nn():
            return Sequential(Catter(), Linear(layer_dim * 2, layer_dim), act_module())

        def get_update_nn():
            return Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                              graph_norm(hidden), act_module())

        self.n_conv = NConv(msg_adj_nn=get_msg_nn(),
                            msg_up_nn=get_lambda_x2x(),
                            update_adj_nn=get_update_nn(),
                            update_up_nn=get_update_nn(),
                            combine_nn=Sequential(Linear(hidden * 2, hidden), graph_norm(hidden), act_module()),
                            args=args)

        self.e_conv = EConv(msg_adj_nn=get_lambda_x2x(),
                            msg_up_nn=get_lambda_x2x(),
                            update_adj_nn=get_update_nn(),
                            update_up_nn=get_update_nn(),
                            combine_nn=Sequential(Linear(hidden * 2, hidden), graph_norm(hidden), act_module()),
                            args=args)

        self.sub_convs = torch.nn.ModuleList()
        for dim in range(cluster_dims):
            mp = SubConv(dim + 1,
                         msg_adj_nn=get_lambda_x2x(),
                         msg_up_nn=get_lambda_x2x(),
                         msg_down_nn=get_lambda_x2x(),
                         update_adj_nn=get_update_nn(),
                         update_up_nn=get_update_nn(),
                         update_down_nn=get_update_nn(),
                         combine_nn=Sequential(Linear(hidden * 3, hidden), graph_norm(hidden), act_module()),
                         args=args)
            self.sub_convs.append(mp)

    def reset_parameters(self):
        reset(self.n_conv)
        reset(self.e_conv)
        reset(self.sub_convs)

    def forward(self, param: MPParam):
        x = self.n_conv.forward(param)
        e = self.e_conv.forward(param)
        xs = [x, e]
        for conv in self.sub_convs:
            xs.append(conv.forward(param))

        return xs


class NConv(torch.nn.Module):
    def __init__(self,
                 msg_adj_nn: Callable,
                 msg_up_nn: Callable,
                 update_adj_nn: Callable,
                 update_up_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 args=None):
        super(NConv, self).__init__()

        self.msg_adj_nn = msg_adj_nn
        self.msg_up_nn = msg_up_nn
        self.update_adj_nn = update_adj_nn
        self.update_up_nn = update_up_nn
        self.combine_nn = combine_nn

        self.initial_eps = eps
        if args.train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_adj_nn)
        reset(self.msg_up_nn)
        reset(self.update_adj_nn)
        reset(self.update_up_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def forward(self, param: MPParam):
        x = param.x
        e = param.e
        i, j = (1, 0)
        x_j = x[param.edge_index[j]]
        # msg from adj and edge

        msg_adj = self.msg_adj_nn((x_j, e))
        agg_adj = scatter(msg_adj, param.edge_index[i], dim=0, dim_size=x.size(0), reduce='sum')
        # scatter will involve duplicate edges, such as self-connect and multigraph.
        # cell-complex in CIN and sstree in TGAA are multigraph, two edges may be adjacent in two cycles.
        # get_dense_e() will ignore duplicate edges, but may loss messages.

        # msg from cluster
        msg_up = self.msg_up_nn(param.attr_list[1])  # b,c,f
        agg_up = torch.matmul(param.nc_list[1], msg_up)[param.x_idx]  # bnc,bcf->bnf

        # As in GIN, we can learn an injective update function for each multi-set
        out_adj = self.update_adj_nn(agg_adj + (1 + self.eps1) * x)
        out_up = self.update_up_nn(agg_up + (1 + self.eps2) * x)

        # We need to combine the two such that the output is injective
        return self.combine_nn(torch.cat([out_adj, out_up], dim=-1))


class EConv(torch.nn.Module):
    def __init__(self,
                 msg_adj_nn: Callable,
                 msg_up_nn: Callable,
                 update_adj_nn: Callable,
                 update_up_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 args=None):
        super(EConv, self).__init__()

        self.msg_adj_nn = msg_adj_nn
        self.msg_up_nn = msg_up_nn
        self.update_adj_nn = update_adj_nn
        self.update_up_nn = update_up_nn
        self.combine_nn = combine_nn

        self.initial_eps = eps
        if args.train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_adj_nn)
        reset(self.msg_up_nn)
        reset(self.update_adj_nn)
        reset(self.update_up_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def forward(self, param: MPParam):
        x = param.x
        e = param.e
        i, j = (1, 0)
        # e_ij will be involved twice by (i, j) and (j, i)

        # msg from nodes
        msg_adj = self.msg_adj_nn(x)
        agg_adj = msg_adj[param.edge_index[i]] + msg_adj[param.edge_index[j]]

        # msg from cluster
        msg_up = torch.matmul(param.nc_list[1], self.msg_up_nn(param.attr_list[1]))[param.x_idx]
        agg_up = msg_up[param.edge_index[i]] + msg_up[param.edge_index[j]]

        # As in GIN, we can learn an injective update function for each multi-set
        out_adj = self.update_adj_nn(agg_adj + (1 + self.eps1) * e)
        out_up = self.update_up_nn(agg_up + (1 + self.eps2) * e)

        # We need to combine the two such that the output is injective
        return self.combine_nn(torch.cat([out_adj, out_up], dim=-1))


class SubConv(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 msg_adj_nn: Callable,
                 msg_up_nn: Callable,
                 msg_down_nn: Callable,
                 update_adj_nn: Callable,
                 update_up_nn: Callable,
                 update_down_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 args=None):
        super(SubConv, self).__init__()

        self.dim = dim

        self.msg_adj_nn = msg_adj_nn
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.update_adj_nn = update_adj_nn
        self.update_up_nn = update_up_nn
        self.update_down_nn = update_down_nn
        self.combine_nn = combine_nn

        self.initial_eps = eps
        if args.train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps3 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
            self.register_buffer('eps3', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_adj_nn)
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.update_adj_nn)
        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)
        self.eps3.data.fill_(self.initial_eps)

    def forward(self, param: MPParam):
        dim = self.dim
        x = param.attr_list[dim]  # b,n,f
        adj = param.adj_list[dim]  # b,n,n, full-connected
        down_attr = param.attr_list[dim - 1]
        m_cn = param.cn_list[dim]  # important
        up_attr = param.attr_list[dim + 1]
        up_nc = param.nc_list[dim + 1]

        B, N, F = x.shape
        # msg from adj
        msg_adj = self.msg_adj_nn(x)
        agg_adj = torch.matmul(adj, msg_adj)

        # msg from up
        msg_up = self.msg_up_nn(up_attr)
        agg_up = torch.matmul(up_nc, msg_up)

        # msg from down
        msg_down = self.msg_down_nn(down_attr)
        agg_down = torch.matmul(m_cn, msg_down)

        # As in GIN, we can learn an injective update function for each multi-set
        out_adj = self.update_adj_nn((agg_adj + (1 + self.eps1) * x).view(B * N, F))
        out_up = self.update_up_nn((agg_up + (1 + self.eps2) * x).view(B * N, F))
        out_down = self.update_down_nn((agg_down + (1 + self.eps3) * x).view(B * N, F))

        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_adj, out_up, out_down], dim=-1)).view(B, N, F)


def get_cluster(in_size=32, k=4, method='none'):
    if method in ['uniform', 'normal', 'bernoulli', 'categorical']:
        return Inv_RandomPool(in_size=in_size, num_cluster=k, pooling_type=method)
    elif method == 'mem':
        return MemPooling(in_size=in_size, num_cluster=k, heads=1)
    else:
        return MasterNode()


def fetch_assign_matrix(random, dim1, dim2):
    if random == 'normal':
        m = torch.randn(dim1, dim2)
    elif random == 'bernoulli':
        m = torch.bernoulli(0.25 * torch.ones(dim1, dim2))
    elif random == 'categorical':
        idxs = torch.multinomial((1.0 / dim2) * torch.ones((dim1, dim2)), 1)
        m = torch.zeros(dim1, dim2)
        m[torch.arange(dim1), idxs.view(-1)] = 1.0
    else:
        m = torch.rand(dim1, dim2)  # uniform

    return m


def normalize(s: Tensor):
    return s / (s.sum(dim=-1, keepdim=True) + EPS)  # normalize


def generate_clusters(s: Tensor, n_x: Tensor, n_adj: Tensor):
    s = F.relu(s)  # sparse connection
    m_nc = normalize(s)  # N,C, membership of node to cluster
    m_cn = normalize(s.transpose(-2, -1))  # C,N, membership of cluster to node
    c_x = torch.matmul(m_cn, n_x)
    c_adj = torch.matmul(torch.matmul(m_cn, n_adj), m_nc)
    return c_x, c_adj, m_nc, m_cn


class MasterNode(torch.nn.Module):
    def __init__(self):
        super(MasterNode, self).__init__()

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor):
        m_nc = mask.unsqueeze(-1)
        m_cn = normalize(m_nc.transpose(-2, -1))
        c_x = torch.matmul(m_cn, x)
        c_adj = x.new_ones([x.size(0), 1, 1])
        loss = x.new_zeros(1)
        return c_x, c_adj, m_nc, m_cn, loss


class Inv_RandomPool(torch.nn.Module):
    def __init__(self, in_size, num_cluster, pooling_type):
        super(Inv_RandomPool, self).__init__()
        # invariant to node order and node number
        self.pooling_type = pooling_type
        self.rm = fetch_assign_matrix(pooling_type, in_size, num_cluster)

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor):
        s = F.relu(torch.matmul(x.detach(), self.rm.repeat(x.size(0), 1, 1)))
        s = s * mask.unsqueeze(-1)
        c_x, c_adj, m_nc, m_cn = generate_clusters(s, x, adj)
        loss = x.new_zeros(1)

        return c_x, c_adj, m_nc, m_cn, loss


class RandomPool(torch.nn.Module):
    def __init__(self, max_num_nodes, num_cluster, pooling_type):
        super(RandomPool, self).__init__()
        # require max_num_nodes of the dataset
        self.pooling_type = pooling_type
        self.rm = fetch_assign_matrix(pooling_type, max_num_nodes, num_cluster)

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor):
        # x: b,n,f
        s = self.rm[:x.size(1), :].repeat(x.size(0), 1, 1)
        s = s * mask.unsqueeze(-1)
        c_x, c_adj, m_nc, m_cn = generate_clusters(s, x, adj)
        loss = x.new_zeros(1)

        return c_x, c_adj, m_nc, m_cn, loss


class MemPooling(torch.nn.Module):
    def __init__(self, in_size: int, num_cluster: int, heads: int = 1, tau: float = 1.):
        super().__init__()
        self.in_size = in_size
        self.heads = heads
        self.num_cluster = num_cluster
        self.tau = tau

        self.k = Parameter(torch.empty(heads, num_cluster, in_size))
        self.conv = Conv2d(heads, 1, kernel_size=1, padding=0, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.k.data)
        reset(self.conv)

    @staticmethod
    def kl_loss(S: Tensor) -> Tensor:

        S_2 = S**2
        P = S_2 / S.sum(dim=1, keepdim=True)
        denom = P.sum(dim=2, keepdim=True)
        denom[S.sum(dim=2, keepdim=True) == 0.0] = 1.0
        P /= denom

        loss = KLDivLoss(reduction='batchmean', log_target=False)
        return loss(S.clamp(EPS).log(), P.clamp(EPS))

    def forward(self, x: Tensor, adj: Tensor, mask: Tensor, hops: int = 3):
        for _ in range(hops):
            x = torch.matmul(adj, x)

        (B, N, _), H, K = x.size(), self.heads, self.num_cluster

        dist = torch.cdist(self.k.view(H * K, -1), x.view(B * N, -1), p=2)**2
        dist = (1. + dist / self.tau).pow(-(self.tau + 1.0) / 2.0)

        dist = dist.view(H, K, B, N).permute(2, 0, 3, 1)  # [B, H, N, K]
        s = dist / dist.sum(dim=-1, keepdim=True)

        s = self.conv(s).squeeze(dim=1).softmax(dim=-1)  # [B, N, K]
        s = s * mask.unsqueeze(-1)

        c_x, c_adj, m_nc, m_cn = generate_clusters(s, x, adj)
        loss = self.kl_loss(s)

        return c_x, c_adj, m_nc, m_cn, loss
