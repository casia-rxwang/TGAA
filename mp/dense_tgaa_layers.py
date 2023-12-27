import torch

from torch import LongTensor, Tensor
from data.complex import CochainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from abc import ABC, abstractmethod

from mp.weight_mp import create_fb_weight, create_sb_weight
from typing import Callable, Optional, Tuple


class TGAAModule(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 layer_dim: int,
                 msg_up_nn: Callable,
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 args=None):
        super(TGAAModule, self).__init__()

        self.dim = dim
        self.layer_dim = layer_dim
        self.msg_up_nn = msg_up_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.update_up_nn = update_up_nn
        self.update_boundaries_nn = update_boundaries_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if args.train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))

        self.fb_weight = create_fb_weight(layer_dim, args)
        self.sb_weight = create_sb_weight(args.sb_cal)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

        reset(self.fb_weight)

    def forward(self, cochain: CochainMessagePassingParams):
        x = cochain.x
        up_attr = cochain.up_attr
        boundary_attr = cochain.boundary_attr

        # up messaging and aggregation
        if up_attr is not None:
            # up_x_i = x[cochain.up_x_i_idx]  # will change BP
            up_x_j = x[cochain.up_x_j_idx]
            out = self.msg_up_nn((up_x_j, up_attr))

            w = self.fb_weight(x[cochain.up_x_i_idx], up_x_j, up_attr)
            out = torch.mul(w, out)
            out_up = self.aggregate_up(out, cochain)
        else:
            out_up = torch.zeros(x.size(0), self.layer_dim, device=x.device)

        # boundary messaging and aggregation
        if boundary_attr is not None:
            out = self.msg_boundaries_nn(boundary_attr)
            out_boundaries = self.aggregate_boundary(out, cochain)
        else:
            out_boundaries = torch.zeros(x.size(0), self.layer_dim, device=x.device)

        # As in GIN, we can learn an injective update function for each multi-set
        out_up += (1 + self.eps1) * x
        out_boundaries += (1 + self.eps2) * x
        out_up = self.update_up_nn(out_up)
        out_boundaries = self.update_boundaries_nn(out_boundaries)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1))

    def aggregate_up(self, msg_up: Tensor, cochain: CochainMessagePassingParams) -> Tensor:
        w = self.sb_weight(cochain.up_adj)
        d_up_attr = get_dense_up_attr(msg_up, w, (cochain.up_index[0], cochain.up_index[1], cochain.up_index[2]))
        d_out_up = torch.matmul(w.unsqueeze(2), d_up_attr).squeeze(2)
        return d_out_up[cochain.x_idx]

    def aggregate_boundary(self, msg_boundary: Tensor, cochain: CochainMessagePassingParams) -> Tensor:
        w = self.sb_weight(cochain.boundary_adj)
        d_boundary = get_dense_x(msg_boundary, cochain.boundary_attr_mask, cochain.boundary_attr_idx)
        d_out_boundaries = torch.matmul(w, d_boundary)
        return d_out_boundaries[cochain.x_idx]


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, xs):
        return torch.cat(xs, dim=-1)


class TGAAConv(torch.nn.Module):
    def __init__(self,
                 passed_msg_up_nn: Optional[Callable] = None,
                 passed_msg_boundaries_nn: Optional[Callable] = None,
                 passed_update_up_nn: Optional[Callable] = None,
                 passed_update_boundaries_nn: Optional[Callable] = None,
                 max_dim: int = 2,
                 hidden: int = 32,
                 act_module=torch.nn.ReLU,
                 layer_dim: int = 32,
                 graph_norm=BN,
                 use_coboundaries=False,
                 args=None):
        super(TGAAConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim + 1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    msg_up_nn = Sequential(Catter(), Linear(layer_dim * 2, layer_dim), act_module())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                          graph_norm(hidden), act_module())

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(),
                                                  Linear(hidden, hidden), graph_norm(hidden), act_module())

            combine_nn = Sequential(Linear(hidden * 2, hidden), graph_norm(hidden), act_module())

            mp = TGAAModule(dim,
                            layer_dim=layer_dim,
                            msg_up_nn=msg_up_nn,
                            msg_boundaries_nn=msg_boundaries_nn,
                            update_up_nn=update_up_nn,
                            update_boundaries_nn=update_boundaries_nn,
                            combine_nn=combine_nn,
                            args=args)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim + 1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class MPNNConv(torch.nn.Module):
    def __init__(self,
                 passed_msg_node_nn: Optional[Callable] = None,
                 passed_msg_edge_nn: Optional[Callable] = None,
                 passed_update_node_nn: Optional[Callable] = None,
                 passed_update_edge_nn: Optional[Callable] = None,
                 hidden: int = 32,
                 act_module=torch.nn.ReLU,
                 layer_dim: int = 32,
                 graph_norm=BN,
                 use_coboundaries=False,
                 eps: float = 0.,
                 args=None):
        super(MPNNConv, self).__init__()

        self.msg_node_nn = passed_msg_node_nn
        if self.msg_node_nn is None:
            if use_coboundaries:
                self.msg_node_nn = Sequential(Catter(), Linear(layer_dim * 2, layer_dim), act_module())
            else:
                self.msg_node_nn = lambda xs: xs[0]

        self.msg_edge_nn = passed_msg_edge_nn
        if self.msg_edge_nn is None:
            self.msg_edge_nn = lambda x: x

        self.update_node_nn = passed_update_node_nn
        if self.update_node_nn is None:
            self.update_node_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(),
                                             Linear(hidden, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                             graph_norm(hidden), act_module())

        self.update_edge_nn = passed_update_edge_nn
        if self.update_edge_nn is None:
            self.update_edge_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(),
                                             Linear(hidden, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                             graph_norm(hidden), act_module())

        self.initial_eps = eps
        if args.train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))

        self.fb_weight = create_fb_weight(layer_dim, args)
        self.sb_weight = create_sb_weight(args.sb_cal)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_node_nn)
        reset(self.msg_edge_nn)
        reset(self.update_node_nn)
        reset(self.update_edge_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

        reset(self.fb_weight)

    def forward(self, *cochain_params: CochainMessagePassingParams):
        n_x = cochain_params[0].x
        n_up_x_i = n_x[cochain_params[0].up_x_i_idx]
        n_up_x_j = n_x[cochain_params[0].up_x_j_idx]
        n_up_attr = cochain_params[0].up_attr
        e_x = cochain_params[1].x

        # node
        out_n = self.msg_node_nn((n_up_x_j, n_up_attr))
        w = self.fb_weight(n_up_x_i, n_up_x_j, n_up_attr)
        out_n = torch.mul(w, out_n)
        n_out = self.aggregate_node(out_n, cochain_params[0])

        # edge
        out_e = self.msg_edge_nn(n_x)
        e_out = self.aggregate_edge(out_e, cochain_params[1])

        # As in GIN, we can learn an injective update function for each multi-set
        n_out += (1 + self.eps1) * n_x
        e_out += (1 + self.eps2) * e_x
        n_out = self.update_node_nn(n_out)
        e_out = self.update_edge_nn(e_out)

        return [n_out, e_out]

    def aggregate_node(self, msg_up: Tensor, cochain: CochainMessagePassingParams) -> Tensor:
        w = self.sb_weight(cochain.up_adj)
        d_up_attr = get_dense_up_attr(msg_up, w, (cochain.up_index[0], cochain.up_index[1], cochain.up_index[2]))
        d_out_up = torch.matmul(w.unsqueeze(2), d_up_attr).squeeze(2)
        return d_out_up[cochain.x_idx]

    def aggregate_edge(self, msg_boundary: Tensor, cochain: CochainMessagePassingParams) -> Tensor:
        w = self.sb_weight(cochain.boundary_adj)
        d_boundary = get_dense_x(msg_boundary, cochain.boundary_attr_mask, cochain.boundary_attr_idx)
        d_out_boundaries = torch.matmul(w, d_boundary)
        return d_out_boundaries[cochain.x_idx]


class InitReduceConv(torch.nn.Module):
    def __init__(self, reduce='add'):
        """
        Args:
            reduce (str): Way to aggregate boundaries. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = create_sb_weight(reduce)

    def forward(self, boundary_x, boundary_adj):
        """
        Args:
            boundary_x (Tensor): B * N_j * F"
            boundary_adj (Tensor): B * N_i * N_j"
        Outs:
            out (Tensor): B * N * F
        """
        boundary_adj = self.reduce(boundary_adj)
        out = torch.matmul(boundary_adj, boundary_x)  # reduce='add', bij,bjf->bif
        return out


class AbstractEmbedVEWithReduce(torch.nn.Module, ABC):
    def __init__(self, v_embed_layer: Callable, e_embed_layer: Optional[Callable], init_reduce: InitReduceConv):
        """
        Args:
            v_embed_layer: Layer to embed the integer features of the vertices
            e_embed_layer: Layer (potentially None) to embed the integer features of the edges.
            init_reduce: Layer to initialise the 2D cell features and potentially the edge features.
        """
        super(AbstractEmbedVEWithReduce, self).__init__()
        self.v_embed_layer = v_embed_layer
        self.e_embed_layer = e_embed_layer
        self.init_reduce = init_reduce

    @abstractmethod
    def _prepare_v_inputs(self, v_params):
        pass

    @abstractmethod
    def _prepare_e_inputs(self, e_params):
        pass

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert 1 <= len(cochain_params) <= 3
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        c_params = cochain_params[2] if len(cochain_params) == 3 else None

        vx = self.v_embed_layer(self._prepare_v_inputs(v_params))
        out = [vx]

        if e_params is None:
            assert c_params is None
            return out

        dense_vx = get_dense_x(vx, v_params.x_mask, v_params.x_idx)  # sparse to dense
        reduced_ex = self.init_reduce(dense_vx, e_params.boundary_adj)  # dense
        ex = reduced_ex[e_params.x_idx]  # dense to sparse

        if e_params.x is not None:
            ex = self.e_embed_layer(self._prepare_e_inputs(e_params))
            # The output of this should be the same size as the vertex features.
            # assert ex.size(1) == vx.size(1)
        out.append(ex)

        if c_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            dense_cx = self.init_reduce(reduced_ex, c_params.boundary_adj) / 2.
            cx = dense_cx[c_params.x_idx]  # dense to sparse
            out.append(cx)

        return out

    def reset_parameters(self):
        reset(self.v_embed_layer)
        reset(self.e_embed_layer)


class EmbedVEWithReduce(AbstractEmbedVEWithReduce):
    def __init__(self, v_embed_layer: torch.nn.Embedding, e_embed_layer: Optional[torch.nn.Embedding],
                 init_reduce: InitReduceConv):
        super(EmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)

    def _prepare_v_inputs(self, v_params):
        # assert v_params.x is not None
        # assert v_params.x.dim() == 2
        # assert v_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return v_params.x.squeeze(1).to(dtype=torch.long)

    def _prepare_e_inputs(self, e_params):
        # assert self.e_embed_layer is not None
        # assert e_params.x.dim() == 2
        # assert e_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return e_params.x.squeeze(1).to(dtype=torch.long)


class OGBEmbedVEWithReduce(AbstractEmbedVEWithReduce):
    def __init__(self, v_embed_layer: AtomEncoder, e_embed_layer: Optional[BondEncoder], init_reduce: InitReduceConv):
        super(OGBEmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)

    def _prepare_v_inputs(self, v_params):
        # assert v_params.x is not None
        # assert v_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return v_params.x.to(dtype=torch.long)

    def _prepare_e_inputs(self, e_params):
        # assert self.e_embed_layer is not None
        # assert e_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return e_params.x.to(dtype=torch.long)


class NoEmbed(torch.nn.Module):
    def __init__(self):
        super(NoEmbed, self).__init__()

    def forward(self, *cochain_params: CochainMessagePassingParams):
        out = []
        for c in cochain_params:
            out.append(c.x)
        return out


def get_dense_x(x: Tensor, d_x_mask: Tensor, d_x_idx: Tuple[LongTensor, LongTensor]):
    dense_x = x.new_zeros(list(d_x_mask.size()) + list(x.size())[1:])
    dense_x[d_x_idx] = x
    return dense_x


def get_dense_up_attr(up_attr: Tensor, up_adj: Tensor, up_idx: Tuple[LongTensor, LongTensor, LongTensor]):
    dense_up_attr = up_attr.new_zeros(list(up_adj.size()) + list(up_attr.size())[1:])
    dense_up_attr[up_idx] = up_attr
    return dense_up_attr
