import torch

from abc import ABC, abstractmethod
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn.inits import reset
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from data.complex import CochainMessagePassingParams
from mp.weight_mp import create_mp_weight
from typing import Callable, Optional


class SparseCINCochainConv(torch.nn.Module):
    """This is a CIN Cochain layer that operates of boundaries and upper adjacent cells."""

    def __init__(self,
                 depth: int,
                 layer_dim: int,
                 msg_up_nn: Callable,
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 args=None):
        super(SparseCINCochainConv, self).__init__()

        self.reduce = args.init_method
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

        self.mp_weight = create_mp_weight(layer_dim, args, depth)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)
        reset(self.mp_weight)

    def forward(self, cochain: CochainMessagePassingParams):
        x = cochain.x
        up_attr = cochain.up_attr
        boundary_attr = cochain.boundary_attr

        i, j = (1, 0)
        # up messaging and aggregation
        if up_attr is not None:
            x_i = x[cochain.up_index[i]]
            x_j = x[cochain.up_index[j]]

            msg_up = self.msg_up_nn((x_j, up_attr))
            agg_up = self.mp_weight(msg_up, x_i, x_j, up_attr, cochain.up_index[i], num_nodes=x.size(0))
        else:
            agg_up = torch.zeros(x.size(0), self.layer_dim, device=x.device)

        # boundary messaging and aggregation
        if boundary_attr is not None:
            msg_boundaries = self.msg_boundaries_nn(boundary_attr)
            msg_boundaries_j = msg_boundaries[cochain.boundary_index[j]]
            agg_boundaries = scatter(msg_boundaries_j, cochain.boundary_index[i], dim=0, dim_size=x.size(0), reduce=self.reduce)
        else:
            agg_boundaries = torch.zeros(x.size(0), self.layer_dim, device=x.device)

        # As in GIN, we can learn an injective update function for each multi-set
        out_up = self.update_up_nn(agg_up + (1 + self.eps1) * x)
        out_boundaries = self.update_boundaries_nn(agg_boundaries + (1 + self.eps1) * x)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1))


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, xs):
        return torch.cat(xs, dim=-1)


class SparseCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from  cellular upper
    neighbors and boundaries, but not from lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self,
                 depth: int,
                 passed_msg_up_nn: Optional[Callable] = None,
                 passed_msg_boundaries_nn: Optional[Callable] = None,
                 passed_update_up_nn: Optional[Callable] = None,
                 passed_update_boundaries_nn: Optional[Callable] = None,
                 max_dim: int = 2,
                 layer_dim: int = 32,
                 hidden: int = 32,
                 act_module=torch.nn.ReLU,
                 graph_norm=BN,
                 use_coboundaries=False,
                 args=None):
        super(SparseCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim + 1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    msg_up_nn = Sequential(Catter(), Linear(layer_dim * 2, layer_dim), act_module())
                else:
                    def msg_up_nn(xs): return xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                def msg_boundaries_nn(x): return x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                          graph_norm(hidden), act_module())

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(),
                                                  Linear(hidden, hidden), graph_norm(hidden), act_module())

            combine_nn = Sequential(Linear(hidden * 2, hidden), graph_norm(hidden), act_module())

            mp = SparseCINCochainConv(depth=depth,
                                      layer_dim=layer_dim,
                                      msg_up_nn=msg_up_nn,
                                      msg_boundaries_nn=msg_boundaries_nn,
                                      update_up_nn=update_up_nn,
                                      update_boundaries_nn=update_boundaries_nn,
                                      combine_nn=combine_nn,
                                      args=args)
            self.mp_levels.append(mp)

    def reset_parameters(self):
        reset(self.mp_levels)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim + 1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class InitReduceConv(torch.nn.Module):
    def __init__(self, reduce='add'):
        """
        Args:
            reduce (str): Way to aggregate boundaries. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = reduce

    def forward(self, boundary_x, boundary_index, out_size):
        features = boundary_x.index_select(0, boundary_index[0])
        return scatter(features, boundary_index[1], dim=0, dim_size=out_size, reduce=self.reduce)


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
        # assert 1 <= len(cochain_params) <= 3
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        c_params = cochain_params[2] if len(cochain_params) == 3 else None

        vx = self.v_embed_layer(self._prepare_v_inputs(v_params))
        out = [vx]

        if e_params is None:
            # assert c_params is None
            return out

        reduced_ex = self.init_reduce(vx, e_params.boundary_index, e_params.cum_num[-1])
        ex = reduced_ex

        if e_params.x is not None:
            ex = self.e_embed_layer(self._prepare_e_inputs(e_params))
            # The output of this should be the same size as the vertex features.
            # assert ex.size(1) == vx.size(1)
        out.append(ex)

        if c_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            cx = self.init_reduce(reduced_ex, c_params.boundary_index, c_params.cum_num[-1]) / 2.
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
