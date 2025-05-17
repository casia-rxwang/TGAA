import torch

from torch import Tensor
from torch.nn import Linear, Embedding, Dropout
from torch_geometric.nn.inits import reset
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from typing import List

from data.complex import ComplexBatch, CochainMessagePassingParams
from mp.tgaa_layers import NoEmbed, InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce, SparseCINConv
from mp.nn import get_nonlinearity, get_graph_norm
from mp.weight_ro import create_ro_weight


class TGAA(torch.nn.Module):
    """
    A cellular version of GIN with some tailoring to nimbly work on molecules from the ZINC and OGB.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 num_layers,
                 hidden,
                 emb_method,
                 embed_dim=None,
                 jump_mode=None,
                 max_dim: int = 2,
                 use_coboundaries=False,
                 final_hidden_multiplier: int = 2,
                 nonlinearity='relu',
                 graph_norm='bn',
                 args=None):
        super(TGAA, self).__init__()

        self.max_dim = max_dim

        if embed_dim is None:
            embed_dim = hidden

        if emb_method == 'embed':
            self.init_conv = EmbedVEWithReduce(Embedding(atom_types, embed_dim), Embedding(bond_types, embed_dim),
                                               InitReduceConv(reduce=args.init_method))
        elif emb_method == 'ogb':
            self.init_conv = OGBEmbedVEWithReduce(AtomEncoder(embed_dim), BondEncoder(embed_dim),
                                                  InitReduceConv(reduce=args.init_method))
        elif emb_method == 'none':
            self.init_conv = NoEmbed()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                SparseCINConv(depth=i + 1,
                              max_dim=self.max_dim,
                              layer_dim=layer_dim,
                              hidden=hidden,
                              act_module=get_nonlinearity(nonlinearity, return_module=True),
                              graph_norm=get_graph_norm(graph_norm),
                              use_coboundaries=use_coboundaries,
                              args=args))

        if jump_mode == 'cat':
            self.jump = lambda xs: torch.cat(xs, dim=-1)
        else:
            self.jump = lambda xs: xs[-1]

        self.lin1s = torch.nn.ModuleList()
        self.ro_weights = torch.nn.ModuleList()

        for i in range(max_dim + 1):
            if jump_mode == 'cat':
                self.ro_weights.append(create_ro_weight(embed_dim, num_layers * hidden, args, i))
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden))
            else:
                self.ro_weights.append(create_ro_weight(embed_dim, hidden, args, i))
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        self.lin2 = Linear(final_hidden_multiplier * hidden * (max_dim + 1), out_size)

        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.layer_drop = Dropout(args.layer_drop) if args.layer_drop > 0 else lambda x: x
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        reset(self.init_conv)
        reset(self.convs)
        reset(self.ro_weights)
        reset(self.lin1s)
        reset(self.lin2)

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        # Embed and populate higher-levels
        params = data.get_all_cochain_params(include_down_features=False)
        xs = list(self.init_conv(*params))

        jump_xs = [[] for _ in xs]
        refs = xs  # for pool weights
        params = update_params(params, xs)

        for c, conv in enumerate(self.convs):
            start_to_process = 0
            xs = conv(*params, start_to_process=start_to_process)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = self.layer_drop(xs[i])
            params = update_params(params, xs)

            for i, x in enumerate(xs):
                jump_xs[i] += [x]

        xs = self.jump_complex(jump_xs)

        # readout
        for i in range(len(xs)):
            xs[i] = self.ro_weights[i](xs[i], refs[i], params[i].batch, params[i].batch_size)  # [B, F]

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


class TGAA_MLP(torch.nn.Module):
    """
    A cellular version of GIN with some tailoring to nimbly work on molecules from the ZINC and OGB.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 emb_method,
                 embed_dim=None,
                 max_dim: int = 2,
                 final_hidden_multiplier: int = 2,
                 nonlinearity='relu',
                 args=None):
        super(TGAA_MLP, self).__init__()

        self.reduce = args.init_method
        self.max_dim = max_dim

        if emb_method == 'embed':
            self.init_conv = EmbedVEWithReduce(Embedding(atom_types, embed_dim), Embedding(bond_types, embed_dim),
                                               InitReduceConv(reduce=args.init_method))
        elif emb_method == 'ogb':
            self.init_conv = OGBEmbedVEWithReduce(AtomEncoder(embed_dim), BondEncoder(embed_dim),
                                                  InitReduceConv(reduce=args.init_method))
        elif emb_method == 'none':
            self.init_conv = NoEmbed()

        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            self.lin1s.append(Linear(embed_dim, final_hidden_multiplier * embed_dim))

        self.lin2 = Linear(final_hidden_multiplier * embed_dim * (max_dim + 1), out_size)

        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        reset(self.init_conv)
        reset(self.lin1s)
        reset(self.lin2)

    def forward(self, data: ComplexBatch):
        # Embed and populate higher-levels
        params = data.get_all_cochain_params(include_down_features=False)
        xs = list(self.init_conv(*params))

        params = update_params(params, xs)

        for i in range(len(xs)):
            xs[i] = scatter(xs[i], params[i].batch, dim=0, dim_size=params[i].batch_size, reduce=self.reduce)

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


def update_params(params: List[CochainMessagePassingParams], xs: List[Tensor]):
    # data may have no 2-cell
    # if len(params) == 2:
    #     return update_params_gcne(params, xs)

    n = params[0]
    e = params[1]
    c = params[2]

    n.x = xs[0]
    n.up_attr = torch.index_select(xs[1], 0, n.shared_coboundaries)
    n.boundary_attr = None

    e.x = xs[1]
    e.up_attr = torch.index_select(xs[2], 0, e.shared_coboundaries)
    e.boundary_attr = xs[0]

    c.x = xs[2]
    c.up_attr = None
    c.boundary_attr = xs[1]

    return [n, e, c]
