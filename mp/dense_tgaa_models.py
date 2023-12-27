import torch

from torch.nn import Linear, Embedding, Dropout
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from data.complex import ComplexBatch, CochainMessagePassingParams
from mp.nn import get_nonlinearity, get_graph_norm

from mp.dense_tgaa_layers import NoEmbed, InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce, TGAAConv, MPNNConv, get_dense_x
from mp.weight_pool import create_pool_weight
from torch_geometric.nn.inits import reset
from torch import Tensor
from typing import List


class TGAA(torch.nn.Module):
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
                 readout='sum',
                 args=None):
        super(TGAA, self).__init__()

        self.max_dim = max_dim
        self.readout = readout

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
                TGAAConv(max_dim=self.max_dim,
                         hidden=hidden,
                         act_module=get_nonlinearity(nonlinearity, return_module=True),
                         layer_dim=layer_dim,
                         graph_norm=get_graph_norm(graph_norm),
                         use_coboundaries=use_coboundaries,
                         args=args))

        if jump_mode == 'cat':
            self.jump = lambda xs: torch.cat(xs, dim=-1)
        else:
            self.jump = lambda xs: xs[-1]

        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden, bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        self.lin2 = Linear(final_hidden_multiplier * hidden * (max_dim + 1), out_size)

        self.pool_weight = create_pool_weight(embed_dim, hidden, args)
        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.layer_drop = Dropout(args.layer_drop) if args.layer_drop > 0 else lambda x: x
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

        reset(self.pool_weight)

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, include_partial=False):
        # Embed and populate higher-levels
        params = init_cells_to_dense_gcnc(data, batch_size=data.num_complexes)
        xs = list(self.init_conv(*params))

        jump_xs = [[] for _ in xs]
        refs = xs  # refs
        params = update_params_gcnc(params, xs)

        for c, conv in enumerate(self.convs):
            xs = conv(*params)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = self.layer_drop(xs[i])
            params = update_params_gcnc(params, xs)

            for i, x in enumerate(xs):
                jump_xs[i] += [x]

        xs = self.jump_complex(jump_xs)

        # pool
        for i in range(len(xs)):
            w = self.pool_weight(xs[i], refs[i])
            xs[i] = torch.mul(w, xs[i])

        # readout
        for i in range(len(xs)):
            d_x_mask = params[i].x_mask
            d_x_idx = params[i].x_idx
            dense_x = get_dense_x(xs[i], d_x_mask, d_x_idx)
            xs[i] = torch.matmul(d_x_mask.unsqueeze(1), dense_x).squeeze(1)

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


class MPNN(torch.nn.Module):
    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 num_layers,
                 hidden,
                 emb_method,
                 embed_dim=None,
                 jump_mode=None,
                 max_dim: int = 1,
                 use_coboundaries=False,
                 final_hidden_multiplier: int = 2,
                 nonlinearity='relu',
                 graph_norm='bn',
                 readout='sum',
                 args=None):
        super(MPNN, self).__init__()

        self.max_dim = max_dim
        self.readout = readout

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
                MPNNConv(hidden=hidden,
                         act_module=get_nonlinearity(nonlinearity, return_module=True),
                         layer_dim=layer_dim,
                         graph_norm=get_graph_norm(graph_norm),
                         use_coboundaries=use_coboundaries,
                         args=args))

        if jump_mode == 'cat':
            self.jump = lambda xs: torch.cat(xs, dim=-1)
        else:
            self.jump = lambda xs: xs[-1]

        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                # These layers don't use a bias. Then, in case a level is not present the output
                # is just zero and it is not given by the biases.
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden, bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        self.lin2 = Linear(final_hidden_multiplier * hidden * (max_dim + 1), out_size)

        self.pool_weight = create_pool_weight(embed_dim, hidden, args)
        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.layer_drop = Dropout(args.layer_drop) if args.layer_drop > 0 else lambda x: x
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

        reset(self.pool_weight)

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, include_partial=False):
        # Embed and populate higher-levels
        params = init_cells_to_dense_gcne(data, batch_size=data.num_complexes)
        xs = list(self.init_conv(*params))

        jump_xs = [[] for _ in xs]
        refs = xs  # refs
        params = update_params_gcne(params, xs)

        for c, conv in enumerate(self.convs):
            xs = conv(*params)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = self.layer_drop(xs[i])
            params = update_params_gcne(params, xs)

            for i, x in enumerate(xs):
                jump_xs[i] += [x]

        xs = self.jump_complex(jump_xs)

        # pool
        for i in range(len(xs)):
            w = self.pool_weight(xs[i], refs[i])
            xs[i] = torch.mul(w, xs[i])

        # readout
        for i in range(len(xs)):
            d_x_mask = params[i].x_mask
            d_x_idx = params[i].x_idx
            dense_x = get_dense_x(xs[i], d_x_mask, d_x_idx)
            xs[i] = torch.matmul(d_x_mask.unsqueeze(1), dense_x).squeeze(1)

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


class TGAA_MLP(torch.nn.Module):
    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 emb_method,
                 embed_dim=None,
                 max_dim: int = 2,
                 final_hidden_multiplier: int = 2,
                 nonlinearity='relu',
                 readout='sum',
                 args=None):
        super(TGAA_MLP, self).__init__()

        self.max_dim = max_dim
        self.readout = readout

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
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ComplexBatch, include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        # Embed and populate higher-levels
        params = init_cells_to_dense_gcnc(data, batch_size=data.num_complexes)
        xs = list(self.init_conv(*params))

        # readout
        for i in range(len(xs)):
            d_x_mask = params[i].x_mask
            d_x_idx = params[i].x_idx
            dense_x = get_dense_x(xs[i], d_x_mask, d_x_idx)
            xs[i] = torch.matmul(d_x_mask.unsqueeze(1), dense_x).squeeze(1)

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


def init_cells_to_dense_gcnc(data: ComplexBatch, batch_size: int):
    '''
    for init, n and e is Scalar (uninit), c is None
    generate adj and idx
    '''
    params = data.get_all_cochain_params(max_dim=2)
    n = params[0]
    e = params[1]
    c = params[2]

    n_num = torch.tensor(n.x_num, device=n.x_batch.device)
    e_num = torch.tensor(e.x_num, device=e.x_batch.device)
    c_num = torch.tensor(c.x_num, device=c.x_batch.device)

    n_cum = torch.cat([n_num.new_zeros(1), n_num.cumsum(dim=0)])
    e_cum = torch.cat([e_num.new_zeros(1), e_num.cumsum(dim=0)])
    c_cum = torch.cat([c_num.new_zeros(1), c_num.cumsum(dim=0)])

    # node
    n.x_mask = torch.zeros((batch_size, n.max_num), device=n.x_batch.device)
    n.x_mask[n.x_idx] = 1

    n.up_adj = torch.zeros((batch_size, n.max_num, n.max_num), device=n.x_batch.device)
    n.up_adj[n.up_index[0], n.up_index[1], n.up_index[2]] = 1

    n.up_x_i_idx = n_cum[n.up_index[0]] + n.up_index[1]
    n.up_x_j_idx = n_cum[n.up_index[0]] + n.up_index[2]
    n.up_attr_idx = e_cum[n.up_index[0]] + n.shared_coboundaries

    # edge
    e.x_mask = torch.zeros((batch_size, e.max_num), device=e.x_batch.device)
    e.x_mask[e.x_idx] = 1

    e.up_adj = torch.zeros((batch_size, e.max_num, e.max_num), device=e.x_batch.device)
    e.up_adj[e.up_index[0], e.up_index[1], e.up_index[2]] = 1

    e.up_x_i_idx = e_cum[e.up_index[0]] + e.up_index[1]
    e.up_x_j_idx = e_cum[e.up_index[0]] + e.up_index[2]
    e.up_attr_idx = c_cum[e.up_index[0]] + e.shared_coboundaries

    e.boundary_adj = torch.zeros((batch_size, e.max_num, n.max_num), device=e.x_batch.device)
    e.boundary_adj[e.boundary_index[0], e.boundary_index[1], e.boundary_index[2]] = 1
    e.boundary_attr_mask = n.x_mask
    e.boundary_attr_idx = n.x_idx

    # cell
    c.x_mask = torch.zeros((batch_size, c.max_num), device=c.x_batch.device)
    c.x_mask[c.x_idx] = 1

    c.boundary_adj = torch.zeros((batch_size, c.max_num, e.max_num), device=c.x_batch.device)
    c.boundary_adj[c.boundary_index[0], c.boundary_index[1], c.boundary_index[2]] = 1
    c.boundary_attr_mask = e.x_mask
    c.boundary_attr_idx = e.x_idx

    return [n, e, c]


def update_params_gcnc(params: List[CochainMessagePassingParams], xs: List[Tensor]):
    n = params[0]
    e = params[1]
    c = params[2]

    n.x = xs[0]
    n.up_attr = xs[1][n.up_attr_idx]
    n.boundary_attr = None

    e.x = xs[1]
    e.up_attr = xs[2][e.up_attr_idx]
    e.boundary_attr = xs[0]

    c.x = xs[2]
    c.up_attr = None
    c.boundary_attr = xs[1]

    return [n, e, c]


def init_cells_to_dense_gcne(data: ComplexBatch, batch_size: int):
    '''
    for init, n and e is Scalar (uninit), c is None
    generate adj and idx
    '''
    params = data.get_all_cochain_params(max_dim=1)
    n = params[0]
    e = params[1]

    n_num = torch.tensor(n.x_num, device=n.x_batch.device)
    e_num = torch.tensor(e.x_num, device=e.x_batch.device)

    n_cum = torch.cat([n_num.new_zeros(1), n_num.cumsum(dim=0)])
    e_cum = torch.cat([e_num.new_zeros(1), e_num.cumsum(dim=0)])

    # node
    n.x_mask = torch.zeros((batch_size, n.max_num), device=n.x_batch.device)
    n.x_mask[n.x_idx] = 1

    n.up_adj = torch.zeros((batch_size, n.max_num, n.max_num), device=n.x_batch.device)
    n.up_adj[n.up_index[0], n.up_index[1], n.up_index[2]] = 1

    n.up_x_i_idx = n_cum[n.up_index[0]] + n.up_index[1]
    n.up_x_j_idx = n_cum[n.up_index[0]] + n.up_index[2]
    n.up_attr_idx = e_cum[n.up_index[0]] + n.shared_coboundaries

    # edge
    e.x_mask = torch.zeros((batch_size, e.max_num), device=e.x_batch.device)
    e.x_mask[e.x_idx] = 1

    e.boundary_adj = torch.zeros((batch_size, e.max_num, n.max_num), device=e.x_batch.device)
    e.boundary_adj[e.boundary_index[0], e.boundary_index[1], e.boundary_index[2]] = 1
    e.boundary_attr_mask = n.x_mask
    e.boundary_attr_idx = n.x_idx

    return [n, e]


def update_params_gcne(params: List[CochainMessagePassingParams], xs: List[Tensor]):
    n = params[0]
    e = params[1]

    n.x = xs[0]
    n.up_attr = xs[1][n.up_attr_idx]
    n.boundary_attr = None

    e.x = xs[1]
    e.up_attr = None
    e.boundary_attr = xs[0]

    return [n, e]
