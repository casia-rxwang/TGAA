import torch

from torch.nn import Linear, Embedding, Dropout
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from data.sstree import SSTreeBatch, MPParams
from model.nn import get_nonlinearity, get_graph_norm

from model.layers import NoEmbed, InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce, GCNCConv, GCNEConv, get_dense_x
from model.pool_tool import create_pool_weight, crate_final_readout
from torch_geometric.nn.inits import reset
from torch import Tensor
from typing import List


class GCNC(torch.nn.Module):
    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 num_layers,
                 hidden,
                 emb_method,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2,
                 jump_mode=None,
                 nonlinearity='relu',
                 readout='sum',
                 final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2),
                 final_readout='sum',
                 embed_edge=False,
                 embed_dim=None,
                 use_coboundaries=False,
                 graph_norm='bn',
                 args=None):
        super(GCNC, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim + 1))

        # self.in_dropout_rate = indropout_rate
        self.layer_drop = args.layer_drop
        self.pool_drop = args.pool_drop
        self.dropout_rate = dropout_rate

        if emb_method == 'embed':
            self.v_embed_init = Embedding(atom_types, embed_dim)
            self.e_embed_init = Embedding(bond_types, embed_dim) if embed_edge else None
            self.reduce_init = InitReduceConv(reduce=args.init_method)
            self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)
        elif emb_method == 'ogb':
            self.v_embed_init = AtomEncoder(embed_dim)
            self.e_embed_init = BondEncoder(embed_dim) if embed_edge else None
            self.reduce_init = InitReduceConv(reduce=args.init_method)
            self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)
            self.layer_drop = dropout_rate  # only for OGB
        elif emb_method == 'none':
            self.init_conv = NoEmbed()

        self.final_readout = final_readout
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.readout = readout
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                GCNCConv(up_msg_size=layer_dim,
                         down_msg_size=layer_dim,
                         boundary_msg_size=layer_dim,
                         passed_msg_boundaries_nn=None,
                         passed_msg_up_nn=None,
                         passed_update_up_nn=None,
                         passed_update_boundaries_nn=None,
                         max_dim=self.max_dim,
                         hidden=hidden,
                         act_module=act_module,
                         layer_dim=layer_dim,
                         graph_norm=self.graph_norm,
                         use_coboundaries=use_coboundaries,
                         args=args))

        if jump_mode == 'cat':
            self.jump = lambda xs: torch.cat(xs, dim=-1)
        else:
            self.jump = lambda xs: xs[-1]

        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden, bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        # final_readout
        self.final_readout_fun = crate_final_readout(final_readout)
        if final_readout == 'cat':
            self.lin2 = Linear(final_hidden_multiplier * hidden * len(self.readout_dims), out_size)
        else:
            self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

        self.pool_weight = create_pool_weight(embed_dim, hidden, args)

        self.layer_drop = Dropout(self.layer_drop) if self.layer_drop > 0 else lambda x: x
        self.pool_drop = Dropout(self.pool_drop) if self.pool_drop > 0 else lambda x: x
        self.final_drop = Dropout(self.dropout_rate) if self.dropout_rate > 0 else lambda x: x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

        reset(self.pool_weight)

    def jump_knowledge(self, jump_xs):
        # Perform JumpingKnowledge at each dim
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: SSTreeBatch):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        params = init_data_to_dense_gcnc(data, batch_size=data.num_data)
        xs = list(self.init_conv(*params))

        jump_xs = [[] for _ in xs]
        refs = xs  # refs
        params = update_params_gcnc(params, xs)

        for c, conv in enumerate(self.convs):
            start_to_process = 0
            xs = conv(*params, start_to_process=start_to_process)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = self.layer_drop(xs[i])
            params = update_params_gcnc(params, xs)

            for i, x in enumerate(xs):
                jump_xs[i] += [x]

        xs = self.jump_knowledge(jump_xs)

        # pool_weight
        for i in range(len(xs)):
            # data.cochains[i].batch for softmax
            w = self.pool_weight(xs[i], refs[i])
            w = self.pool_drop(w)
            xs[i] = torch.mul(w, xs[i])

        # pooling x: [N, C] to [B, C]
        for i in range(len(xs)):
            d_x_mask = params[i].x_mask  # [B, N_b]
            d_x_idx = params[i].x_idx
            dense_x = get_dense_x(xs[i], d_x_mask, d_x_idx)  # [B, N_b, C]
            xs[i] = torch.matmul(d_x_mask.unsqueeze(1), dense_x).squeeze(1)  # bi,bif->bf
        # Select the dimensions we want at the end.
        xs = [xs[i] if i < len(xs) else xs[0].new_zeros(xs[0].size()) for i in self.readout_dims]

        new_xs = []
        for i, x in enumerate(xs):
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = self.final_readout_fun(new_xs)
        x = self.final_drop(x)  # lin2

        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GCNE(torch.nn.Module):
    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 num_layers,
                 hidden,
                 emb_method,
                 dropout_rate: float = 0.5,
                 jump_mode=None,
                 nonlinearity='relu',
                 readout='sum',
                 final_hidden_multiplier: int = 2,
                 final_readout='sum',
                 embed_edge=False,
                 embed_dim=None,
                 use_coboundaries=False,
                 graph_norm='bn',
                 args=None):
        super(GCNE, self).__init__()

        # self.in_dropout_rate = indropout_rate
        self.layer_drop = args.layer_drop
        self.pool_drop = args.pool_drop
        self.dropout_rate = dropout_rate

        if emb_method == 'embed':
            self.v_embed_init = Embedding(atom_types, embed_dim)
            self.e_embed_init = Embedding(bond_types, embed_dim) if embed_edge else None
            self.reduce_init = InitReduceConv(reduce=args.init_method)
            self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)
        elif emb_method == 'ogb':
            self.v_embed_init = AtomEncoder(embed_dim)
            self.e_embed_init = BondEncoder(embed_dim) if embed_edge else None
            self.reduce_init = InitReduceConv(reduce=args.init_method)
            self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)
            self.layer_drop = dropout_rate  # only for OGB
        elif emb_method == 'none':
            self.init_conv = NoEmbed()

        self.final_readout = final_readout
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.readout = readout
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                GCNEConv(passed_msg_node_nn=None,
                         passed_msg_edge_nn=None,
                         passed_update_node_nn=None,
                         passed_update_edge_nn=None,
                         hidden=hidden,
                         act_module=act_module,
                         layer_dim=layer_dim,
                         graph_norm=self.graph_norm,
                         use_coboundaries=use_coboundaries,
                         args=args))

        if jump_mode == 'cat':
            self.jump = lambda xs: torch.cat(xs, dim=-1)
        else:
            self.jump = lambda xs: xs[-1]

        dims = 2
        self.lin1s = torch.nn.ModuleList()
        for _ in range(dims):
            if jump_mode == 'cat':
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden, bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        # final_readout
        self.final_readout_fun = crate_final_readout(final_readout)
        if final_readout == 'cat':
            self.lin2 = Linear(final_hidden_multiplier * hidden * dims, out_size)
        else:
            self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

        self.pool_weight = create_pool_weight(embed_dim, hidden, args)

        self.layer_drop = Dropout(self.layer_drop) if self.layer_drop > 0 else lambda x: x
        self.pool_drop = Dropout(self.pool_drop) if self.pool_drop > 0 else lambda x: x
        self.final_drop = Dropout(self.dropout_rate) if self.dropout_rate > 0 else lambda x: x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

        reset(self.pool_weight)

    def jump_knowledge(self, jump_xs):
        # Perform JumpingKnowledge at each dim
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: SSTreeBatch):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        params = init_data_to_dense_gcne(data, batch_size=data.num_data)
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

        xs = self.jump_knowledge(jump_xs)

        # pool_weight
        for i in range(len(xs)):
            w = self.pool_weight(xs[i], refs[i])
            w = self.pool_drop(w)
            xs[i] = torch.mul(w, xs[i])

        # pooling x: [N, C] to [B, C]
        for i in range(len(xs)):
            d_x_mask = params[i].x_mask  # [B, N_b]
            d_x_idx = params[i].x_idx
            dense_x = get_dense_x(xs[i], d_x_mask, d_x_idx)  # [B, N_b, C]
            xs[i] = torch.matmul(d_x_mask.unsqueeze(1), dense_x).squeeze(1)  # bi,bif->bf

        new_xs = []
        for i, x in enumerate(xs):
            new_xs.append(act(self.lin1s[i](x)))

        x = self.final_readout_fun(new_xs)
        x = self.final_drop(x)  # lin2

        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GCNC_MLP(torch.nn.Module):
    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 emb_method,
                 dropout_rate: float = 0.5,
                 max_dim: int = 2,
                 nonlinearity='relu',
                 readout='sum',
                 final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2),
                 final_readout='sum',
                 embed_edge=False,
                 embed_dim=None,
                 args=None):
        super(GCNC_MLP, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim + 1))

        if emb_method == 'embed':
            self.v_embed_init = Embedding(atom_types, embed_dim)
            self.e_embed_init = Embedding(bond_types, embed_dim) if embed_edge else None
            self.reduce_init = InitReduceConv(reduce=args.init_method)
            self.init_conv = EmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)
        elif emb_method == 'ogb':
            self.v_embed_init = AtomEncoder(embed_dim)
            self.e_embed_init = BondEncoder(embed_dim) if embed_edge else None
            self.reduce_init = InitReduceConv(reduce=args.init_method)
            self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)
        elif emb_method == 'none':
            self.init_conv = NoEmbed()

        self.final_readout = final_readout
        self.nonlinearity = nonlinearity
        self.readout = readout

        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            self.lin1s.append(Linear(embed_dim, final_hidden_multiplier * embed_dim))

        # final_readout
        self.final_readout_fun = crate_final_readout(final_readout)
        if final_readout == 'cat':
            self.lin2 = Linear(final_hidden_multiplier * embed_dim * len(self.readout_dims), out_size)
        else:
            self.lin2 = Linear(final_hidden_multiplier * embed_dim, out_size)

        self.final_drop = Dropout(dropout_rate) if dropout_rate > 0 else lambda x: x

    def reset_parameters(self):
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: SSTreeBatch):
        act = get_nonlinearity(self.nonlinearity, return_module=False)

        params = init_data_to_dense_gcnc(data, batch_size=data.num_data)
        xs = list(self.init_conv(*params))

        # pooling x: [N, C] to [B, C]
        for i in range(len(xs)):
            d_x_mask = params[i].x_mask  # [B, N_b]
            d_x_idx = params[i].x_idx
            dense_x = get_dense_x(xs[i], d_x_mask, d_x_idx)  # [B, N_b, C]
            xs[i] = torch.matmul(d_x_mask.unsqueeze(1), dense_x).squeeze(1)  # bi,bif->bf
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        new_xs = []
        for i, x in enumerate(xs):
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = self.final_readout_fun(new_xs)
        x = self.final_drop(x)  # lin2

        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


def init_data_to_dense_gcnc(data: SSTreeBatch, batch_size: int):
    '''
    for init, n and e is Scalar (uninit), c is None
    generate adj and idx
    '''
    # data may have no cycle
    # if data.dimension == 2:
    #     return init_data_to_dense_gcne(data, batch_size)

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

    # cycle
    c.x_mask = torch.zeros((batch_size, c.max_num), device=c.x_batch.device)
    c.x_mask[c.x_idx] = 1

    c.boundary_adj = torch.zeros((batch_size, c.max_num, e.max_num), device=c.x_batch.device)
    c.boundary_adj[c.boundary_index[0], c.boundary_index[1], c.boundary_index[2]] = 1
    c.boundary_attr_mask = e.x_mask
    c.boundary_attr_idx = e.x_idx

    return [n, e, c]


def update_params_gcnc(params: List[MPParams], xs: List[Tensor]):
    # data may have no cycle
    # if len(params) == 2:
    #     return update_params_gcne(params, xs)

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


def init_data_to_dense_gcne(data: SSTreeBatch, batch_size: int):
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


def update_params_gcne(params: List[MPParams], xs: List[Tensor]):
    n = params[0]
    e = params[1]

    n.x = xs[0]
    n.up_attr = xs[1][n.up_attr_idx]
    n.boundary_attr = None

    e.x = xs[1]
    e.up_attr = None
    e.boundary_attr = xs[0]

    return [n, e]
