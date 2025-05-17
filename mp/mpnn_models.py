import torch
from torch.nn import Linear, Embedding, Dropout, Identity, Sequential, BatchNorm1d

from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from mp.nn import get_nonlinearity, get_graph_norm
from mp.weight_mp import create_mp_weight
from mp.weight_ro import create_ro_weight


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
                 final_hidden_multiplier: int = 2,
                 nonlinearity='relu',
                 graph_norm='bn',
                 args=None):
        super(MPNN, self).__init__()

        self.dims = 2  # node, edge

        if embed_dim is None:
            embed_dim = hidden

        # embed layers
        if emb_method == 'embed':
            self.v_embed_init = Sequential(PreEmbed(), Embedding(atom_types, embed_dim))
            self.e_embed_init = Sequential(PreEmbed(), Embedding(bond_types, embed_dim))
        elif emb_method == 'ogb':
            self.v_embed_init = AtomEncoder(embed_dim)
            self.e_embed_init = BondEncoder(embed_dim)
        elif emb_method == 'none':
            self.v_embed_init = Identity()
            self.e_embed_init = Identity()

        # conv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                MPNNConv(depth=i + 1,
                         layer_dim=layer_dim,
                         hidden=hidden,
                         act_module=get_nonlinearity(nonlinearity, return_module=True),
                         graph_norm=get_graph_norm(graph_norm),
                         args=args))

        # jump model
        if jump_mode == 'cat':
            self.jump = lambda xs: torch.cat(xs, dim=-1)
        else:
            self.jump = lambda xs: xs[-1]

        self.ro_weights = torch.nn.ModuleList()
        self.lin1s = torch.nn.ModuleList()

        for i in range(self.dims):
            if jump_mode == 'cat':
                self.ro_weights.append(create_ro_weight(embed_dim, num_layers * hidden, args, i))
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden))
            else:
                self.ro_weights.append(create_ro_weight(embed_dim, hidden, args, i))
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        self.lin2 = Linear(final_hidden_multiplier * hidden * self.dims, out_size)

        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.layer_drop = Dropout(args.layer_drop) if args.layer_drop > 0 else lambda x: x
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        reset(self.v_embed_init)
        reset(self.e_embed_init)
        reset(self.convs)
        reset(self.ro_weights)
        reset(self.lin1s)
        reset(self.lin2)

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def init_mpparam(self, data: Data):
        edge_index = data.edge_index
        x0 = data.x
        if data.edge_attr is None:
            e0 = x0[edge_index[0]] + x0[edge_index[1]]
        else:
            e0 = data.edge_attr
        # x,e
        x = self.v_embed_init(x0)
        e = self.e_embed_init(e0)

        return x, e

    def forward(self, data: Data):
        x, e = self.init_mpparam(data)
        batch_size = data._num_graphs
        batch = data.batch
        edge_index = data.edge_index

        xs = [x, e]
        jump_xs = [[] for _ in range(self.dims)]
        refs = [x, e]

        for c, conv in enumerate(self.convs):
            xs = conv(xs[0], xs[1], edge_index)
            for i in range(len(xs)):
                xs[i] = self.layer_drop(xs[i])

            for i in range(len(xs)):
                jump_xs[i].append(xs[i])

        xs = self.jump_complex(jump_xs)

        # readout
        batches = [batch, batch[edge_index[0]]]
        for i in range(len(xs)):
            xs[i] = self.ro_weights[i](xs[i], refs[i], batches[i], batch_size)  # [B, F]

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


class MPNNConv(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 layer_dim: int = 32,
                 hidden: int = 32,
                 act_module=torch.nn.ReLU,
                 graph_norm=BatchNorm1d,
                 eps: float = 0.,
                 args=None):
        super(MPNNConv, self).__init__()

        self.initial_eps = eps
        if args.train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))

        self.n_msg_nn = Sequential(Catter(), Linear(layer_dim * 2, layer_dim), act_module())
        self.n_update_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                      graph_norm(hidden), act_module())
        self.e_msg_nn = lambda x: x
        self.e_update_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                      graph_norm(hidden), act_module())

        self.mp_weight = create_mp_weight(layer_dim, args, depth)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.n_msg_nn)
        reset(self.n_update_nn)
        reset(self.e_msg_nn)
        reset(self.e_update_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)
        reset(self.mp_weight)

    def forward(self, x, e, edge_index):
        i, j = (1, 0)
        x_i = x[edge_index[i]]
        x_j = x[edge_index[j]]

        n_msg = self.n_msg_nn((x_j, e))
        n_agg = self.mp_weight(n_msg, x_i, x_j, e, edge_index[i], num_nodes=x.size(0))
        n_out = self.n_update_nn(n_agg + (1 + self.eps1) * x)

        e_msg = self.e_msg_nn(x)
        e_agg = e_msg[edge_index[i]] + e_msg[edge_index[j]]
        e_out = self.e_update_nn(e_agg + (1 + self.eps2) * e)

        return [n_out, e_out]


class PreEmbed(torch.nn.Module):
    def __init__(self):
        super(PreEmbed, self).__init__()

    def forward(self, x):
        # The embedding layer expects integers so we convert the tensor to int.
        return x.squeeze(-1).to(dtype=torch.long)


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, xs):
        return torch.cat(xs, dim=-1)
