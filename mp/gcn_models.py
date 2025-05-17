import torch
from torch.nn import Linear, Embedding, Dropout, Identity, Sequential, BatchNorm1d

from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder

from mp.nn import get_nonlinearity, get_graph_norm
from mp.weight_mp import create_mp_weight
from mp.weight_ro import create_ro_weight


class GCN(torch.nn.Module):
    def __init__(self,
                 atom_types,
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
        super(GCN, self).__init__()

        self.dims = 1  # node

        if embed_dim is None:
            embed_dim = hidden

        # embed layers
        if emb_method == 'embed':
            self.v_embed_init = Sequential(PreEmbed(), Embedding(atom_types, embed_dim))
        elif emb_method == 'ogb':
            self.v_embed_init = AtomEncoder(embed_dim)
        elif emb_method == 'none':
            self.v_embed_init = Identity()

        # conv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                GCNConv(depth=i + 1,
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

        if jump_mode == 'cat':
            self.ro_weight = create_ro_weight(embed_dim, num_layers * hidden, args, 0)
            self.lin1 = Linear(num_layers * hidden, final_hidden_multiplier * hidden)
        else:
            self.ro_weight = create_ro_weight(embed_dim, hidden, args, 0)
            self.lin1 = Linear(hidden, final_hidden_multiplier * hidden)

        self.lin2 = Linear(final_hidden_multiplier * hidden * self.dims, out_size)

        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.layer_drop = Dropout(args.layer_drop) if args.layer_drop > 0 else lambda x: x
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        reset(self.v_embed_init)
        reset(self.convs)
        reset(self.ro_weight)
        reset(self.lin1)
        reset(self.lin2)

    def forward(self, data: Data):
        x = self.v_embed_init(data.x)
        batch_size = data._num_graphs
        batch = data.batch
        edge_index = data.edge_index

        jump_xs = []
        ref = x

        for c, conv in enumerate(self.convs):
            x = self.layer_drop(conv(x, edge_index))
            jump_xs.append(x)

        x = self.jump(jump_xs)

        # readout
        x = self.ro_weight(x, ref, batch, batch_size)  # [B, F]
        out = self.final_drop(self.act(self.lin1(x)))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


class GCNConv(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 layer_dim: int = 32,
                 hidden: int = 32,
                 act_module=torch.nn.ReLU,
                 graph_norm=BatchNorm1d,
                 eps: float = 0.,
                 args=None):
        super(GCNConv, self).__init__()

        self.initial_eps = eps
        if args.train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.msg_nn = lambda x: x
        self.update_nn = Sequential(Linear(layer_dim, hidden), graph_norm(hidden), act_module(), Linear(hidden, hidden),
                                    graph_norm(hidden), act_module())

        self.mp_weight = create_mp_weight(layer_dim, args, depth)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.msg_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)
        reset(self.mp_weight)

    def forward(self, x, edge_index):
        i, j = (1, 0)
        x_i = x[edge_index[i]]
        x_j = x[edge_index[j]]

        msg = self.msg_nn(x_j)
        agg = self.mp_weight(msg, x_i, x_j, x_j, edge_index[i], num_nodes=x.size(0))
        out = self.update_nn(agg + (1 + self.eps) * x)

        return out


class PreEmbed(torch.nn.Module):
    def __init__(self):
        super(PreEmbed, self).__init__()

    def forward(self, x):
        # The embedding layer expects integers so we convert the tensor to int.
        return x.squeeze(-1).to(dtype=torch.long)
