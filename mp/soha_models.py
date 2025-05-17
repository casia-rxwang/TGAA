import torch
from torch.nn import Linear, Embedding, Dropout, Identity, Sequential

from torch_geometric.nn.inits import reset
from torch_scatter import scatter
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from mp.nn import get_nonlinearity, get_graph_norm
from mp.graph_utils import get_dense_x
from mp.soha_layers import get_cluster, MPParam, AHGConv


class SOHA(torch.nn.Module):
    def __init__(self,
                 atom_types,
                 bond_types,
                 out_size,
                 num_layers,
                 hidden,
                 emb_method,
                 embed_dim=None,
                 jump_mode=None,
                 cluster_nums=(4, ),
                 cluster_method='random',
                 final_hidden_multiplier: int = 2,
                 nonlinearity='relu',
                 graph_norm='bn',
                 readout='sum',
                 args=None):
        super(SOHA, self).__init__()

        self.dims = len(cluster_nums) + 2
        self.readout = readout
        self.g_pool = torch.sum if readout == 'sum' else torch.mean

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

        # cluster layers
        self.clusters = torch.nn.ModuleList()
        for k in cluster_nums:
            self.clusters.append(get_cluster(embed_dim, k, cluster_method))
        self.g_cluster = get_cluster()  # MasterNode

        # conv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                AHGConv(cluster_dims=len(cluster_nums),
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

        # output layer
        self.lin1s = torch.nn.ModuleList()

        for i in range(self.dims):
            if jump_mode == 'cat':
                self.lin1s.append(Linear(num_layers * hidden, final_hidden_multiplier * hidden, bias=False))
            else:
                self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))

        self.lin2 = Linear(final_hidden_multiplier * hidden * (len(cluster_nums) + 2), out_size)

        # nonlinearity
        self.act = get_nonlinearity(nonlinearity, return_module=False)
        # dropout
        self.layer_drop = Dropout(args.layer_drop) if args.layer_drop > 0 else lambda x: x
        self.final_drop = Dropout(args.drop_rate) if args.drop_rate > 0 else lambda x: x

    def reset_parameters(self):
        reset(self.v_embed_init)
        reset(self.e_embed_init)
        reset(self.clusters)
        reset(self.g_cluster)
        reset(self.convs)
        reset(self.lin1s)
        reset(self.lin2)

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def init_mpparam(self, p: MPParam):
        # x,e
        p.x = self.v_embed_init(p.x0)
        p.e = self.e_embed_init(p.e0)
        c_loss = p.x.new_zeros(1)

        dense_x = get_dense_x(x=p.x, d_x_mask=p.mask, d_x_idx=p.x_idx)
        p.attr_list.append(dense_x)  # used later
        p.adj_list.append(p.adj)
        p.nc_list.append(None)
        p.cn_list.append(None)

        # sub-structure
        mask = p.mask
        for i, cluster in enumerate(self.clusters):
            c_x, c_adj, m_nc, m_cn, loss = cluster.forward(p.attr_list[i], p.adj_list[i], mask)
            p.attr_list.append(c_x)  # b,n,f
            p.adj_list.append(c_adj)  # b,n,n
            p.nc_list.append(m_nc)  # b,n,c
            p.cn_list.append(m_cn)  # b,c,n

            c_loss = c_loss + loss
            mask = c_x.new_ones(c_x.size()[:2])

        # graph
        g_x, g_adj, g_nc, g_cn, _ = self.g_cluster.forward(p.attr_list[-1], p.adj_list[-1], mask)
        p.attr_list.append(g_x)  # b,1,f
        p.adj_list.append(g_adj)  # b,1,1
        p.nc_list.append(g_nc)  # b,n,1
        p.cn_list.append(g_cn)  # b,1,n

        return c_loss

    def forward(self, data: Data):
        # attr_list: [x, sub]
        param = MPParam(data)
        c_loss = self.init_mpparam(param)

        jump_xs = [[] for _ in range(len(param.attr_list))]  # lens = len([x, e] + attr_list[1:-1])
        refs = [param.x, param.e] + param.attr_list[1:-1]  # not include dense x and graph-level attr

        for c, conv in enumerate(self.convs):
            xs = conv(param)  # xs: [x, e, sub]
            for i in range(len(xs)):
                xs[i] = self.layer_drop(xs[i])
            param.update_attr(x=xs[0], e=xs[1], sub_attrs=xs[2:])

            for i in range(len(xs)):
                jump_xs[i].append(xs[i])

        xs = self.jump_complex(jump_xs)

        # global pooling x: [N, F] to [B, F]
        xs[0] = scatter(xs[0], param.batch, dim=0, dim_size=param.batch_size, reduce=self.readout)
        xs[1] = scatter(xs[1], param.batch[param.edge_index[0]], dim=0, dim_size=param.batch_size, reduce=self.readout)
        # global pooling x: [B, K, F] to [B, F]
        for i in range(2, len(xs)):
            xs[i] = self.g_pool(xs[i], dim=1)

        for i in range(len(xs)):
            xs[i] = self.act(self.lin1s[i](xs[i]))
        out = self.final_drop(torch.cat(xs, dim=-1))

        return self.lin2(out)

    def __repr__(self):
        return self.__class__.__name__


class PreEmbed(torch.nn.Module):
    def __init__(self):
        super(PreEmbed, self).__init__()

    def forward(self, x):
        # The embedding layer expects integers so we convert the tensor to int.
        return x.squeeze(1).to(dtype=torch.long)
