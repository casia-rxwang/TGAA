import torch
import pickle

from data.data_loading import DataLoader, load_dataset, load_graph_dataset
from torch_geometric.data import DataLoader as PyGDataLoader
from mp.gin_models import GIN, GINWithJK
from mp.cin_models import CIN, SparseCIN, MessagePassingAgnostic, EmbedSparseCIN, OGBEmbedSparseCIN
from mp.tgaa_models import TGAA, TGAA_MLP
from mp.mpnn_models import MPNN
from mp.soha_models import SOHA
from mp.gcn_models import GCN

SR_families = ['sr16622', 'sr251256', 'sr261034', 'sr281264', 'sr291467', 'sr351668', 'sr351899', 'sr361446', 'sr401224']
TU_datasets = ['IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 'NCI1', 'NCI109', 'PTC', 'MUTAG']
none_embed_datasets = TU_datasets + SR_families

ogb_embed_datasets = [
    'MOLHIV', 'MOLPCBA', 'MOLTOX21', 'MOLTOXCAST', 'MOLMUV', 'MOLBACE', 'MOLBBBP', 'MOLCLINTOX', 'MOLSIDER', 'MOLESOL',
    'MOLFREESOLV', 'MOLLIPO', 'PPA', 'CODE2'
]

embed_datasets = ['ZINC', 'ZINC-FULL', 'CSL', 'EXPWL1']
embed_datasets_dim = {'ZINC': (28, 4), 'ZINC-FULL': (28, 4), 'CLS': (1, 1), 'EXPWL1': (2, 1)}  # node, edge


def create_dataset(args):
    dataset = None
    num_features = 1
    num_classes = 1

    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        # https://github.com/pytorch/pytorch/issues/973

    if args.data_type == 'graph':
        # load graph dataset
        graph_list, train_ids, val_ids, test_ids, num_classes = load_graph_dataset(args.dataset,
                                                                                   args.data_root,
                                                                                   fold=args.fold,
                                                                                   max_ring_size=args.max_ring_size)
        train_graphs = [graph_list[i] for i in train_ids]
        val_graphs = [graph_list[i] for i in val_ids]
        train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if test_ids is not None:
            test_graphs = [graph_list[i] for i in test_ids]
            test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            test_loader = None
        if args.dataset.startswith('sr'):
            num_features = 1  # otherwise be None
            num_classes = args.emb_dim
        else:
            num_features = graph_list[0].x.shape[1]
    else:
        # data loading
        dataset = load_dataset(args.dataset,
                               args.data_root,
                               max_dim=args.max_dim,
                               fold=args.fold,
                               init_method=args.init_method,
                               emb_dim=args.emb_dim,
                               max_ring_size=args.max_ring_size,
                               use_edge_features=args.use_edge_features,
                               simple_features=args.simple_features,
                               n_jobs=args.preproc_jobs)
        dataset.preload_to_complex()

        split_idx = dataset.get_idx_split()
        num_classes = dataset.num_classes

        # Instantiate data loaders
        train_loader = DataLoader(dataset.get_split('train'),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  max_dim=dataset.max_dim)
        valid_loader = DataLoader(dataset.get_split('valid'),
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  max_dim=dataset.max_dim)
        test_split = split_idx.get("test", None)
        test_loader = None
        if test_split is not None:
            test_loader = DataLoader(dataset.get_split('test'),
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     max_dim=dataset.max_dim)

    return train_loader, valid_loader, test_loader, dataset, num_features, num_classes


def create_model(args, device, dataset, num_features, num_classes):
    # Use coboundaries?
    use_coboundaries = args.use_coboundaries.lower() == 'true'

    # NB: here we assume to have the same number of features per dim
    if args.model == 'gin':
        model = GIN(
            num_features,  # num_input_features
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            num_classes,  # num_classes
            dropout_rate=args.drop_rate,  # dropout rate
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
        ).to(device)
    elif args.model == 'gin_jk':
        model = GINWithJK(
            num_features,  # num_input_features
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            num_classes,  # num_classes
            dropout_rate=args.drop_rate,  # dropout rate
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
        ).to(device)
    elif args.model == 'cin':
        model = CIN(
            dataset.num_features_in_dim(0),  # num_input_features
            dataset.num_classes,  # num_classes
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            dropout_rate=args.drop_rate,  # dropout rate
            max_dim=dataset.max_dim,  # max_dim
            jump_mode=args.jump_mode,  # jump mode
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
        ).to(device)
    elif args.model == 'sparse_cin':
        model = SparseCIN(
            dataset.num_features_in_dim(0),  # num_input_features
            dataset.num_classes,  # num_classes
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            dropout_rate=args.drop_rate,  # dropout rate
            max_dim=dataset.max_dim,  # max_dim
            jump_mode=args.jump_mode,  # jump mode
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
            final_readout=args.final_readout,  # final readout
            apply_dropout_before=args.drop_position,  # where to apply dropout
            use_coboundaries=use_coboundaries,  # whether to use coboundaries in up-msg
            graph_norm=args.graph_norm  # normalization layer
        ).to(device)
    elif args.model == 'mp_agnostic':
        model = MessagePassingAgnostic(
            dataset.num_features_in_dim(0),  # num_input_features
            dataset.num_classes,  # num_classes
            args.emb_dim,  # hidden
            dropout_rate=args.drop_rate,  # dropout rate
            max_dim=dataset.max_dim,  # max_dim
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
        ).to(device)
    elif args.model == 'embed_sparse_cin':
        model = EmbedSparseCIN(
            dataset.num_node_type,  # The number of atomic types
            dataset.num_edge_type,  # The number of bond types
            dataset.num_classes,  # num_classes
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            dropout_rate=args.drop_rate,  # dropout rate
            max_dim=dataset.max_dim,  # max_dim
            jump_mode=args.jump_mode,  # jump mode
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
            final_readout=args.final_readout,  # final readout
            apply_dropout_before=args.drop_position,  # where to apply dropout
            use_coboundaries=use_coboundaries,
            embed_edge=args.use_edge_features,
            graph_norm=args.graph_norm  # normalization layer
        ).to(device)
    # TODO: handle this as above
    elif args.model == 'ogb_embed_sparse_cin':
        model = OGBEmbedSparseCIN(
            dataset.num_tasks,  # out_size
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            dropout_rate=args.drop_rate,  # dropout_rate
            indropout_rate=args.indrop_rate,  # in-dropout_rate
            max_dim=dataset.max_dim,  # max_dim
            jump_mode=args.jump_mode,  # jump_mode
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
            final_readout=args.final_readout,  # final readout
            apply_dropout_before=args.drop_position,  # where to apply dropout
            use_coboundaries=use_coboundaries,  # whether to use coboundaries
            embed_edge=args.use_edge_features,  # whether to use edge feats
            graph_norm=args.graph_norm  # normalization layer
        ).to(device)
    elif args.model == 'tgaa':
        if args.dataset in none_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            embed_dim = dataset.num_features_in_dim(0)
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            embed_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            num_node_type = dataset.num_node_type
            num_edge_type = dataset.num_edge_type
            embed_dim = args.emb_dim
            emb_method = 'embed'
        model = TGAA(
            atom_types=num_node_type,
            bond_types=num_edge_type,
            out_size=num_classes,
            num_layers=args.num_layers,
            hidden=args.emb_dim,
            emb_method=emb_method,
            embed_dim=embed_dim,
            jump_mode=args.jump_mode,
            use_coboundaries=use_coboundaries,
            nonlinearity=args.nonlinearity,
            graph_norm=args.graph_norm,
            args=args
        ).to(device)
    elif args.model == 'tgaa_mlp':
        if args.dataset in none_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            embed_dim = dataset.num_features_in_dim(0)
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            embed_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            num_node_type = dataset.num_node_type
            num_edge_type = dataset.num_edge_type
            embed_dim = args.emb_dim
            emb_method = 'embed'
        model = TGAA_MLP(
            atom_types=num_node_type,
            bond_types=num_edge_type,
            out_size=num_classes,
            emb_method=emb_method,
            embed_dim=embed_dim,
            nonlinearity=args.nonlinearity,
            args=args
        ).to(device)
    elif args.model == 'mpnn':
        num_embed = embed_datasets_dim.get(args.dataset, (-1, -1))
        if args.dataset in none_embed_datasets:
            emb_dim = num_features
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            emb_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            emb_dim = args.emb_dim
            emb_method = 'embed'
        model = MPNN(
            atom_types=num_embed[0],
            bond_types=num_embed[1],
            out_size=num_classes,
            num_layers=args.num_layers,
            hidden=args.emb_dim,
            emb_method=emb_method,
            embed_dim=emb_dim,
            jump_mode=args.jump_mode,
            nonlinearity=args.nonlinearity,
            graph_norm=args.graph_norm,
            args=args
        ).to(device)
    elif args.model == 'gcn':
        num_embed = embed_datasets_dim.get(args.dataset, (-1, -1))
        if args.dataset in none_embed_datasets:
            emb_dim = num_features
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            emb_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            emb_dim = args.emb_dim
            emb_method = 'embed'
        model = GCN(
            atom_types=num_embed[0],
            out_size=num_classes,
            num_layers=args.num_layers,
            hidden=args.emb_dim,
            emb_method=emb_method,
            embed_dim=emb_dim,
            jump_mode=args.jump_mode,
            nonlinearity=args.nonlinearity,
            graph_norm=args.graph_norm,
            args=args
        ).to(device)
    elif args.model == 'soha':
        num_embed = embed_datasets_dim.get(args.dataset, (-1, -1))
        if args.dataset in none_embed_datasets:
            emb_dim = num_features
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            emb_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            emb_dim = args.emb_dim
            emb_method = 'embed'
        model = SOHA(
            atom_types=num_embed[0],
            bond_types=num_embed[1],
            out_size=num_classes,
            num_layers=args.num_layers,
            hidden=args.emb_dim,
            emb_method=emb_method,
            embed_dim=emb_dim,
            jump_mode=args.jump_mode,
            cluster_nums=args.cluster_nums,
            cluster_method=args.cluster_method,
            nonlinearity=args.nonlinearity,
            graph_norm=args.graph_norm,
            args=args
        ).to(device)
    else:
        raise ValueError('Invalid model type {}.'.format(args.model))

    return model


def create_scheduler(args, optimizer):
    if args.lr_scheduler == 'ReduceLROnPlateau':
        mode = 'min' if args.minimize else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=mode,
                                                               factor=args.lr_scheduler_decay_rate,
                                                               patience=args.lr_scheduler_patience,
                                                               verbose=True)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return scheduler


def save_model(path, model, epoch, optimizer=None, scheduler=None):

    state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
    torch.save(data, path)


def load_model(model_path, model, optimizer=None, scheduler=None, resume=False):

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    state_dict = checkpoint['state_dict']
    start_epoch = checkpoint['epoch']

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    for k in missing_keys:
        print('missing parameter {}.'.format(k))
    for k in unexpected_keys:
        print('unexpected parameter {}.'.format(k))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Resumed optimizer with start epoch', start_epoch)
        else:
            print('No optimizer parameters in checkpoint.')
    if scheduler is not None and resume:
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('Resumed scheduler with start epoch', start_epoch)
        else:
            print('No scheduler parameters in checkpoint.')

    return model, start_epoch, optimizer, scheduler


def save_curve(curve_path, epoch, train_curve, valid_curve, test_curve, train_loss_curve):
    curves = {'epoch': epoch, 'train': train_curve, 'val': valid_curve, 'test': test_curve, 'train_loss': train_loss_curve}
    with open(curve_path, 'wb') as handle:
        pickle.dump(curves, handle)
    return


def load_curve(curve_path, model_epoch):
    curves = {}
    with open(curve_path, 'rb') as handle:
        curves = pickle.load(handle)

    curve_epoch = curves['epoch']
    assert curve_epoch == model_epoch, "\n!! Curves file corrupted, delete it and restart."

    train_curve = curves['train']
    valid_curve = curves['val']
    test_curve = curves['test']
    train_loss_curve = curves['train_loss']

    return train_curve, valid_curve, test_curve, train_loss_curve
