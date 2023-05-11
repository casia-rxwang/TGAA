import torch
import pickle

from data.data_loading import DataLoader, load_dataset, load_graph_dataset
from torch_geometric.data import DataLoader as PyGDataLoader

from model.models import GCNC, GCNE, GCNC_MLP

SR_families = ['sr16622', 'sr251256', 'sr261034', 'sr281264', 'sr291467', 'sr351668', 'sr351899', 'sr361446', 'sr401224']
TU_datasets = ['IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 'NCI1', 'NCI109', 'PTC', 'MUTAG']
none_embed_datasets = TU_datasets + SR_families
ogb_embed_datasets = [
    'MOLHIV', 'MOLPCBA', 'MOLTOX21', 'MOLTOXCAST', 'MOLMUV', 'MOLBACE', 'MOLBBBP', 'MOLCLINTOX', 'MOLSIDER', 'MOLESOL',
    'MOLFREESOLV', 'MOLLIPO', 'PPA', 'CODE2'
]
embed_datasets = ['ZINC', 'ZINC-FULL', 'CSL']


def create_dataset(args):
    dataset = None
    num_features = 1
    num_classes = 1

    if args.num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        # https://github.com/pytorch/pytorch/issues/973

    if args.model.startswith('gin'):
        # load graph dataset
        graph_list, train_ids, val_ids, test_ids, num_classes = load_graph_dataset(args.dataset,
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
            num_features = 1
            num_classes = args.emb_dim
        else:
            num_features = graph_list[0].x.shape[1]
    else:
        # data loading
        dataset = load_dataset(args.dataset,
                               max_dim=args.max_dim,
                               fold=args.fold,
                               init_method=args.init_method,
                               emb_dim=args.emb_dim,
                               max_ring_size=args.max_ring_size,
                               use_edge_features=args.use_edge_features,
                               simple_features=args.simple_features,
                               n_jobs=args.preproc_jobs)
        if args.tune:
            split_idx = dataset.get_tune_idx_split()
        else:
            split_idx = dataset.get_idx_split()

        dataset.preload_to_sstree()

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
    # Use coboundaries
    use_coboundaries = args.use_coboundaries.lower() == 'true'

    # Readout dimensions
    readout_dims = tuple(sorted(args.readout_dims))

    # here we assume to have the same number of features per dim
    if args.model == 'gcnc':
        if args.dataset in none_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            num_classes = dataset.num_classes
            emb_dim = dataset.num_features_in_dim(0)
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            num_classes = dataset.num_tasks
            emb_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            num_node_type = dataset.num_node_type
            num_edge_type = dataset.num_edge_type
            num_classes = dataset.num_classes
            emb_dim = args.emb_dim
            emb_method = 'embed'
        model = GCNC(
            num_node_type,  # The number of atomic types
            num_edge_type,  # The number of bond types
            num_classes,  # num_classes
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            emb_method,
            embed_dim=emb_dim,  # embed_dim
            dropout_rate=args.drop_rate,  # dropout rate
            jump_mode=args.jump_mode,  # jump mode
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
            final_readout=args.final_readout,  # final readout
            use_coboundaries=use_coboundaries,
            embed_edge=args.use_edge_features,
            graph_norm=args.graph_norm,  # normalization layer
            readout_dims=readout_dims,  # readout_dims
            args=args  # other args
        ).to(device)
    elif args.model == 'gcne':
        if args.dataset in none_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            num_classes = dataset.num_classes
            emb_dim = dataset.num_features_in_dim(0)
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            num_classes = dataset.num_tasks
            emb_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            num_node_type = dataset.num_node_type
            num_edge_type = dataset.num_edge_type
            num_classes = dataset.num_classes
            emb_dim = args.emb_dim
            emb_method = 'embed'
        model = GCNE(
            num_node_type,  # The number of atomic types
            num_edge_type,  # The number of bond types
            num_classes,  # num_classes
            args.num_layers,  # num_layers
            args.emb_dim,  # hidden
            emb_method,
            embed_dim=emb_dim,  # embed_dim
            dropout_rate=args.drop_rate,  # dropout rate
            jump_mode=args.jump_mode,  # jump mode
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
            final_readout=args.final_readout,  # final readout
            use_coboundaries=use_coboundaries,
            embed_edge=args.use_edge_features,
            graph_norm=args.graph_norm,  # normalization layer
            args=args  # other args
        ).to(device)
    elif args.model == 'gcnc_mlp':
        if args.dataset in none_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            num_classes = dataset.num_classes
            emb_dim = dataset.num_features_in_dim(0)
            emb_method = 'none'
        elif args.dataset in ogb_embed_datasets:
            num_node_type = 0
            num_edge_type = 0
            num_classes = dataset.num_tasks
            emb_dim = args.emb_dim
            emb_method = 'ogb'
        elif args.dataset in embed_datasets:
            num_node_type = dataset.num_node_type
            num_edge_type = dataset.num_edge_type
            num_classes = dataset.num_classes
            emb_dim = args.emb_dim
            emb_method = 'embed'
        model = GCNC_MLP(
            num_node_type,  # The number of atomic types
            num_edge_type,  # The number of bond types
            num_classes,  # num_classes
            emb_method,
            embed_dim=emb_dim,  # embed_dim
            dropout_rate=args.drop_rate,  # dropout rate
            nonlinearity=args.nonlinearity,  # nonlinearity
            readout=args.readout,  # readout
            final_readout=args.final_readout,  # final readout
            embed_edge=args.use_edge_features,
            args=args  # other args
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


def load_model(model_path, model, optimizer=None, scheduler=None, resume=False):

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    # # convert data_parallal to model
    # state_dict_ = checkpoint['state_dict']
    # state_dict = {}
    # for k in state_dict_:
    #     if k.startswith('module') and not k.startswith('module_list'):
    #         state_dict[k[7:]] = state_dict_[k]
    #     else:
    #         state_dict[k] = state_dict_[k]

    state_dict = checkpoint['state_dict']
    start_epoch = checkpoint['epoch']

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
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


def save_model(path, model, epoch, optimizer=None, scheduler=None):
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     state_dict = model.module.state_dict()
    # else:
    #     state_dict = model.state_dict()
    state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
    torch.save(data, path)


def save_curve(curve_path, epoch, train_curve, valid_curve, test_curve):
    curves = {'epoch': epoch, 'train': train_curve, 'val': valid_curve, 'test': test_curve}
    with open(curve_path, 'wb') as handle:
        pickle.dump(curves, handle)
    return


def load_curve(curve_path, epoch):
    curves = {}
    with open(curve_path, 'rb') as handle:
        curves = pickle.load(handle)

    curve_epoch = curves['epoch']
    train_curve = curves['train']
    valid_curve = curves['val']
    test_curve = curves['test']

    assert curve_epoch >= epoch, "\n!! Curves file corrupted, delete it and restart."

    train_curve = train_curve[:epoch]
    valid_curve = valid_curve[:epoch]
    test_curve = test_curve[:epoch]

    return train_curve, valid_curve, test_curve
