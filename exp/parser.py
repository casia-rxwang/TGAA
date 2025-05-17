import time
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='CWN experiment.')
    parser.add_argument('--data_root', type=str, default='../datasets', help='path for data')
    parser.add_argument('--seed', type=int, default=43, help='random seed to set (default: 43, i.e. the non-meaning of life))')
    parser.add_argument('--start_seed', type=int, default=0, help='The initial seed when evaluating on multiple seeds.')
    parser.add_argument('--stop_seed', type=int, default=9, help='The final seed when evaluating on multiple seeds.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='tgaa', help='model, possible choices: mpnn, tgaa, ... (default: tgaa)')
    parser.add_argument('--use_coboundaries',
                        type=str,
                        default='False',
                        help='whether to use coboundary features for up-messages in sparse_cin (default: False)')
    # ^^^ here we explicitly pass it as string as easier to handle in tuning
    parser.add_argument('--indrop_rate', type=float, default=0.0, help='inputs dropout rate for molec models(default: 0.0)')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='dropout rate (default: 0.5)')
    parser.add_argument('--drop_position',
                        type=str,
                        default='lin2',
                        help='where to apply the final dropout (default: lin2, i.e. _before_ lin2)')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='activation function (default: relu)')
    parser.add_argument('--readout', type=str, default='sum', help='readout function (default: sum)')
    parser.add_argument('--final_readout', type=str, default='sum', help='final readout function (default: sum)')
    parser.add_argument('--readout_dims',
                        type=int,
                        nargs='+',
                        default=(0, 1, 2),
                        help='dims at which to apply the final readout (default: 0 1 2, i.e. nodes, edges, 2-cells)')
    parser.add_argument('--jump_mode', type=str, default=None, help='Mode for JK (default: None, i.e. no JK)')
    parser.add_argument('--graph_norm',
                        type=str,
                        default='bn',
                        choices=['bn', 'ln', 'id'],
                        help='Normalization layer to use inside the model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='learning rate decay scheduler (default: StepLR)')
    parser.add_argument('--lr_scheduler_decay_steps',
                        type=int,
                        default=50,
                        help='number of epochs between lr decay (default: 50)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5, help='strength of lr decay (default: 0.5)')
    parser.add_argument('--lr_scheduler_patience',
                        type=float,
                        default=10,
                        help='patience for `ReduceLROnPlateau` lr decay (default: 10)')
    parser.add_argument('--lr_scheduler_min',
                        type=float,
                        default=0.00001,
                        help='min LR for `ReduceLROnPlateau` lr decay (default: 1e-5)')
    parser.add_argument('--num_layers', type=int, default=5, help='number of message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=64, help='dimensionality of hidden units in models (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="PROTEINS", help='dataset name (default: PROTEINS)')
    parser.add_argument('--task_type',
                        type=str,
                        default='classification',
                        help='task type, either (bin)classification, regression or isomorphism (default: classification)')
    parser.add_argument('--eval_metric', type=str, default='accuracy', help='evaluation metric (default: accuracy)')
    parser.add_argument('--iso_eps', type=int, default=0.01, help='Threshold to define (non-)isomorphism')
    parser.add_argument('--minimize', action='store_true', help='whether to minimize evaluation metric or not')
    parser.add_argument('--max_dim', type=int, default="2", help='maximum cellular dimension (default: 2, i.e. two_cells)')
    parser.add_argument('--max_ring_size',
                        type=int,
                        default=None,
                        help='maximum ring size to look for (default: None, i.e. do not look for rings)')
    parser.add_argument('--result_folder', type=str, default='results', help='path for output result')
    parser.add_argument('--exp_name', type=str, default=str(time.time()),
                        help='name for specific experiment; if not provided, a name based on unix timestamp will be used. (default: None)')
    parser.add_argument('--dump_curves', action='store_true', help='whether to dump the training curves to disk')
    parser.add_argument('--untrained', action='store_true', help='whether to skip training')
    parser.add_argument('--fold', type=int, default=None, help='fold index for k-fold cross-validation experiments')
    parser.add_argument('--folds', type=int, default=None, help='The number of folds to run on in cross validation experiments')
    parser.add_argument('--init_method',
                        type=str,
                        default='sum',
                        help='How to initialise features at higher levels (sum, mean)')
    parser.add_argument('--train_eval_period', type=int, default=10, help='How often to evaluate on train.')
    parser.add_argument('--use_edge_features', action='store_true', help="Use edge features for molecular graphs")
    parser.add_argument('--simple_features',
                        action='store_true',
                        help="Whether to use only a subset of original features, specific to ogb-mol*")
    parser.add_argument('--early_stop', action='store_true', help='Stop when minimum LR is reached.')
    parser.add_argument('--preproc_jobs',
                        type=int,
                        default=2,
                        help='Jobs to use for the dataset preprocessing. For all jobs use "-1".'
                        'For sequential processing (no parallelism) use "1"')
    # exp
    parser.add_argument('--data_type', type=str, default='graph', help='graph, complex')
    parser.add_argument('--exp_mode', type=str, default='train', help='train, eval, resume')
    parser.add_argument('--load_model', type=str, default='last', help='last, best')
    parser.add_argument('--recover_best', action='store_true', help='load state_dict from best epoch when reduce learning rate')
    # input cluster and readout cluster
    parser.add_argument('--cluster_nums', type=int, nargs='*', default=(8, ), help='num of clusters at each level')
    parser.add_argument('--cluster_method', type=str, default='random', help='random, none')
    # mp weight and ro weight
    parser.add_argument('--ro_agg', type=str, default='sum', help='ro_agg')
    parser.add_argument('--ro_clusters', type=int, nargs='+', default=(4, 4, 2), help='ro_clusters')
    parser.add_argument('--mp_agg', type=str, default='sum', help='mp_agg')
    parser.add_argument('--mp_clusters', type=int, default=2, help='mp_clusters')
    parser.add_argument('--mp_agg_depth', type=int, default=0, help='the depth (1-N) to start mp_agg')
    parser.add_argument('--agg_qk', type=int, default=0, help='0: q=x_j; 1: q=e_ij; 2: q=x_j||e_ij')
    parser.add_argument('--gatev2_hidden', type=int, default=8, help='hidden dimension of gatev2')
    # others
    parser.add_argument('--layer_drop', type=float, default=0.0, help='layer_drop')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--train_eps', action='store_true', help='eps in GIN')
    parser.add_argument('--tau', type=float, default=10.0, help='temperature')
    parser.add_argument('--gamma', type=float, default=1.0, help='degree of freedom')
    return parser


def validate_args(args):
    """Performs dataset-dependent sanity checks on the supplied args."""
    if args.dataset == 'CSL':
        assert args.model == 'embed_sparse_cin'
        assert args.task_type == 'classification'
        assert not args.minimize
        assert args.lr_scheduler == 'ReduceLROnPlateau'
        assert args.eval_metric == 'accuracy'
        assert args.fold is not None
        assert not args.simple_features
        assert args.graph_norm == 'ln'
    elif args.dataset.startswith('ZINC'):
        assert args.model.startswith('embed')
        assert args.task_type == 'regression'
        assert args.minimize
        assert args.eval_metric == 'mae'
        assert args.lr_scheduler == 'ReduceLROnPlateau'
        assert not args.simple_features
    elif args.dataset in [
            'MOLHIV', 'MOLPCBA', 'MOLTOX21', 'MOLTOXCAST', 'MOLMUV', 'MOLBACE', 'MOLBBBP', 'MOLCLINTOX', 'MOLSIDER', 'MOLESOL',
            'MOLFREESOLV', 'MOLLIPO', 'PPA', 'CODE2'
    ]:
        assert args.model == 'ogb_embed_sparse_cin'
        assert args.eval_metric == 'ogbg-' + args.dataset.lower()
        assert args.jump_mode is None
        if args.dataset in ['MOLESOL', 'MOLFREESOLV', 'MOLLIPO']:
            assert args.task_type == 'mse_regression'
            assert args.minimize
        else:
            assert args.task_type == 'bin_classification'
            assert not args.minimize
    elif args.dataset.startswith('sr'):
        assert args.model in ['sparse_cin', 'mp_agnostic']
        assert args.eval_metric == 'isomorphism'
        assert args.task_type == 'isomorphism'
        assert args.jump_mode is None
        assert args.drop_rate == 0.0
        assert args.untrained
        assert args.nonlinearity == 'elu'
        assert args.readout == 'sum'
        assert args.final_readout == 'sum'
        assert not args.simple_features
