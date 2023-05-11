import os
import time
import argparse

from root_dir import ROOT_DIR


def get_parser():
    parser = argparse.ArgumentParser(description='TGAA Experiment.')
    parser.add_argument('--seed', type=int, default=43, help='random seed to set')
    parser.add_argument('--start_seed', type=int, default=0, help='The initial seed when evaluating on multiple seeds.')
    parser.add_argument('--stop_seed', type=int, default=9, help='The final seed when evaluating on multiple seeds.')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--model', type=str, default='gcnc', help='model')
    parser.add_argument('--use_coboundaries',
                        type=str,
                        default='False',
                        help='whether to use coboundary features for up-messages')
    parser.add_argument('--indrop_rate', type=float, default=0.0, help='inputs dropout rate for molec models')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='activation function')
    parser.add_argument('--readout', type=str, default='sum', help='readout function')
    parser.add_argument('--final_readout', type=str, default='sum', help='final readout function')
    parser.add_argument('--readout_dims',
                        type=int,
                        nargs='+',
                        default=(0, 1, 2),
                        help='dims at which to apply the final readout')
    parser.add_argument('--jump_mode', type=str, default=None, help='Mode for JK')
    parser.add_argument('--graph_norm',
                        type=str,
                        default='bn',
                        choices=['bn', 'ln', 'id'],
                        help='Normalization layer to use inside the model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='learning rate decay scheduler')
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50, help='number of epochs between lr decay')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5, help='strength of lr decay')
    parser.add_argument('--lr_scheduler_patience', type=float, default=10, help='patience for `ReduceLROnPlateau` lr decay')
    parser.add_argument('--lr_scheduler_min', type=float, default=0.00001, help='min LR for `ReduceLROnPlateau` lr decay')
    parser.add_argument('--num_layers', type=int, default=5, help='number of message passing layers')
    parser.add_argument('--emb_dim', type=int, default=64, help='dimensionality of hidden units in models')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--dataset', type=str, default="ZINC", help='dataset name')
    parser.add_argument('--task_type',
                        type=str,
                        default='classification',
                        help='task type, either (bin)classification, regression or isomorphism')
    parser.add_argument('--eval_metric', type=str, default='accuracy', help='evaluation metric')
    parser.add_argument('--iso_eps', type=int, default=0.01, help='Threshold to define (non-)isomorphism')
    parser.add_argument('--minimize', action='store_true', help='whether to minimize evaluation metric or not')
    parser.add_argument('--max_dim', type=int, default="2", help='maximum dimension')
    parser.add_argument('--max_ring_size', type=int, default=None, help='maximum ring size to look for')
    parser.add_argument('--result_folder', type=str, default=os.path.join(ROOT_DIR, 'results'), help='folder to output result')
    parser.add_argument('--exp_name', type=str, default=str(time.time()), help='name for specific experiment')
    parser.add_argument('--dump_curves', action='store_true', help='whether to dump the training curves to disk')
    parser.add_argument('--untrained', action='store_true', help='whether to skip training')
    parser.add_argument('--fold', type=int, default=None, help='fold index for k-fold cross-validation experiments')
    parser.add_argument('--folds', type=int, default=None, help='The number of folds to run on in cross validation experiments')
    parser.add_argument('--init_method',
                        type=str,
                        default='sum',
                        help='How to initialise features at higher levels (sum, mean)')
    parser.add_argument('--train_eval_period', type=int, default=10, help='How often to evaluate on train.')
    parser.add_argument('--tune', action='store_true', help='Use the tuning indexes')
    parser.add_argument('--use_edge_features', action='store_true', help="Use edge features for molecular graphs")
    parser.add_argument('--simple_features',
                        action='store_true',
                        help="Whether to use only a subset of original features, specific to ogb-mol*")
    parser.add_argument('--early_stop', action='store_true', help='Stop when minimum LR is reached.')
    parser.add_argument('--paraid', type=int, default=0, help='model id')
    parser.add_argument('--preproc_jobs',
                        type=int,
                        default=2,
                        help='Jobs to use for the dataset preprocessing. For all jobs use "-1".'
                        'For sequential processing (no parallelism) use "1"')
    parser.add_argument('--train_eps', action='store_true', help='eps in GIN')
    parser.add_argument('--sb_cal', type=str, default='sum', help='mean, degree, sum')
    parser.add_argument('--mp_cal', type=str, default='none', help='diff, mul, mlp, mlpv2, none')
    parser.add_argument('--mp_act', type=str, default='tanh', help='tanh, softmax, id')
    parser.add_argument('--mp_drop', type=float, default=0.0, help='mp_drop')
    parser.add_argument('--mp_channel', action='store_true', help='use channel-wise weight')
    parser.add_argument('--agg_kq', type=int, default=0, help='0: k=v, q=x_i; 1: k=x_j, q=x_i; 2: k=x_j||e_ij, q=x_i||x_i')
    parser.add_argument('--mlpv2_hidden', type=int, default=1, help='hidden layer')
    parser.add_argument('--pool_cal', type=str, default='none', help='diff, mul, mlp, none')
    parser.add_argument('--pool_act', type=str, default='tanh', help='tanh, softmax, id')
    parser.add_argument('--pool_drop', type=float, default=0.0, help='pool_drop')
    parser.add_argument('--pool_channel', action='store_true', help='use channel-wise pooling')
    parser.add_argument('--layer_drop', type=float, default=0.0, help='layer_drop')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--exp_mode', type=str, default='train', help='train, eval, resume')
    parser.add_argument('--load_model', type=str, default='last', help='last, best')
    parser.add_argument('--recover_best', action='store_true', help='load state_dict from best epoch when reduce learning rate')
    return parser
