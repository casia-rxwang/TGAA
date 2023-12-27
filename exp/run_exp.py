import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import random

from exp.train_utils import train, eval, Evaluator
from exp.parser import get_parser, validate_args
from exp.run_exp_utils import create_dataset, create_model, create_scheduler, load_model, save_model, load_curve, save_curve

import warnings

warnings.filterwarnings('ignore')


def main(args):
    """The common training and evaluation script used by all the experiments."""
    # set device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    print("==========================================================")
    print("Using device", str(device))
    print(f"Fold: {args.fold}")
    print(f"Seed: {args.seed}")
    print("======================== Args ===========================")
    print(args)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Set double precision for SR experiments
    if args.task_type == 'isomorphism':
        assert args.dataset.startswith('sr')
        torch.set_default_dtype(torch.float64)

    # Create results folder
    result_folder = os.path.join(args.result_folder, f'{args.dataset}-{args.exp_name}', f'seed-{args.seed}')
    if args.fold is not None:
        result_folder = os.path.join(result_folder, f'fold-{args.fold}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create dataset
    train_loader, valid_loader, test_loader, dataset, num_features, num_classes = create_dataset(args)

    # automatic evaluator, takes dataset name as input
    evaluator = Evaluator(args.eval_metric, eps=args.iso_eps)

    # instantiate model
    model = create_model(args, device, dataset, num_features, num_classes)

    # prepare for load model
    model_path = os.path.join(result_folder, 'model_' + args.load_model + '.pth')
    model_exist = os.path.exists(model_path)
    curve_path = os.path.join(result_folder, 'curve_' + args.load_model + '.pkl')
    curve_exist = os.path.exists(curve_path)

    # eval mode
    if args.exp_mode == 'eval':
        assert model_exist, "\n!! Pre-trained model does not exist."
        model, start_epoch, optimizer, scheduler = load_model(model_path, model, None, None, False)
        return eval_mode(args, model, device, train_loader, valid_loader, test_loader, evaluator)

    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # instantiate learning rate decay
    scheduler = create_scheduler(args, optimizer)

    # train curve
    best_valid_perf = np.finfo(float).max if args.minimize else np.finfo(float).min
    best_val_epoch = 0
    train_curve = []
    valid_curve = []
    test_curve = []
    train_loss_curve = []

    # resume mode
    start_epoch = 0
    if args.exp_mode == 'resume' and model_exist and curve_exist:
        model, start_epoch, optimizer, scheduler = load_model(model_path, model, optimizer, scheduler, True)
        train_curve, valid_curve, test_curve, train_loss_curve = load_curve(curve_path, start_epoch)

        best_val_index = np.argmin(np.array(valid_curve)) if args.minimize else np.argmax(np.array(valid_curve))
        best_valid_perf = valid_curve[best_val_index]
        best_val_epoch = best_val_index + 1
    else:
        print("\n!! Pre-trained model does not exist. Train from scratch.")

    print("============= Model Parameters =================")
    names = []
    params = []
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        names.append(name)
        params.append(param.data.detach().mean().item())
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("============= Params stats ==================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")

    # (!) start training/evaluation
    if not args.untrained:
        for epoch in range(start_epoch, args.epochs):
            # We use a strict inequality here like in the benchmarking GNNs paper code
            # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py#L217
            if args.early_stop and optimizer.param_groups[0]['lr'] < args.lr_scheduler_min:
                print("\n!! The minimum learning rate has been reached.")
                break

            # perform one epoch
            print("=====Epoch {}".format(epoch))
            print('Training...')
            epoch_train_curve = train(model, device, train_loader, optimizer, args.task_type)
            train_loss_curve += epoch_train_curve
            epoch_train_loss = float(np.mean(epoch_train_curve))

            # evaluate model
            print('Evaluating...')
            # NOTICE, eval() will load train data and change following random batch
            # if epoch % args.train_eval_period == 0:
            if epoch == start_epoch or (epoch + 1) % args.train_eval_period == 0:
                train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
            train_curve.append(train_perf)

            valid_perf, epoch_val_loss = eval(model, device, valid_loader, evaluator, args.task_type)
            valid_curve.append(valid_perf)

            if test_loader is not None:
                test_perf, epoch_test_loss = eval(model, device, test_loader, evaluator, args.task_type)
            else:
                test_perf = np.nan
                epoch_test_loss = np.nan
            test_curve.append(test_perf)

            print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
                  f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}'
                  f' | Test Loss {epoch_test_loss:.3f}')

            tmp_lr = optimizer.param_groups[0]['lr']
            # decay learning rate
            if scheduler is not None:
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_perf)
                else:
                    scheduler.step()
            # load state_dict from best epoch when reduce learning rate
            if tmp_lr > optimizer.param_groups[0]['lr'] and args.recover_best:
                cp = torch.load(os.path.join(result_folder, 'model_best.pth'), map_location=lambda storage, loc: storage)
                best_state_dict = cp['state_dict']
                model.load_state_dict(best_state_dict, strict=False)

            # save best model
            if (args.minimize and valid_perf < best_valid_perf) or ((not args.minimize) and valid_perf > best_valid_perf):
                best_val_epoch = epoch
                best_valid_perf = valid_perf
                save_model(os.path.join(result_folder, 'model_best.pth'), model, epoch, optimizer, scheduler)
                save_curve(os.path.join(result_folder, 'curve_best.pkl'), epoch, train_curve, valid_curve, test_curve,
                           train_loss_curve)

            new_params = []
            for name, param in model.named_parameters():
                # print(f"Param {name}: {param.data.view(-1)[0]}")
                # new_params.append(param.data.detach().clone().view(-1)[0])
                new_params.append(param.data.detach().mean().item())

            if epoch % args.train_eval_period == 0:
                # save last model
                save_model(os.path.join(result_folder, 'model_last.pth'), model, epoch, optimizer, scheduler)
                save_curve(os.path.join(result_folder, 'curve_last.pkl'), epoch, train_curve, valid_curve, test_curve,
                           train_loss_curve)

                print("====== Slowly changing params ======= ")
                for i in range(len(names)):
                    if abs(params[i] - new_params[i]) < 1e-6:
                        print(f"Param {names[i]}: {params[i] - new_params[i]}")

            params = new_params
    else:
        train_loss_curve.append(np.nan)
        train_curve.append(np.nan)
        valid_curve.append(np.nan)
        test_curve.append(np.nan)

    print('Final Evaluation...')
    final_train_perf = np.nan
    final_val_perf = np.nan
    final_test_perf = np.nan
    if not args.dataset.startswith('sr'):
        final_train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
        final_val_perf, _ = eval(model, device, valid_loader, evaluator, args.task_type)
    if test_loader is not None:
        final_test_perf, _ = eval(model, device, test_loader, evaluator, args.task_type)

    # save results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'last_train': final_train_perf,
        'last_val': final_val_perf,
        'last_test': final_test_perf,
        'best': best_val_epoch
    }

    msg = (f'========== Result ============\n'
           f'Dataset:        {args.dataset}\n'
           f'------------ Best epoch -----------\n'
           f'Train:          {train_curve[best_val_epoch]}\n'
           f'Validation:     {valid_curve[best_val_epoch]}\n'
           f'Test:           {test_curve[best_val_epoch]}\n'
           f'Best epoch:     {best_val_epoch}\n'
           '------------ Last epoch -----------\n'
           f'Train:          {final_train_perf}\n'
           f'Validation:     {final_val_perf}\n'
           f'Test:           {final_test_perf}\n'
           '-------------------------------\n\n')
    print(msg)

    msg += str(args)
    msg += '\n\n'
    for name, param in model.named_parameters():
        if param.requires_grad:
            msg += f'{name}: {param.size()}\n'
    msg += f'Trainable params: {trainable_params}\n'
    msg += f'Total params    : {total_params}\n'

    with open(os.path.join(result_folder, 'results.txt'), 'w') as handle:
        handle.write(msg)
    if args.dump_curves:
        with open(os.path.join(result_folder, 'curves.pkl'), 'wb') as handle:
            pickle.dump(curves, handle)

    return curves


def eval_mode(args, model, device, train_loader, valid_loader, test_loader, evaluator):

    print('Final Evaluation...')
    best_val_epoch = 0
    final_train_perf = np.nan
    final_val_perf = np.nan
    final_test_perf = np.nan
    if not args.dataset.startswith('sr'):
        final_train_perf, _ = eval(model, device, train_loader, evaluator, args.task_type)
        final_val_perf, _ = eval(model, device, valid_loader, evaluator, args.task_type)
    if test_loader is not None:
        final_test_perf, _ = eval(model, device, test_loader, evaluator, args.task_type)

    curves = {
        'train_loss': [np.nan],
        'train': [final_train_perf],
        'val': [final_val_perf],
        'test': [final_test_perf],
        'last_train': final_train_perf,
        'last_val': final_val_perf,
        'last_test': final_test_perf,
        'best': best_val_epoch
    }

    msg = (f'========== Result ============\n'
           f'Dataset:        {args.dataset}\n'
           f'------------ Best epoch -----------\n'
           f'Train:          {final_train_perf}\n'
           f'Validation:     {final_val_perf}\n'
           f'Test:           {final_test_perf}\n'
           f'Best epoch:     {best_val_epoch}\n'
           '------------ Last epoch -----------\n'
           f'Train:          {final_train_perf}\n'
           f'Validation:     {final_val_perf}\n'
           f'Test:           {final_test_perf}\n'
           '-------------------------------\n\n')
    print(msg)

    return curves


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)

    main(args)
