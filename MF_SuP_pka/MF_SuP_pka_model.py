import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from MF_SuP_pka import build_dataset
from MF_SuP_pka.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch, run_an_eval_epoch_detail, \
    set_random_seed, SuP_pka_Predictor, load_pretrained_model

import warnings

warnings.filterwarnings("ignore")
import argparse


# fix parameters of model
def MF_SuP_pka_train(times, task_name,
                     number_layers=2,
                     num_timesteps=2,
                     graph_feat_size=200,
                     lr=3e-4,
                     weight_decay=0,
                     dropout=0.1,
                     acid_or_base=None,
                     k=0,
                     pretrain_aug=False,
                     stage='transfer'):
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'bond'
    args['metric_name'] = 'r2'
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 32
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['number_layers'] = number_layers
    args['num_timesteps'] = num_timesteps
    args['graph_feat_size'] = graph_feat_size
    args['drop_out'] = dropout
    args['lr'] = lr
    args['weight_decay'] = weight_decay
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name + f'_{k}_hop'
    args['data_name'] = task_name
    args['bin_g_attentivefp_path'] = '../data/MF_SuP_pka_graph_data/' + args['data_name'] + '_graph.bin'
    args['group_path'] = '../data/MF_SuP_pka_graph_data/' + args['data_name'] + '_group.csv'
    args['times'] = times
    args['acid_or_base'] = acid_or_base
    args['k'] = k

    if stage == 'transfer':
        if pretrain_aug == True:
            args['task_name'] += '_aug_transfer'
            if args['acid_or_base'] == 'acid':
                args['pretrain_model'] = '../model/pretrain_stop3_chembl_pka_acidic_71w_2_hop_1_early_stop.pth'
            elif args['acid_or_base'] == 'base':
                args['pretrain_model'] = '../model/pretrain_stop3_chembl_pka_basic_55w_3_hop_1_early_stop.pth'
            else:
                print('acid or base type error')
        else:
            args['task_name'] += '_transfer'
            if args['acid_or_base'] == 'acid':
                args['pretrain_model'] = '../model/pretrain_stop3_chembl_pka_acidic_62w_2_hop_1_early_stop.pth'
            elif args['acid_or_base'] == 'base':
                args['pretrain_model'] = '../model/pretrain_stop3_chembl_pka_basic_49w_3_hop_1_early_stop.pth'
            else:
                print('acid or base type error')


    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    result_pd['index'] = ['r2', 'mae', 'rmse']

    for time_id in range(args['times']):
        # time_id = 32
        set_random_seed(2020 + time_id)
        one_time_train_result = []
        one_time_val_result = []
        one_time_test_result = []
        print('***************************************************************************************************')
        print('{}, {}/{} time'.format(args['task_name'], time_id + 1, args['times']))
        print('***************************************************************************************************')
        # random split load dataset
        train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_pka_random_splited(
            bin_g_attentivefp_path=args['bin_g_attentivefp_path'],
            group_path=args['group_path'],
            shuffle=True)

        if args['task_aug'] == True:
            train_set_aug, val_set_aug, _ = build_dataset.load_graph_from_csv_bin_for_pka_pretrain(
                bin_g_attentivefp_path=args['aug_bin_path'],
                group_path=args['aug_group_path'],
                shuffle=True)  # 8:2

            train_set += train_set_aug
            val_set += val_set_aug
        print("Molecule graph is loaded!")

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args['batch_size'],
                                  shuffle=True,
                                  collate_fn=collate_molgraphs)

        val_loader = DataLoader(dataset=val_set,
                                batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs)

        test_loader = DataLoader(dataset=test_set,
                                 batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)

        loss_criterion = torch.nn.MSELoss(reduction='none')

        model = SuP_pka_Predictor(n_tasks=task_number,
                                  node_feat_size=args['in_feats'],
                                  edge_feat_size=10,
                                  num_layers=args['number_layers'],
                                  num_timesteps=args['num_timesteps'],
                                  graph_feat_size=args['graph_feat_size'],
                                  dropout=args['drop_out'],
                                  acid_or_base=args['acid_or_base'],
                                  k=args['k'])

        filename = '../model/MF_SuP_pka/{}_{}_early_stop.pth'.format(args['task_name'], time_id + 1)

        stopper = EarlyStopping(patience=args['patience'], filename=filename,
                                task_name=args['task_name'] + '_' + str(time_id + 1), mode=args['mode'],
                                pretrained_model=args['pretrain_model'])

        if stage == 'transfer':
            stopper.load_pretrained_model(model)

        model.to(args['device'])
        lr = args['lr']

        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(args['num_epochs']):
            # Train
            # lr = args['lr']
            _, total_loss = run_a_train_epoch(args, model, train_loader, loss_criterion, optimizer)
            # Validation and early stop
            train_score = run_an_eval_epoch_detail(args, model, train_loader, out_path=None)[0]
            val_score = run_an_eval_epoch_detail(args, model, val_loader, out_path=None)[0]
            test_score = run_an_eval_epoch_detail(args, model, test_loader, out_path=None)[0]
            early_stop = stopper.step(val_score, model)

            # schedular.step()
            # lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('epoch {:d}/{:d}, {}, lr: {:.6f}, loss: {:.4f},  train: {:.4f}, valid: {:.4f}, best valid {:.4f}, '
                  'test: {:.4f}'.format(
                epoch + 1, args['num_epochs'], args['metric_name'], lr, total_loss, train_score, val_score,
                stopper.best_score, test_score))
            if early_stop:
                break

        stopper.load_checkpoint(model)
        train_score = run_an_eval_epoch_detail(args, model, train_loader, out_path=None)[0]  # [r2,mae,rmse]
        val_score = run_an_eval_epoch_detail(args, model, val_loader, out_path=None)[0]
        test_score = run_an_eval_epoch_detail(args, model, test_loader, out_path=None)[0]
        pred_name = 'prediction_' + str(time_id + 1)
        stop_test_list = run_an_eval_epoch_detail(args, model, test_loader,
                                                  out_path='../prediction/MF_SuP_pka/' + args[
                                                      'task_name'] + '_' + pred_name + '_test.csv')
        stop_train_list = run_an_eval_epoch_detail(args, model, train_loader,
                                                   out_path='../prediction/MF_SuP_pka/' + args[
                                                       'task_name'] + '_' + pred_name + '_train.csv')
        stop_val_list = run_an_eval_epoch_detail(args, model, val_loader,
                                                 out_path='../prediction/MF_SuP_pka/' + args[
                                                     'task_name'] + '_' + pred_name + '_val.csv')
        result_pd['train_' + str(time_id + 1)] = stop_train_list
        result_pd['val_' + str(time_id + 1)] = stop_val_list
        result_pd['test_' + str(time_id + 1)] = stop_test_list
        print(result_pd[['index', 'train_' + str(time_id + 1), 'val_' + str(time_id + 1), 'test_' + str(time_id + 1)]])
        print('********************************{}, {}_times_result*******************************'.format(
            args['task_name'],
            time_id + 1))
        print("training_result:", round(train_score, 4))
        print("val_result:", round(val_score, 4))
        print("test_result:", round(test_score, 4))

        one_time_train_result.append(train_score)
        one_time_val_result.append(val_score)
        one_time_test_result.append(test_score)
        # except:
        #     task_number = task_number - 1
        all_times_train_result.append(round(np.array(one_time_train_result).mean(), 4))
        all_times_val_result.append(round(np.array(one_time_val_result).mean(), 4))
        all_times_test_result.append(round(np.array(one_time_test_result).mean(), 4))
        # except:
        #     print('{} times is failed!'.format(time_id+1))
        print("************************************{}_times_result************************************".format(
            time_id + 1))
        print('the train result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_train_result))
        print('the average train result of all tasks ({}): {:.4f}'.format(args['metric_name'],
                                                                          np.array(all_times_train_result).mean()))
        print('the train result of all tasks (std): {:.4f}'.format(np.array(all_times_train_result).std()))
        print('the train result of all tasks (var): {:.4f}'.format(np.array(all_times_train_result).var()))

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result))
        print('the average val result of all tasks ({}): {:.4f}'.format(args['metric_name'],
                                                                        np.array(all_times_val_result).mean()))
        print('the val result of all tasks (std): {:.4f}'.format(np.array(all_times_val_result).std()))
        print('the val result of all tasks (var): {:.4f}'.format(np.array(all_times_val_result).var()))

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result))
        print('the average test result of all tasks ({}): {:.4f}'.format(args['metric_name'],
                                                                         np.array(all_times_test_result).mean()))
        print('the test result of all tasks (std): {:.4f}'.format(np.array(all_times_test_result).std()))
        print('the test result of all tasks (var): {:.4f}'.format(np.array(all_times_test_result).var()))

    result_pd.to_csv('../result/MF_SuP_pka/' + args['task_name'] + '_result.csv', index=False)


def MF_SuP_pka_eval(task_name,
                    number_layers=2,
                    num_timesteps=2,
                    graph_feat_size=200,
                    dropout=0.1,
                    acid_or_base=None,
                    k=0,
                    checkpoint=None):
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'bond'
    args['metric_name'] = 'r2'
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 32
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['number_layers'] = number_layers
    args['num_timesteps'] = num_timesteps
    args['graph_feat_size'] = graph_feat_size
    args['drop_out'] = dropout
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name + f'_{k}_hop'
    args['data_name'] = task_name
    args['bin_g_attentivefp_path'] = '../data/MF_SuP_pka_graph_data/' + args['data_name'] + '_graph.bin'
    args['group_path'] = '../data/MF_SuP_pka_graph_data/' + args['data_name'] + '_group.csv'
    args['acid_or_base'] = acid_or_base
    args['k'] = k
    args['checkpoint'] = checkpoint

    # load data
    test_set, task_number = build_dataset.load_graph_from_csv_bin_for_external_test(
        bin_g_attentivefp_path=args['bin_g_attentivefp_path'],
        group_path=args['group_path'],)
    print("Molecule graph is loaded!")

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)

    # load model
    model = SuP_pka_Predictor(n_tasks=task_number,
                              node_feat_size=args['in_feats'],
                              edge_feat_size=10,
                              num_layers=args['number_layers'],
                              num_timesteps=args['num_timesteps'],
                              graph_feat_size=args['graph_feat_size'],
                              dropout=args['drop_out'],
                              acid_or_base=args['acid_or_base'],
                              k=args['k'])
    load_pretrained_model(args['checkpoint'], model)
    model.to(args.device)
     
    # make prediction and sava results
    run_an_eval_epoch_detail(args, model, test_loader, out_path='../prediction/MF_SuP_pka/' + args['task_name'] + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, help="task name", default='pka_acidic_2750')
    parser.add_argument("--type", type=str, help="task type(acid or base)", default='acid')
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--pretrain_aug", action='store_true', default=False)
    parser.add_argument("--stage", type=str, default='transfer')  # [before_transfer, transfer]
    parser.add_argument("--mode", type=str, default='transfer')  # [train, eval]
    parser.add_argument("--checkpoint", type=str, default=None)  # [train, eval]

    args = parser.parse_args()

    if args.mode == 'train':
        MF_SuP_pka_train(times=10, task_name=args.task_name, acid_or_base=args.type,
                        k=args.k_hop, pretrain_aug=args.pretrain_aug, stage=args.stage)
    elif args.mode == 'eval':
        if args.checkpoint is None:
            if args.type == 'acid':
                args.checkpoint = '../model/pka_acidic_2750_aug_transfer_2_hop_1_early_stop.pth'
            elif args.type == 'base':
                args.checkpoint = '../model/pka_basic_2992_aug_transfer_3_hop_1_early_stop.pth'
        MF_SuP_pka_eval(task_name=args.task_name, acid_or_base=args.type,
                        k=args.k_hop, checkpoint=args.checkpoint)