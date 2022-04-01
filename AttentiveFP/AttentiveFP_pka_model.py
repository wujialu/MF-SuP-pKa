import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from AttentiveFP import build_dataset
from AttentiveFP.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch, run_an_eval_epoch_detail, \
    set_random_seed, AttentiveFPPredictor


# fix parameters of model
def AttentiveFP_model(times, task_name,
                      number_layers=2,
                      num_timesteps=2,
                      graph_feat_size=200,
                      lr=3e-4,
                      weight_decay=0,
                      dropout=0.1):
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
    args['task_name'] = task_name
    args['data_name'] = task_name
    args['bin_g_attentivefp_path'] = '../data/Attentivefp_graph_data/' + args['data_name'] + '_graph.bin'
    args['group_path'] = '../data/Attentivefp_graph_data/' + args['data_name'] + '_group.csv'
    args['times'] = times

    all_times_train_result = []
    all_times_val_result = []
    all_times_test_result = []
    result_pd = pd.DataFrame()
    result_pd['index'] = ['r2', 'mae', 'rmse']

    for time_id in range(args['times']):
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
            shuffle=True
        )
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

        model = AttentiveFPPredictor(n_tasks=task_number,
                                     node_feat_size=args['in_feats'],
                                     edge_feat_size=10,
                                     num_layers=args['number_layers'],
                                     num_timesteps=args['num_timesteps'],
                                     graph_feat_size=args['graph_feat_size'],
                                     dropout=args['drop_out'])

        filename = '../model/AttentiveFP/{}_early_stop.pth'.format(args['task_name'] + '_' + str(time_id + 1))
        stopper = EarlyStopping(patience=args['patience'], mode=args['mode'], filename=filename)
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
            print('epoch {:d}/{:d}, {}, lr: {:.6f}, loss: {:.4f},  train: {:.4f}, valid: {:.4f}, best valid {:.4f}, '
                  'test: {:.4f}'.format(
                epoch + 1, args['num_epochs'], args['metric_name'], lr, total_loss, train_score, val_score,
                stopper.best_score, test_score))
            if early_stop:
                break
        stopper.load_checkpoint(model)
        train_score = run_an_eval_epoch_detail(args, model, train_loader, out_path=None)[0]
        val_score = run_an_eval_epoch_detail(args, model, val_loader, out_path=None)[0]
        test_score = run_an_eval_epoch_detail(args, model, test_loader, out_path=None)[0]
        pred_name = 'prediction_' + str(time_id + 1)
        stop_test_list = run_an_eval_epoch_detail(args, model, test_loader,
                                                  out_path='../prediction/AttentiveFP/' +
                                                           args['task_name'] + '_' + pred_name + '_test.csv')
        stop_train_list = run_an_eval_epoch_detail(args, model, train_loader,
                                                   out_path='../prediction/AttentiveFP/' +
                                                            args['task_name'] + '_' + pred_name + '_train.csv')
        stop_val_list = run_an_eval_epoch_detail(args, model, val_loader,
                                                 out_path='../prediction/AttentiveFP/' +
                                                          args['task_name'] + '_' + pred_name + '_val.csv')
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
        print('the average train result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                          np.array(all_times_train_result).mean()))
        print('the train result of all tasks (std): {:.3f}'.format(np.array(all_times_train_result).std()))
        print('the train result of all tasks (var): {:.3f}'.format(np.array(all_times_train_result).var()))

        print('the val result of all tasks ({}): '.format(args['metric_name']), np.array(all_times_val_result))
        print('the average val result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                        np.array(all_times_val_result).mean()))
        print('the val result of all tasks (std): {:.3f}'.format(np.array(all_times_val_result).std()))
        print('the val result of all tasks (var): {:.3f}'.format(np.array(all_times_val_result).var()))

        print('the test result of all tasks ({}):'.format(args['metric_name']), np.array(all_times_test_result))
        print('the average test result of all tasks ({}): {:.3f}'.format(args['metric_name'],
                                                                         np.array(all_times_test_result).mean()))
        print('the test result of all tasks (std): {:.3f}'.format(np.array(all_times_test_result).std()))
        print('the test result of all tasks (var): {:.3f}'.format(np.array(all_times_test_result).var()))

    result_pd.to_csv('../result/AttentiveFP/' + args['task_name'] + '_result.csv', index=False)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, help="task name")
    args = parser.parse_args()
    AttentiveFP_model(times=10, task_name=args.task_name)
