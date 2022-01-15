"""
MLP Binary Classifier
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
from numpy import argmax
import time
import random
import torch
import math
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from utils import *



class MLP(torch.nn.Module):
    def __init__(self, dim, h_1=80, h_2=10, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, h_1)
        self.fc_2 = torch.nn.Linear(h_1, h_2)
        self.fc_3 = torch.nn.Linear(h_2, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


def MLP_classification(args, X_train, X_test, y_train, y_test):
    """
    binary classification with an MLP classifier
    """

    # set parameters
    GPU = args.gpu
    BATCH_SIZE = args.bs
    DROP_OUT = args.drop_out
    # Set device
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    train_y_pred_prob_list = []
    test_y_pred_prob_list = []
    train_y_true_list = []
    test_y_true_list = []

    for i in range(args.n_iter):

        # Initialize Model
        num_instance = X_train.shape[0]
        num_batch = int(math.ceil(num_instance / BATCH_SIZE))

        decoder = MLP(X_train.shape[1], h_1=args.h_1, h_2=args.h_2, drop=DROP_OUT)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        decoder = decoder.to(device)
        decoder_loss_criterion = torch.nn.BCELoss()

        train_losses = []
        for epoch in range(args.n_epoch):

            decoder = decoder.train()
            loss = 0

            for k in range(num_batch):
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance, s_idx + BATCH_SIZE)

                x_train_batch = np.array(X_train[s_idx: e_idx])
                y_true_batch = np.array(y_train[s_idx: e_idx])

                decoder_optimizer.zero_grad()

                y_true_batch_torch = torch.from_numpy(y_true_batch).float().to(device)
                x_train_batch = torch.from_numpy(x_train_batch).float().to(device)
                pred = decoder(x_train_batch).sigmoid()
                decoder_loss = decoder_loss_criterion(pred, y_true_batch_torch)
                decoder_loss.backward()
                decoder_optimizer.step()
                loss += decoder_loss.item()

            train_losses.append(loss / num_batch)

        train_pred_prob = eval_MLP_node_clf(decoder, X_train, BATCH_SIZE, device)
        train_y_pred_prob_list.append(train_pred_prob)
        train_y_true_list.append(y_train)

        test_pred_prob = eval_MLP_node_clf(decoder, X_test, BATCH_SIZE, device)
        test_y_pred_prob_list.append(test_pred_prob)
        test_y_true_list.append(y_test)

    return train_y_pred_prob_list, test_y_pred_prob_list, train_y_true_list, test_y_true_list



def eval_MLP_node_clf(decoder, x, batch_size, device):
    pred_prob = np.zeros(x.shape[0])
    num_instance = x.shape[0]
    num_batch = int(math.ceil(num_instance / batch_size))

    with torch.no_grad():
        decoder.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            x_batch = np.array(x[s_idx: e_idx])
            x_batch = torch.from_numpy(x_batch).float().to(device)
            pred_prob_batch = decoder(x_batch).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    return pred_prob


def convert_prob_to_lbl(y_true, y_pred_prob):
    """
    convert prediction probabilities to labels based on the threshold of the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    J = tpr - fpr
    ix = argmax(J)
    best_threshold = thresholds[ix]
    y_pred = (y_pred_prob > best_threshold)

    return np.array(y_pred)


def get_perf_agg_train_test(train_y_true, train_y_pred, train_y_pred_proba,
                            test_y_true, test_y_pred, test_y_prob_proba):
    train_perf_dict = compute_measures_wo_clf(train_y_true, train_y_pred, train_y_pred_proba)
    test_perf_dict = compute_measures_wo_clf(test_y_true, test_y_pred, test_y_prob_proba)
    perf_dict = {
        # train
        'train_prec': train_perf_dict['prec'],
        'train_rec': train_perf_dict['rec'],
        'train_f1': train_perf_dict['f1'],
        'train_acc': train_perf_dict['acc'],
        'train_auc_roc': train_perf_dict['auc_roc'],
        # test
        'test_prec': test_perf_dict['prec'],
        'test_rec': test_perf_dict['rec'],
        'test_f1': test_perf_dict['f1'],
        'test_acc': test_perf_dict['acc'],
        'test_auc_roc': test_perf_dict['auc_roc'],
    }
    return perf_dict


def get_perf_one_run_mlp_clf(network, train_pred_prob_list, test_pred_prob_list, train_y_true_list, test_y_true_list):
    """
    average performance results of one run MLP node classification
    """
    perf_dict_list = []
    for i in range(len(train_pred_prob_list)):
        i_y_pred_train = convert_prob_to_lbl(train_y_true_list[i], train_pred_prob_list[i])
        i_y_pred_test = convert_prob_to_lbl(test_y_true_list[i], test_pred_prob_list[i])

        # get performance results
        i_perf_dict = get_perf_agg_train_test(train_y_true_list[i], i_y_pred_train, train_pred_prob_list[i],
                                              test_y_true_list[i], i_y_pred_test, test_pred_prob_list[i])
        perf_dict_list.append(i_perf_dict)

    perf_pd = pd.DataFrame(perf_dict_list)
    avg_perf_dict = perf_pd.mean()
    avg_perf_dict['ID'] = f'{network}_MLP'
    return avg_perf_dict


def end_to_end_n_clf_with_MLP(args):
    """
    Binary node classification with MLP in a general way
    """
    network = args.network
    # make stats file
    stats_filename = f"./logs/{network}_stats_MLP.csv"
    stats_file = open(stats_filename, 'w')
    header_line = 'ID,train_prec,train_rec,train_f1,train_acc,train_auc_roc,test_prec,test_rec,test_f1,test_acc,' \
                  'test_auc_roc\n '
    stats_file.write(header_line)
    stats_file.close()

    data_path_partial = f'./data/{network}/'
    meta_col_names = ['node', 'label', 'train_mask', 'val_mask', 'test_mask', 'is_anchor']

    for i_run in tqdm(range(args.n_runs)):

        node_feats_df = pd.read_csv(f'./data/{network}/{network}_node_feats.csv')
        masks_df = pd.read_csv(f'{data_path_partial}/masks/masks_{i_run}.csv')

        node_feats_df = node_feats_df.merge(masks_df, how='inner', on='node')
        node_feats_df = node_feats_df.loc[node_feats_df['is_anchor'] == 1]

        train_node_feats = node_feats_df.loc[((node_feats_df['train_mask'] == 1) | (node_feats_df['val_mask'] == 1))]
        test_node_feats = node_feats_df.loc[node_feats_df['test_mask'] == 1]

        y_train = train_node_feats['label'].tolist()
        y_test = test_node_feats['label'].tolist()

        X_train = train_node_feats.drop(meta_col_names, axis=1)
        X_test = test_node_feats.drop(meta_col_names, axis=1)

        # scaling
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train.values)
        X_test = scaler.transform(X_test.values)

        # classification
        train_y_pred_prob_list, test_y_pred_prob_list, train_y_true_list, test_y_true_list = MLP_classification(args,
                                                                                                                X_train,
                                                                                                                X_test,
                                                                                                                y_train,
                                                                                                                y_test)
        i_run_perf_dict = get_perf_one_run_mlp_clf(network, train_y_pred_prob_list, test_y_pred_prob_list,
                                                   train_y_true_list, test_y_true_list)
        # append the results to file
        stats_df = pd.read_csv(stats_filename)
        stats_df = stats_df.append(i_run_perf_dict, ignore_index=True)
        stats_df.to_csv(stats_filename, index=False)

    # append average to the stats
    stats_df = pd.read_csv(stats_filename)
    clf_avg_stats = stats_df.mean(axis=0)
    clf_avg_stats['ID'] = 'Average_MLP'
    stats_df = stats_df.append(clf_avg_stats, ignore_index=True)
    stats_df.to_csv(stats_filename, index=False)


def end_to_end_n_clf_with_MLP_dynG(args):
    """
    Binary node classification with MLP for dynamic networks
    """
    network = args.network
    # make stats file
    stats_filename = f"./logs/{network}_stats_MLP_Dynamic.csv"
    stats_file = open(stats_filename, 'w')
    header_line = 'ID,train_prec,train_rec,train_f1,train_acc,train_auc_roc,test_prec,test_rec,test_f1,test_acc,' \
                  'test_auc_roc\n '
    stats_file.write(header_line)
    stats_file.close()

    data_path_partial = f'./data/{network}/'
    meta_col_names = ['node_id', 'timestamp', 'label', 'train_mask', 'val_mask', 'test_mask']

    train_emb_df = pd.read_csv(f'{data_path_partial}/{network}_TGBase_emb_train.csv')
    val_emb_df = pd.read_csv(f'{data_path_partial}/{network}_TGBase_emb_val.csv')
    training = [train_emb_df, val_emb_df]
    train_emb_df = pd.concat(training)
    y_train = train_emb_df['label'].tolist()
    X_train = train_emb_df.drop(meta_col_names, axis=1)

    test_emb_df = pd.read_csv(f'{data_path_partial}/{network}_TGBase_emb_test.csv')
    y_test = test_emb_df['label'].tolist()
    X_test = test_emb_df.drop(meta_col_names, axis=1)


    # scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)

    # classification
    train_y_pred_prob_list, test_y_pred_prob_list, train_y_true_list, test_y_true_list = MLP_classification(args,
                                                                                                            X_train,
                                                                                                            X_test,
                                                                                                            y_train,
                                                                                                            y_test)
    i_run_perf_dict = get_perf_one_run_mlp_clf(network, train_y_pred_prob_list, test_y_pred_prob_list,
                                               train_y_true_list, test_y_true_list)
    # append the results to file
    stats_df = pd.read_csv(stats_filename)
    stats_df = stats_df.append(i_run_perf_dict, ignore_index=True)
    stats_df.to_csv(stats_filename, index=False)




