"""
utility functions for node classification; dynamic graphs
"""

import argparse
import sys
import pandas as pd
import numpy as np
from scipy.stats import entropy
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from utils import *
from tqdm import tqdm

rnd_seed = 2021


def base_rflr_classification(clf, identifier, X_train, X_test, y_train, y_test, binary):
    """
    train the model on the train set and test it on the test set.
    to be consistent among different run, the indices are passed.
    important NOTE: it is implicitly inferred that the positive label is 1.
    """

    # train the model
    clf.fit(X_train, y_train)

    # predict the training set labels
    y_train_pred = clf.predict(X_train)

    # predict the test set labels
    y_test_pred = clf.predict(X_test)

    # evaluate the performance for the training set
    tr_perf_dict = perf_report(clf, X_train, y_train, y_train_pred, binary)
    ts_perf_dict = perf_report(clf, X_test, y_test, y_test_pred, binary)

    perf_dict = {
        'ID': identifier,
        # train
        'train_prec': tr_perf_dict['prec'],
        'train_rec': tr_perf_dict['rec'],
        'train_f1': tr_perf_dict['f1'],
        'train_acc': tr_perf_dict['acc'],
        'train_auc_roc': tr_perf_dict['auc_roc'],
        # test
        'test_prec': ts_perf_dict['prec'],
        'test_rec': ts_perf_dict['rec'],
        'test_f1': ts_perf_dict['f1'],
        'test_acc': ts_perf_dict['acc'],
        'test_auc_roc': ts_perf_dict['auc_roc'],
    }

    return perf_dict


def rf_lr_classification(X_train, X_test, y_train, y_test, stats_file, network, clf_name, binary):
    """
    apply classification to input X with label y with "Random Forest" & "Logistic Regression"
    :param X_train: train set
    :param X_test: test set
    :param y_train: train set labels
    :param y_test: test set labels
    :return the classification results
    """
    # define classifier
    if clf_name == 'RF':
        clf = RandomForestClassifier(n_estimators=50, max_features=10, max_depth=5, random_state=rnd_seed)
        # rf_clf = RandomForestClassifier(n_estimators=500, random_state=rnd_seed)
    elif clf_name == 'LR':
        clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1e5, random_state=rnd_seed)

    # apply classification
    clf_perf = base_rflr_classification(clf, f'{network}_{clf_name}', X_train, X_test, y_train, y_test, binary)

    # append the results to file
    stats_df = pd.read_csv(stats_file)
    stats_df = stats_df.append(clf_perf, ignore_index=True)
    stats_df.to_csv(stats_file, index=False)

    return clf_perf


def end_to_end_rf_lr_clf(args):

    ############
    # main task
    ############
    network = args.network
    n_iter = args.n_runs
    clf_name = args.clf
    binary = True

    # make stats file
    stats_filename = f"./logs/{network}_stats_{args.clf}.csv"
    stats_file = open(stats_filename, 'w')
    header_line = 'ID,train_prec,train_rec,train_f1,train_acc,train_auc_roc,test_prec,test_rec,test_f1,test_acc,' \
                  'test_auc_roc\n '
    stats_file.write(header_line)
    stats_file.close()

    meta_col_names = ['node', 'label', 'train_mask', 'val_mask', 'test_mask', 'is_anchor']

    # read node features
    node_feats_filename = f'./data/{network}/{network}_node_feats.csv'

    for i in tqdm(range(n_iter)):
        node_feats_df = pd.read_csv(node_feats_filename)
        # read masks
        masks_filename = f'./data/{network}/masks/masks_{i}.csv'
        masks_df = pd.read_csv(masks_filename)

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
        clf_perf = rf_lr_classification(X_train, X_test, y_train, y_test, stats_filename, network, clf_name, binary)

    # append average to the stats
    stats_df = pd.read_csv(stats_filename)
    id = f'{network}_{clf_name}'
    clf_stats = stats_df.loc[stats_df['ID'] == id]
    clf_avg_stats = clf_stats.mean(axis=0)
    clf_avg_stats['ID'] = f'{id}_avg'
    stats_df = stats_df.append(clf_avg_stats, ignore_index=True)
    stats_df.to_csv(stats_filename, index=False)


def end_to_end_rf_lr_clf_DynG(args):

    network = args.network
    n_iter = args.n_runs
    clf_name = args.clf
    binary = True

    # make stats file
    stats_filename = f"./logs/{network}_stats_{args.clf}_Dynamic.csv"
    stats_file = open(stats_filename, 'w')
    header_line = 'ID,train_prec,train_rec,train_f1,train_acc,train_auc_roc,test_prec,test_rec,test_f1,test_acc,' \
                  'test_auc_roc\n '
    stats_file.write(header_line)
    stats_file.close()

    meta_col_names = ['node_id', 'timestamp', 'label', 'train_mask', 'val_mask', 'test_mask']
    data_path_partial = f'./data/{network}/'

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
    clf_perf = rf_lr_classification(X_train, X_test, y_train, y_test, stats_filename, network, clf_name, binary)


