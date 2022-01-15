"""
general utility functions
"""

from sklearn.metrics import *
import numpy as np
from scipy.stats import entropy
import math
import pandas as pd
import random


def perf_report(clf, X, y_true, y_pred, binary):
    """
    calculates and prints the performance results
    """
    if binary:
        prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred, average='binary')
        micro_f1 = f1_score(y_true, y_pred, average='binary')
        auc_roc = roc_auc_score(y_true, clf.predict_proba(X)[:, 1])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
    else:
        prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='macro')
        auc_roc = roc_auc_score(y_true, clf.predict_proba(X), multi_class='ovr')
        tn, fp, fn, tp = None, None, None, None
        TPR, FPR = None, None
    acc = accuracy_score(y_true, y_pred)

    perf_dict = {'prec': prec,
                 'rec': rec,
                 'f1': f1,
                 'micro_f1': micro_f1,
                 'auc_roc': auc_roc,
                 'acc': acc,
                 'tn': tn,
                 'fp': fp,
                 'fn': fn,
                 'tp': tp,
                 'TPR': TPR,
                 'FPR': FPR
                 }

    return perf_dict


def compute_measures_wo_clf(y_true, y_pred, y_pred_prob):
    """
    compute performance measures for the Binary classification without passing classifier
    """

    prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred, average='binary')
    micro_f1 = f1_score(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR_conf_mat = tp / (tp + fn)
    FPR_conf_mat = fp / (fp + tn)

    # roc-auc score & plot
    auc_roc = roc_auc_score(y_true, y_pred_prob)
    fpr, tpr, roc_auc_threshorlds = roc_curve(y_true, y_pred_prob)

    # pr-auc score & plot
    prec_curve, rec_curve, pr_auc_thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc_score = auc(rec_curve, prec_curve)

    # save measures
    perf_dict = {'prec': prec,
                 'rec': rec,
                 'f1': f1,
                 'micro_f1': micro_f1,
                 'auc_roc': auc_roc,
                 'acc': acc,
                 'tn': tn,
                 'fp': fp,
                 'fn': fn,
                 'tp': tp,
                 'TPR': TPR_conf_mat,
                 'FPR': FPR_conf_mat,
                 }
    return perf_dict


def agg_stats_of_list(value_list):
    """
    get the aggregated statistics of a list of values
    """
    if len(value_list) > 0:
        feat_dict = {'avg': np.mean(value_list),
                     'min': np.min(value_list),
                     'max': np.max(value_list),
                     'sum': np.sum(value_list),
                     'std': np.std(value_list),
                     'ent': entropy(value_list) if np.sum(value_list) != 0 else 0,
                     'len': len(value_list),
                     }
    else:
        feat_dict = {'avg': 0,
                     'min': 0,
                     'max': 0,
                     'sum': 0,
                     'std': 0,
                     'ent': 0,
                     'len': 0,
                     }
    if math.isinf(feat_dict['ent']):
        feat_dict['ent'] = 0

    return feat_dict


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, val_ratio, test_ratio, use_validation=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_features = np.load('./data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_features = np.load('./data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data

