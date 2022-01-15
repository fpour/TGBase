"""
TGBase features generation for dynamic node classification
"""
import argparse
import sys

import numpy as np
import random
from utils import *
import pandas as pd
from tqdm import tqdm

rnd_seed = 2021
random.seed(rnd_seed)


def init_node_memory(node_list, no_edge_feats):
    """
    initialize the nodes' feature state list for the given node_list
    """
    init_node_states = {}
    for node in node_list:
        init_node_states[node] = {}
        init_node_states[node]['no_event'] = 0  # number of events involving this node so far
        init_node_states[node]['time_avg'] = 0
        init_node_states[node]['time_min'] = float('inf'),
        init_node_states[node]['time_max'] = 0
        init_node_states[node]['time_sum'] = 0
        init_node_states[node]['time_last'] = 0  # timestamp of the previous seen event
        init_node_states[node]['time_std'] = 0

        for feat_idx in range(no_edge_feats):
            init_node_states[node]['feat_' + str(feat_idx) + '_avg'] = 0
            init_node_states[node]['feat_' + str(feat_idx) + '_min'] = float('inf')
            init_node_states[node]['feat_' + str(feat_idx) + '_max'] = 0
            init_node_states[node]['feat_' + str(feat_idx) + '_sum'] = 0
            init_node_states[node]['feat_' + str(feat_idx) + '_std'] = 0

    return init_node_states


def update_node_state(current_node_state, timestamp, edge_feature):
    """
    update the state of one node based on the event characteristics
    """
    time_interval = timestamp - current_node_state['time_last']
    new_node_state = {'no_event': current_node_state['no_event'] + 1,
                      'time_avg': (current_node_state['time_avg'] * current_node_state['no_event'] + time_interval) / (
                              current_node_state['no_event'] + 1),
                      'time_min': min(current_node_state['time_min'], time_interval),
                      'time_max': max(current_node_state['time_max'], time_interval),
                      'time_sum': current_node_state['time_sum'] + time_interval,
                      'time_last': timestamp,
                      }
    new_node_state['time_std'] = np.sqrt(
        ((time_interval - current_node_state['time_avg']) * (time_interval - new_node_state['time_avg']) +
         (current_node_state['no_event']) * current_node_state['time_std'] ** 2) / new_node_state['no_event'])
    for feat_idx in range(len(edge_feature)):
        id = 'feat_' + str(feat_idx)
        new_node_state[id + '_avg'] = (current_node_state[id + '_avg'] * current_node_state['no_event'] + edge_feature[
            feat_idx]) / (current_node_state['no_event'] + 1)
        new_node_state[id + '_min'] = min(current_node_state[id + '_min'], edge_feature[feat_idx])
        new_node_state[id + '_max'] = max(current_node_state[id + '_max'], edge_feature[feat_idx])
        new_node_state[id + '_sum'] = current_node_state[id + '_sum'] + edge_feature[feat_idx]
        new_node_state[id + '_std'] = np.sqrt(((edge_feature[feat_idx] - new_node_state[id + '_avg']) * (
                edge_feature[feat_idx] - current_node_state[id + '_avg']) +
                                               current_node_state['no_event'] * current_node_state[
                                                   id + '_std'] ** 2) / (new_node_state['no_event']))
    return new_node_state


def gen_dynamic_emb_for_data_split(data, node_memory, edge_features):
    """
    generate dynamic embeddings for a list of nodes
    """
    emb_list = []
    print("Info: Number of interactions:", len(data.sources))
    for idx, source in tqdm(enumerate(data.sources)):  # NB: Only "source" nodes
        prev_source_state = node_memory[source]  # current state features
        current_source_state = update_node_state(prev_source_state, data.timestamps[idx],
                                                 edge_features[data.edge_idxs[idx]])
        node_memory[source] = current_source_state
        # if 'time_last' in node_states[source]: del node_states[source]['time_last']
        current_source_state['node_id'] = source
        current_source_state['timestamp'] = data.timestamps[idx]
        current_source_state['label'] = data.labels[idx]
        emb_list.append(current_source_state)

    return node_memory, emb_list


def append_mask_to_emb(emb_list, mask_triplet):
    for emb in emb_list:
        emb['train_mask'] = mask_triplet[0]
        emb['val_mask'] = mask_triplet[1]
        emb['test_mask'] = mask_triplet[2]
    return emb_list


def generate_TGBase_DynEmb(network, val_ratio, test_ratio, use_validation):
    """
    generate TGBase dynamic embeddings for a dataset
    """
    full_data, node_features, edge_features, \
    train_data, val_data, test_data = get_data_node_classification(network, val_ratio, test_ratio, use_validation)

    node_list = full_data.unique_nodes
    print("Info: Total Number of nodes: {}".format(len(node_list)))
    no_edge_feats = len(edge_features[0])
    node_memory = init_node_memory(node_list, no_edge_feats)

    node_emb_list = []
    # train split
    print("Info: Generating embeddings for training set...")
    node_memory, emb_list_train = gen_dynamic_emb_for_data_split(train_data, node_memory, edge_features)
    train_embs = append_mask_to_emb(emb_list_train, (1, 0, 0))
    dyEmb_filename = f'./data/{network}/{network}_TGBase_emb_train.csv'
    node_emb_df = pd.DataFrame(train_embs)
    node_emb_df.to_csv(dyEmb_filename, index=False)

    # val split
    print("Info: Generating embeddings for validation set...")
    node_memory, emb_list_val = gen_dynamic_emb_for_data_split(val_data, node_memory, edge_features)
    val_embs = append_mask_to_emb(emb_list_val, (0, 1, 0))
    dyEmb_filename = f'./data/{network}/{network}_TGBase_emb_val.csv'
    node_emb_df = pd.DataFrame(val_embs)
    node_emb_df.to_csv(dyEmb_filename, index=False)

    # test split
    print("Info: Generating embeddings for test set...")
    node_memory, emb_list_test = gen_dynamic_emb_for_data_split(test_data, node_memory, edge_features)
    test_embs = append_mask_to_emb(emb_list_test, (0, 0, 1))
    dyEmb_filename = f'./data/{network}/{network}_TGBase_emb_test.csv'
    node_emb_df = pd.DataFrame(test_embs)
    node_emb_df.to_csv(dyEmb_filename, index=False)


def main():
    # Argument passing
    parser = argparse.ArgumentParser(description='Generate TGBase Features for Dynamic Networks.')
    # path setting
    parser.add_argument('--network', type=str, default='wikipedia', help='Network name.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio.')
    parser.add_argument('--use_validation', action='store_true', help='Whether to use a validation set.')


    sys_argv = sys.argv
    try:
        args = parser.parse_args()
        print("Arguments:", args)
    except:
        parser.print_help()
        sys.exit()

    generate_TGBase_DynEmb(args.network, args.val_ratio, args.test_ratio, args.use_validation)


if __name__ == '__main__':
    main()

