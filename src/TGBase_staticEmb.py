"""
TGBase features generation for static node classification
"""

import argparse
import datetime
import sys
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import entropy
import multiprocessing as mp
import networkx as nx
import pickle
from utils import *


def get_edge_attributes(edgelist_df, node, e_attr_opt, dir_opt):
    """
    get attributes of incident edges of a node
    """
    if dir_opt == 'in':  # incoming edge
        e_feat_list = edgelist_df.loc[edgelist_df['target'] == node, e_attr_opt].tolist()
    elif dir_opt == 'out':  # outgoing edge
        e_feat_list = edgelist_df.loc[edgelist_df['source'] == node, e_attr_opt].tolist()
    elif dir_opt == 'all':  # all edges
        e_feat_list = edgelist_df.loc[
            ((edgelist_df['source'] == node) | (edgelist_df['target'] == node)), e_attr_opt].tolist()
    else:
        raise ValueError("Undefined option!")

    if e_attr_opt == 'timestamp':
        timestamp_list = [datetime.datetime.fromtimestamp(t) for t in e_feat_list]
        timestamp_list.sort()
        # interval of txs in minutes
        e_feat_list = [((timestamp_list[i + 1] - timestamp_list[i]).total_seconds() / 60) for i in
                       range(len(timestamp_list) - 1)]

    return e_feat_list


def generate_edge_attr_feats(edgelist_df, node, e_attr_opt):
    """
    generate node's features that are based on the edge attributes
    """
    in_edge_attr_list = get_edge_attributes(edgelist_df, node, e_attr_opt, 'in')
    in_edge_stats = agg_stats_of_list(in_edge_attr_list)

    out_edge_attr_list = get_edge_attributes(edgelist_df, node, e_attr_opt, 'out')
    out_edge_stats = agg_stats_of_list(out_edge_attr_list)

    all_edge_attr_list = get_edge_attributes(edgelist_df, node, e_attr_opt, 'all')
    all_edge_stats = agg_stats_of_list(all_edge_attr_list)

    return in_edge_stats, out_edge_stats, all_edge_stats


def get_neighborhood_feat(G, node):
    """
        do some pre-processing for generating neighborhood features
    """
    neighbors = list(G.neighbors(node))
    # degree
    degree_nodeView = G.degree(neighbors)
    degree_list = list(dict(degree_nodeView).values())
    degree_feats = agg_stats_of_list(degree_list)
    # in_degree
    in_degree_nodeView = G.in_degree(neighbors)
    in_degree_list = list(dict(in_degree_nodeView).values())
    in_degree_feats = agg_stats_of_list(in_degree_list)
    # out_degree
    out_degree_nodeView = G.out_degree(neighbors)
    out_degree_list = list(dict(out_degree_nodeView).values())
    out_degree_feats = agg_stats_of_list(out_degree_list)

    return in_degree_feats, out_degree_feats, degree_feats


def generate_node_feats(G, edgelist_df, node):
    """
    generate all features of a node
    """
    # timestamp features
    stats_time_in, stats_time_out, stats_time_all = generate_edge_attr_feats(edgelist_df, node, 'timestamp')
    # weight features
    stats_w_in, stats_w_out, stats_w_all = generate_edge_attr_feats(edgelist_df, node, 'weight')
    # neighborhood features
    nei_in_degree_feats, nei_out_degree_feats, nei_degree_feats = get_neighborhood_feat(G, node)

    node_feats_dict = {
        # node intrinsic features
        'node': node,

        # self
        'degree': G.degree(node),
        'in_degree': G.in_degree(node),
        'out_degree': G.out_degree(node),

        # intensity
        # node is 'target'
        'avg_w_in': stats_w_in['avg'],
        'min_w_in': stats_w_in['min'],
        'max_w_in': stats_w_in['max'],
        'sum_w_in': stats_w_in['sum'],
        'std_w_in': stats_w_in['std'],
        'ent_w_in': stats_w_in['ent'],

        # node is 'source'
        'avg_w_out': stats_w_out['avg'],
        'min_w_out': stats_w_out['min'],
        'max_w_out': stats_w_out['max'],
        'sum_w_out': stats_w_out['sum'],
        'std_w_out': stats_w_out['std'],
        'ent_w_out': stats_w_out['ent'],

        # either source or target
        'avg_w_all': stats_w_all['avg'],
        'min_w_all': stats_w_all['min'],
        'max_w_all': stats_w_all['max'],
        'sum_w_all': stats_w_all['sum'],
        'std_w_all': stats_w_all['std'],
        'ent_w_all': stats_w_all['ent'],

        # time
        # node is 'target
        'avg_t_in': stats_time_in['avg'],
        'min_t_in': stats_time_in['min'],
        'max_t_in': stats_time_in['max'],
        'sum_t_in': stats_time_in['sum'],
        'std_t_in': stats_time_in['std'],
        'ent_t_in': stats_time_in['ent'],

        # node is 'source'
        'avg_t_out': stats_time_out['avg'],
        'min_t_out': stats_time_out['min'],
        'max_t_out': stats_time_out['max'],
        'sum_t_out': stats_time_out['sum'],
        'std_t_out': stats_time_out['std'],
        'ent_t_out': stats_time_out['ent'],

        # either source or target
        'avg_t_all': stats_time_all['avg'],
        'min_t_all': stats_time_all['min'],
        'max_t_all': stats_time_all['max'],
        'sum_t_all': stats_time_all['sum'],
        'std_t_all': stats_time_all['std'],
        'ent_t_all': stats_time_all['ent'],

        # neighborhood features
        'avg_neighbor_degree': nei_degree_feats['avg'],
        'min_neighbor_degree': nei_degree_feats['min'],
        'max_neighbor_degree': nei_degree_feats['max'],
        'sum_neighbor_degree': nei_degree_feats['sum'],
        'std_neighbor_degree': nei_degree_feats['std'],
        'ent_neighbor_degree': nei_degree_feats['ent'],

        'avg_neighbor_in_degree': nei_in_degree_feats['avg'],
        'min_neighbor_in_degree': nei_in_degree_feats['min'],
        'max_neighbor_in_degree': nei_in_degree_feats['max'],
        'sum_neighbor_in_degree': nei_in_degree_feats['sum'],
        'std_neighbor_in_degree': nei_in_degree_feats['std'],
        'ent_neighbor_in_degree': nei_in_degree_feats['ent'],

        'avg_neighbor_out_degree': nei_out_degree_feats['avg'],
        'min_neighbor_out_degree': nei_out_degree_feats['min'],
        'max_neighbor_out_degree': nei_out_degree_feats['max'],
        'sum_neighbor_out_degree': nei_out_degree_feats['sum'],
        'std_neighbor_out_degree': nei_out_degree_feats['std'],
        'ent_neighbor_out_degree': nei_out_degree_feats['ent'],
    }
    return node_feats_dict


def generate_all_nodes_feat(G, edgelist):
    """
    generate features for all nodes of the graph
    """
    node_list = list(G.nodes())
    node_feats_list = []
    for node in tqdm(node_list):
        node_feats_dict = generate_node_feats(G, edgelist, node)
        node_feats_list.append(node_feats_dict)
    node_feats_df = pd.DataFrame(node_feats_list)
    return node_feats_df


def generate_graph(edgelist_path):
    """
    read an edge-list of one of the REV2 datasets
    """
    edgelist_df = pd.read_csv(edgelist_path, header=None)
    edgelist_df.columns = ['source', 'target', 'weight', 'timestamp']
    G = nx.from_pandas_edgelist(edgelist_df, source='source', target='target', edge_attr=['weight', 'timestamp'],
                                create_using=nx.MultiDiGraph)
    return edgelist_df, G


def main():
    # Argument passing
    parser = argparse.ArgumentParser(description='Generate TGBase Features for Static Networks.')
    # path setting
    parser.add_argument('--network', type=str, default='', help='Network name.')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
        print("Arguments:", args)
    except:
        parser.print_help()
        sys.exit()

    network_name = args.network
    edgelist_path = f'./data/{network_name}/{network_name}_network.csv'
    edgelist_df, G = generate_graph(edgelist_path)
    node_feats_df = generate_all_nodes_feat(G, edgelist_df)
    node_feats_df_filename = f'./data/{network_name}/{network_name}_node_feats.csv'
    node_feats_df.to_csv(node_feats_df_filename, index=False)



if __name__ == '__main__':
    main()