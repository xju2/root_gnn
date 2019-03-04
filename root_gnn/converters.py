"""
functions that convert different inputs to networkx
"""
import numpy as np
import pandas as pd
import networkx as nx

from .prepare import get_edge_features
from trackml.dataset import load_event
import re
import os


vlids = [(7,2), (7,4), (7,6), (7,8), (7,10), (7,12), (7,14),
         (8,2), (8,4), (8,6), (8,8),
         (9,2), (9,4), (9,6), (9,8), (9,10), (9,12), (9,14),
         (12,2), (12,4), (12,6), (12,8), (12,10), (12,12),
         (13,2), (13,4), (13,6), (13,8),
         (14,2), (14,4), (14,6), (14,8), (14,10), (14,12),
         (16,2), (16,4), (16,6), (16,8), (16,10), (16,12),
         (17,2), (17,4),
         (18,2), (18,4), (18,6), (18,8), (18,10), (18,12)]
n_det_layers = len(vlids)

def create_evt_pairs_converter(evt_file_name, use_all_nodes_=False):
    use_all_nodes = use_all_nodes_
    evt_id = int(re.search('event00000([0-9]*)', os.path.basename(evt_file_name)).group(1))

    hits, particles, truth = load_event(
        evt_file_name, parts=['hits', 'particles', 'truth'])

    truth = truth.merge(particles[['particle_id']], on='particle_id')
    hits = hits.merge(truth[['hit_id', 'particle_id']], on='hit_id', how='left')
    hits = hits.fillna(value=0)

    # Assign convenient layer number [0-47]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])

    # add new features
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    hits = hits.assign(r=r, phi=phi)

    # add hit indexes to column hit_idx
    hits_with_idx = hits.rename_axis('hit_idx').reset_index()
    feature_scale = np.array([1000., np.pi, 1000.])

    print("Event {} pairs --> graphs with {:09d} nodes".format(evt_id, hits_with_idx.shape[0]))

    def pairs_to_graph(pairs):
        """
        pairs: np.array[n_pairs, 2]
        """
        # dealing with pairs
        df_in_nodes  = pd.DataFrame(pairs[:, 0], columns=['hit_id'])
        df_out_nodes = pd.DataFrame(pairs[:, 1], columns=['hit_id'])

        df_in_nodes  = df_in_nodes.merge(hits_with_idx, on='hit_id', how='left')
        df_out_nodes = df_out_nodes.merge(hits_with_idx, on='hit_id', how='left')

        # find out if edge is true edge from particle ID != 0
        n_edges = df_in_nodes.shape[0]
        y = np.zeros(n_edges, dtype=np.float32)
        pid1 = df_in_nodes ['particle_id'].values
        pid2 = df_out_nodes['particle_id'].values
        y[:] = (pid1 == pid2) & (pid1 != 0)

        graph = nx.DiGraph()
        # only add the hits that are used in edges (not all nodes)
        if not use_all_nodes:
            hits_id_dict = {}
            used_hits_set = np.unique(np.concatenate([df_in_nodes['hit_idx'], df_out_nodes['hit_idx']]))
            for ii,idx in enumerate(used_hits_set):
                hits_id_dict[idx] = ii
                graph.add_node(ii, pos=hits.iloc[idx][['r', 'phi', 'z']].values/feature_scale, solution=0.0)
        else:
            # use all hits in the event!
            for idx in hits_with_idx['hit_idx']:
                graph.add_node(idx, pos=hits.iloc[idx][['r', 'phi', 'z']].values/feature_scale, solution=0.0)

        # add edges
        for idx in range(n_edges):
            in_hit_idx  = int(df_in_nodes.iloc[idx, 1])
            out_hit_idx = int(df_out_nodes.iloc[idx, 1])

            if use_all_nodes:
                in_node_idx  = in_hit_idx
                out_node_idx = out_hit_idx
            else:
                in_node_idx  = hits_id_dict[in_hit_idx]
                out_node_idx = hits_id_dict[out_hit_idx]

            f1 = graph.node[in_node_idx]['pos']
            f2 = graph.node[out_node_idx]['pos']
            distance = get_edge_features(f1, f2)
            graph.add_edge(in_node_idx,  out_node_idx, distance=distance, solution=y[idx])
            graph.add_edge(out_node_idx, in_node_idx,  distance=distance, solution=y[idx])
            graph.node[in_node_idx].update(solution=y[idx])
            graph.node[out_node_idx].update(solution=y[idx])


        graph.graph['features'] = np.array([0.])
        return graph

    return pairs_to_graph

