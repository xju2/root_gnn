import numpy as np
import pandas as pd
import itertools
from typing import Optional

from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet
from root_gnn.src.datasets.base import linecount

n_node_features = 7

is_signal = False
def make_graph(event, debug=False, data_dict=False):
    """
    Build a graph where the nodes are the particles 
    constituting the leading jet, edges are any connections of them.
    The graph will feed to GNN to spearate w boson events from qcd.
    """
    scale = 0.001
    # information of each particle: px, py, pz, E, pdgID, isFromW, isInLeadingJet
    n_particles = len(event) // n_node_features
    nodes = [[
        event[inode*n_node_features+0], # px
        event[inode*n_node_features+1], # py
        event[inode*n_node_features+2], # pz
        event[inode*n_node_features+3]  # E
    ] for inode in range(n_particles) if event[inode*n_node_features+6] == 1]

    if len(nodes) < 1:
        return [(None, None)]

    nodes = np.array(nodes, dtype=np.float32) / scale
    n_nodes = nodes.shape[0]
    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)


    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)

    if debug:
        print(n_edges, "edges")
        print("senders:", senders)
        print("receivers:", receivers)
        # print("edge:", edge_target)
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([n_nodes], dtype=np.float32)
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([float(is_signal)], dtype=np.float32)
    }
    if data_dict:
        return [(input_datadict, target_datadict)]
    else:
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        return [(input_graph, target_graph)]

def read(filename, start_entry, nentries):
    ievt = 0
    with open(filename, 'r') as f:
        for line in f:
            if ievt < start_entry:
                continue
            if ievt >= start_entry + nentries:
                break
            yield [float(x) for x in line.split()]
            ievt += 1


class WTaggerLeadingJetDataset(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph

    def signal(self, ss=True):
        global is_signal
        is_signal = ss

    def _num_evts(self, filename: str) -> int:
        return linecount(filename)