import numpy as np
import pandas as pd
import itertools
from typing import Optional

from graph_nets import utils_tf


from root_gnn.src.datasets.base import DataSet

tree_name = "output"
def make_graph(event, debug=False):
    # selecting events with at least one reco jets
    # and two true tau leptons

    if event.nJets < 1 or len(event.truthTauEt) < 2:
        return [(None, None)]

    # globals
    global_attr =  [event.truthTauEt[0], event.truthTauEta[0], event.truthTauPhi[0]]
    global_attr += [event.truthTauEt[1], event.truthTauEta[1], event.truthTauPhi[1]]        
    
    # nodes
    n_nodes = 0
    
    def get_track_info(idx):
        return [event.TrackPt[idx], event.TrackEta[idx], event.TrackPhi[idx]]

    def get_tower_info(idx):
        return [event.JetTowerEt[idx], event.JetTowerEta[idx], event.JetTowerPhi[idx]]

    
    nodes = []    
    prev_num_tower = 0
    for indv_jet in range(event.nJets):
        
        # adding tracks associated with each jet
        for num_track in range(event.JetGhostTrackN[indv_jet]):
            track_idx = event.JetGhostTrackIdx[num_track]
            nodes.append(get_track_info(track_idx))
            
        # add towers associated with each jet
        for num_tower in range(event.JetTowerN[indv_jet]):
            tower_idx = num_tower+prev_num_tower
            nodes.append(get_tower_info(tower_idx))

        prev_num_tower += event.JetTowerN[indv_jet]
    
    nodes = np.array(nodes, dtype=np.float32)
    n_nodes = nodes.shape[0]
    
    if debug:
        #print(np.array(event.TrackPt).shape)
        print(n_nodes)
        print(nodes)

    # edges
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": None,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([0], dtype=np.float32)
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": None,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array(global_attr, dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    
    return [(input_graph, target_graph)]

def read(filename):
    import ROOT
    chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
    chain.Add(filename)
    n_entries = chain.GetEntries()
    print("Total {:,} Events".format(n_entries))

    for ientry in range(n_entries):
        chain.GetEntry(ientry)
        yield chain


class DiTauMassDataset(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph

    def _num_evts(self, filename):
        import ROOT
        chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
        chain.Add(filename)
        n_entries = chain.GetEntries()
        return n_entries