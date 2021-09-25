# %%
from types import DynamicClassAttribute
import numpy as np
import pandas as pd
import itertools
from typing import Optional

from graph_nets import utils_tf


from root_gnn.src.datasets.base import DataSet

# %%
tree_name = "output"
scales = np.array([100, 5, np.pi, 100, 5, np.pi, 100], dtype=np.float32)
node_scales = np.array([5, 3, np.pi], dtype=np.float32)
do_disconnect_jets = False

def make_graph(event, debug=False):
    # selecting events with at least one reco jets
    # and two true tau leptons
    if event.nJets < 1 or event.truthTauN != 2:
        return [(None, None)]
    
    # globals
    Pt1, Pt2 = event.truthTauEt[0], event.truthTauEt[1]
    eta1, eta2 = event.truthTauEta[0], event.truthTauEta[1]
    phi1, phi2 = event.truthTauPhi[0], event.truthTauPhi[1]
    ditau_inv_mass = np.sqrt(2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(phi1-phi2)))
    # global_attr = [Pt1, eta1, phi1, Pt2, eta2, phi2, ditau_inv_mass]
    global_attr = [ditau_inv_mass]
    
    global_attr = np.array(global_attr) / scales

    # nodes
    def get_track_info(idx):
        return [event.TrackPt[idx], event.TrackEta[idx], event.TrackPhi[idx]]

    def get_tower_info(idx):
        return [event.JetTowerEt[idx], event.JetTowerEta[idx], event.JetTowerPhi[idx]]

    
    nodes = []
    node_indx = []
    prev_num_track = 0
    prev_num_tower = 0
    inode = 0
    for indv_jet in range(event.nJets):
        
        # adding tracks associated with each jet
        for num_track in range(event.JetGhostTrackN[indv_jet]):
            track_idx = event.JetGhostTrackIdx[num_track+prev_num_track]
            nodes.append(get_track_info(track_idx))
            inode += 1
            
        # add towers associated with each jet
        for num_tower in range(event.JetTowerN[indv_jet]):
            nodes.append(get_tower_info(num_tower+prev_num_tower))
            inode += 1

        prev_num_tower += event.JetTowerN[indv_jet]
        
        node_indx.append(inode)
        
    nodes = np.array(nodes, dtype=np.float32) / node_scales
    n_nodes = nodes.shape[0]
    
    if debug:
        #print(np.array(event.TrackPt).shape)
        assert node_indx[-1] == n_nodes, 'Nodes Error'
        print(n_nodes)
        print(nodes)

    # edges
    if do_disconnect_jets:
        all_edges = []
        prev_node = 0
        for i in node_indx:
            all_edges += list(itertools.permutations(range(prev_node, i), 2))
            prev_node = i
    else:
        all_edges = list(itertools.permutations(range(n_nodes), 2))

    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([0], dtype=np.float32)
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array(global_attr, dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    
    if n_nodes < 1 or n_edges < 1:
        return [(None, None)]

    return [(input_graph, target_graph)]

def read(filename, start_entry, nentries):
    import ROOT
    chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
    chain.Add(filename)
    tot_entries = chain.GetEntries()
    nentries = nentries if (start_entry + nentries) <= tot_entries\
        else tot_entries - start_entry

    # print("Total {:,} Events".format(n_entries))
    chain.SetBranchStatus("*", 0)
    branches = [
        'nJets',
        'truthTauN', 'truthTauEt', 'truthTauEta', 'truthTauPhi',
        'TrackPt', 'TrackEta', 'TrackPhi',
        'JetTowerEt', 'JetTowerEta', 'JetTowerPhi',
        'JetGhostTrackN', 'JetGhostTrackIdx', 'JetTowerN'
    ]
    [chain.SetBranchStatus(brname, 1) for brname in branches]

    for ientry in range(nentries):
        chain.GetEntry(ientry + start_entry)
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

    def disconnect_jets(self):
        global do_disconnect_jets
        do_disconnect_jets = False