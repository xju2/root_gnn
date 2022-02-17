import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

import ROOT
from ROOT import TChain, AddressOf, std
from array import array

def make_graph(chain, debug=False):
    isTau = 0
    scale_factors = np.array([1.0e-3,1.0/3.0,1.0/math.pi],dtype=np.float32)
    track_idx = 0
    tower_idx = 0
    graph_list = []
    for ijet in range(chain.nJets):
        # Match jet to truth jet that minimizes angular distance
        nodes = []
        min_index = 0
        if chain.nTruthJets > 0:
            min_dR = math.sqrt((chain.JetPhi[ijet]-chain.TruthJetPhi[0])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[0])**2)
        for itruth in range(chain.nTruthJets):
            dR = math.sqrt((chain.JetPhi[ijet]-chain.TruthJetPhi[itruth])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[itruth])**2)
            if dR < min_dR:
                min_dR = dR
                min_index = itruth
        if chain.nTruthJets > 0 and min_dR < 0.4:
            isTau = chain.TruthJetIsTautagged[min_index]
        else:
            isTau = 0

        for itower in range(chain.JetTowerN[ijet]):
            nodes.append([chain.JetTowerEt[tower_idx],chain.JetTowerEta[tower_idx],chain.JetTowerPhi[tower_idx]])
            tower_idx += 1

        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            nodes.append([chain.TrackPt[ghost_track_idx],chain.TrackEta[ghost_track_idx],chain.TrackPhi[ghost_track_idx]])
            track_idx+=1


        n_nodes = len(nodes)
        if n_nodes<1:
            continue
        nodes = np.array(nodes,dtype=np.float32)*scale_factors
        if debug:
            print(nodes.shape)
            print(n_nodes)
            print(nodes)
        
        # edges
        all_edges = list(itertools.combinations(range(n_nodes), 2))
        senders = np.array([x[0] for x in all_edges])
        receivers = np.array([x[1] for x in all_edges])
        n_edges = len(all_edges)
        edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
        zeros = np.array([0.0], dtype=np.float32)

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
            "globals": np.array([1. if isTau else 0.],dtype=np.float32)
        }
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        graph_list.append((input_graph, target_graph))
    if not len(graph_list):
        return [(None,None)]
    else:
        return graph_list

def read(filename):
    import ROOT
    tree_name = "output"
    chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
    chain.Add(filename)
    n_entries = chain.GetEntries()
    print("Total {:,} Events".format(n_entries))

    for ientry in range(n_entries):
        chain.GetEntry(ientry)
        yield chain

class TauIdentificationDataset(DataSet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read = read
        self.make_graph = make_graph
