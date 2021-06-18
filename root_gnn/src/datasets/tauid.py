import numpy as np
import itertools

from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet


def make_graph(event, debug=False):
    # n_max_nodes = 60
    n_nodes = event.n_nodes
    #nodes = np.hstack((event.jet_pt, event.jet_eta, event.jet_phi, event.jet_e))
    #nodes = nodes.reshape( (n_nodes, 4) )
    nodes = np.array(event.nodes)
    if debug:
        #print(np.array(event.jet_pt).shape)
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
        "globals": np.array([1. if event.is_tau_jet else 0.],dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]

class JetInfo:
    def __init__(self, is_tau_jet: bool, nodes: list[list[float]], n_nodes: int):
        self.is_tau_jet = is_tau_jet
        self.n_nodes = n_nodes
        self.nodes = nodes

def read(filename):
    import ROOT
    from ROOT import TChain, AddressOf, std
    from array import array
    tree_name = "output"
    chain = TChain(tree_name,tree_name)
    chain.Add(filename)
    n_entries = chain.GetEntries()

#    chain.Scan("*")
    JetPhi = std.vector('float')()
    JetEta = std.vector('float')()

    nTruthJets = array('i',[0])
    nJets = array('i',[0])
    TruthJetIsTautagged = std.vector('int')()
    TruthJetEta = std.vector('float')()
    TruthJetPhi = std.vector('float')()

    JetTowerN = std.vector('int')()
    JetTowerEt = std.vector('float')()
    JetTowerEta = std.vector('float')()
    JetTowerPhi = std.vector('float')()

    TrackPt = std.vector('float')()
    TrackEta = std.vector('float')()
    TrackPhi = std.vector('float')()

    JetGhostTrackN = std.vector('int')()
    JetGhostTrackIdx = std.vector('int')()

    chain.SetBranchAddress("JetPhi",JetPhi)
    chain.SetBranchAddress("JetEta",JetEta)

    chain.SetBranchAddress("nTruthJets",nTruthJets)
    chain.SetBranchAddress("nJets",nJets)
    chain.SetBranchAddress("TruthJetIsTautagged",TruthJetIsTautagged)
    chain.SetBranchAddress("TruthJetEta",TruthJetEta)
    chain.SetBranchAddress("TruthJetPhi",TruthJetPhi)

    chain.SetBranchAddress("JetTowerN",JetTowerN)
    chain.SetBranchAddress("JetTowerEt",JetTowerEt)
    chain.SetBranchAddress("JetTowerEta",JetTowerEta)
    chain.SetBranchAddress("JetTowerPhi",JetTowerPhi)

    chain.SetBranchAddress("TrackPt",TrackPt)
    chain.SetBranchAddress("TrackEta",TrackEta)
    chain.SetBranchAddress("TrackPhi",TrackPhi)

    chain.SetBranchAddress("JetGhostTrackN",JetGhostTrackN)
    chain.SetBranchAddress("JetGhostTrackIdx",JetGhostTrackIdx)

    isTau = 0
    for ientry in range(n_entries):
        chain.GetEntry(ientry)
        track_idx = 0
        tower_idx = 0
        #print(nJets[0],nTruthJets[0],len(JetPhi),len(TruthJetPhi))
        for ijet in range(nJets[0]):
            # Match jet to truth jet that minimizes angular distance
            nodes = []
            min_index = 0
            if nTruthJets[0] > 0:
                min_dR2 = (JetPhi[ijet]-TruthJetPhi[0])**2 + (JetEta[ijet]-TruthJetEta[0])**2
            for itruth in range(nTruthJets[0]):
                dR2 = (JetPhi[ijet]-TruthJetPhi[itruth])**2 + (JetEta[ijet]-TruthJetEta[itruth])**2
                if dR2 < min_dR2:
                    min_dR2 = dR2
                    min_index = itruth
            if nTruthJets[0] > 0:
                isTau = TruthJetIsTautagged[min_index]
            else:
                isTau = 0 

            # find minimum angular distance between recon and truth jets
            # for each jet
            # check which index to use for jet
            # find the label and nodes from tracks and towers
            # yield a JetInfo object
            for itower in range(JetTowerN[ijet]):
                nodes.append([JetTowerEt[tower_idx],JetTowerEta[tower_idx],JetTowerPhi[tower_idx]])
                tower_idx += 1
                #generate tower list
       
            for itrack in range(JetGhostTrackN[ijet]):
                ghost_track_idx = JetGhostTrackIdx[track_idx]
                track_idx+=1
                nodes.append([TrackPt[ghost_track_idx],TrackEta[ghost_track_idx],TrackPhi[ghost_track_idx]])

            yield JetInfo(isTau,nodes,len(nodes))

class TauIdentificationDataSet(DataSet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read = read
        self.make_graph = make_graph

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from graph_nets import utils_np
from graph_nets import graphs

t = TauIdentificationDataSet()

def print_graphs_tuple(g, data=True):
    for field_name in graphs.ALL_FIELDS:
        per_replica_sample = getattr(g, field_name)
        if per_replica_sample is None:
            print(field_name, "EMPTY")
        else:
            print(field_name, "is with shape", per_replica_sample.shape)
            if data and  field_name != "edges":
                print(per_replica_sample)

for y in t.read("~/tauid_gnn/root_gnn/root_gnn/src/datasets/v0/Ntuple_ditau_processed.root"):
    g = make_graph(y,False)[0][0] 
    print_graphs_tuple(g,data=True)
    input()
