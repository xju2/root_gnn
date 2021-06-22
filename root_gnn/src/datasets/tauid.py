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
    def __init__(self, is_tau_jet: bool, nodes, n_nodes: int):
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
    print("Entries",n_entries,len(nJets),len(nTruthJets))

    for ientry in range(n_entries):
        chain.GetEntry(ientry)
        track_idx = 0
        tower_idx = 0
        print(nJets[0],nTruthJets[0],len(JetPhi),len(TruthJetPhi))

        #################################################
        # Match truth jets to the closest (in terms of angular distance)
        # reconstructed jet while ensuring each reconstructed jet is
        # matched only once
        #################################################
        used_buffer = [ False for x in range(nJets[0]) ]
        # Initialize jet labels to mean false. Iterate over truth jets to find
        # which ones need to be overriden with true labels
        is_tau_jet = [ 0 for x in range(nJets[0]) ]
        for itruth in range(nTruthJets[0]):
            min_index = 0
            if nJets[0] > 0:
                min_dR2 = (JetPhi[0]-TruthJetPhi[itruth])**2 + (JetEta[0]-TruthJetEta[itruth])**2
            for ijet in range(nJets[0]):
                dR2 = (JetPhi[ijet]-TruthJetPhi[itruth])**2 + (JetEta[ijet]-TruthJetEta[itruth])**2
                if dR2 < min_dR2 and not used_buffer[ijet]:
                    min_dR2 = dR2
                    min_index = ijet
            # Change label if reconstructed jet with index min_index is unused
            if nJets[0] > 0 and not used_buffer[min_index]:
                is_tau_jet[min_index] = TruthJetIsTautagged[itruth]
                used_buffer[min_index] = True
        #################################################
        # Retrieve the track and tower information for
        # each reconstructed jet and yield the information
        # needed to build the jet's graph representation
        #################################################
        for ijet in range(nJets[0]):
            nodes = []
            for itower in range(JetTowerN[ijet]):
                nodes.append([JetTowerEt[tower_idx],JetTowerEta[tower_idx],JetTowerPhi[tower_idx]])
                tower_idx += 1
                #generate tower list

            for itrack in range(JetGhostTrackN[ijet]): # towers associated with this jet
                ghost_track_idx = JetGhostTrackIdx[track_idx]
                nodes.append([TrackPt[ghost_track_idx],TrackEta[ghost_track_idx],TrackPhi[ghost_track_idx]])
                track_idx+=1

            yield JetInfo(is_tau_jet[ijet],nodes,len(nodes))

        # Print debug info
        #if nTruthJets[0] > 0 and nJets[0] > nTruthJets[0]:
        if 1:
            chain.Scan("nJets:nTruthJets:JetTowerN:JetTowerEt:JetTowerEta:JetTowerPhi:JetGhostTrackN:JetGhostTrackIdx:TrackPt:TrackEta:TrackPhi","","",50,ientry)
            chain.Scan("nJets:nTruthJets:TruthJetIsTautagged:TruthJetEta:TruthJetPhi:JetEta:JetPhi:JetTowerN:JetGhostTrackN","","",50,ientry)
            input()

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
    gin, gtarget = make_graph(y,False)[0]
    print("Input")
    print_graphs_tuple(gin,data=True)
    print("Target")
    print_graphs_tuple(gtarget,data=True)
    #Pause execution for user to read debug info
    input()
