import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.utils import calc_dphi, load_yaml
from root_gnn.src.datasets.base import DataSet
from root_gnn.src.datasets import hetero_graphs
from root_gnn import utils

import ROOT
from ROOT import TChain, AddressOf, std
from array import array

# from graph_nets import utils_tf
from root_gnn.src.datasets import graph
import subprocess

tower_lim=None
track_lim=None
cutoff=False

import ROOT
from ROOT import TChain, AddressOf, std
from array import array

tower_lim=None
track_lim=None
cutoff=False

tree_name = "output"

def make_graph(chain, debug=False, connectivity="disconnected"):
    isTau = 0
    track_idx = 0
    tower_idx = 0
    graph_list = []

    scale_factors = np.array([1.0,1.0,1.0,1.0/3.0,1.0/math.pi,1.0,1.0],dtype=np.float32)
    
    
    for ijet in range(chain.nJets):
        
        # Filtering out reco jets
        if chain.JetPt[ijet] < 30 or abs(chain.JetEta[ijet]) >= 3:
            continue
            
        # Match jet to truth jet that minimizes angular distance
        split_point = 0
        nodes = []
        tower_nodes = []
        track_nodes = []
        min_index = 0
        if chain.nTruthJets > 0:
            min_dR = math.sqrt(calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[0])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[0])**2)
        for itruth in range(chain.nTruthJets):
            dR = math.sqrt(calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[itruth])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[itruth])**2)
            if dR < min_dR:
                min_dR = dR
                min_index = itruth
        if chain.nTruthJets > 0 and min_dR < 0.4:
            isTau = chain.TruthJetIsTautagged[min_index]
        else:
            isTau = 0

        # Nodes
        for itower in range(chain.JetTowerN[ijet]):
            if(not cutoff or chain.JetTowerEt[tower_idx] >= 1.0):
                tower_nodes.append([0.0, math.log10(chain.JetPt[ijet]),\
                              math.log10(chain.JetTowerEt[tower_idx]),\
                              chain.TowerEta[tower_idx],\
                              chain.TowerPhi[tower_idx],\
                              0.0,\
                              0.0])

            tower_idx += 1
        
        tower_nodes.sort(reverse=True)
        if tower_lim != None:
            tower_nodes = tower_nodes[0:min(len(tower_nodes),tower_lim)]

        split_point = len(tower_nodes)
        
        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            if(not cutoff or chain.TrackPt[ghost_track_idx] >= 1.0):
                theta = 2*math.atan(-math.exp(chain.TrackEta[ghost_track_idx]))
                z0 = math.log10(10e-3+math.fabs(chain.TrackZ0[ghost_track_idx]*math.sin(theta)))
                d0 = math.log10(10e-3+math.fabs(chain.TrackD0[ghost_track_idx]))
                track_nodes.append([1.0, math.log10(chain.JetPt[ijet]),\
                              math.log10(chain.TrackPt[ghost_track_idx]),\
                              chain.TrackEta[ghost_track_idx],\
                              chain.TrackPhi[ghost_track_idx],\
                              z0,\
                              d0])
            track_idx+=1

        track_nodes.sort(reverse=True)
        if track_lim != None:
            track_nodes = track_nodes[0:min(len(track_nodes),track_lim)]

        nodes = np.array(tower_nodes + track_nodes,dtype=np.float32)
        node_type = np.array([0] * len(tower_nodes) + [1] * len(track_nodes), dtype=np.int32)

        n_nodes = len(nodes)
        if n_nodes < 1:
            continue
        
        
        if debug:
            #print(nodes.shape)
            #print(n_nodes)
            #print(nodes)
            pass
     
        # edges
        if connectivity == 'disconnected':
            all_edges = list(itertools.combinations(range(split_point), 2)) + list(itertools.combinations(range(split_point, n_nodes), 2))
        elif connectivity == 'KNN':
            nbrs = NearestNeighbors(n_neighbors=3).fit(nodes)
            distances, indices = nbrs.kneighbors(nodes)
            all_edges = indices
        else:
            all_edges = list(itertools.combinations(range(n_nodes), 2))
        senders = np.array([x[0] for x in all_edges])
        receivers = np.array([x[1] for x in all_edges])
        n_edges = len(all_edges)
        
            
        edges = []
        edge_type = []
        zeros = np.array([0.0], dtype=np.float32)
        
        Delta = lambda ya, yb, phiA, phiB: np.sqrt((ya-yb)**2+(phiA-phiB)**2)
        kT = lambda ptA, ptB, delta: min(ptA, ptB) * delta
        Z = lambda ptA, ptB: min(ptA, ptB) / (ptA + ptB)
        M2 = lambda Pt1, Pt2, eta1, eta2, phi1, phi2: 2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(phi1-phi2))
        
        
        for e in all_edges:
            v1, v2 = nodes[e[0]], nodes[e[1]]
            delta = Delta(v1[2], v2[2], v1[3], v2[3])
            kt = kT(10**v1[1], 10**v2[1], delta)
            z = Z(10**v1[1], 10**v2[1])
            m2 = M2(10**v1[1], 10**v2[1], v1[2], v2[2], v1[3], v2[3])
            if m2 <= 0:
                m2 = 1e-10
            edges.append([np.log(delta), np.log(kt), np.log(z), np.log(m2)])
            if node_type[e[0]] > 0.5:
                if node_type[e[1]] > 0.5: # track-track
                    edge_type.append([0, 0, 1])
                else: #track-cluster should not be possible
                    raise("Error: Tracks and clusters out of order")
            elif node_type[e[1]] > 0.5: # cluster-track
                edge_type.append([0,1,0])
            else: # cluster-cluster
                edge_type.append([1,0,0])
        
        nodes = np.array(nodes,dtype=np.float32)*scale_factors
        edges = np.array(edges, dtype=np.float32)
        if n_edges < 1:
            edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
        

        input_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "node_types": node_type,
            "edges": edges,
            "edge_types": edge_type,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([n_nodes], dtype=np.float32)
        }
        target_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "node_types": node_type,
            "edges": edges,
            "edge_types": edge_type,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([1. if isTau else 0.],dtype=np.float32)
        }
        input_graph = hetero_graphs.data_dicts_to_hetero_graphs_tuple([input_datadict])
        target_graph = hetero_graphs.data_dicts_to_hetero_graphs_tuple([target_datadict])
        graph_list.append((input_graph, target_graph))

    if len(graph_list) == 0:
        return [(None, None)]
    
    return graph_list


def read(filename, start_entry, nentries):
    import ROOT
    chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
    chain.Add(filename)
    tot_entries = chain.GetEntries()
    nentries = nentries if (start_entry + nentries) <= tot_entries\
        else tot_entries - start_entry
    #print("Total {:,} Events".format(nentries))
    for ientry in range(nentries):
        chain.GetEntry(ientry + start_entry)
        yield chain

class HetTauIdentificationDataset(DataSet):
    """
    Tau Identification dataset with heterogeneous nodes. The graph can be set to be fully-connected, fully-connected only among the same type of nodes (denoted as 'disconnected'), or KNN based. 
    Each node in the graph is a vector of length 6, with components:
        * Pt of the jet
        * Et of tower/Pt of track
        * Eta of tower/track
        * Phi of tower/track
        * d0 of track, 0 if tower
        * z0 of track, 0 if tower
    Each edge contains 4 edge features, delta, kT, z, mass square
    """
    def __init__(self,
                 use_cutoff=False,\
                 track_limit=None,\
                 tower_limit=None):
        super().__init__()
        self.read = read
        self.make_graph = make_graph
        tower_lim=tower_limit
        track_lim=track_limit
        if use_cutoff:
            cutoff=True

    def set_config_file(self,config):
        config = load_yaml(config)
        self.tower_lim = config.get('tower_limit',None)
        self.track_lim = config.get('track_limit',None)
        self.use_cutoff = config.get('use_cutoff',False)

    def _num_evts(self, filename):
        import ROOT
        chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
        chain.Add(filename)
        n_entries = chain.GetEntries()
        return n_entries

