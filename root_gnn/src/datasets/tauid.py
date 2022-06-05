import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.utils import calc_dphi, load_yaml
from root_gnn.src.datasets.base import DataSet
from sklearn.neighbors import NearestNeighbors

import ROOT
from ROOT import TChain, AddressOf, std
from array import array

tower_lim=None
track_lim=None
cutoff=False

tree_name = "output"
def make_graph(chain, debug=False, connectivity=None):
    isTau = 0
    scale_factors = np.array([1.0e-3,1.0/3.0,1.0/math.pi],dtype=np.float32)
    track_idx = 0
    tower_idx = 0
    graph_list = []
    for ijet in range(chain.nJets):
        # Match jet to truth jet that minimizes angular distance
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

        for itower in range(chain.JetTowerN[ijet]):
            if(not cutoff or chain.JetTowerEt[tower_idx] >= 1.0):
                tower_nodes.append([chain.JetTowerEt[tower_idx],\
                              chain.TowerEta[tower_idx],\
                              chain.TowerPhi[tower_idx]])
            tower_idx += 1
        
        tower_nodes.sort(reverse=True)
        if tower_lim != None:
            tower_nodes = tower_nodes[0:min(len(tower_nodes),tower_lim)]

        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            if(not cutoff or chain.TrackPt[ghost_track_idx] >= 1.0):
                track_nodes.append([chain.TrackPt[ghost_track_idx],\
                              chain.TrackEta[ghost_track_idx],\
                              chain.TrackPhi[ghost_track_idx]])
            track_idx+=1
        
        track_nodes.sort(reverse=True)
        if track_lim != None:
            track_nodes = track_nodes[0:min(len(track_nodes),track_lim)]
        
        
        scale_factors = np.array([1.0e-3,1.0/3.0,1.0/math.pi],dtype=np.float32)
        nodes = np.array(tower_nodes + track_nodes,dtype=np.float32)*scale_factors
        n_nodes = len(nodes)
        if n_nodes < 1:
            continue
        nodes = np.array(nodes,dtype=np.float32)*scale_factors
        if debug:
            print(nodes.shape)
            print(n_nodes)
            print(nodes)
        
        # edges
        if connectivity == 'disconnected':
            all_edges = list(itertools.combinations(tower_nodes, 2)) + list(itertools.combinations(track_nodes, 2))
        elif connectivity == 'KNN':
            nbrs = NearestNeighbors(n_neighbors=3).fit(nodes)
            distances, indices = nbrs.kneighbors(nodes)
            all_edges = indices
        else:
            all_edges = list(itertools.combinations(range(n_nodes), 2))
        senders = np.array([x[0] for x in all_edges])
        receivers = np.array([x[1] for x in all_edges])
        n_edges = len(all_edges)
        
        if n_edges < 0:
            continue
            
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

        for ientry in range(nentries):
            chain.GetEntry(ientry + start_entry)
            yield chain

class TauIdentificationDataset(DataSet):
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

