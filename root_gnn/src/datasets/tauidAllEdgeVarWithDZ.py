import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

tree_name = "output"
def make_graph(chain, debug=False):
    
    def get_tower_info(idx):
        return [chain.JetTowerEt[idx], chain.JetTowerEta[idx], chain.JetTowerPhi[idx], 0, 0]
    
    def get_track_info(idx):
        return [chain.TrackPt[idx], chain.TrackEta[idx], chain.TrackPhi[idx], chain.TrackD0[idx], chain.TrackZ0[idx]]
        
    isTau = 0
    scale_factors = np.array([1.0e-3,1.0/6, 1.0/2/math.pi, 1, 1],dtype=np.float32)
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

        tower_nodes, track_nodes = [], []
        inode = 0
        for itower in range(chain.JetTowerN[ijet]):
            nodes.append(get_tower_info(tower_idx))
            tower_nodes.append(inode)
            inode += 1
            tower_idx += 1

        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            nodes.append(get_track_info(ghost_track_idx))
            track_nodes.append(inode)
            inode += 1
            track_idx+=1

        n_nodes = len(nodes)
        if n_nodes < 1:
            continue
        
        if debug:
            print(nodes.shape)
            print(n_nodes)
            print(nodes)

        # edges
        all_edges = list(itertools.combinations(tower_nodes, 2)) + list(itertools.combinations(track_nodes, 2))
        #print(all_edges)
        senders = np.array([x[0] for x in all_edges])
        receivers = np.array([x[1] for x in all_edges])
        n_edges = len(all_edges)
        edges = []
        zeros = np.array([0.0], dtype=np.float32)
        
        Delta = lambda ya, yb, phiA, phiB: np.sqrt((ya-yb)**2+(phiA-phiB)**2)
        kT = lambda ptA, ptB, delta: min(ptA, ptB) * delta
        Z = lambda ptA, ptB: min(ptA, ptB) / (ptA + ptB)
        M2 = lambda Pt1, Pt2, eta1, eta2, phi1, phi2: 2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(phi1-phi2))
        
        for e in all_edges:
            #print(e)
            v1, v2 = nodes[e[0]], nodes[e[1]]
            delta = Delta(v1[1], v2[1], v1[2], v2[2])
            kt = kT(v1[0], v2[0], delta)
            z = Z(v1[0], v2[0])
            m2 = M2(v1[0], v2[0], v1[1], v2[1], v1[2], v2[2])
            edges.append([np.log(delta), np.log(kt), np.log(z), np.log(m2)])
        edges = np.array(edges, dtype=np.float32)
        
        # Feature Scaling
        #nodes = (np.array(nodes,dtype=np.float32) - np.array([2., 0., 0.], dtype=np.float32))*scale_factors
        nodes = np.array(nodes, dtype=np.float32) * scale_factors
        
        n_nodes = len(nodes)
        if n_nodes < 1 or n_edges < 1:
            continue
        

        input_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([chain.JetEta[ijet], chain.JetPhi[ijet]], dtype=np.float32)
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

class tauidAllEdgeVarWithDZ(DataSet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read = read
        self.make_graph = make_graph
        
    def _num_evts(self, filename):
        import ROOT
        chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
        chain.Add(filename)
        n_entries = chain.GetEntries()
        return n_entries
