from platform import node
import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.utils import calc_dphi, load_yaml
from root_gnn.src.datasets.base import DataSet
from sklearn.neighbors import NearestNeighbors

tree_name = "output"

class TauIdentificationDataset(DataSet):
    """
    Tau Identification dataset with heterogeneous nodes.
    The graph can be set to be fully-connected, 
        fully-connected only among the same type of nodes (denoted as 'disconnected'),
        or KNN based. 
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
                 use_cutoff: bool =False,
                 track_limit: int = None,
                 tower_limit: int = None, *args, **kwargs):

        self.use_cutoff = use_cutoff
        self.track_limit = track_limit
        self.tower_limit = tower_limit
        super(TauIdentificationDataset, self).__init__(*args, **kwargs)

    def set_config_file(self,config):
        config = load_yaml(config)
        self.tower_limit = config.get('tower_limit',None)
        self.track_limit = config.get('track_limit',None)
        self.use_cutoff = config.get('use_cutoff',False)

    def _num_evts(self, filename):
        import ROOT
        chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
        chain.Add(filename)
        n_entries = chain.GetEntries()
        return n_entries

    def read(self, filename, start_entry, nentries):
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

    
    def make_graph(self, chain, debug=False, connectivity="disconnected"):
        isTau = 0
        track_idx = 0
        tower_idx = 0
        graph_list = []

        scale_factors = np.array([1.0,1.0,1.0/3.0,1.0/math.pi,1.0,1.0],dtype=np.float32)
        
        
        for ijet in range(chain.nJets):
            # Match jet to truth jet that minimizes angular distance
            split_point = 0
            nodes = []
            tower_nodes = []
            track_nodes = []
            min_index = -1
            if chain.nTruthJets > 0:
                min_dR = 9999
                for itruth in range(chain.nTruthJets):
                    dR = math.sqrt(
                        calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[itruth])**2 \
                      + (chain.JetEta[ijet]-chain.TruthJetEta[itruth])**2)
                    if dR < min_dR:
                        min_dR = dR
                        min_index = itruth


            if min_index >= 0 and min_dR < 0.4:
                isTau = chain.TruthJetIsTautagged[min_index]
            else:
                isTau = 0

        ### Nodes ###
        for itower in range(chain.JetTowerN[ijet]):
            if(not cutoff or chain.JetTowerEt[tower_idx] >= 1.0):
                if use_delta_angles:
                    deta = chain.JetEta[ijet]-chain.JetTowerEta[tower_idx]
                    dphi = calc_dphi(chain.JetPhi[ijet],chain.JetTowerPhi[tower_idx])
                    tower_feature = [0.0, math.log10(chain.JetPt[ijet]),
                                     math.log10(chain.JetTowerEt[tower_idx]),
                                     math.fabs(deta),
                                     math.fabs(dphi),
                                     0.0,
                                     0.0]
                else:
                    tower_feature = [0.0, math.log10(chain.JetPt[ijet]),\
                                     math.log10(chain.JetTowerEt[tower_idx]),\
                                     chain.TowerEta[tower_idx]/3,\
                                     chain.TowerPhi[tower_idx]/math.pi,\
                                     0.0,
                                     0.0]
                if not use_jetPt:
                    tower_feature = [tower_feature[0]] + tower_feature[2:]
                if not with_node_type:
                    tower_feature = tower_feature[1:]
                
                tower_nodes.append(tower_feature)

            tower_idx += 1
        
        tower_nodes.sort(reverse=True)
        if tower_lim != None:
            tower_nodes = tower_nodes[0:min(len(tower_nodes),tower_lim)]

        split_point = len(tower_nodes)
        
        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            if(not cutoff or chain.TrackPt[ghost_track_idx] >= 1.0):
                deta = chain.JetEta[ijet]-chain.TrackEta[ghost_track_idx]
                dphi = calc_dphi(chain.JetPhi[ijet],chain.TrackPhi[ghost_track_idx])
                theta = 2*math.atan(-math.exp(chain.TrackEta[ghost_track_idx]))
                z0 = math.log10(10e-3+math.fabs(chain.TrackZ0[ghost_track_idx]*math.sin(theta)))
                d0 = math.log10(10e-3+math.fabs(chain.TrackD0[ghost_track_idx]))
                
                if use_delta_angles:
                    track_feature = [1.0, math.log10(chain.JetPt[ijet]),
                                     math.log10(chain.TrackPt[ghost_track_idx]),
                                     math.fabs(deta),
                                     math.fabs(dphi),
                                     z0,
                                     d0]
                else:
                    track_feature = [1.0, math.log10(chain.JetPt[ijet]),\
                                     math.log10(chain.TrackPt[ghost_track_idx]),\
                                     chain.TrackEta[ghost_track_idx]/3,\
                                     chain.TrackPhi[ghost_track_idx]/math.pi,\
                                     z0,\
                                     d0]
                if not use_jetPt:
                    track_feature = [track_feature[0]] + track_feature[2:]
                if not with_node_type:
                    track_feature = track_feature[1:]
                track_nodes.append(track_feature)
            track_idx+=1

        track_nodes.sort(reverse=True)
        if track_lim != None:
            track_nodes = track_nodes[0:min(len(track_nodes),track_lim)]

        nodes = np.array(tower_nodes + track_nodes,dtype=np.float32)
        node_type = np.array([0] * len(tower_nodes) + [1] * len(track_nodes), dtype=np.int8)

        n_nodes = len(nodes)
        if n_nodes < 1:
            continue
        
        
        if debug:
            #print(nodes.shape)
            #print(n_nodes)
            #print(nodes)
            pass
     
        ### edges ###
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
        Delta = lambda ya, yb, phiA, phiB: np.sqrt((ya-yb)**2+(phiA-phiB)**2)
        kT = lambda ptA, ptB, delta: min(ptA, ptB) * delta
        Z = lambda ptA, ptB: min(ptA, ptB) / (ptA + ptB)
        M2 = lambda Pt1, Pt2, eta1, eta2, phi1, phi2: 2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(calc_dphi(phi1, phi2)))
        
        
        edge_starting = 2 if with_node_type else 1
        for e in all_edges:
            v1, v2 = nodes[e[0]], nodes[e[1]]
            edge_feature = []
            if with_edge_features:
                
                pt1, eta1, phi1 = v1[edge_starting:edge_starting+3]
                pt2, eta2, phi2 = v2[edge_starting:edge_starting+3]
                delta = Delta(eta1, eta2, phi1, phi2)
                kt = kT(10**pt1, 10**pt2, delta)
                z = Z(10**pt1, 10**pt2)
                m2 = M2(10**pt1, 10**pt2, eta1, eta2, phi1, phi2)
                if m2 <= 0:
                    m2 = 1e-10
                edge_feature = [np.log(delta), np.log(kt), np.log(z), np.log(m2)]
            # Determine Edge Types
            if node_type[e[0]] > 0.5:
                if node_type[e[1]] > 0.5: # track-track
                    edge_type = [0.0, 0.0, 1.0]
                else: #track-cluster should not be possible
                    raise("Error: Tracks and clusters out of order")
            elif node_type[e[1]] > 0.5: # cluster-track
                edge_type = [0.0, 1.0, 0.0]
            else: # cluster-cluster
                edge_type = [1.0, 0.0, 0.0]
            
            edges.append(edge_type + edge_feature)
            
            
        ### Globals ###
        if use_jetVar:
            global_features = [chain.JetPt[ijet], chain.JetEta[ijet], chain.JetPhi[ijet]]
        else:
            global_features = [n_nodes]
        if with_hlv_features:
            global_features += [chain.JetLeadingTrackFracP[ijet],
                                chain.JetTrackR[ijet],
                                chain.JetNumISOTracks[ijet],
                                chain.JetMaxDRInCore[ijet],
                                chain.JetTrackMass[ijet]]
            
        
        nodes = np.array(nodes,dtype=np.float32)
        edges = np.array(edges, dtype=np.float32)
        globals = np.array(global_features, dtype=np.float32)  
        if n_edges < 1:
            edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
        

        input_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": globals
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
    #print("Total {:,} Events".format(nentries))
    for ientry in range(nentries):
        chain.GetEntry(ientry + start_entry)
        yield chain

class TauIdentificationDataset(DataSet):
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

