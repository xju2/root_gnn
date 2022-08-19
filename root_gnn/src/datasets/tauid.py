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
    def __init__(self, with_padding=False, name="TauIdentificationDataset"):
        super(TauIdentificationDataset, self).__init__(with_padding, name)

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

    
    def make_graph(self, chain, debug=False,
                    connectivity="disconnected",
                    signal=None,
                    with_edge_features=False, with_node_type=True, 
                    with_hlv_features=False,
                    use_delta_angles=True, ## angles differences between jets and tracks and towers
                    tower_lim=None, track_lim=None, cutoff=False,
                    use_jetPt=True, use_jetVar=True,
                    knn_nbrs=3, # number of nearest neighbors to use for KNN
                    knn_radius=1.0, # radius for KNN
                   ):
        

        graph_list = []

        track_idx = 0
        tower_idx = 0        

        for ijet in range(chain.nJets):

            ## jets cuts: pT > 30 GeV and |eta| < 2.5
            if chain.JetPt[ijet] < 30 or abs(chain.JetEta[ijet]) > 2.5:
                tower_idx += chain.JetTowerN[ijet]
                track_idx += chain.JetGhostTrackN[ijet]
                continue

            # Match the reco-level jet to a truth-level jet that minimizes angular distance
            # to see if it is a true tau jet, i.e. signal.
            isTau = 0
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

            ## only use 1 prong and 3 prong decayed tau leptons
            isTau = 0 if (isTau != 1 and isTau != 3) else isTau
            if (signal != 0 and isTau == 0) \
                or (signal == 0 and isTau) \
                or (signal != None and signal != 0 and signal != 10 and signal != isTau):
                tower_idx += chain.JetTowerN[ijet]
                track_idx += chain.JetGhostTrackN[ijet]
                continue

            split_point = 0 # an index where to split the graph into tracks only and towers only
            nodes = []
            tower_nodes = []
            track_nodes = []

            ### Nodes ###
            jet_eta = chain.JetEta[ijet]
            jet_phi = chain.JetPhi[ijet]
            jet_pt = chain.JetPt[ijet]

            node_type_idx = 0 # 0 for towers, 1 for tracks
            for itower in range(chain.JetTowerN[ijet]):
                tower_et = chain.JetTowerEt[tower_idx]
                if(not cutoff or tower_et >= 1.0):
                    if use_delta_angles:
                        deta = jet_eta - chain.JetTowerEta[tower_idx]
                        dphi = calc_dphi(jet_phi, chain.JetTowerPhi[tower_idx])
                    else:
                        deta = chain.TowerEta[tower_idx]/3
                        dphi = chain.TowerPhi[tower_idx]/math.pi

                    tower_feature = [node_type_idx, math.log10(jet_pt), math.log10(tower_et), deta, dphi, 0.0, 0.0]

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
            
            node_type_idx = 1 # 0 for towers, 1 for tracks
            for itrack in range(chain.JetGhostTrackN[ijet]):
                ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
                track_pt = chain.TrackPt[ghost_track_idx]

                if(not cutoff or track_pt >= 1.0):
                    if use_delta_angles:
                        deta = chain.JetEta[ijet]-chain.TrackEta[ghost_track_idx]
                        dphi = calc_dphi(chain.JetPhi[ijet],chain.TrackPhi[ghost_track_idx])
                    else:
                        deta = chain.TrackEta[ghost_track_idx]/3
                        dphi = chain.TrackPhi[ghost_track_idx]/math.pi

                    theta = 2*math.atan(-math.exp(chain.TrackEta[ghost_track_idx]))
                    z0 = math.log10(10e-3+math.fabs(chain.TrackZ0[ghost_track_idx]*math.sin(theta)))
                    d0 = math.log10(10e-3+math.fabs(chain.TrackD0[ghost_track_idx]))
                    track_feature = [node_type_idx, math.log10(jet_pt), math.log10(track_pt), deta, dphi, z0, d0]


                    if not use_jetPt:
                        track_feature = [track_feature[0]] + track_feature[2:]
                    if not with_node_type:
                        track_feature = track_feature[1:]
                    track_nodes.append(track_feature)
                track_idx+=1

            track_nodes.sort(reverse=True)
            if track_lim != None:
                track_nodes = track_nodes[0:min(len(track_nodes),track_lim)]

            nodes = np.array(tower_nodes + track_nodes, dtype=np.float32)
            node_type = np.array([0] * len(tower_nodes) + [1] * len(track_nodes), dtype=np.int8)

            n_nodes = len(nodes)
            

        
            ### edges ###
            if connectivity == 'disconnected':
                ## In the graph, tracks only connect to tracks, towers only connect to towers
                all_edges = list(itertools.combinations(range(split_point), 2)) \
                    + list(itertools.combinations(range(split_point, n_nodes), 2))
                senders = np.array([x[0] for x in all_edges])
                receivers = np.array([x[1] for x in all_edges])
            elif connectivity == 'KNN':
                ## nodes are connected to their nearest neighbors
                ## the last two columns of the nodes array are the eta and phi of the nodes
                nbrs = NearestNeighbors(n_neighbors=knn_nbrs, radius=knn_radius).fit(nodes[:, -2:])
                senders, receivers = nbrs.kneighbors_graph(nodes[:, -2:]).nonzero()
                all_edges = zip(senders, receivers)
            else:
                ## fully connected graph
                all_edges = list(itertools.combinations(range(n_nodes), 2))
                senders = np.array([x[0] for x in all_edges])
                receivers = np.array([x[1] for x in all_edges])

            n_edges = len(all_edges)
                
            ## edge features always include the edge type
            edges = []
            Delta = lambda ya, yb, phiA, phiB: np.sqrt((ya-yb)**2+(phiA-phiB)**2)
            kT = lambda ptA, ptB, delta: min(ptA, ptB) * delta
            Z_fn = lambda ptA, ptB: min(ptA, ptB) / (ptA + ptB)
            M2_fn = lambda Pt1, Pt2, eta1, eta2, phi1, phi2: 2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(calc_dphi(phi1, phi2)))
            
            
            edge_starting = 2 if with_node_type else 1
            for e in all_edges:
                v1, v2 = nodes[e[0]], nodes[e[1]]
                edge_feature = []
                if with_edge_features:
                    pt1, eta1, phi1 = v1[edge_starting:edge_starting+3]
                    pt2, eta2, phi2 = v2[edge_starting:edge_starting+3]
                    delta = Delta(eta1, eta2, phi1, phi2)
                    kt = kT(10**pt1, 10**pt2, delta)
                    z = Z_fn(10**pt1, 10**pt2)
                    m2 = M2_fn(10**pt1, 10**pt2, eta1, eta2, phi1, phi2)
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
                global_features = [math.log10(jet_pt), jet_eta/3, jet_phi/math.pi]
            else:
                global_features = [0.]

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