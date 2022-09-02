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
        self.branches = [
            'nJets', 'JetPt', 'JetEta', 'JetPhi', 'JetTowerN', 'JetGhostTrackN',
            'nTruthJets', 'TruthJetPt', 'TruthJetEta', 'TruthJetPhi', 'TruthJetIsTautagged',
            'JetTowerEt', 'JetTowerEta', 'JetTowerPhi',
            'TrackPt', 'TrackEta', 'TrackPhi', 'TrackD0', 'TrackZ0',
            'JetGhostTrackIdx',
            'JetLeadingTrackFracP', 'JetTrackR', 'JetNumISOTracks',
            'JetMaxDRInCore', 'JetTrackMass'
        ]

    def _num_evts(self, filename):
        import uproot
        chain = uproot.open(f"{filename}:{tree_name}")
        n_entries = chain.num_entries
        return n_entries

    def read(self, filename, start_entry, nentries):
        import uproot
        tree = uproot.open(f"{filename}:{tree_name}")

        for batch in tree.iterate(self.branches, step_size=1, library="np"):
            yield batch

    
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
        num_jets = chain['nJets'][0]
        num_true_jets = chain['nTruthJets'][0]

        for ijet in range(num_jets):
            
            jet_eta = chain['JetEta'][0][ijet]
            jet_phi = chain['JetPhi'][0][ijet]
            jet_pt = chain['JetPt'][0][ijet]
            jet_ntracks = chain['JetGhostTrackN'][0][ijet]
            jet_ntowers = chain['JetTowerN'][0][ijet]

            ## jets cuts: pT > 30 GeV and |eta| < 2.5
            if jet_pt < 30 or abs(jet_eta) > 2.5:
                tower_idx += jet_ntowers
                track_idx += jet_ntracks
                continue

            # Match the reco-level jet to a truth-level jet that minimizes angular distance
            # to see if it is a true tau jet, i.e. signal.
            isTau = 0
            min_index = -1
            min_dR = 9999
            if num_true_jets > 0:
                for itruth in range(num_true_jets):
                    truth_jet_eta = chain['TruthJetEta'][0][itruth]
                    truth_jet_phi = chain['TruthJetPhi'][0][itruth]
                    dR = math.sqrt(
                        calc_dphi(jet_phi, truth_jet_phi)**2 \
                        + (jet_eta- truth_jet_eta)**2)
                    if dR < min_dR:
                        min_dR = dR
                        min_index = itruth

            if min_index >= 0 and min_dR < 0.4:
                isTau = chain['TruthJetIsTautagged'][0][min_index]
            else:
                isTau = 0

            ## only use 1 prong and 3 prong decayed tau leptons
            isTau = 0 if (isTau != 1 and isTau != 3) else isTau
            if (signal != 0 and isTau == 0) \
                or (signal == 0 and isTau) \
                or (signal != None and signal != 0 and signal != 10 and signal != isTau):
                tower_idx += jet_ntowers
                track_idx += jet_ntracks
                continue

            split_point = 0 # an index where to split the graph into tracks only and towers only
            nodes = []
            tower_nodes = []
            track_nodes = []

            ### Nodes ###

            node_type_idx = 0 # 0 for towers, 1 for tracks
            for _ in range(jet_ntowers):
                tower_et = chain['JetTowerEt'][0][tower_idx]
                tower_eta = chain['JetTowerEta'][0][tower_idx]
                tower_phi = chain['JetTowerPhi'][0][tower_idx]
                if(not cutoff or tower_et >= 1.0):
                    if use_delta_angles:
                        deta = jet_eta - tower_eta
                        dphi = calc_dphi(jet_phi, tower_phi)
                    else:
                        deta = tower_eta/3
                        dphi = tower_phi/math.pi

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
            for _ in range(jet_ntracks):
                ghost_track_idx = chain['JetGhostTrackIdx'][0][track_idx]
                track_pt = chain['TrackPt'][0][ghost_track_idx]
                track_eta = chain['TrackEta'][0][ghost_track_idx]
                track_phi = chain['TrackPhi'][0][ghost_track_idx]
                track_d0 = chain['TrackD0'][0][ghost_track_idx]
                track_z0 = chain['TrackZ0'][0][ghost_track_idx]

                if(not cutoff or track_pt >= 1.0):
                    if use_delta_angles:
                        deta = jet_eta - track_eta
                        dphi = calc_dphi(jet_phi, track_phi)
                    else:
                        deta = track_eta/3
                        dphi = track_phi/math.pi

                    theta = 2*math.atan(-math.exp(track_eta))
                    z0 = math.log10(10e-3+math.fabs(track_z0*math.sin(theta)))
                    d0 = math.log10(10e-3+math.fabs(track_d0))
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
                        raise RuntimeError("Error: Tracks and clusters out of order")
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
                global_features += [chain['JetLeadingTrackFracP'][0][ijet],
                                    chain['JetTrackR'][0][ijet],
                                    chain['JetNumISOTracks'][0][ijet],
                                    chain['JetMaxDRInCore'][0][ijet],
                                    chain['JetTrackMass'][0][ijet]]
                
            
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