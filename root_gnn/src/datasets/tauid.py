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

            ## <NOTE, for heteregenous nodes, do we want to remove the padding zeros?>
            for _ in range(chain.JetTowerN[ijet]):
                if(not self.use_cutoff or chain.JetTowerEt[tower_idx] >= 1.0):
                    tower_nodes.append([
                        chain.JetPt[ijet],
                        chain.JetTowerEt[tower_idx],
                        chain.TowerEta[tower_idx],
                        chain.TowerPhi[tower_idx],
                        0.0, ## d0
                        0.0  ## z0
                        ])

                tower_idx += 1
            
            tower_nodes.sort(reverse=True)
            if self.tower_limit != None:
                tower_nodes = tower_nodes[0:min(len(tower_nodes),self.tower_limit)]

            split_point = len(tower_nodes)
            
            for _ in range(chain.JetGhostTrackN[ijet]):
                ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
                if(not self.use_cutoff or chain.TrackPt[ghost_track_idx] >= 1.0):

                    theta = 2*math.atan(-math.exp(chain.TrackEta[ghost_track_idx]))
                    z0 = math.log10(10e-3+math.fabs(chain.TrackZ0[ghost_track_idx]*math.sin(theta)))
                    d0 = math.log10(10e-3+math.fabs(chain.TrackD0[ghost_track_idx]))
                    track_nodes.append([
                        chain.JetPt[ijet],
                        chain.TrackPt[ghost_track_idx],
                        chain.TrackEta[ghost_track_idx],
                        chain.TrackPhi[ghost_track_idx],
                        z0,
                        d0])
                track_idx+=1

            track_nodes.sort(reverse=True)
            if self.track_limit != None:
                track_nodes = track_nodes[0:min(len(track_nodes),self.track_limit)]

            nodes = np.array(tower_nodes + track_nodes,dtype=np.float32)

            n_nodes = len(nodes)
            if n_nodes < 1:
                continue
            
            
            if debug:
                print(nodes.shape)
                print(n_nodes)
                print(nodes)
        
            # edges
            if connectivity == 'disconnected':
                all_edges = list(itertools.combinations(range(split_point), 2)) +\
                    list(itertools.combinations(range(split_point, n_nodes), 2))
            elif connectivity == 'KNN':
                nbrs = NearestNeighbors(n_neighbors=3).fit(nodes)
                _, indices = nbrs.kneighbors(nodes)
                all_edges = indices
            else:
                all_edges = list(itertools.combinations(range(n_nodes), 2))

            all_edges = np.array(all_edges, dtype=np.int32)
            senders = all_edges[:, 0]
            receivers = all_edges[:, 1]
            n_edges = all_edges.shape[0]
            
                
            edges = []
            
            Delta = lambda ya, yb, phiA, phiB: np.sqrt((ya-yb)**2+(phiA-phiB)**2)
            kT = lambda ptA, ptB, delta: np.minimum(ptA, ptB) * delta
            Z = lambda ptA, ptB: np.minimum(ptA, ptB) / (ptA + ptB)
            M2 = lambda Pt1, Pt2, eta1, eta2, phi1, phi2: 2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(phi1-phi2))
            
            delta = Delta(nodes[senders,2], nodes[receivers,2], nodes[senders,3], nodes[receivers,3])
            kt = kT(nodes[senders,1], nodes[receivers,1], delta)
            z = Z(nodes[senders,1], nodes[receivers,1])
            m2 = M2(
                nodes[senders,1], nodes[receivers,1],
                nodes[senders,2], nodes[receivers,2],
                nodes[senders,3], nodes[receivers,3])
            
            edges = np.stack([np.log(delta), np.log(kt), np.log(z), np.log(m2)], axis=1)

            ## take log10 of pT of jets and pT of tracks
            nodes[: 0:2] = np.log10(nodes[: 0:2])
            nodes = np.array(nodes,dtype=np.float32)*scale_factors
            edges = np.array(edges, dtype=np.float32)

            if n_edges < 1:
                edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
            

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