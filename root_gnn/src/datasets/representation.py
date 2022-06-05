import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet
from root_gnn.utils import calc_dphi

#class for generating graphs for use in representation learning, this implementation centers the jets at the jet axis when making the graph
def make_graph(chain, debug=False):
    scale_factors = np.array([1.0e-3,1.0/3.0,1.0/math.pi],dtype=np.float32) #using same scale factors Landon found were most useful
    track_idx = 0
    tower_idx = 0
    graph_list = []
    for ijet in range(chain.nJets):
        
        nodes = []
        min_index = 0
    #each node is a tower/track, node info is pt/Et, Eta, and Phi (centered at jet axis)
        for itower in range(chain.JetTowerN[ijet]):
            nodes.append([chain.JetTowerEt[tower_idx],chain.JetTowerEta[tower_idx] - chain.JetEta[ijet], calc_dphi(chain.JetTowerPhi[tower_idx], chain.JetPhi[ijet])])
            tower_idx += 1

        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            nodes.append([chain.TrackPt[ghost_track_idx],chain.TrackEta[ghost_track_idx]-chain.JetEta[ijet],calc_dphi(chain.TrackPhi[ghost_track_idx],chain.JetPhi[ijet])])
            track_idx+=1

        n_nodes = len(nodes)
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
        
        #globals 
        jet_data = np.array([chain.JetPt[ijet], chain.JetEta[ijet], chain.JetPhi[ijet]], dtype=np.float32)*scale_factors 
        
        input_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": jet_data
        }
        target_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": 0
        }
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        graph_list.append((input_graph, target_graph))
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

class RepresentationDataSet(DataSet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.read = read
        self.make_graph = make_graph