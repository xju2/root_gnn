import ROOT as root
import numpy as np
import itertools
from root_gnn.src.datasets.base import DataSet

filename = '/global/homes/x/xju/m3443/data/TauStudies/v0/Ntuple_ditau_processed.root'

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
        
def make_graph(event, debug=False):
    
    
    # globals
    tau_vec = [event.truthTauEt, event.truthTauEta, event.truthTauPhi]
    tau1_vec = [param[0] for param in tau_vec]
    tau2_vec = [param[1] for param in tau_vec]
    
    global_attr = tau1_vec + tau2_vec
    
    
    
    # nodes
    n_nodes = 0
    
    track_info = event.TrackPt, event.TrackEta, event.TrackPhi
    tower_info = event.JetTowerEt, event.JetTowerEta, event.JetTowerPhi
    
    nodes = []
    
    prev_num_track = 0
    prev_num_tower = 0
    
    for indv_jet in range(len(event.JetGhostTrackN)):
        indv_nodes = [] # dim(nodes) = (num_tracks + num_towers) * 3 for each jet
        
        for num_track in range(prev_num_track, prev_num_track+event.JetGhostTrackN[indv_jet]):
            track_idx = event.JetGhostTrackIdx[num_track]
            indv_track_info = [] # contains the triplet info for one track
            for param in track_info:
                indv_track_info.append(param[track_idx])    
            indv_nodes.append(np.array(indv_track_info, dtype=np.float32))
            n_nodes += 1
        prev_num_track += event.JetGhostTrackN[indv_jet]
            
        for num_tower in range(prev_num_tower, prev_num_tower+event.JetTowerN[indv_jet]):
            tower_idx = num_tower
            indv_tower_info = [] # contains the triplet info for one tower
            for param in tower_info:
                indv_tower_info.append(param[tower_idx])
            indv_nodes.append(np.array(indv_tower_info, dtype=np.float32))
            n_nodes += 1
        prev_num_tower += event.JetTowerN[indv_jet]
        
        nodes.append(np.array(indv_nodes, dtype=np.float32))
    
    nodes = np.array(nodes, dtype=np.float32)
    
    if debug:
        print(np.array(event.TrackPt).shape)
        print(n_nodes)
        print(nodes)
        
        

    # edges
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
    true_edges = set(list(itertools.combinations(tau1_vec, 2)) \
        + list(itertools.combinations(tau2_vec, 2)))
    truth_labels = [int(x in true_edges) for x in all_edges]
    if debug:
        print(all_edges)
        print(truth_labels)
    truth_labels = np.array(truth_labels, dtype=np.float32)
    zeros = np.array([0.0], dtype=np.float32)

    
    
    # Graphs
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": zeros
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": zeros,
        "edges": truth_labels,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array(global_attr, dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    
    return [(input_graph, target_graph)]


class diTauMass(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph