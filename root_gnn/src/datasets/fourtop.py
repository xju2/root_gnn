import numpy as np
import itertools
import math
from typing import Optional
from graph_nets import utils_tf
import ROOT

from root_gnn.src.datasets.base import DataSet

is_signal = False
def make_graph(event, debug=False):
    n_jets = len(event.jet_pt)
    n_ele = len(event.ele_pt) if hasattr(event, 'ele_pt') else 0
    n_muon = len(event.mu_pt) if hasattr(event, 'mu_pt') else 0
    n_nodes = n_jets + n_ele + n_muon
    
    # scale = np.array([0.001, 3., math.pi, 0.001], dtype=np.float32)
    nodes = [[
        event.jet_pt[idx]*0.001,
        event.jet_eta[idx]/3.,
        event.jet_phi[idx]/math.pi,
        event.jet_e[idx]*0.001
    ] for idx in range(n_jets)]

    if n_ele > 0:
        nodes += [[
            event.ele_pt[idx]*0.001,
            event.ele_eta[idx]/3.,
            event.ele_phi[idx]/math.pi,
            event.ele_e[idx]*0.001
        ] for idx in range(n_ele)]
    
    if n_muon > 0:
        nodes += [[
            event.mu_pt[idx]*0.001,
            event.mu_eta[idx]/3.,
            event.mu_phi[idx]/math.pi,
            event.mu_e[idx]*0.001
        ] for idx in range(n_muon)]
    
    nodes = np.array(nodes, dtype=np.float32) 
    node_target =  np.expand_dims(np.array([0.0]*n_nodes, dtype=np.float32), axis=1)

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)


    # edges 1) fully connected, 2) objects nearby in eta/phi are connected
    # TODO: implement 2). <xju>
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)

    if debug:
        print(n_edges, "edges")
        print("senders:", senders)
        print("receivers:", receivers)
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([0.0], dtype=np.float32)
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": node_target,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([float(is_signal)], dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]

def read(filename, nevts=-1):
    tree_name = "nominal_Loose"
    chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
    chain.Add(filename)
    n_entries = chain.GetEntries()        
    print("Total {:,} Events".format(n_entries))
    for ientry in range(n_entries):
        chain.GetEntry(ientry)
        if nevts > 0 and ientry > nevts:
            break
        yield chain


class FourTopDataset(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph