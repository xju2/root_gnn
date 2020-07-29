import ROOT # use TLorentzVector

import numpy as np
import itertools

from graph_nets import utils_tf

n_node_features = 7
ZERO = ROOT.TLorentzVector()

def make_graph(event, debug=False):
    scale = 0.001
    # information of each particle: px, py, pz, E, pdgID, isFromW, isInLeadingJet
    n_nodes = len(event) // n_node_features
    nodes = [[
        event[inode*n_node_features+0], # px
        event[inode*n_node_features+1], # py
        event[inode*n_node_features+2], # pz
        event[inode*n_node_features+3]  # E
    ] for inode in range(n_nodes) ]
    node_target = [
        event[inode*n_node_features+5] # isFromW
        for inode in range(n_nodes)
    ]

    nodes = np.array(nodes, dtype=np.float32) / scale
    node_target = np.array(node_target, dtype=np.float32)
    true_nodes = np.where(node_target==1)[0].tolist()
    node_target = np.expand_dims(node_target, axis=1)

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)
        print('node target:', node_target.shape)
        print("truth nodes:", sum(true_nodes))


    # edges 1) fully connected, 2) objects nearby in eta/phi are connected
    # TODO: implement 2). <xju>
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)

    edge_target = [
        int(edge[0] in true_nodes and edge[1] in true_nodes)
        for edge in all_edges
    ]
    # print("Truth Edges:", sum(edge_target))
    edge_target = np.expand_dims(np.array(edge_target, dtype=np.float32), axis=1)

    if debug:
        print(n_edges, "edges")
        print("senders:", senders)
        print("receivers:", receivers)
        print("edge_target:", edge_target.shape)
        # print("edge:", edge_target)
    
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
        "nodes": node_target,
        "edges": edge_target,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([0.0], dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]

def evaluate_evt(event):

    n_features = n_node_features
    n_particles = len(event) // n_features

    particles = [
        ROOT.TLorentzVector(ROOT.TVector3(
            event[inode*n_features+0],
            event[inode*n_features+1],
            event[inode*n_features+2]),
            event[inode*n_features+3])
        for inode in range(n_particles)
    ]
    leading_jet_idxs = [
        inode
        for inode in range(n_particles) if event[inode*n_features+6] == 1
    ]
    w_jet_idxs = [
        inode
        for inode in range(n_particles) if event[inode*n_features+5] == 1        
    ]

    if len(leading_jet_idxs) > 0:
        tlv_leading_jet = particles[leading_jet_idxs[0]]
        for leading_jet_idx in leading_jet_idxs[1:]:
            tlv_leading_jet += particles[leading_jet_idx]
    else:
        tlv_leading_jet = ZERO

    if len(w_jet_idxs) > 0:
        tlv_wboson = particles[w_jet_idxs[0]]
        for wjet_idx in w_jet_idxs[1:]:
            tlv_wboson += particles[wjet_idx]
    else:
        tlv_wboson = ZERO
    
    return tlv_leading_jet, tlv_wboson

def invariant_mass(event, p_list):
    tlv = ZERO
    if len(p_list) < 1:
        return tlv
    
    particles = [
        ROOT.TLorentzVector(ROOT.TVector3(
            event[inode*n_node_features+0],
            event[inode*n_node_features+1],
            event[inode*n_node_features+2]),
            event[inode*n_node_features+3])
        for inode in p_list
    ]
    for pp in particles:
        tlv += pp
    return tlv


def read(filename, nevts=1, skip_nevts=0):
    iskip = 0
    ievt = 0
    with open(filename, 'r') as f:
        for line in f:
            if iskip < skip_nevts:
                iskip += 1
                continue
            ievt += 1
            if ievt > nevts:
                break
            event = [float(x) for x in line.split()]
            yield event