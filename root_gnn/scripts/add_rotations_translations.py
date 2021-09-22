#!/usr/bin/env python
import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.utils import calc_dphi

#function for adding rotations about jet axis and translations to a jet assuming the input is a graph
def add_rotations_translations(jet_graph):
    #assumes the jet graph is already centered at the origin
    theta = np.random.uniform(0,2*np.pi) 
    shift_eta = np.random.uniform(-1,1)
    shift_phi = np.random.uniform(-1,1)
    jet_eta = jet_graph.globals.numpy()[0][1]
    jet_phi = jet_graph.globals.numpy()[0][2]
    new_eta = jet_eta*np.cos(theta) - jet_phi*np.sin(theta) + shift_eta
    new_phi = jet_eta*np.sin(theta) + jet_phi*np.cos(theta)  + shift_phi
    new_globals = [jet_graph.globals.numpy()[0][0], new_eta, new_phi]
    nodes = jet_graph.nodes.numpy()
    new_nodes = []
    for node_idx in range(len(jet_graph.nodes)):
        new_node_eta = nodes[node_idx][1]*np.cos(theta) - nodes[node_idx][2]*np.sin(theta) +shift_eta
        new_node_phi = nodes[node_idx][1]*np.sin(theta) + nodes[node_idx][2]*np.cos(theta)+shift_phi
        new_nodes.append([jet_graph.nodes.numpy()[node_idx][0], new_node_eta, new_node_phi])
    new_graph = jet_graph.replace(nodes = new_nodes, globals = new_globals)
    return new_graph