import numpy as np
import itertools
from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

n_input_particle_features = 5
n_target_node_features = 5
n_node_features = 5
n_max_tops = 4

def num_particles(event):
    return len(event) // n_input_particle_features

def make_graph(event, debug=False):
    # input particle features are: pdgID (or isBjet), px, py, pz, E
    # abs(pdgID) == 6 --> top quark, otherwise
    # the first one is: isBjet, saying if the jet comes from B hadrons.

    scale = 0.0001
    n_particles = num_particles(event)

    # reserve a number of ghost nodes for top candidates
    nodes = np.zeros((n_max_tops, n_node_features), dtype=np.float32)

    # find how many top quarks in the event
    n_tops = 0
    for inode in range(n_particles):
        if abs(event[inode*n_input_particle_features]) == 6:
            n_tops += 1
        else:
            break

    if n_particles == n_tops:
        # event only contains top quark information.
        return [(None, None)]

    if debug:
        print("{} top quarks in the event".format(n_tops))

    jet_nodes = [[
        event[inode*n_input_particle_features+1], # px
        event[inode*n_input_particle_features+2], # py
        event[inode*n_input_particle_features+3], # pz
        event[inode*n_input_particle_features+4],  # E
        0, 
    ] for inode in range(n_tops, n_particles)]

    jet_nodes = np.array(jet_nodes, dtype=np.float32) * scale
    nodes = np.concatenate([nodes, jet_nodes], axis=0)
    n_nodes = n_particles-n_tops+n_max_tops

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)


    # all input nodes connecting to each other, except 
    # the reserved ghost nodes. they only connect to 
    # non-ghost nodes.
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    excluded_edges = list(itertools.combinations(range(n_max_tops), 2))
    all_edges = [x for x in all_edges if x not in excluded_edges]

    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)

    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([0], dtype=np.float32)
    }

    # prepare target nodes... edge information not used.
    top_nodes = [[
        event[inode*n_input_particle_features+1], # px
        event[inode*n_input_particle_features+2], # py
        event[inode*n_input_particle_features+3], # pz
        event[inode*n_input_particle_features+4], # E
        1 # is a top
    ] for inode in range(0, n_tops)]
    top_nodes = np.array(top_nodes, dtype=np.float32)
    if n_tops < n_max_tops:
        top_nodes = np.concatenate([top_nodes, np.zeros((n_max_tops-n_tops, n_target_node_features), dtype=np.float32)], axis=0)

    target_datadict = {
        "n_node": n_max_tops,
        "n_edge": 1,
        "nodes": top_nodes,
        "edges": np.array([0], dtype=np.float32),
        "senders": np.array([0]),
        "receivers": np.array([0]),
        "globals": np.array([0], dtype=np.float32)
    }

    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]


def read(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield [float(x) for x in line.split()]


class TopReco(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph