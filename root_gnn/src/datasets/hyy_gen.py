import numpy as np
import itertools
from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

n_node_features = 5
max_nodes = 14 # including the particle that decays

def num_particles(event):
    return len(event) // n_node_features

one_hot_matrix = [
    [0, 0], # photon
    [0, 1], # leptons
    [1, 0], # jets
    [1, 1], # MET
]
id_dict = {
    100: 0,
    200: 1,
    300: 2,
    400: 3
}

def one_hot_encoder(id):
    return one_hot_matrix(id_dict[id])


def make_graph(event, debug=False):
    # each particle contains: pdgID, px, py, pz, E
    scale = 0.0001
    n_particles = num_particles(event)
    nodes = [[
        event[inode*n_node_features+1], # px
        event[inode*n_node_features+2], # py
        event[inode*n_node_features+3], # pz
        event[inode*n_node_features+4],  # E
        *one_hot_encoder(inode*n_node_features) # particle type
    ] for inode in range(n_particles) if event[inode*n_node_features] != -5 ]

    nodes = np.array(nodes, dtype=np.float32) * scale
    n_nodes = nodes.shape[0]

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)

    if nodes.shape[0] > max_nodes:
        print("cluster decays to more than {} nodes".format(max_nodes))
        return [(None, None)]
    elif nodes.shape[0] < max_nodes:
        print("nodes: {} less than maximum {}".format(nodes.shape[0], max_nodes))
        print(event)
        new_nodes = np.zeros([max_nodes, 6], dtype=np.float32)
        new_nodes[:nodes.shape[0], :] = nodes
        nodes = new_nodes

    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)

    input_datadict = {
        "n_node": 1,
        "n_edge": 1,
        "nodes": nodes[0, :].reshape((1, -1)), # this part will be replaced with a normal distribution
        "edges": np.expand_dims(np.array([1.0]*1, dtype=np.float32), axis=1),
        "senders": np.array([0]),
        "receivers": np.array([0]),
        "globals": np.array([1], dtype=np.float32)
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([1]*(n_nodes-1)+[0]*(max_nodes-n_nodes+1), dtype=np.float32)
    }

    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]


def read(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield [float(x) for x in line.split(',')]


class HiggsYYGen(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph