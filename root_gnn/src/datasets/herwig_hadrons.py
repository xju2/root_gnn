import numpy as np
import itertools
from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

n_node_features = 6
max_nodes = 3 # including the particle that decays

def num_particles(event):
    return len(event) // n_node_features

def make_graph(event, debug=False):
    scale = 0.0001
    n_nodes = num_particles(event)
    nodes = [[
        event[inode*n_node_features+1], # E
        event[inode*n_node_features+2], # px
        event[inode*n_node_features+3], # py
        event[inode*n_node_features+4]  # pz
    ] for inode in range(n_nodes)]
    nodes = np.array(nodes, dtype=np.float32) * scale

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)

    if nodes.shape[0] > max_nodes:
        print("cluster decays to more than {} nodes".format(max_nodes))
        return [(None, None)]
    elif nodes.shape[0] < max_nodes:
        print("nodes: {} less than maximum {}".format(nodes.shape[0], max_nodes))
        new_nodes = np.zeros([max_nodes, 4], dtype=np.float32)
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
        "nodes": nodes[0, :].reshape((1, -1)),
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
            yield [float(x) for x in line.split()]


class HerwigHadrons(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph