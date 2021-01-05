import numpy as np
import itertools
from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

n_input_particle_features = 5
n_max_nodes = 18

def num_particles(event):
    return len(event) // n_input_particle_features


def make_graph(event, debug=False):
    """
    Convert events to graphs. Each event contains a list of particles,
    each particle contains 5 features. The first particle is the input,
    and the following particles are the outputs.

    Args:
        event: a vector of features
    Returns:
        [(input-graph, target-graph)]
    """

    scale = np.array([0.001]*4, dtype=np.float32)
    n_particles = num_particles(event)

    nodes = [[
        event[inode*n_input_particle_features+1], # px
        event[inode*n_input_particle_features+2], # py
        event[inode*n_input_particle_features+3], # pz
        event[inode*n_input_particle_features+4],  # E
    ] for inode in range(n_particles)]

    nodes = np.array(nodes, dtype=np.float32) * scale
    n_nodes = nodes.shape[0] - 1
    if n_nodes > n_max_nodes:
        print("ERROR, {} nodes larger than maximum nodes {}.".format(n_nodes, n_max_nodes))
        return [(None, None)]

    if n_nodes < n_max_nodes:
        nodes = np.concatenate([nodes, np.zeros([n_max_nodes-n_nodes, nodes.shape[1]], dtype=np.float32)], axis=0)
    
    n_nodes = n_max_nodes
    # all input nodes connecting to each other
    all_edges = list(itertools.combinations(range(n_nodes), 2))

    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    all_senders = np.concatenate([senders, receivers], axis=0)
    all_receivers = np.concatenate([receivers, senders], axis=0)
    n_edges = len(all_edges*2)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)


    zero = np.array([0], dtype=np.float32)
    input_datadict = {
        "n_node": 1,
        "n_edge": 1,
        "nodes": nodes[0:1, ],
        "edges": np.expand_dims(np.array([0.0], dtype=np.float32), axis=1),
        "senders": zero,
        "receivers": zero,
        "globals": zero,
    }

    target_datadict = {
        "n_node": n_max_nodes,
        "n_edge": n_edges,
        "nodes": nodes[1:, ],
        "edges": edges,
        "senders": all_senders,
        "receivers": all_receivers,
        "globals": zero,
    }

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)
        print(nodes)

    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]


def read(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield [float(x) for x in line.split()]

class HadronicInteraction(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph