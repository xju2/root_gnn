import numpy as np
import itertools
from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

n_input_particle_features = 5
n_target_node_features = 6 # for each top [top 4-vector, charge (2 bits), and p is-there]
n_node_features = 5 # jet 4-vector and b-tagging
n_max_tops = 4

onehot_charge_matrix = [
    [1, 1], # 0
    [0, 1], # +1
    [1, 0], # -1
    [0, 0], # No charge info
]
onehot_charge_dict = {
  0: 0,  # 0
  1: 1,  # +1
  -1: 2, # -1
  2: 3   # not defined
}

def one_hot_encoder(id):
    return onehot_charge_matrix[onehot_charge_dict[id]]

def sign(a):
    return int(a > 0) * 1 + int(a < 0) * (-1)

def num_particles(event):
    return len(event) // n_input_particle_features

def make_graph(event, debug=False):
    # input particle features are: pdgID (or isBjet), px, py, pz, E
    # abs(pdgID) == 6 --> top quark, otherwise
    # the first one is: isBjet, saying if the jet comes from B hadrons.

    scale = np.array([0.01]*4 + [1], dtype=np.float32)
    n_particles = num_particles(event)

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
        print(event)
        print("{} top quarks in the event".format(n_tops))

    nodes = [[
        event[inode*n_input_particle_features+1], # px
        event[inode*n_input_particle_features+2], # py
        event[inode*n_input_particle_features+3], # pz
        event[inode*n_input_particle_features+4],  # E
        event[inode*n_input_particle_features] # b-tagging
    ] for inode in range(n_tops, n_particles)]

    nodes = np.array(nodes, dtype=np.float32) * scale
    n_nodes = nodes.shape[0]


    # all input nodes connecting to each other
    all_edges = list(itertools.combinations(range(n_nodes), 2))

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
        sign(event[inode*n_input_particle_features]),
        # *one_hot_encoder(sign(event[inode*n_input_particle_features])), # charge info
        1 # is a top
    ] for inode in range(0, n_tops)]
    
    top_nodes = np.array(top_nodes, dtype=np.float32)
    
    if n_tops < n_max_tops:
        top_nodes = np.concatenate([top_nodes, np.zeros((n_max_tops-n_tops, n_target_node_features), dtype=np.float32)], axis=0)

    target_datadict = {
        "n_node": 1,
        "n_edge": 1,
        "nodes": np.array([0], dtype=np.float32),
        "edges": np.array([0], dtype=np.float32),
        "senders": np.array([0]),
        "receivers": np.array([0]),
        "globals": top_nodes.T.reshape((-1,))
    }

    if debug:
        print(n_nodes, "nodes")
        print("node features:", nodes.shape)
        print(nodes)
        print(top_nodes)

    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]


def read(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield [float(x) for x in line.split(',')]


class TopReco(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph