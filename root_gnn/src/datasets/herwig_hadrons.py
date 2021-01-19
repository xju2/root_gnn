import numpy as np
import itertools
import re
from graph_nets import utils_tf
from root_gnn.src.datasets.base import DataSet

n_node_features = 6
n_max_nodes = 2 # maximum number of out-going particles

def make_graph(event, debug=False):
    
    particles = event[:-2].split(';')
    
    is_light = True
    array = []
    node_dict = {}
    node_idx = 0
    root_uid = 0
    for ipar, par in enumerate(particles):
        items = par.split(',')
        if ipar == 0:
            is_light = items[0] == 'L'
            items = items[1:]

        if len(items) < 2:
            continue
            
        uid = int(re.search('([0-9]+)', items[0]).group(0))
        if ipar == 0:
            root_uid = uid

        node_dict[uid] = node_idx
        node_idx += 1
        pdgid = [int(items[1])]
        children, idx = ([int(x) for x in re.search('[0-9]+ [0-9]*', items[2]).group(0).strip().split(' ')],3) if '[' in items[2] else ([-1, -1], 2)
        # WARN: assuming the number of decay products could be 0, 1 and 2
        if len(children) == 1:
            children += [-1]
        momentum = [float(x) for x in items[idx:]]
        all_info = [uid] + children + pdgid + momentum
        array.append(all_info)

    # rows: particles, 
    # columns: uid, child1, child2, pdgid, 4-momentum
    array = np.array(array)
    nodes = array[:, 4:].astype(np.float32)
    n_nodes = nodes.shape[0] - 1

    if n_nodes > n_max_nodes:
        print("cluster decays to more than {} nodes".format(n_max_nodes))
        return [(None, None)]

    if n_nodes < n_max_nodes:
        nodes = np.concatenate([nodes, np.zeros([n_max_nodes-n_nodes, nodes.shape[1]], dtype=np.float32)], axis=0)

    n_nodes = nodes.shape[0]
    senders = np.concatenate([array[array[:, 1] > 0, 0].astype(np.int32), array[array[:, 2] > 0, 0].astype(np.int32)])
    receivers = np.concatenate([array[array[:, 1] > 0, 1].astype(np.int32), array[array[:, 2] > 0, 2].astype(np.int32)])
    
    # convert node id to [0, xxx]
    # remove root id in the edges
    # use fully-connected graph?
    senders = np.array([node_dict[i] for i in senders], dtype=np.int32)
    receivers = np.array([node_dict[j] for i,j in zip(senders,receivers)], dtype=np.int32)
    n_edges = senders.shape[0]
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
    zero = np.array([0], dtype=np.float32)

    input_datadict = {
        "n_node": 1,
        "n_edge": 1,
        "nodes": nodes[0:1, :],
        "edges": np.expand_dims(np.array([0.], dtype=np.float32), axis=1),
        "senders":zero,
        "receivers": zero,
        "globals": zero,
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        # "globals": np.array([1]*(n_nodes-1)+[0]*(max_nodes-n_nodes+1), dtype=np.float32)
        "globals": zero,
    }
    # print("input: ", input_datadict)
    # print("target:", target_datadict)
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    # padding the graph if number of nodes is less than max-nodes??
    # not sure it is nessary..

    return [(input_graph, target_graph)]


def read(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line[:-1]


class HerwigHadrons(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph