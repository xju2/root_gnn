import numpy as np
import pandas as pd
import itertools
from typing import Optional

from graph_nets import utils_tf

from root_gnn.src.datasets.base import DataSet


features = ['E', 'PX', 'PY', 'PZ']
scale = 0.001
solution = 'is_signal_new'

def make_graph(event, debug: Optional[bool] = False):
    n_max_nodes = 200
    n_nodes = 0
    nodes = []
    for inode in range(n_max_nodes):
        E_name = 'E_{}'.format(inode)
        if event[E_name] < 0.1:
            continue

        f_keynames = ['{}_{}'.format(x, inode) for x in features]
        n_nodes += 1
        nodes.append(event[f_keynames].values*scale)
    nodes = np.array(nodes, dtype=np.float32)
    # print(n_nodes, "nodes")
    # print("node features:", nodes.shape)

    # edges 1) fully connected, 2) objects nearby in eta/phi are connected
    # TODO: implement 2). <xju>
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
    # print(n_edges, "edges")
    # print("senders:", senders)
    # print("receivers:", receivers)

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
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([event[solution]], dtype=np.float32)
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]

def read(filename, nevts: Optional[int] = -1):
    with pd.HDFStore(filename, mode='r') as store:
        df = store['table']
    
    print("{:,} Events".format(df.shape[0]))
    for ievt in range(df.shape[0]):
        if nevts > 0 and ievt > nevts:
            break
        yield df.iloc[ievt]


class TopTaggerDataset(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = read
        self.make_graph = make_graph