import os
import tensorflow as tf
import numpy as np

from graph_nets import utils_tf

from root_gnn.src.datasets.base import DataSet
from root_gnn.src.datasets import graph
from root_gnn.utils import load_model

class WTaggerFilteredDataset(DataSet):
    def __init__(self, name="WTaggerFilteredDataset", *args, **kwargs):
        self.edge_cut = 0.5
        self.is_signal = False
        super().__init__(name=name, *args, **kwargs)

    def set_configuration(self, config):
        self.model, self.num_mp, self.batch_size = load_model(config)

    def signal(self, ss=True):
        self.is_signal = ss

    def read(self, filename):
        filenames = tf.io.gfile.glob(filename)
        dataset = tf.data.TFRecordDataset(filenames)
        AUTO = tf.data.experimental.AUTOTUNE
        dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
        total_evts = sum([1 for _ in dataset])
        print("Total {:,} events".format(total_evts))

        for data in dataset:
            yield data

    
    def make_graph(self, event, debug):
        inputs_tr, _ = event
        
        # apply the GNN model and filter out the edges with a score less than the threshold 0.5.
        outputs_tr = self.model(inputs_tr, self.num_mp, is_training=False)
        output_graph = outputs_tr[-1]

        # calculate similar variables for GNN-based reconstruction
        # method-one, place a threshold on edge score
        edge_predict = np.squeeze(output_graph.edges.numpy())
        edge_passed = edge_predict > self.edge_cut
        nodes_sel = np.unique(np.concatenate([output_graph.receivers.numpy()[edge_passed],\
            output_graph.senders.numpy()[edge_passed]], axis=0))

        n_nodes = nodes_sel.shape[0]
        n_edges = sum(edge_passed)
        nodes = inputs_tr.nodes.numpy()[nodes_sel]
        edges = inputs_tr.edges.numpy()[edge_passed]

        node_dicts = {}
        for idx, val in enumerate(nodes_sel):
            node_dicts[val] = idx

        senders = np.array([node_dicts[x] for x in inputs_tr.senders.numpy()[edge_passed]])
        receivers = np.array([node_dicts[x] for x in inputs_tr.receivers.numpy()[edge_passed]])

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
            "globals": np.array([float(self.is_signal)], dtype=np.float32)
        }
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        return [(input_graph, target_graph)]