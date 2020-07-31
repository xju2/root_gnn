import os
import tensorflow as tf
import numpy as np

from graph_nets import utils_tf

from root_gnn.src.datasets.base import DataSet
from root_gnn.src.datasets import graph
from root_gnn.src.models import model_utils

class WTaggerFilteredDataset(DataSet):
    def __init__(self, *args, **kwargs):
        self.edge_cut = 0.5
        self.is_signal = False
        super().__init__(*args, **kwargs)

    def set_gnn_config(self, config):
        self.model, self.num_mp, self.batch_size = model_utils.create_load_model(config)

    def signal(self, ss=True):
        self.is_signal = ss

    def read(self, filename, nevts):
        filenames = tf.io.gfile.glob(filename)
        dataset = tf.data.TFRecordDataset(filenames)
        AUTO = tf.data.experimental.AUTOTUNE
        dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
        total_evts = sum([1 for _ in dataset])
        print("Total {:,} events".format(total_evts))

        for batch, data in enumerate(dataset):
            if nevts > 0 and batch >= nevts:
                break
            yield data

    
    def make_graph(self, event, debug):
        inputs_tr, _ = event
        
        # apply the GNN model and filter out the edges with a score less than the threshold 0.5.
        outputs_tr = self.model(inputs_tr, self.num_mp)
        output_graph = outputs_tr[-1]

        # calculate similar variables for GNN-based reconstruction
        # method-one, place a threshold on edge score
        edge_predict = np.squeeze(output_graph.edges.numpy())
        edge_passed = edge_predict > self.edge_cut
        nodes_sel = np.unique(np.concatenate([output_graph.receivers.numpy()[edge_passed],\
            output_graph.senders.numpy()[edge_passed]], axis=0))

        n_nodes = nodes_sel.shape[0]
        n_edges = sum(edge_passed)
        nodes = output_graph.nodes.numpy()[nodes_sel]
        edges = output_graph.edges.numpy()[edge_passed]
        senders = output_graph.senders.numpy()[edge_passed]
        receivers = output_graph.receivers.numpy()[edge_passed]
        # print("n-nodes:", n_nodes)
        # print("n-edges:", n_edges)
        # print("nodes:", nodes.shape)
        # print("edges:", edges.shape)
        # print("senders:", senders.shape)
        # print("receivers:", receivers.shape)

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
    