"""
Functions that convert usual inputs to graph
"""
import time
import os
import itertools
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import graphs

from root_gnn import tools_tf


class TopTaggerDataset:
    def __init__(self, filename, with_padding=False):
        self.filename = filename
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_padding = with_padding
        self.n_files_saved = 0

    def process(self, save=False, outname=None, n_evts_per_record=10):
        self.graphs = []
        with pd.HDFStore(self.filename, mode='r') as store:
            df = store['table']
        
        print("{:,} Events".format(df.shape[0]))
        event = df.iloc[0]

        print(event.index)
        print(event['E_199'])
        features = ['E', 'PX', 'PY', 'PZ']
        scale = 0.001
        zeros = np.array([0.0], dtype=np.float32)
        solution = 'is_signal_new'
        if save and not outname:
            raise ValueError("need output name in save mode")

        def make_graph(event):
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
            nodes = np.array(nodes)
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
                "globals": zeros
            }
            target_datadict = {
                "n_node": n_nodes,
                "n_edge": n_edges,
                "nodes": nodes,
                "edges": edges,
                "senders": senders,
                "receivers": receivers,
                "globals": np.array(event[solution], dtype=np.float32)
            }
            input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
            target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
            return [(input_graph, target_graph)]

        now = time.time()
        logged_time = now
        self.n_graphs_per_evt = 1

        ievt = 0
        self.n_evts = 0
        for ievt in range(df.shape[0]):
            self.graphs += make_graph(df.iloc[ievt])
            self.n_evts += 1
            ievt += 1
            # if ievt % 100 == 0:
            #     print("{} mins processed {} events".format((time.time()-logged_time)/60, ievt))
            #     logged_time = time.time()
            if save and ievt % n_evts_per_record == 0:
                self.tot_data = len(self.graphs)
                self.write_tfrecord(outname, n_evts_per_record)
                self.graphs = []
                self.n_evts = 0

        self.tot_data = len(self.graphs)
        read_time = time.time() - now
        print("TopTaggerDataset added {} events, Total {} graphs, in {:.1f} mins".format(
            self.n_evts, len(self.graphs), read_time/60.))

    def _get_signature(self):
        if self.input_dtype and self.target_dtype:
            return 
        ex_input, ex_target = self.graphs[0]
        self.input_dtype, self.input_shape = tools_tf.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        self.target_dtype, self.target_shape = tools_tf.dtype_shape_from_graphs_tuple(
            ex_target, with_padding=self.with_padding)

    def write_tfrecord(self, filename, n_evts_per_record=10):
        self._get_signature()
        def generator():
            for G in self.graphs:
                yield (G[0], G[1])

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(self.input_dtype, self.target_dtype),
            output_shapes=(self.input_shape, self.target_shape),
            args=None)

        n_graphs_per_evt = self.n_graphs_per_evt
        n_evts = self.n_evts
        n_files = n_evts//n_evts_per_record
        if n_evts%n_evts_per_record > 0:
            n_files += 1

        print("In total {} graphs, {} graphs per event".format(self.tot_data, n_graphs_per_evt))
        print("In total {} events, write to {} files".format(n_evts, n_files))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        igraph = -1
        ifile = -1
        writer = None
        n_graphs_per_record = n_graphs_per_evt * n_evts_per_record
        for data in dataset:
            igraph += 1
            if igraph % n_graphs_per_record == 0:
                ifile += 1
                if writer is not None:
                    writer.close()
                outname = "{}_{}.tfrec".format(filename, self.n_files_saved+ifile)
                writer = tf.io.TFRecordWriter(outname)
            example = tools_tf.serialize_graph(*data)
            writer.write(example)
        self.n_files_saved += n_files
            