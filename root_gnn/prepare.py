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

# <TODO: factorize ROOT related out.>
import ROOT 

from graph_nets import utils_tf
from graph_nets import graphs

from root_gnn import tools_tf

from root_gnn.datasets import wprime
from root_gnn.datasets import fourtop

class DataSet(object):
    def __init__(self, with_padding=False, n_graphs_per_evt=1):
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_padding = with_padding
        self.n_files_saved = 0
        self.graphs = []
        self.n_graphs_per_evt = n_graphs_per_evt
        self.n_evts = 0

    def _get_signature(self):
        if self.input_dtype and self.target_dtype:
            return 
        if len(self.graphs) <  1:
            raise RuntimeError("No graphs")

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
        if not os.path.exists(os.path.dirname(os.path.abspath(filename))):
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


    def process(self, save, outname, n_evts_per_record):
        raise NotImplementedError


class TopTaggerDataset(DataSet):
    def __init__(self, filename, with_padding=False):
        super().__init__(with_padding=with_padding)
        self.filename = filename

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
        # zeros = np.array([0.0], dtype=np.float32)
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

        now = time.time()
        # logged_time = now
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
            ievt, len(self.graphs), read_time/60.))


class ToppairDataSet(DataSet):
    def __init__(self, filename, with_padding=False):
        super(ToppairDataSet, self).__init__(with_padding=with_padding)
        self.filename = filename

    def process(self, save=False, outname=None, n_evts_per_record=10):
        self.graphs = []
        tree_name = "output"
        chain = ROOT.TChain(tree_name, tree_name)
        chain.Add(self.filename)
        n_entries = chain.GetEntries()
        print("Total {:,} Events".format(n_entries))
        n_3jets = 0
        n_one_top = 0
        n_two_top = 0
        evtid = 0
        max_jets = 0
        for ientry in range(n_entries):
            chain.GetEntry(ientry)
            if len(chain.m_jet_pt) < 3:
                continue
            n_3jets += 1
            max_jets = max([max_jets, len(chain.m_jet_pt)])
            if (-1 not in chain.reco_triplet_1 or -1 not in chain.reco_triplet_2):
                n_one_top += 1
            if (-1 not in chain.reco_triplet_1 and -1 not in chain.reco_triplet_2):
                n_two_top += 1

        print("At least 3 jets:   {:10,}, {:.1f}%".format(n_3jets, 100*n_3jets/n_entries))
        print("At least one top:  {:10,}, {:.1f}%".format(n_one_top, 100*n_one_top/n_entries))
        print("At least two tops: {:10,}, {:.1f}%".format(n_two_top, 100*n_two_top/n_entries))
        print("Maximum jets in an event:", max_jets)
        return
        print("event id:", evtid)
        print(chain.m_jet_pt)
        print(chain.m_jet_eta)
        zeros = np.array([0.0], dtype=np.float32)
        def make_graph(event):
            n_max_nodes = 60
            n_nodes = len(event.m_jet_pt)
            print(np.array(event.m_jet_pt).shape)
            nodes = np.hstack((event.m_jet_pt, event.m_jet_eta, event.m_jet_phi, event.m_jet_E))
            nodes = nodes.reshape(-1, n_nodes).transpose()
            print(n_nodes)
            print(nodes)

            # edges
            all_edges = list(itertools.combinations(range(n_nodes), 2))
            senders = np.array([x[0] for x in all_edges])
            receivers = np.array([x[1] for x in all_edges])
            n_edges = len(all_edges)
            edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
            true_edges = set(list(itertools.combinations(event.reco_triplet_1, 2)) \
                + list(itertools.combinations(event.reco_triplet_2, 2)))
            truth_labels = [int(x in true_edges) for x in all_edges]
            print(true_edges)
            print(event.reco_triplet_1)
            print(event.reco_triplet_2)
            print(truth_labels)
            truth_labels = np.array(truth_labels)

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
                "nodes": zeros,
                "edges": truth_labels,
                "senders": senders,
                "receivers": receivers,
                "globals": zeros
            }
            input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
            target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
            return [(input_graph, target_graph)]

        make_graph(chain)
            

class WTaggerDataset(DataSet):
    def __init__(self, filename, with_padding=False):
        super().__init__(with_padding=with_padding)
        self.filename = filename
        self.n_node_features = wprime.n_node_features

    def process(self, save=False, outname=None, n_evts_per_record=10, debug=False):
        self.graphs = []

        now = time.time()
        ievt = 0
        self.n_evts = 0

        with open(self.filename, 'r') as f:
            for line in f:
                event = line.split()
                if len(event) % self.n_node_features != 0:
                    print("event info cannot be factorized by {} features".format(self.n_node_features))
                self.graphs += wprime.make_graph(event, debug)
                self.n_evts += 1
                ievt += 1
                if save and ievt % n_evts_per_record == 0:
                    self.tot_data = len(self.graphs)
                    self.write_tfrecord(outname, n_evts_per_record)
                    self.graphs = []
                    self.n_evts = 0

        self.tot_data = len(self.graphs)
        read_time = time.time() - now
        print("WTaggerDataset added {} events, in {:.1f} mins".format(
            ievt, read_time/60.))


class FourTopDataset(DataSet):
    def __init__(self, filename, tree_name="nominal_Loose", with_padding=False):
        super().__init__(with_padding=with_padding)
        self.filename = filename
        self.tree_name = tree_name

    def process(self, save=False, outname=None, n_evts_per_record=10, debug=False):
        self.graphs = []

        now = time.time()
        ievt = 0
        self.n_evts = 0

        tree_name = self.tree_name
        chain = ROOT.TChain(tree_name, tree_name)
        chain.Add(self.filename)
        n_entries = chain.GetEntries()        
        print("Total {:,} Events".format(n_entries))
        for ientry in range(n_entries):
            chain.GetEntry(ientry)
            self.graphs += fourtop.make_graph(chain, debug)
            self.n_evts += 1
            ievt += 1
            if save and ievt % n_evts_per_record == 0:
                self.tot_data = len(self.graphs)
                self.write_tfrecord(outname, n_evts_per_record)
                self.graphs = []
                self.n_evts = 0

        self.tot_data = len(self.graphs)
        read_time = time.time() - now
        print("WTaggerDataset added {} events, in {:.1f} mins".format(
            ievt, read_time/60.))