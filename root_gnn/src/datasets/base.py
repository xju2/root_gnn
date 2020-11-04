"""
base class defines the procedure with that the TFrecord data is produced.
"""
import time
import os

import tensorflow as tf
from graph_nets import utils_tf
from root_gnn.src.datasets import graph
from typing import Optional

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

    def read(self, filename, nevts: Optional[int] = -1):
        """
        read the file from `filename` and return an event
        """
        raise NotImplementedError

    def make_graph(self, event, debug):
        """
        Convert the event into a graphs_tuple. 
        """
        raise NotImplementedError

    def _get_signature(self):
        if self.input_dtype and self.target_dtype:
            return 
        if len(self.graphs) <  1:
            raise RuntimeError("No graphs")

        ex_input, ex_target = self.graphs[0]
        self.input_dtype, self.input_shape = graph.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        self.target_dtype, self.target_shape = graph.dtype_shape_from_graphs_tuple(
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
            example = graph.serialize_graph(*data)
            writer.write(example)
        self.n_files_saved += n_files

    def process(self, filename, save, outname, n_evts_per_record, debug, max_evts):
        self.graphs = []

        now = time.time()
        ievt = 0
        self.n_evts = 0

        ifailed = 0
        nskips = 0
        for event in self.read(filename):
            output_name = "{}_{}.tfrec".format(outname, self.n_files_saved)
            if os.path.exists(output_name):
                self.n_files_saved += 1
                nskips += n_evts_per_record
                print("{} is there; {:,} Events to skip".format(output_name, nskips))
            if nskips > 0:
                nskips -= 1
                ievt += 1
                continue

            if max_evts > 0 and ievt > max_evts:
                break

            gen_graphs = self.make_graph(event, debug)
            if gen_graphs[0][0] is None:
            
                ifailed += 1
                continue

            self.graphs += gen_graphs
            self.n_evts += 1
            ievt += 1

            if save and ievt % n_evts_per_record == 0:
                self.tot_data = len(self.graphs)
                self.write_tfrecord(outname, n_evts_per_record)
                self.graphs = []
                self.n_evts = 0

        if len(self.graphs) > 0:
            # save left over graphs
            self.write_tfrecord(outname, len(self.graphs))
            
        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            ievt, read_time/60.))
        print("{:,} events failed in being converted to graph".format(ifailed))  