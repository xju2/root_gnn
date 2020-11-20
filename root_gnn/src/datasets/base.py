"""
base class defines the procedure with that the TFrecord data is produced.
"""
import time
import os
from typing import Optional
from multiprocessing import Pool
from functools import partial

import tensorflow as tf
from graph_nets import utils_tf
from root_gnn.src.datasets import graph


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

    def _get_signature(self, this_graph):
        if self.input_dtype and self.target_dtype:
            return 
        if this_graph[0] is None:
            raise RuntimeError("Wrong graph input")

        ex_input, ex_target = this_graph
        self.input_dtype, self.input_shape = graph.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        self.target_dtype, self.target_shape = graph.dtype_shape_from_graphs_tuple(
            ex_target, with_padding=self.with_padding)
    

    def write_tfrecord(self, graphs, filename, n_evts_per_record=10, n_graphs_per_evt=1):
        self._get_signature(graphs[0])
        def generator():
            for G in graphs:
                yield (G[0], G[1])

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(self.input_dtype, self.target_dtype),
            output_shapes=(self.input_shape, self.target_shape),
            args=None)

        n_evts = len(graphs) // n_graphs_per_evt
        n_files = n_evts//n_evts_per_record
        if n_evts%n_evts_per_record > 0:
            n_files += 1

        print("In total {} graphs, {} graphs per event".format(len(graphs), n_graphs_per_evt))
        print("In total {} events, write to {} files".format(n_evts, n_files))
        abs_outdir = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(abs_outdir):
            os.makedirs(abs_outdir)

        n_files_saved = 0
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
                outname = "{}_{}.tfrec".format(filename, n_files_saved+ifile)
                writer = tf.io.TFRecordWriter(outname)
            example = graph.serialize_graph(*data)
            writer.write(example)

    def subprocess(self, ijob, n_evts_per_record, num_evts, filename, outname, debug):
        ievt = -1
        ifailed = 0
        all_graphs = []
        start_entry = ijob * n_evts_per_record
        for event in self.read(filename):
            ievt += 1
            if ievt < start_entry:
                continue
            
            gen_graphs = self.make_graph(event, debug)
            if gen_graphs[0][0] is None:
                ifailed += 1
                continue

            all_graphs += gen_graphs
            if ievt == start_entry + n_evts_per_record - 1:
                break
        
        isaved = len(all_graphs)
        ex_input, ex_target = all_graphs[0]
        input_dtype, input_shape = graph.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        target_dtype, target_shape = graph.dtype_shape_from_graphs_tuple(
            ex_target, with_padding=self.with_padding)
        def generator():
            for G in all_graphs:
                yield (G[0], G[1])

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(input_dtype, target_dtype),
            output_shapes=(input_shape, target_shape),
            args=None)

        outname = "{}_{}.tfrec".format(outname, ijob)
        writer = tf.io.TFRecordWriter(outname)
        for data in dataset:
            example = graph.serialize_graph(*data)
            writer.write(example)
        writer.close()
        return ifailed, isaved
        

    def process(self, filename, outname, n_evts_per_record, debug, max_evts, num_workers=1, **kwargs):
        now = time.time()

        all_evts = sum([1 for _ in self.read(filename)])
        all_evts = max_evts if max_evts > 0 and all_evts > max_evts else all_evts

        n_files = all_evts // n_evts_per_record
        if all_evts%n_evts_per_record > 0:
            n_files += 1

        print("In total {} events, write to {} files".format(all_evts, n_files))
        with Pool(num_workers) as p:
            process_fnc = partial(self.subprocess,
                        n_evts_per_record=n_evts_per_record,
                        num_evts=all_evts,
                        filename=filename,
                        outname=outname,
                        debug=debug)
            res = p.map(process_fnc, list(range(n_files)))

        ifailed = sum([x[0] for x in res])
        isaved = sum([x[1] for x in res])
            
        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            isaved, read_time/60.))
        print("{:,} events failed in being converted to graph".format(ifailed))