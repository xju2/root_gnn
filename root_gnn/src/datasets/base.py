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

    def _num_evts(self, filename: str) -> int:
        """
        return total number of events in the filename
        """
        return sum([1 for _ in self.read(filename)])

    def make_graph(self, event, debug):
        """
        Convert the event into a graphs_tuple. 
        """
        raise NotImplementedError

    def subprocess(self, ijob, n_evts_per_record, filename, outname, overwrite, debug):
       
        outname = "{}_{}.tfrec".format(outname, ijob)
        if os.path.exists(outname) and not overwrite:
            print(outname,"is there. skip...")
            return 0, n_evts_per_record

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

        writer = tf.io.TFRecordWriter(outname)
        for data in dataset:
            example = graph.serialize_graph(*data)
            writer.write(example)
        writer.close()
        return ifailed, isaved
        

    def process(self, filename, outname, n_evts_per_record, debug, max_evts, num_workers=1, overwrite=False, **kwargs):
        now = time.time()

        all_evts = self._num_evts(filename)
        all_evts = max_evts if max_evts > 0 and all_evts > max_evts else all_evts

        n_files = all_evts // n_evts_per_record
        if all_evts%n_evts_per_record > 0:
            n_files += 1

        print("Total {:,} events are requested to be written to {:,} files with {:,} workers".format(all_evts, n_files, num_workers))
        out_dir = os.path.abspath(os.path.dirname(outname))
        os.makedirs(out_dir, exist_ok=True)
        
        if num_workers == 1:
            ifailed, isaved = self.subprocess(0, n_evts_per_record, filename, outname, overwrite, debug)
        else:
            with Pool(num_workers) as p:
                process_fnc = partial(self.subprocess,
                            n_evts_per_record=n_evts_per_record,
                            filename=filename,
                            outname=outname,
                            overwrite=overwrite,
                            debug=debug)
                res = p.map(process_fnc, list(range(n_files)))

            ifailed = sum([x[0] for x in res])
            isaved = sum([x[1] for x in res])
            
        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            isaved, read_time/60.))
        print("{:,} events failed in being converted to graph".format(ifailed))