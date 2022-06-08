"""
base class defines the procedure with that the TFrecord data is produced.
"""
import time
import os
from multiprocessing import Pool
from functools import partial
from typing import Optional

import tensorflow as tf
from root_gnn.src.datasets import graph

def linecount(filename):
    return sum([1 for lin in open(filename)])
    # out = subprocess.Popen(['wc', '-l', filename],
    #                      stdout=subprocess.PIPE,
    #                      stderr=subprocess.STDOUT
    #                      ).communicate()[0]
    # return int(out.partition(b' ')[0])

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

    def read(self, filename, start_entry: Optional[int] = 0, nevts: Optional[int] = -1):
        """
        read the file from `filename` and return an event
        """
        raise NotImplementedError

    def _num_evts(self, filename: str) -> int:
        """
        return total number of events in the filename
        """
        return sum([1 for _ in self.read(filename)])

    def make_graph(self, event, debug, connectivity=None):
        """
        Convert the event into a graphs_tuple. 
        """
        raise NotImplementedError

    def subprocess(self, ijob, n_evts_per_record, filename, outname,
            overwrite, debug, connectivity=None):
       
        outname = "{}_{}.tfrec".format(outname, ijob)
        if os.path.exists(outname) and not overwrite:
            print(outname,"is there. skip...")
            return 0, n_evts_per_record

        ifailed = 0
        all_graphs = []
        start_entry = ijob * n_evts_per_record
        
        if debug:
            print(">>> Debug 0", ijob)
        
        t0 = time.time()
        jevt = 0
        kgraphs = 0
        for event in self.read(filename, start_entry, n_evts_per_record):
            gen_graphs = self.make_graph(event, debug, connectivity=connectivity)
            
            if debug:
                print(">>> Debug 1", ijob, jevt, kgraphs)
            
            if len(gen_graphs)==0 or gen_graphs[0][0] == None:
                ifailed += 1
                continue

            all_graphs += gen_graphs
            kgraphs += len(gen_graphs)
            jevt += 1
            
        if debug:
            print(">>> Debug 2", ijob)
            
        isaved = len(all_graphs)
        if isaved > 0:
            ex_input, ex_target = all_graphs[0]
            input_dtype, input_shape = graph.dtype_shape_from_graphs_tuple(
                ex_input, with_padding=self.with_padding)
            target_dtype, target_shape = graph.dtype_shape_from_graphs_tuple(
                ex_target, with_padding=self.with_padding)
            is_hetero_graph = True if hasattr(ex_input, 'node_types') \
                and ex_input.node_types is not None else False

            def generator():
                for G in all_graphs:
                    yield (G[0], G[1])

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=(input_dtype, target_dtype),
                output_shapes=(input_shape, target_shape),
                args=None)
            if debug:
                print(">>> Debug 3", ijob)
            writer = tf.io.TFRecordWriter(outname)
            serializer = graph.serialize_hetero_graph if is_hetero_graph else graph.serialize_graph
            for data in dataset:
                example = serializer(*data)
                writer.write(example)
            writer.close()
            if debug:
                print(">>> Debug 4", ijob)
            t1 = time.time()
            all_graphs = []
            print(f">>> Job {ijob} Finished in {abs(t1-t0)/60:.2f} min")
        else:
            print(ijob, "all failed")
        return ifailed, isaved

    def process(self, filename, outname, n_evts_per_record,
        debug, max_evts, num_workers=1, overwrite=False, connectivity=None, **kwargs):
        now = time.time()

        all_evts = self._num_evts(filename)
        all_evts = max_evts if max_evts > 0 and all_evts > max_evts else all_evts

        n_files = all_evts // n_evts_per_record
        if all_evts%n_evts_per_record > 0:
            n_files += 1

        print("Total {:,} events are requested to be written to "
              "{:,} files with {:,} workers".format(all_evts, n_files, num_workers))
        out_dir = os.path.abspath(os.path.dirname(outname))
        os.makedirs(out_dir, exist_ok=True)
        
        if num_workers < 2:
            ifailed, isaved=0, 0
            for ijob in range(n_files):
                n_failed, n_saved = self.subprocess(
                    ijob, n_evts_per_record, filename, outname, overwrite,
                    debug, connectivity=connectivity)
                ifailed += n_failed
                isaved += n_saved
        else:
            with Pool(num_workers) as p:
                process_fnc = partial(self.subprocess,
                        n_evts_per_record=n_evts_per_record,
                        filename=filename,
                        outname=outname,
                        overwrite=overwrite,
                        debug=debug,
                        connectivity=connectivity)
                res = p.map(process_fnc, list(range(n_files)))

            ifailed = sum([x[0] for x in res])
            isaved = sum([x[1] for x in res])
            
        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            isaved, read_time/60.))
        print("{:,} events failed in being converted to graph".format(ifailed))