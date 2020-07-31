"""
Make doublet GraphNtuple
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


max_graph_dict = {
    "eta2-phi12": [3500, 56500],
    'eta2-phi1': [37000, 680000],
    # 'eta1-phi1': [74000, 1430000] # mine doublets
    # 'eta1-phi1': [96500, 566000] # embeded with duplicated nodes
    # 'eta1-phi1': [68700, 565900] # embeded without duplicated nodes
    'eta1-phi1': [58112, 417664]    # embeded keep 99% of graphs
}

def get_max_graph_size(n_eta, n_phi):
    try:
        res = max_graph_dict['eta{}-phi{}'.format(n_eta, n_phi)]
    except KeyError:
        print("{} and {} is unknown".format(n_eta, n_phi))
        res = max_graph_dict['eta2-phi1']
    return res


graph_types = {
    'n_node': tf.int32,
    'n_edge': tf.int32,
    'nodes': tf.float32,
    'edges': tf.float32,
    'receivers': tf.int32,
    'senders': tf.int32,
    'globals': tf.float32,
}

def padding(g, max_nodes, max_edges, do_concat=True):
    f_dtype = np.float32
    n_nodes = np.sum(g.n_node)
    n_edges = np.sum(g.n_edge)
    n_nodes_pad = max_nodes - n_nodes
    n_edges_pad = max_edges - n_edges

    if n_nodes_pad < 0:
        raise ValueError("Max Nodes: {}, but {} nodes in graph".format(max_nodes, n_nodes))

    if n_edges_pad < 0:
        raise ValueError("Max Edges: {}, but {} edges in graph".format(max_edges, n_edges))

    # padding edges all pointing to the last node
    # TODO: make the graphs more general <xju>
    edges_idx = tf.constant([0] * n_edges_pad, dtype=np.int32)
    # print(edges_idx)
    zeros = np.array([0.0], dtype=f_dtype)
    n_node_features = g.nodes.shape[-1]
    n_edge_features = g.edges.shape[-1]
    # print("input graph global: ", g.globals.shape)
    # print("zeros: ", np.zeros_like(g.globals.numpy()))
    # print("input edges", n_edges, "padding edges:", n_edges_pad)

    padding_datadict = {
        "n_node": n_nodes_pad,
        "n_edge": n_edges_pad,
        "nodes": np.zeros((n_nodes_pad, n_node_features), dtype=f_dtype),
        'edges': np.zeros((n_edges_pad, n_edge_features), dtype=f_dtype),
        'receivers': edges_idx,
        'senders': edges_idx,
        'globals':zeros
    }
    padding_graph = utils_tf.data_dicts_to_graphs_tuple([padding_datadict])
    if do_concat:
        return utils_tf.concat([g, padding_graph], axis=0)
    else:
        return padding_graph

def splitting(g_input, n_devices, verbose=False):
    """
    split the graph so that edges are distributed
    to all devices, of which the number is specified by n_devices.
    """
    def reduce_edges(gg, n_edges_fixed, edge_slice):
        edges = gg.edges[edge_slice]
        return gg.replace(n_edge=tf.convert_to_tensor(np.array([edges.shape[0]]), tf.int32), 
            edges=edges,
            receivers=gg.receivers[edge_slice],
            senders=gg.senders[edge_slice])

    n_edges = tf.math.reduce_sum(g_input.n_edge)
    n_nodes = tf.math.reduce_sum(g_input.n_node)
    splitted_graphs = []
    n_edges_fixed = n_edges // n_devices
    if verbose:
        print("Total {:,} Edges in input graph, splitted into {:,} devices".format(n_edges, n_devices))
        print("Resulting each device contains {:,} edges".format(n_edges_fixed))

    for idevice in range(n_devices):
        if idevice < n_devices - 1:
            edge_slice = slice(idevice*n_edges_fixed, (idevice+1)*n_edges_fixed)
        else:
            edge_slice = slice(idevice*n_edges_fixed, n_edges)
        splitted_graphs.append(reduce_edges(g_input, n_edges_fixed, edge_slice))

    return splitted_graphs


def parse_tfrec_function(example_proto):
    features_description = dict(
        [(key+"_IN",  tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS] + 
        [(key+"_OUT", tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS])

    example = tf.io.parse_single_example(example_proto, features_description)
    input_dd = graphs.GraphsTuple(**dict([(key, tf.io.parse_tensor(example[key+"_IN"], graph_types[key]))
        for key in graphs.ALL_FIELDS]))
    out_dd = graphs.GraphsTuple(**dict([(key, tf.io.parse_tensor(example[key+"_OUT"], graph_types[key]))
        for key in graphs.ALL_FIELDS]))
    return input_dd, out_dd

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_graph(G1, G2):
    feature = {}
    for key in graphs.ALL_FIELDS:
        feature[key+"_IN"] = _bytes_feature(tf.io.serialize_tensor(getattr(G1, key)))
        feature[key+"_OUT"] = _bytes_feature(tf.io.serialize_tensor(getattr(G2, key)))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def concat_batch_dim(G):
    """
    G is a GraphNtuple Tensor, with additional dimension for batch-size.
    Concatenate them along the axis for batch
    It only works for batch size of 1
    """
    n_node = tf.reshape(G.n_node, [-1])
    n_edge = tf.reshape(G.n_edge, [-1])
    nodes = tf.reshape(G.nodes, [-1, tf.shape(G.nodes)[-1]])
    edges = tf.reshape(G.edges, [-1, tf.shape(G.edges)[-1]])
    senders = tf.reshape(G.senders, [-1])
    receivers = tf.reshape(G.receivers, [-1])
    globals_ = tf.reshape(G.globals, [-1, tf.shape(G.globals)[-1]])
    return G.replace(n_node=n_node, n_edge=n_edge, nodes=nodes,\
        edges=edges, senders=senders, receivers=receivers, globals=globals_)


def _concat_batch_dim(G):
    """
    G is a GraphNtuple Tensor, with additional dimension for batch-size.
    Concatenate them along the axis for batch
    """
    input_graphs = []
    for ibatch in [0, 1]:
        data_dict = {
            "nodes": G.nodes[ibatch],
            "edges": G.edges[ibatch],
            "receivers": G.receivers[ibatch],
            'senders': G.senders[ibatch],
            'globals': G.globals[ibatch],
            'n_node': G.n_node[ibatch],
            'n_edge': G.n_edge[ibatch],
        }
        input_graphs.append(graphs.GraphsTuple(**data_dict))
        return (tf.add(ibatch, 1), input_graphs)
    print("{} graphs".format(len(input_graphs)))
    return utils_tf.concat(input_graphs, axis=0)


def add_batch_dim(G, axis=0):
    """
    G is a GraphNtuple Tensor, without a dimension for batch.
    Add a dimensioin them along the axis for batch
    """
    n_node = tf.expand_dims(G.n_node, axis=0)
    n_edge = tf.expand_dims(G.n_edge, axis=0)
    nodes = tf.expand_dims(G.nodes, axis=0)
    edges = tf.expand_dims(G.edges, axis=0)
    senders = tf.expand_dims(G.senders, axis=0)
    receivers = tf.expand_dims(G.receivers, 0)
    globals_ = tf.expand_dims(G.globals, 0)
    return G.replace(n_node=n_node, n_edge=n_edge, nodes=nodes,\
        edges=edges, senders=senders, receivers=receivers, globals=globals_)


def data_dicts_to_graphs_tuple(input_dd, target_dd, with_batch_dim=True):
    # if type(input_dd) is not list:
    #     input_dd = [input_dd]
    # if type(target_dd) is not list:
    #     target_dd = [target_dd]
        
    # input_graphs = utils_tf.data_dicts_to_graphs_tuple(input_dd)
    # target_graphs = utils_tf.data_dicts_to_graphs_tuple(target_dd)
    input_graphs = utils_tf.concat(input_dd, axis=0)
    target_graphs = utils_tf.concat(target_dd, axis=0)
    # # fill zeros
    # input_graphs = utils_tf.set_zero_global_features(input_graphs, 1, dtype=tf.float64)
    # target_graphs = utils_tf.set_zero_global_features(target_graphs, 1, dtype=tf.float64)
    # target_graphs = utils_tf.set_zero_node_features(target_graphs, 1, dtype=tf.float64)
    
    # expand dims
    if with_batch_dim:
        input_graphs = add_batch_dim(input_graphs)
        target_graphs = add_batch_dim(target_graphs)
    return input_graphs, target_graphs


## TODO place holder for the case PerReplicaSpec is exported.
def get_signature(graphs_tuple_sample):
    from graph_nets import graphs

    graphs_tuple_description_fields = {}

    for field_name in graphs.ALL_FIELDS:
        per_replica_sample = getattr(graphs_tuple_sample, field_name)
        def spec_from_value(v):
            shape = list(v.shape)
            dtype = list(v.dtype)
            if shape:
                shape[1] = None
            return tf.TensorSpec(shape=shape, dtype=dtype)

        per_replica_spec = tf.distribute.values.PerReplicaSpec(
            *(spec_from_value(v) for v in per_replica_sample.values)
        )

        graphs_tuple_description_fields[field_name] = per_replica_spec
    return graphs.GraphsTuple(**graphs_tuple_description_fields)


def specs_from_graphs_tuple(
    graphs_tuple_sample, with_batch_dim=False,
    dynamic_num_graphs=False,
    dynamic_num_nodes=True,
    dynamic_num_edges=True,
    description_fn=tf.TensorSpec,
    ):
    graphs_tuple_description_fields = {}
    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]

    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(graphs_tuple_sample, field_name)
        if field_sample is None:
            raise ValueError(
                "The `GraphsTuple` field `{}` was `None`. All fields of the "
                "`GraphsTuple` must be specified to create valid signatures that"
                "work with `tf.function`. This can be achieved with `input_graph = "
                "utils_tf.set_zero_{{node,edge,global}}_features(input_graph, 0)`"
                "to replace None's by empty features in your graph. Alternatively"
                "`None`s can be replaced by empty lists by doing `input_graph = "
                "input_graph.replace({{nodes,edges,globals}}=[]). To ensure "
                "correct execution of the program, it is recommended to restore "
                "the None's once inside of the `tf.function` by doing "
                "`input_graph = input_graph.replace({{nodes,edges,globals}}=None)"
                "".format(field_name))

        shape = list(field_sample.shape)
        dtype = field_sample.dtype

        # If the field is not None but has no field shape (i.e. it is a constant)
        # then we consider this to be a replaced `None`.
        # If dynamic_num_graphs, then all fields have a None first dimension.
        # If dynamic_num_nodes, then the "nodes" field needs None first dimension.
        # If dynamic_num_edges, then the "edges", "senders" and "receivers" need
        # a None first dimension.
        if shape:
            if with_batch_dim:
                shape[1] = None
            elif (dynamic_num_graphs \
                or (dynamic_num_nodes \
                    and field_name == graphs.NODES) \
                or (dynamic_num_edges \
                    and field_name in edge_dim_fields)): shape[0] = None

        print(field_name, shape, dtype)
        graphs_tuple_description_fields[field_name] = description_fn(
            shape=shape, dtype=dtype)

    return graphs.GraphsTuple(**graphs_tuple_description_fields)


def dtype_shape_from_graphs_tuple(input_graph, with_batch_dim=False, with_padding=True, debug=False, with_fixed_size=False):
    graphs_tuple_dtype = {}
    graphs_tuple_shape = {}

    edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]
    for field_name in graphs.ALL_FIELDS:
        field_sample = getattr(input_graph, field_name)
        shape = list(field_sample.shape)
        dtype = field_sample.dtype
        print(field_name, shape, dtype)

        if not with_fixed_size and shape and not with_padding:
            if with_batch_dim:
                shape[1] = None
            else:
                if field_name == graphs.NODES or field_name in edge_dim_fields:
                    shape[0] = None

        graphs_tuple_dtype[field_name] = dtype
        graphs_tuple_shape[field_name] = tf.TensorShape(shape)
        if debug:
            print(field_name, shape, dtype)
    
    return graphs.GraphsTuple(**graphs_tuple_dtype), graphs.GraphsTuple(**graphs_tuple_shape)


# TODO: use one-hot-encoding to add layer info for nodes,
# attach the flattened encoding to node features
def make_graph_ntuples(hits, segments, n_eta, n_phi,
                    node_features=['r', 'phi', 'z'],
                    edge_features=None,
                    dphi=0.0, deta=0.0, with_pad=False, verbose=False):
    phi_range = (-np.pi, np.pi)
    eta_range = (-5, 5)
    phi_edges = np.linspace(*phi_range, num=n_phi+1)
    eta_edges = np.linspace(*eta_range, num=n_eta+1)
    if edge_features is None:
        n_edge_features = 1
    else:
        n_edge_features = len(edge_features)
    n_node_features = len(node_features)
    N_MAX_NODES, N_MAX_EDGES = get_max_graph_size(n_eta, n_phi)

    n_graphs = n_eta * n_phi
    if verbose:
        print("{} graphs".format(n_graphs))

    f_dtype = np.float32
    zeros = np.array([0.0], dtype=f_dtype)
    def make_subgraph(mask):
        hit_id = hits[mask].hit_id.values
        sub_doublets = segments[segments.hit_id_in.isin(hit_id) & segments.hit_id_out.isin(hit_id)]

        # TODO: include all edges, uncomment following lines. <>
        # sub_doublets = segments[segments.hit_id_in.isin(hit_id)]
        # # extend the hits to include the hits used in the sub-doublets.
        # hit_id = hits[mask | hits.hit_id.isin(sub_doublets.hit_id_out.values)].hit_id.values

        n_nodes = hit_id.shape[0]
        n_edges = sub_doublets.shape[0]
        nodes = hits[mask][node_features].values.astype(f_dtype)
        if edge_features is None:
            edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
        else:
            edges = sub_doublets[edge_features].values.astype(f_dtype)
        # print(nodes.dtype)

        hits_id_dict = dict([(hit_id[idx], idx) for idx in range(n_nodes)])
        in_hit_ids = sub_doublets.hit_id_in.values
        out_hit_ids = sub_doublets.hit_id_out.values
        senders   = [hits_id_dict[in_hit_ids[idx]]  for idx in range(n_edges)]
        receivers = [hits_id_dict[out_hit_ids[idx]] for idx in range(n_edges)]
        if verbose:
            print("\t{} nodes and {} edges".format(n_nodes, n_edges))
        senders = np.array(senders)
        receivers = np.array(receivers)

        input_datadict = {
            "n_node": n_nodes,
            'n_edge': n_edges,
            'nodes': nodes,
            'edges': edges,
            'senders': senders,
            'receivers': receivers,
            'globals': zeros
        }
        target_datadict = {
            "n_node": n_nodes,
            'n_edge': n_edges,
            'nodes': np.zeros((n_nodes, n_node_features), dtype=f_dtype),
            'edges': np.expand_dims(sub_doublets.solution.values.astype(f_dtype), axis=1),
            'senders': senders,
            'receivers': receivers,
            'globals': zeros
        }
        
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])

        if with_pad:
            input_graph = padding(input_graph, N_MAX_NODES, N_MAX_EDGES)
            target_graph = padding(target_graph, N_MAX_NODES, N_MAX_EDGES)

        return [(input_graph, target_graph)]

        # # HACK, reverse a fixed number of edges, for testing distributing edges, <xju>
        # # distribute the edges to 8 devices
        # n_devices = 8
        # def reduce_edges(gg, n_edges_fixed, edge_slice):
        #     # n_edges_fixed = 170000
        #     return gg.replace(n_edge=tf.convert_to_tensor(np.array([n_edges_fixed]), tf.int32), 
        #         edges=gg.edges[edge_slice],
        #         receivers=gg.receivers[edge_slice],
        #         senders=gg.senders[edge_slice])

        # splitted_graphs = []
        # n_edges_fixed = N_MAX_EDGES // n_devices
        # for idevice in range(n_devices):
        #     edge_slice = slice(idevice*n_edges_fixed, (idevice+1)*n_edges_fixed)
        #     splitted_graphs.append((
        #         reduce_edges(input_graph, n_edges_fixed, edge_slice),
        #         reduce_edges(target_graph, n_edges_fixed, edge_slice)
        #         ))

        # return splitted_graphs
        # add pad graph to have a constant

    all_graphs = []
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]
        phi_max += dphi
        phi_min -= dphi
        phi_mask = (hits.phi > phi_min) & (hits.phi < phi_max)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            eta_min -= deta
            eta_max += deta
            eta_mask = (hits.eta > eta_min) & (hits.eta < eta_max)
            all_graphs += make_subgraph(eta_mask & phi_mask)
    tot_nodes = sum([x.n_node for x, _ in all_graphs])
    tot_edges = sum([x.n_edge for x, _ in all_graphs])
    if verbose:
        print("\t{} nodes and {} edges".format(tot_nodes, tot_edges))
    return all_graphs


class IndexMgr:
    def __init__(self, n_total, training_frac=0.8):
        self.max_tr = int(n_total*training_frac)
        self.total = n_total
        self.n_test = n_total - self.max_tr
        self.tr_idx = 0
        self.te_idx = self.max_tr

    def next(self, is_training=False):
        if is_training:
            self.tr_idx += 1
            if self.tr_idx > self.max_tr:
                self.tr_idx = 0
            return self.tr_idx
        else:
            self.te_idx += 1
            if self.te_idx > self.total:
                self.te_idx = self.max_tr
            return self.te_idx


class DoubletGraphGenerator:
    def __init__(self, n_eta, n_phi, node_features, edge_features, \
        with_batch_dim=True, with_pad=False, split_edge=False, n_devices=None, verbose=False):
        self.n_eta = n_eta
        self.n_phi = n_phi
        self.node_features = node_features
        self.edge_features = edge_features
        self.verbose = verbose
        self.graphs = []
        self.evt_list = []
        self.idx_mgr = None
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_batch_dim = with_batch_dim
        self.with_pad = with_pad
        self.tf_record_idx = 0
        self.n_evts = 0
        if split_edge and not n_devices:
            raise ValueError("split_edge set to True, please provide n_devices")
        self.split_edge = split_edge
        self.n_devices = n_devices

        print("DoubletGraphGenerator settings: \
            \n\twith_batch_dim={},\
            \n\twith_pad={},\
            \n\tsplit_edge={},\
            \n\tn_devices={},\
            \n\tverbose={}".format(
            self.with_batch_dim, self.with_pad, self.split_edge, self.n_devices, self.verbose
        ))
        if self.with_pad:
            print("all graphs are padded to {} nodes and {} edges".format(*get_max_graph_size(self.n_eta, self.n_phi)))
        else:
            print("dynamic graph sizes will be generated")
        self.n_files_saved = 0

    def add_file(self, hit_file, doublet_file):
        """
        hit-file: pandas dataframe for hits, minimum  columns:
            ['hit_id', 'eta', 'phi'] + [node-features],
        doublet-file: pandas dataframe for edges, minimum columns:
            ['hit_id_in', 'hit_id_out', 'solution'], where 'solution' identifies if the edge is true
            + [edge_features]
        """
        now = time.time()
        with pd.HDFStore(hit_file, 'r') as hit_store:
            n_evts = len(list(hit_store.keys()))
            self.n_evts += n_evts
            with pd.HDFStore(doublet_file, 'r') as doublet_store:
                n_doublet_keys = len(list(doublet_store.keys()))
                for key in hit_store.keys(): # loop over events
                    doublets = []
                    try:
                        # FIXME: hardced 9 layers. <>
                        for ipair in range(9):
                            pair_key = key+'/pair{}'.format(ipair)
                            doublets.append(doublet_store[pair_key])
                    except KeyError:
                        continue
                    hit = hit_store[key]
                    doublets = pd.concat(doublets)
                    all_graphs = make_graph_ntuples(
                                        hit, doublets,
                                        self.n_eta, self.n_phi,
                                        node_features=self.node_features,
                                        edge_features=self.edge_features,
                                        with_pad=self.with_pad,
                                        verbose=self.verbose)
                    self.n_graphs_per_evt = len(all_graphs)
                    self.graphs += all_graphs
                    self.evt_list.append(key)

        self.tot_data = len(self.graphs)
        self.idx_mgr = IndexMgr(self.tot_data)
        read_time = time.time() - now
        print("DoubletGraphGenerator added {} events, Total {} graphs, in {:.1f} mins".format(n_evts, len(self.graphs), read_time/60.))

    def add_daniels_doublets(self, base_filename, evtid, all_hits, n_sections=8):
        # base /project/projectdirs/m3443/usr/dtmurnane/doublets/high_fullsplit/event{}_{}
        # node_features = ['r', 'phi', 'z']

        # NOTE: this uses default [r, phi, z], it yields bad GNN results. <xju>
        solutions = []
        doublets = []
        hit_ids = []
        for isec in range(n_sections):
            id_filename = base_filename.format(evtid, isec)+"_ID.npz"
            id_array = np.load(id_filename)
            hit_id = id_array['I']
            hit_ids.append(hit_id)

            file_name = base_filename.format(evtid, isec)+".npz"
            array = np.load(file_name)
            solutions.append(array['y'])
            edge = array['e']
            new_edge = np.apply_along_axis(lambda x: hit_id[x], 1, edge)
            doublets.append(new_edge)
        hit_ids = np.unique(np.concatenate(hit_ids))
        doublets = np.concatenate(doublets, axis=1)
        solutions = np.concatenate(solutions)
        hits = all_hits[all_hits.hit_id.isin(hit_ids)][list(set(['hit_id', 'eta', 'phi']+self.node_features))]

        doublets = pd.DataFrame.from_dict({
            "hit_id_in": doublets[0], 
            "hit_id_out": doublets[1],
            'solution': solutions,
        })
        all_graphs = make_graph_ntuples(
            hits, doublets, self.n_eta, self.n_phi,
            node_features=self.node_features,
            edge_features=None,
            with_pad=False,
            verbose=self.verbose
        )
        
        input_graphs = utils_tf.concat([x[0] for x in all_graphs], axis=0)
        target_graphs = utils_tf.concat([x[1] for x in all_graphs], axis=0)
        if self.with_pad:
            max_nodes, max_edges = get_max_graph_size(self.n_eta, self.n_phi)
            try:
                input_graphs = padding(input_graphs, max_nodes, max_edges)
                target_graphs = padding(target_graphs, max_nodes, max_edges)
            except ValueError:
                print("Discarding the graph with size greater than the maximum")
                return
        
        if self.split_edge:
            splitted_inputs = splitting(input_graphs, self.n_devices)
            splitted_targets = splitting(target_graphs, self.n_devices)
            all_graphs = [(x, y) for x, y in zip(splitted_inputs, splitted_targets)]
        else:
            all_graphs = [(input_graphs, target_graphs)]

        self.n_graphs_per_evt = len(all_graphs)
        self.graphs += all_graphs
        self.evt_list.append(str(evtid))
        self.tot_data = len(self.graphs)
        self.idx_mgr = IndexMgr(self.tot_data)
        self.n_evts += 1


    def _get_signature(self):
        if self.input_dtype and self.target_dtype:
            return
        ex_input, ex_target = self.create_graph(num_graphs=1)
        self.input_dtype, self.input_shape = dtype_shape_from_graphs_tuple(
            ex_input, self.with_batch_dim, self.verbose)
        self.target_dtype, self.target_shape = dtype_shape_from_graphs_tuple(
            ex_target, self.with_batch_dim, self.verbose)
        
    def _graph_generator(self, is_training=True): # one graph a dataset
        min_idx, max_idx = 0, int(self.tot_data * 0.8)

        if not is_training:
            min_idx, max_idx = int(self.tot_data*0.8), self.tot_data-1

        for idx in range(min_idx, max_idx):
            yield data_dicts_to_graphs_tuple([self.graphs[idx][0]], [self.graphs[idx][1]], self.with_batch_dim)


    def create_dataset(self, is_training=True):
        self._get_signature()
        dataset = tf.data.Dataset.from_generator(
            self._graph_generator,
            output_types=(self.input_dtype, self.target_dtype),
            output_shapes=(self.input_shape, self.target_shape),
            args=(is_training,)
        )
        return dataset

    # FIXME: 
    # everytime check if one event is completely used (used all subgraphs)
    # shuffle the events, but feed the subgraphs in order
    def create_graph(self, num_graphs, is_training=True):
        if not self.idx_mgr:
            raise ValueError("No Doublet Graph is created")

        inputs = []
        targets = []
        for _ in range(num_graphs):
            idx = self.idx_mgr.next(is_training)
            input_dd, target_dd =  self.graphs[idx]
            inputs.append(input_dd)
            targets.append(target_dd)

        return utils_tf.concat(inputs, axis=0), utils_tf.concat(targets, axis=0)
        # return data_dicts_to_graphs_tuple(inputs, targets, self.with_batch_dim)

    
    def write_tfrecord(self, filename, n_evts_per_record=10):
        self._get_signature()
        def generator():
            for G in self.graphs:
                yield (G[0], G[1])
                # yield data_dicts_to_graphs_tuple([G[0]], [G[1]], self.with_batch_dim)
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(self.input_dtype, self.target_dtype),
            output_shapes=(self.input_shape, self.target_shape),
            args=None
        )

        # n_graphs_per_evt = self.n_eta * self.n_phi
        # n_evts = int(self.tot_data//n_graphs_per_evt)
        # n_evts = len(self.evt_list)
        n_graphs_per_evt = self.n_graphs_per_evt
        n_evts = self.n_evts
        n_files = n_evts//n_evts_per_record
        if n_evts%n_evts_per_record > 0:
            n_files += 1

        print("In total {} graphs, {} graphs per event".format(self.tot_data, n_graphs_per_evt))
        print("In total {} events, write to {} files".format(n_evts, n_files))
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

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
            example = serialize_graph(*data)
            writer.write(example)
        self.n_files_saved += n_files
