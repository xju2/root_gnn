"""
Make doublet GraphNtuple
"""
import tensorflow as tf
from graph_nets import graphs
from graph_nets.graphs import GraphsTuple
from root_gnn.src.datasets.hetero_graphs import HeteroGraphsTuple

graph_types = {
    'n_node': tf.int32,
    'n_edge': tf.int32,
    'nodes': tf.float32,
    'edges': tf.float32,
    'receivers': tf.int32,
    'senders': tf.int32,
    'globals': tf.float32,
}

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

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int32_list from a int32 / int64."""
    return tf.train.Feature(int32_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_graph(G1: GraphsTuple, G2: GraphsTuple):
    feature = {}
    for key in ('n_node', 'n_edge', 'receivers', 'senders'):
        feature[key+"_IN"] = _int64_feature(getattr(G1, key))
        feature[key+"_OUT"] = _int64_feature(getattr(G2, key))

    for key in ('nodes', 'edges', 'globals'):
        feature[key+"_IN"] = _float_feature(getattr(G1, key))
        feature[key+"_OUT"] = _float_feature(getattr(G2, key))

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_hetero_graph(G1: HeteroGraphsTuple, G2: HeteroGraphsTuple):
    feature = {}
    for key in ('n_node', 'n_edge', 'receivers', 'senders', 'node_types', 'edge_types'):
        feature[key+"_IN"] = _int64_feature(getattr(G1, key))
        feature[key+"_OUT"] = _int64_feature(getattr(G2, key))

    for key in ('nodes', 'edges', 'globals'):
        feature[key+"_IN"] = _float_feature(getattr(G1, key))
        feature[key+"_OUT"] = _float_feature(getattr(G2, key))

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

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

        # print(field_name, shape, dtype)
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
        # print(field_name, shape, dtype)

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