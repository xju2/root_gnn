from graph_nets import graphs
import tensorflow as tf

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