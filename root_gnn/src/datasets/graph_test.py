#!/usr/bin/env python
"""
Test make_graph_ntuples
"""
from heptrkx.dataset.graph import make_graph_ntuples
from heptrkx.dataset.graph import DoubletGraphGenerator
from heptrkx.dataset.graph import specs_from_graphs_tuple
# from heptrkx.dataset.graph import N_MAX_EDGES, N_MAX_NODES
from heptrkx.dataset.graph import parse_tfrec_function
import pandas as pd
import tensorflow as tf
from graph_nets import graphs
from heptrkx.dataset import graph



hit_file_name = '/global/cscratch1/sd/xju/heptrkx/codalab/inputs/hitfiles/evt21001_test.h5'
doublet_file_name = '/global/cscratch1/sd/xju/heptrkx/codalab/inputs/doublet_files/doublets-evt21001_test.h5'

# hit_file_name = '/Volumes/GoogleDrive/My Drive/HEPTrk/Data//hitfiles/evt21001_test.h5'
# doublet_file_name = '/Volumes/GoogleDrive/My Drive/HEPTrk/Data//doublet_files/doublets-evt21001_test.h5'

def test_graph():
    with pd.HDFStore(hit_file_name, 'r') as hit_store:
        print(hit_store.keys())
        with pd.HDFStore(doublet_file_name, 'r') as doublet_store:
            print(doublet_store.keys())
            key = 'evt21001'
            hit = hit_store[key]
            doublets = []
            for ipair in range(9):
                pair_key = key+'/pair{}'.format(ipair)
                doublets.append(doublet_store[pair_key])
            doublets = pd.concat(doublets)
            print("{:,} hits and {:,} doublets".format(hit.shape[0], doublets.shape[0]))

            all_graphs = make_graph_ntuples(
                                hit, doublets, 2, 10,
                                verbose=True)

    return all_graphs

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_graph(G, G2):
    feature = {}
    for key in graphs.ALL_FIELDS:
        data = getattr(G, key)
        print("key:", key, data.shape)
        feature[key+"_IN"] = _bytes_feature(tf.io.serialize_tensor(getattr(G, key)))
        feature[key+"_OUT"] = _bytes_feature(tf.io.serialize_tensor(getattr(G2, key)))
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


tf_file_name = 'test.tfrecord_0.tfrec'
def write_tfrecord():
    print("Is eager mode {}".format(tf.executing_eagerly()))
    global_batch_size = 1
    graph_gen = DoubletGraphGenerator(2, 12, ['x', 'y', 'z'], ['deta', 'dphi'], with_pad=True, with_batch_dim=False)
    graph_gen.add_file(hit_file_name, doublet_file_name)
    # training_dataset = graph_gen.create_dataset(is_training=True)

    graph_gen.write_tfrecord(tf_file_name)

    # with tf.io.TFRecordWriter(tf_file_name) as writer:
    #     for data in training_dataset:
    #         example = serialize_graph(*data)
    #         writer.write(example)

def read_tfrecord():
    filenames = [tf_file_name]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # print(raw_dataset)
    # features_description = {
    #     'nodes': tf.io.FixedLenFeature([1, N_MAX_NODES, 3], tf.float32),
    #     'edges': tf.io.FixedLenFeature([1, N_MAX_EDGES, 2], tf.float32),
    #     'receivers': tf.io.FixedLenFeature([1, N_MAX_EDGES], tf.int64),
    #     'senders': tf.io.FixedLenFeature([1, N_MAX_EDGES], tf.int64),
    #     'globals': tf.io.FixedLenFeature([1, 1], tf.float32),
    #     'n_node': tf.io.FixedLenFeature([1, 1], tf.float32),
    #     'n_edge': tf.io.FixedLenFeature([1, 1], tf.float32)
    # }
    features_description = dict(
        [(key+"_IN",  tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS] + 
        [(key+"_OUT", tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS])

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, features_description)
        n_node = tf.io.parse_tensor(example['globals_IN'], tf.float64)
        return n_node

    parsed_dataset = raw_dataset.map(parse_tfrec_function)
    for raw_record in parsed_dataset.take(1):
        print(raw_record)

        # print("parse example")
        # example = tf.io.parse_single_example(raw_record, features_description)
        # # print(example)
        # nodes = tf.io.parse_tensor(example['edges_OUT'], tf.float16)
        # print(nodes)
        # for field in graphs.ALL_FIELDS:
        #     print(field)
        #     nodes = tf.io.parse_tensor(example[field+'_OUT'], tf.float64)
        #     print(nodes)
        # decoded_ex = tf.io.parse_single_example(example, features_description)
        # print(decoded_ex.features.feature['n_node'])



def test_dataset():
    print("Is eager mode {}".format(tf.executing_eagerly()))
    global_batch_size = 1
    graph_gen = DoubletGraphGenerator(2, 8, ['x', 'y', 'z'], ['deta', 'dphi'], with_pad=True)
    graph_gen.add_file(hit_file_name, doublet_file_name)
    training_dataset = graph_gen.create_dataset(is_training=True)

    # serialized_dataset = training_dataset.map(tf_serialize_graph)

    # file_name = 'test.tfrecord'
    # writer = tf.data.experimental.TFRecordWriter(file_name)
    # writer.write(serialized_dataset)

    for g_input, g_target in training_dataset.take(1):
        print("Input Graph")
        print(g_input)
        print("Target Graph")
        print(g_target)
        g1, g2 = serialize_graph(g_input, g_target)
        print(g1)

    # training_dataset = training_dataset.repeat().shuffle(2048).batch(global_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # print(list(training_dataset.take(1).as_numpy_iterator()))

    # testing_dataset = graph_gen.create_dataset()
    # print(list(testing_dataset.take(1).as_numpy_iterator()))

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # dist_training_dataset = mirrored_strategy.experimental_distribute_dataset(training_dataset)
    # with mirrored_strategy.scope():
    #     for inputs in dist_training_dataset:
    #         input, target = inputs
    #         print(input.n_node)
    #         break

def test_signature():
    with_batch_dim = False
    graph_gen = DoubletGraphGenerator(2, 8, ['x', 'y', 'z'], ['deta', 'dphi'], \
        with_batch_dim=with_batch_dim)
    graph_gen.add_file(hit_file_name, doublet_file_name)
    in_graphs, out_graphs = graph_gen.create_graph(1, is_training=True)

    input_signature = (
        specs_from_graphs_tuple(in_graphs, with_batch_dim),
        specs_from_graphs_tuple(out_graphs, with_batch_dim)
    )
    print("Input graph: ", in_graphs)
    print("Input signature: ", input_signature[0])
    print("Target graph: ", out_graphs)
    print("Target signature:", input_signature[1])

def test_mask():
    with_batch_dim = False
    with_pad = True
    graph_gen = DoubletGraphGenerator(2, 8, ['x', 'y', 'z'], ['deta', 'dphi'], \
        with_batch_dim=with_batch_dim, with_pad=with_pad)
    graph_gen.add_file(hit_file_name, doublet_file_name)

    input_graph, out_graph = graph_gen.create_graph(1, is_training=True)
    print("Edges:", out_graph.n_edge)

    row_index = tf.range(tf.math.reduce_sum(out_graph.n_edge))
    n_valid_edges = out_graph.n_edge[0]
    mask = tf.cast(row_index < n_valid_edges, tf.float16)
    mask = tf.expand_dims(mask, axis=1)
    print("Mask:", mask)
    print("Mask shape:", tf.shape(mask))
    print("Edge shape", tf.shape(out_graph.edges))
    print("Edge:", out_graph.edges)

    real_weight = 100
    fake_weight = 1.0

    weights = out_graph.edges * real_weight + (1 - out_graph.edges) * fake_weight
    print("Weight shape:", tf.shape(weights))


def test_concat():
    file_names = ['/global/cfs/cdirs/m3443/usr/xju/heptrkx/codalab/tfdata_doublets/one_evt_24regions/doublets_24regions_111evts_0.tfrec']
    raw_dataset = tf.data.TFRecordDataset(file_names)
    training_dataset = raw_dataset.map(parse_tfrec_function)

    global_batch_size = 1
    AUTO = tf.data.experimental.AUTOTUNE
    training_dataset = training_dataset.batch(global_batch_size).prefetch(AUTO)
    training_viz_iterator = training_dataset.as_numpy_iterator()
    g1, g2 = next(training_viz_iterator)
    print(g1.edges.shape)
    g1_merged = graph.concat_batch_dim(g1)
    g2_merged = graph.concat_batch_dim(g2)
    print(g1_merged)
    print(g2_merged)

def test_gs_tf():
    # file_names = ['gs://gnn-v1/one_evt_24regions_padding/doublets_24regions_110evts_noPadding_0.tfrec']
    file_names = tf.io.gfile.glob("gs://gnn-v1/one_evt_24regions_padding/doublets_24regions_110evts_noPadding_*.tfrec")
    print(file_names)
    raw_dataset = tf.data.TFRecordDataset(file_names)
    training_dataset = raw_dataset.map(graph.parse_tfrec_function)
    global_batch_size = 32
    training_dataset = training_dataset.batch(global_batch_size, drop_remainder=True).prefetch(1)
    num_batches = 0
    for inputs in training_dataset:
        num_batches += 1

    print("total batches:", num_batches)

def test_edge_distributed():
    with_batch_dim = False
    with_pad = False
    graph_gen = DoubletGraphGenerator(1, 1, ['x', 'y', 'z'], ['deta', 'dphi'], \
        with_batch_dim=with_batch_dim, with_pad=with_pad)
    graph_gen.add_file(hit_file_name, doublet_file_name)

    g = graph_gen.graphs[0]
    print(g[0].n_node, g[0].n_edge)

def test_daniel_graph():
    base_filename = '/project/projectdirs/m3443/usr/dtmurnane/doublets/high_fullsplit/event{}_{}'
    evtid = 9999

    with_batch_dim = False
    with_pad = False
    graph_gen = DoubletGraphGenerator(1, 1, ['x', 'y', 'z'], ['deta', 'dphi'], \
        with_batch_dim=with_batch_dim, with_pad=with_pad)
    graph_gen.add_daniels_doublets(base_filename, evtid)
    print(graph_gen.graphs[0][0])

def test_split():
    input_tfdata = "/global/cscratch1/sd/xju/heptrkx/codalab/Daniel_Doublets_RemoveDuplicatedHits/embedded_edges_correct_0.tfrec"
    file_names = tf.io.gfile.glob(input_tfdata)
    print(file_names)
    n_devices = 8
    raw_dataset = tf.data.TFRecordDataset(file_names)
    training_dataset = raw_dataset.map(graph.parse_tfrec_function)
    for data in training_dataset.take(1).as_numpy_iterator():
        input_tr, target_tr = data
        splitted_graphs = graph.splitting(input_tr, n_devices, verbose=True)
        n_total_edges = sum([tf.math.reduce_sum(x.n_edge) for x in splitted_graphs])
        n_total_receivers = sum([x.receivers.shape[0] for x in splitted_graphs])
        print("total edges summed from splitted graphs:", n_total_edges.numpy())
        print("total receivers:", n_total_receivers)

if __name__ == "__main__":
    # test_graph()
    # test_dataset()
    # test_signature()
    # test_mask()
    # write_tfrecord()
    # read_tfrecord()
    # test_concat()
    # test_gs_tf()
    # test_edge_distributed()
    # test_daniel_graph()
    test_split()