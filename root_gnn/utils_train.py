
import numpy as np
import sklearn.metrics

import tensorflow as tf

from graph_nets import utils_tf
from graph_nets import utils_np

import yaml

def create_feed_dict(generator, batch_size, input_ph, target_ph, is_trained=True):
    inputs, targets = generator(batch_size, is_trained)
    input_graphs  = utils_np.data_dicts_to_graphs_tuple(inputs)
    target_graphs = utils_np.data_dicts_to_graphs_tuple(targets)
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}

    return feed_dict


def create_loss_ops(target_op, output_ops):
    # only use edges
    loss_ops = [
        tf.losses.log_loss(target_op.globals, output_op.globals)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


def eval_output(target, output):

    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)

    test_target = []
    test_pred = []
    for td, od in zip(tdds, odds):
        test_target.append(td['edges'])
        test_pred.append(od['edges'])

    test_target = np.concatenate(test_target, axis=0)
    test_pred   = np.concatenate(test_pred,   axis=0)
    return test_pred, test_target


def compute_matrics(target, output):
    test_pred, test_target = eval_output(target, output)
    thresh = 0.5
    y_pred, y_true = (test_pred > thresh), (test_target > thresh)
    return sklearn.metrics.precision_score(y_true, y_pred), sklearn.metrics.recall_score(y_true, y_pred)


def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)
