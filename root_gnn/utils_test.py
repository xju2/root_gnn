"""
utils for testing trained tf model
"""

import tensorflow as tf
from graph_nets import utils_tf

import yaml
import sklearn.metrics
import os
import numpy as np

from .utils_train import load_config
from .utils_train import create_feed_dict
from .utils_train import eval_output

from .prepare import inputs_generator

from . import get_model

import matplotlib.pyplot as plt

ckpt_name = 'checkpoint_{:05d}.ckpt'


def create_trained_model(config_name, input_ckpt=None):
    """
    @config: configuration for train_nx_graph
    """
    # load configuration file
    config = load_config(config_name)
    config_tr = config['train']

    log_every_seconds       = config_tr['time_lapse']
    batch_size = n_graphs   = config_tr['batch_size']   # need optimization
    num_processing_steps_tr = config_tr['n_iters']      ## level of message-passing
    prod_name = config['prod_name']
    if input_ckpt is None:
        input_ckpt = os.path.join(config['output_dir'], prod_name)


    # generate inputs
    generate_input_target = inputs_generator(config['data']['output_nxgraph_dir'])

    # build TF graph
    tf.reset_default_graph()
    model = get_model(config['model']['name'])

    input_graphs, target_graphs = generate_input_target(n_graphs)
    input_ph  = utils_tf.placeholders_from_data_dicts(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_data_dicts(target_graphs, force_dynamic_num_graphs=True)

    output_ops_tr = model(input_ph, num_processing_steps_tr)

    def evaluator(iteration, n_test_graphs=10):
        try:
            sess.close()
        except NameError:
            pass

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(input_ckpt, ckpt_name.format(iteration)))
        odds = []
        tdds = []
        for _ in range(n_test_graphs):
            feed_dict = create_feed_dict(generate_input_target, batch_size, input_ph, target_ph, is_trained=False)
            predictions = sess.run({
                "outputs": output_ops_tr,
                'target': target_ph
            }, feed_dict=feed_dict)
            output = predictions['outputs'][-1]
            target = predictions['target']
            odd, tdd = eval_output(target, output)
            odds.append(odd)
            tdds.append(tdd)
        return np.concatenate(odds), np.concatenate(tdds)

    return evaluator


def plot_metrics(odd, tdd, odd_th=0.5, tdd_th=0.5):
    y_pred, y_true = (odd > odd_th), (tdd > tdd_th)
    accuracy  = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall    = sklearn.metrics.recall_score(y_true, y_pred)

    print('Accuracy:  %.4f' % accuracy)
    print('Precision: %.4f' % precision)
    print('Recall:    %.4f' % recall)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, odd)


    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12,5))

    # Plot the model outputs
    # binning=dict(bins=50, range=(0,1), histtype='step', log=True)
    binning=dict(bins=50, histtype='step', log=True)
    ax0.hist(odd[y_true==False], label='fake', **binning)
    ax0.hist(odd[y_true], label='true', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot the ROC curve
    auc = sklearn.metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], '--')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve, AUC = %.4f' % auc)

    plt.tight_layout()
    plt.savefig('roc_graph_nets.eps')
