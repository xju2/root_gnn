import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.set_verbosity("INFO")
logging.info("TF Version:{}".format(tf.__version__))
try:
    import horovod.tensorflow as hvd
    no_horovod = False
except ModuleNotFoundError:
    logging.warning("No horvod module, cannot perform distributed training")
    no_horovod = True


import os
import six
from types import SimpleNamespace
import pprint
import time
import functools

import numpy as np

from tensorflow.python.profiler import profiler_v2 as profiler

from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from root_gnn.utils import load_yaml
from root_gnn.src.datasets import graph
from root_gnn import model as all_models
from root_gnn import losses

verbosities = ['DEBUG','ERROR', "FATAL", "INFO", "WARN"]
printer = pprint.PrettyPrinter(indent=2)

def read_dataset(filenames):
    """
    Read dataset...
    """
    AUTO = tf.data.experimental.AUTOTUNE
    tr_filenames = tf.io.gfile.glob(filenames)
    n_files = len(tr_filenames)

    dataset = tf.data.TFRecordDataset(tr_filenames)
    dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
    n_graphs = sum([1 for _ in dataset])
    return dataset, n_graphs


def loop_dataset(datasets, batch_size):
    if batch_size > 0:
        in_list = []
        target_list = []
        for dataset in datasets:
            inputs_tr, targets_tr = dataset
            in_list.append(inputs_tr)
            target_list.append(targets_tr)
            if len(in_list) == batch_size:
                inputs_tr = utils_tf.concat(in_list, axis=0)
                targets_tr = utils_tf.concat(target_list, axis=0)
                yield (inputs_tr, targets_tr)
                in_list = []
                target_list = []
    else:
        for dataset in datasets:
            yield dataset


def get_signature(dataset, batch_size):
    with_batch_dim = False
    input_list = []
    target_list = []
    for dd in dataset.take(batch_size).as_numpy_iterator():
        input_list.append(dd[0])
        target_list.append(dd[1])

    inputs = utils_tf.concat(input_list, axis=0)
    targets = utils_tf.concat(target_list, axis=0)
    input_signature = (
        graph.specs_from_graphs_tuple(inputs, with_batch_dim),
        graph.specs_from_graphs_tuple(targets, with_batch_dim),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    )
    return input_signature

class TrainerBase(object):
    def __init__(self, input_dir, output_dir, lr,
                batch_size, num_epochs,
                num_iters,
                decay_lr=True, # if to use decay learning rate...
                decay_lr_start_epoch=10,
                patterns='*', distributed=False, verbose="INFO", *args, **kwargs):
        self.model = None
        self.loss_fcn = None
        self.num_iters = num_iters

        # datasets
        self.input_dir = input_dir
        self.output_dir = output_dir

        # create optimizer
        self.init_lr = lr
        self.lr = tf.Variable(lr, trainable=False, name='lr', dtype=tf.float32)
        self.optimizer = snt.optimizers.Adam(learning_rate=self.lr)
        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
        self.decay_lr_start_epoch = tf.constant(decay_lr_start_epoch, dtype=tf.int32)
        self.decay_lr = decay_lr # if use decay lr

        # perform distributed training
        self.distributed = distributed

        # calcualte metrics to be recorded
        self.metric_dict = {}

    def setup_training_loop(self, model, loss_fcn):
        input_signature = self.input_signature()

        def update_step(inputs, targets):
            print("Tracing update_step")
            with tf.GradientTape() as tape:
                output_ops = model(inputs, self.num_iters)
                loss_ops_tr = loss_fcn(targets, output_ops)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(self.num_iters, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, model.trainable_variables)
            self.optimizer.apply(gradients, model.trainable_variables)
            return loss_op_tr

        self.training_step = tf.function(update_step, input_signature=input_signature)


    def update_step(self, model, loss_fcn):
        self.setup_training_loop()

        self.train_one_epoch()
        self.ckpt_manager.save()

        self.after_train_one_epoch()

    def eval(self, model):
        raise NotImplementedError
        

    def after_train_one_epoch(self):
        pass

    def validate_one_epoch(self):
        for data in loop_dataset(self.data_val):
            inputs, targets = data
            outputs = self.apply(inputs)
            if len(outputs) > 1:
                outputs = outputs[-1]
            self.update_metrics(targets, outputs)


    def load_training_data(self, filenames):
        self.data_train, _ = read_dataset(filenames)
        self.ngraphs_train = sum([1 for _ in self.data_train])

    def load_validating_data(self, filenames):
        self.data_val, _ = read_dataset(filenames)
        self.ngraphs_val =  sum([1 for _ in self.data_val])

    def load_testing_data(self, filenames):
        self.data_test, _ = read_dataset(filenames)
        self.ngraphs_test =  sum([1 for _ in self.data_test])

    def optimizer(self, lr):
        self.optimizer = snt.optimizers.Adam(lr)

    def input_signature(self):
        with_batch_dim = False
        input_list = []
        target_list = []
        for dd in self.data_train.take(self.train_batch_size).as_numpy_iterator():
            input_list.append(dd[0])
            target_list.append(dd[1])

        inputs = utils_tf.concat(input_list, axis=0)
        targets = utils_tf.concat(target_list, axis=0)
        input_signature = (
            graph.specs_from_graphs_tuple(inputs, with_batch_dim),
            graph.specs_from_graphs_tuple(targets, with_batch_dim),
        )
        return input_signature


    def train_one_epoch(self):
        num_batches = 0
        total_loss = 0
        for inputs in loop_dataset(self.data_train):
            inputs_tr, targets_tr = inputs
            total_loss += self.training_step(inputs_tr, targets_tr).numpy()
            num_batches += 1
        return total_loss, num_batches

