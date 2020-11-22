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
    return dataset, tr_filenames

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
    else:
        for dataset in datasets:
            yield dataset

class TrainerBase(object):
    def __init__(self, config, distributed=False, verbose="INFO"):
        pass

    def loss_fcn(self, truth_graph, predict_graph):
        raise NotImplementedError

    def train(self):
        self.setup_training_loop()

        self.train_one_epoch()
        self.ckpt_manager.save()

        self.after_train_one_epoch()
        
    
    def train_one_epoch(self):
        num_batches = 0
        total_loss = 0
        for inputs in loop_dataset(self.data_train):
            inputs_tr, targets_tr = inputs
            total_loss += self.training_step(inputs_tr, targets_tr).numpy()
            num_batches += 1
        return total_loss, num_batches


    def aftet_train_one_epoch():
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

    def apply(self, inputs):
        outputs = self.model(inputs, self.num_iters)
        return outputs

    def setup_training_loop(self):
        input_signature = self.input_signature()

        def update_step(inputs, targets):
            print("Tracing update_step")
            with tf.GradientTape() as tape:
                output_ops = self.apply(inputs)
                loss_ops_tr = self.loss_fcn(targets, output_ops)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(self.num_iters, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, self.model.trainable_variables)
            self.optimizer.apply(gradients, self.model.trainable_variables)
            return loss_op_tr

        self.training_step = tf.function(update_step, input_signature=input_signature)