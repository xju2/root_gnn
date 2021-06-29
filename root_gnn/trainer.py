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
import argparse

import numpy as np

import sklearn.metrics

from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from root_gnn.utils import load_yaml
from root_gnn.src.datasets import graph
from root_gnn import model as all_models
from root_gnn import losses

verbosities = ['DEBUG','ERROR', "FATAL", "INFO", "WARN"]
printer = pprint.PrettyPrinter(indent=2)

AUTO = tf.data.experimental.AUTOTUNE

def read_dataset(filenames):
    """
    Read dataset...
    """
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


def get_signature(
        dataset, batch_size, with_bool=False,
        dynamic_num_nodes=True,
        dynamic_num_edges=True,
        ):
    """
    Get signature of inputs for the training loop.
    The signature is used by the tf.function
    """
    with_batch_dim = False
    input_list = []
    target_list = []
    for dd in dataset.take(batch_size).as_numpy_iterator():
        input_list.append(dd[0])
        target_list.append(dd[1])

    inputs = utils_tf.concat(input_list, axis=0)
    targets = utils_tf.concat(target_list, axis=0)
    input_signature = (
        graph.specs_from_graphs_tuple(
            inputs, with_batch_dim,
            dynamic_num_nodes=dynamic_num_nodes,
            dynamic_num_edges=dynamic_num_edges
        ),
        graph.specs_from_graphs_tuple(
            targets, with_batch_dim,
            dynamic_num_nodes=dynamic_num_nodes,
            dynamic_num_edges=dynamic_num_edges
        )
    )
    if with_bool:
        input_signature = input_signature + (tf.TensorSpec(shape=[], dtype=tf.bool), )
        
    return input_signature

class TrainerBase(object):
    
    """
    The base class to implement a simple trainer and model.
    
    ...
    
    Important Attributes
    --------------------
    input_dir: The input directory for data
    
    output_dir: The output directory for metrics and outputs
   
    model: The model class to use for training, validating, and
    testing
    
    loss_fcn: The loss function class to use for training,
    validating, and testing
    """

    def __init__(self, input_dir=None, output_dir=None, 
                model=None, loss_fcn=None, loss_name="GlobalLoss", lr=0.0005,
                optimizer=None,
                batch_size=50, num_epochs=5, num_iters=10,
                metric_mode=None,
                early_stop=None, max_attempts=1,
                shuffle_size=1,
                patterns='*', distributed=False, verbose="INFO", 
                **extra_configs):
        """
        TrainerBase constructor, which initializes configurations, hyperparameters,
        and metrics.
        
        Parameters
        ----------
        Sets input, output, model and loss, as well as relevant hyperparameters.
        The user can directly unpack the arguments from the ArgumentParser
        created in get_arg_parser().
        """
        self.input_dir = input_dir
        self.train_dir = os.path.join(self.input_dir, "train_*.tfrec")
        self.val_dir = os.path.join(self.input_dir, "val_*.tfrec")
        self.output_dir = output_dir
        self.shuffle_size = shuffle_size

        self.num_iters = num_iters
        self.batch_size = batch_size
        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
        self.epoch_count = tf.Variable(0, trainable=False, name='epoch_count', dtype=tf.int64)
        self.lr = tf.Variable(lr, trainable=False, name='lr', dtype=tf.float32)

        self.model = model
        self.loss_fcn = loss_fcn
        self.loss_name = loss_name
        if optimizer:
            self.optimizer = optimizer(learning_rate=self.lr)
        else:
            self.optimizer = snt.optimizers.Adam(learning_rate=self.lr)
        
        self.init_metrics(metric_mode, early_stop, max_attempts)

        self.distributed = distributed
        self.extra_configs = extra_configs

    # Preprocessing
    # --------------------
    @staticmethod
    def get_arg_parser():
        """
        This static method returns an ArgumentParser with important arguments for
        the TrainerBase's configuration. The parser can be used as a parent parser.
        """
        parser = argparse.ArgumentParser(add_help=False)
        add_arg = parser.add_argument
        add_arg("--input-dir", help="name of dir for input", default="tfrec")
        add_arg("--output-dir", help="name of dir for output", default="trained")
        add_arg("--batch-size", type=int, help="training/evaluation batch size", default=50)
        add_arg("--num-epochs", type=int, help="number of epochs", default=5)
        add_arg("--num-iters", type=int, help="level of message passing", default=10)
        add_arg("--shuffle-size", type=int, help="number for shuffling", default=-1)
        return parser

    def init_metrics(self, mode=None, early_stop=None, max_attempts=1):
        """
        Function to initialize TrainerBase metrics, which is called in
        __init__().
        
        Parameters
        ----------
        mode: Either 'clf' (classification) or 'rgr' (regression).
        Determines which metrics will be used:
            - 'clf' uses 'auc', 'acc', 'pre', 'rec', 'loss'
            - 'rgr' uses 'loss', 'pull'
        
        early_stop: Sets a metric to use as a stopping condition
        
        max_attempts: The number of attempts allowed for the early_stop
        metric to fail before training stops
        
        Raises
        ------
        ValueError: if self.metric_mode is neither 'clf' or 'rgr'
        """
        self.metric_mode = mode
        if mode is None:
            self.metric_dict = {}
            self.early_stop = None
        elif mode == "clf":
            self.metric_dict = {
                "auc": 0.0, "acc": 0.0, "pre": 0.0, "rec": 0.0, "loss": 0.0
            }
            self.early_stop = early_stop if early_stop else "auc"
        elif mode == "rgr":
            self.metric_dict = {
                "loss": 0.0, "pull": 0.0
            }
            self.early_stop = early_stop if early_stop else "loss"
        else:
            raise ValueError("Unsupported metric mode: must either be 'clf', 'rgr', or None")
        self.max_attempts = max_attempts
        self.attempts = 0
        if self.early_stop in ["loss", "pull"]:
            self.should_max_metric = True
            self.best_metric = float('inf')
        else:
            self.should_max_metric = False
            self.best_metric = 0.0
        time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        self.metric_writer = tf.summary.create_file_writer(os.path.join(self.output_dir, "logs/{}/metrics".format(time_stamp)))
    
    def _early_stop_condition(self):
        """
        Helper function to check whether, given self.early_stop
        and self.should_max_metric, training should stop at the
        current moment.
        """
        current_metric = self.metric_dict[self.early_stop]
        if (self.should_max_metric and current_metric > self.best_metric) or\
                (not self.should_max_metric and current_metric < self.best_metric):
            print("Current metric {} {:.4f} is {} than best {:.4f}.".format(
                self.early_stop, current_metric, "higher" if self.should_max_metric else "lower", self.best_metric))
            if self.attempts >= self.max_attempts:
                print("Reached maximum failed attempts: {} attempts. Stopping training".format(self.max_attempts))
                return True
            self.attempts += 1
        else:
            self.best_metric = current_metric
        return False

    def _get_shuffle_size(self, shuffle_size):
        """
        Helper function that returns the buffer size to shuffle with.
        """
        return self.ngraphs_train if shuffle_size < 0 else shuffle_size

    def load_training_data(self, filenames=None, shuffle=False, shuffle_size=-1, repeat=1):
        """
        Loads, shuffles, and sets up training data from the train directory.
        
        Parameters
        ----------
        filenames: A list of pattern-matching files, produced by tf.io.gfile.glob.
        Defaults to None, in which case self.train_dir is used for glob
        
        shuffle: A boolean indicating whether to shuffle the dataset or not
        
        shuffle_size: If shuffle is True, specifies the shuffle buffer size. If
        negative, uses the entire dataset as is
        
        repeat: Specifies how many times the dataset is repeated
        """
        if not filenames:
            filenames = tf.io.gfile.glob(self.train_dir)
        if shuffle_size is None:
            shuffle_size = self.shuffle_size
        self.data_train, self.ngraphs_train = read_dataset(filenames)
        if shuffle:
            self.data_train = self.data_train.shuffle(self._get_shuffle_size(shuffle_size), seed=12345, reshuffle_each_iteration=False)
        if repeat > 1:
            self.data_train = self.data_train.repeat(repeat)
        self.data_train = self.data_train.prefetch(AUTO)
        return self.data_train, self.ngraphs_train

    def load_validating_data(self, filenames=None, shuffle=False, shuffle_size=-1, repeat=1):
        """
        Loads, shuffles, and sets up validation data from the val directory.
        
        Parameters
        ----------
        filenames: A list of pattern-matching files, produced by tf.io.gfile.glob.
        Defaults to None, in which case self.val_dir is used for glob
        
        shuffle: A boolean indicating whether to shuffle the dataset or not
        
        shuffle_size: If shuffle is True, specifies the shuffle buffer size. If
        negative, uses the entire dataset as is
        
        repeat: Specifies how many times the dataset is repeated
        """
        if not filenames:
            filenames = tf.io.gfile.glob(self.val_dir)
        if shuffle_size is None:
            shuffle_size = self.shuffle_size
        self.data_val, self.ngraphs_val = read_dataset(filenames)
        if shuffle:
            self.data_val = self.data_val.shuffle(self._get_shuffle_size(shuffle_size), seed=12345, reshuffle_each_iteration=False)
        if repeat > 1:
            self.data_val = self.data_val.repeat(repeat)
        self.data_val = self.data_val.prefetch(AUTO)
        return self.data_val, self.ngraphs_val

    def load_testing_data(self, filenames, shuffle=False, shuffle_size=-1, repeat=1):
        """
        Loads, shuffles, and sets up testing data from the test directory.
        
        Parameters
        ----------
        filenames: A list of pattern-matching files, produced by tf.io.gfile.glob
        
        shuffle: A boolean indicating whether to shuffle the dataset or not
        
        shuffle_size: If shuffle is True, specifies the shuffle buffer size. If
        negative, uses the entire dataset as is
        
        repeat: Specifies how many times the dataset is repeated
        """
        self.data_test, self.ngraphs_test = read_dataset(filenames)
        if shuffle_size is None:
            shuffle_size = self.shuffle_size
        if shuffle:
            self.data_test = self.data_test.shuffle(self._get_shuffle_size(shuffle_size), seed=12345, reshuffle_each_iteration=False)
        if repeat > 1:
            self.data_test = self.data_test.repeat(repeat)
        self.data_test = self.data_test.prefetch(AUTO)
        return self.data_test, self.ngraphs_test

    def setup_training_loop(self, model=None, loss_fcn=None):
        """
        Sets up training step for the optimizer using tf.function
        and tf.GradientTape. Optionally, the user can pass in a
        model and loss_fcn to replace the current model and loss_fcn
        attributes of TrainerBase.
        
        Parameters
        ----------
        model: The model class to use for training
        
        loss_fcn: The loss function class to use for training
        """
        if model:
            self.model = model
        if loss_fcn:
            self.loss_fcn = loss_fcn
        input_signature = self._input_signature()

        def update_step(inputs, targets):
            print("Tracing update_step")
            with tf.GradientTape() as tape:
                output_ops = self.model(inputs, self.num_iters)
                loss_ops_tr = self.loss_fcn(targets, output_ops)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(self.num_iters, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, self.model.trainable_variables)
            self.optimizer.apply(gradients, self.model.trainable_variables)
            return loss_op_tr

        self.training_step = tf.function(update_step, input_signature=input_signature)

    def _input_signature(self):
        """
        Helper function to generate an input signature, which is
        passed into tf.function() to generate the training_step.
        """
        return get_signature(self.data_train, self.batch_size)

    def optimizer(self, lr):
        """
        Sets the optimizer.
        """
        self.optimizer = snt.optimizers.Adam(lr)

    def set_model(self, model):
        """
        Sets the model.
        """
        self.model = model

    def set_loss(loss_fcn):
        """
        Sets the loss function.
        """
        self.loss_fcn = loss_fcn

    # Training
    # --------------------        

    def train_one_epoch(self, train_data=None):
        """
        Performs one epoch of training, returning the training loss
        and number of batches.
        
        Parameters
        ----------
        train_data: The data to use for training. Must have been
        processed by load_training_data() beforehand. If None is
        specified, defaults to self.data_train
        """
        if train_data is None:
            train_data = self.data_train
        num_batches = 0
        total_loss = 0.
        for inputs in loop_dataset(train_data, self.batch_size):
            inputs_tr, targets_tr = inputs
            total_loss += self.training_step(inputs_tr, targets_tr).numpy()
            num_batches += 1
        return total_loss, num_batches

    def validate_one_epoch(self, val_data=None):
        """
        Performs one epoch of validation, returning the validation loss,
        number of batches, predictions, as well as truth info.
        
        Parameters
        ----------
        val_data: The data to use for validation. Must have been
        processed by load_validating_data() beforehand. If None is
        specified, defaults to self.data_val
        """
        if val_data is None:
            val_data = self.data_val
        total_loss = 0.
        num_batches = 0
        predictions, truth_info = [], []

        for data in loop_dataset(val_data, self.batch_size):
            inputs, targets = data
            outputs = self.model(inputs, self.num_iters)
            total_loss += (tf.math.reduce_sum(
                self.loss_fcn(targets, outputs))/tf.constant(
                    self.num_iters, dtype=tf.float32)).numpy()
            if len(outputs) > 1:
                outputs = outputs[-1]
            if self.loss_name == "GlobalLoss":
                predictions.append(outputs.globals)
                truth_info.append(targets.globals)
            else:
                predictions.append(outputs.edges)
                truth_info.append(targets.edges)
            num_batches += 1
        predictions = np.concatenate(predictions, axis=0)
        truth_info = np.concatenate(truth_info, axis=0)
        return total_loss, num_batches, predictions, truth_info
    
    def update_metrics(self, predictions, truth_info, loss, threshold=0.5):
        """
        Updates metrics after receiving both predictions and truth_info from validation.
        
        Parameters
        ----------
        predictions: Predictions generated from validation
        
        truth_info: Truth info used in validation
        
        loss: Validation loss
        
        Threshold: For the 'clf' metric mode (classification), indicates the threshold
        for classifying on-class or off-class. Defaults to 0.5
        
        Raises
        ------
        ValueError: if self.metric_mode is neither 'clf' or 'rgr'
        """
        if self.metric_mode == "clf":
            y_true, y_pred = (truth_info > threshold), (predictions > threshold)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, predictions)
            self.metric_dict['auc'] = sklearn.metrics.auc(fpr, tpr)
            self.metric_dict['acc'] = sklearn.metrics.accuracy_score(y_true, y_pred)
            self.metric_dict['pre'] = sklearn.metrics.precision_score(y_true, y_pred)
            self.metric_dict['rec'] = sklearn.metrics.recall_score(y_true, y_pred)
        elif self.metric_mode == "rgr":
            self.metric_dict['pull'] = np.mean((predictions - truth_info) / truth_info)
        else:
            raise ValueError("currently " + self.metric_mode + " is not supported")
        self.metric_dict['loss'] = loss
        print(self.metric_dict)
        with self.metric_writer.as_default():
            for key,val in self.metric_dict.items():
                tf.summary.scalar(key, val, step=self.epoch_count)
            self.metric_writer.flush()
        self.epoch_count.assign_add(1)

    def train(self, model=None, loss_fcn=None, train_data=None, val_data=None, num_epochs=None):
        """
        Trains the dataset, potentially using the specified model and loss.
        
        Parameters
        ----------
        model: Class of model used for training. If not specified, uses current
        model attribute of TrainerBase
        
        loss_fcn: Class of loss function used for training. If not specified,
        uses current loss_fcn attribute of TrainerBase
        
        train_data: Data to train on. Defaults to self.data_train, which should
        be loaded beforehand by load_training_data()
        
        val_data: Data to use for validation. Defaults to self.data_val, which
        should be loaded beforehand by load_validating_data()
        
        num_epochs: The maximum number of epochs for training. Defaults to 
        self.num_epochs
        """
        if num_epochs is None:
            num_epochs = self.num_epochs
        self.setup_training_loop(model, loss_fcn)
        for epoch in range(num_epochs):
            loss_tr, num_batches_tr = self.train_one_epoch(train_data)
            loss_val, num_batches_val, predictions, truth_info = self.validate_one_epoch(val_data)
            if self.metric_mode:
                self.update_metrics(predictions, truth_info, loss_val / num_batches_val)
            if self._early_stop_condition():
                print("breaking after {} epoch(s)".format(epoch + 1))
                break

    # Prediction
    # --------------------  

    def predict(self, test_data):
        """
        Uses the current model/loss to generate predictions on test_data
        """
        raise NotImplementedError

    def score(self, y_pred, y_true):
        """
        Returns a dictionary of relevant metrics.
        
        Parameters
        ----------
        y_pred: Values predicted by the model
        
        y_true: The ground truth values to score against
        """
        raise NotImplementedError

    # future: add hyperparameter optimization
    # search for tf hyperparam optimization packages, integrate with tensorboard
    # can try multiple hyperparams, construct surrogate fn predicting hyperparam relationships