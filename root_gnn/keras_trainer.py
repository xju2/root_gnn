from operator import is_
from ipykernel import kernel_protocol_version
from numpy.lib.arraysetops import isin
import tensorflow as tf
from tensorflow.compat.v1 import logging
from tensorflow.keras import Model
from keras.models import load_model
logging.set_verbosity("INFO")
logging.info("TF Version:{}".format(tf.__version__))
# try:
#     import horovod.tensorflow as hvd
#     no_horovod = False
# except ModuleNotFoundError:
#     logging.warning("No horvod module, cannot perform distributed training")
#     no_horovod = True


import os
import pprint
import time
import functools
import argparse
import tqdm

import numpy as np

import sklearn.metrics

from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from root_gnn.utils import load_yaml
from root_gnn.src.datasets import graph
from root_gnn import losses
from root_gnn import model as Models

verbosities = ['DEBUG','ERROR', "FATAL", "INFO", "WARN"]
printer = pprint.PrettyPrinter(indent=2)

AUTO = tf.data.experimental.AUTOTUNE

def read_dataset(filenames, nEvtsPerFile=5000):
    """
    Read dataset...
    """
    tr_filenames = tf.io.gfile.glob(filenames)
    n_files = len(tr_filenames)

    dataset = tf.data.TFRecordDataset(tr_filenames)
    dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
    # n_graphs = sum([1 for _ in dataset]) # this is computational expensive.
    n_graphs = n_files * nEvtsPerFile
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
    data, with_bool=False,
    with_batch_dim=False,
    dynamic_num_nodes=True,
    dynamic_num_edges=True,
    ):
    """
    Get signature of inputs for the training loop.
    The signature is used by the tf.function
    """
    inputs, targets = next(data)
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


def add_args(parser):
    """
    The method adds options for a parser
    """
    add_arg = parser.add_argument
    add_arg("--input-dir", help="name of dir for input", default="inputs")
    add_arg("--evts-per-file", help="Events per TFRecords", default=5000)
    add_arg("--output-dir", help='output directory', default='trained')
    add_arg("--model", help='predefined ModelName', choices=list(Models.__all__), required=True)
    add_arg("--loss-name", help='predefined Loss Function', choices=list(losses.__all__), required=True)
    add_arg("--loss-pars", help='weights for loss function', default=None)
    add_arg("--learning-rate", type=float, help="learning rate", default=0.0005)
    add_arg("--batch-size", type=int, help='batch size', default=100)
    add_arg("--num-epochs", type=int, help="number of epochs", default=5)
    add_arg("--num-iters", type=int, help="number of message passing", default=4)
    add_arg('--stop-on', help='metric for early stop.'
        '\"val_loss, auc, acc, pre, rec\" for classification, \"val_loss, pull\" for regression.', default='val_loss')
    add_arg('--patiences', help='number of allowed no improvement', default=3, type=int)
    add_arg("--shuffle-size", type=int, help="number for shuffling", default=-1)
    add_arg("--log-freq", type=int, help='log frequence in terms of batches', default=100)
    add_arg("--val-batches", type=int, help='number of batches used for each validation', default=50)
    add_arg("--file-pattern", default='*', help='file patterns for input TFRecords')
    add_arg("--disable-tqdm", action='store_true', help='disable tqdm progressing bar')
    add_arg("--encoder-size", help='MLP size for encoder', default=None)
    add_arg("--core-size", help='MLP size for core', default=None)
    add_arg("--decoder-size", help='MLP size for decoder', default=None)
    add_arg("--with-edge-inputs", action='store_true', help='input graph contains edge information')
    add_arg("--with-edges", help='duplicate variable', default=False)
    add_arg("--with-global-inputs", help='input graph contains global information', default=False)
    add_arg("--with-globals", help='duplicate with global flag', default=False)
    add_arg("--output-size", help='output size of global regression', default=1)
    add_arg("--num-transformation", help='the number of times transformation is performed for representational learning', default=1)
    add_arg("--agument-type", help='type of augmentation for representation learning', default='rotation')
    add_arg("--cosine-decay",help='learning rate schedule function.',default=False)
    add_arg("--decay-steps", help='Steps for cosine decay in learning rate', default=0)

    
class Trainer(snt.Module):
    
    """
    The class to implement a simple trainer and model.
    
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

    def __init__(self, input_dir, evts_per_file, output_dir, 
                model, loss_fcn, optimizer,
                mode, # mode, 'clf,globals', 'clf,edges', 'rgr,globals'
                batch_size=100, num_epochs=1, num_iters=4,
                stop_on='val_loss', patiences=2,
                shuffle_size=-1, log_freq=100,
                val_batches=50,
                file_pattern='*', #distributed=False,
                disable_tqdm=False,
                encoder_size=None, core_size=None, decoder_size=None,
                with_edge_inputs=False, with_edges=False, 
                with_global_inputs=False, with_globals=False,
                activation=tf.nn.relu, learning_rate=0.005, 
                output_size=1, num_transformation=1, augment_type="rotation", 
                cosine_decay=False, decay_steps=0,
                verbose="INFO", name='Trainer', **kwargs):
        """
        Trainer constructor, which initializes configurations, hyperparameters,
        and metrics.
        
        Parameters
        ----------
        Sets input, output, model and loss, as well as relevant hyperparameters.
        The user can directly unpack the arguments from the ArgumentParser
        created in get_arg_parser().
        """
        super().__init__(name=name)
        self.input_dir = input_dir

        # read training and testing data
        self.training_step = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.file_pattern = file_pattern
        self.evts_per_file = evts_per_file
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.read_all_data()

        self.ckpt_manager = None
        self.output_dir = output_dir
        self.output_size = output_size
        self.num_transformation = num_transformation
        self.augment_type = augment_type
        
        self.cosine_decay = cosine_decay
        self.decay_steps = decay_steps
        
        if isinstance(activation, str):
            activation = getattr(tf.nn, activation)
            
        if isinstance(model, str):
            if "regression" in model or "Regression" in model:
                self.model = getattr(Models, model)(
                    self.output_size,
                    with_edge_inputs=with_edges,
                    with_global_inputs=with_global_inputs,
                    encoder_size=encoder_size,
                    core_size=core_size,
                    decoder_size=decoder_size,
                    activation=activation
                    )
            else:
                self.model = getattr(Models, model)(
                    with_edge_inputs=with_edges,
                    with_global_inputs=with_global_inputs,
                    encoder_size=encoder_size,
                    core_size=core_size,
                    decoder_size=decoder_size,
                    activation=activation
                    )
        elif isinstance(model, snt.Module):
            self.model = model
        else:
            raise RuntimeError("model:", model, "is not supported")

        if isinstance(loss_fcn, str):
            loss_config = loss_fcn.split(',')
            loss_name = loss_config[0]
            if len(loss_config) > 1:
                self.loss_fcn = getattr(losses, loss_name)(*[float(x) for x in loss_config[1:]])
            else:
                self.loss_fcn = getattr(losses, loss_name)
        else:
            self.loss_fcn = loss_fcn
            
        if isinstance(optimizer, snt.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, float):
            self.optimizer = snt.optimizers.Adam(learning_rate=optimizer)
        elif isinstance(optimizer, str):
            self.optimizer = getattr(snt.optimizers, optimizer)(learning_rate=learning_rate)
        else:
            self.optimizer = snt.optimizers.Adam(learning_rate=0.0005)


        self.mode = mode.split(',')
        self.num_iters = num_iters

        # training hyperparameters
        self.log_freq = log_freq
        self.val_batches = val_batches
        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
        self.step_count = tf.Variable(0, trainable=False, name='step_count', dtype=tf.int64)
        # self.lr = tf.Variable(0.0005, trainable=False, name='lr', dtype=tf.float32)

        # self.distributed = distributed
        self.extra_configs = kwargs
        self.disable_tqdm = disable_tqdm

        ## setup monitoring issues
        self.stop_on = stop_on
        self.attempts = 0
        self.patiences = patiences
        self.metric_dict = dict()
        time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        self.metric_writer = tf.summary.create_file_writer(
            os.path.join(self.output_dir, "logs/{}/metrics".format(time_stamp)))

        if "loss" in stop_on or "pull" in stop_on:
            self.should_max_metric = False
            self.best_metric = float('inf')
        else:
            self.should_max_metric = True
            self.best_metric = 0.0

    def train(self, num_steps: int = None, stop_on: str = None, epochs: int = None):
        """
        The training step.
        """
        ngraphs = self.ngraphs_train
        train_data = self.data_train
        batch_size = self.batch_size
        steps_per_epoch = ngraphs // batch_size
        max_epochs = self.num_epochs
        if epochs is not None and num_steps is not None:
            raise RuntimeError("Please either use `num_steps` or `epochs`")

        tot_steps = num_steps if num_steps else epochs*steps_per_epoch if epochs else self.num_epochs * steps_per_epoch
        self.stop_on = stop_on if stop_on else self.stop_on
        stop_on = self.stop_on

        print(f"Training starts with {ngraphs} graphs with batch size of {batch_size} for {max_epochs} epochs")
        print(f"runing {tot_steps} steps, {steps_per_epoch} steps per epoch, and stop on variable {stop_on}")

        #if self.output_dir is not None and os.path.exists(self.output_dir):
            #self.gnn_model = load_model(self.output_dir)
        #else:
        self.gnn_model = Model(self.model)

        self.gnn_model.compile(optimizer='adam',loss=self.loss_fcn, metrics=[tf.keras.metrics.AUC()])
        self.es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10)
        self.ckpt = tf.keras.callbacks.ModelCheckpoint(monitor='val_auc', filepath=self.output_dir, save_best_only=True)
        #self.gnn_model.summary()
        counter = 1000
        x, y = [], []
        for _ in tqdm.trange(counter):
            i, j = next(train_data)
            x.append(i)
            y.append(j)

        self.gnn_model.fit(x, y, batch_size=batch_size, epochs=10, validation_split=0.11, callbacks=[self.es, self.ckpt])
        self.gnn_model.save(self.output_dir)

    def _setup_training_loop(self):
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
        if self.training_step:
            return 

        input_signature = get_signature(self.data_train)

        
        



    def load_data(self, dirname):
        """
        Loads, shuffles, and sets up training data from the train directory
        """
        _input_dir = os.path.join(self.input_dir, dirname)
        _files = tf.io.gfile.glob(os.path.join(_input_dir, self.file_pattern))
        if (not os.path.exists(_input_dir)) or len(_files) < 1:
            raise RuntimeError(f"{_input_dir} directory does not contain data.")

        tfdata, ngraphs = read_dataset(_files, self.evts_per_file)
        shuffle_size = self.shuffle_size
        if shuffle_size > ngraphs:
            shuffle_size = ngraphs

        if shuffle_size > 0:
            tfdata = tfdata.shuffle(
                shuffle_size, seed=12345, reshuffle_each_iteration=False)

        tfdatasets = tfdata.repeat().prefetch(AUTO)
        tfdata = loop_dataset(tfdatasets, self.batch_size)
        return tfdata, ngraphs

    def load_training_data(self):
        """
        Loads, shuffles, and sets up training data from the train directory
        """
        if self.data_train is not None:
            return 
        self.data_train, self.ngraphs_train = self.load_data('train')

    def load_validating_data(self):
        """
        Loads, shuffles, and sets up validation data from the val directory.
        """
        if self.data_val is not None:
            return 
        self.data_val, self.ngraphs_val = self.load_data('val')

    def load_testing_data(self):
        """
        Loads, shuffles, and sets up testing data from the test directory.
        """
        if self.data_test is not None:
            return 
        self.data_test, self.ngraphs_test = self.load_data('test')

    def read_all_data(self):
        self.load_training_data()
        self.load_validating_data()