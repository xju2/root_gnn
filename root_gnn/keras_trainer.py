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
    add_arg("--use-node-attn", help='Use node attention', default=False)
    add_arg("--use-global-attn", help='Use global attention', default=False)
    add_arg("--node-mlp-size", help="MLP size for nodes", default=None)
    add_arg("--edge-mlp-size", help="MLP size for edge", default=None)
    add_arg("--global-mlp-size", help="MLP size for global", default=None)
    add_arg("--num-attn-heads", help="Number of Attention Heads", default=8)
    add_arg("--num-attn-blocks", help="Number of Attention Blocks", default=2)
    add_arg("--num-attn-layers", help="Number of Attention Layers", default=1)
    add_arg("--manual-ngraph", help="Manually set the number of graphs", default=0, type=int)
    add_arg("--hom-model", help="Use homogeneous model for GlobalHetGraphClassifier", default=False)
    add_arg('--global-in-nodes', help='Put global attributes in nodes if with global inputs', type=bool, default=None)
    add_arg('--global-in-edges', help='Put global attributes in edges if with global inputs', type=bool, default=None)
    add_arg("--core-mp-size", help="MLP size for RGN MP", default=[64,64])
    add_arg("--attn-layer-size", help="MLP sizes of Attention Layers", default=[64,64])
    add_arg("--lstm-unit", help="Hidden size for LSTM aggregation layer", type=int, default=32)
    add_arg("--global-core-size", help="MLP sizes for global LSTM merging block", default=[64])
    add_arg("--use-MP", help="Use message passing in GRN", default=True)
    add_arg("--core-model", help="The core model name", default="MLPGN")
    add_arg("--use-edge-encoder", help="Use edge encoder for GRN model", default=False)
    add_arg("--global-shape", help="Global feature shape", default=[1,8])
    add_arg("--node-shape", help="Node feature shape", default=[16,7])


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
                encoder_size=None, core_size=None, decoder_size=[],
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
        #self.read_all_data()

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
                    with_global_inputs=with_globals,
                    encoder_size=encoder_size,
                    core_size=core_size,
                    decoder_size=decoder_size,
                    activation=activation,
                    **kwargs
                    )
        elif isinstance(model, snt.Module):
            self.model = model
        else:
            raise RuntimeError("model:", model, "is not supported")
        self.patience = patiences
        print(">>> Using keras Trainer")
        self.num_iters = num_iters
        self.loss_fcn = 'binary_crossentropy'
        self.num_epochs = num_epochs
        self.optimizer = tf.keras.optimizers.Adam(optimizer)
        print(">>> Loading data")
        self.load_npz()
        
        
        if os.path.exists(self.output_dir):
            print(f">>> Loading model at {self.output_dir}")
            self.gnn_model = tf.keras.models.load_model(self.output_dir)
        else:
            print(">>> Creating new model")
            self.gnn_model = Models.kerasClassifier(self.model, **kwargs)
            self.gnn_model.compile(optimizer=self.optimizer,loss=self.loss_fcn, metrics=[tf.keras.metrics.AUC()])
        self.es = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=self.patience)
        self.ckpt = tf.keras.callbacks.ModelCheckpoint(monitor='val_auc', filepath=self.output_dir, save_best_only=True)

    
    def train(self, num_steps: int = None, stop_on: str = None, epochs: int = None):
        """
        The training step.
        """
        self.gnn_model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, 
                           shuffle=True, epochs=self.num_epochs, validation_data=self.data_val,
                           callbacks=[self.es, self.ckpt])
        self.gnn_model.save(self.output_dir)



    def load_npz(self):
        train_input = os.path.join(self.input_dir, "train.npz")
        val_input = os.path.join(self.input_dir, "val.npz")

        train_data = np.load(train_input)
        val_data = np.load(val_input)

        graph_attr = ['n_node', 'n_edge', 'nodes', 'edges', 'senders', 'receivers', 'globals']
        y_train = train_data['labels']
        n_train = len(y_train) // self.batch_size * self.batch_size # FIXME: currently cannot deal with tail case
        y_train = y_train[:n_train]
        x_train = [train_data[key][:n_train] for key in graph_attr]
        

        self.x_train = x_train
        self.y_train = y_train

        y_val = val_data['labels']
        n_val = len(y_val) // self.batch_size * self.batch_size
        x_val = [val_data[key][:n_val] for key in graph_attr]
        y_val = y_val[:n_val]
        
        self.data_val = (x_val, y_val)
        