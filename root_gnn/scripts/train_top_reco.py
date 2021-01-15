#!/usr/bin/env python
import tensorflow as tf

import os
import sys
import argparse

import re
import time
import random
import functools
import six

import numpy as np
import sklearn.metrics


from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from types import SimpleNamespace
import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
import horovod.tensorflow as hvd


from root_gnn import model as all_models
from root_gnn.src.datasets import topreco
from root_gnn.src.datasets import graph
from root_gnn.utils import load_yaml

from root_gnn.trainer import read_dataset
from root_gnn.trainer import loop_dataset
from root_gnn.trainer import get_signature


target_scales = np.array([145.34593924, 145.57711889, 432.92148524, 281.44161905, 1, 1]*topreco.n_max_tops).T.reshape((-1,))
target_mean = np.array([6.74674671e-02, -6.17142186e-02,  4.18239305e-01, 4.24881531e+02, 0, 0]*topreco.n_max_tops).T.reshape((-1,))


def init_workers(distributed=False):
    if distributed:
        hvd.init()
        assert hvd.mpi_threads_supported()
        from mpi4py import MPI
        assert hvd.size() == MPI.COMM_WORLD.Get_size()
        comm = MPI.COMM_WORLD
        print("Rank: {}, Size: {}".format(hvd.rank(), hvd.size()))
        return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                                local_rank=hvd.local_rank(),
                                local_size=hvd.local_size(), comm=comm)
    else:
        print("not doing distributed")
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)
        

def train_and_evaluate(args):
    dist = init_workers(args.distributed)

    device = 'CPU'
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logging.info("found {} GPUs".format(len(gpus)))

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if len(gpus) > 0:
        device = "{}GPUs".format(len(gpus))
    if gpus and args.distributed:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    output_dir = args.output_dir
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    num_processing_steps_tr = args.num_iters     ## level of message-passing
    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(n_epochs, args.batch_size))
    logging.info("{} processing steps in the model".format(num_processing_steps_tr))
    logging.info("I am in hvd rank: {} of  total {} ranks".format(dist.rank, dist.size))

    if dist.rank == 0:
        train_input_dir = os.path.join(args.input_dir, 'train') 
        val_input_dir = os.path.join(args.input_dir, 'val')
        train_files = tf.io.gfile.glob(os.path.join(train_input_dir, args.patterns))
        eval_files = tf.io.gfile.glob(os.path.join(val_input_dir, args.patterns))
        ## split the number of files evenly to all ranks
        train_files = [x.tolist() for x in np.array_split(train_files, dist.size)]
        eval_files = [x.tolist() for x in np.array_split(eval_files, dist.size)]
    else:
        train_files = None
        eval_files = None

    if args.distributed:
        train_files = dist.comm.scatter(train_files, root=0)
        eval_files = dist.comm.scatter(eval_files, root=0)
    else:
        train_files = train_files[0]
        eval_files = eval_files[0]

    logging.info("rank {} has {} training files and {} evaluation files".format(
        dist.rank, len(train_files), len(eval_files)))

    AUTO = tf.data.experimental.AUTOTUNE
    training_dataset, ngraphs_train = read_dataset(train_files)
    training_dataset = training_dataset.prefetch(AUTO)

    input_signature = get_signature(training_dataset, args.batch_size)


    learning_rate = args.learning_rate
    optimizer = snt.optimizers.Adam(learning_rate)
    model = getattr(all_models, 'FourTopPredictor')()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir,
                                max_to_keep=5, keep_checkpoint_every_n_hours=8)
    logging.info("Loading latest checkpoint from: {}".format(output_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)

    target_scales = np.array([145.34593924, 145.57711889, 432.92148524, 281.44161905, 1, 1]*topreco.n_max_tops).reshape((topreco.n_max_tops, -1)).T.reshape((-1,))
    target_mean = np.array([6.74674671e-02, -6.17142186e-02,  4.18239305e-01, 4.24881531e+02, 0, 0]*topreco.n_max_tops).reshape((topreco.n_max_tops, -1)).T.reshape((-1,))
    # training loss
    def loss_fcn(target_op, output_ops):
        # print("target size: ", target_op.nodes.shape)
        # print("output size: ", output_ops[0].nodes.shape)
        # output_op = output_ops[-1]
        # print("loss of 4-vect: ", tf.nn.l2_loss((target_op.nodes[:, :4] - output_op.nodes[:topreco.n_max_tops, :4])))
        # print("loss of charge: ", tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.cast(target_op.nodes[:, 4:6], tf.int32),  output_op.nodes[:topreco.n_max_tops, 4:6])))
        # print("loss of predictions: ", tf.compat.v1.losses.log_loss(tf.cast(target_op.nodes[:, 6], tf.int32),  tf.math.sigmoid(output_op.nodes[:topreco.n_max_tops, 6])))

        # loss_ops = [tf.nn.l2_loss((target_op.nodes[:, :4] - output_op.nodes[:topreco.n_max_tops, :4]))
        #     + tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.cast(target_op.nodes[:, 4:6], tf.int32),  output_op.nodes[:topreco.n_max_tops, 4:6]))
        #     + tf.compat.v1.losses.log_loss(tf.cast(target_op.nodes[:, 6], tf.int32),  tf.math.sigmoid(output_op.nodes[:topreco.n_max_tops, 6]))
        #     for output_op in output_ops
        # ]

        # loss_ops = [tf.nn.l2_loss((target_op.globals[:, :topreco.n_max_tops*4] - output_op.globals[:, :topreco.n_max_tops*4])) / target_op.globals.shape[0]
        #     + tf.compat.v1.losses.log_loss(
        #         tf.cast(target_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*5], tf.int32),\
        #         tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*5]))
        #     + tf.compat.v1.losses.log_loss(
        #         tf.cast(target_op.globals[:, topreco.n_max_tops*5:], tf.int32),\
        #         tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*5:]))
        #     for output_op in output_ops
        # ]
        # alpha = tf.constant(1, dtype=tf.float32)
        # loss_ops = [alpha * tf.compat.v1.losses.mean_squared_error(target_op.globals[:, :topreco.n_max_tops*4], output_op.globals[:, :topreco.n_max_tops*4])
        #     + tf.compat.v1.losses.log_loss(
        #         tf.cast(target_op.globals[:, topreco.n_max_tops*4:], tf.int32),\
        #         tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*4:]))
        #     for output_op in output_ops
        # ]

        # loss_ops = [ tf.nn.l2_loss((target_op.globals[:, :topreco.n_max_tops*4] - output_op.globals[:, :topreco.n_max_tops*4]))
        #     for output_op in output_ops
        # ]
        loss_ops = [ tf.compat.v1.losses.absolute_difference(
                            target_op.globals[:, :topreco.n_max_tops*4],\
                            output_op.globals[:, :topreco.n_max_tops*4])
            for output_op in output_ops
        ]

        # loss_ops = [tf.compat.v1.losses.mean_squared_error(target_op.globals[:, :topreco.n_max_tops*4], output_op.globals[:, :topreco.n_max_tops*4])
        #     for output_op in output_ops
        # ]

        return tf.stack(loss_ops)

    @functools.partial(tf.function, input_signature=input_signature)
    def train_step(inputs_tr, targets_tr, first_batch):
        print("Tracing update_step")
        print("inputs nodes", inputs_tr.nodes.shape)
        print("inputs edges", inputs_tr.edges.shape)
        print("input n_node", inputs_tr.n_node.shape)
        print(inputs_tr.nodes)
        with tf.GradientTape() as tape:
            outputs_tr = model(inputs_tr, num_processing_steps_tr, is_training=True)
            loss_ops_tr = loss_fcn(targets_tr, outputs_tr)
            loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(num_processing_steps_tr, dtype=tf.float32)

        # Horovod: add Horovod Distributed GradientTape.
        if args.distributed:
            tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss_op_tr, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if args.distributed and first_batch:
            hvd.broadcast_variables(model.trainable_variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables, root_rank=0)

        return loss_op_tr


    def train_epoch(dataset):
        total_loss = 0.
        batch = 0
        for inputs in loop_dataset(dataset, args.batch_size):
            input_tr, targets_tr = inputs
            new_target = (targets_tr.globals - target_mean) / target_scales
            targets_tr = targets_tr.replace(globals=new_target)
            total_loss += train_step(input_tr, targets_tr, batch==0).numpy()
            batch += 1
        logging.info("total batches: {}".format(batch))
        return total_loss/batch, batch
        # return total_loss/batch/args.batch_size, batch

    out_str  = "Start training " + time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "Epoch, Time [mins], Loss\n"
    log_name = os.path.join(output_dir, "training_log.txt")
    if dist.rank == 0:
        with open(log_name, 'a') as f:
            f.write(out_str)
    now = time.time()

    for epoch in range(n_epochs):
        logging.info("start epoch {} on {}".format(epoch, device))

        # shuffle the dataset before training
        training_dataset = training_dataset.shuffle(args.shuffle_size, seed=12345, reshuffle_each_iteration=True)
        loss,batches = train_epoch(training_dataset)
        this_epoch = time.time()

        logging.info("{} epoch takes {:.2f} mins with loss {:.4f} in {} batches".format(
            epoch, (this_epoch-now)/60., loss, batches))
        out_str = "{}, {:.2f}, {:.4f}\n".format(epoch, (this_epoch-now)/60., loss)

        now = this_epoch
        if dist.rank == 0:
            with open(log_name, 'a') as f:
                f.write(out_str)
            ckpt_manager.save()

    if dist.rank == 0:
        out_log = "End @ " + time.strftime('%d %b %Y %H:%M:%S', time.localtime()) + "\n"
        with open(log_name, 'a') as f:
            f.write(out_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("--input-dir", help='input directory that contains subfolder of train, val and test')
    add_arg("--patterns", help='file patterns', default='*')
    add_arg("--output-dir", help="where the model and training info saved")
    add_arg('-d', '--distributed', action='store_true', help='data distributed training')
    add_arg("--num-iters", help="number of message passing steps", default=8, type=int)
    add_arg("--learning-rate", help='learing rate', default=0.0005, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)
    add_arg("--batch-size", type=int, help='training/evaluation batch size', default=500)
    add_arg("--shuffle-size", type=int, help="number of events for shuffling", default=650)
    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],\
        default="INFO")
    args, _ = parser.parse_known_args()

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train_and_evaluate(args)