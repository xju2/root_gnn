#!/usr/bin/env python
"""
Training a GAN for modeling hadronic interactions
"""
from root_gnn.trainer import get_signature
from root_gnn.trainer import loop_dataset
from root_gnn.trainer import read_dataset

from root_gnn.src.generative import sets_gan
from scipy.sparse import coo_matrix
import tensorflow as tf
# import tensorflow.experimental.numpy as tnp

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
import tqdm


from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
import sonnet as snt

from types import SimpleNamespace
import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))


MAX_NODES = 2
OUT_DIM = 4


def my_print(g, data=False):
    for field_name in graphs.ALL_FIELDS:
        per_replica_sample = getattr(g, field_name)
        if per_replica_sample is None:
            print(field_name, "EMPTY")
        else:
            print(field_name, ":", per_replica_sample.shape)
            if data and field_name != "edges":
                print(per_replica_sample)


def init_workers(distributed=False):
    return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)


def hacky_sigmoid_l2(nodes):
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    # TODO: make tunable param
    r = 10
    return tf.math.sigmoid(r * (1 - D))


def scaled_hacky_sigmoid_l2(nodes):
    dim = tf.shape(nodes)[1]
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    # TODO: make tunable param
    D /= tf.sqrt(tf.dtypes.cast(dim, tf.float32))
    r = 10
    return tf.math.sigmoid(r * (1 - D))


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
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank()], 'GPU')

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    output_dir = args.output_dir
    # output_dir = os.path.join(args.output_dir, "{}".format(time_stamp))
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(
        n_epochs, args.batch_size))
    logging.info("I am in hvd rank: {} of  total {} ranks".format(
        dist.rank, dist.size))

    if dist.rank == 0:
        train_input_dir = os.path.join(args.input_dir, 'train')
        val_input_dir = os.path.join(args.input_dir, 'val')
        train_files = tf.io.gfile.glob(
            os.path.join(train_input_dir, args.patterns))
        eval_files = tf.io.gfile.glob(
            os.path.join(val_input_dir, args.patterns))
        # split the number of files evenly to all ranks
        train_files = [x.tolist()
                       for x in np.array_split(train_files, dist.size)]
        eval_files = [x.tolist()
                      for x in np.array_split(eval_files, dist.size)]
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
    training_dataset = training_dataset.repeat(n_epochs).prefetch(AUTO)
    training_dataset = training_dataset.shuffle(
                args.shuffle_size, seed=12345, reshuffle_each_iteration=False)


    logging.info("rank {} has {} training graphs".format(
        dist.rank, ngraphs_train))

    input_signature = get_signature(
        training_dataset, args.batch_size, dynamic_num_nodes=False)


    gan = sets_gan.SetGAN(
        noise_dim=args.noise_dim,
        max_nodes=MAX_NODES,
        num_iters=args.disc_num_iters,
        batch_size=args.batch_size)

    optimizer = sets_gan.SetGANOptimizer(
                        gan,
                        batch_size=args.batch_size,
                        noise_dim=args.noise_dim,
                        num_epcohs=n_epochs
                        )
    step = optimizer.step
    if not args.debug:
        step = tf.function(step)

    training_data = loop_dataset(training_dataset, args.batch_size)
    steps_per_epoch = ngraphs_train // args.batch_size


    log_dir = os.path.join(output_dir, "logs/{}/train".format(time_stamp))
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        gan=gan)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir,
                                              max_to_keep=5, keep_checkpoint_every_n_hours=8)
    logging.info("Loading latest checkpoint from: {}".format(ckpt_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)


    pre_gen_loss = pre_disc_loss = 0
    start_time = time.time()
    with tqdm.trange(n_epochs*steps_per_epoch) as t:

        for step_num in t:
            epoch = tf.constant(int(step_num / steps_per_epoch))
            inputs_tr, targets_tr = next(training_data)
            disc_loss, gen_loss, lr_mult = step(inputs_tr, targets_tr, epoch)
            disc_loss = disc_loss.numpy()
            gen_loss = gen_loss.numpy()
            if step_num and (step_num % steps_per_epoch == 0):
                t.set_description('Epoch {}/{}'.format(epoch.numpy(), n_epochs))
                t.set_postfix(
                    G_loss=gen_loss, G_loss_change=gen_loss-pre_gen_loss,
                    D_loss=disc_loss, D_loss_change=disc_loss-pre_disc_loss,
                )                
                ckpt_manager.save()
                pre_gen_loss = gen_loss
                pre_disc_loss = disc_loss

                # log some metrics
                this_epoch = time.time()
                with train_summary_writer.as_default():
                    tf.summary.scalar("generator loss", gen_loss, step=epoch)
                    tf.summary.scalar("discriminator loss", disc_loss, step=epoch)
                    tf.summary.scalar(
                        "time", (this_epoch-start_time)/60., step=epoch)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("input_dir",
            help='input directory that contains subfolder of train, val and test')
    add_arg("output_dir", help="where the model and training info saved")
    add_arg("--patterns", help='file patterns', default='*')
    add_arg('-d', '--distributed', action='store_true',
            help='data distributed training')
    add_arg("--disc-lr", help='learning rate for discriminator', default=0.0005, type=float)
    add_arg("--gen-lr", help='learning rate for generator', default=0.0005, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)
    add_arg("--batch-size", type=int,
            help='training/evaluation batch size', default=500)
    add_arg("--shuffle-size", type=int,
            help="number of events for shuffling", default=650)

    add_arg("--noise-dim", type=int, help='dimension of noises', default=8)
    add_arg("--disc-num-iters", type=int,
            help='number of message passing for discriminator', default=4)
    add_arg("--disc-alpha", type=float,
            help='inversely scale true dataset in the loss calculation', default=0.1)
    add_arg("--disc-beta", type=float,
            help='scales generated dataset in the loss calculation', default=0.8)

    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
            default="INFO")
    add_arg("--debug", help='in debug mode', action='store_true')
    args, _ = parser.parse_known_args()
    # print(args)

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train_and_evaluate(args)
