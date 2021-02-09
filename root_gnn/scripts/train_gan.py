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

node_mean = np.array([
    [14.13, 0.05, -0.10, -0.04], 
    [7.73, 0.02, -0.04, -0.08],
    [6.41, 0.04, -0.06, 0.04]
], dtype=np.float32)

node_scales = np.array([
    [13.29, 10.54, 10.57, 12.20], 
    [8.62, 6.29, 6.35, 7.29],
    [6.87, 5.12, 5.13, 5.90]
], dtype=np.float32)

node_min = np.array([
    [0.75, -46.3, -46.0, -47.0],
    [0.13, -40.5, -37.0, -39.5],
    [0.14, -36.4, -35.5, -35.0]
], dtype=np.float32)

node_max = np.array([
    [49.1, 47.7, 44.2, 46.6],
    [46.2, 35.7, 41.0, 37.8],
    [42.8, 36.3, 37.0, 35.5]
], dtype=np.float32)

node_abs_max = np.array([
    [49.1, 47.7, 46.0, 47.0],
    [46.2, 40.5, 41.0, 39.5],
    [42.8, 36.4, 37.0, 35.5]
], dtype=np.float32)

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
    batch_size = args.batch_size

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
        n_epochs, batch_size))
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
    if args.shuffle_size > 0:
        training_dataset = training_dataset.shuffle(
                args.shuffle_size, seed=12345, reshuffle_each_iteration=False)


    logging.info("rank {} has {} training graphs".format(
        dist.rank, ngraphs_train))

    input_signature = get_signature(
        training_dataset, batch_size, dynamic_num_nodes=False)


    gan = sets_gan.SetGAN(
        noise_dim=args.noise_dim,
        max_nodes=MAX_NODES,
        num_iters=args.disc_num_iters,
        batch_size=args.batch_size)

    optimizer = sets_gan.SetGANOptimizer(
                        gan,
                        batch_size=batch_size,
                        noise_dim=args.noise_dim,
                        num_epcohs=n_epochs,
                        disc_lr=args.disc_lr,
                        gen_lr=args.gen_lr,
                        )
    
    disc_step = optimizer.disc_step
    step = optimizer.step
    if not args.debug:
        step = tf.function(step)
        disc_step = tf.function(disc_step)

    

    training_data = loop_dataset(training_dataset, batch_size)
    steps_per_epoch = ngraphs_train // batch_size


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

    
    if args.warm_up:
        # train discriminator for certain batches
        # to "warm up" the discriminator
        print("start to warm up discriminator with {} batches".format(args.disc_batches))
        for _ in range(args.disc_batches):
            inputs_tr, targets_tr = next(training_data)
            inputs_tr = inputs_tr.replace(nodes=inputs_tr.nodes/node_abs_max[0])
            target_nodes = np.reshape(targets_tr.nodes, [batch_size, -1, 4])
            target_nodes = np.reshape(target_nodes/node_abs_max, [-1, 4])
            targets_tr = targets_tr.replace(nodes=target_nodes)
            disc_step(inputs_tr, targets_tr)
        print("finished the warm up")



    pre_gen_loss = pre_disc_loss = 0
    start_time = time.time()

    with tqdm.trange(n_epochs*steps_per_epoch) as t:

        for step_num in t:
            epoch = tf.constant(int(step_num / steps_per_epoch), dtype=tf.int32)
            inputs_tr, targets_tr = next(training_data)

            # --------------------------------------------------------
            # scale the inputs and outputs to norm distribution
            # inputs_tr = inputs_tr.replace(nodes=(inputs_tr.nodes - node_mean[0]) / node_scales[0])
            # target_nodes = np.reshape(targets_tr.nodes, [batch_size, -1, 4])
            # target_nodes = np.reshape((target_nodes - node_mean) / node_scales, [-1, 4])
            # targets_tr = targets_tr.replace(nodes=target_nodes)
            # --------------------------------------------------------
            # scale the inputs and outputs to [-1, 1]
            inputs_tr = inputs_tr.replace(nodes=inputs_tr.nodes/node_abs_max[0])
            target_nodes = np.reshape(targets_tr.nodes, [batch_size, -1, 4])
            target_nodes = np.reshape(target_nodes/node_abs_max, [-1, 4])
            targets_tr = targets_tr.replace(nodes=target_nodes)
            # --------------------------------------------------------

            disc_loss, gen_loss, lr_mult = step(inputs_tr, targets_tr, epoch)
            disc_loss = disc_loss.numpy()
            gen_loss = gen_loss.numpy()
            if step_num and (step_num % 50 == 0):
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
                    # epoch = epoch.numpy()
                    tf.summary.scalar("generator loss", gen_loss, step=step_num)
                    tf.summary.scalar("discriminator loss", disc_loss, step=step_num)
                    tf.summary.scalar(
                        "time", (this_epoch-start_time)/60., step=step_num)
                    

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
    add_arg("--disc-lr", help='learning rate for discriminator', default=2e-4, type=float)
    add_arg("--gen-lr", help='learning rate for generator', default=5e-5, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)
    add_arg("--batch-size", type=int,
            help='training/evaluation batch size', default=500)
    add_arg("--shuffle-size", type=int,
            help="number of events for shuffling", default=650)
    add_arg("--warm-up", action='store_true', help='warm up discriminator first')
    add_arg("--disc-batches", help='number of batches training discriminator only', type=int, default=100)

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
