#!/usr/bin/env python
"""
Training a GAN for modeling hadronic interactions
"""
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


from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
import sonnet as snt

from types import SimpleNamespace
import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))

from scipy.sparse import coo_matrix

from root_gnn.src.generative import sets_gan

from root_gnn.trainer import read_dataset
from root_gnn.trainer import loop_dataset
from root_gnn.trainer import get_signature

MAX_NODES = 18

def my_print(g, data=False):
    for field_name in graphs.ALL_FIELDS:
        per_replica_sample = getattr(g, field_name)
        if per_replica_sample is None:
            print(field_name, "EMPTY")
        else:
            print(field_name, ":", per_replica_sample.shape)
            if data and  field_name != "edges":
                print(per_replica_sample)

def init_workers(distributed=False):
    return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)

def hacky_sigmoid_l2(nodes):
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    #TODO: make tunable param
    r = 10
    return tf.math.sigmoid(r * (1 - D))

def scaled_hacky_sigmoid_l2(nodes):
    dim = tf.shape(nodes)[1]
    r = tf.reduce_sum(tf.square(nodes), 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(nodes, nodes, transpose_b=True) + tf.transpose(r)
    #TODO: make tunable param
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
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    output_dir = args.output_dir
    # output_dir = os.path.join(args.output_dir, "{}".format(time_stamp))
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(n_epochs, args.batch_size))
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
    logging.info("rank {} has {} training graphs".format(dist.rank, ngraphs_train))

    input_signature = get_signature(training_dataset, args.batch_size, dynamic_num_nodes=False)


    learning_rate = args.learning_rate
    optimizer_gen = snt.optimizers.Adam(learning_rate)
    optimizer_disc = snt.optimizers.Adam(learning_rate)

    model_gen = sets_gan.SetsGenerator(input_dim=4+args.noise_dim, out_dim=4)
    model_disc = sets_gan.SetsDiscriminator()

    # regularization terms for discriminator
    l2_reg = snt.regularizers.L2(0.01)


    def generator_loss(fake_output):
        loss_ops = [tf.compat.v1.losses.log_loss(tf.ones_like(output_op.globals, dtype=tf.float32), output_op.globals)
                    for output_op in fake_output
                ]
        return tf.stack(loss_ops)

    def discriminator_loss(real_output, fake_output):
        loss_ops = [tf.compat.v1.losses.log_loss(
                                    tf.ones_like(output_op.globals, dtype=tf.float32), output_op.globals, weights=1-args.disc_alpha)
                for output_op in real_output]
        loss_ops += [tf.compat.v1.losses.log_loss(
                                    tf.zeros_like(output_op.globals, dtype=tf.float32), output_op.globals, weights=args.disc_beta)
                for output_op in fake_output]
        return tf.stack(loss_ops)


    n_node = tf.constant([MAX_NODES]*args.batch_size, dtype=tf.int32)
    n_edge = tf.constant([0]*args.batch_size, dtype=tf.int32)
    out_dim = 4
    # @functools.partial(tf.function, input_signature=input_signature)
    def train_step(inputs_tr, targets_tr):
        noise = tf.random.normal([args.batch_size, args.noise_dim])
        incident_info = inputs_tr.nodes
        input_op = tf.concat([inputs_tr.nodes, noise], axis=-1)
        inputs_tr = inputs_tr.replace(nodes=input_op)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generator
            node_pred = model_gen(inputs_tr, max_nodes=MAX_NODES, training=True)

            # concatnate the incident particle 4-vector to the predicted 4 vectors
            incident_info = tf.reshape(incident_info, [args.batch_size, 1, out_dim])
            node_pred = tf.concat([inputs_tr, node_pred], axis=1)
            node_pred = tf.reshape(node_pred, [-1, out_dim])

            pred_graph = graphs.GraphsTuple(
                nodes=node_pred, edges=None, globals=tf.constant([0]*args.batch_size, dtype=tf.float32),
                receivers=None, senders=None, n_node=n_node,
                n_edge=n_edge
            )
            pred_graph = utils_tf.fully_connect_graph_static(pred_graph, exclude_self_edges=True)
            pred_graph = pred_graph.replace(edges=tf.zeros([pred_graph.senders.shape[0], 1], dtype=tf.float32))
            my_print(pred_graph, data=False)
            my_print(targets_tr, data=True)

            # discriminator
            # add noise to target nodes
            noise_target = tf.random.normal(targets_tr.nodes.shape)
            targets_tr = targets_tr.replace(nodes=tf.math.add_n([targets_tr.nodes, noise_target]))
            real_output = model_disc(targets_tr, args.disc_num_iters)
            fake_output = model_disc(pred_graph, args.disc_num_iters)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gen_loss = tf.math.reduce_mean(gen_loss) / tf.constant(args.disc_num_iters, dtype=tf.float32)
            disc_loss = tf.math.reduce_mean(disc_loss) / tf.constant(args.disc_num_iters, dtype=tf.float32)

        gradients_of_generator = gen_tape.gradient(gen_loss, model_gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, model_disc.trainable_variables)

        optimizer_gen.apply(gradients_of_generator, model_gen.trainable_variables)
        optimizer_disc.apply(gradients_of_discriminator, model_disc.trainable_variables)

        return gen_loss, disc_loss

    def train_epoch(dataset):
        total_gen_loss = 0
        total_disc_loss = 0
        batch = 0
        max_batches = 100
        for inputs in loop_dataset(dataset, args.batch_size):
            inputs_tr, targets_tr = inputs
            gen_loss, disc_loss = train_step(inputs_tr, targets_tr)
            batch += 1
            total_gen_loss += gen_loss.numpy()
            total_disc_loss += disc_loss.numpy()
            if batch > max_batches:
                break
        return total_gen_loss/batch, total_disc_loss/batch, batch

    
    log_dir = os.path.join(output_dir, "logs/{}/train".format(time_stamp))
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        optimizer_gen=optimizer_gen,
        optimizer_disc=optimizer_disc,
        model_gen=model_gen, model_disc=model_disc)

    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir,
                                max_to_keep=5, keep_checkpoint_every_n_hours=8)
    logging.info("Loading latest checkpoint from: {}".format(ckpt_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)


    start_time = time.time()
    for epoch in range(n_epochs):

        training_dataset = training_dataset.shuffle(args.shuffle_size, seed=12345, reshuffle_each_iteration=True)

        gen_loss, disc_loss, batches = train_epoch(training_dataset)
        this_epoch = time.time()

        logging.info("{} epoch takes {:.2f} mins with generator loss {:.4f}, discriminator loss {:.4f} in {} batches".format(
            epoch, (this_epoch-start_time)/60., gen_loss, disc_loss, batches))

        ckpt_manager.save()
        # log some metrics
        with train_summary_writer.as_default():
            tf.summary.scalar("generator loss", gen_loss, step=epoch)
            tf.summary.scalar("discriminator loss", disc_loss, step=epoch)
            tf.summary.scalar("time", (this_epoch-start_time)/60., step=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("input_dir", help='input directory that contains subfolder of train, val and test')
    add_arg("output_dir", help="where the model and training info saved")
    add_arg("--patterns", help='file patterns', default='*')
    add_arg('-d', '--distributed', action='store_true', help='data distributed training')
    add_arg("--learning-rate", help='learing rate', default=0.0005, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)
    add_arg("--batch-size", type=int, help='training/evaluation batch size', default=500)
    add_arg("--shuffle-size", type=int, help="number of events for shuffling", default=650)

    add_arg("--noise-dim", type=int, help='dimension of noises', default=8)
    add_arg("--disc-num-iters", type=int, help='number of message passing for discriminator', default=4)
    add_arg("--disc-alpha", type=float, help='inversely scale true dataset in the loss calculation', default=0.1)
    add_arg("--disc-beta", type=float, help='scales generated dataset in the loss calculation', default=0.8)

    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],\
        default="INFO")
    args, _ = parser.parse_known_args()
    # print(args)

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train_and_evaluate(args)
