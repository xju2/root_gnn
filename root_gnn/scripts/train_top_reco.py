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
import tqdm

import numpy as np
import sklearn.metrics


from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from types import SimpleNamespace
import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
try:
    import horovod.tensorflow as hvd
except ModuleNotFoundError:
    logging.warning("No horvod module, cannot perform distributed training")


from root_gnn import model as all_models
from root_gnn.src.datasets import topreco
from root_gnn.src.datasets import graph
from root_gnn.utils import load_yaml

from root_gnn.trainer import read_dataset
from root_gnn.trainer import loop_dataset
from root_gnn.trainer import get_signature


target_scales = np.array([145.4, 145.6, 432.9, 281.4, 1, 1, 1]*topreco.n_max_tops).reshape(
    (topreco.n_max_tops, -1)).T.reshape((-1,))
target_mean = np.array([0.067, -0.062, 0.042, 420, 0, 0, 0]*topreco.n_max_tops).reshape(
    (topreco.n_max_tops, -1)).T.reshape((-1,))


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
    # dist = init_workers(args.distributed)
    dist = init_workers()

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
    batch_size = args.batch_size

    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    num_processing_steps_tr = args.num_iters     ## level of message-passing
    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(n_epochs, batch_size))
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
    training_dataset = training_dataset.repeat(n_epochs).prefetch(AUTO).shuffle(
        args.shuffle_size, seed=12345, reshuffle_each_iteration=False
    )

    input_signature = get_signature(training_dataset, batch_size, with_bool=True)


    learning_rate = args.learning_rate
    optimizer = snt.optimizers.Adam(learning_rate)
    model = getattr(all_models, 'FourTopPredictor')()

    # training loss
    def loss_fcn(target_op, output_ops):
        # print("target size: ", target_op.nodes.shape)
        # print("output size: ", output_ops[0].nodes.shape)

        # loss_ops = [tf.compat.v1.losses.absolute_difference(
        #                 target_op.globals[:, :topreco.n_max_tops*4],
        #                 output_op.globals[:, :topreco.n_max_tops*4]) * 10
        #             + tf.compat.v1.losses.log_loss(
        #                 tf.cast(target_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*7], tf.int32),\
        #                 tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*7]))
        #     for output_op in output_ops
        # ]

        loss_ops = [tf.compat.v1.losses.absolute_difference(
                        target_op.globals[:, :topreco.n_max_tops*4],
                        output_op.globals[:, :topreco.n_max_tops*4])
            for output_op in output_ops
        ]

        # The following works well for the top 4-vector predictions
        # loss_ops = [ tf.compat.v1.losses.absolute_difference(
        #                     target_op.globals[:, :topreco.n_max_tops*4],\
        #                     output_op.globals[:, :topreco.n_max_tops*4])
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


    training_data = loop_dataset(training_dataset, batch_size)
    steps_per_epoch = ngraphs_train // batch_size
    log_dir = os.path.join(output_dir, "logs/{}/train".format(time_stamp))
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir,
                                max_to_keep=5, keep_checkpoint_every_n_hours=8)
    logging.info("Loading latest checkpoint from: {}".format(ckpt_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)


    start_time = time.time()
    with tqdm.trange(n_epochs*steps_per_epoch) as t:
        for step_num in t:
            epoch = tf.constant(int(step_num / steps_per_epoch), dtype=tf.int32)
            inputs_tr, targets_tr = next(training_data)
            new_target = (targets_tr.globals - target_mean) / target_scales
            targets_tr = targets_tr.replace(globals=new_target)
            loss =  train_step(inputs_tr, targets_tr, step_num==0).numpy()

            if step_num and (step_num % args.log_every_iter == 0):
                t.set_description('Epoch {}/{}'.format(epoch.numpy(), n_epochs))
                t.set_postfix(loss=loss)             
                ckpt_manager.save()

                # log some metrics
                this_epoch = time.time()
                with train_summary_writer.as_default():
                    # epoch = epoch.numpy()
                    tf.summary.scalar("loss", loss, step=step_num)
                    tf.summary.scalar("time", (this_epoch-start_time)/60., step=step_num)


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
    add_arg("--log-every-iter", type=int, help='logger every N interactions', default=100)
    args, _ = parser.parse_known_args()

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train_and_evaluate(args)