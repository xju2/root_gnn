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

from root_gnn import model as all_models
from root_gnn.src.datasets import topreco_v2 as topreco
from root_gnn.src.datasets import graph
from root_gnn.utils import load_yaml


target_scales = np.array([145.34593924, 145.57711889, 432.92148524, 281.44161905, 1, 1]*topreco.n_max_tops).T.reshape((-1,))
target_mean = np.array([6.74674671e-02, -6.17142186e-02,  4.18239305e-01, 4.24881531e+02, 0, 0]*topreco.n_max_tops).T.reshape((-1,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN for predicting Top quark properties')
    add_arg = parser.add_argument
    add_arg('config', help='configuration file')
    add_arg("--no-gpu", help='do not use GPU', action='store_true')
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # add ops to save and restore all the variables
    prod_name = config['prod_name']
    output_dir = os.path.join(config['output_dir'], prod_name)
    os.makedirs(output_dir, exist_ok=True)

    config_tr = config['parameters']
    global_batch_size = config_tr['batch_size']   # need optimization
    num_processing_steps_tr = config_tr['num_iters']
    metric_name = config_tr['earlystop_metric']
    metric_dict = {
        "auc_te": 0.0, "acc_te": 0.0, "prec_te": 0.0, "rec_te": 0.0, "loss_te": 0.0
    }
    if metric_name not in metric_dict.keys():
        msg = "earlystop_metric: {} not supported. Use one of the following\n".format(metric_name) + "\n".join(list(metric_dict.keys()))
        raise ValueError(msg)
    acceptable_fails = config_tr['acceptable_failure']
    n_epochs = config_tr['epochs']  

 
    # prepare inputs
    tr_filenames = tf.io.gfile.glob(config['tfrec_dir_train'])
    n_train = len(tr_filenames)
    val_filenames = tf.io.gfile.glob(config['tfrec_dir_val'])
    n_val = len(val_filenames)

    print("Input file names: ", tr_filenames)
    print("{} training files".format(n_train))
    print("{} evaluation files".format(n_val))
    print("Model saved at {}".format(output_dir))

    shuffle_buffer_size = config_tr.get("shuffle_buffer_size", -1)
    AUTO = tf.data.experimental.AUTOTUNE
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # training_dataset = training_dataset.with_options(options)

    training_dataset = tf.data.TFRecordDataset(tr_filenames)
    training_dataset = training_dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
    n_train_graphs = sum([1 for _ in training_dataset])
    shuffle_train_size = n_train_graphs if shuffle_buffer_size < 0 else shuffle_buffer_size
    training_dataset = training_dataset.shuffle(shuffle_train_size, seed=12345, reshuffle_each_iteration=False)
    training_dataset = training_dataset.prefetch(AUTO)

    testing_dataset = tf.data.TFRecordDataset(val_filenames)
    testing_dataset = testing_dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
    n_test_graphs = sum([1 for _ in testing_dataset])
    shuffle_eval_size = n_test_graphs if shuffle_buffer_size < 0 else shuffle_buffer_size
    testing_dataset = testing_dataset.shuffle(shuffle_eval_size, seed=12345, reshuffle_each_iteration=False)
    testing_dataset = testing_dataset.prefetch(AUTO)

    print("Total {} graphs for training".format(n_train_graphs))

    learning_rate = config_tr['learning_rate']
    # learning_rate = tf.compat.v1.train.exponential_decay(start_learning_rate, )
    optimizer = snt.optimizers.Adam(learning_rate)
    model = getattr(all_models, 'FourTopPredictor')()

    # inputs, targets = doublet_graphs.create_graph(batch_size)
    with_batch_dim = False
    input_list = []
    target_list = []
    for dd in training_dataset.take(global_batch_size).as_numpy_iterator():
        input_list.append(dd[0])
        target_list.append(dd[1])

    inputs = utils_tf.concat(input_list, axis=0)
    targets = utils_tf.concat(target_list, axis=0)
    input_signature = (
        graph.specs_from_graphs_tuple(inputs, with_batch_dim),
        graph.specs_from_graphs_tuple(targets, with_batch_dim),
    )

    # n_max_tops = topreco.n_max_tops
    n_max_tops = 2
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
        
        alpha = tf.constant(1.0, dtype=tf.float32)
        loss_ops = [alpha * tf.compat.v1.losses.mean_squared_error(
                        target_op.globals[:, :n_max_tops*4],
                        output_op.globals[:, :n_max_tops*4])
            + tf.compat.v1.losses.log_loss(
                tf.cast(target_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*5], tf.int32),\
                tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*4:topreco.n_max_tops*5]))
            # + tf.compat.v1.losses.log_loss(
            #     tf.cast(target_op.globals[:, topreco.n_max_tops*5:], tf.int32),\
            #     tf.math.sigmoid(output_op.globals[:, topreco.n_max_tops*5:])) 
            for output_op in output_ops
        ]

        return tf.stack(loss_ops)


    @functools.partial(tf.function, input_signature=input_signature)
    def update_step(inputs_tr, targets_tr):
        print("Tracing update_step")
        with tf.GradientTape() as tape:
            output_graphs_tr = model(inputs_tr, num_processing_steps_tr)
            # print(output_graphs_tr[-1].globals.shape)
            # print(targets_tr.globals.shape)
            loss_ops_tr = loss_fcn(targets_tr, output_graphs_tr)
            loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(num_processing_steps_tr, dtype=tf.float32)

        gradients = tape.gradient(loss_op_tr, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return loss_op_tr

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir,\
        max_to_keep=3, keep_checkpoint_every_n_hours=1)
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restore from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    start_time = time.time()

    # log information
    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), T (elapsed seconds), Ltr (training loss), Lge (testing loss)"\
        "AUC, Accuracy, Precision, Recall\n"
    log_name = os.path.join(output_dir, "log_training.txt")
    with open(log_name, 'a') as f:
        f.write(out_str)

    previous_metric = 0.0
    threshold = 0.5
    n_fails = 0

    epoch_count = tf.Variable(0, trainable=False, name='epoch_count', dtype=tf.int64)
    now = time.time()
    out_str = ""
    for epoch in range(n_epochs):
        total_loss = 0.
        num_batches = 0

        in_list = []
        target_list = []
        for inputs in training_dataset:
            inputs_tr, targets_tr = inputs
            in_list.append(inputs_tr)
            target_list.append(targets_tr)
            if len(in_list) == global_batch_size:
                inputs_tr = utils_tf.concat(in_list, axis=0)
                targets_tr = utils_tf.concat(target_list, axis=0)

                # scale target values
                new_target = (targets_tr.globals - target_mean) / target_scales
                targets_tr = targets_tr.replace(globals=new_target)

                total_loss += update_step(inputs_tr, targets_tr).numpy()
                in_list = []
                target_list = []
                num_batches += 1


        ckpt_manager.save()
        elapsed = time.time() - start_time
        out_the_epoch = "{:.2f} minutes, epoch {:,} with loss {:.4f} in {:,} batches".format(
            elapsed/60., epoch, total_loss/num_batches/global_batch_size, num_batches)
        print(out_the_epoch)
        with open(log_name, 'a') as f:
            f.write(out_the_epoch + "\n")
        start_time = time.time()