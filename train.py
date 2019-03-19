#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import networkx as nx
    import numpy as np
    import pandas as pd

    from graph_nets import utils_tf
    from graph_nets import utils_np

    from root_gnn import dataset
    from root_gnn import utils_train
    from root_gnn.model import GeneralClassifier
    import tensorflow as tf

    input_dir = '/global/homes/x/xju/project/xju/gnn_examples/H4l_ggF_vs_VBF/input'
    input_data = dataset.dataset(input_dir, 'ggf.root', 'vbf.root', 'tree_incl_all')

    n_graphs = 500

    tf.reset_default_graph()
    input_graphs, target_graphs = input_data.get_graphs(n_graphs, is_training=False)
    input_ph  = utils_tf.placeholders_from_networkxs(input_graphs, force_dynamic_num_graphs=True)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs, force_dynamic_num_graphs=True)

    model = GeneralClassifier()
    num_processing_steps_tr = 3
    output_ops_tr = model(input_ph, num_processing_steps_tr)

    loss_ops_tr = utils_train.create_loss_ops(target_ph, output_ops_tr)
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

    global_step = tf.Variable(0, trainable=False)
    start_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(
        start_learning_rate, global_step,
        decay_steps=400,
        decay_rate=0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

    input_ph, target_ph = utils_train.make_all_runnable_in_session(input_ph, target_ph)

    output_dir = 'trained_results/try_007'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    ckpt_name = 'checkpoint_{:05d}.ckpt'
    import glob, re
    files = glob.glob(output_dir+"/*.ckpt.meta")
    last_iteration = 0 if len(files) < 1 else max([
            int(re.search('checkpoint_([0-9]*).ckpt.meta', os.path.basename(x)).group(1))
            for x in files])
    print("last iteration:", last_iteration)


    sess = tf.Session()
    saver = tf.train.Saver()

    if last_iteration > 0:
        print("loading checkpoint:", os.path.join(output_dir, ckpt_name.format(last_iteration)))
        saver.restore(sess, os.path.join(output_dir, ckpt_name.format(last_iteration)))
    else:
        init_ops = tf.global_variables_initializer()
        # saver must be created before init_ops is run!
        sess.run(init_ops)

    import time
    log_name = 'big.log'
    log_every_seconds = 60
    iterations = 8000000
    iter_per_job = 10000

    out_str  = time.strftime('%d %b %Y %H:%M:%S', time.localtime())
    out_str += '\n'
    out_str += "# (iteration number), T (elapsed seconds), Ltr (training loss), Lte (testing loss), Precision, Recall\n"
    log_name = os.path.join(output_dir, log_name)
    with open(log_name, 'a') as f:
        f.write(out_str)

    start_time = time.time()
    last_log_time = start_time

    irun = 0
    for itr in range(last_iteration, iterations):
        if irun > iter_per_job:
            break
        else:
            irun += 1
        input_graphs, target_graphs = input_data.get_graphs(n_graphs, is_training=True)

        input_graphs_ntuple = utils_np.networkxs_to_graphs_tuple(input_graphs)
        target_graphs_ntuple = utils_np.networkxs_to_graphs_tuple(target_graphs)

        feed_dict = {
            input_ph: input_graphs_ntuple,
            target_ph: target_graphs_ntuple
        }

        train_values = sess.run({
            "step": step_op,
            "target": target_ph,
            "loss": loss_op_tr,
            "outputs": output_ops_tr
        }, feed_dict=feed_dict)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        if elapsed_since_last_log > log_every_seconds:
            # save a checkpoint
            last_log_time = the_time
            input_graphs, target_graphs = input_data.get_graphs(n_graphs, is_training=False)
            input_graphs_ntuple = utils_np.networkxs_to_graphs_tuple(input_graphs)
            target_graphs_ntuple = utils_np.networkxs_to_graphs_tuple(target_graphs)
            feed_dict = {
                input_ph: input_graphs_ntuple,
                target_ph: target_graphs_ntuple
            }

            test_values = sess.run({
                "target": target_ph,
                "loss": loss_op_tr,
                "outputs": output_ops_tr
            }, feed_dict=feed_dict)

            elapsed = the_time - start_time

            correct_tr, solved_tr = utils_train.compute_matrics(
                test_values["target"], test_values["outputs"][-1])

            out_str = "# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Precision {:.4f}, Recall {:.4f}\n".format(
                itr, elapsed, train_values["loss"], test_values["loss"],
                correct_tr, solved_tr)
            with open(log_name, 'a') as f:
                f.write(out_str)


            save_path = saver.save(
                sess, os.path.join(output_dir, ckpt_name.format(itr)))
    sess.close()

