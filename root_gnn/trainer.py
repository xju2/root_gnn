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

import numpy as np

from tensorflow.python.profiler import profiler_v2 as profiler

from graph_nets import utils_tf
from graph_nets import utils_np
import sonnet as snt

from root_gnn.utils import load_yaml
from root_gnn.src.datasets import graph
from root_gnn import model as all_models
from root_gnn import losses

verbosities = ['DEBUG','ERROR', "FATAL", "INFO", "WARN"]
printer = pprint.PrettyPrinter(indent=2)
class Trainer(object):
    def __init__(self, config, distributed=False, verbose="INFO"):
        if verbose.upper() not in verbosities:
            raise ValueError("Allowed verbosities: {}".format(
                ", ".join(verbosities)))
        self._dist = self._init_workers(distributed)
        if self._dist.rank == 0:
            self._read_config(config)
        else:
            self._args = None

        if self._distributed:
            self._args = self._dist.comm.bcast(self._args, root=0)

    def execute(self):
        logging.info("I am rank {} of  total {} ranks".format(self._dist.rank, self._dist.size))
        train_files, eval_files = self._get_train_eval_data()
        logging.info("rank {} has {} training files and {} evaluation files".format(
            self._dist.rank, len(train_files), len(eval_files)))
        
        training_dataset = self._read_data(train_files)
        testing_dataset = self._read_data(eval_files)

        batch_size = self._args.batch_size

        learning_rate = self._args.learning_rate
        optimizer = snt.optimizers.Adam(learning_rate)
        model = getattr(all_models, self._args.model_name)

        with_batch_dim = False
        input_list = []
        target_list = []
        for dd in training_dataset.take(batch_size).as_numpy_iterator():
            input_list.append(dd[0])
            target_list.append(dd[1])

        inputs = utils_tf.concat(input_list, axis=0)
        targets = utils_tf.concat(target_list, axis=0)
        input_signature = (
            graph.specs_from_graphs_tuple(inputs, with_batch_dim),
            graph.specs_from_graphs_tuple(targets, with_batch_dim)
        )

        loss_fcn = getattr(losses, self._args.loss_name)(*[float(x) for x in self._args.loss_args])

        num_processing_steps_tr = self._args.num_message_passing

        @functools.partial(tf.function, input_signature=input_signature)
        def update_step(inputs_tr, targets_tr):
            print("Tracing update_step")
            with tf.GradientTape() as tape:
                outputs_tr = model(inputs_tr, num_processing_steps_tr)
                loss_ops_tr = loss_fcn(targets_tr, outputs_tr)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(num_processing_steps_tr, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, model.trainable_variables)
            optimizer.apply(gradients, model.trainable_variables)
            return outputs_tr, loss_op_tr

        time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        output_dir = self._args.output_dir

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir,\
            max_to_keep=3, keep_checkpoint_every_n_hours=1)
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        if ckpt_manager.latest_checkpoint:
            print("Restore from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        data_iterator = training_dataset.as_numpy_iterator()
        # computational graphs
        func_log_dir = "logs/{}/funcs".format(time_stamp)
        func_writer = tf.summary.create_file_writer(os.path.join(output_dir, func_log_dir))
        profiling_steps = config_tr.get("profiling_steps", 10)
        profile_logdir = os.path.join(output_dir, "logs/{}/profiling".format(time_stamp))

        previous_metric = 0.0
        threshold = 0.5
        n_fails = 0
        n_epochs = self._args.epochs

        epoch_count = tf.Variable(0, trainable=False, name='epoch_count', dtype=tf.int64)
        for epoch in range(n_epochs):
            total_loss = 0.
            num_batches = 0

            in_list = []
            target_list = []

            data_iterator = training_dataset.as_numpy_iterator()

            for inputs in training_dataset:
                inputs_tr, targets_tr = inputs
                in_list.append(inputs_tr)
                target_list.append(targets_tr)
                if len(in_list) == batch_size:
                    inputs_tr = utils_tf.concat(in_list, axis=0)
                    targets_tr = utils_tf.concat(target_list, axis=0)
                    total_loss += update_step(inputs_tr, targets_tr)[1].numpy()
                    in_list = []
                    target_list = []
                    num_batches += 1
                    if self._args.do_profiling:
                        if epoch == 0 and num_batches==self._args.start_batch:


            ckpt_manager.save()

            eval_output_name = os.path.join(output_dir, "eval_{}.npz".format(
                ckpt_manager.checkpoint.save_counter.numpy()))

            loss_tr = total_loss/num_batches

            elapsed = time.time() - start_time
            inputs_te_list = []
            target_te_list = []
            predictions = []
            truth_info = []
            num_batches_te = 0
            total_loss_te = 0
            for inputs in testing_dataset:
                inputs_te, targets_te = inputs
                inputs_te_list.append(inputs_te)
                target_te_list.append(targets_te)
                if len(inputs_te_list) == global_batch_size:
                    inputs_te = utils_tf.concat(inputs_te_list, axis=0)
                    targets_te = utils_tf.concat(target_te_list, axis=0)
                    outputs_te = model(inputs_te, num_processing_steps_tr)
                    total_loss_te += (tf.math.reduce_sum(
                        loss_fcn(targets_te, outputs_te))/tf.constant(
                            num_processing_steps_tr, dtype=tf.float32)).numpy()
                    if loss_name == "GlobalLoss":
                        predictions.append(outputs_te[-1].globals)
                        truth_info.append(targets_te.globals)
                    else:
                        predictions.append(outputs_te[-1].edges)
                        truth_info.append(targets_te.edges)
                    inputs_te_list = []
                    target_te_list = []
                    num_batches_te += 1

            loss_te = total_loss_te / num_batches_te
            predictions = np.concatenate(predictions, axis=0)
            truth_info = np.concatenate(truth_info, axis=0)
            # print(tf.math.reduce_sum(predictions).numpy(), tf.math.reduce_sum(truth_info).numpy())

            y_true, y_pred = (truth_info > threshold), (predictions > threshold)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, predictions)
            metric_dict['auc_te'] = sklearn.metrics.auc(fpr, tpr)
            metric_dict['acc_te'] = sklearn.metrics.accuracy_score(y_true, y_pred)
            metric_dict['pre_te'] = sklearn.metrics.precision_score(y_true, y_pred)
            metric_dict['rec_te'] = sklearn.metrics.recall_score(y_true, y_pred)
            metric_dict['loss_te'] = loss_te
            out_str = "* {:05d}, T {:.1f}, Ltr {:.4f}, Lge {loss_te:.4f}, AUC {auc_te:.4f}, A {acc_te:.4f}, P {pre_te:.4f}, R {rec_te:.4f}".format(
                epoch, elapsed, loss_tr, **metric_dict)
            print(out_str)
            with open(log_name, 'a') as f:
                f.write(out_str+"\n")
            np.savez(eval_output_name, predictions=predictions, truth_info=truth_info)

            # save metrics to the summary file
            metric_dict['loss_tr'] = loss_tr
            with writer.as_default():
                for key,val in metric_dict.items():
                    tf.summary.scalar(key, val, step=epoch_count)
                writer.flush()
            epoch_count.assign_add(1)

            metric = metric_dict[metric_name]
            if metric < previous_metric:
                print("Current metric {} {:.4f} is lower than previous {:.4f}.".format(metric_name, metric, previous_metric))
                if n_fails < acceptable_fails:
                    n_fails += 1
                else:
                    print("Reached maximum failure threshold: {} times. Stop Training".format(acceptable_fails))
                    break
            else:
                previous_metric = metric


    def _init_workers(self, distributed):
        if not no_horovod and distributed:
            self._distributed = True
            hvd.init()
            assert hvd.mpi_threads_supported()
            from mpi4py import MPI
            assert hvd.size() == MPI.COMM_WORLD.Get_size()
            comm = MPI.COMM_WORLD
            return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                                local_rank=hvd.local_rank(),
                                local_size=hvd.local_size(), comm=comm)
        else:
            self._distributed = False
            return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)

    def _read_config(self, config_name):
        config = load_yaml(config_name)
        self._args = SimpleNamespace()

        train_files = config.get("tfrec_dir_train", None)
        if train_files is None:
            raise ValueError("provide Training data via 'tfrec_dir_train'")

        eval_files = config.get("tfrec_dir_val", None)
        if eval_files is None:
            raise ValueError("provide Evaluation data via 'tfrec_dir_val'")

        output_dir = config.get("output_dir", "TestArea")
        if "gs" not in output_dir:
            os.makedirs(output_dir, exist_ok=True)

        model_name = config.get("model_name", None)
        if model_name is None:
            raise ValueError("model cannot be empty, provide via 'model_name'")
        if model_name not in all_models.__all__:
            raise ValueError("{} is not valid. Choices are {}".format(
                model_name, ", ".join(all_models.__all__)))

        loss_name = config.get("loss_name", None)
        if loss_name is None:
            raise ValueError("loss name cannot be empty, provide via 'loss_name'")
        if loss_name not in losses.__all__:
            raise ValueError("{} is not valid. Choices are {}".format(
                loss_name, ", ".join(losses.__all__)))

        self._args.train_files = train_files
        self._args.eval_files = eval_files
        self._args.output_dir = output_dir
        self._args.model_name = model_name
        self._args.loss_name = loss_name
        self._args.loss_args = config.get("loss_args", None)
        if self._args.loss_args is None:
            raise ValueError("arguments for loss function could not be None, provide via 'loss_args' ")

        self._args.__dict__.update(**dict({
            "batch_size": 1,
            "num_message_passing": 5,
            "learing_rate": 0.005,
            "epochs": 1,
            "earlystop_metric": "auc_te",
            "acceptable_fails": 1,
            "shuffle_buffer_size": 4,

        }))
        parameters = config.get('parameters', None)
        if parameters:
            self._args.__dict__.update(parameters)

        self._args.do_profiling = config.get('do_profiling', False)
        self._args.update_freq = config.get("update_freq", 'epoch')
        self._init_profile_batch(config.get('profile_batch', 2))

    def _init_profile_batch(self, profile_batch):
        """ Taken from Tensorboard """
        profile_error_message = (
            'profile_batch must be a non-negative integer or 2-tuple of positive '
            'integers. A pair of positive integers signifies a range of batches '
            'to profile. Found: {}'.format(profile_batch))

        if isinstance(profile_batch, six.string_types):
            profile_batch = str(profile_batch).split(',')
            profile_batch = tf.nest.map_structure(int, profile_batch)
        
        if isinstance(profile_batch, int):
            self._args.start_batch = profile_batch
            self._args.stop_batch = profile_batch
        elif isinstance(profile_batch, (tuple, list)) and len(profile_batch) == 2:
            self._args.start_batch, self._args.stop_batch = profile_batch
        else:
            raise ValueError(profile_error_message)

        if self._args.start_batch < 0 or self._args.stop_batch < self._args.start_batch:
            raise ValueError(profile_error_message)

        if self._args.start_batch > 0:
            profiler.warmup()  # Improve the profiling accuracy.

        # True when a trace is running.
        self._args.is_tracing = False

        # Setting `profile_batch=0` disables profiling.
        self._args.should_trace = not (self._args.start_batch == 0 and self._args.stop_batch == 0)

    def _read_data(self, filenames):
        AUTO = tf.data.experimental.AUTOTUNE
        tr_filenames = tf.io.gfile.glob(filenames)
        n_files = len(tr_filenames)
        # print("Read {} files".format(n_files))

        dataset = tf.data.TFRecordDataset(tr_filenames)
        dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
        buffer_size = self._args.shuffle_buffer_size * self._args.batch_size
        reshuffle = True
        if buffer_size <= 0:
            buffer_size =  sum([1 for _ in dataset])
            reshuffle = False # too much computation, not reshuffle
        dataset = dataset.shuffle(buffer_size, seed=12345, reshuffle_each_iteration=reshuffle).prefetch(AUTO)
        return dataset

    def _get_train_eval_data(self):
        if self._dist.rank == 0:
            train_files = tf.io.gfile.glob(self._args.train_files)
            eval_files = tf.io.gfile.glob(self._args.eval_files)
            train_files = [x.tolist() for x in np.array_split(train_files, self._dist.size)]
            eval_files = [x.tolist() for x in np.array_split(eval_files, self._dist.size)]
        else:
            train_files = None
            eval_files = None

        if self._distributed:
            train_files = self._dist.comm.scatter(train_files, root=0)
            eval_files = self._dist.comm.scatter(eval_files, root=0)
        else:
            train_files = train_files[0]
            eval_files = eval_files[0]
        return train_files, eval_files