import tensorflow as tf
from tensorflow.compat.v1 import logging 
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

from tensorflow.python.profiler import profiler_v2 as profiler

from root_gnn.utils import load_yaml
from root_gnn.src.datasets import graph
from root_gnn import model as all_models
from root_gnn import losses

printer = pprint.PrettyPrinter(indent=2)
class Trainer(object):
    def __init__(self, config, distribute=False):
        self._dist = self._init_workers(distribute)
        if self._dist.rank == 0:
            self._read_config(config)
            printer.pprint(self._args.__dict__)
        else:
            self._args = None

        if self._distributed:
            self._args = self._dist.comm.bcast(self._args, root=0)

    def _init_workers(self, distribute):
        if not no_horovod and distribute:
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

        self._args.__dict__.update(**dict({
            "batch_size": 1,
            "num_message_passing": 5,
            "learing_rate": 0.005,
            "epochs": 1,
            "earlystop_metric": "auc_te",
            "acceptable_fails": 1
        }))
        parameters = config.get('parameters', None)
        if parameters:
            self._args.__dict__.update(parameters)

        profiling = config.get('profiling', None)
        if profiling:
            self._args.do_profiling = True
            self._update_freq = profiling.get("update_freq", 'epoch')
            self._init_profile_batch(profiling.get('profile_batch', 2))
            self._do_profiling_only = profiling.get('do_profiling_only', False)
        else:
            self._args.do_profiling = False


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

    def execute(self):
        pass
    
    def _read_data(self, filenames):
        AUTO = tf.data.experimental.AUTOTUNE
        tr_filenames = tf.io.gfile.glob(filenames)
        n_files = len(tr_filenames)
        print("Read {} files".format(n_files))

        dataset = tf.data.TFRecordDataset(tr_filenames)
        dataset = dataset.map(graph.parse_tfrec_function, num_parallel_calls=AUTO)
        n_graphs = sum([1 for _ in dataset])
        return (dataset, n_graphs)


        


