import os

import tensorflow as tf
import sonnet as snt
from root_gnn.utils import load_yaml
from root_gnn import model as GNN



def create_load_model(config):
    config = load_yaml(config)
    config_tr = config['parameters']

    prod_name = config['prod_name']
    ckpt_name = 'checkpoint'
    modeldir = os.path.join(config['output_dir'], prod_name)
    global_batch_size = config_tr['batch_size']
    num_processing_steps_tr = config_tr.get('num_iters', 0)      ## level of message-passing
    
    learning_rate = config_tr['learning_rate']
    optimizer = snt.optimizers.Adam(learning_rate)
    model = getattr(GNN, config['model_name'])()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=modeldir, max_to_keep=5)
    if os.path.exists(os.path.join(modeldir, ckpt_name)):
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Loaded latest checkpoint from {}".format(modeldir))
    else:
        raise ValueError("Cannot find model at:", modeldir)
    
    return (model, num_processing_steps_tr, global_batch_size)