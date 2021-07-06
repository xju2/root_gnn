import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from root_gnn.src.models.base import MLPGraphNetwork
from root_gnn.src.models.base import make_mlp_model
from functools import partial


class GlobalLearnerBase(snt.Module):

    def __init__(self,
        global_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="GlobalLearnerBase", **kwargs):
        super(GlobalLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(make_mlp_model, mlp_size=encoder_size, **kwargs)
        else:
            encoder_mlp_fn = partial(make_mlp_model, **kwargs)

        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False)
        edge_block_args=dict(use_edges=False, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
        global_block_args=dict(use_edges=True, use_nodes=True, use_globals=False)
        
        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = True
            node_block_args['use_sent_edges'] = True
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False
        if with_global_inputs:
            global_block_args['use_globals']=True

        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=make_mlp_model,
            name='edge_encoder_block', **edge_block_args)

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=make_mlp_model,
            name='node_encoder_block', **node_block_args)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=make_mlp_model, name='global_encoder_block', **global_block_args)
        
        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)

        self._core = MLPGraphNetwork(nn_fn=core_mlp_fn)

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)

        latent = self._global_block(
            self._edge_encoder_block(self._node_encoder_block(input_op, node_kwargs), edge_kwargs), global_kwargs)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops


class GlobalClassifier(GlobalLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        name="GlobalClassifier", **kwargs):

        edge_output_size = 1
        if decoder_size is not None:
            decoder_size += [edge_output_size]
        else:
            decoder_size = [edge_output_size]

        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='edge_classifier_output'),
            tf.sigmoid])
        
        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size,
            name=name, **kwargs)


class GlobalRegression(GlobalLearnerBase):
    def __init__(self, global_output_size,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        name="GlobalRegression", **kwargs):

        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='global_regresssion_output')])

        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size, name=name, **kwargs)