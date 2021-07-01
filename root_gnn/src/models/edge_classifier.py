import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from root_gnn.src.models.base import InteractionNetwork
from root_gnn.src.models.base import make_mlp_model
from functools import partial


class EdgeLearnerBase(snt.Module):
    def __init__(self,
        edge_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        encoder_size: list=None,
        core_size: list=None,
        name="EdgeLearnerBase", **kwargs):
        super(EdgeLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(make_mlp_model, mlp_size=encoder_size, **kwargs)
        else:
            encoder_mlp_fn = partial(make_mlp_model, **kwargs)

        edge_block_args=dict(use_edges=False, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False)
        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = True
            node_block_args['use_sent_edges'] = True
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=encoder_mlp_fn,
            use_edges=False,
            name='edge_encoder_block',
            **edge_block_args
        )


        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=encoder_mlp_fn,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )

        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)

        self._core = InteractionNetwork(
            edge_model_fn=core_mlp_fn,
            node_model_fn=core_mlp_fn,
            reducer=tf.math.unsorted_segment_sum
        )

        self._output_transform = modules.GraphIndependent(edge_fn, None, None)


    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = dict(is_training=is_training)

        latent = self._edge_block(self._node_encoder_block(input_op, node_kwargs), edge_kwargs)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_model_kwargs=edge_kwargs, node_model_kwargs=node_kwargs)

            output_ops.append(self._output_transform(latent))

        return output_ops


# class EdgeClassifier