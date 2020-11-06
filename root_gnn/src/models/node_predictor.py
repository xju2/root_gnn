import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from graph_nets.modules import InteractionNetwork

from root_gnn.src.datasets.topreco import n_target_node_features
from root_gnn.src.models.base import make_mlp_model

LATENT_SIZE = 128

class FourTopPredictor(snt.Module):
    """
    The mdoel predicts 7 parameters for each node with only the
    leading 4 nodes are used since they are the ghost nodes serving
    as the top quark candidates.
    """
    def __init__(self, name="FourTopPredictor"):
        super(FourTopPredictor, self).__init__(name=name)


        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=make_mlp_model,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block'
        )
        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=make_mlp_model,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )

        self._core = InteractionNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            reducer=tf.math.unsorted_segment_sum
        )

        # Transforms the outputs into appropriate shapes.
        node_output_size = n_target_node_features
        node_fn = lambda: snt.nets.MLP([node_output_size],
                        activation=tf.nn.relu, # default is relu
                        name='node_output')

        self._output_transform = modules.GraphIndependent(node_model_fn=node_fn)


    def __call__(self, input_op, num_processing_steps):
        latent = self._edge_block(self._node_encoder_block(input_op))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

            output_ops.append(self._output_transform(latent))

        return output_ops