import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from graph_nets.modules import InteractionNetwork

from root_gnn.src.datasets.topreco import n_target_node_features, n_max_tops
from root_gnn.src.models.base import MLPGraphNetwork
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

        self._global_block = blocks.GlobalBlock(
            global_model_fn=make_mlp_model,
            use_edges=True,
            use_nodes=True,
            use_globals=False,
        )

        self._core = MLPGraphNetwork()

        # Transforms the outputs into appropriate shapes.
        global_output_size = n_target_node_features * n_max_tops
        self._global_nn = snt.nets.MLP([128, 128, global_output_size],
                        activation=tf.nn.leaky_relu, # default is relu, tanh
                        dropout_rate=0.30,
                        name='global_output'
                    )

    def __call__(self, input_op, num_processing_steps, is_training=True):
        latent = self._global_block(self._edge_block(self._node_encoder_block(input_op)))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

            output = latent.replace(globals=self._global_nn(latent.globals, is_training))
            output_ops.append(output)

        return output_ops