"""
Generative model based on Graph Neural Network
"""
import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import blocks
from graph_nets import modules

from root_gnn.src.models.base import make_mlp
from root_gnn.src.models.base import make_mlp_model
from root_gnn.src.models.base import MLPGraphNetwork

LATENT_SIZE = 128
NUM_LAYERS = 2

class SetsGenerator(snt.Module):
    def __init__(self, name="SetsGenerator", input_dim=12, out_dim=4):
        super(SetsGenerator, self).__init__(name=name)

        # graph creation
        self._node_linear = snt.Sequential([snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS,
                        activation=tf.nn.relu,
                        activate_final=False,
                        dropout_rate=0.30,
                        name="node_encoder_nn"),
                        snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
        self._node_reduction = snt.nets.MLP([input_dim], name='node_reduction_nn')

        # node properties
        self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name="node_rnn")
        self._node_prop_nn = snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS+[out_dim],
                activation=tf.nn.relu, activate_final=False,
                dropout_rate=0.30,
                name="node_prop_nn")

    def __call__(self, input_op, max_nodes, training):
        """
        Args: 
            input_op: 2D vector with dimensions [batch-size, features], 
            the latter containing [px, py, pz, E, N-dimension noises]
            max_nodes: maximum number of output nodes
            training: if in training mode, needed for `dropout`.

        Retruns:
            predicted node featurs with dimension of [batch-size, max-nodes, out-features] 
        """
        node_pred = []
        batch_size = input_op.shape[0]

        # predicting edges
        # TODO: concate the new embedding to the old ones and do the reduction before RNN.
        node_hidden_state = tf.zeros([batch_size, LATENT_SIZE], dtype=tf.float32, name='initial_node_hidden')
        node_embedding = input_op

        for inode in range(max_nodes):
            node_embedding = self._node_linear(node_embedding, training)

            # node properties
            node_embedding, node_hidden_state = self._node_rnn(node_embedding, node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding, training)
            node_pred.append(node_prop)
            node_embedding = self._node_reduction(node_embedding)

        node_pred = tf.transpose(tf.stack(node_pred, axis=0), perm=[1, 0, 2])
        node_pred = tf.reshape(node_pred, [-1, node_pred.shape[-1]])
        return node_pred


class SetsDiscriminator(snt.Module):
    def __init__(self, name="SetsDiscriminator"):
        super(SetsDiscriminator, self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=make_mlp_model,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block')

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
        global_output_size = 1
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, global_output_size],
                         name='global_output'), tf.sigmoid])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._global_block(self._edge_block(self._node_encoder_block(input_op)))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            output_ops.append(self._output_transform(latent))

        return output_ops