"""
Generative model based on Graph Neural Network
"""
import tensorflow as tf
import sonnet as snt
from root_gnn.src.models.base import make_mlp

LATENT_SIZE = 128
NUM_LAYERS = 2

class GraphGenerator(snt.Module):
    def __init__(self, name="GraphGenerator", max_nodes=84, max_prev_nodes=15):
        super(GraphGenerator, self).__init__(name=name)

        self.max_nodes = max_nodes
        # graph creation
        self._node_linear = snt.Sequential([snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS,
                        activation=tf.nn.relu,
                        activate_final=False,
                        dropout_rate=0.30,
                        name="node_encoder_mlp"),
                        snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
        self._edges_rnn = snt.GRU(hidden_size=LATENT_SIZE, name='edge_rnn')
        self._edges_decoder = snt.nets.MLP([LATENT_SIZE, max_prev_nodes],
                activation=tf.nn.relu, activate_final=False,
                dropout_rate=0.30,
                name="node_prope_nn")

        # node properties
        self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, node="node_rnn")
        self._node_prop_nn = snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS+[4],
                activation=tf.nn.relu, activate_final=False,
                dropout_rate=0.30,
                name="node_prope_nn")

    def __call__(self, input_op, is_training):
        """
        input_op: a vector containing [px, py, pz, E, N-dimension noises]
        """
        encoded_adj = []
        node_pred = []

        # predicting edges
        # TODO: concate the new embedding to the old ones and do the reduction before RNN.
        edge_hidden_state = tf.zeros([1, LATENT_SIZE], dtype=tf.float32, name='initial_edge_hidden')
        node_hidden_state = tf.zeros([1, LATENT_SIZE], dtype=tf.float32, name='initial_node_hidden')
        node_embedding = input_op

        for inode in range(self.max_nodes):
            node_embedding = self._node_linear(node_embedding, is_training)
            edge_embedding, edge_hidden_state = self._edges_rnn(node_embedding, edge_hidden_state)
            edge_pred = self._edges_decoder(edge_embedding)
            encoded_adj.append(edge_pred)

            # node properties
            node_embedding, node_hidden_state = self._edges_rnn(node_embedding, node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding)
            node_pred.append(tf.squeeze(node_prop))

        return encoded_adj, node_pred
