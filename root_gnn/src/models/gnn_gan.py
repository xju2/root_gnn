"""
Generative model based on Graph Neural Network
"""
import tensorflow as tf
import sonnet as snt
from root_gnn.src.models.base import make_mlp

LATENT_SIZE = 128
NUM_LAYERS = 2

class GraphGenerator(snt.Module):
    def __init__(self, name="GraphGenerator", max_prev_nodes=15, noise_dimension=8):
        super(GraphGenerator, self).__init__(name=name)

        # graph creation
        self._node_linear = snt.Sequential([snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS,
                        activation=tf.nn.relu,
                        activate_final=False,
                        dropout_rate=0.30,
                        name="node_encoder_nn"),
                        snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
        self._node_reduction = snt.nets.MLP([noise_dimension + 4], name='node_reduction_nn')

        self._edges_rnn = snt.GRU(hidden_size=LATENT_SIZE, name='edge_rnn')
        self._edges_decoder = snt.nets.MLP([LATENT_SIZE, max_prev_nodes],
                activation=tf.nn.relu, activate_final=False,
                dropout_rate=0.30,
                name="edge_list_nn")

        # node properties
        self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name="node_rnn")
        self._node_prop_nn = snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS+[4],
                activation=tf.nn.relu, activate_final=False,
                dropout_rate=0.30,
                name="node_prop_nn")

    def __call__(self, input_op, max_nodes, training):
        """
        input_op: a vector containing [px, py, pz, E, N-dimension noises]
        """
        encoded_adj = []
        node_pred = []
        batch_size = input_op.shape[0]

        # predicting edges
        # TODO: concate the new embedding to the old ones and do the reduction before RNN.
        edge_hidden_state = tf.zeros([batch_size, LATENT_SIZE], dtype=tf.float32, name='initial_edge_hidden')
        node_hidden_state = tf.zeros([batch_size, LATENT_SIZE], dtype=tf.float32, name='initial_node_hidden')
        node_embedding = input_op

        for inode in range(max_nodes-1):
            node_embedding = self._node_linear(node_embedding, training)

            # adjancy matrix
            edge_embedding, edge_hidden_state = self._edges_rnn(node_embedding, edge_hidden_state)
            edge_pred = self._edges_decoder(edge_embedding, training)
            encoded_adj.append(edge_pred)

            # node properties
            node_embedding, node_hidden_state = self._node_rnn(node_embedding, node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding, training)
            node_pred.append(node_prop)
            node_embedding = self._node_reduction(node_embedding)

        # node_pred = tf.concat(node_pred, axis=0, name='concat_node_props')

        return encoded_adj, node_pred