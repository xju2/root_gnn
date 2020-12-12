import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from root_gnn.src.models.base import InteractionNetwork
from root_gnn.src.models.base import make_mlp_model
from root_gnn.src.models.base import MLPGraphNetwork


LATENT_SIZE = 128 # do not change.. should be the same value as in base.py

class DecaySimulator(snt.Module):
    def __init__(self, name="DecaySimulator"):
        super(DecaySimulator, self).__init__(name=name)

        self._node_linear = make_mlp_model()
        self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name='node_rnn')
        self._node_proper =  snt.nets.MLP([4], activate_final=False)


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

        self._global_encoder_block = blocks.GlobalBlock(
            global_model_fn=make_mlp_model,
            use_edges=True,
            use_nodes=True,
            use_globals=False,
            nodes_reducer=tf.math.unsorted_segment_sum,
            edges_reducer=tf.math.unsorted_segment_sum,
            name='global_encoder_block'
        )

        self._core = MLPGraphNetwork()

        # self._core = InteractionNetwork(
        #     edge_model_fn=make_mlp_model,
        #     node_model_fn=make_mlp_model,
        #     reducer=tf.math.unsorted_segment_sum
        # )

        # # Transforms the outputs into appropriate shapes.
        node_output_size = 64
        node_fn =lambda: snt.Sequential([
            snt.nets.MLP([node_output_size],
                        activation=tf.nn.relu, # default is relu
                        name='node_output')])

        global_output_size = 1
        global_fn = lambda: snt.Sequential([
            snt.nets.MLP([global_output_size],
                        activation=tf.nn.relu, # default is relu
                        name='global_output'),
            tf.sigmoid])

        self._output_transform = modules.GraphIndependent(
            edge_model_fn=None, 
            node_model_fn=node_fn,
            global_model_fn=global_fn)


    def __call__(self, input_op, max_nodes):
        node_pred = []
        global_pred = []

        node_hidden_state = tf.zeros([1, LATENT_SIZE], dtype=tf.float32, name='initial_node_hidden')
        latent = input_op
        for inode in range(max_nodes):
            # print("----loopping----", inode)
            # print(latent.nodes.numpy())
            # print(latent.n_node.numpy())
            # print(latent.edges.numpy())
            # print(latent.senders.numpy())
            # print(latent.receivers.numpy())
            nodes = latent.nodes

            # encode nodes, edges and globals
            # write separately for easily debugging
            latent = self._node_encoder_block(latent)
            latent = self._edge_block(latent)
            latent = self._global_encoder_block(latent)
            # message passing and output predictions
            latent = self._core(latent)
            latent = self._output_transform(latent)

            node_embedding = self._node_linear(latent.nodes)
            node_embedding = tf.math.reduce_sum(node_embedding, axis=0, keepdims=True, name='reduce_node_embedding')
            # print("node embedding:", node_embedding.shape)
            # print("node hiddent state:", node_hidden_state.shape)
            node_embedding, node_hidden_state = self._node_rnn(node_embedding, node_hidden_state)
            node_output = self._node_proper(node_embedding)

            # save output for loss calculations
            global_pred.append(latent.globals)
            node_pred.append(tf.squeeze(node_output))

            # update the graph by adding a new node with features as predicted
            # construct a fully-connected graph
            # n_node_tobe = [tf.add(latent.n_node[0], 1)]
            n_nodes = tf.math.reduce_sum(latent.n_node)
            n_nodes = tf.add(n_nodes, 1)
            nodes_tobe = tf.concat([nodes, node_output], axis=0, name='add_new_node')
            rng = tf.range(n_nodes)
            receivers, senders = tf.meshgrid(rng, rng)
            n_edge = n_nodes * n_nodes
            ind = tf.cast(1 - tf.eye(n_nodes), bool)
            receivers = tf.boolean_mask(receivers, ind)
            senders = tf.boolean_mask(senders, ind)
            n_edge -= n_nodes
            receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
            senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
            edges = tf.ones([1, 1], dtype=tf.float32)
            n_edge = tf.reshape(n_edge, [1])
            n_node = tf.reshape(n_nodes, [1])

            latent = latent.replace(nodes=nodes_tobe, n_node=n_node,\
                n_edge=n_edge, edges=edges, senders=senders, receivers=receivers)

        return node_pred, global_pred