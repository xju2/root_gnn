"""
Generative model based on Graph Neural Network
"""
import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import blocks
from graph_nets import modules
import graph_nets as gn

from root_gnn.src.models.base import make_mlp
from root_gnn.src.models.base import make_mlp_model
from root_gnn.src.models.base import MLPGraphNetwork

LATENT_SIZE = 128
NUM_LAYERS = 2

class SetsGenerator(snt.Module):
    def __init__(self, name="SetsGenerator", input_dim=12, out_dim=4):
        super(SetsGenerator, self).__init__(name=name)

        # self._edge_block = blocks.EdgeBlock(
        #     edge_model_fn=make_mlp_model,
        #     use_edges=False,
        #     use_receiver_nodes=True,
        #     use_sender_nodes=True,
        #     use_globals=False,
        #     name='edge_encoder_block'
        # )
        # self._node_encoder_block = blocks.NodeBlock(
        #     node_model_fn=make_mlp_model,
        #     use_received_edges=False,
        #     use_sent_edges=False,
        #     use_nodes=True,
        #     use_globals=False,
        #     name='node_encoder_block'
        # )

        # self._global_encoder_block = blocks.GlobalBlock(
        #     global_model_fn=make_mlp_model,
        #     use_edges=True,
        #     use_nodes=True,
        #     use_globals=False,
        #     nodes_reducer=tf.math.unsorted_segment_sum,
        #     edges_reducer=tf.math.unsorted_segment_sum,
        #     name='global_encoder_block'
        # )

        # self._core = MLPGraphNetwork()

        # # Transforms the outputs into appropriate shapes.
        # node_fn =lambda: snt.Sequential([
        #     snt.nets.MLP([LATENT_SIZE],
        #                 activation=tf.nn.relu, # default is relu
        #                 name='node_output')])

        # # global_output_size = 1
        # # global_fn = lambda: snt.Sequential([
        # #     snt.nets.MLP([global_output_size],
        # #                 activation=tf.nn.relu, # default is relu
        # #                 name='global_output'),
        # #     tf.sigmoid])

        # self._output_transform = modules.GraphIndependent(
        #     edge_model_fn=None, 
        #     node_model_fn=node_fn,
        #     global_model_fn=None)


        # graph creation
        self._node_linear = snt.Sequential([snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS+[out_dim],
                        activation=tf.nn.relu,
                        activate_final=False,
                        dropout_rate=0.30,
                        name="node_encoder_nn"),
                        snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
        # self._node_reduction = snt.nets.MLP([input_dim], name='node_reduction_nn')

        # node properties
        self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name="node_rnn")
        self._node_prop_nn = snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS+[out_dim],
                activation=tf.nn.relu, activate_final=False,
                dropout_rate=0.30,
                name="node_prop_nn")

        self.out_dim = out_dim

    def __call__(self, 
        target_op: gn.GraphsTuple, 
        max_nodes: int, 
        training: bool) -> tf.Tensor:
        """
        Args: 
            input_op: 2D vector with dimensions [batch-size, features], 
            the latter containing [px, py, pz, E, N-dimension noises]
            max_nodes: maximum number of output nodes
            training: if in training mode, needed for `dropout`.

        Retruns:
            predicted node featurs with dimension of [batch-size, max-nodes, out-features] 
        """

        batch_size = input_op.nodes.shape[0]

        # predicting edges
        # TODO: concate the new embedding to the old ones and do the reduction before RNN.
        node_hidden_state = tf.zeros([batch_size, LATENT_SIZE], dtype=tf.float32, name='initial_node_hidden')
        latent = input_op.replace(nodes=self._node_linear(input_op.nodes, training))
        
        nodes = latent.nodes
        nodes = tf.reshape(nodes, [batch_size, -1, self.out_dim])
        for inode in range(max_nodes):
            # print(nodes.shape)

            # # usual business
            # latent = self._node_encoder_block(latent)
            # latent = self._edge_block(latent)
            # latent = self._global_encoder_block(latent)
            # latent = self._core(latent)
            # latent = self._output_transform(latent)

            # node properties
            node_embedding = tf.math.reduce_sum(nodes, axis=1)
            node_embedding, node_hidden_state = self._node_rnn(node_embedding, node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding, training)
            node_prop = tf.reshape(node_prop, [batch_size, -1, self.out_dim])

            # add new node to the existing nodes
            nodes = tf.concat([nodes, node_prop], axis=1, name='add_new_node')

            # update the graph by adding a new node with features as predicted
            # construct a fully-connected graph
            # n_nodes = tf.math.reduce_sum(latent.n_node)
            # n_nodes = tf.add(n_nodes, 1)
            # nodes_tobe = tf.concat([nodes, node_prop], axis=0, name='add_new_node')
            # rng = tf.range(n_nodes)
            # receivers, senders = tf.meshgrid(rng, rng)
            # n_edge = n_nodes * n_nodes
            # ind = tf.cast(1 - tf.eye(n_nodes), bool)
            # receivers = tf.boolean_mask(receivers, ind)
            # senders = tf.boolean_mask(senders, ind)
            # n_edge -= n_nodes
            # receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
            # senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
            # edges = tf.ones([1, 1], dtype=tf.float32)
            # n_edge = tf.reshape(n_edge, [1])
            # n_node = tf.reshape(n_nodes, [1])

            # latent = latent.replace(nodes=nodes_tobe, n_node=n_node,\
            #     n_edge=n_edge, edges=edges, senders=senders, receivers=receivers)
        # print("HERE", nodes[1:].shape)    
        nodes = tf.reshape(nodes[:, 1:, :], [-1, self.out_dim])
        return nodes


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