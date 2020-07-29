"""
The implementation of Graph Networks are mostly inspired by the ones in deepmind/graphs_nets
https://github.com/deepmind/graph_nets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import blocks
import sonnet as snt

NUM_LAYERS = 2    # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 128  # Hard-code latent layer sizes for demos.
# DROPOUT_RATE = 0.2

def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  # the activation function choices:
  # swish, relu, relu6, leaky_relu
  return snt.Sequential([
      snt.nets.MLP([128, 64]*NUM_LAYERS,
                    activation=tf.nn.relu,
                    activate_final=True, 
                #    dropout_rate=DROPOUT_RATE
        ),
      snt.LayerNorm(axis=-1, create_scale=True, create_offset=False)
  ])

class MLPGraphIndependent(snt.Module):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = modules.GraphIndependent(
        edge_model_fn=make_mlp_model,
        node_model_fn=make_mlp_model,
        global_model_fn=make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""
    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._network = modules.GraphNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model
            )

    def __call__(self, inputs):
        return self._network(inputs)


class GeneralClassifier(snt.Module):

    def __init__(self, name="GeneralClassifier"):
        super(GeneralClassifier, self).__init__(name=name)

        self._encoder = MLPGraphIndependent()
        self._core    = MLPGraphNetwork()
        self._decoder = MLPGraphIndependent()

        # Transforms the outputs into appropriate shapes.
        global_output_size = 1
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, global_output_size],
                         name='global_output'), tf.sigmoid])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

        decoded_op = self._decoder(latent)
        output_ops.append(self._output_transform(decoded_op))
        return output_ops


class InteractionNetwork(snt.Module):
  """Implementation of an Interaction Network, similarly to
  https://arxiv.org/abs/1612.00222, except that it does not require inputput 
  edge features.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               name="interaction_network"):
    super(InteractionNetwork, self).__init__(name=name)
    self._edge_block = blocks.EdgeBlock(
        edge_model_fn=edge_model_fn, use_globals=False)
    self._node_block = blocks.NodeBlock(
        node_model_fn=node_model_fn,
        use_received_edges=True,
        use_sent_edges=True,
        use_globals=False,
        received_edges_reducer=reducer)

  def __call__(self, graph):
    return self._edge_block(self._node_block(graph))


class GlobalClassifierNoEdgeInfo(snt.Module):

    def __init__(self, name="GlobalClassifierNoEdgeInfo"):
        super(GlobalClassifierNoEdgeInfo, self).__init__(name=name)

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

class EdgeGlobalClassifier(snt.Module):
    def __init__(self, name="EdgeGlobalClassifier"):
        super(EdgeGlobalClassifier, self).__init__(name=name)

        self._encoder = MLPGraphIndependent()
        self._core    = MLPGraphNetwork()
        self._decoder = MLPGraphIndependent()

        # Transforms the outputs into appropriate shapes
        # assuming the target of the outputs of
        # global and edge is binary.
        global_output_size = 1
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, global_output_size],
                         name='global_output'), tf.sigmoid])
        edge_output_size = 1
        edge_fn = lambda: snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, edge_output_size],
                         name='edge_output'), tf.sigmoid])

        self._output_transform = modules.GraphIndependent(edge_fn, None, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))

        return output_ops

class MultiClassifier(snt.Module):
    def __init__(self, name="MultiClassifier"):
        super(MultiClassifier, self).__init__(name=name)

        self._encoder = MLPGraphIndependent()
        self._core    = MLPGraphNetwork()
        self._decoder = MLPGraphIndependent()

        # Transforms the outputs into appropriate shapes.
        global_output_size = 3
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, global_output_size],
                         name='global_output')])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

        decoded_op = self._decoder(latent)
        output_ops.append(self._output_transform(decoded_op))
        return output_ops


class NodeEdgeClassifier(snt.Module):
    def __init__(self, name="NodeEdgeClassifier"):
        super(NodeEdgeClassifier, self).__init__(name=name)


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
        edge_output_size = 1
        node_output_size = 1
        edge_fn =lambda: snt.Sequential([
            snt.nets.MLP([edge_output_size],
                        activation=tf.nn.relu, # default is relu
                        name='edge_output'),
            tf.sigmoid])

        node_fn =lambda: snt.Sequential([
            snt.nets.MLP([node_output_size],
                        activation=tf.nn.relu, # default is relu
                        name='node_output'),
            tf.sigmoid])

        self._output_transform = modules.GraphIndependent(edge_fn, node_fn, None)


    def __call__(self, input_op, num_processing_steps):
        latent = self._edge_block(self._node_encoder_block(input_op))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

            output_ops.append(self._output_transform(latent))

        return output_ops

class EdgeClassifier(snt.Module):
    def __init__(self, name="EdgeClassifier"):
        super(EdgeClassifier, self).__init__(name=name)


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
        edge_output_size = 1
        edge_fn =lambda: snt.Sequential([
            snt.nets.MLP([edge_output_size],
                        activation=tf.nn.relu, # default is relu
                        name='edge_output'),
            tf.sigmoid])

        self._output_transform = modules.GraphIndependent(edge_fn, None, None)


    def __call__(self, input_op, num_processing_steps):
        latent = self._edge_block(self._node_encoder_block(input_op))
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

            output_ops.append(self._output_transform(latent))

        return output_ops