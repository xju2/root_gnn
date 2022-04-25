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

class MultiMLP(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False,dropout_rate=0.05):
        super(MultiMLP, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)#,dropout_rate=dropout_rate)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)#,dropout_rate=dropout_rate)
    def __call__(self,x,is_training=True):
        return self.tower_mlp(tf.reshape(x[:,1:],[len(x), len(x[0])-1]))*tf.reshape(x[:,0],[len(x),1])+self.track_mlp(tf.reshape(x[:,1:],[len(x), len(x[0])-1]))*(1.-tf.reshape(x[:,0],[len(x),1]))
    
class ConcatMLP(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False,dropout_rate=0.05):
        super(ConcatMLP, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final,dropout_rate=dropout_rate)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final,dropout_rate=dropout_rate)
    def __call__(self,x,is_training=True):
        return tf.concatenate(self.tower_mlp(tf.reshape(x[0][1:],[1, len(x[0])-1]))*x[0][0],self.track_mlp(tf.reshape(x[0][1:],[1, len(x[0])-1]))*(1.-x[0][0]))

class EdgeMLP(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False,dropout_rate=0.05):
        super(EdgeMLP, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.mix_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
    def __call__(self,x,is_training=True):
        return \
    self.tower_mlp(tf.reshape(x[:,3:],[len(x), len(x[0])-3]))*tf.reshape(x[:,0],[len(x),1])+\
    self.mix_mlp(tf.reshape(x[:,3:],[len(x), len(x[0])-3]))*tf.reshape(x[:,1],[len(x),1])+\
    self.track_mlp(tf.reshape(x[:,3:],[len(x), len(x[0])-3]))*tf.reshape(x[:,2],[len(x),1])

def make_multi_mlp_model(
    mlp_size: list = [128]*2,
    dropout_rate: float = 0.05,
    activations=tf.nn.relu,
    activate_final: bool =True,
    name: str = 'MLP', *args, **kwargs):
    create_scale = True if not "create_scale" in kwargs else kwargs['create_scale']
    create_offset = True if not "create_offset" in kwargs else kwargs['create_offset']
    return snt.Sequential([
      MultiMLP(mlp_size,
                  activation=activations,
                  activate_final=activate_final,
                  dropout_rate=dropout_rate
        ),
      snt.LayerNorm(axis=-1, create_scale=create_scale, create_offset=create_offset)
  ], name=name)

def make_heterogeneous_edges_model(
    mlp_size: list = [128]*2,
    dropout_rate: float = 0.05,
    activations=tf.nn.relu,
    activate_final: bool =True,
    name: str = 'MLP', *args, **kwargs):
    create_scale = True if not "create_scale" in kwargs else kwargs['create_scale']
    create_offset = True if not "create_offset" in kwargs else kwargs['create_offset']
    return snt.Sequential([
      EdgeMLP(mlp_size,
                  activation=activations,
                  activate_final=activate_final,
                  dropout_rate=dropout_rate
        ),
      snt.LayerNorm(axis=-1, create_scale=create_scale, create_offset=create_offset)
  ], name=name)

def make_concat_mlp_model(
    mlp_size: list = [128]*2,
    dropout_rate: float = 0.05,
    activations=tf.nn.relu,
    activate_final: bool =True,
    name: str = 'MLP', *args, **kwargs):
    create_scale = True if not "create_scale" in kwargs else kwargs['create_scale']
    create_offset = True if not "create_offset" in kwargs else kwargs['create_offset']
    return snt.Sequential([
      ConcatMLP(mlp_size,
                  activation=activations,
                  activate_final=activate_final,
                  dropout_rate=dropout_rate
        ),
      snt.LayerNorm(axis=-1, create_scale=create_scale, create_offset=create_offset)
  ], name=name)

def make_mlp_model(
    mlp_size: list = [128]*2,
    dropout_rate: float = 0.05,
    activations=tf.nn.relu,
    activate_final: bool =True,
    name: str = 'MLP', *args, **kwargs):
    create_scale = True if not "create_scale" in kwargs else kwargs['create_scale']
    create_offset = True if not "create_offset" in kwargs else kwargs['create_offset']
    return snt.Sequential([
      snt.nets.MLP(mlp_size,
                  activation=activations,
                  activate_final=activate_final,
                  dropout_rate=dropout_rate
        ),
      snt.LayerNorm(axis=-1, create_scale=create_scale, create_offset=create_offset)
  ], name=name)

class MLPGraphIndependent(snt.Module):
  """GraphIndependent with same Neural Network type---such as MLPs---for edge, node, and global models."""

  def __init__(self, nn_fn, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = modules.GraphIndependent(
        edge_model_fn=nn_fn,
        node_model_fn=nn_fn,
        global_model_fn=nn_fn)

    def __call__(self, inputs,
            edge_model_kwargs=None,
            node_model_kwargs=None,
            global_model_kwargs=None):
        return self._network(
      inputs,
      edge_model_kwargs=edge_model_kwargs,
      node_model_kwargs=node_model_kwargs,
      global_model_kwargs=global_model_kwargs
      )


class MLPGraphNetwork(snt.Module):
    """GraphIndependent with same Neural Network type---such as MLPs---for edge, node, and global models."""
    def __init__(self, nn_fn, reducer=tf.math.unsorted_segment_sum, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._network = modules.GraphNetwork(
            edge_model_fn=nn_fn,
            node_model_fn=nn_fn,
            global_model_fn=nn_fn,reducer=reducer)

    def __call__(self, inputs,
            edge_model_kwargs=None,
            node_model_kwargs=None,
            global_model_kwargs=None):
        return self._network(inputs,
                      edge_model_kwargs=edge_model_kwargs,
                      node_model_kwargs=node_model_kwargs,
                      global_model_kwargs=global_model_kwargs)



class InteractionNetwork(snt.Module):
  """Implementation of an Interaction Network, similarly to
  https://arxiv.org/abs/1612.00222, except that it does not require input 
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
    def __call__(self,
    graph,
    edge_model_kwargs=None,
    node_model_kwargs=None
  ):
        return self._edge_block(self._node_block(graph, node_model_kwargs), edge_model_kwargs)
