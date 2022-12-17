"""
The implementation of Graph Networks are mostly inspired by the ones in deepmind/graphs_nets
https://github.com/deepmind/graph_nets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa

from graph_nets import modules
from graph_nets import utils_tf
from graph_nets import blocks
from graph_nets import _base

from functools import partial

import sonnet as snt



class MultiMLP(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False,dropout_rate=0.05):
        super(MultiMLP, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)#,dropout_rate=dropout_rate)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)#,dropout_rate=dropout_rate)
    def __call__(self,x,is_training=True):
        return self.tower_mlp(tf.reshape(x[:,1:],[len(x), len(x[0])-1]))*tf.reshape(x[:,0],[len(x),1])+self.track_mlp(tf.reshape(x[:,1:],[len(x), len(x[0])-1]))*(1.-tf.reshape(x[:,0],[len(x),1]))

class MultiMLPKeras(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False, batch_size=500*16, feature_size=7, dropout_rate=0.05):
        super(MultiMLPKeras, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)#,dropout_rate=dropout_rate)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)#,dropout_rate=dropout_rate)
        self.batch_size = batch_size
        self.feature_size = feature_size
    def __call__(self,x,is_training=True):
        batch_size = self.batch_size
        feature_size = self.feature_size
        return self.tower_mlp(tf.reshape(x[:,1:],[batch_size, feature_size-1]))*tf.reshape(x[:,0],[batch_size,1])+ \
            self.track_mlp(tf.reshape(x[:,1:],[batch_size, feature_size-1]))*(1.-tf.reshape(x[:,0],[batch_size,1]))
    
class ConcatMLP(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False,dropout_rate=0.05):
        super(ConcatMLP, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final,dropout_rate=dropout_rate)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final,dropout_rate=dropout_rate)
    def __call__(self,x,is_training=True):
        return tf.concatenate([self.tower_mlp(tf.reshape(x[0][1:],[1, len(x[0])-1]))*x[0][0],self.track_mlp(tf.reshape(x[0][1:],[1, len(x[0])-1]))*(1.-x[0][0])])

class EdgeMLP(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False,dropout_rate=0.05):
        super(EdgeMLP, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.mix_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
    def __call__(self,x,is_training=True):
        return self.tower_mlp(tf.reshape(x[:,3:],[len(x), len(x[0])-3]))*tf.reshape(x[:,0],[len(x),1])+self.mix_mlp(tf.reshape(x[:,3:],[len(x), len(x[0])-3]))*tf.reshape(x[:,1],[len(x),1])+self.track_mlp(tf.reshape(x[:,3:],[len(x), len(x[0])-3]))*tf.reshape(x[:,2],[len(x),1])
    
class EdgeMLPKeras(snt.Module):
    def __init__(self,mlp_size,activation=tf.nn.relu,activate_final=False, batch_size=500*8*15, feature_size=5, dropout_rate=0.05):
        super(EdgeMLPKeras, self).__init__()
        self.tower_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.track_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.mix_mlp = snt.nets.MLP(mlp_size,activation=activation,activate_final=activate_final)
        self.batch_size = batch_size
        self.feature_size = feature_size

    def __call__(self,x,is_training=True):
        batch_size = self.batch_size
        feature_size = self.feature_size
        return self.tower_mlp(tf.reshape(x[:,3:],[batch_size, feature_size-3]))*tf.reshape(x[:,0],[batch_size,1])+ \
            self.mix_mlp(tf.reshape(x[:,3:],[batch_size, feature_size-3]))*tf.reshape(x[:,1],[batch_size,1])+ \
                self.track_mlp(tf.reshape(x[:,3:],[batch_size, feature_size-3]))*tf.reshape(x[:,2],[batch_size,1])

        

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

    
### Experimental Attention Modules ###

class AttentionBlock(snt.Module):
    """
    The attention block inspired by https://arxiv.org/pdf/2202.03772.pdf
    """
    def __init__(self, node_embed_dim=128, num_attn_heads=8, layer_dim=[128], ffn=None, 
                 mlp_fn=make_mlp_model, embedding_fcn=None):
        super(AttentionBlock, self).__init__(name='AttentionBlock')
        
        
        self.use_embedding = False
        if embedding_fcn is not None:
            self.embedding = embedding_fcn(mlp_size=[node_embed_dim])
            self.use_embedding = True
            
        self.pre_attn_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.attention = layers.MultiHeadAttention(num_attn_heads, node_embed_dim)
        self.post_attn_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        self.pre_fc_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.fc = mlp_fn(mlp_size=layer_dim, activations=activations.gelu)

        
        
    def __call__(self, x, q=None, attn_mask=None, is_training=True, **kwargs):
        
        if self.use_embedding:
            x = self.embedding(x)
        residual = x
        x = self.pre_attn_norm(x)
        
        x = tf.expand_dims(x, axis=0)
        if q is None:
            x = self.attention(x, x, x, attention_mask=attn_mask, return_attention_scores=False)
        else:
            x = self.attention(q, x, x, attention_mask=attn_mask, return_attention_scores=False)
        x = tf.squeeze(x, axis=0)
        
        x = self.post_attn_norm(x) + residual

        x = self.pre_fc_norm(x)
        x = self.fc(x, is_training=is_training)

        return x
    
    
class AttentionAggregator(snt.Module):
    def __init__(self, num_heads, key_dim):
        super(AttentionAggregator, self).__init__(name='AttentionAggregator')
        
        self.pre_attn_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.post_attn_norm = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        
        self.reducer = tf.math.unsorted_segment_sum
        self.global_query = tf.Variable(tf.zeros([1, 1, key_dim]), trainable=True)

        
    def __call__(self, x, segment_ids, num_segments, global_q=None, **kwargs):

        residual = x 
        x = self.pre_attn_norm(x)
        
        x = tf.expand_dims(x, axis=1)
        if global_q is None:
            global_q = self.global_query
        else:
            global_q = global_q
        x = self.attention(global_q, x, x, return_attention_scores=False) # Check
        x = tf.squeeze(x, axis=1)        
        
        x = self.post_attn_norm(x) + residual

        return self.reducer(x, segment_ids, num_segments)
    
    
class GlobalAttentionBlock(blocks.GlobalBlock):

    def __init__(self,
               global_model_fn,
               use_edges=True,
               use_nodes=True,
               use_globals=True,
               num_heads=8, key_dim=64,
               name="global_attn_block",
               **kwargs):
    
        super(GlobalAttentionBlock, self).__init__(global_model_fn,
                                                   use_edges=use_edges,
                                                   use_nodes=use_nodes,
                                                   use_globals=use_globals,
                                                   name=name)
        
        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                           "use_nodes or use_globals must be True.")

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = EdgeToGlobalAttAggregator(num_heads, key_dim)
            if self._use_nodes:
                self._nodes_aggregator = NodeToGlobalAttAggregator(num_heads, key_dim)

                    
                    
class GraphAttentionNetwork(modules.GraphNetwork):
    """
    Implementation of a Graph Network with Attention Global Block instead of general Global Block
    """
    def __init__(self,
               edge_model_fn,
               node_model_fn,
               global_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               edge_block_opt=None,
               node_block_opt=None,
               global_block_opt=None,
               num_heads=8, key_dim=64,
               name="GraphAttentionNetwork"):
    
        super(GraphAttentionNetwork, self).__init__(edge_model_fn,
                                                    node_model_fn,
                                                    global_model_fn,
                                                    reducer=reducer,
                                                    edge_block_opt=edge_block_opt,
                                                    node_block_opt=node_block_opt,
                                                    global_block_opt=global_block_opt,
                                                    name=name)
        
        global_block_opt = modules._make_default_global_block_opt(global_block_opt, reducer)

        with self._enter_variable_scope():
            self._global_block = GlobalAttentionBlock(
              global_model_fn=global_model_fn, num_heads=num_heads, key_dim=key_dim,
              **global_block_opt)
    
    
    
class GlobalAttentionAggregatorBase(snt.Module):
    def __init__(self, num_heads, key_dim, name="GlobalAttentionAggregatorBase"):
        super(GlobalAttentionAggregatorBase, self).__init__(name=name)
        self._normalizer = modules._unsorted_segment_softmax
        #self.mha = layers.MultiHeadAttention(num_heads, key_dim)
        #self._setup = False

        self.k_layer = make_mlp_model([key_dim], dropout_rate=0.)
        self.q_layer = make_mlp_model([key_dim], dropout_rate=0.)
        self.v_layer = make_mlp_model([key_dim], dropout_rate=0.)
    
    def _setup_MHA(self, k, q, v):
        if self._setup:
            return
        #self.mha._build_from_signature(q, v, k)
        #self.q_layer = self.mha._query_dense
        #self.k_layer = self.mha._key_dense
        #self.v_layer = self.mha._value_dense
        self._setup = True

        
class NodeToGlobalAttAggregator(GlobalAttentionAggregatorBase):
    def __init__(self, num_heads, key_dim, name="NodeToGlobalAttAggregator"):
        super(NodeToGlobalAttAggregator, self).__init__(num_heads, key_dim, name)
        
    def __call__(self, graph, **kwargs):
        k = graph.nodes
        v = graph.nodes
        q = graph.globals
        #self._setup_MHA(k, q, v)

        #print(f">>> k, {k}")                               
        node_keys = self.k_layer(k)
        node_values = self.v_layer(v)
        global_queries = self.q_layer(q)
        
        queries = blocks.broadcast_globals_to_nodes(
            graph.replace(globals=global_queries))

        attention_weights_logits = tf.reduce_sum(
            node_keys * queries, axis=-1)
        normalized_attention_weights = blocks.NodesToGlobalsAggregator(
            reducer=self._normalizer)(graph.replace(nodes=attention_weights_logits))

        attented_nodes = node_values * normalized_attention_weights[..., None]
        #attention_output = self.mha._output_dense(attented_nodes)
        attention_output = attented_nodes

        node_to_global_aggregator = blocks.NodesToGlobalsAggregator(
            reducer=tf.math.unsorted_segment_sum)
        aggregated_attended_values = node_to_global_aggregator(
            graph.replace(nodes=attention_output))
        #print(">>> k after,", aggregated_attended_values)
        return aggregated_attended_values
    
    
class EdgeToGlobalAttAggregator(GlobalAttentionAggregatorBase):
    def __init__(self, num_heads, key_dim, name="EdgeToGlobalAttAggregator"):
        super(EdgeToGlobalAttAggregator, self).__init__(num_heads, key_dim, name)
        
    def __call__(self, graph, **kwargs):
        k = v = graph.edges
        q = graph.globals
        #self._setup_MHA(k, q, v)
        
        edge_keys = self.k_layer(k)
        edge_values = self.v_layer(v)
        global_queries = self.q_layer(q)
        
        queries = blocks.broadcast_globals_to_edges(
            graph.replace(globals=global_queries))

        attention_weights_logits = tf.reduce_sum(
            edge_keys * queries, axis=-1)
        normalized_attention_weights = blocks.EdgesToGlobalsAggregator(
            reducer=self._normalizer)(graph.replace(edges=attention_weights_logits))

        attented_edges = edge_values * normalized_attention_weights[..., None]
        #attention_output = self.mha._output_dense(attented_edges)
        attention_output = attented_edges
        
        edge_to_global_aggregator = blocks.EdgesToGlobalsAggregator(
            reducer=tf.math.unsorted_segment_sum)
        aggregated_attended_values = edge_to_global_aggregator(
            graph.replace(edges=attention_output))

        return aggregated_attended_values



class GlobalLSTMBlock(blocks.GlobalBlock):
    def __init__(self,
               global_model_fn,
               node_embedding_dim=32,
               edge_embedding_dim=64,
               lstm_unit=32,
               use_edges=True,
               use_nodes=True,
               use_globals=True,
               name="global_lstm_block",
               **kwargs):
    
        super(GlobalLSTMBlock, self).__init__(global_model_fn,
                                                   use_edges=use_edges,
                                                   use_nodes=use_nodes,
                                                   use_globals=use_globals,
                                                   name=name)
        
        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                           "use_nodes or use_globals must be True.")

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = EdgeToGlobalLSTMAggregator(lstm_unit, edge_embedding_dim)
            if self._use_nodes:
                self._nodes_aggregator = NodeToGlobalLSTMAggregator(lstm_unit, node_embedding_dim)



class NodeToGlobalLSTMAggregator(snt.Module):
    def __init__(self, lstm_unit=32, node_embedding_dim=32, 
                 equal_length=True, name="NodeToGlobalLSTMAggregator"):
        super(NodeToGlobalLSTMAggregator, self).__init__(name)
        self.LSTM = layers.LSTM(lstm_unit, unroll=True, go_backwards=False)
        self.num_nodes = None
        self.batch_size = None
        self.embedding_dim = node_embedding_dim
        self.equal_length = equal_length

    def __call__(self, graph, **kwargs):
        nodes = graph.nodes
        #print("nodes", nodes)
        if self.num_nodes is None or self.batch_size is None or not self.equal_length:
            self.batch_size = utils_tf.get_num_graphs(graph)
            #self.num_nodes = blocks._get_static_num_nodes(graph) // self.batch_size
            self.num_nodes = 16
            
        input_nodes = tf.reshape(nodes, [self.batch_size, self.num_nodes, self.embedding_dim])
        #print("nodes", input_nodes)
        aggregated_values = self.LSTM(input_nodes)
        #print("nodes", aggregated_values)
        return aggregated_values


class EdgeToGlobalLSTMAggregator(snt.Module):
    def __init__(self, lstm_unit=32, edge_embedding_dim=64,
                 equal_length=True, name="EdgeToGlobalLSTMAggregator"):
        super(EdgeToGlobalLSTMAggregator, self).__init__(name)
        self.LSTM = layers.LSTM(lstm_unit, unroll=True, go_backwards=False)
        self.num_edges = None
        self.batch_size = None
        self.embedding_dim = edge_embedding_dim
        self.equal_length = equal_length

    def __call__(self, graph, **kwargs):
        edges = graph.edges
        #print("edges", edges)
        if self.num_edges is None or self.batch_size is None or not self.equal_length:
            self.batch_size = utils_tf.get_num_graphs(graph)            
            self.num_edges = 16 * 15 // 2
            
        input_edges = tf.reshape(edges, [self.batch_size, self.num_edges, self.embedding_dim])
        #print("edges", edges)
        aggregated_values = self.LSTM(input_edges)
        #print("edges", aggregated_values)
        return aggregated_values

    

### LSTM Encoder Functions ###
    
class rnn_mlp_snt(snt.Module):
    def __init__(self, input_shape, dense_shape, lstm_1, lstm_2,
                       name='rnn_mlp', activation=tf.nn.relu, **kwargs):
        super(rnn_mlp_snt, self).__init__(name)
        self.mlp = snt.nets.MLP(dense_shape, activation=activation)
        self.LSTM1 = snt.UnrolledLSTM(lstm_1)
        self.LSTM2 = snt.UnrolledLSTM(lstm_2)
        self.lstm_1_init = self.LSTM1.initial_state(500) # batch_size = 500
        self.lstm_2_init = self.LSTM2.initial_state(500)
        #print(tf.shape(self.lstm_1_init))
        #exit()
    
    def __call__(self, input_seq):
        mlp_out = self.mlp(input_seq)
        lstm_out = self.LSTM1(mlp_out, self.lstm_1_init)[0] # 0 is the seq, 1 is final state, final state is (hidden, cell)
        return self.LSTM2(lstm_out, self.lstm_2_init)[1][0]

    
    
def make_rnn_mlp(input_shape,
    dense_unit_1, dense_unit_2, lstm_1, lstm_2,
    backwards=False, unroll=True, mask_value=0.0, **kwargs):

    return keras.Sequential([
      layers.Input(shape=input_shape),
      layers.Masking(mask_value=mask_value),
      layers.TimeDistributed(layers.Dense(dense_unit_1)),
      layers.TimeDistributed(layers.Dense(dense_unit_2)),
      layers.LSTM(lstm_1, unroll=unroll, go_backwards=backwards, return_sequences=True),
      layers.LSTM(lstm_2, unroll=unroll, go_backwards=backwards)])

def keras_two_layer(input_shape, dense_unit_1, dense_unit_2, mask_value=0.0,):
    return keras.Sequential([
      layers.Input(shape=input_shape),
      layers.Masking(mask_value=mask_value),
      layers.TimeDistributed(layers.Dense(dense_unit_1)),
      layers.TimeDistributed(layers.Dense(dense_unit_2))])



### Attention Encoder Blocks ###

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(Layer):
    def __init__(self, length, d_model):
        super().__init__()
        self.d_model = d_model
        self.length = length
        #self.embedding = Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        #length = tf.shape(x)[1]
        #x = self.embedding(x)
        length = self.length
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class AttentionEncoder(snt.Module):
    def __init__(self, input_shape, dense_shape, attn_1, attn_2, 
                 position_encoding=False, name='attn_mlp', 
                 activation=tf.nn.relu, **kwargs):
        super(rnn_mlp_snt, self).__init__(name)
        
        if position_encoding:
            self.embedding = PositionalEmbedding(*input_shape)
        else:
            self.embedding = None
        self.mlp = snt.nets.MLP(dense_shape, activation=activation)
        self.self_attn = layers.MultiHeadAttention(8, attn_1)
        self.glb_attn = layers.MultiHeadAttention(8, attn_2)
    
    
    def __call__(self, input_seq, glb_embedding):
        
        if self.embedding is not None:
            input_seq = self.embedding(input_seq)
        mlp_out = self.mlp(input_seq)
        attn_out = self.self_attn(mlp_out, mlp_out)
        return tf.reduce_sum(self.glb_attn(query=glb_embedding, value=attn_out, key=attn_out), axis=1)
    

    
    