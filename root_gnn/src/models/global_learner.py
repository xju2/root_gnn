import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow_addons as tfa
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from root_gnn.src.models.base import MLPGraphNetwork
from root_gnn.src.models.base import make_mlp_model, make_multi_mlp_model, make_concat_mlp_model, make_heterogeneous_edges_model, AttentionBlock, AttentionAggregator, GraphAttentionNetwork
from functools import partial


class GlobalLearnerBase(snt.Module):
    def __init__(self,
        global_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="GlobalLearnerBase",
        reducer=tf.math.unsorted_segment_sum,
        encoder_fn = make_mlp_model,node_encoder_fn = make_mlp_model,edge_encoder_fn = make_mlp_model,
        global_in_nodes=None, global_in_edges=None,**kwargs):
        super(GlobalLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(encoder_fn, mlp_size=encoder_size, name="EncoderMLP", **kwargs)
            node_encoder_mlp_fn = partial(node_encoder_fn, mlp_size=encoder_size, name="NodeEncoderMLP", **kwargs)
            edge_encoder_mlp_fn = partial(edge_encoder_fn, mlp_size=encoder_size, name="EdgeEncoderMLP", **kwargs)
        else:
            encoder_mlp_fn = partial(encoder_fn, name='EncoderMLP', **kwargs)
            node_encoder_mlp_fn = partial(node_encoder_fn, name='NodeEncoderMLP', **kwargs)
            edge_encoder_mlp_fn = partial(edge_encoder_fn, name='EdgeEncoderMLP', **kwargs)

        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False,received_edges_reducer=reducer,sent_edges_reducer=reducer)
        edge_block_args=dict(use_edges=True, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
        global_block_args=dict(use_edges=True, use_nodes=True, use_globals=False,nodes_reducer=reducer,edges_reducer=reducer)
        #print("Global Inputs?:",with_global_inputs)

        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = False
            node_block_args['use_sent_edges'] = False
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False
        if with_global_inputs:
            global_block_args['use_globals'] = True
            #global_block_args['use_nodes'] = False
            #global_block_args['use_edges'] = False
            node_block_args['use_globals'] = True
            edge_block_args['use_globals'] = True
            
        if global_in_nodes is not None:
            node_block_args['use_globals'] = global_in_nodes
        if global_in_edges is not None:
            edge_block_args['use_globals'] = global_in_edges
            
        print(f'>>> Node Block Options: {node_block_args}')
        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=edge_encoder_mlp_fn,
            name='edge_encoder_block', **edge_block_args)

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=node_encoder_mlp_fn,
            name='node_encoder_block', **node_block_args)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='global_encoder_block', **global_block_args)

        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)

        self._core = MLPGraphNetwork(nn_fn=core_mlp_fn, reducer=reducer)

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)
        
        node_encoding = self._node_encoder_block(input_op, node_kwargs)
        edge_encoding = self._edge_encoder_block(node_encoding, edge_kwargs)


        latent = self._global_block(edge_encoding, global_kwargs)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops


class GlobalClassifier(GlobalLearnerBase):
    def __init__(self,
        with_edge_inputs=True, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        name="GlobalClassifier", **kwargs):

        global_output_size = 1
        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]

        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='edge_classifier_output'),
            tf.sigmoid])

        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size,
            name=name, **kwargs)


class GlobalRegression(GlobalLearnerBase):
    def __init__(self, global_output_size,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        name="GlobalRegression", **kwargs):

        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='global_regresssion_output')])

        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size, name=name, **kwargs)


class GlobalSetClassifier(snt.Module):
    def __init__(self,with_global_inputs=False,core_size: list=None,decoder_size: list=None,name="GlobalSetClassifier",**kwargs):
        super(GlobalSetClassifier, self).__init__(name=name)

        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=with_global_inputs)
        global_block_args=dict(use_edges=False, use_nodes=True, use_globals=with_global_inputs)

        if core_size is not None:
            node_mlp_fn = partial(make_mlp_model, dropout_rate=None,mlp_size=core_size, **kwargs)
        else:
            node_mlp_fn = partial(make_mlp_model, dropout_rate=None,**kwargs)

        global_size = core_size + [1]

        global_fn = lambda: snt.Sequential([snt.nets.MLP(global_size,activation=tf.nn.relu, name='set_classifier_output'),tf.sigmoid])

        self.deep_set = modules.DeepSets(node_mlp_fn,global_fn)


    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = global_kwargs = dict(is_training=is_training)
        output_op = self.deep_set(input_op)
        return [output_op]


class GlobalAttentionRegression(snt.Module):
    """Attention model with interaction GNN"""
    def __init__(self, global_output_size=1,
        with_node_inputs=True, with_edge_inputs=False, with_global_inputs=False, 
        encoder_size: list=None, core_size: list=None, 
        decoder_size: list=None, activation_func=tf.nn.relu, 
        name="GlobalAttentionRegression",**kwargs):
        
        super(GlobalAttentionRegression, self).__init__(name=name)

        self.attention = modules.SelfAttention()
        
        self.global_regression = GlobalRegression(
            global_output_size=global_output_size,
            with_global_inputs=with_global_inputs,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            encoder_size=encoder_size, 
            core_size=core_size, 
            decoder_size=decoder_size,
            activation_func=tf.nn.relu, **kwargs)
        
        self.k_layer = snt.nets.MLP(core_size)
        self.q_layer = snt.nets.MLP(core_size)
        self.v_layer = snt.nets.MLP(core_size)
    
    def __call__(self, input_graph, num_processing_steps, is_training=True):
        node = input_graph.nodes
        k = self.k_layer(node)
        q = self.q_layer(node)
        v = self.v_layer(node)
        
        # shape of V, K, Q: [total_num_nodes, num_heads, key_size]
        weighted_input = self.attention(v, k, q, input_graph)  # Apply the Attention Mechanism
        output_graph = self.global_regression(
            weighted_input, num_processing_steps, is_training=is_training)
        
        return output_graph
    

class GlobalAttentionClassifier(snt.Module):
    def __init__(self, activation_func=tf.nn.relu,
                 with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
                 encoder_size: list=None, core_size: list=None, decoder_size: list=None,
                 node_input_shape=(10, 5), node_encoder_size=[64, 64], dense_units=32, 
                 lstm_units=32, edge_input_shape=(256), edge_encoder_size=[64, 64, 64, 8],
                 num_attn_blocks=2, num_attn_heads=8, attn_layer_dim=[128], attn_fnn_ratio=4,
                 name="GlobalAttentionClassifier", **kwargs):
        
        super(GlobalAttentionClassifier, self).__init__(name=name)
             
        ###### Encoder ######
        
        ## Encoder Blocks Args ##
        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False)
        edge_block_args=dict(use_edges=False, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
        global_block_args=dict(use_edges=True, use_nodes=True, use_globals=False)

        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = True
            node_block_args['use_sent_edges'] = True
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False
        if with_global_inputs:
            global_block_args['use_globals']=True
            
            
        ## General Encoder MLP ##
        if encoder_size is not None:
            encoder_mlp_fn = partial(make_mlp_model, mlp_size=encoder_size, name="EncoderMLP", **kwargs)
        else:
            encoder_mlp_fn = partial(make_mlp_model, name='EncoderMLP', **kwargs)
            
            
        ## Edge Encoder ##
        self.edge_encoder_mlp = self._make_edge_encoder(edge_input_shape=edge_input_shape, 
                                                   encoder_size=edge_encoder_size)
        
        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=encoder_mlp_fn,
            name='edge_encoder_block', **edge_block_args)

        
        ## Node Encoder ##
        
        node_enocder_mlp = self._make_node_encoder(node_input_shape=node_input_shape,
                                                   encoder_size=node_encoder_size, 
                                                   dense_units=dense_units, 
                                                   lstm_units=lstm_units)
        
        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=encoder_mlp_fn,
            name='node_encoder_block', **node_block_args)

        
        ## Global Encoder ##          
        self._global_block = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='global_encoder_block', **global_block_args)
        
        
        ###### Core ######
        
        ## Node Core ##
        node_embed_dim = node_encoder_size[-1] if len(node_encoder_size) > 1 else lstm_units
        node_fn = lambda: snt.Sequential([AttentionBlock(node_embed_dim, num_attn_heads, 
                                                 attn_layer_dim, attn_fnn_ratio)
                                  for _ in range(num_attn_blocks)])
        
        ## Edge and Global Core ##
        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)
        
        self._core = modules.GraphNetwork(edge_model_fn=core_mlp_fn,
                                          node_model_fn=node_fn,
                                          global_model_fn=core_mlp_fn)
        
        ###### Decoder ######
        
        edge_output_size = 1
        if decoder_size is not None:
            decoder_size += [edge_output_size]
        else:
            decoder_size = [edge_output_size]

        global_fn = lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='edge_classifier_output'),
            tf.sigmoid])
        
        self._output_transform = modules.GraphIndependent(None, None, global_fn)
        
        
    def _make_node_encoder(self, node_input_shape, encoder_size=[64, 64], dense_units=32, lstm_units=32):
        def _call(x, **kwargs):
            
            embedding = layers.Dense(encoder_size[0], name="nodeEmbedding")
            shared_dense_1_1 = layers.TimeDistributed(layers.Dense(dense_units, activation="relu"))
            shared_dense_1_2 = layers.TimeDistributed(layers.Dense(dense_units, activation="relu"))
            lstm_1_1 = layers.LSTM(lstm_units, go_backwards=False, activation="relu", return_sequences=True)
            lstm_1_2 = layers.LSTM(lstm_units, go_backwards=False, activation="relu")
            if len(encoder_size) < 1:
                return tf.keras.Sequential([embedding, shared_dense_1_1, shared_dense_1_2, lstm_1_1, lstm_1_2])
            second_embedding = layers.Dense(encoder_size[-1], name="2ndNodeEmbedding")
            result = tf.keras.Sequential([embedding, shared_dense_1_1, shared_dense_1_2, second_embedding])(tf.expand_dims(x, axis=0))
            return result

        return lambda: _call
    
    
    def _make_edge_encoder(self, edge_input_shape=(256), encoder_size=[64, 64, 64, 8]):
        module_list = []
        for dim in encoder_size:
            module_list.extend([layers.Conv1D(dim, 1), layers.BatchNormalization(), tfa.layers.GELU()])
        ops = tf.keras.Sequential(module_list)

        return lambda x: tf.squeeze(ops(tf.expand_dims(x, axis=0)), axis=0)
    
    
    def __call__(self, input_op, num_processing_steps, is_training=True):
        
        edge_kwargs = node_kwargs = global_kwargs = dict(is_training=is_training)
        
        latent = self._global_block(
            self._edge_encoder_block(self._node_encoder_block(input_op, node_kwargs),
                edge_kwargs), global_kwargs)
        latent0 = latent
        
        edge_kwargs = dict(is_training=is_training)
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            edge_feature = latent.edges
            #attn_mask = self.edge_encoder_mlp(edge_feature)
            #node_kwargs = dict(is_training=is_training, attn_mask=edge_feature)
            
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops
        

    

    def __init__(self,
        global_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="GlobalLearnerBaseEdgesFirst",
        reducer=tf.math.unsorted_segment_sum,
        encoder_fn = make_mlp_model, **kwargs):
        super(GlobalLearnerBaseEdgesFirst, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(encoder_fn, mlp_size=encoder_size, name="EncoderMLP", **kwargs)
        else:
            encoder_mlp_fn = partial(ecoder_fn, name='EncoderMLP', **kwargs)

        node_block_args=dict(use_received_edges=True, use_sent_edges=True, use_nodes=True, use_globals=False,received_edges_reducer=reducer,sent_edges_reducer=reducer)
        edge_block_args=dict(use_edges=False, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
        global_block_args=dict(use_edges=True, use_nodes=True, use_globals=False,nodes_reducer=reducer,edges_reducer=reducer)
        print("Global Inputs?:",with_global_inputs)

        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = False
            node_block_args['use_sent_edges'] = False
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False
        if with_global_inputs:
            global_block_args['use_globals'] = True
            global_block_args['use_nodes'] = False
            global_block_args['use_edges'] = False

        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=encoder_mlp_fn,
            name='edge_encoder_block', **edge_block_args)

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=encoder_mlp_fn,
            name='node_encoder_block', **node_block_args)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='global_encoder_block', **global_block_args)

        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)

        self._core = MLPGraphNetwork(nn_fn=core_mlp_fn, reducer=reducer)

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)

        latent = self._global_block(
            self._node_encoder_block(self._edge_encoder_block(input_op, node_kwargs), edge_kwargs), global_kwargs)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops

        
class HeterogeneousLearnerBase(snt.Module):
    def __init__(self,
        global_fn,
        with_edge_inputs=True,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="HeterogeneousLearnerBase",
        reducer=tf.math.unsorted_segment_sum,
        activation=tf.nn.relu,
        node_encoder_fn = make_mlp_model,
        edge_encoder_fn = make_mlp_model,
        global_encoder_fn = make_mlp_model,
        global_in_nodes=None, global_in_edges=None,
        **kwargs):
        super(HeterogeneousLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            node_encoder_mlp_fn = partial(node_encoder_fn, mlp_size=encoder_size, activations=activation, name="EncoderMLP")
            edge_encoder_mlp_fn = partial(edge_encoder_fn, mlp_size=encoder_size, activations=activation, name="EncoderMLP")
            global_encoder_mlp_fn = partial(node_encoder_fn, mlp_size=encoder_size, activations=activation, name="EncoderMLP")
        else:
            node_encoder_mlp_fn = partial(node_encoder_fn, activations=activation, name='EncoderMLP')
            edge_encoder_mlp_fn = partial(edge_encoder_fn, activations=activation, name='EncoderMLP')
            global_encoder_mlp_fn = partial(global_encoder_fn, activations=activation, name='EncoderMLP')

        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False,received_edges_reducer=reducer,sent_edges_reducer=reducer)
        edge_block_args=dict(use_edges=True, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
        global_block_args=dict(use_edges=True, use_nodes=True, use_globals=False,nodes_reducer=reducer,edges_reducer=reducer)

        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = False
            node_block_args['use_sent_edges'] = False
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False
        if with_global_inputs:
            global_block_args['use_globals'] = True
            #global_block_args['use_nodes'] = False 
            #global_block_args['use_edges'] = False
            node_block_args['use_globals'] = True
            edge_block_args['use_globals'] = True
        
        if global_in_nodes is not None:
            node_block_args['use_globals'] = global_in_nodes
        if global_in_edges is not None:
            edge_block_args['use_globals'] = global_in_edges
        
        print(f'>>> Node Block Options: {node_block_args}')
        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=edge_encoder_mlp_fn,
            name='edge_encoder_block', **edge_block_args)

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=node_encoder_mlp_fn,
            name='node_encoder_block', **node_block_args)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=global_encoder_mlp_fn, name='global_encoder_block', **global_block_args)

        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, activations=activation, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, activations=activation, **kwargs)

        self._core = MLPGraphNetwork(nn_fn=core_mlp_fn, reducer=reducer)

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)
        
        node_encoding = self._node_encoder_block(input_op, node_kwargs)
        edge_encoding = self._edge_encoder_block(node_encoding, edge_kwargs)
        latent = self._global_block(edge_encoding, global_kwargs)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops
    

class GlobalClassifierHeterogeneousNodes(HeterogeneousLearnerBase):
    def __init__(self,
        with_edge_inputs=True, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,node_encoder_fn=make_multi_mlp_model, activation=tf.nn.relu,
        name="GlobalClassifierHeterogeneousNodes", **kwargs):

        global_output_size = 1
        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]

        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=activation, # default is relu
                        name='global_classifier_output'),
            tf.sigmoid])

        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size,
            name=name, node_encoder_fn = node_encoder_fn, activation=activation, 
            **kwargs)
        

class GlobalClassifierHeterogeneousEdges(HeterogeneousLearnerBase):
    def __init__(self,
        with_edge_inputs=True, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,node_encoder_fn=make_multi_mlp_model, activation=tf.nn.relu,
        name="GlobalClassifierHeterogeneousEdges", **kwargs):

        global_output_size = 1
        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]

        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=activation, # default is relu
                        name='global_classifier_output'),
            tf.sigmoid])

        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size,
            name=name,node_encoder_fn = node_encoder_fn, edge_encoder_fn = make_heterogeneous_edges_model, activation=activation, **kwargs)

        
class GlobalHeterogeneousAttentionClassifier(GlobalClassifierHeterogeneousEdges):
    def __init__(self,
        with_edge_inputs=True, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        node_encoder_fn=make_multi_mlp_model, activation=tf.nn.relu,
        num_attn_blocks=2, num_attn_heads=8, attn_fnn_ratio=4,
        use_node_attn=True, use_global_attn=True,
        name="GlobalHeterogeneousAttentionClassifier", **kwargs):


        super().__init__(name=name,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size,
            decoder_size=decoder_size,
            node_encoder_fn=node_encoder_fn, 
            activation=activation, **kwargs)
        
        ### Core ###
        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, activations=activation, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, activations=activation, **kwargs)
            
        embed_dim = encoder_size[-1]
        attn_layer_dim = [core_size[-1]]
        if use_node_attn:
            node_core_fn = lambda: snt.Sequential([AttentionBlock(embed_dim, num_attn_heads, 
                                              attn_layer_dim, attn_fnn_ratio)
                                              for _ in range(num_attn_blocks)])
        else:
            node_core_fn = core_mlp_fn
        
        if use_global_attn:
            global_core_fn = lambda: AttentionBlock(embed_dim, num_attn_heads, 
                                                    attn_layer_dim, attn_fnn_ratio)
        else:
            global_core_fn = core_mlp_fn
        
        self._core = GraphAttentionNetwork(edge_model_fn=core_mlp_fn,
                                           node_model_fn=node_core_fn,
                                           global_model_fn=global_core_fn,
                                           num_heads=num_attn_heads, 
                                           key_dim=embed_dim)
        

class GlobalHetGraphClassifier(snt.Module):
    def __init__(self, hom_model=False,
        with_edge_inputs=True, with_node_inputs=True, with_global_inputs=False,
        encoder_size=None, core_size=None, decoder_size:list=[],
        node_mlp_size: list=[64,64], edge_mlp_size: list=[64,64], global_mlp_size: list=[64,64], 
        num_attn_layers=1, use_node_attn=True, use_global_attn=True,
        global_in_nodes=None, global_in_edges=None,
        use_encoder=False, use_edge_mp=True,
        reducer=tf.math.unsorted_segment_sum, activation=tf.nn.relu,
        name="GlobalHetGraphClassifier", **kwargs):

        super(GlobalHetGraphClassifier, self).__init__(name=name)


        if core_size is None:
            core_size = [64, 64]
        if node_mlp_size is None:
            print(">>> No node mlp size received, using default config!")
            node_mlp_size = core_size
        if edge_mlp_size is None:
            edge_mlp_size = core_size
        if global_mlp_size is None:
            global_mlp_size = core_size
        
        if hom_model:
            edge_mlp_fn = partial(make_mlp_model, mlp_size=edge_mlp_size, activations=activation, name="EdgeMLP")
            node_mlp = make_mlp_model
        else:
            edge_mlp_fn = partial(make_heterogeneous_edges_model, mlp_size=edge_mlp_size, activations=activation, name="EdgeMLP")
            node_mlp = make_multi_mlp_model
        
        empty = lambda: lambda x: x
        embed_dim = node_mlp_size[:num_attn_layers]
        attn_layer_dim = node_mlp_size[num_attn_layers:]
        self.use_node_attn = use_node_attn
        if use_node_attn:
            self.self_attention = modules.SelfAttention()
            self.k_layer = node_mlp(embed_dim, dropout_rate=0.)
            self.q_layer = node_mlp(embed_dim, dropout_rate=0.)
            self.v_layer = node_mlp(embed_dim, dropout_rate=0.)
            node_mlp_fn = partial(make_mlp_model, mlp_size=attn_layer_dim, activations=activation, name="NodeMLP")
        else:
            node_mlp_fn = partial(node_mlp, mlp_size=node_mlp_size, activations=activation, name="NodeMLP")
        
        global_mlp_fn = partial(make_mlp_model, mlp_size=global_mlp_size, activations=activation, name="GlobalMLP")

        node_block_args, edge_block_args, global_block_args = make_block_args(with_edge_inputs, with_node_inputs, with_global_inputs, reducer)
        
        if global_in_nodes is not None:
            node_block_args['use_globals'] = global_in_nodes
        if global_in_edges is not None:
            edge_block_args['use_globals'] = global_in_edges
        
        print(f'>>> Node Block Options: {node_block_args}')
        
        
        if use_global_attn:
            self._enc = GraphAttentionNetwork(edge_model_fn=edge_mlp_fn,
                                          node_model_fn=node_mlp_fn,
                                          global_model_fn=global_mlp_fn,
                                          reducer=reducer,
                                          edge_block_opt=edge_block_args,
                                          node_block_opt=node_block_args,
                                          global_block_opt=global_block_args,
                                          key_dim=embed_dim[0])
        else:
            self._enc = modules.GraphNetwork(edge_model_fn=edge_mlp_fn,
                                            node_model_fn=node_mlp_fn,
                                            global_model_fn=global_mlp_fn,
                                            reducer=reducer,
                                            edge_block_opt=edge_block_args,
                                            node_block_opt=node_block_args,
                                            global_block_opt=global_block_args)
        
        if use_encoder:
            node_block_args, edge_block_args, global_block_args = get_mp_args(with_edge_inputs, with_node_inputs, with_global_inputs, reducer)
            self._core = modules.GraphNetwork(edge_model_fn=edge_mlp_fn,
                                              node_model_fn=node_mlp_fn,
                                              global_model_fn=global_mlp_fn,
                                              reducer=reducer,
                                              edge_block_opt=edge_block_args,
                                              node_block_opt=node_block_args,
                                              global_block_opt=global_block_args)
            self.use_encoder = True
        else:
            self._core = self._enc
            self.use_encoder = False
        
                
                

        global_output_size = 1
        decoder_size += [global_output_size]
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=activation, # default is relu
                        name='global_classifier_output'),
            tf.sigmoid])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)


    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)
               
        latent = input_op
        if self.use_node_attn:
            node = latent.nodes
            k = self.k_layer(node)
            q = self.q_layer(node)
            v = self.v_layer(node)
            latent = self.self_attention(v, k, q, latent)
        if self.use_encoder:
            latent = self._enc(latent, node_kwargs, edge_kwargs, global_kwargs)
        output_ops = []
        for _ in range(num_processing_steps):
            latent = self._core(latent, node_kwargs, edge_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops


def make_block_args(with_edge_inputs, with_node_inputs, with_global_inputs, reducer):
    node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False,received_edges_reducer=reducer,sent_edges_reducer=reducer)
    edge_block_args=dict(use_edges=False, use_receiver_nodes=True, use_sender_nodes=True, use_globals=False)
    global_block_args=dict(use_edges=True, use_nodes=True, use_globals=False,nodes_reducer=reducer,edges_reducer=reducer)

    if with_edge_inputs:
        edge_block_args['use_edges'] = True
        node_block_args['use_received_edges'] = False
        node_block_args['use_sent_edges'] = False
    if not with_node_inputs:
        edge_block_args['use_receiver_nodes'] = False
        edge_block_args['use_sender_nodes'] = False
        node_block_args['use_nodes'] = False
    if with_global_inputs:
        global_block_args['use_globals'] = True
        #global_block_args['use_nodes'] = False
        #global_block_args['use_edges'] = False
        node_block_args['use_globals'] = True
        edge_block_args['use_globals'] = True
        

    return node_block_args, edge_block_args, global_block_args


def empty_fn(x, is_training=True, **kwargs):
    return x


class GlobalRGNGraphClassifier(snt.Module):
    def __init__(self, hom_model=False,
        with_edge_inputs=True, with_node_inputs=True, with_global_inputs=False,
        encoder_size=None, core_size=None, decoder_size:list=[],
        node_mlp_size: list=[64,64], edge_mlp_size: list=[64,64], global_mlp_size: list=[64,64], 
        core_mp_size: list=[64,64],
        attn_layer_size=[64,64], use_node_attn=True, use_global_attn=True,
        global_in_nodes=None, global_in_edges=None,
        reducer=tf.math.unsorted_segment_sum, activation=tf.nn.relu,
        name="GlobalRGNGraphClassifier", **kwargs):

        super(GlobalRGNGraphClassifier, self).__init__(name=name)


        if core_size is None:
            core_size = [64, 64]
        if node_mlp_size is None:
            print(">>> No node mlp size received, using default config!")
            node_mlp_size = core_size
        if edge_mlp_size is None:
            edge_mlp_size = core_size
        if global_mlp_size is None:
            global_mlp_size = core_size
        
        empty = lambda: empty_fn
        

        if hom_model:
            edge_mlp_fn = partial(make_mlp_model, mlp_size=edge_mlp_size, activations=activation, name="EdgeMLP")
            node_mlp = make_mlp_model
        else:
            edge_mlp_fn = partial(make_heterogeneous_edges_model, mlp_size=edge_mlp_size, activations=activation, name="EdgeMLP")
            node_mlp = make_multi_mlp_model
        
        if not with_edge_inputs:
            edge_mlp_fn = empty

        attn_layer_dim = attn_layer_size[:len(attn_layer_size)-1]
        self.use_node_attn = use_node_attn
        if use_node_attn:
            self.self_attention = modules.SelfAttention()
            self.k_layer = make_mlp_model(attn_layer_dim, dropout_rate=0.)
            self.q_layer = make_mlp_model(attn_layer_dim, dropout_rate=0.)
            self.v_layer = make_mlp_model(attn_layer_dim, dropout_rate=0.)
            core_node_mlp = partial(make_mlp_model, mlp_size=node_mlp_size, activations=activation, name="AttnOutput")
            node_mlp_fn = partial(node_mlp, mlp_size=node_mlp_size, activations=activation, name="NodeMLP")
            #exit()
        else:
            node_mlp_fn = None
            core_node_mlp = partial(node_mlp, mlp_size=node_mlp_size, activations=activation, name="NodeMLP")
        
        global_enc_fn = partial(make_mlp_model, mlp_size=global_mlp_size, activations=activation, name="GlobalMLP")
        core_mlp_fn = partial(make_mlp_model, mlp_size=core_mp_size, activations=activation, name="CoreMLP")

        node_block_args, edge_block_args, global_block_args = self.make_mp_args(with_edge_inputs, with_node_inputs, with_global_inputs, reducer)
        
        if global_in_nodes is not None:
            node_block_args['use_globals'] = global_in_nodes
        if global_in_edges is not None:
            edge_block_args['use_globals'] = global_in_edges
        
        print(f'>>> Node Block Options: {node_block_args}')
        
        
        #node_block_args, edge_block_args, global_block_args = self.get_enc_args(with_edge_inputs, with_node_inputs, with_global_inputs, reducer)
        self._enc = modules.GraphIndependent(edge_model_fn=None,
                                             node_model_fn=node_mlp_fn,
                                             global_model_fn=global_enc_fn)


        if use_global_attn:
            self._core = GraphAttentionNetwork(edge_model_fn=edge_mlp_fn,
                                               node_model_fn=core_node_mlp,
                                               global_model_fn=core_mlp_fn,
                                               reducer=reducer,
                                               edge_block_opt=edge_block_args,
                                               node_block_opt=node_block_args,
                                               global_block_opt=global_block_args,
                                               key_dim=attn_layer_size[-1])
        else:
            self._core = modules.GraphNetwork(edge_model_fn=edge_mlp_fn,
                                              node_model_fn=core_node_mlp,
                                              global_model_fn=core_mlp_fn,
                                              reducer=reducer,
                                              edge_block_opt=edge_block_args,
                                              node_block_opt=node_block_args,
                                              global_block_opt=global_block_args)

        global_output_size = 1
        decoder_size += [global_output_size]
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=activation, # default is relu
                        name='global_classifier_output'),
            tf.sigmoid])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)


    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)
               
        latent = input_op
        latent = self._enc(latent, {}, edge_kwargs, global_kwargs)
        if self.use_node_attn:
            node = latent.nodes
            k = self.k_layer(node)
            q = self.q_layer(node)
            v = self.v_layer(node)
            latent = self.self_attention(v, k, q, latent)
        
        output_ops = []
        for _ in range(num_processing_steps):
            latent = self._core(latent, node_kwargs, edge_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops

    def make_mp_args(self, with_edge_inputs, with_node_inputs, with_global_inputs, reducer):
        node_block_args=dict(use_received_edges=False, 
                            use_sent_edges=False, 
                            use_nodes=True, 
                            use_globals=False,
                            received_edges_reducer=reducer,
                            sent_edges_reducer=reducer)

        edge_block_args=dict(use_edges=False, 
                            use_receiver_nodes=True, 
                            use_sender_nodes=True, 
                            use_globals=False)

        global_block_args=dict(use_edges=False, #changed default
                            use_nodes=True, 
                            use_globals=False,
                            nodes_reducer=reducer,
                            edges_reducer=reducer)

        if with_edge_inputs:
            edge_block_args['use_edges'] = True
            node_block_args['use_received_edges'] = False
            node_block_args['use_sent_edges'] = False
            global_block_args['use_edges'] = True
        if not with_node_inputs:
            edge_block_args['use_receiver_nodes'] = False
            edge_block_args['use_sender_nodes'] = False
            node_block_args['use_nodes'] = False
        if with_global_inputs:
            global_block_args['use_globals'] = True
            #global_block_args['use_nodes'] = False
            #global_block_args['use_edges'] = False
            #node_block_args['use_globals'] = True
            #edge_block_args['use_globals'] = True

        return node_block_args, edge_block_args, global_block_args