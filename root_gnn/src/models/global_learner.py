import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules
from graph_nets import blocks

from root_gnn.src.models.base import MLPGraphNetwork
from root_gnn.src.models.base import make_mlp_model, make_multi_mlp_model, make_concat_mlp_model
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
        encoder_fn = make_mlp_model, **kwargs):
        super(GlobalLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(encoder_fn, mlp_size=encoder_size, name="EncoderMLP", **kwargs)
        else:
            encoder_mlp_fn = partial(encoder_fn, name='EncoderMLP', **kwargs)

        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=False,received_edges_reducer=reducer,sent_edges_reducer=reducer)
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
            self._edge_encoder_block(self._node_encoder_block(input_op, node_kwargs), edge_kwargs), global_kwargs)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
            output_ops.append(self._output_transform(latent))

        return output_ops


class GlobalClassifier(GlobalLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
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

class GlobalInputLearnerBase(snt.Module):

    def __init__(self,
        global_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="GlobalInputLearnerBase",
        reducer=tf.math.unsorted_segment_sum,
        encoder_fn = make_mlp_model, **kwargs):
        super(GlobalInputLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(encoder_fn, mlp_size=encoder_size, name="EncoderMLP", **kwargs)
        else:
            encoder_mlp_fn = partial(ecoder_fn, name='EncoderMLP', **kwargs)

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
            global_block_args['use_globals'] = False

        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=encoder_mlp_fn,
            name='edge_encoder_block', **edge_block_args)

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=encoder_mlp_fn,
            name='node_encoder_block', **node_block_args)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='global_encoder_block', **global_block_args)
        
        self._global_encoder = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='hlv_encoder_block',use_edges=False,use_nodes=False)
        
        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)

        self._core = MLPGraphNetwork(nn_fn=core_mlp_fn, reducer=reducer)

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)

        latent = self._global_block(
            self._edge_encoder_block(self._node_encoder_block(input_op, node_kwargs), edge_kwargs), global_kwargs)
        latent0 = latent
        latent_globals = self._global_encoder(input_op, global_kwargs)

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
        latent.replace(globals=tf.concat([latent.globals,latent_globals.globals],axis=-1))
        output_ops.append(self._output_transform(latent))

        return output_ops
                       
class GlobalInputClassifier(GlobalInputLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        name="GlobalInputClassifier", **kwargs):

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
        
        
class GlobalOnlyLearnerBase(snt.Module):

    def __init__(self,
        global_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="GlobalInputLearnerBase",
        reducer=tf.math.unsorted_segment_sum,
        encoder_fn = make_mlp_model, **kwargs):
        super(GlobalInputLearnerBase, self).__init__(name=name)

        if encoder_size is not None:
            encoder_mlp_fn = partial(encoder_fn, mlp_size=encoder_size, name="EncoderMLP", **kwargs)
        else:
            encoder_mlp_fn = partial(ecoder_fn, name='EncoderMLP', **kwargs)

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
            global_block_args['use_globals'] = False

        self._edge_encoder_block = blocks.EdgeBlock(
            edge_model_fn=encoder_mlp_fn,
            name='edge_encoder_block', **edge_block_args)

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=encoder_mlp_fn,
            name='node_encoder_block', **node_block_args)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='global_encoder_block', **global_block_args)
        
        self._global_encoder = blocks.GlobalBlock(
            global_model_fn=encoder_mlp_fn, name='hlv_encoder_block',use_edges=False,use_nodes=False)
        
        if core_size is not None:
            core_mlp_fn = partial(make_mlp_model, mlp_size=core_size, **kwargs)
        else:
            core_mlp_fn = partial(make_mlp_model, **kwargs)

        self._core = MLPGraphNetwork(nn_fn=core_mlp_fn, reducer=reducer)

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        node_kwargs = edge_kwargs = global_kwargs = dict(is_training=is_training)

        latent = self._global_block(
            self._edge_encoder_block(self._node_encoder_block(input_op, node_kwargs), edge_kwargs), global_kwargs)
        latent0 = latent
        latent_globals = self._global_encoder(input_op, global_kwargs)

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input, edge_kwargs, node_kwargs, global_kwargs)
        latent.replace(globals=tf.concat([latent.globals,latent_globals.globals],axis=-1))
        output_ops.append(self._output_transform(latent))

        return output_ops
                       
class GlobalOnlyClassifier(GlobalInputLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,
        name="GlobalInputClassifier", **kwargs):

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

class GlobalClassifierMultiMLP(GlobalLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,encoder_fn=make_multi_mlp_model,
        name="GlobalClassifier", **kwargs):

        global_output_size = 1
        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]

        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='global_classifier_output'),
            tf.sigmoid])

        super().__init__(global_fn,
            with_edge_inputs=with_edge_inputs,
            with_node_inputs=with_node_inputs,
            with_global_inputs=with_global_inputs,
            encoder_size=encoder_size, core_size=core_size,
            name=name, **kwargs)
        
class GlobalClassifierConcatMLP(GlobalLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
        encoder_size: list=None, core_size: list=None, decoder_size: list=None,encoder_fn=make_concat_mlp_model,
        name="GlobalClassifier", **kwargs):

        global_output_size = 1
        if decoder_size is not None:
            decoder_size += [global_output_size]
        else:
            decoder_size = [global_output_size]

        global_fn =lambda: snt.Sequential([
            snt.nets.MLP(decoder_size,
                        activation=tf.nn.relu, # default is relu
                        name='global_classifier_output'),
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




class GlobalGraphNetClassifier(snt.Module):
    def __init__(self,with_global_inputs=False,core_size: list=None,decoder_size: list=None,name="GlobalSetClassifier",**kwargs):
        super(GlobalGraphNetClassifier, self).__init__(name=name)

        node_block_args=dict(use_received_edges=False, use_sent_edges=False, use_nodes=True, use_globals=with_global_inputs)
        global_block_args=dict(use_edges=False, use_nodes=True, use_globals=with_global_inputs)

        if core_size is not None:
            node_mlp_fn = partial(make_mlp_model, dropout_rate=None,mlp_size=core_size, **kwargs)
        else:
            node_mlp_fn = partial(make_mlp_model, dropout_rate=None,**kwargs)

        global_size = core_size + [1]

        global_fn = lambda: snt.Sequential([snt.nets.MLP(global_size,activation=tf.nn.relu, name='set_classifier_output'),tf.sigmoid])

        self.deep_set = modules.GraphNetwork(node_mlp_fn,node_mlp_fn,global_fn)


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
    """Attention model with interaction GNN"""
    def __init__(self, global_output_size=1,
        with_node_inputs=True, with_edge_inputs=False, with_global_inputs=False, 
        encoder_size: list=None, core_size: list=None, 
        decoder_size: list=None, activation_func=tf.nn.relu, 
        name="GlobalAttentionClassifier",**kwargs):
        
        super(GlobalAttentionClassifier, self).__init__(name=name)

        self.attention = modules.SelfAttention()
        
        self.global_classifier = GlobalClassifier(
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
        output_graph = self.global_classifier(
            weighted_input, num_processing_steps, is_training=is_training)
        
        return output_graph

    
class GlobalLearnerBaseEdgesFirst(snt.Module):
    def __init__(self,
        global_fn,
        with_edge_inputs=False,
        with_node_inputs=True,
        with_global_inputs=False,
        encoder_size: list=None,
        core_size: list=None,
        name="GlobalLearnerBase",
        reducer=tf.math.unsorted_segment_sum,
        encoder_fn = make_mlp_model, **kwargs):
        super(GlobalLearnerBase, self).__init__(name=name)

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


class GlobalClassifierEdgesFirst(GlobalLearnerBase):
    def __init__(self,
        with_edge_inputs=False, with_node_inputs=True, with_global_inputs=False,
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