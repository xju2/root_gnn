"""
Generative model based on Graph Neural Network
"""
from types import SimpleNamespace
import functools
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import blocks
from graph_nets import modules

from graph_nets import graphs

LATENT_SIZE = 128
NUM_LAYERS = 2

def my_print(g, data=False):
    for field_name in graphs.ALL_FIELDS:
        per_replica_sample = getattr(g, field_name)
        if per_replica_sample is None:
            print(field_name, "EMPTY")
        else:
            print(field_name, ":", per_replica_sample.shape)
            if data and field_name != "edges":
                print(per_replica_sample)

# Use Spectral Normalization, arXiv:1802.05957,
# in Generator and discriminators.
# The implementation is taken from deepmind/sonnet examples
# https://github.com/deepmind/sonnet/blob/v2/examples/little_gan_on_mnist.ipynb

class SpectralNormalizer(snt.Module):

    def __init__(self, epsilon=1e-12, name=None):
        super().__init__(name=name)
        self.l2_normalize = functools.partial(
            tf.math.l2_normalize, epsilon=epsilon)

    @snt.once
    def _initialize(self, weights):
        init = self.l2_normalize(snt.initializers.TruncatedNormal()(
            shape=[1, weights.shape[-1]], dtype=weights.dtype))
        # 'u' tracks our estimate of the first spectral vector for the given weight.
        self.u = tf.Variable(init, name='u', trainable=False)

    def __call__(self, weights, is_training=True):
        self._initialize(weights)
        if is_training:
            # Do a power iteration and update u and weights.
            weights_matrix = tf.reshape(weights, [-1, weights.shape[-1]])
            v = self.l2_normalize(self.u @ tf.transpose(weights_matrix))
            v_w = v @ weights_matrix
            u = self.l2_normalize(v_w)
            sigma = tf.stop_gradient(tf.reshape(v_w @ tf.transpose(u), []))
            self.u.assign(u)
            weights.assign(weights / sigma)
        return weights


class SpectrallyNormedLinear(snt.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_normalizer = SpectralNormalizer()

    def __call__(self, inputs, is_training=True):
        self._initialize(inputs)

        normed_w = self.spectral_normalizer(self.w, is_training=is_training)
        outputs = tf.matmul(inputs, normed_w)
        if self.with_bias:
            outputs = tf.add(outputs, self.b)
        return outputs


class SimpleBlock(snt.Module):

    def __init__(self, embed_dim, with_batch_norm=False, name=None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.hidden = SpectrallyNormedLinear(self.embed_dim, with_bias=True)
        if with_batch_norm:
            self.bn = snt.BatchNorm(create_scale=True, create_offset=True)
        else:
            self.bn = None

    def __call__(self, inputs, is_training=True):
        output = self.hidden(inputs, is_training=is_training)
        if self.bn:
            output = self.bn(output, is_training=is_training)
        return output


class RegulizedMLP(snt.Module):

    def __init__(self,
                 output_sizes: Iterable[int],
                 with_batch_norm: bool = False,
                 dropout_rate=None,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 activate_final: bool = False,
                 name: Optional[Text] = None):
        """
        Constructs an MLP with batch-normalization, spectral regulization, and drop out.
        """
        super().__init__(name=name)

        self._activation = activation
        self._activate_final = activate_final
        self._dropout_rate = dropout_rate
        self._layers = []
        for index, output_size in enumerate(output_sizes):
            self._layers.append(
                SimpleBlock(
                    embed_dim=output_size,
                    with_batch_norm=with_batch_norm,
                    name="linear_%d" % index))

    def __call__(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        use_dropout = self._dropout_rate not in (None, 0)
        num_layers = len(self._layers)

        for i, layer in enumerate(self._layers):
            inputs = layer(inputs, is_training=is_training)
            if i < (num_layers - 1) or self._activate_final:
                if use_dropout and is_training:
                    inputs = tf.nn.dropout(inputs, rate=self._dropout_rate)
                inputs = self._activation(inputs)

        return inputs


def make_regulized_mlp():
    return snt.Sequential([
            RegulizedMLP([LATENT_SIZE]*NUM_LAYERS,
                    activation=tf.nn.relu,
                    activate_final=True,
                    dropout_rate=0.30,
                    with_batch_norm=True,
                    ),
            snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ])


class SetsGenerator(snt.Module):
    def __init__(self, input_dim: int = 12, out_dim: int = 4,
        with_regulization: bool = True,
        with_batch_norm: bool = False,
        name="SetsGenerator",
        ):
        """
        initilize the generator by giving the input dimension and output dimension.
        """
        super(SetsGenerator, self).__init__(name=name)

        MLP_fn = RegulizedMLP if with_regulization else snt.nets.MLP
        # graph creation
        self._node_linear = snt.Sequential([
            MLP_fn([LATENT_SIZE]*NUM_LAYERS+[out_dim],
                    activation=tf.nn.relu,
                    activate_final=False,
                    dropout_rate=0.30,
                    with_batch_norm=with_batch_norm,
                    name="node_encoder_nn"
                    ),
            snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ])


        # node properties
        self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name="node_rnn")
        self._node_prop_nn = MLP_fn([LATENT_SIZE]*NUM_LAYERS+[out_dim],
                                    activation=tf.nn.relu,
                                    activate_final=False,
                                    dropout_rate=0.30,
                                    with_batch_norm=with_batch_norm,
                                    name="node_prop_nn")

        self.out_dim = out_dim

    def __call__(self,
                 input_op,
                 max_nodes: int,
                 training: bool = True) -> tf.Tensor:
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
        node_hidden_state = tf.zeros(
            [batch_size, LATENT_SIZE], dtype=tf.float32, name='initial_node_hidden')
        latent = input_op.replace(
            nodes=self._node_linear(input_op.nodes, training))

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
            node_embedding, node_hidden_state = self._node_rnn(
                node_embedding, node_hidden_state)
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
        # nodes = tf.reshape(nodes[:, 1:, :], [-1, self.out_dim])
        return nodes[:, 1:, :]



class MLPGraphNetwork(snt.Module):
    """GraphIndependent with MLP edge, node, and global models."""
    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        self._network = modules.GraphNetwork(
            edge_model_fn=make_regulized_mlp,
            node_model_fn=make_regulized_mlp,
            global_model_fn=make_regulized_mlp
            )

    def __call__(self, inputs,
            edge_model_kwargs=None,
            node_model_kwargs=None,
            global_model_kwargs=None):
        return self._network(inputs,
                    edge_model_kwargs=edge_model_kwargs,
                    node_model_kwargs=node_model_kwargs,
                    global_model_kwargs=global_model_kwargs)


class SetsDiscriminator(snt.Module):
    def __init__(self, name="SetsDiscriminator"):
        super(SetsDiscriminator, self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=make_regulized_mlp,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
            name='edge_encoder_block')

        self._node_encoder_block = blocks.NodeBlock(
            node_model_fn=make_regulized_mlp,
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name='node_encoder_block'
        )

        self._global_block = blocks.GlobalBlock(
            global_model_fn=make_regulized_mlp,
            use_edges=True,
            use_nodes=True,
            use_globals=False,
        )

        self._core = MLPGraphNetwork()
        # Transforms the outputs into appropriate shapes.
        global_output_size = 1

        def global_fn(): return snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, global_output_size],
                         name='global_output'), tf.sigmoid])

        self._output_transform = modules.GraphIndependent(
            None, None, global_fn)

    def __call__(self, input_op, num_processing_steps, is_training=True):
        kwargs = dict(is_training=is_training)
        latent = self._node_encoder_block(input_op, kwargs)
        latent = self._edge_block(latent, kwargs)
        latent = self._global_block(latent, kwargs)

        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(
                core_input,
                edge_model_kwargs=kwargs,
                node_model_kwargs=kwargs,
                global_model_kwargs=kwargs
            )
            output_ops.append(self._output_transform(latent))

        return output_ops



class SetGAN(snt.Module):

    def __init__(self, noise_dim, max_nodes, num_iters, batch_size, name=None):
        super().__init__(name=name)
        self.generator = SetsGenerator(noise_dim+4, 4, with_regulization=True, with_batch_norm=True)
        self.discriminator = SetsDiscriminator()
        self._max_nodes = max_nodes
        self._num_iters = num_iters
        self._batch_size = batch_size

    def generate(self, inputs_tr, noise, is_training=True):
        incident_info = inputs_tr.nodes
        input_op = tf.concat([inputs_tr.nodes, noise], axis=-1)
        inputs_tr = inputs_tr.replace(nodes=input_op)

        node_pred = self.generator(inputs_tr, max_nodes=self._max_nodes)
        incident_info = tf.reshape(
            incident_info, [self._batch_size, 1, 4])
        node_pred = tf.concat([incident_info, node_pred], axis=1)
        node_pred = tf.reshape(node_pred, [-1, 4])

        n_node = tf.constant([self._max_nodes]*self._batch_size, dtype=tf.int32)
        n_edge = tf.constant([0]*self._batch_size, dtype=tf.int32)
        pred_graph = graphs.GraphsTuple(
            nodes=node_pred, edges=None, globals=tf.constant([0]*self._batch_size, dtype=tf.float32),
            receivers=None, senders=None, n_node=n_node,
            n_edge=n_edge
        )
        pred_graph = utils_tf.fully_connect_graph_static(
            pred_graph, exclude_self_edges=True)
        pred_graph = pred_graph.replace(edges=tf.zeros(
            [pred_graph.senders.shape[0], 1], dtype=tf.float32))
        return pred_graph

    def discriminate(self, input_graph):
        return self.discriminator(input_graph, self._num_iters)


def generator_loss(fake_output):
    loss_ops = [tf.compat.v1.losses.log_loss(tf.ones_like(output_op.globals, dtype=tf.float32), output_op.globals)
                for output_op in fake_output
                ]
    return tf.reduce_mean(tf.stack(loss_ops))


def discriminator_loss(real_output, fake_output, disc_alpha, disc_beta):
    loss_ops = [tf.compat.v1.losses.log_loss(
        tf.ones_like(output_op.globals, dtype=tf.float32), output_op.globals, weights=disc_alpha)
        for output_op in real_output]
    loss_ops += [tf.compat.v1.losses.log_loss(
        tf.zeros_like(output_op.globals, dtype=tf.float32), output_op.globals, weights=disc_beta)
        for output_op in fake_output]
    return tf.reduce_mean(tf.stack(loss_ops))


class SetGANOptimizer(snt.Module):

    def __init__(self,
                gan, 
                batch_size=100,
                noise_dim=128,
                disc_lr=2e-4,
                gen_lr=5e-5,
                num_epochs=100,
                decay_lr_start_epoch=50,
                decay_disc_lr=True,
                decay_gen_lr=True,
                disc_alpha=0.1,
                disc_beta=0.8,
                name=None, *args, **kwargs):
        super().__init__(name=name)
        self.gan = gan
        self.hyparams = SimpleNamespace(
            batch_size=batch_size,
            disc_lr=disc_lr,
            gen_lr=gen_lr,
            num_epochs=num_epochs,
            decay_lr_start_epoch=decay_lr_start_epoch,
            decay_disc_lr=decay_disc_lr,
            decay_gen_lr=decay_gen_lr,
            noise_dim=noise_dim,
        )

        self.disc_lr = tf.Variable(disc_lr, trainable=False, name='disc_lr', dtype=tf.float32)
        self.gen_lr = tf.Variable(
            gen_lr, trainable=False, name='gen_lr', dtype=tf.float32)
        
        self.disc_opt = snt.optimizers.Adam(learning_rate=self.disc_lr, beta1=0.0)
        self.gen_opt = snt.optimizers.Adam(learning_rate=self.gen_lr, beta1=0.0)

        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
        self.decay_lr_start_epoch = tf.constant(decay_lr_start_epoch, dtype=tf.int32)

        self.disc_alpha = tf.constant(1-disc_alpha, dtype=tf.float32)
        self.disc_beta = tf.constant(disc_beta, dtype=tf.float32)

    def get_noise_batch(self):
        noise_shape = [self.hyparams.batch_size, self.hyparams.noise_dim]
        return tf.random.normal(noise_shape, dtype=tf.float32)

    def disc_step(self, inputs_tr, targets_tr, lr_mult=1.0):
        gan = self.gan
        with tf.GradientTape() as tape:
            gen_graph = gan.generate(inputs_tr, self.get_noise_batch())
            # my_print(gen_graph)
            # my_print(targets_tr)

            real_output = gan.discriminate(gen_graph)
            fake_output = gan.discriminate(targets_tr)
            
            loss = discriminator_loss(
                real_output, fake_output,
                self.disc_alpha,
                self.disc_beta)

        disc_params = gan.discriminator.trainable_variables
        disc_grads = tape.gradient(loss, disc_params)
        if self.hyparams.decay_disc_lr:
            self.disc_lr.assign(self.hyparams.disc_lr * lr_mult)
        self.disc_opt.apply(disc_grads, disc_params)
        return loss


    def gen_step(self, inputs_tr, lr_mult=1.0):
        gan = self.gan
        noise = self.get_noise_batch()
        with tf.GradientTape() as tape:
            gen_graph = gan.generate(inputs_tr, noise)
            fake_output = gan.discriminate(gen_graph)
            loss = generator_loss(fake_output)

        gen_params = gan.generator.trainable_variables
        gen_grads = tape.gradient(loss, gen_params)
        if self.hyparams.decay_gen_lr:
            self.gen_lr.assign(self.hyparams.gen_lr * lr_mult)
        self.gen_opt.apply(gen_grads, gen_params)
        return loss

    def _get_lr_mult(self, epoch):
        # Linear decay to 0.
        decay_epoch = tf.cast(epoch - self.decay_lr_start_epoch, tf.float32)
        if decay_epoch < tf.constant(0, dtype=tf.float32):
            return tf.constant(1., dtype=tf.float32)

        num_decay_epochs = tf.cast(self.hyparams.num_epochs - self.decay_lr_start_epoch,
                                dtype=tf.float32)
        return (num_decay_epochs - decay_epoch) / num_decay_epochs

    def step(self, inputs_tr, targets_tr, epoch):
        lr_mult = self._get_lr_mult(epoch)
        disc_loss = self.disc_step(inputs_tr, targets_tr, lr_mult=lr_mult)
        gen_loss = self.gen_step(inputs_tr, lr_mult=lr_mult)
        return disc_loss, gen_loss, lr_mult