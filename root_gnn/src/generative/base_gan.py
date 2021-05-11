"""
Generative model based on Graph Neural Network
"""
from types import SimpleNamespace
import functools
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

def generator_loss(fake_output, alt_gen_loss=False):
    if alt_gen_loss:
        loss_ops = -tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output)
    else:
        loss_ops = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output)
    return tf.reduce_mean(loss_ops)


def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)
                        + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))


def Discriminator_Regularizer(p_true, grad_D_true_logits, p_gen, grad_D_gen_logits, batch_size):
    """
    Args:
        p_true: probablity from Discriminator for true events
        grad_D_true_logits: gradient of Discrimantor logits w.r.t its input variables
        p_gen: probability from Discriminator for generated events
        grad_D_gen_logits: gradient of Discrimantor logits w.r.t its input variables
    Returns:
        discriminator regularizer
    """
    grad_D_true_logits_norm = tf.norm(
        tf.reshape(grad_D_true_logits, [batch_size, -1]),
        axis=1, keepdims=True
    )
    grad_D_gen_logits_norm = tf.norm(
        tf.reshape(grad_D_gen_logits, [batch_size, -1]),
        axis=1, keepdims=True
    )
    assert grad_D_true_logits_norm.shape == p_true.shape
    assert grad_D_gen_logits_norm.shape == p_gen.shape
        
    reg_true = tf.multiply(tf.square(1.0 - p_true), tf.square(grad_D_true_logits_norm))
    reg_gen = tf.multiply(tf.square(p_gen), tf.square(grad_D_gen_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_true + reg_gen)
    return disc_regularizer, grad_D_true_logits_norm, grad_D_gen_logits_norm, reg_true, reg_gen


class GANOptimizer(snt.Module):

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
                with_disc_reg=True,
                gamma_reg=1e-3, 
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
            with_disc_reg=with_disc_reg,
            gamma_reg=gamma_reg,
        )

        self.disc_lr = tf.Variable(disc_lr, trainable=False, name='disc_lr', dtype=tf.float32)
        self.gen_lr = tf.Variable(
            gen_lr, trainable=False, name='gen_lr', dtype=tf.float32)
        
        self.disc_opt = snt.optimizers.SGD(learning_rate=self.disc_lr)
        self.gen_opt = snt.optimizers.Adam(learning_rate=self.gen_lr)

        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
        self.decay_lr_start_epoch = tf.constant(decay_lr_start_epoch, dtype=tf.int32)

        self.disc_alpha = tf.constant(1-disc_alpha, dtype=tf.float32)
        self.disc_beta = tf.constant(disc_beta, dtype=tf.float32)


    def get_noise_batch(self):
        noise_shape = [self.hyparams.batch_size, self.hyparams.noise_dim]
        return tf.random.normal(noise_shape, dtype=tf.float32)


    def disc_step(self, inputs_tr, targets_tr, lr_mult=1.0):
        gan = self.gan
        flip = tf.random.uniform(shape=[1], dtype=tf.float32) > 0.9
        noises = self.get_noise_batch()
        inputs = tf.concat([inputs_tr, noises], axis=-1)
        with tf.GradientTape() as tape, tf.GradientTape() as true_tape, tf.GradientTape() as fake_tape:
            gen_evts = gan.generate(inputs)
            gen_evts = tf.concat([inputs_tr, gen_evts], axis=-1)

            true_tape.watch(targets_tr)
            fake_tape.watch(gen_evts)
            real_output = gan.discriminate(targets_tr)
            fake_output = gan.discriminate(gen_evts)
            loss = discriminator_loss(real_output, fake_output)

            if self.hyparams.with_disc_reg:
                grad_logits_true = true_tape.gradient(real_output, targets_tr)
                grad_logits_gen = fake_tape.gradient(fake_output, gen_evts)
                regularizers = Discriminator_Regularizer(
                    tf.sigmoid(real_output),
                    grad_logits_true,
                    tf.sigmoid(fake_output),
                    grad_logits_gen,
                    self.hyparams.batch_size,
                )
                reg_loss = regularizers[0]
                assert reg_loss.shape == loss.shape
                loss += self.hyparams.gamma_reg*reg_loss

        disc_params = gan.discriminator.trainable_variables
        disc_grads = tape.gradient(loss, disc_params)
        if self.hyparams.decay_disc_lr:
            self.disc_lr.assign(self.hyparams.disc_lr * lr_mult)

        self.disc_opt.apply(disc_grads, disc_params)
        if self.hyparams.with_disc_reg:
            return loss, *regularizers
        else:
            return loss,


    def gen_step(self, inputs_tr, lr_mult=1.0):
        gan = self.gan
        noises = self.get_noise_batch()
        inputs = tf.concat([inputs_tr, noises], axis=-1)
        with tf.GradientTape() as tape:
            gen_graph = gan.generate(inputs)
            gen_graph = tf.concat([inputs_tr, gen_graph], axis=-1)
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

        num_decay_epochs = tf.cast(self.num_epochs - self.decay_lr_start_epoch,
                                dtype=tf.float32)
        return (num_decay_epochs - decay_epoch) / num_decay_epochs

    def step(self, inputs_tr, targets_tr, epoch):
        lr_mult = self._get_lr_mult(epoch)
        disc_loss = self.disc_step(inputs_tr, targets_tr, lr_mult=lr_mult)
        gen_loss = self.gen_step(inputs_tr, lr_mult=lr_mult)
        return disc_loss, gen_loss, lr_mult

    def cond_gen(self, inputs_tr):
        gan = self.gan
        noises = self.get_noise_batch()
        inputs = tf.concat([inputs_tr, noises], axis=-1)
        gen_evts = gan.generate(inputs)
        return gen_evts