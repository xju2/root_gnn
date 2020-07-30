# implementation is similar to
# https://github.com/deepmind/sonnet/blob/v2/sonnet/src/optimizers/adam.py


"""Cyclic Consine Anealing"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from sonnet import Optimizer
from root_gnn.src import types
from root_gnn.src.optimizers import optimizer_utils

import tensorflow as tf
from typing import Optional, Sequence, Text, Union
import math

def cca_update(g, t, lr, iterations):
    return 0.5*lr*(tf.math.cos((t-1)%iterations/iterations*math.pi) + 1) * g

class CyclicCosineAnealing(Optimizer):
    """
    Cyclic Consine Anealing method
    cite: https://arxiv.org/pdf/1608.03983.pdf
    """
    def __init__(
        self,
        iterations: Union[types.IntegerLike, tf.Variable],
        learning_rate: Union[types.FloatLike, tf.Variable] = 0.1,
        name: Optional[Text] = None):
        super(CyclicCosineAnealing, self).__init__(name=name)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.step = tf.Variable(0, trainable=False, name="t", dtype=tf.int64)

    def apply(self, updates: Sequence[types.ParameterUpdate],\
        parameters: Sequence[tf.Variable]):
        
        # optimizer_utils.check_distribution_strategy()
        optimizer_utils.check_updates_parameters(updates, parameters)
        self.step.assign_add(1)
        for update, param in zip(updates, parameters):
            if update is not None:
                optimizer_utils.check_same_dtype(update, param)
                learning_rate = tf.cast(self.learning_rate, update.dtype)
                iterations = tf.cast(self.iterations, update.dtype)
                step = tf.cast(self.step, update.dtype)
                if isinstance(update, tf.IndexedSlices):
                    update, indices = optimizer_utils.deduplicate_indexed_slices(update)
                    update = cca_update(g=update, t=step, lr=learning_rate, iterations=iterations)
                    param.scatter_sub(tf.IndexedSlices(update, indices))
                else:
                    update = cca_update(g=update, t=step, lr=learning_rate, iterations=iterations)
                    param.assign_sub(update) # substract the gradient*step