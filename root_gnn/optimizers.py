# implementation is similar to
# https://github.com/deepmind/sonnet/blob/v2/sonnet/src/optimizers/adam.py


"""Cyclic Consine Anealing"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from sonnet.src import base
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
from sonnet.src.optimizers import optimizer_utils

import tensorflow as tf
from typing import Optional, Sequence, Text, Union
import math

def cca_update(g, t, lr, iterations):

    return 0.5*lr*(tf.math.cos((t-1)%iterations/iterations*math.pi) + 1) * g

class CyclicCosineAnealing(base.Optimizer):
    """
    Cyclic Consine Anealing method
    cite: https://arxiv.org/pdf/1608.03983.pdf
    """
    def __init__(
        self,
        iterations: Union[types.IntLike, tf.Variable],
        learning_rate: Union[types.FloatLike, tf.Variable] = 0.1,
        name: Optional[Text] = None):
        super(CyclicCosineAnealing, self).__init__(name=name)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def apply(self, updates: Sequence[types.ParameterUpdate],\
        parameters: Sequence[tf.Variable]):
        
        optimizer_utils.check_distribution_strategy()
        optimizer_utils.check_updates_parameters(updates, parameters)
