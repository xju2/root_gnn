"""Tests for sonnet.v2.src.sgd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src.optimizers import optimizer_tests
from sonnet.src.optimizers import sgd
import tensorflow as tf

CONFIGS = optimizer_tests.named_product()
class CCATest(optimizer_tests.OptimizerTestBase):

    def make_optimizer(self, *args, **kwargs):
        pass