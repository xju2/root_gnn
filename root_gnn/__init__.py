"""
GNN models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model          import SegmentClassifier as mm

def get_model(model_name=None):
    """
    model_name could be used for future testing different models
    """
    return mm()
