"""
The implementation of Graph Networks are mostly inspired by the ones in deepmind/graphs_nets
https://github.com/deepmind/graph_nets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from root_gnn.src.models.edge_learner import EdgeClassifier
from root_gnn.src.models.edge_learner import EdgeRegression
from root_gnn.src.models.global_learner import GlobalClassifier
from root_gnn.src.models.global_learner import GlobalClassifierHeterogeneousEdges, GlobalClassifierHeterogeneousNodes
from root_gnn.src.models.global_learner import GlobalRNNClassifierSNT, GlobalAttnEncoderClassifier

from root_gnn.src.models.global_learner import *

__all__ = (
    "EdgeClassifier",
    "EdgeRegression",
    "GlobalClassifier",
    "GlobalClassifierHeterogeneousEdges",
    "GlobalClassifierHeterogeneousNodes",
    "GlobalRNNClassifierSNT",
    "GlobalAttnEncoderClassifier"
)
