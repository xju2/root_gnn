"""
The implementation of Graph Networks are mostly inspired by the ones in deepmind/graphs_nets
https://github.com/deepmind/graph_nets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from root_gnn.src.models.general_classifier import GeneralClassifier
from root_gnn.src.models.edge_classifier import EdgeClassifier
from root_gnn.src.models.edge_global_classifier import EdgeGlobalClassifier
from root_gnn.src.models.global_classifier import GlobalClassifierNoEdgeInfo
from root_gnn.src.models.node_edge_classifier import NodeEdgeClassifier
from root_gnn.src.models.decay_simulator import DecaySimulator
from root_gnn.src.models.node_predictor_v3 import FourTopPredictor

__all__ = (
    "GeneralClassifier",
    "EdgeClassifier",
    "EdgeGlobalClassifier",
    "GlobalClassifierNoEdgeInfo",
    "NodeEdgeClassifier",
    "DecaySimulator",
    'FourTopPredictor',
)