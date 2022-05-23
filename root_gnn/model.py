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
from root_gnn.src.models.global_learner import GlobalRegression
from root_gnn.src.models.global_learner import GlobalSetClassifier
from root_gnn.src.models.global_learner import GlobalGraphNetClassifier
from root_gnn.src.models.global_learner import GlobalClassifierMultiMLP
from root_gnn.src.models.global_learner import GlobalClassifierConcatMLP
from root_gnn.src.models.global_learner import GlobalAttentionClassifier
from root_gnn.src.models.global_learner import GlobalClassifierEdgesFirst, GlobalClassifierHeterogeneousEdges, GlobalClassifierHeterogeneousNodes

__all__ = (
    "EdgeClassifier",
    "EdgeRegression",
    "GlobalClassifier",
    "GlobalRegression",
    "GlobalSetClassifier",
    "GlobalGraphNetClassifier",
    "GlobalClassifierMultiMLP",
    "GlobalClassifierConcatMLP",
    "GlobalAttentionClassifier",
    "GlobalClassifierEdgesFirst",
    "GlobalClassifierHeterogeneousEdges",
    "GlobalClassifierHeterogeneousNodes"
)
