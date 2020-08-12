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

__all__ = (
    "GeneralClassifier",
    "EdgeClassifier",
    "EdgeGlobalClassifier",
    "GlobalClassifierNoEdgeInfo",
    "NodeEdgeClassifier",   
)

# class MultiClassifier(snt.Module):
#     def __init__(self, name="MultiClassifier"):
#         super(MultiClassifier, self).__init__(name=name)

#         self._encoder = MLPGraphIndependent()
#         self._core    = MLPGraphNetwork()
#         self._decoder = MLPGraphIndependent()

#         # Transforms the outputs into appropriate shapes.
#         global_output_size = 3
#         global_fn =lambda: snt.Sequential([
#             snt.nets.MLP([LATENT_SIZE, global_output_size],
#                          name='global_output')])

#         self._output_transform = modules.GraphIndependent(None, None, global_fn)

#     def __call__(self, input_op, num_processing_steps):
#         latent = self._encoder(input_op)
#         latent0 = latent

#         output_ops = []
#         for _ in range(num_processing_steps):
#             core_input = utils_tf.concat([latent0, latent], axis=1)
#             latent = self._core(core_input)

#         decoded_op = self._decoder(latent)
#         output_ops.append(self._output_transform(decoded_op))
#         return output_ops