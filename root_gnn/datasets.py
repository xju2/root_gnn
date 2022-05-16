"""
Functions that convert usual inputs to graph
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from root_gnn.src.datasets.toptagger import TopTaggerDataset
from root_gnn.src.datasets.wprime import WTaggerDataset
from root_gnn.src.datasets.wprimefiltered import WTaggerFilteredDataset
from root_gnn.src.datasets.wprimeljet import WTaggerLeadingJetDataset
from root_gnn.src.datasets.ditaumass import DiTauMassDataset

from root_gnn.src.datasets.tauid import TauIdentificationDataset
from root_gnn.src.datasets.tauid_trackvars import TauIdentificationDatasetTrackvars
from root_gnn.src.datasets.tauid_trackvars_towervars import TauIdentificationDatasetTrackvarsTowervars
from root_gnn.src.datasets.tauid_heterogeneous_nodes import TauIdentificationDatasetHeterogeneousNodes
from root_gnn.src.datasets.tauid_heterogeneous_edges import TauIdentificationDatasetHeterogeneousEdges

__all__ = (
    "TopTaggerDataset",
    "WTaggerDataset",
    "WTaggerFilteredDataset",
    "WTaggerLeadingJetDataset",
    "DiTauMassDataset",
    "TauIdentificationDataset",
    "TauIdentificationDatasetJetPt",
    "TauIdentificationDatasetTrackvars",
    "TauIdentificationDatasetTrackvarsTowervars",
    "TauIdentificationDatasetHeterogeneousNodes",
    "TauIdentificationDatasetHeterogeneousEdges"
)
