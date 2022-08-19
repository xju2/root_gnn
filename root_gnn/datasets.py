"""
Functions that convert usual inputs to graph
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = (
    "TopTaggerDataset",
    "WTaggerDataset",
    "WTaggerFilteredDataset",
    "WTaggerLeadingJetDataset",
    "DiTauMassDataset",
    "TauIdentificationDataset",
)

from root_gnn.src.datasets.toptagger import TopTaggerDataset
from root_gnn.src.datasets.wprime import WTaggerDataset
from root_gnn.src.datasets.wprimefiltered import WTaggerFilteredDataset
from root_gnn.src.datasets.wprimeljet import WTaggerLeadingJetDataset
from root_gnn.src.datasets.ditaumass import DiTauMassDataset

from root_gnn.src.datasets.tauid import TauIdentificationDataset