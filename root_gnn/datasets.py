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
from root_gnn.src.datasets.tauid_cutoff import TauIdentificationDatasetCutoff
from root_gnn.src.datasets.tauid_perm import TauIdentificationDatasetPermutation
from root_gnn.src.datasets.tauid_cutoff_perm import TauIdentificationDatasetCutoffPermutation
from root_gnn.src.datasets.tauid_cutoff2 import TauIdentificationDatasetCutoff2
from root_gnn.src.datasets.tauid_mmlp import TauIdentificationDatasetMMLP




__all__ = (
    "TopTaggerDataset",
    "WTaggerDataset",
    "WTaggerFilteredDataset",
    "WTaggerLeadingJetDataset",
    "DiTauMassDataset",
    "TauIdentificationDataset",
    "TauIdentificationDatasetCutoff",
    "TauIdentificationDatasetCutoff2",
    "TauIdentificationDatasetPermutation",
    "TauIdentificationDatasetCutoffPermutation",
    "TauIdentificationDatasetMMLP"
)
