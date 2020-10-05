"""
Functions that convert usual inputs to graph
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from root_gnn.src.datasets.toptagger import TopTaggerDataset
from root_gnn.src.datasets.fourtop import FourTopDataset
from root_gnn.src.datasets.toppair import ToppairDataSet

from root_gnn.src.datasets.wprime import WTaggerDataset
from root_gnn.src.datasets.wprimefiltered import WTaggerFilteredDataset
from root_gnn.src.datasets.wprimeljet import WTaggerLeadingJetDataset

from root_gnn.src.datasets.herwig_hadrons import HerwigHadrons

__all__ = (
    "TopTaggerDataset",
    "FourTopDataset",
    "ToppairDataSet",
    "WTaggerDataset",
    "WTaggerFilteredDataset",
    "WTaggerLeadingJetDataset",
    "HerwigHadrons",
)