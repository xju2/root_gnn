"""
Functions that convert usual inputs to graph
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from root_gnn.src.datasets.toptagger import TopTaggerDataset
from root_gnn.src.datasets.wprime import WTaggerDataset
from root_gnn.src.datasets.fourtop import FourTopDataset
from root_gnn.src.datasets import fourtop
from root_gnn.src.datasets.toptagger import ToppairDataSet

def is_signal(val=True):
    fourtop.is_signal = val

__all__ = (
    "TopTaggerDataset",
    "WTaggerDataset",
    "FourTopDataset",
    "ToppairDataSet",
)