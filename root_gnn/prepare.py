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

def is_signal(val=True):
    fourtop.is_signal = val

__all__ = (
    "TopTaggerDataset",
    "WTaggerDataset",
    "FourTopDataset",
)

# class ToppairDataSet(DataSet):
#     def __init__(self, filename, with_padding=False):
#         super(ToppairDataSet, self).__init__(with_padding=with_padding)
#         self.filename = filename

#     def process(self, save=False, outname=None, n_evts_per_record=10):
#         self.graphs = []
#         tree_name = "output"
#         chain = ROOT.TChain(tree_name, tree_name)
#         chain.Add(self.filename)
#         n_entries = chain.GetEntries()
#         print("Total {:,} Events".format(n_entries))
#         n_3jets = 0
#         n_one_top = 0
#         n_two_top = 0
#         evtid = 0
#         max_jets = 0
#         for ientry in range(n_entries):
#             chain.GetEntry(ientry)
#             if len(chain.m_jet_pt) < 3:
#                 continue
#             n_3jets += 1
#             max_jets = max([max_jets, len(chain.m_jet_pt)])
#             if (-1 not in chain.reco_triplet_1 or -1 not in chain.reco_triplet_2):
#                 n_one_top += 1
#             if (-1 not in chain.reco_triplet_1 and -1 not in chain.reco_triplet_2):
#                 n_two_top += 1

#         print("At least 3 jets:   {:10,}, {:.1f}%".format(n_3jets, 100*n_3jets/n_entries))
#         print("At least one top:  {:10,}, {:.1f}%".format(n_one_top, 100*n_one_top/n_entries))
#         print("At least two tops: {:10,}, {:.1f}%".format(n_two_top, 100*n_two_top/n_entries))
#         print("Maximum jets in an event:", max_jets)
#         return
#         print("event id:", evtid)
#         print(chain.m_jet_pt)
#         print(chain.m_jet_eta)
#         zeros = np.array([0.0], dtype=np.float32)
#         def make_graph(event):
#             n_max_nodes = 60
#             n_nodes = len(event.m_jet_pt)
#             print(np.array(event.m_jet_pt).shape)
#             nodes = np.hstack((event.m_jet_pt, event.m_jet_eta, event.m_jet_phi, event.m_jet_E))
#             nodes = nodes.reshape(-1, n_nodes).transpose()
#             print(n_nodes)
#             print(nodes)

#             # edges
#             all_edges = list(itertools.combinations(range(n_nodes), 2))
#             senders = np.array([x[0] for x in all_edges])
#             receivers = np.array([x[1] for x in all_edges])
#             n_edges = len(all_edges)
#             edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
#             true_edges = set(list(itertools.combinations(event.reco_triplet_1, 2)) \
#                 + list(itertools.combinations(event.reco_triplet_2, 2)))
#             truth_labels = [int(x in true_edges) for x in all_edges]
#             print(true_edges)
#             print(event.reco_triplet_1)
#             print(event.reco_triplet_2)
#             print(truth_labels)
#             truth_labels = np.array(truth_labels)

#             input_datadict = {
#                 "n_node": n_nodes,
#                 "n_edge": n_edges,
#                 "nodes": nodes,
#                 "edges": edges,
#                 "senders": senders,
#                 "receivers": receivers,
#                 "globals": np.array([n_nodes], dtype=np.float32)
#             }
#             target_datadict = {
#                 "n_node": n_nodes,
#                 "n_edge": n_edges,
#                 "nodes": zeros,
#                 "edges": truth_labels,
#                 "senders": senders,
#                 "receivers": receivers,
#                 "globals": zeros
#             }
#             input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
#             target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
#             return [(input_graph, target_graph)]

#         make_graph(chain)