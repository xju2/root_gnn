
import ROOT
from ROOT import TFile
import os
import numpy as np
import networkx as nx

from . import prepare

class index_mgr:
    def __init__(self, n_total, training_frac=0.8):
        self.max_tr = int(n_total*training_frac)
        self.total = n_total
        self.n_test = n_total - self.max_tr
        self.tr_idx = 0
        self.te_idx = self.max_tr

    def next(self, is_training=False):
        if is_training:
            self.tr_idx += 1
            if self.tr_idx > self.max_tr:
                self.tr_idx = 0
            return self.tr_idx
        else:
            self.te_idx += 1
            if self.te_idx > self.total:
                self.te_idx = self.max_tr
            return self.te_idx


class dataset:
    def __init__(self, file_name, tree_name, branches=None):
        self.file_ = TFile.Open(file_name, 'READ')
        self.tree_ = self.file_.Get(tree_name)
        if branches is not None:
            self.tree_.SetBranchStatus('*', 0)
            for var_name in branches:
                self.tree_.SetBranchStatus(var_name, 1)
            self.tree_.SetBranchStatus('weight', 1)

        # use 80% for training and 20% for testing
        self.idx_ = index_mgr(self.tree_.GetEntries())


    def generate_nxgraph(self, is_training=True):
        mychain = self.tree_

        evtid = self.idx_.next(is_training)
        mychain.GetEntry(evtid)
        #TODO need to find smart way to apply selections
        # maybe use a function as argument?
        while mychain.n_jets < 2:
            evtid = self.idx_.next(is_training)
            mychain.GetEntry(evtid)

        myEvent = nx.DiGraph()
        node_idx = 0
        scale = np.array([100, 2.5, np.pi, 1])


        if mychain.n_jets > 0:
            for idx_jet in reversed(np.argsort(list(mychain.jet_pt)).tolist()):
                j_pt = mychain.jet_pt[idx_jet]
                j_eta = mychain.jet_eta[idx_jet]
                j_phi = mychain.jet_phi[idx_jet]
                j_m   = mychain.jet_m[idx_jet]
                j_lv = ROOT.TLorentzVector()
                j_lv.SetPtEtaPhiM(j_pt, j_eta, j_phi, j_m)
                j_r = j_pt/j_lv.E()

                myEvent.add_node(node_idx, pos=np.array([j_pt,j_eta,j_phi, j_m])/scale)
                node_idx += 1


        for idx_lep in reversed(np.argsort(list(mychain.lepton_pt)).tolist()):
            l_pt  = mychain.lepton_pt[idx_lep]
            l_eta = mychain.lepton_eta[idx_lep]
            l_phi = mychain.lepton_phi[idx_lep]
            l_m   = l_pt/mychain.lepton_m[idx_lep]
            l_lv = ROOT.TLorentzVector()
            l_lv.SetPtEtaPhiM(l_pt, l_eta, l_phi, l_m)
            l_r = l_pt/l_lv.E()
            myEvent.add_node(node_idx, pos=np.array([l_pt,l_eta,l_phi, l_m])/scale)
            node_idx += 1

        for i in range(node_idx):
            for j in range(i+1, node_idx):
                delta_eta = myEvent.node[j]['pos'][1]-myEvent.node[i]['pos'][1]
                delta_phi = prepare.calc_dphi(myEvent.node[j]['pos'][2], myEvent.node[i]['pos'][2])
                delta_r = np.sqrt(delta_eta**2+delta_phi**2)
                myEvent.add_edge(i,j, distance = [delta_eta,  delta_phi, delta_r])
                myEvent.add_edge(j,i, distance = [-delta_eta, -delta_phi, delta_r])

        myEvent.graph['attributes'] = np.array([mychain.n_jets, len(mychain.lepton_pt)])
        # myEvent.graph['solution'] = np.array([int(is_signal)])

        return myEvent


    def get_graphs(self, n_graphs, is_training=True):
        input_graphs = [None]*n_graphs
        target_graphs = [None]*n_graphs

        for iph in range(n_graphs):
            newgraph_signal = self._generate_graph(is_signal=True,
                                                   is_training=is_training)
            newgraph_bkg = self._generate_graph(is_signal=False,
                                                is_training=is_training)

            input_g1, target_g1 = prepare.graph_to_input_target(newgraph_signal)
            input_g2, target_g2 = prepare.graph_to_input_target(newgraph_bkg)
            input_graphs[iph*2:iph*2+1] = [input_g1, input_g2]
            target_graphs[iph*2:iph*2+1] = [target_g1, target_g2]

        return input_graphs, target_graphs
