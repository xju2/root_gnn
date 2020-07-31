#!/usr/bin/env python

from root_gnn import prepare

if __name__ == "__main__":
    # filename = "/global/cfs/cdirs/atlas/allx123/root_gnn/samples/mc16a.PowhegPy8_ttH125_fixweight.MxAODDetailed.e7488_s3126_r9364_p4097_h025.root"
    # filename = "/global/cfs/cdirs/atlas/allx123/root_gnn/samples/mc16d.PowhegPy8_ttH125_fixweight.MxAODDetailed.e7488_s3126_r10201_p4097_h025.root"
    filename = "/global/cfs/cdirs/atlas/allx123/root_gnn/samples/mc16e.PowhegPy8_ttH125_fixweight.MxAODDetailed.e7488_s3126_r10724_p4097_h025.root"

    def check_tth_file(filename):
        tree_name = "output"
        chain = ROOT.TChain(tree_name, tree_name)
        chain.Add(filename)
        n_entries = chain.GetEntries()
        print("Total {:,} Events".format(n_entries))
        n_3jets = 0
        n_one_top = 0
        n_two_top = 0
        # evtid = 0
        max_jets = 0
        for ientry in range(n_entries):
            chain.GetEntry(ientry)
            if len(chain.m_jet_pt) < 3:
                continue
            n_3jets += 1
            max_jets = max([max_jets, len(chain.m_jet_pt)])
            if (-1 not in chain.reco_triplet_1 or -1 not in chain.reco_triplet_2):
                n_one_top += 1
            if (-1 not in chain.reco_triplet_1 and -1 not in chain.reco_triplet_2):
                n_two_top += 1

        print("At least 3 jets:   {:10,}, {:.1f}%".format(n_3jets, 100*n_3jets/n_entries))
        print("At least one top:  {:10,}, {:.1f}%".format(n_one_top, 100*n_one_top/n_entries))
        print("At least two tops: {:10,}, {:.1f}%".format(n_two_top, 100*n_two_top/n_entries))
        print("Maximum jets in an event:", max_jets)
    
    # dataset = prepare.ToppairDataSet(filename)
    # dataset.process()