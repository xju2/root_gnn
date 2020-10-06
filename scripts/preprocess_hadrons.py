#!/usr/bin/env python

import pandas as pd

class HadronParser:
    def __init__(self):
        self.feature_names = ['pdgID','E', 'px', 'py', 'pz', 'invMass']
        self.reset()
        
    def process(self, evt):
        pid = evt.pdgID
        if pid == 81 and len(self.evt_info) > 0:
            if self.n_nodes != 3:
                print(evt)
            evt_info = []
            for key in sorted(self.evt_info.keys(), reverse=True):
                evt_info += self.evt_info[key]
            self.evts.append(" ".join(["{:.04f}".format(x) for x in evt_info]))
            self.evt_info = {}
            self.n_nodes = 0
        self.n_nodes += 1
        values = [getattr(evt, name) for name in self.feature_names]
        if evt.invMass in self.evt_info:
            self.evt_info[evt.invMass-1.1] = values
        else:
            self.evt_info[evt.invMass] = values
    
    def save(self, outname):
        with open(outname, 'w') as f:
            for evt in self.evts:
                f.write(evt +"\n")
                
    def reset(self):
        self.evt_info = {}
        self.evts = []
        self.n_nodes = 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="parse the input file")
    add_arg = parser.add_argument
    add_arg("filename", help='input csv Herwig file')
    add_arg("outname", help='output filename')
    args = parser.parse_args()

    pp = HadronParser()
    columns = ['evtID', 'pdgID', 'E', "px", 'py', 'pz', 'invMass', 'onShell']
    df =  pd.read_csv(args.filename, names=columns, header=None)
    for i in range(df.shape[0]):
        pp.process(df.iloc[i])

    pp.save(args.outname)