import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import networkx as nx
from root_gnn.src.datasets.tauid import TauIdentificationDataset
from tqdm import trange

dataset = TauIdentificationDataset()

ditauPU = '/global/cfs/cdirs/m3443/data/TauStudies/v5/ditau_train.root'
qcdPU = '/global/cfs/cdirs/m3443/data/TauStudies/v5/qcd_train.root'
ditau_test = '/global/cfs/cdirs/m3443/data/TauStudies/v5/ditau_test.root'
qcd_test = '/global/cfs/cdirs/m3443/data/TauStudies/v5/qcd_test.root'
test = '/global/cfs/cdirs/m3443/data/TauStudies/v5/all_test.root'
hlv_names = ["JetLeadingTrackFracP", "JetTrackR", 
                "JetMaxDRInCore", "JetNumISOTracks", "JetTrackMass"]

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi

def get_info(chain, sig=False):
    signals = []
    pt, et = [], []
    ntrk, ntwr = [], []
    pt_frac, et_frac = [], []
    hlv = {}
    #for name in hlv_names:
        #hlv[name] = []
    o, t, b = 0, 0, 0
    track_idx = 0
    tower_idx = 0
    for ijet in range(chain.nJets):
        
        min_index = 0
        if chain.nTruthJets > 0:
            min_dR = math.sqrt(calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[0])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[0])**2)
        for itruth in range(chain.nTruthJets):
            dR = math.sqrt(calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[itruth])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[itruth])**2)
            if dR < min_dR:
                min_dR = dR
                min_index = itruth
        if chain.nTruthJets > 0 and min_dR < 0.4:
            isTau = chain.TruthJetIsTautagged[min_index]
        else:
            isTau = 0
        
        tower_Et, track_Pt = [], []
        for itower in range(chain.JetTowerN[ijet]):    
            tower_Et.append(chain.JetTowerEt[tower_idx])
            tower_idx += 1
        
        
        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            track_Pt.append(chain.TrackPt[ghost_track_idx])
            track_idx += 1
            
        isTau = 0 if isTau != 1 and isTau != 3 else isTau
        if sig and not isTau:
            continue
        if not sig and isTau:
            continue
        if chain.JetPt[ijet] < 30 or abs(chain.JetEta[ijet]) >= 3:
            continue
        
        #track_Pt.sort(reverse=True)
        #tower_Et.sort(reverse=True)
        
        if isTau == 1:
            o += 1
        elif isTau == 3:
            t += 1
        elif isTau == 0:
            b += 1
        #pt.append(np.array(track_Pt, dtype=np.float32))
        #et.append(np.array(tower_Et, dtype=np.float32))
        #ntrk.append(len(track_Pt))
        #ntwr.append(len(tower_Et))
        
       
    return o, t, b

nditau = dataset._num_evts(ditauPU)
#nqcd = dataset._num_evts(qcdPU)
ditau = dataset.read(ditauPU, 0, nditau)
#qcd = dataset.read(qcdPU, 0, nevt)
one_p, three_p, background = 0, 0, 0

for _ in trange(nditau, desc="ditau"):
    f = next(ditau)
    o, t, b = get_info(f, sig=True)
    one_p += o
    three_p += t
    background += b


print(one_p, three_p, background)
np.savez('ntau.npz', one_prong=one_p, three_prong=three_p, background=background)
