#!/usr/bin/env python
import numpy as np
import itertools
import ROOT
from ROOT import TChain, AddressOf, std
from array import array
import matplotlib.pyplot as plt

def plot_jet_rings(jet_eta,jet_phi,tau_tagged):
    """
    Function for graphing circles displaying the core (R=0.1), central (R=0.2) and outer (R=0.4) regions of the jet
    """
    theta = np.linspace(0, 2*np.pi, 100)
    radius_core = 0.1
    a_core = radius_core*np.cos(theta) + jet_eta
    b_core = radius_core*np.sin(theta) + jet_phi

    radius_cent = 0.2
    a_cent = radius_cent*np.cos(theta) + jet_eta
    b_cent = radius_cent*np.sin(theta) + jet_phi

    radius_out = 0.4
    a_out = radius_out*np.cos(theta) + jet_eta
    b_out = radius_out*np.sin(theta) + jet_phi
    if tau_tagged == 0: 
        plt.plot(a_core, b_core, 'r')
        plt.plot(a_cent, b_cent, 'r')
        plt.plot(a_out, b_out, 'r')
        plt.plot(jet_eta, jet_phi, 'r')
    else:
        plt.plot(a_core, b_core, 'y')
        plt.plot(a_cent, b_cent, 'y')
        plt.plot(a_out, b_out, 'y')
        plt.plot(jet_eta, jet_phi, 'y')

def evt_display(filename, evt_type, evt_id, jet_idxs = 'None'):
    """
    Displays all tracks and towers in an event along with rings for each jet. Includes functionallity to only graph particular jets and their associated tracks/towers
    """
    tree_name = "output"
    chain = ROOT.TChain(tree_name, tree_name) 
    chain.Add(filename)
    chain.GetEntry(evt_id)
    ax = plt.gca() #you first need to get the axis handle
    ax.set_aspect(1)
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    if jet_idxs == 'None':
        plt.plot(chain.TowerEta,chain.TowerPhi, 'mo', fillstyle ='none', markersize = 5, label = 'towers')
        plt.plot( chain.TrackEta,chain.TrackPhi, 'cx', markersize = 5, label = 'tracks')
        
        tower_phi = []
        tower_eta = []
        track_phi = []
        track_eta = []
        last_tower_idx = 0 
        last_track_idx = 0 #keeps track of last final index for tracks and towers
        #this for loop splits up all of the tower/track phi and eta data into different lists for each jet, compiles together as list of lists
        for jet_id in range(chain.nJets):
            jet_tower_n = chain.JetTowerN[jet_id]
            tower_phi.append(chain.JetTowerPhi[last_tower_idx:last_tower_idx+jet_tower_n])
            tower_eta.append(chain.JetTowerEta[last_tower_idx:last_tower_idx+jet_tower_n])
            last_tower_idx += jet_tower_n
            
            jet_track_n = chain.JetGhostTrackN[jet_id]
            jet_track_idxs = chain.JetGhostTrackIdx[last_track_idx:last_track_idx+jet_track_n]
            
            jet_track_eta = []
            jet_track_phi = []
            for jet_track_idx in jet_track_idxs:
                jet_track_eta.append(chain.TrackEta[jet_track_idx])
                jet_track_phi.append(chain.TrackPhi[jet_track_idx])
            track_eta.append(jet_track_eta)
            track_phi.append(jet_track_phi)
            last_track_idx += jet_track_n
            
            plot_jet_rings(chain.JetEta[jet_id],chain.JetPhi[jet_id], chain.TruthJetIsTautagged[jet_id])
            plt.plot(tower_eta[jet_id],tower_phi[jet_id], 'bo', fillstyle ='none', markersize = 7, label = 'towers')
            plt.plot(track_eta[jet_id],track_phi[jet_id], 'gx', markersize = 7, label = 'tracks')
        plt.title('{} Event {}'.format(evt_type, evt_id))
        plt.xlabel('Pseudorapiditiy ($\eta$)',fontsize = 18)
        plt.ylabel('Azimuthal angle ($\phi$)', fontsize = 18)
        # plt.legend()
        plt.savefig('{}_Event_Display_{}'.format(evt_type, evt_id))
        plt.show()
        
    else:
        tower_phi = []
        tower_eta = []
        track_phi = []
        track_eta = []
        last_tower_idx = 0 
        last_track_idx = 0 #keeps track of last final index for tracks and towers
        #this for loop splits up all of the tower/track phi and eta data into different lists for each jet, compiles together as list of lists
        for jet_id in range(chain.nJets):
            jet_tower_n = chain.JetTowerN[jet_id]
            tower_phi.append(chain.JetTowerPhi[last_tower_idx:last_tower_idx+jet_tower_n])
            tower_eta.append(chain.JetTowerEta[last_tower_idx:last_tower_idx+jet_tower_n])
            last_tower_idx += jet_tower_n
            
            jet_track_n = chain.JetGhostTrackN[jet_id]
            jet_track_idxs = chain.JetGhostTrackIdx[last_track_idx:last_track_idx+jet_track_n]
            
            jet_track_eta = []
            jet_track_phi = []
            for jet_track_idx in jet_track_idxs:
                jet_track_eta.append(chain.TrackEta[jet_track_idx])
                jet_track_phi.append(chain.TrackPhi[jet_track_idx])
            track_eta.append(jet_track_eta)
            track_phi.append(jet_track_phi)
            last_track_idx += jet_track_n
            
        
        #now, looks at each jet_idx given to function on plots that jet and it's associated towers and tracks
        for jet_idx in jet_idxs:
            plot_jet_rings(chain.JetEta[jet_idx],chain.JetPhi[jet_idx], chain.TruthJetIsTautagged[jet_idx])
            plt.plot(tower_eta[jet_idx],tower_phi[jet_idx], 'bo', fillstyle ='none', markersize = 6, label = 'towers')
            plt.plot(track_eta[jet_idx],track_phi[jet_idx], 'gx', markersize = 6, label = 'tracks')
            
        #finishes plotting the graph
        plt.title('{} Event {} Jets  {}'.format(evt_type, evt_id,jet_idxs), fontsize = 18)
        plt.xlabel('Pseudorapiditiy ($\eta$)',fontsize = 18)
        plt.ylabel('Azimuthal angle ($\phi$)', fontsize = 18)
        plt.legend()
        plt.savefig('{}_Event_Display_{}_Jets_{}'.format(evt_type, evt_id,jet_idxs))
        plt.show()

def jet_display_edges(filename, evt_type, evt_id, jet_idxs):
    """
    Displays a single jet with color coded edges conntecting tracks-tracks and towers-towers, will add functionality of track-tower edges
    """
    tree_name = "output"
    chain = ROOT.TChain(tree_name, tree_name) 
    chain.Add(filename)
    chain.GetEntry(evt_id)
    ax = plt.gca() #you first need to get the axis handle
    ax.set_aspect(1)
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    tower_phi = []
    tower_eta = []
    track_phi = []
    track_eta = []
    
    edge_tower_phi = []
    edge_tower_eta = []
    edge_track_phi = []
    edge_track_eta = []
    
    last_tower_idx = 0 
    last_track_idx = 0 #keeps track of last final index for tracks and towers
    #this for loop splits up all of the tower/track phi and eta data into different lists for each jet, compiles together as list of lists
    for jet_id in range(chain.nJets):
        jet_tower_n = chain.JetTowerN[jet_id]
        jet_tower_phi = chain.JetTowerPhi[last_tower_idx:last_tower_idx+jet_tower_n]
        jet_tower_eta = chain.JetTowerEta[last_tower_idx:last_tower_idx+jet_tower_n]
        tower_phi.append(jet_tower_phi)
        tower_eta.append(jet_tower_eta)
        last_tower_idx += jet_tower_n
        
        #tower-tower edges
        jet_tower_phi_edges = list(itertools.permutations(jet_tower_phi, 2))
        jet_tower_eta_edges = list(itertools.permutations(jet_tower_eta, 2))
        edge_tower_phi.append(jet_tower_phi_edges)
        edge_tower_eta.append(jet_tower_eta_edges)
    
        jet_track_n = chain.JetGhostTrackN[jet_id]
        jet_track_idxs = chain.JetGhostTrackIdx[last_track_idx:last_track_idx+jet_track_n]

        jet_track_eta = []
        jet_track_phi = []
        for jet_track_idx in jet_track_idxs:
            jet_track_eta.append(chain.TrackEta[jet_track_idx])
            jet_track_phi.append(chain.TrackPhi[jet_track_idx])
        track_eta.append(jet_track_eta)
        track_phi.append(jet_track_phi)
        last_track_idx += jet_track_n
        
        #track-track edges
        jet_track_phi_edges = list(itertools.permutations(jet_track_phi, 2))
        jet_track_eta_edges = list(itertools.permutations(jet_track_eta, 2))
        edge_track_phi.append(jet_track_phi_edges)
        edge_track_eta.append(jet_track_eta_edges)

    #now, looks at each jet_idx given to function on plots that jet and it's associated towers and tracks
    for jet_idx in jet_idxs:
        plot_jet_rings(chain.JetEta[jet_idx],chain.JetPhi[jet_idx], chain.TruthJetIsTautagged[jet_idx])
        plt.plot(tower_eta[jet_idx],tower_phi[jet_idx], 'bo', fillstyle ='none', markersize = 6, label = 'towers')
        plt.plot(track_eta[jet_idx],track_phi[jet_idx], 'gx', markersize = 6, label = 'tracks')
        for edge_idx in range(len(edge_tower_phi[jet_idx])):
            plt.plot(edge_tower_eta[jet_idx][edge_idx],edge_tower_phi[jet_idx][edge_idx], 'b')
        for edge_idx in range(len(edge_track_phi[jet_idx])):
            plt.plot(edge_track_eta[jet_idx][edge_idx],edge_track_phi[jet_idx][edge_idx], 'g')
    #finishes plotting the graph
    plt.title('{} Event {} Jets  {}'.format(evt_type, evt_id,jet_idxs), fontsize = 18)
    plt.xlabel('Pseudorapiditiy ($\eta$)',fontsize = 18)
    plt.ylabel('Azimuthal angle ($\phi$)', fontsize = 18)
    plt.legend()
    plt.savefig('{}_Event_Display_{}_Jets_{}'.format(evt_type, evt_id,jet_idxs))
    plt.show()
    
if __name__ == "__main__":
    filename_tau = '/global/homes/j/jacobl/v0/Ntuple_ditau_processed.root'
    filename_qcd = '/global/homes/j/jacobl/v0/Ntuple_qcd_processed.root'
    filename_tau_pileup = '/global/cfs/cdirs/m3443/data/TauStudies/v2/Ntuple_ditau_PU.root'
    evt_display(filename_tau,'tau',4)