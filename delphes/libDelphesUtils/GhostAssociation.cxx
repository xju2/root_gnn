#include "DelphesUtils/GhostAssociation.hpp"
#include "classes/DelphesClasses.h"

#include "fastjet/ClusterSequence.hh"

#include <TLorentzVector.h>
#include <TClonesArray.h>
// #include "fastjet/ClusterSequenceArea.hh"
// #include "fastjet/JetDefinition.hh"
// #include "fastjet/PseudoJet.hh"

using namespace fastjet;


vector<PseudoJet> GhostAssociation::Associate(
  TClonesArray* jets, TClonesArray* tracks, GhostAssociation::Config& config)
{
  vector<Jet*> jetContainer;
  vector<Track*> trackContainer;

  // Loop over all jets
  Int_t i;
  Jet* jet;
  Track* track;

  for(i = 0; i < jets->GetEntriesFast(); ++i) {
    jet = (Jet*) jets->At(i);
    jetContainer.push_back(jet);
  }

  for(i = 0; i < tracks->GetEntriesFast(); ++i) {
    track = (Track*) tracks->At(i);
    trackContainer.push_back(track);
  }

  return Associate(jetContainer, trackContainer, config);
}


vector<PseudoJet> GhostAssociation::Associate(
  vector<Jet*>& jets, vector<Track*>& tracks, GhostAssociation::Config& config)
{
  vector<PseudoJet> particles;
  TObject* object;
  Tower* tower;
  int j;
  for(auto jet: jets) {
    for(j = 0; j < jet->Constituents.GetEntriesFast(); ++j){
      object = jet->Constituents.At(j);
      if (object->IsA() == Tower::Class()) {
        tower = (Tower*) object;
        const auto& ltv = tower->P4();
        particles.push_back( PseudoJet(ltv.Px(), ltv.Py(), ltv.Pz(), ltv.E()) );
      }
    }
  }

  // Loop over tracks, with energy and pT being zero
  for(auto track: tracks) {
    const auto& ltv = track->P4();
    TLorentzVector ghost_ltv;
    ghost_ltv.SetPtEtaPhiM(0, ltv.Eta(), ltv.Phi(), 0);
    particles.push_back( PseudoJet(ghost_ltv.Px(), ghost_ltv.Py(), ghost_ltv.Pz(), ghost_ltv.E()) );
  }

  // choose a jet definition
  JetDefinition jet_def(config.jet_alg, config.radius);

  // run the clustering, extract the jets
  ClusterSequence sequence(particles, jet_def);
  vector<PseudoJet> jetCandidates = sorted_by_pt(sequence.inclusive_jets(config.jet_ptmin));
  return jetCandidates;
}

vector<PseudoJet> GhostAssociation::inclusive_jets(
  vector<Tower*>& towers, vector<Track*>& tracks, GhostAssociation::Config& config)
{
  vector<PseudoJet> particles;
  for(auto tower: towers) {
    const auto& ltv = tower->P4();
    PseudoJet pjet(ltv.Px(), ltv.Py(), ltv.Pz(), ltv.E());
    pjet.set_user_index(-1);
    particles.push_back(pjet);
  }
  int idx = 0;
  for(auto track: tracks) {
    const auto& ltv = track->P4();
    TLorentzVector ghost_ltv;
    ghost_ltv.SetPtEtaPhiM(0.00001, ltv.Eta(), ltv.Phi(), 0);
    PseudoJet pjet(ghost_ltv.Px(), ghost_ltv.Py(), ghost_ltv.Pz(), ghost_ltv.E());
    pjet.set_user_index(idx++);
    particles.push_back(pjet);
  }

  // choose a jet definition
  JetDefinition jet_def(config.jet_alg, config.radius);

  // run the clustering, extract the jets
  ClusterSequence sequence(particles, jet_def);
  vector<PseudoJet> jetCandidates = sorted_by_pt(sequence.inclusive_jets(config.jet_ptmin));
  // vector<PseudoJet> jetCandidates = sorted_by_pt(sequence.inclusive_jets());

  printf("Jets from towers and tracks: %ld\n", jetCandidates.size());
  unsigned k;
  int j;
  for (k = 0; k < jetCandidates.size(); k++) {
    auto& pseudo_jet = jetCandidates[k];
    printf("Jets with Ghosts: %d, %.2f %.2f %.2f\n",
      k, pseudo_jet.pt(), pseudo_jet.eta(), pseudo_jet.phi_std());
    
    // loop constituents
    j = 0;
    for (const auto& jet_comps: pseudo_jet.constituents()) {
      printf("\t constituent %d: idx %d, pT %.2f\n", j++, jet_comps.user_index(), jet_comps.pt());
    }
  }

  return jetCandidates;
}


vector<int> GhostAssociation::Associate(
  Jet* jet, vector<Track*>& tracks, GhostAssociation::Config& config)
{

  Tower* tower;
  int j;
  TObject* object;

  vector<PseudoJet> particles;
  for(j = 0; j < jet->Constituents.GetEntriesFast(); ++j) {
    object = jet->Constituents.At(j);
    if (object->IsA() == Tower::Class()) {
      tower = (Tower*) object;
      const auto& ltv = tower->P4();
      PseudoJet pjet(ltv.Px(), ltv.Py(), ltv.Pz(), ltv.E());
      pjet.set_user_index(-1);
      particles.push_back(pjet);
    }
  }

  int idx = 0;
  for(auto track: tracks) {
    const auto& ltv = track->P4();
    TLorentzVector ghost_ltv;
    ghost_ltv.SetPtEtaPhiM(0.00001, ltv.Eta(), ltv.Phi(), 0);
    PseudoJet pjet(ghost_ltv.Px(), ghost_ltv.Py(), ghost_ltv.Pz(), ghost_ltv.E());
    pjet.set_user_index(idx++);
    particles.push_back(pjet);
  }

  // choose a jet definition
  JetDefinition jet_def(config.jet_alg, config.radius);

  // run the clustering, extract the jets
  ClusterSequence sequence(particles, jet_def);
  vector<PseudoJet> jetCandidates = sorted_by_pt(sequence.inclusive_jets(config.jet_ptmin));

  unsigned k;
  vector<int> trackIdx;
  trackIdx.clear();
  for (k = 0; k < jetCandidates.size(); k++) {
    auto& pseudo_jet = jetCandidates[k];
    // loop constituents
    j = 0;
    for (const auto& jet_comps: pseudo_jet.constituents()) {
      if(jet_comps.user_index() < 0) continue;
      trackIdx.push_back(jet_comps.user_index());
    }
  }
  return trackIdx;
}