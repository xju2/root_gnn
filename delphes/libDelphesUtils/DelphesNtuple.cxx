#include "DelphesUtils/DelphesNtuple.hpp"

#include "classes/DelphesClasses.h"


DelphesNtuple::DelphesNtuple(std::string& filename):
  DelphesNtupleBase(filename){
  useTruthJets = false;
  useRecoJets = false;
  useJetTowers = false;
  useTracks = false;
  useTowers = false;
  useJetGhostTracks = false;
}

DelphesNtuple::~DelphesNtuple(){
}

void DelphesNtuple::BookGenJets() {
  useTruthJets = true;
  tree->Branch("nTruthJets", &br_nTruthJets, "nTruthJets/I");
  tree->Branch("nTruthBJets", &br_nTruthBJets, "nTruthBJets/I");
  tree->Branch("nTruthTauJets", &br_nTruthTauJets, "nTruthTauJets/I");
  tree->Branch("TruthJetPt", &br_truthJetPt);
  tree->Branch("TruthJetEta", &br_truthJetEta);
  tree->Branch("TruthJetPhi", &br_truthJetPhi);
  tree->Branch("TruthJetE", &br_truthJetE);
  tree->Branch("TruthJetIsBtagged", &br_truthJetIsBtagged);
  tree->Branch("TruthJetIsTautagged", &br_truthJetIsTautagged);
}

void DelphesNtuple::FillGenJet(Jet* jet) {
  if(!useTruthJets) BookGenJets();
  br_truthJetPt.push_back(jet->PT);
  br_truthJetEta.push_back(jet->Eta);
  br_truthJetPhi.push_back(jet->Phi);
  br_truthJetE.push_back(jet->PT);
  br_truthJetIsBtagged.push_back(jet->BTag);
  br_truthJetIsTautagged.push_back(jet->TauTag);  
}

void DelphesNtuple::FillGenJetsCnt(int njets, int nbjets, int ntaujets){
  if(!useTruthJets) BookGenJets();
  br_nTruthJets = njets;
  br_nTruthBJets = nbjets;
  br_nTruthTauJets = ntaujets;
}

void DelphesNtuple::ClearGenJets() {
  if(!useTruthJets) BookGenJets();
  br_truthJetPt.clear();
  br_truthJetEta.clear();
  br_truthJetPhi.clear();
  br_truthJetE.clear();
  br_truthJetIsBtagged.clear();
  br_truthJetIsTautagged.clear();
}

void DelphesNtuple::BookRecoJets(bool withTowers) {
  useRecoJets = true;
  tree->Branch("nJets", &br_nRecoJets, "nJets/I");
  tree->Branch("nBJets", &br_nRecoBJets, "nBJets/I");
  tree->Branch("nTauJets", &br_nRecoTauJets, "nTauJets/I");
  tree->Branch("JetPt", &br_recoJetPt);
  tree->Branch("JetEta", &br_recoJetEta);
  tree->Branch("JetPhi", &br_recoJetPhi);
  tree->Branch("JetE", &br_recoJetE);
  tree->Branch("JetIsBtagged", &br_recoJetIsBtagged);
  tree->Branch("JetIsTautagged", &br_recoJetIsTautagged);
  tree->Branch("JetTowerN", &br_recoJetNTowers);
  tree->Branch("JetTrackN", &br_recoJetNTracks);

  if(withTowers) BookJetTowers();
}

void DelphesNtuple::FillRecoJet(Jet* jet) {
  if(!useRecoJets) BookRecoJets();
  br_recoJetPt.push_back(jet->PT);
  br_recoJetEta.push_back(jet->Eta);
  br_recoJetPhi.push_back(jet->Phi);
  br_recoJetE.push_back(jet->PT);
  br_recoJetIsBtagged.push_back(jet->BTag);
  br_recoJetIsTautagged.push_back(jet->TauTag);
  // number of towers associated with the jet
  int jetTowerN = 0;
  int jetTracksN = 0;
  Int_t j;
  TObject *object;
  Tower* tower;
  Track* track;
  for(j = 0; j < jet->Constituents.GetEntriesFast(); ++j) {
    object = jet->Constituents.At(j);
    if (object == 0) continue;
    if (object->IsA() == Tower::Class()){
      jetTowerN ++;
      tower = (Tower*) object;
      if(useJetTowers) FillJetTower(tower);
    } else if (object->IsA() == Track::Class()) {
      jetTracksN ++;
    } else {}
  }
  br_recoJetNTowers.push_back(jetTowerN);
  br_recoJetNTracks.push_back(jetTracksN);
}

void DelphesNtuple::FillRecoJetCnt(int njets, int nbjets, int ntaujets){
  if(!useRecoJets) BookRecoJets();
  br_nRecoJets = njets;
  br_nRecoBJets = nbjets;
  br_nRecoTauJets = ntaujets;
}

void DelphesNtuple::BookGhostTracks() {
  useJetGhostTracks = true;
  tree->Branch("JetGhostTrackN", &br_recoJetGhostTracksN);
  tree->Branch("JetGhostTrackIdx", &br_jetGhostTrackIdx);
}

void DelphesNtuple::ClearGhostTracks() {
  br_recoJetGhostTracksN.clear();
  br_jetGhostTrackIdx.clear();
}

void DelphesNtuple::FillRecoJetGhostTracks(vector<int>& trackIdx){
  if(!useJetGhostTracks) BookGhostTracks();

  if (trackIdx.size() < 1) {
    br_recoJetGhostTracksN.push_back(0);
    return;
  }

  int cnt = 0;
  for(auto idx: trackIdx) {
    if(idx < 0) continue;
    cnt ++;
    br_jetGhostTrackIdx.push_back(idx);
  }
  br_recoJetGhostTracksN.push_back(cnt);
}


void DelphesNtuple::ClearRecoJets() {
  if(!useRecoJets) BookRecoJets();
  br_recoJetPt.clear();
  br_recoJetEta.clear();
  br_recoJetPhi.clear();
  br_recoJetE.clear();
  br_recoJetIsBtagged.clear();
  br_recoJetIsTautagged.clear();
  br_recoJetNTowers.clear();
  br_recoJetNTracks.clear();
}

void DelphesNtuple::BookJetTowers() {
  useJetTowers = true;
  tree->Branch("JetTowerEt",   &br_jetTowerEt);
  tree->Branch("JetTowerEta",  &br_jetTowerEta);
  tree->Branch("JetTowerPhi",  &br_jetTowerPhi);
  tree->Branch("JetTowerE",    &br_jetTowerE);
  tree->Branch("JetTowerEem",  &br_jetTowerEem);
  tree->Branch("JetTowerEhad", &br_jetTowerEhad);
}

void DelphesNtuple::FillJetTower(Tower* tower) {
  if(!useJetTowers) BookJetTowers();
  br_jetTowerEt.push_back(tower->ET);
  br_jetTowerEta.push_back(tower->Eta);
  br_jetTowerPhi.push_back(tower->Phi);
  br_jetTowerE.push_back(tower->E);
  br_jetTowerEem.push_back(tower->Eem);
  br_jetTowerEhad.push_back(tower->Ehad);  
}

void DelphesNtuple::ClearJetTower() {
  if(!useJetTowers) BookJetTowers();
  br_jetTowerEt.clear();
  br_jetTowerEta.clear();
  br_jetTowerPhi.clear();
  br_jetTowerE.clear();
  br_jetTowerEem.clear();
  br_jetTowerEhad.clear();  
}

void DelphesNtuple::BookTracks() {
  useTracks = true;
  tree->Branch("nTracks", &br_nTracks, "nTracks/I");
  tree->Branch("TracKPID",    &br_trackPID);
  tree->Branch("TrackCharge", &br_trackCharge);
  tree->Branch("TrackEtaOuter", &br_trackEtaOut);
  tree->Branch("TrackPhiOuter", &br_trackPhiOut);
  tree->Branch("TrackPt",     &br_trackPt);
  tree->Branch("TrackEta",    &br_trackEta);
  tree->Branch("TrackPhi",    &br_trackPhi);
  tree->Branch("TrackD0",     &br_trackD0);
  tree->Branch("TrackZ0",     &br_trackZ0);
}

void DelphesNtuple::FillTrack(Track* track) {
  if(!useTracks) BookTracks();
  br_trackPID.push_back(track->PID);
  br_trackCharge.push_back(track->Charge);
  br_trackEtaOut.push_back(track->EtaOuter);
  br_trackPhiOut.push_back(track->PhiOuter);
  br_trackPt.push_back(track->PT);
  br_trackEta.push_back(track->Eta);
  br_trackPhi.push_back(track->Phi);
  br_trackD0.push_back(track->D0);
  br_trackZ0.push_back(track->DZ);
}

void DelphesNtuple::ClearTracks() {
  if(!useTracks) BookTracks();
  br_trackPID.clear();
  br_trackCharge.clear();
  br_trackEtaOut.clear();
  br_trackPhiOut.clear();
  br_trackPt.clear();
  br_trackEta.clear();
  br_trackPhi.clear();
  br_trackD0.clear();
  br_trackZ0.clear();
}

void DelphesNtuple::BookTowers() {
  useTowers = true;
  tree->Branch("nTowers", &br_nTowers, "nTowers/I");
  tree->Branch("TowerEt",   &br_towerEt);
  tree->Branch("TowerEta",  &br_towerEta);
  tree->Branch("TowerPhi",  &br_towerPhi);
  tree->Branch("TowerE",    &br_towerE);
  tree->Branch("TowerEem",  &br_towerEem);
  tree->Branch("TowerEhad", &br_towerEhad);
}

void DelphesNtuple::FillTower(Tower* tower) {
  if(!useTowers) BookTowers();
  br_towerEt.push_back(tower->ET);
  br_towerEta.push_back(tower->Eta);
  br_towerPhi.push_back(tower->Phi);
  br_towerE.push_back(tower->E);
  br_towerEem.push_back(tower->Eem);
  br_towerEhad.push_back(tower->Ehad);
}

void DelphesNtuple::ClearTowers() {
  br_towerEt.clear();
  br_towerEta.clear();
  br_towerPhi.clear();
  br_towerE.clear();
  br_towerEem.clear();
  br_towerEhad.clear();
}

void DelphesNtuple::BookTaus() {
  useTruthTaus = true;
  tree->Branch("truthTauN", &br_nTruthTaus, "truthTauN/I");
  tree->Branch("truthTauEt",   &br_truthTauEt);
  tree->Branch("truthTauEta",  &br_truthTauEta);
  tree->Branch("truthTauPhi",  &br_truthTauPhi);
  tree->Branch("truthTauE",    &br_truthTauE);
  tree->Branch("truthTauCharge",    &br_truthTauCharge);
}

void DelphesNtuple::FillTau(GenParticle* particle) {
  if(!useTruthTaus) BookTaus();
  br_truthTauEt.push_back(particle->PT);
  br_truthTauEta.push_back(particle->Eta);
  br_truthTauPhi.push_back(particle->Phi);
  br_truthTauE.push_back(particle->E);
  br_truthTauCharge.push_back(particle->Charge);
}

void DelphesNtuple::ClearTruthTaus() {
  br_truthTauEt.clear();
  br_truthTauEta.clear();
  br_truthTauPhi.clear();
  br_truthTauE.clear();  
  br_truthTauCharge.clear();
}

void DelphesNtuple::Clear(){
  if(useTruthJets)  ClearGenJets();
  if(useRecoJets)   ClearRecoJets();
  if(useJetTowers)  ClearJetTower();
  if(useTracks)     ClearTracks();
  if(useTowers)     ClearTowers();
  if(useJetGhostTracks) ClearGhostTracks();
  if(useTruthTaus)  ClearTruthTaus();
}

void DelphesNtuple::Fill() {
  if(tree) {
    if(useTracks) br_nTracks = (int) br_trackEta.size();
    if(useTowers) br_nTowers = (int) br_towerEta.size();
    if(useTruthTaus)  br_nTruthTaus = (int) br_truthTauEta.size();
    tree->Fill();
  }
}


