#include "DelphesUtils/DelphesNtuple.hpp"

#include "classes/DelphesClasses.h"


DelphesNtuple::DelphesNtuple(std::string& filename):
  DelphesNtupleBase(filename){
  useTruthJets = false;
  useRecoJets = false;
}

DelphesNtuple::~DelphesNtuple(){
}

void DelphesNtuple::BookGenJets() {
  useTruthJets = true;
  tree->Branch("nTruthJets", &br_nTruthJets, "nTruthJets/I");
  tree->Branch("nTruthBJets", &br_nTruthBJets, "nTruthBJets/I");
  tree->Branch("nTruthTauJets", &br_nTruthTauJets, "nTruthTauJets/I");
  tree->Branch("truthJetPt", &br_truthJetPt);
  tree->Branch("truthJetEta", &br_truthJetEta);
  tree->Branch("truthJetPhi", &br_truthJetPhi);
  tree->Branch("truthJetE", &br_truthJetE);
  tree->Branch("truthJetIsBtagged", &br_truthJetIsBtagged);
  tree->Branch("truthJetIsTautagged", &br_truthJetIsTautagged);
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
  br_truthJetPt.clear();
  br_truthJetEta.clear();
  br_truthJetPhi.clear();
  br_truthJetE.clear();
  br_truthJetIsBtagged.clear();
  br_truthJetIsTautagged.clear();
}

void DelphesNtuple::BookRecoJets() {
  useRecoJets = true;
  tree->Branch("nRecoJets", &br_nRecoJets, "nRecoJets/I");
  tree->Branch("nRecoBJets", &br_nRecoBJets, "nRecoBJets/I");
  tree->Branch("nRecoTauJets", &br_nRecoTauJets, "nRecoTauJets/I");
  tree->Branch("recoJetPt", &br_recoJetPt);
  tree->Branch("recoJetEta", &br_recoJetEta);
  tree->Branch("recoJetPhi", &br_recoJetPhi);
  tree->Branch("recoJetE", &br_recoJetE);
  tree->Branch("recoJetIsBtagged", &br_recoJetIsBtagged);
  tree->Branch("recoJetIsTautagged", &br_recoJetIsTautagged);
}

void DelphesNtuple::FillRecoJet(Jet* jet) {
  if(!useRecoJets) BookRecoJets();
  br_recoJetPt.push_back(jet->PT);
  br_recoJetEta.push_back(jet->Eta);
  br_recoJetPhi.push_back(jet->Phi);
  br_recoJetE.push_back(jet->PT);
  br_recoJetIsBtagged.push_back(jet->BTag);
  br_recoJetIsTautagged.push_back(jet->TauTag);
}

void DelphesNtuple::FillRecoJetCnt(int njets, int nbjets, int ntaujets){
  if(!useRecoJets) BookRecoJets();
  br_nRecoJets = njets;
  br_nRecoBJets = nbjets;
  br_nRecoTauJets = ntaujets;
}

void DelphesNtuple::Clear(){
  if(useTruthJets) ClearGenJets();
  if(useRecoJets) ClearRecoJets();
}

void DelphesNtuple::Fill() {
  if(tree) tree->Fill();
}

void DelphesNtuple::ClearRecoJets() {
  br_recoJetPt.clear();
  br_recoJetEta.clear();
  br_recoJetPhi.clear();
  br_recoJetE.clear();
  br_recoJetIsBtagged.clear();
  br_recoJetIsTautagged.clear();
}
