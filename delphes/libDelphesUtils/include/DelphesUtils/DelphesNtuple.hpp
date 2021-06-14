#ifndef __DELPHES_NTUPLE_H__
#define __DELPHES_NTUPLE_H__

#include "DelphesNtupleBase.hpp"
#include <vector>

class Jet;
class Track;
class Tower;

using namespace std;
class DelphesNtuple: public DelphesNtupleBase{
  public:
  DelphesNtuple(std::string& filename);
  ~DelphesNtuple();

  void Clear();
  void Fill();

  // Truth jets
  void BookGenJets();
  void FillGenJet(Jet* jet);
  void FillGenJetsCnt(int njets, int nbjets, int ntaujets);

  // Reco Jets
  void BookRecoJets(bool withTowers=false);
  void FillRecoJet(Jet* jet);  // Reco Jets
  void FillRecoJetCnt(int njets, int nbjets, int ntaujets);

  // Towers associated with Reco jets
  void BookJetTowers();
  void FillJetTower(Tower* tower);

  // Tracks
  void BookTracks();
  void FillTrack(Track* track);
  
  // Towers
  void BookTowers();
  void FillTower(Tower* tower);

  protected:
  // Truth Jet variables
  bool useTruthJets;
  int br_nTruthJets;
  int br_nTruthBJets;
  int br_nTruthTauJets;
  vector<float> br_truthJetPt;
  vector<float> br_truthJetEta;
  vector<float> br_truthJetPhi;
  vector<float> br_truthJetE;
  vector<int> br_truthJetIsBtagged;
  vector<int> br_truthJetIsTautagged;
  void ClearGenJets();

  // Reco Jet variables
  bool useRecoJets;
  int br_nRecoJets;
  int br_nRecoBJets;
  int br_nRecoTauJets;
  vector<float> br_recoJetPt;
  vector<float> br_recoJetEta;
  vector<float> br_recoJetPhi;
  vector<float> br_recoJetE;
  vector<int> br_recoJetIsBtagged;
  vector<int> br_recoJetIsTautagged;
  vector<int> br_recoJetNTowers;
  vector<int> br_recoJetNTracks;
  void ClearRecoJets();

  // Towers associated with Jets
  bool useJetTowers;
  vector<float> br_jetTowerEt;
  vector<float> br_jetTowerEta;
  vector<float> br_jetTowerPhi;
  vector<float> br_jetTowerE;
  vector<float> br_jetTowerEem;
  vector<float> br_jetTowerEhad;
  void ClearJetTower();

  // Tracks
  bool useTracks;
  int br_nTracks;
  vector<int> br_trackPID;
  vector<int> br_trackCharge;
  vector<float> br_trackEtaOut;
  vector<float> br_trackPhiOut;
  vector<float> br_trackPt;
  vector<float> br_trackEta;
  vector<float> br_trackPhi;
  vector<float> br_trackD0;
  vector<float> br_trackZ0;
  void ClearTracks();

  // Towers
  bool useTowers;
  int br_nTowers;
  vector<float> br_towerEt;
  vector<float> br_towerEta;
  vector<float> br_towerPhi;
  vector<float> br_towerE;
  vector<float> br_towerEem;
  vector<float> br_towerEhad;
  void ClearTowers();

};

#endif