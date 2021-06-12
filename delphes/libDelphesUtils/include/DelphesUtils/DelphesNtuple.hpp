#ifndef __DELPHES_NTUPLE_H__
#define __DELPHES_NTUPLE_H__

#include "DelphesNtupleBase.hpp"
#include <vector>

class Jet;

using namespace std;
class DelphesNtuple: public DelphesNtupleBase{
  public:
  DelphesNtuple(std::string& filename);
  ~DelphesNtuple();

  void Clear();
  void Fill();

  void BookGenJets();
  void FillGenJet(Jet* jet); // Truth Jets
  void FillGenJetsCnt(int njets, int nbjets, int ntaujets);

  void BookRecoJets();
  void FillRecoJet(Jet* jet);  // Reco Jets
  void FillRecoJetCnt(int njets, int nbjets, int ntaujets);

  protected:
  // Truth Jet variables
  bool useTruthJets;
  int br_nTruthJets;
  int br_nTruthBJets;
  int br_nTruthTauJets;
  vector<double> br_truthJetPt;
  vector<double> br_truthJetEta;
  vector<double> br_truthJetPhi;
  vector<double> br_truthJetE;
  vector<int> br_truthJetIsBtagged;
  vector<int> br_truthJetIsTautagged;
  void ClearGenJets();

  // Reco Jet variables
  bool useRecoJets;
  int br_nRecoJets;
  int br_nRecoBJets;
  int br_nRecoTauJets;
  vector<double> br_recoJetPt;
  vector<double> br_recoJetEta;
  vector<double> br_recoJetPhi;
  vector<double> br_recoJetE;
  vector<int> br_recoJetIsBtagged;
  vector<int> br_recoJetIsTautagged;
  void ClearRecoJets();

};

#endif