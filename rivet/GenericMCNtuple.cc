// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/DressedLeptons.hh"
#include "Rivet/Projections/MissingMomentum.hh"
#include "Rivet/Projections/PromptFinalState.hh"
#include "Rivet/Projections/VetoedFinalState.hh"

#include "Rivet/Projections/SmearedParticles.hh"
#include "Rivet/Projections/SmearedJets.hh"
#include "Rivet/Projections/SmearedMET.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Tools/SmearingFunctions.hh"

#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <unistd.h>

using namespace std;

namespace Rivet {


  /// @brief the top quark matching is inspired by 
  /// the charge asymmetry measurement in top quark pair production
  /// in dilepton channel
  /// https://rivet.hepforge.org/analyses/ATLAS_2016_I1449082.html
  class GenericMCNtuple : public Analysis {
  public:

    const double MW = 80.300*GeV;
    const double MTOP = 172.5*GeV;

    /// Constructor
    DEFAULT_RIVET_ANALYSIS_CTOR(GenericMCNtuple);


    /// @name Analysis methods
    //@{

    /// Book branches and initialise projections before the run
    void init() {

      // Cuts
      const Cut eta_full = Cuts::abseta < 4.9;
      const Cut electron_cuts = Cuts::abseta < 2.47 && Cuts::pT > 20*GeV;
      const Cut muon_cuts = Cuts::abseta < 2.5 && Cuts::pT > 10*GeV;

      // All final state particles
      FinalState fs(eta_full);
      // Get photons to dress leptons
      IdentifiedFinalState photons(fs, PID::PHOTON);
      // photon smearing function is not implemented in Rivet.
      SmearedParticles recoPhotons(photons, PHOTON_EFF_ATLAS_RUN1, PHOTON_SMEAR_ATLAS_RUN1);

      // Electron projections
      // ---------------------
      IdentifiedFinalState electrons(electron_cuts);
      electrons.acceptIdPair(PID::ELECTRON);
      declare(electrons, "TruthElectron");
      // Electron smearing
      SmearedParticles recoEle(electrons, ELECTRON_EFF_ATLAS_RUN1_MEDIUM, ELECTRON_SMEAR_ATLAS_RUN1);
      declare(recoEle, "RecoElectron");

      // Muon projections
      // ---------------------
      IdentifiedFinalState muons(muon_cuts);
      muons.acceptIdPair(PID::MUON);
      declare(muons, "TruthMuon");
      // Muon smearing
      SmearedParticles recoMuon(muons, MUON_EFF_ATLAS_RUN1, MUON_SMEAR_ATLAS_RUN1);
      declare(recoMuon, "RecoMuon");

      // Missing ET projections
      // ---------------------
      VisibleFinalState calofs(eta_full);
      MissingMomentum met(calofs);
      declare(met, "TruthMET");
      SmearedMET recoMET(met, MET_SMEAR_ATLAS_RUN1);
      declare(recoMET, "RecoMET");  

      // Jets projections
      // ---------------------
      FastJets jets(FinalState(eta_full), FastJets::ANTIKT, 0.4);
      declare(jets, "TruthJet");
      SmearedJets recojets(jets, JET_SMEAR_ATLAS_RUN1);
      declare(recojets, "RecoJet");

      // Prepare ROOT file and Book branches
      char buffer[512];
      sprintf(buffer, "GenericMCNtuple_%d.root", getpid());
      string out_filename(buffer);
      file_handle = TFile::Open(out_filename.c_str(), "RECREATE");
      tree = new TTree("output", "GenericMCNtuple output");
      bookBranch();
    }
    void bookBranch(){
      tree->Branch("nTruthEle", &br_nTruthEle, "nTruthEle/I");
      tree->Branch("truthElePt", &br_truthElePt);
      tree->Branch("truthEleEta", &br_truthEleEta);
      tree->Branch("truthElePhi", &br_truthElePhi);
      tree->Branch("truthEleE", &br_truthEleE);

      tree->Branch("nTruthMuon", &br_nTruthMuon, "nTruthMuon/I");
      tree->Branch("truthMuonPt", &br_truthMuonPt);
      tree->Branch("truthMuonEta", &br_truthMuonEta);
      tree->Branch("truthMuonPhi", &br_truthMuonPhi);
      tree->Branch("truthMuonE", &br_truthMuonE);

      tree->Branch("nTruthJet", &br_nTruthJet, "nTruthJet/I");
      tree->Branch("truthJetPt", &br_truthJetPt);
      tree->Branch("truthJetEta", &br_truthJetEta);
      tree->Branch("truthJetPhi", &br_truthJetPhi);
      tree->Branch("truthJetE", &br_truthJetE);

      tree->Branch("truthMETPt", &br_truthMETPt);
      tree->Branch("truthMETPhi", &br_truthMETPhi);

      tree->Branch("nRecoEle", &br_nRecoEle, "nRecoEle/I");
      tree->Branch("recoElePt", &br_recoElePt);
      tree->Branch("recoEleEta", &br_recoEleEta);
      tree->Branch("recoElePhi", &br_recoElePhi);
      tree->Branch("recoEleE", &br_recoEleE);

      tree->Branch("nRecoMuon", &br_nRecoMuon, "nRecoMuon/I");
      tree->Branch("recoMuonPt", &br_recoMuonPt);
      tree->Branch("recoMuonEta", &br_recoMuonEta);
      tree->Branch("recoMuonPhi", &br_recoMuonPhi);
      tree->Branch("recoMuonE", &br_recoMuonE);

      tree->Branch("nRecoJet", &br_nRecoJet, "nRecoJet/I");
      tree->Branch("recoJetPt", &br_recoJetPt);
      tree->Branch("recoJetEta", &br_recoJetEta);
      tree->Branch("recoJetPhi", &br_recoJetPhi);
      tree->Branch("recoJetE", &br_recoJetE);

      tree->Branch("recoMETPt", &br_recoMETPt);
      tree->Branch("recoMETPhi", &br_recoMETPhi);
    }

    void clearBranch(){
      br_truthElePt.clear();
      br_truthEleEta.clear();
      br_truthElePhi.clear();
      br_truthEleE.clear();

      br_truthMuonPt.clear();
      br_truthMuonEta.clear();
      br_truthMuonPhi.clear();
      br_truthMuonE.clear();

      br_truthJetPt.clear();
      br_truthJetEta.clear();
      br_truthJetPhi.clear();
      br_truthJetE.clear();

      br_recoElePt.clear();
      br_recoEleEta.clear();
      br_recoElePhi.clear();
      br_recoEleE.clear();

      br_recoMuonPt.clear();
      br_recoMuonEta.clear();
      br_recoMuonPhi.clear();
      br_recoMuonE.clear();

      br_recoJetPt.clear();
      br_recoJetEta.clear();
      br_recoJetPhi.clear();
      br_recoJetE.clear();
    }

    /// Perform the per-event analysis
    void analyze(const Event& event) {
      clearBranch();

      // Truth studies
      const Jets truthJets  = apply<JetAlg>(event, "TruthJet").jetsByPt(
        Cuts::pT > 30*GeV && Cuts::abseta < 2.8);
      
      const Particles truthEle = apply<IdentifiedFinalState>(event, "TruthElectron").particlesByPt();
      const Particles truthMuon = apply<IdentifiedFinalState>(event, "TruthMuon").particlesByPt();
      
      // Discard jets very close to electrons (dR < 0.2), 
      // or with low track multiplicity and close to muons (dR < 0.4)
      const Jets isoTruthJet = filter_discard(truthJets, [&](const Jet& j){
          if (any(truthEle, deltaRLess(j, 0.2))) return true;
 				  if (j.particles(Cuts::abscharge > 0 && Cuts::pT > 0.4*GeV).size() <= 3 && \
					  any(truthMuon, deltaRLess(j, 0.4))) return true;
				  return false;
        }
      );
      br_nTruthJet = (int) isoTruthJet.size();
      for(const Jet& j: isoTruthJet){
        const FourMomentum tlv = j.momentum();
        br_truthJetPt.push_back(tlv.pt()/GeV);
        br_truthJetE.push_back(tlv.E()/GeV);
        br_truthJetEta.push_back(tlv.eta());
        br_truthJetPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
      }

      // Discard electrons close to remaining jets (dR < 0.4)
      const Particles isoTruthEle = filter_discard(
        truthEle, [&](const Particle& e) {return any(isoTruthJet, deltaRLess(e, 0.4)); }
      );
      br_nTruthEle = (int) isoTruthEle.size();
      for(const Particle& ele: isoTruthEle){
        const FourMomentum tlv = ele.momentum();
        br_truthElePt.push_back(tlv.pt()/GeV);
        br_truthEleE.push_back(tlv.E()/GeV);
        br_truthEleEta.push_back(tlv.eta());
        br_truthElePhi.push_back(tlv.phi(MINUSPI_PLUSPI));
      }

      // Discard muons close to remaining jets (dR < 0.4)
      const Particles isoTruthMuon = filter_discard(truthMuon, [&](const Particle& m) {
				for (const Jet& j : isoTruthJet) {
					if (deltaR(j,m) > 0.4) continue;
					if (j.particles(Cuts::abscharge > 0 && Cuts::pT > 0.4*GeV).size() > 3) return true;
				}
				return false;
      });
      br_nTruthMuon = (int) isoTruthMuon.size();
      for(const Particle& muon: isoTruthMuon){
        const FourMomentum tlv = muon.momentum();
        br_truthMuonPt.push_back(tlv.pt()/GeV);
        br_truthMuonE.push_back(tlv.E()/GeV);
        br_truthMuonEta.push_back(tlv.eta());
        br_truthMuonPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
      }

      // ET miss
      const Vector3& truthMET = apply<MissingMomentum>(event, "TruthMET").vectorMissingEt();
      br_truthMETPt = truthMET.mod()/GeV;
      br_truthMETPhi = truthMET.phi(MINUSPI_PLUSPI);

      // Top quark matching

      if (tree) tree->Fill();
    }


    /// Normalise histograms etc., after the run
    void finalize() {
      if (file_handle && tree) {
        file_handle->cd();
        tree->Write();
        file_handle->Close();
        tree = nullptr;
        file_handle = nullptr;
      }
    }

    //@}


    TFile* file_handle;
    TTree* tree;
    // truth info
    int br_nTruthEle;
    vector<double> br_truthElePt;
    vector<double> br_truthEleEta;
    vector<double> br_truthElePhi;
    vector<double> br_truthEleE;

    int br_nTruthMuon;
    vector<double> br_truthMuonPt;
    vector<double> br_truthMuonEta;
    vector<double> br_truthMuonPhi;
    vector<double> br_truthMuonE;

    int br_nTruthJet;
    vector<double> br_truthJetPt;
    vector<double> br_truthJetEta;
    vector<double> br_truthJetPhi;
    vector<double> br_truthJetE;

    double br_truthMETPt;
    double br_truthMETPhi;

    // reco info
    int br_nRecoEle;
    vector<double> br_recoElePt;
    vector<double> br_recoEleEta;
    vector<double> br_recoElePhi;
    vector<double> br_recoEleE;

    int br_nRecoMuon;
    vector<double> br_recoMuonPt;
    vector<double> br_recoMuonEta;
    vector<double> br_recoMuonPhi;
    vector<double> br_recoMuonE;

    int br_nRecoJet;
    vector<double> br_recoJetPt;
    vector<double> br_recoJetEta;
    vector<double> br_recoJetPhi;
    vector<double> br_recoJetE;

    double br_recoMETPt;
    double br_recoMETPhi;

    // @TODO: To add top Matching Info
    int br_nTopMatched;

  };


  DECLARE_RIVET_PLUGIN(GenericMCNtuple);

}
