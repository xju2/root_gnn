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

#include <algorithm>
using namespace std;

namespace Rivet {


  /// @brief This analysis applies pre-selections on electrons, muons and jets,
  /// then save four-memotum information for each object.
  /// The overlap removeal was done in a ATLAS way.
  /// In addition, the top quark matching is inspired by 
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
      const Cut muon_cuts = Cuts::abseta < 2.5 && Cuts::pT > 20*GeV;
      const Cut ph_cuts = Cuts::abseta < 2.47 && Cuts::pT > 20*GeV;

      // All final state particles
      FinalState fs(eta_full);
      // Get photons to dress leptons
      IdentifiedFinalState photons(ph_cuts);
      photons.acceptIdPair(PID::PHOTON);
      declare(photons, "TruthPhoton");
      // photon smearing function is not implemented in Rivet.
      SmearedParticles recoPhotons(photons, PHOTON_EFF_ATLAS_RUN1, PHOTON_SMEAR_ATLAS_RUN1);
      declare(recoPhotons, "RecoPhoton");

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

      // Neutrino projection
      // ----------------------
      IdentifiedFinalState nu_id;
      nu_id.acceptNeutrinos();
      PromptFinalState neutrinos(nu_id);
      neutrinos.acceptTauDecays(true);
      declare(neutrinos, "TruthNeutrinos");

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
      truthJetPTCut = 20.;
      recoJetPTCut = 25.;
      bookBranch();
    }
    void bookBranch(){
      
      tree->Branch("nTruthJet", &br_nTruthJet, "nTruthJet/I");
      tree->Branch("nTruthBJet", &br_nTruthBJet, "nTruthBJet/I");
      tree->Branch("truthJetPt", &br_truthJetPt);
      tree->Branch("truthJetEta", &br_truthJetEta);
      tree->Branch("truthJetPhi", &br_truthJetPhi);
      tree->Branch("truthJetE", &br_truthJetE);
      tree->Branch("truthJetIsBtagged", &br_truthJetIsBtagged);

      tree->Branch("nTruthEle", &br_nTruthEle, "nTruthEle/I");
      tree->Branch("truthElePt", &br_truthElePt);
      tree->Branch("truthEleEta", &br_truthEleEta);
      tree->Branch("truthElePhi", &br_truthElePhi);
      tree->Branch("truthEleE", &br_truthEleE);
      tree->Branch("truthEleCharge", &br_truthEleCharge);

      tree->Branch("nTruthPhoton", &br_nTruthPhoton, "nTruthPhoton/I");
      tree->Branch("truthPhotonPt", &br_truthPhotonPt);
      tree->Branch("truthPhotonEta", &br_truthPhotonEta);
      tree->Branch("truthPhotonPhi", &br_truthPhotonPhi);
      tree->Branch("truthPhotonE", &br_truthPhotonE);
      
      tree->Branch("nTruthMuon", &br_nTruthMuon, "nTruthMuon/I");
      tree->Branch("truthMuonPt", &br_truthMuonPt);
      tree->Branch("truthMuonEta", &br_truthMuonEta);
      tree->Branch("truthMuonPhi", &br_truthMuonPhi);
      tree->Branch("truthMuonE", &br_truthMuonE);
      tree->Branch("truthMuonCharge", &br_truthMuonCharge);

      tree->Branch("nTruthNeutrino", &br_nTruthNeutrino, "nTruthNeutrino/I");
      tree->Branch("truthNeutrinoPt", &br_truthNeutrinoPt);
      tree->Branch("truthNeutrinoEta", &br_truthNeutrinoEta);
      tree->Branch("truthNeutrinoPhi", &br_truthNeutrinoPhi);
      tree->Branch("truthNeutrinoE", &br_truthNeutrinoE);
      tree->Branch("truthNeutrinoPID", &br_truthNeutrinoPID);
      
      tree->Branch("nTruthLepton", &br_nTruthLepton, "nTruthLepton/I");

      tree->Branch("truthMETPt", &br_truthMETPt);
      tree->Branch("truthMETPhi", &br_truthMETPhi);

      tree->Branch("nRecoEle", &br_nRecoEle, "nRecoEle/I");
      tree->Branch("recoElePt", &br_recoElePt);
      tree->Branch("recoEleEta", &br_recoEleEta);
      tree->Branch("recoElePhi", &br_recoElePhi);
      tree->Branch("recoEleE", &br_recoEleE);
      tree->Branch("recoEleCharge", &br_recoEleCharge);

      tree->Branch("nRecoPhoton", &br_nRecoPhoton, "nRecoPhoton/I");
      tree->Branch("recoPhotonPt", &br_recoPhotonPt);
      tree->Branch("recoPhotonEta", &br_recoPhotonEta);
      tree->Branch("recoPhotonPhi", &br_recoPhotonPhi);
      tree->Branch("recoPhotonE", &br_recoPhotonE);

      tree->Branch("nRecoMuon", &br_nRecoMuon, "nRecoMuon/I");
      tree->Branch("recoMuonPt", &br_recoMuonPt);
      tree->Branch("recoMuonEta", &br_recoMuonEta);
      tree->Branch("recoMuonPhi", &br_recoMuonPhi);
      tree->Branch("recoMuonE", &br_recoMuonE);
      tree->Branch("recoMuonCharge", &br_recoMuonCharge);

      tree->Branch("nRecoLepton", &br_nRecoLepton, "nRecoLepton/I");

      tree->Branch("nRecoJet", &br_nRecoJet, "nRecoJet/I");
      tree->Branch("recoJetPt", &br_recoJetPt);
      tree->Branch("recoJetEta", &br_recoJetEta);
      tree->Branch("recoJetPhi", &br_recoJetPhi);
      tree->Branch("recoJetE", &br_recoJetE);
      tree->Branch("recoJetIsBtagged", &br_recoJetIsBtagged);

      tree->Branch("recoMETPt", &br_recoMETPt);
      tree->Branch("recoMETPhi", &br_recoMETPhi);
    }

    void clearBranch(){
      br_truthElePt.clear();
      br_truthEleEta.clear();
      br_truthElePhi.clear();
      br_truthEleE.clear();
      br_truthEleCharge.clear();

      br_truthPhotonPt.clear();
      br_truthPhotonEta.clear();
      br_truthPhotonPhi.clear();
      br_truthPhotonE.clear();

      br_truthMuonPt.clear();
      br_truthMuonEta.clear();
      br_truthMuonPhi.clear();
      br_truthMuonE.clear();
      br_truthMuonCharge.clear();

      br_truthNeutrinoPt.clear();
      br_truthNeutrinoEta.clear();
      br_truthNeutrinoPhi.clear();
      br_truthNeutrinoE.clear();
      br_truthNeutrinoPID.clear();

      br_truthJetPt.clear();
      br_truthJetEta.clear();
      br_truthJetPhi.clear();
      br_truthJetE.clear();
      br_truthJetIsBtagged.clear();

      br_recoElePt.clear();
      br_recoEleEta.clear();
      br_recoElePhi.clear();
      br_recoEleE.clear();
      br_recoEleCharge.clear();

      br_recoPhotonPt.clear();
      br_recoPhotonEta.clear();
      br_recoPhotonPhi.clear();
      br_recoPhotonE.clear();

      br_recoMuonPt.clear();
      br_recoMuonEta.clear();
      br_recoMuonPhi.clear();
      br_recoMuonE.clear();
      br_recoMuonCharge.clear();

      br_recoJetPt.clear();
      br_recoJetEta.clear();
      br_recoJetPhi.clear();
      br_recoJetE.clear();
      br_recoJetIsBtagged.clear();
    }

    /// Perform the per-event analysis
    void analyze(const Event& event) {
      clearBranch();
      // const double PHV = -999999.; // place holder value

      // Truth studies
      // ---------------------
      const Jets truthJets  = apply<JetAlg>(event, "TruthJet").jetsByPt(
        Cuts::pT > truthJetPTCut*GeV && Cuts::abseta < 2.8);
      
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
        br_truthJetIsBtagged.push_back((int) j.bTagged(Cuts::pT > 5*GeV));
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
        br_truthEleCharge.push_back((int) ele.charge());
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
        br_truthMuonCharge.push_back((int) muon.charge());
      }
      br_nTruthLepton = br_nTruthEle + br_nTruthMuon;

      // Photons
      const Particles truthPhoton = apply<IdentifiedFinalState>(event, "TruthPhoton").particlesByPt();
      br_nTruthPhoton = (int) truthPhoton.size();
      for(const Particle& ph: truthPhoton){
        const FourMomentum tlv = ph.momentum();
        br_truthPhotonPt.push_back(tlv.pt()/GeV);
        br_truthPhotonE.push_back(tlv.E()/GeV);
        br_truthPhotonEta.push_back(tlv.eta());
        br_truthPhotonPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
      }

      // ET miss
      const Vector3& truthMET = apply<MissingMomentum>(event, "TruthMET").vectorMissingEt();
      br_truthMETPt = truthMET.mod()/GeV;
      br_truthMETPhi = truthMET.phi(MINUSPI_PLUSPI);

      // Neutrinos
      const Particles truthNeutrinos = apply<PromptFinalState>(event, "TruthNeutrinos").particlesByPt();
      br_nTruthNeutrino = (int) truthNeutrinos.size();
      for(const Particle& nu: truthNeutrinos) {
        const FourMomentum tlv = nu.momentum();
        br_truthNeutrinoPt.push_back(tlv.pt()/GeV);
        br_truthNeutrinoE.push_back(tlv.E()/GeV);
        br_truthNeutrinoEta.push_back(tlv.eta());
        br_truthNeutrinoPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
        br_truthNeutrinoPID.push_back((int) nu.pid());
      }

      // B jets
      Jets truthBjets, truthLightjets;
      for (const Jet& jet: isoTruthJet) {
        (jet.bTagged(Cuts::pT > 5*GeV)? truthBjets: truthLightjets) += jet;
      }
      br_nTruthBJet = (int) truthBjets.size();

      // detector level studies
      // ------------------------------------------
      const Jets recoJets  = apply<JetAlg>(event, "RecoJet").jetsByPt(
        Cuts::pT > recoJetPTCut*GeV && Cuts::abseta < 2.8);
      
      const Particles recoEle = apply<ParticleFinder>(event, "RecoElectron").particlesByPt();
      const Particles recoMuon = apply<ParticleFinder>(event, "RecoMuon").particlesByPt();
      
      // Discard jets very close to electrons (dR < 0.2), 
      // or with low track multiplicity and close to muons (dR < 0.4)
      const Jets isoRecoJet = filter_discard(recoJets, [&](const Jet& j){
          if (any(recoEle, deltaRLess(j, 0.2))) return true;
 				  if (j.particles(Cuts::abscharge > 0 && Cuts::pT > 0.4*GeV).size() <= 3 && \
					  any(recoMuon, deltaRLess(j, 0.4))) return true;
				  return false;
        }
      );
      br_nRecoJet = (int) isoRecoJet.size();
      for(const Jet& j: isoRecoJet){
        const FourMomentum tlv = j.momentum();
        br_recoJetPt.push_back(tlv.pt()/GeV);
        br_recoJetE.push_back(tlv.E()/GeV);
        br_recoJetEta.push_back(tlv.eta());
        br_recoJetPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
        br_recoJetIsBtagged.push_back((int) j.bTagged(Cuts::pT > 5*GeV));
      }

      // Discard electrons close to remaining jets (dR < 0.4)
      const Particles isoRecoEle = filter_discard(
        recoEle, [&](const Particle& e) {return any(isoRecoJet, deltaRLess(e, 0.4)); }
      );
      br_nRecoEle = (int) isoRecoEle.size();
      for(const Particle& ele: isoRecoEle){
        const FourMomentum tlv = ele.momentum();
        br_recoElePt.push_back(tlv.pt()/GeV);
        br_recoEleE.push_back(tlv.E()/GeV);
        br_recoEleEta.push_back(tlv.eta());
        br_recoElePhi.push_back(tlv.phi(MINUSPI_PLUSPI));
        br_recoEleCharge.push_back((int) ele.charge());
      }

      // Discard muons close to remaining jets (dR < 0.4)
      const Particles isoRecoMuon = filter_discard(recoMuon, [&](const Particle& m) {
				for (const Jet& j : isoRecoJet) {
					if (deltaR(j,m) > 0.4) continue;
					if (j.particles(Cuts::abscharge > 0 && Cuts::pT > 0.4*GeV).size() > 3) return true;
				}
				return false;
      });
      br_nRecoMuon = (int) isoRecoMuon.size();
      for(const Particle& muon: isoRecoMuon){
        const FourMomentum tlv = muon.momentum();
        br_recoMuonPt.push_back(tlv.pt()/GeV);
        br_recoMuonE.push_back(tlv.E()/GeV);
        br_recoMuonEta.push_back(tlv.eta());
        br_recoMuonPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
        br_recoMuonCharge.push_back((int) muon.charge());
      }
      br_nRecoLepton = br_nRecoEle + br_nRecoMuon;

      // reco photons
      const Particles recoPhoton = apply<ParticleFinder>(event, "RecoPhoton").particlesByPt();
      for(const Particle& ph: recoPhoton){
        const FourMomentum tlv = ph.momentum();
        br_recoPhotonPt.push_back(tlv.pt()/GeV);
        br_recoPhotonE.push_back(tlv.E()/GeV);
        br_recoPhotonEta.push_back(tlv.eta());
        br_recoPhotonPhi.push_back(tlv.phi(MINUSPI_PLUSPI));
      }

      // ET miss
      const Vector3& recoMET = apply<SmearedMET>(event, "RecoMET").vectorMissingEt();
      br_recoMETPt = recoMET.mod()/GeV;
      br_recoMETPhi = recoMET.phi(MINUSPI_PLUSPI);

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

    double truthJetPTCut;
    double recoJetPTCut;

    // variables saved in the ntuple
    // event level information


    // truth objects
    // -------------------------------
    int br_nTruthEle;
    vector<double> br_truthElePt;
    vector<double> br_truthEleEta;
    vector<double> br_truthElePhi;
    vector<double> br_truthEleE;
    vector<int>    br_truthEleCharge;

    int br_nTruthPhoton;
    vector<double> br_truthPhotonPt;
    vector<double> br_truthPhotonEta;
    vector<double> br_truthPhotonPhi;
    vector<double> br_truthPhotonE;    

    int br_nTruthMuon;
    vector<double> br_truthMuonPt;
    vector<double> br_truthMuonEta;
    vector<double> br_truthMuonPhi;
    vector<double> br_truthMuonE;
    vector<int>    br_truthMuonCharge;

    int br_nTruthNeutrino;
    vector<double> br_truthNeutrinoPt;
    vector<double> br_truthNeutrinoEta;
    vector<double> br_truthNeutrinoPhi;
    vector<double> br_truthNeutrinoE;
    vector<int> br_truthNeutrinoPID;

    int br_nTruthJet;
    int br_nTruthBJet;
    vector<double> br_truthJetPt;
    vector<double> br_truthJetEta;
    vector<double> br_truthJetPhi;
    vector<double> br_truthJetE;
    vector<int> br_truthJetIsBtagged;

    double br_truthMETPt;
    double br_truthMETPhi;
    int br_nTruthLepton;

    // reco info
    int br_nRecoEle;
    vector<double> br_recoElePt;
    vector<double> br_recoEleEta;
    vector<double> br_recoElePhi;
    vector<double> br_recoEleE;
    vector<int>    br_recoEleCharge;

    int br_nRecoPhoton;
    vector<double> br_recoPhotonPt;
    vector<double> br_recoPhotonEta;
    vector<double> br_recoPhotonPhi;
    vector<double> br_recoPhotonE;

    int br_nRecoMuon;
    vector<double> br_recoMuonPt;
    vector<double> br_recoMuonEta;
    vector<double> br_recoMuonPhi;
    vector<double> br_recoMuonE;
    vector<int>    br_recoMuonCharge;
    int br_nRecoLepton;

    int br_nRecoJet;
    vector<double> br_recoJetPt;
    vector<double> br_recoJetEta;
    vector<double> br_recoJetPhi;
    vector<double> br_recoJetE;
    vector<int> br_recoJetIsBtagged;

    double br_recoMETPt;
    double br_recoMETPhi;

  };


  DECLARE_RIVET_PLUGIN(GenericMCNtuple);

}
