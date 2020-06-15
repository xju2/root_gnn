// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/DressedLeptons.hh"
#include "Rivet/Projections/MissingMomentum.hh"
#include "Rivet/Projections/PromptFinalState.hh"

namespace Rivet {


  /// @brief This analysis is based on an existing Rivet analysis
  /// "Charge asymmetry in top quark pair production in dilepton channel"
  ///  https://rivet.hepforge.org/analyses/ATLAS_2016_I1449082.html

  class MC_HplusX : public Analysis {
  public:

    /// Constructor
    DEFAULT_RIVET_ANALYSIS_CTOR(MC_HplusX);

    const double MW = 80.300*GeV;
    const double MTOP = 172.5*GeV;

    /// @name Analysis methods
    //@{

    /// Book histograms and initialise projections before the run
    void init() {

      // Cuts
      const Cut eta_full = Cuts::abseta < 5.0;
      const Cut lep_cuts = Cuts::abseta < 2.5 && Cuts::pT > 25*GeV;
      // All final state particles
      FinalState fs(eta_full);
      // Get photons to dress leptons
      IdentifiedFinalState photons(fs, PID::PHOTON);

      // Electron projections
      // ---------------------
      // Electron/muons are defined from electron/muon and photons within a cone of DR = 0.1.
      // No isolation condition is imposed. 
      // The parent of the electron/muon is required not to be a hadron or quark.
      IdentifiedFinalState el_id(fs, {PID::ELECTRON,-PID::ELECTRON});
      PromptFinalState electrons(el_id);
      electrons.acceptTauDecays(true);
      // Electron dressing
      DressedLeptons dressedelectrons(photons, electrons, 0.1, lep_cuts, true);
      declare(dressedelectrons, "dressedelectrons");
      DressedLeptons dressedelectrons_full(photons, electrons, 0.1, eta_full, true);

      // Muon projections
      // ---------------------
      IdentifiedFinalState mu_id(fs, {PID::MUON,-PID::MUON});
      PromptFinalState muons(mu_id);
      muons.acceptTauDecays(true);
      // Muon dressing
      DressedLeptons dressedmuons(photons, muons, 0.1, lep_cuts, true);
      declare(dressedmuons, "dressedmuons");
      DressedLeptons dressedmuons_full(photons, muons, 0.1, eta_full, true);

      // Neutrino projections
      // ---------------------
      // Missing ET is calculated as the 4–vector sum of neutrinos from W/Z-boson decays. Tau decays are
      // included. A neutrino is treated as a detectable particle and is selected for consideration in the same
      // way as electrons or muons, i.e. the parent is required not to be a hadron or quark (u − b).
      IdentifiedFinalState nu_id;
      nu_id.acceptNeutrinos();
      PromptFinalState neutrinos(nu_id);
      neutrinos.acceptTauDecays(true);
      declare(neutrinos, "neutrinos");

      // Jets projections
      // ---------------------
      // Jets are defined with the anti-kt algorithm, 
      // clustering all stable particles excluding the electrons,
      // muons, neutrinos, and photons used in the definition of the selected leptons.
      VetoedFinalState vfs(fs);
      vfs.addVetoOnThisFinalState(dressedelectrons_full);
      vfs.addVetoOnThisFinalState(dressedmuons_full);
      vfs.addVetoOnThisFinalState(neutrinos);
      declare(FastJets(vfs, FastJets::ANTIKT, 0.4), "Jets");

    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {

      // Get the electrons and muons
      const vector<DressedLepton> dressedelectrons = apply<DressedLeptons>(event, "dressedelectrons").dressedLeptons();
      const vector<DressedLepton> dressedmuons     = apply<DressedLeptons>(event, "dressedmuons").dressedLeptons();
      const vector<DressedLepton> leptons = dressedelectrons + dressedmuons;
      // Require at least 2 leptons in the event
      if (leptons.size() < 2) vetoEvent;

      // Get the neutrinos
      const Particles neutrinos = apply<PromptFinalState>(event, "neutrinos").particlesByPt();
      // Require at least 2 neutrinos in the event (ick)
      if (neutrinos.size() < 2)  vetoEvent;

      // Get jets and apply selection
      const Jets jets = apply<FastJets>(event, "Jets").jetsByPt(Cuts::pT > 25*GeV && Cuts::abseta < 2.5);
      // Require at least 2 jets in the event
      if (jets.size() < 2)  vetoEvent;

      // Remaining selections
      // Events where leptons and jets overlap, within dR = 0.4, are rejected.
      for (const DressedLepton& lepton : leptons) {
        if (any(jets, deltaRLess(lepton, 0.4))) vetoEvent;
      }

      // Construct pseudo-tops
      // Exactly 2 opposite-sign leptons are required (e/mu)
      if (leptons.size() != 2) vetoEvent;
      if ( (leptons[0].charge() * leptons[1].charge()) > 0.) vetoEvent;
      const FourMomentum lep_p = (leptons[0].charge3() > 0) ? leptons[0] : leptons[1];
      const FourMomentum lep_n = (leptons[0].charge3() > 0) ? leptons[1] : leptons[0];

      // Only the 2 leading pT selected neutrinos are considered
      const FourMomentum nu1 = neutrinos[0].momentum();
      const FourMomentum nu2 = neutrinos[1].momentum();

      // Two jets correspond to the two leading jets in the event.
      // If there is any b-tagged jet in the event, then the b-tagged jets
      // are preferentially selected over the non-tagged jets without taking into account its pT.
      // A jet is a b–jet if any B–hadron is included in the jet.
      // Only B-hadrons with an initial pT > 5 GeV are considered.
      Jets bjets, lightjets;
      for (const Jet& jet : jets) {
        (jet.bTagged(Cuts::pT > 5*GeV) ? bjets : lightjets) += jet;
      }
      // Already sorted by construction, since jets is sorted by decreasing pT
      // std::sort(bjets.begin()    , bjets.end()    , cmpMomByPt);
      // std::sort(lightjets.begin(), lightjets.end(), cmpMomByPt);

      // Initially take 2 highest pT jets
      FourMomentum bjet1 = jets[0];
      FourMomentum bjet2 = jets[1];
      if (!bjets.empty()) {
        bjet1 = bjets[0];
        bjet2 = (bjets.size() > 1) ? bjets[1] : lightjets[0]; //< We should have a light jet because >=2 jets requirement
      } else {
        // No btagged jets --> should have >= 2 light jets
        bjet1 = lightjets[0];
        bjet2 = lightjets[1];
      }

      // Construct pseudo-W bosons from lepton-neutrino combinations
      // Minimize the difference between the mass computed from each lepton-neutrino combination and the W boson mass
      const double massDiffW1 = fabs( (nu1 + lep_p).mass() - MW ) + fabs( (nu2 + lep_n).mass() - MW );
      const double massDiffW2 = fabs( (nu1 + lep_n).mass() - MW ) + fabs( (nu2 + lep_p).mass() - MW );
      const FourMomentum Wp = (massDiffW1 < massDiffW2) ? nu1+lep_p : nu2+lep_p;
      const FourMomentum Wn = (massDiffW1 < massDiffW2) ? nu2+lep_n : nu1+lep_n;

      // Construct pseudo-tops from jets and pseudo-W bosons
      // Minimize the difference between the mass computed from each W-boson and b-jet combination and the top mass
      const double massDiffT1 = fabs( (Wp+bjet1).mass()*GeV - MTOP ) + fabs( (Wn+bjet2).mass()*GeV - MTOP );
      const double massDiffT2 = fabs( (Wp+bjet2).mass()*GeV - MTOP ) + fabs( (Wn+bjet1).mass()*GeV - MTOP );
      const FourMomentum top_p = (massDiffT1 < massDiffT2) ? Wp+bjet1 : Wp+bjet2;
      const FourMomentum top_n = (massDiffT1 < massDiffT2) ? Wn+bjet2 : Wn+bjet1;

      // Calculate d|eta|, d|y|, etc.
      double dEta = lep_p.abseta() - lep_n.abseta();
      double dY   = top_p.absrapidity() - top_n.absrapidity();
      double mtt  = (top_p + top_n).mass()*GeV;
      double beta = fabs( (top_p + top_n).pz() ) / (top_p + top_n).E();
      double pttt = (top_p + top_n).pt()*GeV;

      // Fill histos, counters
      _h_dEta->fill(dEta);
      _h_dY  ->fill(dY  );
      // Histos for inclusive and differential asymmetries
      int mttBinID  = getBinID(kmttMeas , mtt);
      int betaBinID = getBinID(kbetaMeas, beta);
      int ptttBinID = getBinID(kptMeas  , pttt);
      for (int iM = 0; iM < kNmeas; ++iM) {
        int binID = -1;
        switch (iM) {
          case kInclMeas : binID = 0;         break;
          case kmttMeas  : binID = mttBinID ; break;
          case kbetaMeas : binID = betaBinID; break;
          case kptMeas   : binID = ptttBinID; break;
          default: binID = -1; break;
        }
        if (binID >= 0) {
          _h_dY_asym  [iM][binID] ->fill(dY  );
          _h_dEta_asym[iM][binID] ->fill(dEta);
        }
      }

    }


    /// Normalise histograms etc., after the run
    void finalize() {
    }

    //@}


    /// @name Histograms
    //@{

    //@}


  };


  DECLARE_RIVET_PLUGIN(MC_HplusX);

}
