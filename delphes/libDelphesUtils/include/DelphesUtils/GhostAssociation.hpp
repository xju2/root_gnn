#ifndef __GHOST_ASSOCIATION_H__
#define __GHOST_ASSOCIATION_H__

#include <vector>
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"

class TClonesArray;
class Track;
class Tower;
class Jet;

using namespace std;

namespace GhostAssociation {
  
  struct Config {
    float radius;
    float jet_ptmin;
    fastjet::JetAlgorithm jet_alg; // antikt_algorithm
  };
  Config DefaultConfig() {
    return {0.4, 10.0, fastjet::antikt_algorithm};
  }

  vector<fastjet::PseudoJet> Associate(TClonesArray* jets, TClonesArray* tracks, Config& config);
  vector<fastjet::PseudoJet> Associate(vector<Jet*>& jets, vector<Track*>& tracks, Config& config);
  vector<int> Associate(Jet* jet, vector<Track*>& tracks, Config& config);
  vector<fastjet::PseudoJet> inclusive_jets(vector<Tower*>& towers, vector<Track*>& tracks, Config& config);

};

#endif