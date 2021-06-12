#ifndef __GHOST_ASSOCIATION_H__
#define __GHOST_ASSOCIATION_H__

#include <vector>

class PseudoJet;
class TCloneArray;

using namespace std;

class GhostAssociation {
  public:
  GhostAssociation(double dR);
  ~GhostAssociation();

  vector<PseudoJet> Associate(TClonesArray* jets, TClonesArray* tracks);

  private:
  double radius;
  
};

#endif