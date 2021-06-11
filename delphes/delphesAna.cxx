#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#include "ExRootAnalysis/ExRootResult.h"

#include <TClonesArray.h>
#include <TSystem.h>

using namespace std;

void AnalysisEvents(ExRootTreeReader* treeReader) {
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchElectron = treeReader->UseBranch("Electron");
  TClonesArray *branchPhoton = treeReader->UseBranch("Photon");
  TClonesArray *branchMuon = treeReader->UseBranch("Muon");
  TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
  TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
  TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");

  Long64_t allEntries = treeReader->GetEntries();
  printf("Total %lld entries\n", allEntries);
}


int main(int argc, char** argv)
{
  const char* del_lib_path = "/global/cfs/cdirs/atlas/xju/software/Delphes-3.5.0/build/libDelphes.so";
  gSystem->Load(del_lib_path);
  TChain *chain = new TChain("Delphes");

  std::string filename("/global/cfs/cdirs/m3443/usr/xju/TauStudies/run_Ztautau/Ntuple_ditau.root");
  chain->Add(filename.c_str());


  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  AnalysisEvents(treeReader);

  delete chain;
  return 0;
}