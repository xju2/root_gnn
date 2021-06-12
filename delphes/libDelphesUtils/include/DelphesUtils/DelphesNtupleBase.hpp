#ifndef __DELPHES_NTUPLE_BASE_H__
#define __DELPHES_NTUPLE_BASE_H__

#include <TFile.h>
#include <TTree.h>

class DelphesNtupleBase {
  public:
  DelphesNtupleBase(std::string& filename) {
    file = TFile::Open(filename.c_str(), "recreate");
    tree = new TTree("output", "Ntuple derived from Delphes");
  }
  virtual ~DelphesNtupleBase() {
    if(tree) {tree->Write(); tree = nullptr;}
    if(file) {file->Close(); file = nullptr;}
  }

  protected:
  TFile* file;
  TTree* tree;
};

#endif