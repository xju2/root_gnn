#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include <iostream>
#include <fstream>
#include <string.h>

using namespace Pythia8;
using namespace std;

// It's the same as example:main42.cc
int main(int argc, char** argv) {

  // Check that correct number of command-line arguments
  if (argc < 3) {
    cerr << " Unexpected number of command-line arguments. \n You are"
         << " expected to provide one input and one output file name. \n"
         << " Program stopped! " << endl;
    return 1;
  }

  // Check that the provided input name corresponds to an existing file.
  ifstream is(argv[1]);
  if (!is) {
    cerr << " Command-line file " << argv[1] << " was not found. \n"
         << " Program stopped! " << endl;
    return 1;
  }

  // Confirm that external files will be used for input and output.
  cout << "\n >>> PYTHIA settings will be read from file " << argv[1]
       << " <<< \n >>> HepMC events will be written to file "
       << argv[2] << " <<< \n" << endl;


	// Set up pythia to hepmc object
	HepMC::Pythia8ToHepMC ToHepMC;
	HepMC::IO_GenEvent ascii_io(argv[2], std::ios::out);
	// Generator
	Pythia pythia;

	// Read in commands from external file.
	pythia.readFile(argv[1]);

	Event& event = pythia.event;

	
	// check other options
	const char* tune_file = NULL;
	for(int i = 3; i < argc; i++){
		const char* key = strtok(argv[i], "=");
		const char* val = strtok(NULL, " ");
		if(strcmp(key, "-t") == 0) tune_file = val;
	}
	if(tune_file){
		cout <<" tunning file: " << tune_file << endl;
		pythia.readFile(tune_file);
	}else{
		cout << "no tunning file" << endl;
	}

	int nEvent = pythia.mode("Main:numberOfEvents");
	if (pythia.mode("Beams:frameType") == 4) {
		nEvent = 1000000000;
	}
	int nAbort = pythia.mode("Main:timesAllowErrors");

	pythia.init();
	
	// show settings
	// pythia.settings.listChanged();
	// pythia.particleData.listChanged();

	// Loop over events
	int iAbort = 0;
	for(int iEvent = 0; iEvent < nEvent; ++iEvent) {

		// generate event
		if(!pythia.next()) {
			if (pythia.info.atEndOfFile()) {
				cout << " Aborted since reached end of Les Houches Event File\n";
				break;
			}

			if(++iAbort < nAbort) continue;
			cout << " Event generation aborted prematurely, owing to error!\n";
			break;
		}

		HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
		ToHepMC.fill_next_event(pythia, hepmcevt);

		// write the HepMC event to file. Done with it.
		ascii_io << hepmcevt;
		delete hepmcevt;
	}

	pythia.stat();
	return 0;
}
