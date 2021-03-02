# Introduction
This is to provide a simple interface to run Pythia generator and write out HepMC data. One can use `Rivet` to further analyze the HepMC data.

We will compile the code with the docker image: `docexoty/mctuning:1.0.0`. In Cori, one can do the following
```bash
shifterimg pull docexoty/mctuning:1.0.0
```

Then to compile the code
```bash
mkdir build && cd build 

cmake .. && make 
```

To run the program:
```bash
./bin/pythia ../cmnds/zee.cmnd test.hepmc
```