# Introduction
This is a general [Rivet](https://rivet.hepforge.org/) that reads in HepMC data and produce a [ROOT](https://root.cern.ch/) ntuple where lepton and jet information are stored.

# Instructions
To compile the code, first checkout and run the docker image: `docexoty/mctuning:1.0.0`. In Cori, one can do the following:
```bash
shifter --image=docker:docexoty/mctuning:1.0.0 bash
```
In other system, one can do the following:
```bash
docker run -it -rm -v $PWD:$PWD -w $PWD docexoty/mctuning:1.0.0 bash
```

Then it is ready to compile:
```bash
rivet-build --with-root RivetGenericMCNtuple.so GenericMCNtuple.cc
```

To allow `Rivet` to find the analysis, setup the global environment variable:
```bash
export RIVET_ANALYSIS_PATH=$PWD
```

Now one can use the analysis to process HepMC data
```bash
rivet --analysis=GenericMCNtuple test.hepmc
```