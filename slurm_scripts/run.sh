#!/bin/bash
#SBATCH -J ditaumass
#SBATCH -C gpu
#SBATCH --constraint gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 4:00:00
#SBATCH -A m3443


conda activate gnn
cd /global/cscratch1/sd/andrish/training_data/config
#train_gnn config tauid_fcn_we.yaml
#train_gnn config tauid_fcn_ne.yaml
#train_gnn config tauid_dis_we.yaml
#train_gnn config tauid_dis_ne.yaml
#train_gnn config config82.yaml

#train_gnn config config71.yaml
#cd /global/cscratch1/sd/andrish/training_data
#cp /global/cfs/cdirs/m3443/data/TauStudies/v5/ditau_test.root #/global/cscratch1/sd/andrish/datasets/tauid_dis_we_2k.root

#python tauid.py --inputfile /global/cfs/cdirs/m3443/data/TauStudies/v5/ditau_test.root  --outputfile /global/cscratch1/sd/andrish/datasets/tauid_dis_we_2k.root --num-events 500 --config /global/cscratch1/sd/andrish/training_data/config/tauid_dis_we.yaml --type tauidHeterogeneousNodes --disconnect-graph

#./tauid_plots -i /global/cscratch1/sd/andrish/datasets --ntuple tauid_dis_we_2k.root --name tauid_dis_we_2k