#!/bin/bash
#SBATCH -J wboson
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -A m1759
#SBATCH -o backup/%x-%j.out

which python
cd /global/homes/x/xju/code/root_gnn
#train_classifier configs/train_toptaggers.yaml
#train_classifier configs/train_wtaggers.yml
train_classifier configs/train_wtaggers_edges.yaml
