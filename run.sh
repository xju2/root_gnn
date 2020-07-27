#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres gpu:1
#SBATCH -c 3
#SBATCH -t 4:00:00
#SBATCH -A m1759

which python
cd /global/homes/x/xju/code/root_gnn
#train_classifier configs/train_toptaggers.yaml
#train_classifier configs/train_wtaggers.yml
train_classifier configs/train_wtaggers_edges.yaml
