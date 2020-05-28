#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres gpu:1
#SBATCH -c 2
#SBATCH -t 2:50:00
#SBATCH -A m1759

which python
cd /global/homes/x/xju/code/root_gnn
train_classifier configs/train_toptaggers.yaml
