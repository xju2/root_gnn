#!/bin/bash
#SBATCH -J topreco_v2
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 4:00:00
#SBATCH -A m1759
#SBATCH -o backup/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=xju@lbl.gov

which python
cd /global/homes/x/xju/code/root_gnn
#train_classifier configs/train_toptaggers.yaml
#train_classifier configs/train_wtaggers.yml
#train_classifier configs/train_wtaggers_edges.yaml
#train_classifier configs/train_toppair.yaml
train_top_reco configs/train_topreco_2tops_v2.yaml
