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
train_classifier \
	--train-files "/global/cscratch1/sd/xju/WbosonTagger/tfrec_bigger/*_*.tfrec" \
	--eval-files  "/global/cscratch1/sd/xju/WbosonTagger/tfrec_val_bigger/*_*.tfrec" \
	--job-dir	  "/global/cscratch1/sd/xju/WbosonTagger/trained/EdgeOnly_SmallerGNN" \
	--model-name "EdgeClassifier" \
	--loss-name "EdgeLoss" \
	--loss-args "4, 1" \
	--train-batch-size 1 \
	--eval-batch-size 1 \
	--shuffle-buffer-size 4 \
	--num-iters 8 \
	--num-epochs 2 \
	--learning-rate 0.0005 \
	--early-stop-metric "auc_te" \
	--acceptable-fails 2 

