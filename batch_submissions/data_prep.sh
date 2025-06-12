#!/bin/bash
# This file is used to prepare the data for the RNN training, by taking in the ROOT data, converting it 
# to NPZ format, and then splitting it into training and validation sets

echo "Creating NPZ files"

create_npz $1/ditau_train_final2.root $2/ditau --signal --inclusive
create_npz $1/qcd_train.root $2/qcd --inclusive

split_npz $2/ditau_inclusive.npz 0.2
split_npz $2/qcd_inclusive.npz 0.2

echo "NPZ files created"