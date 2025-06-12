#!/bin/bash
# Applies and evaluates the RNN model on the testing files, and outputs results to out_dir.

echo "Applying RNN model to testing files"

apply_keras $1 -i $2/ditau_inclusive_test.npz -o $3/inclusive_ditau.npz -l 5
apply_keras $1 -i $2/qcd_inclusive_test.npz -o $3/inclusive_qcd1.npz -l 5

echo "Application complete"