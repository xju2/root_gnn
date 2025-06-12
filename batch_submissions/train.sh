#!/bin/bash
### Trains the RNN model (implemented in Keras) using the inputted NPZ files, outputting the model to the specified path

echo "Training RNN"

train_keras $1/ditau_inclusive_train_val.npz $1/qcd_inclusive_train_val.npz --model-path $2 -l 10

echo "Training complete"