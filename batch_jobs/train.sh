### Train RNN ###

# Example shown below is an inclusive RNN with LSTM layers and loss weight 5:1 for signal:background,
# loss weight can be changed by the -l <int> command
# input_path=/global/cscratch1/sd/andrish/training_data/graphsrnn
input_path=/global/cfs/cdirs/m3443/usr/ahuang/HeteroGNN/graphs/npz/rnn
model_path=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/model

echo "Training RNN"
train_keras ${input_path}/ditau_train_inclusive.npz ${input_path}/qcd_train_inclusive.npz --model-path ${model_path} -l 5

echo "Training complete"