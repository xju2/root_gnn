### Train RNN ###

# Example shown below is an inclusive RNN with LSTM layers and loss weight 5:1 for signal:background,
# loss weight can be changed by the -l <int> command
# input_path=/global/cscratch1/sd/andrish/training_data/graphsrnn
input_path=/global/cfs/cdirs/m3443/usr/ahuang/HeteroGNN/graphs/npz/rnn
model_path=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/model

echo "Training RNN"
train_keras ${input_path}/ditau_train_inclusive.npz ${input_path}/qcd_train_inclusive.npz --model-path ${model_path} -l 10

echo "Training complete"

### Inference ###

# Create testing files
model_dir=${model_path}
out_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1
test_root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
test_npz_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/test_npz #/global/cscratch1/sd/andrish/results/tauid_final/rnn

create_npz ${test_root_dir}/ditau_test.root ${test_npz_dir}/rnn_test_ditau --signal --inclusive
create_npz ${test_root_dir}/qcd_test.root ${test_npz_dir}/rnn_test_qcd --inclusive

echo "Testing files created"

# Apply on testing files
apply_keras ${model_dir} -i ${test_npz_dir}/rnn_test_ditau_inclusive.npz -o ${out_dir}/rnn_inclusive_ditau.npz -l 17
apply_keras ${model_dir} -i ${test_npz_dir}/rnn_test_qcd_inclusive.npz -o ${out_dir}/rnn_inclusive_qcd1.npz

echo "Model applied to testing files"

### Evaluation ###
out_dir=${out_dir}/evaluation

plot_tauid ${out_dir}/rnn_inclusive

echo "Evaluation complete"