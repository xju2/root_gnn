### Train RNN ###

# Example shown below is an inclusive RNN with LSTM layers and loss weight 5:1 for signal:background,
# loss weight can be changed by the -l <int> command

root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
npz_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/npz 

create_npz ${root_dir}/ditau_train_final2.root ${npz_dir}/ditau --signal --inclusive
create_npz ${root_dir}/qcd_train.root ${npz_dir}/qcd --inclusive

split_npz ${npz_dir}/ditau_inclusive.npz 0.2
split_npz ${npz_dir}/qcd_inclusive.npz 0.2

# echo "Testing files created

echo "Training RNN"

model_path=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/model
train_keras ${npz_dir}/ditau_inclusive_train_val.npz ${npz_dir}/qcd_inclusive_train_val.npz --model-path ${model_path} -l 10

echo "Training complete"

### Inference ###

# Create testing files
out_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1

# Apply on testing files
apply_keras ${model_path} -i ${npz_dir}/ditau_inclusive_test.npz -o ${out_dir}/inclusive_ditau.npz -l 5
apply_keras ${model_path} -i ${npz_dir}/qcd_inclusive_test.npz -o ${out_dir}/inclusive_qcd1.npz -l 5

echo "Model applied to testing files"

### Evaluation ###

plot_tauid ${out_dir}/inclusive

echo "Evaluation complete"