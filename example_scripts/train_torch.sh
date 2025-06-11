### Train RNN ###

# Example shown below is an inclusive RNN with LSTM layers and loss weight 5:1 for signal:background,
# loss weight can be changed by the -l <int> command

root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
npz_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/npz 

# create_npz ${root_dir}/ditau_train_final2.root ${npz_dir}/ditau --signal --inclusive
# create_npz ${root_dir}/qcd_train.root ${npz_dir}/qcd --inclusive

# split_npz ${npz_dir}/ditau_inclusive.npz 0.2
# split_npz ${npz_dir}/qcd_inclusive.npz 0.2

# echo "Testing files created

echo "Training RNN"

model_path=/global/cfs/cdirs/m3443/usr/akaul/torch/rnn_1/model
train_torch.py ${npz_dir}/ditau_inclusive_train_val.npz ${npz_dir}/qcd_inclusive_train_val.npz --m ${model_path} -l 10 -n "rnn"

echo "Training complete"

### Inference ###

# Create testing files
out_dir=/global/cfs/cdirs/m3443/usr/akaul/torch/rnn_1

# Apply on testing files
# Must be done this way because apply_torch uses the same model as train_torch, which means it needs to import from apply_torch.py
apply_torch.py ${model_path} -i ${npz_dir}/ditau_inclusive_test.npz -o ${out_dir}/output_ditau.npz -l 5 -n "rnn"
apply_torch.py ${model_path} -i ${npz_dir}/qcd_inclusive_test.npz -o ${out_dir}/output_qcd.npz -l 5 -n "rnn"

echo "Model applied to testing files"

### Evaluation ###

plot_tauid ${out_dir}/output

echo "Evaluation complete"