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

echo "Training $1"

model_path=/global/cfs/cdirs/m3443/usr/akaul/torch/$1_1/model
# train_torch.py ${npz_dir}/ditau_inclusive_train_val.npz ${npz_dir}/qcd_inclusive_train_val.npz --m ${model_path} -l 10 -n $1

echo "Training complete"

### Inference ###

# Create testing files
out_dir=/global/cfs/cdirs/m3443/usr/akaul/torch/$1_1

# Apply on testing files
#apply_torch.py ${model_path} -i ${npz_dir}/ditau_inclusive.npz -o ${out_dir}/output_ditau_all.npz -l 5 -n $1
#apply_torch.py ${model_path} -i ${npz_dir}/qcd_inclusive.npz -o ${out_dir}/output_qcd_all.npz -l 5 -n $1

echo "Model applied to testing files"

### Evaluation ###

#plot_tauid ${out_dir}/output

echo "Evaluation complete"


# Only include this line if you want to plot the distributions of the features, to see why the model might be acting weird
if [ $1 == "lstm" ]; then
    echo "Starting data analysis"
    analyze_inclusive.py --data_path ${out_dir} --output_dir ${out_dir}
else
    echo "Invalid model type"
fi