model_path=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/model
# Create testing files
model_dir=${model_path}
out_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1
test_root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
test_npz_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/test_npz #/global/cscratch1/sd/andrish/results/tauid_final/rnn

create_npz ${test_root_dir}/ditau_test.root ${test_npz_dir}/rnn_test_ditau --signal --inclusive
create_npz ${test_root_dir}/qcd_test.root ${test_npz_dir}/rnn_test_qcd --inclusive

#echo "Testing files created"

# Apply on testing files
apply_keras ${model_dir} -i ${test_npz_dir}/rnn_test_ditau_inclusive.npz -o ${out_dir}/rnn_inclusive_ditau.npz -l 5
apply_keras ${model_dir} -i ${test_npz_dir}/rnn_test_qcd_inclusive.npz -o ${out_dir}/rnn_inclusive_qcd1.npz -l 5

#echo "Model applied to testing files"

### Evaluation ###
echo "Evaluating model"

plot_tauid ${out_dir}/rnn_inclusive

echo "Evaluation complete" 