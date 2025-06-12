### Train RNN ###

root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
npz_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/npz 
out_dir=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1
model_path=/global/cfs/cdirs/m3443/usr/akaul/tau_id/rnn_1/model

data_prep ${root_dir} ${npz_dir}

train ${npz_dir} ${model_path}

apply ${model_path} ${npz_dir} ${out_dir}

plot_tauid ${out_dir}/inclusive