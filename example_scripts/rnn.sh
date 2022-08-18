### Creating the training files ###
# These commands create processes running in parallel that are meant to 
# be executed on an interactive cpu node. Each one processes 100000 events
# and creates files with 1-prong jets, 3-prong jets, and jets with any number 
# of prongs. Right now the number of prongs is determined by the reconstructed 
# tracks and reconstructed jets instead of using truth information about each jet.

basePath=/global/cscratch1/sd/andrish/training_data/graphs/rnn/npz
ditau=/global/cfs/cdirs/m3443/data/TauStudies/v5/ditau_train.root
qcd=/global/cfs/cdirs/m3443/data/TauStudies/v5/qcd_train.root

create_npz ${ditau} \
${basePath}/ditau_train_0 -n 100000 --initial-event 0 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_0 -n 100000 --initial-event 0 \
&
create_npz ${ditau} \
${basePath}/ditau_train_1 -n 100000 --initial-event 100000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_1 -n 100000 --initial-event 100000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_2 -n 100000 --initial-event 200000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_2 -n 100000 --initial-event 200000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_3 -n 100000 --initial-event 300000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_3 -n 100000 --initial-event 300000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_4 -n 100000 --initial-event 400000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_4 -n 100000 --initial-event 400000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_5 -n 100000 --initial-event 500000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_5 -n 100000 --initial-event 500000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_6 -n 100000 --initial-event 600000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_6 -n 100000 --initial-event 600000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_7 -n 100000 --initial-event 700000 --signal &
create_npz ${qcd} \
${basePath}/qcd_train_7 -n 100000 --initial-event 700000 \
&
create_npz ${ditau} \
${basePath}/ditau_train_8 -n 100000 --initial-event 800000 --signal


### Merging the separate training files ###
# The --signal flag indicates that only tau jets should be included in the 
# training files. These are the only jets we want from the ditau events since 
# we draw the background jets exclusively from the independently generated qcd events 
# for the RNN.

outPath=/global/cscratch1/sd/andrish/training_data/graphs/rnn

merge_npz --output-path ${outPath}/ditau_train_1prong --npz ${basePath}/ditau_train_0_1prong.npz ${basePath}/ditau_train_1_1prong.npz ${basePath}/ditau_train_2_1prong.npz ${basePath}/ditau_train_3_1prong.npz ${basePath}/ditau_train_4_1prong.npz ${basePath}/ditau_train_5_1prong.npz ${basePath}/ditau_train_6_1prong.npz ${basePath}/ditau_train_7_1prong.npz ${basePath}/ditau_train_8_1prong.npz

merge_npz --output-path ${outPath}/ditau_train_3prong --npz ${basePath}/ditau_train_0_3prong.npz ${basePath}/ditau_train_1_3prong.npz ${basePath}/ditau_train_2_3prong.npz ${basePath}/ditau_train_3_3prong.npz ${basePath}/ditau_train_4_3prong.npz ${basePath}/ditau_train_5_3prong.npz ${basePath}/ditau_train_6_3prong.npz ${basePath}/ditau_train_7_3prong.npz ${basePath}/ditau_train_8_3prong.npz

merge_npz --output-path ${outPath}/qcd_train_inclusive --npz ${basePath}/qcd_train_0_inclusive.npz ${basePath}/qcd_train_1_inclusive.npz ${basePath}/qcd_train_2_inclusive.npz ${basePath}/qcd_train_3_inclusive.npz ${basePath}/qcd_train_4_inclusive.npz ${basePath}/qcd_train_5_inclusive.npz ${basePath}/qcd_train_6_inclusive.npz ${basePath}/qcd_train_7_inclusive.npz

merge_npz --output-path ${outPath}/ditau_train_inclusive --npz ${basePath}/ditau_train_0_1prong.npz ${basePath}/ditau_train_1_1prong.npz ${basePath}/ditau_train_2_1prong.npz ${basePath}/ditau_train_3_1prong.npz ${basePath}/ditau_train_4_1prong.npz ${basePath}/ditau_train_5_1prong.npz ${basePath}/ditau_train_6_1prong.npz ${basePath}/ditau_train_7_1prong.npz ${basePath}/ditau_train_8_1prong.npz ${basePath}/ditau_train_0_3prong.npz ${basePath}/ditau_train_1_3prong.npz ${basePath}/ditau_train_2_3prong.npz ${basePath}/ditau_train_3_3prong.npz ${basePath}/ditau_train_4_3prong.npz ${basePath}/ditau_train_5_3prong.npz ${basePath}/ditau_train_6_3prong.npz ${basePath}/ditau_train_7_3prong.npz ${basePath}/ditau_train_8_3prong.npz 


### Train RNN ###

# Example shown below is an inclusive RNN with LSTM layers and loss weight 5:1 for signal:background,
# loss weight can be changed by the -l <int> command

input_path=/global/cscratch1/sd/andrish/training_data/graphs/rnn
model_path=/global/cscratch1/sd/andrish/training_data/rnn

train_rnn ${input_path}/ditau_train_inclusive.npz ${input_path}/qcd_train_inclusive.npz --model_path ${model_path} -l 5

### Inference ###

# Create testing files
model_dir=/global/cscratch1/sd/andrish/training_data/rnn
out_dir=/global/cscratch1/sd/andrish/results/tauid_final/rnn
test_root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
test_npz_dir=/global/cscratch1/sd/andrish/results/tauid_final/rnn

create_npz ${test_root_dir}/ditau_test.root ${test_npz_dir}/rnn_test_ditau --signal
create_npz ${test_root_dir}/qcd_test.root ${test_npz_dir}/rnn_test_qcd --signal

# Apply on testing files
apply_rnn ${model_dir} -i ${test_npz_dir}/rnn_test_ditau_inclusive.npz -o ${out_dir}/rnn_inclusive_ditau.npz -l 5
apply_rnn ${model_dir} -i ${test_npz_dir}/rnn_test_qcd_other.npz -o ${out_dir}/rnn_inclusive_qcd1.npz -l 5

### Evalutaion ###
out_dir=/global/cscratch1/sd/andrish/results/tauid_final/rnn

plot_tauid ${out_dir}/rnn_inclusive