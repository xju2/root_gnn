
### Make Graphs ###
# Note: Graph making usually takes around 1h on Cori CPU

input_root_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
graph_dir=/global/cscratch1/sd/andrish/training_data/graphs_4/HetHLV
  
# di-tau events (signal)
# Note: The --signal <int> command here has argument 10, it means the signals
# are processed inclusively with 1-prong and 3-prong jets. Processing them separately
# can be done by using --signal 1 or --signal 3. The --with-node-type command adds
# in one-hot encodings for the nodes and edges for hetero models. Remove if homo models
# are used.
create_tfrecord ${input_root_dir}/ditau_train.root \
    ${graph_dir}/tfrec/ditau \
    --max-evts 10000000 \
    --evts-per-record 10000 \
    --type TauIdentificationDataset \
    --num-workers 10 \
    --signal 10 \
    --use-jetPt \
    --use-jetVar \
    --use-delta-angles \
    --with-node-type \
    --with-hlv-features \
    --tower-lim 6 \
    --track-lim 10

# QCD events (background)
create_tfrecord ${input_root_dir}/qcd_train.root \
    ${graph_dir}/tfrec/qcd \
    --max-evts 10000000 \
    --evts-per-record 10000 \
    --type TauIdentificationDataset \
    --num-workers 10 \
    --use-jetPt \
    --use-jetVar \
    --use-delta-angles \
    --with-hlv-features \
    --with-node-type \
    --signal 0 \
    --tower-lim 6 \
    --track-lim 10

# Split graphs for training
split_files_for_nn ${graph_dir}/tfrec ${graph_dir}/inputs


### Training ###
# Note: Training usually takes 8h on Cori GPU for model to converge
train_gnn config hetero_encoder.yaml

### Inference ###
# Note: Inference time could vary depending on the batch size. Usually takes
# 2h to finish 100k di-tau events and 3h to finish 40k QCD events.

model_dir=/global/cscratch1/sd/andrish/training_data/hetero_encoder
test_dir=/global/cfs/cdirs/m3443/data/TauStudies/v5
config_name=hetero_encoder
# Config name should match the .yaml file name
result_dir=/global/cscratch1/sd/andrish/results/tauid_final

mkdir ${result_dir}

# di-tau tests
# Note: all the commands should be same as the make_graph commands EXCEPT the --signal command
# is replaced with --signal-only to separate 1-prong and 3-prong jets
evaluate_tauid -i ${test_dir}/ditau_test.root \
                -o ${result_dir}/${config_name} \
                --num-events 100000 \
                --config ${config_name}.yaml\
                --type TauIdentificationDataset \
                --use-delta-angles \
                --use-jetPt \
                --use-jetVar \
                --with-node-type \
                --with-hlv-features \
                --tower-lim 6 \
                --track-lim 10 \
                --signal-only \

# QCD tests
# Note: there are around 100k events, so it could take more than 4h to run on GPU, might need 
# multiple runs or separate runs, using [--initial-event 40000, --num-events 80000], etc.
# Note: --num-events actually means the ending event index, so [--initial-event 40000, --num-events 80000]
# would mean to process event number 40000 to event number 80000
evaluate_tauid -i ${test_dir}/qcd_test.root \
                -o ${result_dir}/${config_name} \
                --initial-event 0 \
                --num-events 100000 \
                --config ${config_name}.yaml\
                --type TauIdentificationDataset \
                --use-delta-angles \
                --use-jetPt \
                --use-jetVar \
                --with-node-type \
                --with-hlv-features \
                --tower-lim 6 \
                --track-lim 10 \


### Evaluation ###
result_dir=/global/cscratch1/sd/andrish/results/tauid_final
config_name=hetero_encoder

plot_tauid ${result_dir}/${config_name}


### Comparison ###
result_dir=/global/cscratch1/sd/andrish/results/tauid_final
config1_name=hetero_encoder
config2_name=homo_encoder

tauid_compare \
--npz ${result_dir}/${config1_name}.npz \
${result_dir}/${config2_name}.npz \
--legend 'Hetero Encoder (Node & Edge) with HLV' 'Homo Encoder (Node & Edge) with HLV' \
--title 'Inclusive' \
--prefix ${result_dir}/Comparison