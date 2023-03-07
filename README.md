# root_gnn
GNN classification/Regression for reconstructed HEP events. 

[![DOI](https://zenodo.org/badge/173806807.svg)](https://zenodo.org/badge/latestdoi/173806807)

## Install
```bash
conda create --name gnnana python=3.8
conda activate gnnana
conda install -c conda-forge jupyterlab
conda install -c conda-forge root

git clone https://github.com/xju2/root_gnn.git

cd root_gnn
pip install -e .
```

## Input data
Download the W' and QCD data from [https://zenodo.org/record/3981290#.XzQs5zVlAUF](https://zenodo.org/record/3981290#.XzQs5zVlAUF)the folder `data`, and split the W' data into `wboson_big.txt` and `wboson_small.txt`.

## Graph Construction
Construct signal and background jets in to graphs and store them in `.tfrec` files. The command is in the following format:
```
create_tfrecord <Input ROOT file path> <Output path>
                --max-evts <INT: Maximum number of events to process> 
                --evts-per-record <INT: Number of events per output file>
                --type TauIdentificationDataset 
                --num-workers <INT: Number of threads>
                --signal [0: Background only | 1: 1-prong signals only | 3: 3-prong signals only | 10: 1-prong and 3-prong inclusive signals]
                --use-jetPt (Optional: Include jet Pt as node features)
                --use-jetVar (Optional: Include jet variables as global features)
                --use-delta-angles (Optional: Use delta eta and phi for the node variables)
                --with-node-type (Optional: Add in one-hot encodings for node and edge types)
                --with-hlv-features (Optional: Add high-level variables to globals)
                --tower-lim <Optional INT: Limits on the number of towers>
                --track-lim <Optional INT: Limits on the number of tracks>
```

The graphs then need to be splitted into different sets for training and validation, using the following command:

```
split_files_for_nn <Graph Path> <Destination>
```

## Training
Training usually takes 12-16h on Cori GPU for model to converge. Training can be done using the following command:
```
train_gnn config <config_name.yaml>
```
The configuration file should be in the format:
```
input_dir: <input directory>
evts_per_file: <number of events per file>
output_dir: <output directory>
model: <model name>
loss_name: GlobalLoss
loss_pars: <loss weights in a tuple (signal, background)>

batch_size: <batch size>
num_iters: <number of message passing steps>
learning_rate: <learning rate>
num_epochs: <number of epochs>
stop_on: [early stop criteria, options: auc | val_loss | acc | pre | rec]
patiences: <number of steps allowed with no improvement>
log_freq: <log frequence in terms of batches>
val_batches: <number of batches used for each validation>
shuffle_size: <number for shuffling batches>
encoder_size: <MLP size for encoder>
core_size: <MLP size for core>
with_edges: [input graph contains edge information, options: True | False]
with_globals: [input graph contains global information, options: True | False]
```
Usable models include:
* `GlobalClassifier`: Homogeneous encoder model for global classification
* `GlobalClassifierHeterogeneousEdges`: Global classifer model with Heterogeneous Node encoder
* `GlobalClassifierHeterogeneousNodes`: Global classifer model with Heterogeneous Node and Edge encoder
* `GlobalRNNClassifierSNT`: The global classifer model using recurrent LSTM functions as the encoder, implemented with sonnet modules
* `GlobalAttnEncoderClassifier`: The global classifer model using MultiHeadAttention blocks as the encoder


## Inference
Inference time could vary depending on the batch size. Usually takes
2h to finish 100k di-tau events and 3h to finish 40k QCD events. Inference can be done using the following command:
```
evaluate_tauid -i <Path to test ROOT file>
               -o <Path to output directory>
               --num-events <INT: Number of events to process>
               --config <config_name.yaml>
               --type TauIdentificationDataset
               --use-jetPt (Optional: Include jet Pt as node features)
               --use-jetVar (Optional: Include jet variables as global features)
               --use-delta-angles (Optional: Use delta eta and phi for the node variables)
               --with-node-type (Optional: Add in one-hot encodings for node and edge types)
               --with-hlv-features (Optional: Add high-level variables to globals)
               --tower-lim <Optional INT: Limits on the number of towers>
               --track-lim <Optional INT: Limits on the number of tracks>
               --signal-only (Optional: Use signals only)
```

The optional flags should match the graph construction flags to ensure that the model uses the same types of graph inputs.


## Evaluation
The inference result can then be evaluated by plotting the prediction socres and related metrics using:
```
plot_tauid <Path to directory that stores inferences>
```
Results from different models can also be compared using:
```
tauid_compare --npz <Inference from Model 1> <Inference from Model 2> ...
              --qcd <Optional: Inference on qcd jets if stored separately>
              --others <Optional: Other inference results if stored separately>
              --colors <Optional STR: Colors to plot with>
              --styles <Optional STR: Line styles e.g. 'solid' 'dashed' 'dotted'>
              --legend <Optional STR: Labels to use in the plot legends>
              --prefix <Optional STR: Prefix for the output files>
              --efficiencies <Optional List of FLOATS: Find rejection power associated with interesting efficiencies and place them in a table>
              --no-log (Optional: Do not use Log scale)
```
The command above will draw the ROC and rejection curves for all the models listed and create an Excel table summarizing the rejection and AUC values.


## Training using Keras
The following models are also implemented in the Keras framework, as it has potential faster speed for training than the sonnet modules:
* `RecurrentEncoder`: The global classifer model using recurrent LSTM functions as the encoder
* `AttentionEncoder`: The global classifer model using MultiHeadAttention blocks as the encoder
The implementation are the same as the respective ones using sonnet modules. This framework can be used with the following procedures:

The inputs need to be first stored in `.npz` files for efficiency, using:
```
create_npz <input ROOT file> <output file name> 
           -n <number of events to process> 
           --initial-event <index of starting event>
           --signal (Optional: only use signal jets in the event)
```
The command can be ran in parallel to process multiple files at a time. The files can then be merged using
```
merge_npz --output-path <output file name>
          --npz <list of files to merge>
```
The signals and backgrounds need to be stored in separate files.

The model can then be trained using 
```
train_keras <input npz file for signals> <input npz file for backgrounds>
            --model-path <path to store the model>
            -l <loss weight for signal>
            -n [Model name, options: None (default to `RecurrentEncoder`) | attn_inv (`AttentionEncoder`) | attn_pos (`AttentionEncoder` with positional encoding)]
```

The inference can then be done using
```
apply_keras <model directory> -i <input test npz file> -l <loss weight used>
```

The evaluation and comparison can be done in the same procedure as the models in sonnet framework.
