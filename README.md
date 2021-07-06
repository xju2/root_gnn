# root_gnn
GNN classification for reconstructed HEP events stored in ROOT

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

## Commands for the edge classifier
* Graph Construction
```bash
create_tfrecord data/wboson_big.txt tfRec/fully_connected --max-evts 200 --evts-per-record 200 --type WTaggerDataset
```
The graphs are saved into the folder `tfRec`. Then split the folder into `train`, `val`, and `test` via the command:
```bash
split_files_for_nn tfRec inputs
```

* Graph Training
```bash
train_classifier train_wtaggers_edges.yaml
```
If `train_wtaggers_edges.yaml` does not exist in current folder, the program will look for the default one in the package, `root_gnn/configs/train_wtaggers_edges.yaml`.

* Evaluation
```bash
evaluate_wtagger data/wboson_small.txt train_wtaggers_edges.yaml test --nevts 10
```

* Metrics calculation
```bash
calculate_wtagger_metrics test.npz test
```

## Commands for event classifier
Traing two event classifiers with different inputs, one from the edge classifier and the other from the anti-$k_t$ algorithm.

### event classifier using outputs from the edge classifier
* Graph Construction for W boson events
```bash
create_tfrecord "tfRec/fully_connected*.tfrec" tfRec_filtered/wboson \
    --type WTaggerFilteredDataset \
    --signal --model-config configs/train_wtaggers_edges.yaml \
    --max-evts 100 --evts-per-record 100
```

* Graph Construction for q* events
First create `tfrecords` for the edge classifier.
```bash
create_tfrecord data/qstar.txt tfrec_qcd/qcd --type WTaggerDataset --max-evts 100 --evts-per-record 100
```

Apply the edge classifier on the q* events
```bash
create_tfrecord "tfrec_qcd/qcd*.tfrec" \
	tfRec_filtered/qcd \
	--type WTaggerFilteredDataset  \
    --model-config configs/train_wtaggers_edges.yaml \
	--max-evts 100 --evts-per-record 100
```

Split these files
```bash
split_files_for_nn tfRec_filtered FilteredJets
```

* Training Global Classifier
```bash
train_classifier train_w_qcd.yaml
```

### event classifier using outputs from the anti-$k_t$ algorithm
* Graph construction for W boson events
```bash
create_tfrecord data/wboson.txt \
	tfRec_ljet/wboson \
	--type WTaggerLeadingJetDataset \
	--signal \
	--max-evts 95000 --evts-per-record 1000
```

* Graphc osntruction for q* events
```bash
create_tfrecord data/qstar.txt \
	tfRec_ljet/qcd \
	--type WTaggerLeadingJetDataset \
	--max-evts 95000 --evts-per-record 1000
```

* Split the events
```bash
split_files_for_nn tfRec_ljet LeadingJets
```
* Graph training

```bash
train_classifier train_w_qcd_ljet.yaml
```

### Evaluate both GNNs
```bash
evaluate_w_qcd_classifier "FilteredJets/test/*.tfrec" train_w_qcd.yaml classifier_gnn
```
```bash
evaluate_w_qcd_classifier "LeadingJets/test/*.tfrec" train_w_qcd_ljet.yaml classifier_ljet
```
