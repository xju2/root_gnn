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
Save data in the folder `data`.

## Commands for the edge classifier
* Graph Construction
```bash
create_tfrecord data/wboson_big.txt tfRec/fully_connected --max-evts 200 --evts-per-record 200 --type WTaggerDataset
```
The graphs are saved into the folder `tfRec`. Then create another folder `tfRec_val` and move some TFRecord files to the folder `tfRec_val` for validation purpose.
Then modify `tfrec_dir_train` and `tfrec_dir_val` in the `configs/train_wtaggers_edges.yaml` so that they point to the training and validation data. 
Modify `output_dir` so it points to a output directory.
* Graph Training
```bash
train_classifier configs/train_wtaggers_edges.yaml
```

create a `data/wboson_small.txt` file from the `wboson.txt` using the events that are not used in training.
* Evaluation
```bash
evaluate_wtagger data/wboson_small.txt configs/train_wtaggers_edges.yaml test --nevts 10
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
```bash
create_tfrecord data/qstar.txt tfrec_qcd/qcd --type WTaggerDataset --max-evts 100 --evts-per-record 100
```
```bash
create_tfrecord "tfrec_qcd/qcd*.tfrec" \
	tfRec_filtered/qcd \
	--type WTaggerFilteredDataset  \
    --model-config configs/train_wtaggers_edges.yaml \
	--max-evts 100 --evts-per-record 100
```
Create `tfRec_filtered_val` and put events there for validation.
Change the `tfrec_dir_train`, `tfrec_dir_val` and `output_dir` in `configs/train_w_qcd.yaml` accordingly.
* Graph training
```bash
train_classifier configs/train_w_qcd.yaml
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
Create `tfRec_ljet_val` and put events there for validation.
* Graph training
Again, change the `tfrec_dir_train`, `tfrec_dir_val` and `output_dir` in `configs/train_w_qcd_ljet.yaml` accordingly.
```bash
train_classifier configs/train_w_qcd_ljet.yaml
```

### Evaluate both GNNs
```bash
evaluate_w_qcd_classifier "tfRec_filtered_val/*.tfrec" configs/train_w_qcd.yaml classifier_gnn
```
```bash
evaluate_w_qcd_classifier "tfRec_ljet_val/*.tfrec" configs/train_w_qcd_ljet.yaml classifier_ljet
```
