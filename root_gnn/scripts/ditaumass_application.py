import argparse
import os

from root_gnn import model as Models
from root_gnn import losses
from root_gnn.trainer import Trainer

import numpy as np
import matplotlib.pyplot as plt

from graph_nets import utils_tf
import sonnet as snt

from root_gnn import model
from root_gnn.utils import load_yaml
from root_gnn import utils_plot

from root_gnn.trainer import read_dataset
from root_gnn.trainer import loop_dataset
from ROOT import TChain, AddressOf, std

import tensorflow as tf

from heapq import nsmallest

all_names = ['Et for Tau1', 'Eta for Tau1', 'Phi for Tau1', 'Et for Tau2', 'Eta for Tau2', 'Phi for Tau2']
            

def application(input_dir=None, output_dir=None, evtsPerRecord=1000, vBatches=20, names=all_names, num_iters=4, learning_rate=0.0005, core_size=[128,128], savefig=False, fwhm=True, inverse_output_scaling=lambda x: x):
    """
    Plot Pull Distribution Graphs for given variables using validation dataset in the input directory and trained model in the output directory
    To use the function, call application() with the desired arguments
    """
    
    # Load needed variables
    file_in = input_dir
    n_evts_per_record = evtsPerRecord
    v_batches = vBatches
    if type(names) == str:
        names = [names]

    
    modeldir = os.path.join(output_dir, 'checkpoints')

    global_batch_size = 600
    optimizer = snt.optimizers.Adam(learning_rate)
    loss_fcn = losses.GlobalRegressionLoss()


    # Load the latest checkpoint
    model = Models.GlobalRegression(6, encoder_size=core_size, core_size=core_size)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=modeldir, max_to_keep=5)
    if os.path.exists(os.path.join(modeldir, 'checkpoint')):
        status = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Loaded latest checkpoint from {}".format(modeldir),status)
    else:
        raise ValueError(f'Cannot find model at: {modeldir}')

    # Create a trainer object with the current model
    trainer = Trainer(input_dir=file_in, output_dir=output_dir,
                  model=model, loss_fcn=loss_fcn, optimizer=optimizer,
                  evts_per_file=n_evts_per_record, mode='rgr,globals', batch_size=global_batch_size,
                  val_batches=10, log_freq=300, patiences=float('inf'), num_iters=num_iters)
    
    def load_val_data(num_batch, trainer, dirc='val'):
        """
        Load validation data/graphs from directory
        """
        inputs = []
        data = trainer.load_data(dirc)[0]
        for i in range(num_batch):
            inputs.append(next(data))
        return inputs

    def get_components(p, t, x):
        """
        Get component x of the graphs and return the prediction and truth values of x
        Input: prediction and truth arrays p, t with shape (6, ), so for example, p[0] would be the Et of tau 1
        Output: predictions and truth scaled back to their original values using INVERSE_OUTPUT_SCALING function passed in as an argument to the application() fuction
        """
        p1 = []
        for i in p:
            for j in i:
                p1.append(float(j[x]))
        t1 = []
        for i in t:
            for j in i:
                t1.append(float(j[x]))
        
        return inverse_output_scaling(np.array(p1)), inverse_output_scaling(np.array(t1))


    def predict(model, test_data):
        """
        Uses the current model/loss to generate predictions on test_data
        """
        predictions, truth_info = [], []
        for data in test_data:
            inputs, targets = data
            outputs = model(inputs, num_iters, is_training=False)
            output = outputs[-1]
            predictions.append(output.globals)
            truth_info.append(targets.globals)

        return predictions, truth_info
    
    def FWHM(arr_x, arr_y):
        difference = max(arr_y) - min(arr_y)
        HM = difference / 2

        pos_extremum = arr_y.argmax()

        nearest_above = (np.abs(arr_y[pos_extremum:-1] - HM)).argmin()
        nearest_below = (np.abs(arr_y[0:pos_extremum] - HM)).argmin()

        
        left = np.mean(arr_x[nearest_below])
        right = np.mean(arr_x[nearest_above + pos_extremum])
        fwhm = right - left
        
        return left, right, fwhm

    def plot_pull_graphs(pred, truth, name, debug=False):
        """
        Display four graphs for given prediction and truth values
        Input: PRED: prediction values; TRUTH: truth values; NAME: variable name
        """
        assert len(pred) == len(truth), 'Dimension of prediction and truth needs to be equal'

        num_evts = len(pred)
        print(f"Number of events: {num_evts}")
        pull_dist = (pred - truth) / truth
        pull_val = round(np.mean(pull_dist), 3)

        pred2 = list(pred)
        pred2.sort()

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8.5,8.5))

        weights_p = np.ones_like(pred) / len(pred)
        weights_t = np.ones_like(truth) / len(truth)

        rg_t = [int(min(truth))-0.5, int(max(truth)+0.5)]
        rg_p = [int(min(pred))-0.5, int(max(pred)+0.5)]
        bin_wid = np.mean(np.diff(pred2)) * 100
        num_bins = int((rg_p[1]-rg_p[0])/bin_wid)
        rg = (min(rg_t+rg_p), max(rg_t+rg_p))


        if debug:
            print(bin_wid, num_bins)
            print(weights_p==weights_t)
            if 'Et' in name:
                print(pred)
                print(len(pred))

        ax[0,0].hist(pred, histtype='step', align='left', weights=weights_p, range=rg, bins=num_bins, lw=2, label="Prediction")
        ax[0,0].hist(truth, histtype='step',  align='left', weights=weights_t, range=rg, bins=num_bins, lw=2, label="Truth")
        ax[0,0].set_title("Normalized Value Distribution")
        ax[0,0].set_xlabel(name)
        ax[0,0].set_ylabel("Normalized Frequency")
        ax[0,0].legend(loc='upper right')

        rg = (-2, 2)
        pull_dist_adj = np.array([i for i in pull_dist if rg[0]<=i<=rg[1]])
        pull_val_adj = round(np.mean(pull_dist_adj), 3)
        weights_d = np.ones_like(pull_dist_adj) / len(pull_dist_adj)
        bin_wid = 0.02
        num_bins = int((rg[1]-rg[0])/bin_wid)

        counts, vals, patches = ax[1,0].hist(pull_dist_adj, weights=weights_d, align='left', range=rg, bins=num_bins, label=name)
        ax[1,0].set_title(name)
        ax[1,0].set_xlabel("Pull Value")
        ax[1,0].set_ylabel("Normalized Frequency")
        ax[1,0].axvline(pull_val_adj, color='red', linestyle='dashed', alpha=0.6, label=f'Mean Value={pull_val_adj}')
        if fwhm:
            fwhm_l, fwhm_r, fwhm_val = FWHM(vals, counts)
            fwhm_val = round(fwhm_val, 3)
            ax[1,0].axvspan(fwhm_l, fwhm_r, facecolor='yellow', alpha=0.4, label=f"FWHM={fwhm_val}")
        ax[1,0].legend()



        weights_d = np.ones_like(pull_dist) / len(pull_dist)
        #rg = (int(min(pull_dist))-0.5, int(max(pull_dist)+0.5))
        rg = (-10, 10)
        bin_wid = 0.02
        num_bins = 100


        ax[1,1].hist(pull_dist, weights=weights_d, align='left', range=rg, bins=num_bins, label=name)
        ax[1,1].set_title("All Pull Value Distribution")
        ax[1,1].set_xlabel("Pull Value")
        ax[1,1].set_ylabel("Normalized Frequency")
        ax[1,1].axvline(pull_val, color='red', linestyle='dashed', alpha=0.6, label=f'Mean Value={pull_val}')
        ax[1,1].legend(loc='upper right')


        ax[0,1].scatter(pred, truth, marker='.', s=0.1, label=name)
        left_end, right_end = int(min(truth)), int(max(truth))
        x = np.linspace(left_end, right_end, int(right_end-left_end))
        ax[0,1].plot(x, x, linestyle='--', color='red', alpha=0.6, label='Prediction = Truth')
        ax[0,1].set_xlabel('Prediction Value')
        ax[0,1].set_ylabel('Truth Value')
        ax[0,1].set_title("Correspondence Between Prediction and Truth Values")
        ax[0,1].legend(loc='upper right')


        plt.tight_layout()

        if savefig:
            try:
                plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
            except:
                print(f"WARNING: Failed to save image {name}.png")

        plt.show()
        print(f"Fraction of Outliers: {round(1-len(pull_dist_adj)/num_evts, 3)}")

        if debug:
            #outliers = [i for i in pull_dist if not rg[0]<=i<=rg[1]]
            #outliers.sort()
            #print(f"Outliers: {None if outliers==[] else outliers}")
            print(bin_wid, num_bins)

    # Plot the graphs for global variables
    val_data = load_val_data(v_batches, trainer=trainer)
    pred, tru = predict(model, val_data)
    mpe_6 = []
    for name in names:
        i = all_names.index(name)
        prediction, truth = get_components(pred, tru, i)
        mpe = np.mean(abs((prediction-truth)/truth))
        mpe_6.append(round(mpe, 3))
        plot_pull_graphs(prediction, truth, name)
        print()
    print("Mean Percentage Error: ")
    for i in range(len(names)):
        print(f"{names[i]}: {mpe_6[i]}")
    print(f"Average Mean Percentage Error: {1/len(names)*round(sum(mpe_6),3)}")
