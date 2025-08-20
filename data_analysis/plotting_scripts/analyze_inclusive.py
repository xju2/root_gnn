import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data_path", type=str, required=True)
args.add_argument("--output_dir", type=str, required=True)
args = args.parse_args()

data_path = args.data_path
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# This script attempts to plot the distribution of the features in the ditau inclusive dataset and the qcd dataset
# THIS SCRIPT IS NOT USED IN THE FINAL ANALYSIS; JUST WAS AN INTERMEDIARY ATTEMPT THAT WORKED ONLY FOR THE HLVs AND WAS IN THE PROCESS OF BEING EXTENDED TO THE LLVs AND TRACKS

def plot_histogram_with_errors(data, bins, range, use_log_scale, label, color):
    raw_counts, bin_edges = np.histogram(data, bins=bins, range=range)
    normalized_counts, _, _ = plt.hist(data, bins=bin_edges, density=True, histtype='step', linewidth = 2, label=label, color=color)
    errors = np.sqrt(normalized_counts)
    total_count = np.sum(raw_counts)
    bin_widths = np.diff(bin_edges)
    normalized_errors = np.zeros_like(normalized_counts)

    if total_count > 0:
        non_zero_bin_widths_mask = bin_widths > 0
        if np.any(non_zero_bin_widths_mask):
            valid_errors = errors[non_zero_bin_widths_mask]
            valid_bin_widths = bin_widths[non_zero_bin_widths_mask]
            normalized_errors[non_zero_bin_widths_mask] = valid_errors / (total_count * valid_bin_widths)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, normalized_counts, yerr=normalized_errors, fmt='none', ecolor=color, capsize=3)


def plot_feature_dist(signal_data_below_threshold, signal_data_above_threshold, bg_data_below_threshold, bg_data_above_threshold, feature_name, feature_label, output_filename, bins=100, plot_range=None, use_log_scale=False, threshold=0.2):
    plt.figure(figsize=(12, 7))

    #pt_range = (0, 250)

    plot_histogram_with_errors(signal_data_below_threshold, bins, plot_range, use_log_scale, f'Signal Inclusive + prediction below {threshold} (Ditau, label=1)', 'dodgerblue')
    plot_histogram_with_errors(bg_data_below_threshold, bins, plot_range, use_log_scale, f'Background + prediction below {threshold} (QCD, label=0)', 'red')
    plot_histogram_with_errors(signal_data_above_threshold, bins, plot_range, use_log_scale, f'Signal Inclusive + prediction above {1 - threshold} (Ditau, label=1)', 'green')
    plot_histogram_with_errors(bg_data_above_threshold, bins, plot_range, use_log_scale, f'Background + prediction above {1 - threshold} (QCD, label=0)', 'orange')

    plt.title(f'{feature_label} distribution')
    plt.xlabel(f'{feature_label}')
    plt.ylabel('Normalized Frequency')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    print(f"\nDisplaying ditau and qcd jet {feature_name} distribution...")
    plt.tight_layout()
    plt.show()

    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved plot to ${output_filename}")

    plt.close()

hlv_features_meta = {
    0: ('JetPt', 'Jet $p_T$ [GeV]', 20, (0, 250), False),
    1: ('JetEta', 'Jet $\eta$', 15, (-3.5, 3.5), False),
    2: ('JetPhi', 'Jet $\phi$', 15, (-3.5, 3.5), False),
    3: ('JetLeadingTrackFracP', 'Jet Leading Track $p/p_{jet}$', 40, (0, 1), False),
    4: ('JetTrackRadius', 'Jet Track Radius [GeV*rad?]', 40, (0, 0.7), False),
    5: ('JetNumISOTracks', 'Jet Num ISO Tracks', 30, (0, 30), False),
    6: ('JetMaxDRInCore', 'Jet Max $\Delta R$ in Core', 25, (0.10, 0.20), False),
    7: ('JetTrackMass', 'Jet Track Mass [GeV]', 50, (0, 150), False), 
}
output_hlv_dir = os.path.join(output_dir, "hlv")
os.makedirs(output_hlv_dir, exist_ok=True)

ditau_inclusive_npz_path = os.path.join(data_path, 'output_ditau.npz')
qcd_npz_path = os.path.join(data_path, 'output_qcd.npz')


try:
    print("Loading ditau inclusive NPZ file...")
    if not os.path.exists(ditau_inclusive_npz_path):
        raise FileNotFoundError(f"The file {ditau_inclusive_npz_path} does not exist.")
    ditau_inclusive_data = np.load(ditau_inclusive_npz_path)
    print("ditau NPZ file loaded successfully.")

    print("Loading qcd NPZ file...")
    if not os.path.exists(qcd_npz_path):
        raise FileNotFoundError(f"The file {qcd_npz_path} does not exist.")
    qcd_data = np.load(qcd_npz_path)
    print("qcd NPZ file loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {qcd_npz_path} does not exist.")
    exit()
except Exception as e:
    print(f"Error while loading ditau NPZ file: {e}")
    exit()

ditau_truth = ditau_inclusive_data['truth_info']
ditau_predictions = ditau_inclusive_data['predictions']
ditau_hlv = ditau_inclusive_data['hlv']
ditau_track = ditau_inclusive_data['track']
ditau_cluster = ditau_inclusive_data['cluster']


print(f"DEBUG: Shape of ditau_inclusive_track: {ditau_track.shape}")
print(f"DEBUG: Data type of ditau_inclusive_track: {ditau_track.dtype}")

ditau_pt = ditau_hlv[:, 0]

qcd_hlv = qcd_data['hlv']
qcd_truth = qcd_data['truth_info']
qcd_predictions = qcd_data['predictions']
qcd_track = qcd_data['track']
qcd_cluster = qcd_data['cluster']

# TODO: Weight ditau events by the ratio; weight ditau events by about 10 in loss calculation

print(f"\nThere are {len(ditau_hlv[:, 0])} ditau inclusive events.")
print(f"\nThere are {len(qcd_hlv[:, 0])} qcd events.")

print("Processing ditau inclusive hlv features...")
for index, (name, label, bins, range_of_vals, use_log) in hlv_features_meta.items():
    print(f" - processing feature: {label} (Index {index})")
    threshold = 0.2
    ditau_indices_below_threshold = np.where((ditau_truth == 1) & (ditau_predictions <= threshold))[0]
    ditau_indices_above_threshold = np.where((ditau_truth == 1) & (ditau_predictions >= 1 - threshold))[0]

    qcd_indices_below_threshold = np.where((qcd_truth == 0) & (qcd_predictions <= threshold))[0]
    qcd_indices_above_threshold = np.where((qcd_truth == 0) & (qcd_predictions >= 1 - threshold))[0]

    ditau_feature_data_below_threshold = ditau_hlv[ditau_indices_below_threshold, index]
    ditau_feature_data_above_threshold = ditau_hlv[ditau_indices_above_threshold, index]

    qcd_feature_data_below_threshold = qcd_hlv[qcd_indices_below_threshold, index]
    qcd_feature_data_above_threshold = qcd_hlv[qcd_indices_above_threshold, index]

    output_filename = os.path.join(output_hlv_dir, f'{name}_distribution.png')

    plot_feature_dist(
        signal_data_below_threshold=ditau_feature_data_below_threshold,
        signal_data_above_threshold=ditau_feature_data_above_threshold,
        bg_data_below_threshold=qcd_feature_data_below_threshold,
        bg_data_above_threshold=qcd_feature_data_above_threshold,
        feature_name=name,
        feature_label=label, 
        output_filename=output_filename,
        bins=bins,
        plot_range=range_of_vals,
        use_log_scale=use_log,
        threshold=threshold
    )

ditau_inclusive_data.close()
qcd_data.close()
print("NPZ files closed successfully.")