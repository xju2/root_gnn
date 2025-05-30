import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "./plots/rnn_PU40/hlv"
os.makedirs(output_dir, exist_ok=True)


# This script attempts to plot the distribution of the features in the PU40 events dataset
# THIS SCRIPT IS NOT USED IN THE FINAL ANALYSIS; JUST WAS AN INTERMEDIARY ATTEMPT THAT WORKED ONLY FOR THE HLVs AND WAS IN THE PROCESS OF BEING EXTENDED TO THE LLVs AND TRACKS
def plot_feature_dist(signal_1prong_data, signal_3prong_data, bg_data, feature_name, feature_label, output_filename, bins=100, range=None, use_log_scale=False):
    plt.figure(figsize=(12, 7))

    #pt_range = (0, 250)

    plt.hist(signal_1prong_data, bins=bins, density=True, histtype='step', range=range, linewidth = 2, label='1 Prong Signal (Ditau, label=1)', color='dodgerblue')
    plt.hist(signal_3prong_data, bins=bins, density=True, histtype='step', range=range, linewidth = 2, label='3 Prong Signal (Ditau, label=3)', color='black')
    plt.hist(bg_data, bins=bins, density=True, histtype='step', linewidth = 2, label='Background (QCD, label=0)', color='red')

    plt.title(f'{feature_label} distribution')
    plt.xlabel(f'{feature_label}')
    plt.xlim(range)
    plt.ylabel('Normalized Frequency')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    print(f"\nDisplaying ditau and qcd jet {feature_name} distribution...")
    plt.tight_layout()
    plt.show()

    output_filename = os.path.join(output_dir, f'hlv_{feature_name}_distribution.png')
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved plot to {output_filename}")

    plt.close()

hlv_features_meta = {
    0: ('JetPt', 'Jet $p_T$ [GeV]', 100, (0, 250), False),
    1: ('JetEta', 'Jet $\eta$', 18, (-3.5, 3.5), False),
    2: ('JetPhi', 'Jet $\phi$', 20, (-3.5, 3.5), False),
    3: ('JetLeadingTrackFracP', 'Jet Leading Track $p/p_{jet}$', 100, (0, 1), False),
    4: ('JetTrackRadius', 'Jet Track Radius [GeV*rad?]', 100, (0, 0.7), False),
    5: ('JetNumISOTracks', 'Jet Num ISO Tracks', 100, (0, 30), False),
    6: ('JetMaxDRInCore', 'Jet Max $\Delta R$ in Core', 100, (0.10, 0.20), False),
    7: ('JetTrackMass', 'Jet Track Mass [GeV]', 100, (0, 150), False), 
}

base_path = '/global/cfs/cdirs/m3443/usr/ahuang/HeteroGNN/graphs/npz/rnn_PU40'

ditau_1prong_filename = 'ditau_train_1prong.npz'
ditau_3prong_filename = 'ditau_train_3prong.npz'
qcd_filename = 'qcd_train_inclusive.npz'

ditau_1prong_npz_path = os.path.join(base_path, ditau_1prong_filename)
ditau_3prong_npz_path = os.path.join(base_path, ditau_3prong_filename)
qcd_npz_path = os.path.join(base_path, qcd_filename)


try:
    print("Loading ditau 1 prong NPZ file...")
    if not os.path.exists(ditau_1prong_npz_path):
        raise FileNotFoundError(f"The file {ditau_1prong_npz_path} does not exist.")
    ditau_1prong_data = np.load(ditau_1prong_npz_path)
    print("ditau 1 prong NPZ file loaded successfully.")

    print("Loading ditau 3 prong NPZ file...")
    if not os.path.exists(ditau_3prong_npz_path):
        raise FileNotFoundError(f"The file {ditau_3prong_npz_path} does not exist.")
    ditau_3prong_data = np.load(ditau_3prong_npz_path)
    print("ditau 3 prong NPZ file loaded successfully.")

    print("Loading qcd NPZ file...")
    if not os.path.exists(qcd_npz_path):
        raise FileNotFoundError(f"The file {qcd_npz_path} does not exist.")
    qcd_data = np.load(qcd_npz_path)
    print("qcd NPZ file loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {ditau_npz_path} does not exist.")
    exit()
except Exception as e:
    print(f"Error while loading ditau NPZ file: {e}")
    exit()

ditau_1prong_hlv = ditau_1prong_data['hlv_info']
ditau_1prong_labels = ditau_1prong_data['labels']
ditau_3prong_hlv = ditau_3prong_data['hlv_info']
ditau_3prong_labels = ditau_3prong_data['labels']

ditau_1prong_pt = ditau_1prong_hlv[:, 0]
ditau_3prong_pt = ditau_3prong_hlv[:, 0]

qcd_hlv = qcd_data['hlv_info']
qcd_labels = qcd_data['labels']

# TODO: Weight ditau events by the ratio; weight ditau events by about 10 in loss calculation

print(f"\nThere are {len(ditau_1prong_hlv[:, 0])} ditau 1 prong events and {len(ditau_3prong_hlv[:, 0])} ditau 3 prong events.")
print(f"\nThere are {len(qcd_hlv[:, 0])} qcd events.")

for index, (name, label, bins, range, use_log) in hlv_features_meta.items():
    print(f" - processing feature: {label} (Index {index})")
    ditau_1prong_feature_data = ditau_1prong_hlv[:, index]
    ditau_3prong_feature_data = ditau_3prong_hlv[:, index]
    qcd_feature_data = qcd_hlv[:, index]

    output_filename = os.path.join(output_dir, f'hlv_{name}_distribution.png')

    plot_feature_dist(
        signal_1prong_data=ditau_1prong_feature_data,
        signal_3prong_data=ditau_3prong_feature_data,
        bg_data=qcd_feature_data, 
        feature_name=name,
        feature_label=label, 
        output_filename=output_filename,
        bins=bins,
        range=range,
        use_log_scale=use_log
    )

ditau_1prong_data.close()
ditau_3prong_data.close()
qcd_data.close()
print("NPZ files closed successfully.")