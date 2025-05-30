import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "./plots/rnn/"
os.makedirs(output_dir, exist_ok=True)

# This script attempts to plot the distribution of the features in the ditau inclusive dataset and the qcd dataset
# THIS SCRIPT IS NOT USED IN THE FINAL ANALYSIS; JUST WAS AN INTERMEDIARY ATTEMPT THAT WORKED ONLY FOR THE HLVs AND WAS IN THE PROCESS OF BEING EXTENDED TO THE LLVs AND TRACKS

def plot_feature_dist(signal_inclusive_data, bg_data, feature_name, feature_label, output_filename, bins=100, range=None, use_log_scale=False):
    plt.figure(figsize=(12, 7))

    #pt_range = (0, 250)

    plt.hist(signal_inclusive_data, bins=bins, range=range_of_vals,density=True, histtype='step', linewidth = 2, label='Signal Inclusive (Ditau, label=1)', color='dodgerblue')
    plt.hist(bg_data, bins=bins, density=True, range=range_of_vals, histtype='step', linewidth = 2, label='Background (QCD, label=0)', color='red')

    plt.title(f'{feature_label} distribution')
    plt.xlabel(f'{feature_label}')
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
    0: ('JetPt', 'Jet $p_T$ [GeV]', 25, (0, 250), False),
    1: ('JetEta', 'Jet $\eta$', 18, (-3.5, 3.5), False),
    2: ('JetPhi', 'Jet $\phi$', 20, (-3.5, 3.5), False),
    3: ('JetLeadingTrackFracP', 'Jet Leading Track $p/p_{jet}$', 100, (0, 1), False),
    4: ('JetTrackRadius', 'Jet Track Radius [GeV*rad?]', 100, (0, 0.7), False),
    5: ('JetNumISOTracks', 'Jet Num ISO Tracks', 100, (0, 30), False),
    6: ('JetMaxDRInCore', 'Jet Max $\Delta R$ in Core', 100, (0.10, 0.20), False),
    7: ('JetTrackMass', 'Jet Track Mass [GeV]', 100, (0, 150), False), 
}
output_hlv_dir = os.path.join(output_dir, "hlv")
os.makedirs(output_hlv_dir, exist_ok=True)

track_features_meta = {
    0: ('Track_log10TrackPt', 'Track log10($p_T$) [GeV]', 40, (-1, 3), False,),
    1: ('Track_abs_dEta', 'Track [$\Delta \eta$] (relative to Jet)', 50, (0, 0.5), False),
    2: ('Track_abs_dPhi', 'Track [$\Delta \phi$] (relative to Jet)', 50, (0, 0.5), False),
    3: ('Track_z0_scaled', 'Track Scaled $z_0$ sin($\\theta$)', 40, (-2, 2), False),
    4: ('Track_d0_scaled', 'Track Scaled $d_0$', 40, (-2, 2), False),
}
output_track_dir = os.path.join(output_dir, "track")
os.makedirs(output_track_dir, exist_ok=True)


cluster_features_meta = {
    0: ('Cluster_log10TowerEt', 'Cluster log10($E_T$) [GeV]', 40, (-1, 3), False),
    1: ('Cluster_abs_dEta', 'Cluster [$\Delta \eta$] (relative to jet)', 50, (0, 0.5), False),
    2: ('Cluster_abs_dPhi', 'Cluster [$\Delta \phi$] (relative to Jet)', 50, (0, 0.5), False),
}
output_cluster_dir = os.path.join(output_dir, "cluster")
os.makedirs(output_cluster_dir, exist_ok=True)

base_path = '/global/cfs/cdirs/m3443/usr/ahuang/HeteroGNN/graphs/npz/rnn'

ditau_inclusive_filename = 'ditau_train_inclusive.npz'
qcd_filename = 'qcd_train_inclusive.npz'

ditau_inclusive_npz_path = os.path.join(base_path, ditau_inclusive_filename)
qcd_npz_path = os.path.join(base_path, qcd_filename)


try:
    print("Loading ditau inclusive NPZ file...")
    if not os.path.exists(ditau_inclusive_npz_path):
        raise FileNotFoundError(f"The file {ditau_inclusive_npz_path} does not exist.")
    ditau_inclusive_data = np.load(ditau_inclusive_npz_path)
    print("ditau 1 prong NPZ file loaded successfully.")

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

ditau_inclusive_hlv = ditau_inclusive_data['hlv_info']
ditau_inclusive_labels = ditau_inclusive_data['labels']
ditau_inclusive_track = ditau_inclusive_data['track_info']
print(f"DEBUG: Shape of ditau_inclusive_track: {ditau_inclusive_track.shape}")
print(f"DEBUG: Data type of ditau_inclusive_track: {ditau_inclusive_track.dtype}")
ditau_inclusive_cluster = ditau_inclusive_data['cluster_info']

ditau_inclusive_pt = ditau_inclusive_hlv[:, 0]

qcd_hlv = qcd_data['hlv_info']
qcd_labels = qcd_data['labels']
qcd_track = qcd_data['track_info']
qcd_cluster = qcd_data['cluster_info']

# TODO: Weight ditau events by the ratio; weight ditau events by about 10 in loss calculation

print(f"\nThere are {len(ditau_inclusive_hlv[:, 0])} ditau inclusive events.")
print(f"\nThere are {len(qcd_hlv[:, 0])} qcd events.")

print("Processing ditau inclusive hlv features...")
for index, (name, label, bins, range_of_vals, use_log) in hlv_features_meta.items():
    print(f" - processing feature: {label} (Index {index})")
    ditau_inclusive_feature_data = ditau_inclusive_hlv[:, index]
    qcd_feature_data = qcd_hlv[:, index]

    output_filename = os.path.join(output_hlv_dir, f'hlv_{name}_distribution.png')

    plot_feature_dist(
        signal_inclusive_data=ditau_inclusive_feature_data,
        bg_data=qcd_feature_data, 
        feature_name=name,
        feature_label=label, 
        output_filename=output_filename,
        bins=bins,
        range=range_of_vals,
        use_log_scale=use_log
    )

print("Processing ditau inclusive track features...")
for index, (name, label, bins, range_of_vals, use_log) in track_features_meta.items():
    print(f" - processing feature: {label} (Index {index})")
    ditau_inclusive_feature_data = ditau_inclusive_track[:, :, index]
    qcd_feature_data = qcd_track[:, :, index]
    print("\n\n\n SHAPE", ditau_inclusive_feature_data.shape, "\n\n\n")


    output_filename = os.path.join(output_track_dir, f'/track_{name}_distribution.png')
    plot_feature_dist(
        signal_inclusive_data=ditau_inclusive_feature_data,
        bg_data=qcd_feature_data, 
        feature_name=name,
        feature_label=label, 
        output_filename=output_filename,
        bins=bins,
        range=range_of_vals,
        use_log_scale=use_log
    )



print("Processing ditau inclusive cluster features...")
for index, (name, label, bins, range, use_log) in cluster_features_meta.items():
    print(f" - processing feature: {label} (Index {index})")
    ditau_inclusive_feature_data = ditau_inclusive_cluster[:, index]
    qcd_feature_data = qcd_cluster[:, index]

    output_filename = os.path.join(output_dir, f'cluster_{name}_distribution.png')

    plot_feature_dist(
        signal_inclusive_data=ditau_inclusive_feature_data,
        bg_data=qcd_feature_data, 
        feature_name=name,
        feature_label=label, 
        output_filename=output_filename,
        bins=bins,
        range=range,
        use_log_scale=use_log
    )

ditau_inclusive_data.close()
qcd_data.close()
print("NPZ files closed successfully.")