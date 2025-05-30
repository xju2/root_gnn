import numpy as np
import os

# Define file paths
ditau_path = '/global/cfs/cdirs/m3443/usr/ahuang/HeteroGNN/graphs/npz/rnn/ditau_train_inclusive.npz'
qcd_path = '/global/cfs/cdirs/m3443/usr/ahuang/HeteroGNN/graphs/npz/rnn/qcd_train_inclusive.npz'

# Check if files exist before attempting to load
print(f"Checking for file: {ditau_path}")
if not os.path.exists(ditau_path):
    print(f"Error: Ditau file not found.")
else:
    print("Ditau file found.")

print(f"Checking for file: {qcd_path}")
if not os.path.exists(qcd_path):
    print(f"Error: QCD file not found.")
else:
    print("QCD file found.")

# Load the NPZ files and print their keys
try:
    with np.load(ditau_path) as ditau_data:
        print("\nKeys in ditau_train_inclusive.npz:")
        ditau_keys = list(ditau_data.keys())
        print(ditau_keys)
        # Let's also print the shape of the first few arrays to get an idea of the data size
        print("\nSample shapes in ditau data:")
        for key in ditau_keys[:5]:
             print(f"  '{key}': {ditau_data[key].shape}")

        print("\nLoading 'labels' array from ditau_train_inclusive.npz...")
        labels = ditau_data['labels']
        print(f"Shape of 'labels' array: {labels.shape}")
        print(f"Data type of 'labels' array: {labels.dtype}")

        # Check for unique values in labels to see what's present
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nUnique values in 'labels' and their counts:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} occurrences")

        # Create masks for label == 1 and label == 3
        mask_1 = (labels == 1)
        mask_3 = (labels != 1)

        # Check if any elements match the masks
        found_1 = np.any(mask_1)
        found_3 = np.any(mask_3)

        if found_1:
            print(f"  Number of labels equal to 1: {np.sum(mask_1)}")

        if found_3:
            print(f"  Number of labels not equal to 1: {np.sum(mask_3)}")

        # Interpretation
        if found_1 and found_3:
            print("\n'labels' array distinguishes event types.")
            # We can proceed assuming label 1 = 1-prong, label 3 = 3-prong if this looks correct
        else:
            print("\nNeither label 1 nor label 3 were found.")

        print("\nInspecting 'track_info' array:")
        if 'track_info' in ditau_data:
            track_info = ditau_data['track_info']
            print(f"  Shape: {track_info.shape}")
            print(f"  Data type: {track_info.dtype}")
            print(f"  First few entries:\n{track_info[:2]}") # Print first 2 entries
        else:
            print("  'track_info' key not found.")

        print("\nInspecting 'cluster_info' array:")
        if 'cluster_info' in ditau_data:
            cluster_info = ditau_data['cluster_info']
            print(f"  Shape: {cluster_info.shape}")
            print(f"  Data type: {cluster_info.dtype}")
            print(f"  First few entries:\n{cluster_info[:2]}") # Print first 2 entries
        else:
            print("  'cluster_info' key not found.")

        print("\nInspecting 'hlv_info' array again:")
        if 'hlv_info' in ditau_data:
            hlv_info = ditau_data['hlv_info']
            print(f"  Shape: {hlv_info.shape}")
            print(f"  Data type: {hlv_info.dtype}")
            print(f"  First few entries:\n{hlv_info[:2]}") # Print first 2 entries

            # Let's also look at unique values in each column of hlv_info
            print("\n  Unique values in the first few columns of 'hlv_info':")
            num_cols_to_check = min(hlv_info.shape[1], 5) # Check up to 5 columns
            for i in range(num_cols_to_check):
                 unique_vals = np.unique(hlv_info[:, i])
                 print(f"    Column {i}: {unique_vals[:10]} {'...' if len(unique_vals) > 10 else ''}") # Show up to 10 unique values

        else:
            print("  'hlv_info' key not found.")

except FileNotFoundError:
    print(f"Skipping loading ditau file as it wasn't found.")
except Exception as e:
    print(f"An error occurred while loading {ditau_path}: {e}")


try:
    with np.load(qcd_path) as qcd_data:
        print("\nKeys in qcd_train_inclusive.npz:")
        qcd_keys = list(qcd_data.keys())
        print(qcd_keys)
        # Print shapes for QCD data as well
        print("\nSample shapes in qcd data:")
        for key in qcd_keys[:5]:
             print(f"  '{key}': {qcd_data[key].shape}")

except FileNotFoundError:
    print(f"Skipping loading QCD file as it wasn't found.")
except Exception as e:
    print(f"An error occurred while loading {qcd_path}: {e}")
