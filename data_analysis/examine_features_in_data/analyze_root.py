import uproot
import argparse
import awkward as ak # uproot often returns data in awkward arrays

# This script is used to get a high-level understanding of the data in the ROOT file
def analyze_root(file_path, tree_name="output;380"): # Assuming a common TTree name "Events"
    # Construct the full path similar to the original script
    # You might want to make this path handling more robust or configurable
    base_dir = '/global/cfs/cdirs/m3443/data/TauStudies/v5/' 
    full_path = base_dir + file_path

    try:
        # Open the ROOT file
        with uproot.open(full_path) as file:
            print(f"Successfully opened ROOT file: {full_path}")

            # List the top-level objects (usually includes TTrees)
            print("\nObjects (TTrees, histograms, etc.) in the file:")
            tree_keys = [key for key in file.keys(filter_classname="TTree")]
            other_keys = [key for key in file.keys() if key not in tree_keys]
            
            if tree_keys:
                 print("  TTrees:", tree_keys)
            if other_keys:
                print("  Other objects:", other_keys)


            # Analyze a specific TTree (adjust tree_name if needed)
            if tree_name in tree_keys:
                print(f"\n--- Analyzing TTree: '{tree_name}' ---")
                tree = file[tree_name]

                # Print the number of entries (events)
                print(f"Number of entries: {tree.num_entries}")

                # List the branches (like columns or arrays in NPZ)
                print("\nBranches in the TTree:")
                branch_names = tree.keys()
                for name in branch_names:
                    # You can add more info here, e.g., tree[name].typename
                    print(f"  {name}") 
                
                # Attempt to access and print labels, similar to the NPZ script
                # Adjust 'labels' if your branch has a different name
                label_branch_name = 'labels' 
                if label_branch_name in branch_names:
                    print(f"\nAttempting to load branch: '{label_branch_name}'")
                    labels_array = tree[label_branch_name].array(library="np") # Load as numpy array
                    print(f"'{label_branch_name}' branch content (first 10 entries): ")
                    print(labels_array[:10]) 
                    print(f"Shape of '{label_branch_name}' array: {labels_array.shape}")
                else:
                    print(f"\nBranch '{label_branch_name}' not found in TTree '{tree_name}'.")

            elif tree_keys:
                 print(f"\nSpecified TTree '{tree_name}' not found. Found TTrees: {tree_keys}")
                 print("Please provide the correct TTree name using the --tree argument.")
            else:
                 print(f"\nNo TTrees found in the file.")


    except FileNotFoundError:
        print(f"Error: File not found at {full_path}")
    except uproot.exceptions.KeyInFileError as e:
        print(f"Error accessing object in file: {e}")
        print(f"Check if the TTree name '{tree_name}' exists in the file.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ROOT file')
    parser.add_argument('file_path', type=str, help='Path to the ROOT file (relative to the base directory)')
    args = parser.parse_args()
    analyze_root(args.file_path)
