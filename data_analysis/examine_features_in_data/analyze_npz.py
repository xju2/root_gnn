import numpy as np
import argparse

# This script looks at the content of the NPZ file with name file_path in the directory dir_path
def analyze_npz():
    file_path = '/global/cfs/cdirs/m3443/usr/' + args.dir_path + '/' + args.file_path


    try:
        # Load the NPZ file
        data = np.load(file_path)

        # List the keys (names of the arrays) stored in the file
        print(f"Arrays stored in {file_path}:")
        print(list(data.files))

        print("\nShapes of the arrays:")
        for key in data.files:
            print(f"  {key}: {data[key].shape}")
            
        # Close the file handle
        data.close()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze NPZ file')
    parser.add_argument('dir_path', type=str, default='ahuang/HeteroGNN/graphs/npz/rnn/', help='Path to the directory containing the NPZ file')
    parser.add_argument('file_path', type=str, default='ditau_train_inclusive.npz', help='Path to the NPZ file')
    args = parser.parse_args()
    analyze_npz()