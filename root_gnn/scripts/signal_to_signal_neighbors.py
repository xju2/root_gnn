#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def load_signal_data(signal_file):
    """Load LSTM prediction file for signal events."""
    print("Loading signal (ditau) predictions...")
    signal_data = np.load(signal_file)
    return signal_data

def flatten_features(track, cluster, hlv):
    """Flatten all features into a single feature vector per event."""
    # Flatten track features: (N, 10, 6) -> (N, 60)
    track_flat = track.reshape(track.shape[0], -1)
    
    # Flatten cluster features: (N, 6, 4) -> (N, 24)  
    cluster_flat = cluster.reshape(cluster.shape[0], -1)
    
    # HLV features: (N, 8) -> (N, 8)
    hlv_flat = hlv
    
    # Concatenate all features: (N, 60 + 24 + 8) = (N, 92)
    features = np.concatenate([track_flat, cluster_flat, hlv_flat], axis=1)
    
    return features

def find_candidate_neighbors_for_signal(args):
    """Find candidate neighbors for a single signal event (for parallel processing)."""
    i, distances, max_neighbors = args
    
    # Sort by distance (ascending)
    sorted_indices = np.argsort(distances)
    
    # Collect neighbors that meet criteria
    candidates = []
    candidate_distances = []
    
    for idx in sorted_indices:
        # Add to candidates (we'll filter for uniqueness later)
        candidates.append(idx)
        candidate_distances.append(distances[idx])
        
        # Stop if we have enough candidates (we'll need extras in case of conflicts)
        if len(candidates) >= max_neighbors * 3:  # Get 3x candidates to handle conflicts
            break
    
    return i, candidates, candidate_distances

def normalize_delta_phi(delta_phi):
    """Computes delta_phi in range [-pi,pi] for a scalar or numpy array."""
    delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi
    return delta_phi

def track_distance(correct_features, incorrect_features):
    """Calculate track distance using broadcasting for efficiency."""
    # Reshape from (N, 60) to (N, 10 tracks, 6 features)
    c_tracks = correct_features.reshape(correct_features.shape[0], 10, 6)
    i_tracks = incorrect_features.reshape(incorrect_features.shape[0], 10, 6)

    # Use [:, None, ...] to add a dimension for broadcasting
    # This results in a (N_correct, N_incorrect, 10 tracks) matrix for each variable
    delta_eta = c_tracks[:, None, :, 2] - i_tracks[None, :, :, 2]
    delta_phi = normalize_delta_phi(c_tracks[:, None, :, 3] - i_tracks[None, :, :, 3])
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

    c_pt = 10**c_tracks[:, :, 1]
    i_pt = 10**i_tracks[:, :, 1]
    delta_pt = np.abs(c_pt[:, None, :] - i_pt[None, :, :])

    # Calculate the average distance across all 10 tracks (axis=2)
    avg_distance = np.mean(delta_r + delta_pt, axis=2)
    return avg_distance

def cluster_distance(correct_features, incorrect_features):
    """Calculate cluster distance using broadcasting for efficiency."""
    # Reshape from (N, 24) to (N, 6 clusters, 4 features)
    c_clusters = correct_features.reshape(correct_features.shape[0], 6, 4)
    i_clusters = incorrect_features.reshape(incorrect_features.shape[0], 6, 4)

    # Broadcast to get (N_correct, N_incorrect, 6 clusters) matrices
    delta_eta = c_clusters[:, None, :, 2] - i_clusters[None, :, :, 2]
    delta_phi = normalize_delta_phi(c_clusters[:, None, :, 3] - i_clusters[None, :, :, 3])
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

    c_et = 10**c_clusters[:, :, 1]
    i_et = 10**i_clusters[:, :, 1]
    delta_et = np.abs(c_et[:, None, :] - i_et[None, :, :])

    # Average distance across all 6 clusters
    avg_distance = np.mean(delta_r + delta_et, axis=2)
    return avg_distance

def hlv_distance(correct_hlv_features, incorrect_hlv_features):
    """Calculate HLV distance based on Euclidean distance."""
    return euclidean_distances(correct_hlv_features, incorrect_hlv_features)

def distance_metric(correct_features, incorrect_features):
    # Note: an unweighted sum of distances assumes each component is equally important.
    track_dist = track_distance(correct_features[:, :60], incorrect_features[:, :60])
    cluster_dist = cluster_distance(correct_features[:, 60:84], incorrect_features[:, 60:84])
    hlv_dist = hlv_distance(correct_features[:, 84:92], incorrect_features[:, 84:92])
    return track_dist + cluster_dist + hlv_dist

def find_signal_to_signal_neighbors_parallel(distance_matrix,
                                           correct_indices, incorrect_indices,
                                           max_neighbors=10, n_jobs=None):
    """Parallel version: Find nearest neighbors from incorrectly predicted signals."""
    
    print("Computing similarity metric between correctly and incorrectly predicted signal events...")
    start_time = time.time()
    
    # The distance matrix is now pre-computed and passed in
    
    # Determine number of processes
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(correct_indices))
    elif n_jobs == -1:
        n_jobs = cpu_count()
    
    print(f"Finding neighbors using {n_jobs} parallel processes...")
    
    # Prepare arguments for parallel processing
    args_list = [(i, distance_matrix[i], max_neighbors)
                 for i in range(len(correct_indices))]
    
    # Find candidate neighbors in parallel
    with Pool(n_jobs) as pool:
        candidate_results = pool.map(find_candidate_neighbors_for_signal, args_list)
    
    # Sort results by mean distance to prioritize low-distance events
    mean_dists = []
    for i, candidates, candidate_dists in candidate_results:
        if len(candidate_dists) > 0:
            mean_dists.append((i, np.mean(candidate_dists[:max_neighbors])))
        else:
            mean_dists.append((i, np.inf)) # Use infinity for events with no neighbors
    
    # Sort by mean distance (ascending)
    mean_dists.sort(key=lambda x: x[1], reverse=False)
    priority_order = [x[0] for x in mean_dists]
    
    # Now assign neighbors sequentially in priority order to avoid conflicts
    all_neighbors = [[] for _ in range(len(correct_indices))]
    all_distances = [[] for _ in range(len(correct_indices))]
    used_incorrect_indices = set()
    
    for i in priority_order:
        _, candidates, candidate_dists = candidate_results[i]
        neighbors = []
        neighbor_distances = []
        
        for idx, dist in zip(candidates, candidate_dists):
            if idx not in used_incorrect_indices:
                neighbors.append(idx)
                neighbor_distances.append(dist)
                used_incorrect_indices.add(idx)
                
                if len(neighbors) >= max_neighbors:
                    break
        
        all_neighbors[i] = neighbors
        all_distances[i] = neighbor_distances
    
    print(f"Neighbor finding completed in {time.time() - start_time:.2f} seconds")
    return all_neighbors, all_distances

def find_signal_to_signal_neighbors(distance_matrix, 
                                  correct_indices, incorrect_indices,
                                  max_neighbors=10,
                                  use_parallel=True, n_jobs=None):
    """Find nearest neighbors from incorrectly predicted signals for each correctly predicted signal."""
    
    if use_parallel and len(correct_indices) > 10:  # Use parallel for larger datasets
        return find_signal_to_signal_neighbors_parallel(
            distance_matrix, correct_indices, incorrect_indices,
            max_neighbors, n_jobs)
    
    # Original sequential version for small datasets
    # The distance matrix is now pre-computed and passed in
    
    # Initialize results
    all_neighbors = []
    all_distances = []
    used_incorrect_indices = set()
    
    # For each correctly predicted signal event
    for i in range(len(correct_indices)):
        # Get distance scores for this event
        distances = distance_matrix[i]
        
        # Sort by distance (ascending)
        sorted_indices = np.argsort(distances)
        
        # Collect neighbors that meet criteria
        neighbors = []
        neighbor_distances = []
        
        for idx in sorted_indices:
            # Skip if already used
            if idx in used_incorrect_indices:
                continue
                
            # Add to neighbors
            neighbors.append(idx)
            neighbor_distances.append(distances[idx])
            used_incorrect_indices.add(idx)
            
            # Stop if we have enough neighbors
            if len(neighbors) >= max_neighbors:
                break
        
        all_neighbors.append(neighbors)
        all_distances.append(neighbor_distances)
    
    return all_neighbors, all_distances

def analyze_signal_to_signal_lstm(signal_file, prediction_threshold=0.98, 
                                max_neighbors=10,
                                output_dir='signal_to_signal_analysis_distance',
                                use_parallel=True, n_jobs=None, batch_size=1000,
                                debug_subset=None):
    """Main analysis function for signal-to-signal nearest neighbors."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    signal_data = load_signal_data(signal_file)
    
    print(f"Signal data shapes:")
    print(f"  Track: {signal_data['track'].shape}")
    print(f"  Cluster: {signal_data['cluster'].shape}")
    print(f"  HLV: {signal_data['hlv'].shape}")
    print(f"  Predictions: {signal_data['predictions'].shape}")
    print(f"  Truth: {signal_data['truth_info'].shape}")
    
    # Find correctly predicted signal events (high confidence true positives)
    signal_correct_mask = (signal_data['truth_info'] == 1) & (signal_data['predictions'] >= prediction_threshold)
    signal_correct_indices = np.where(signal_correct_mask)[0]
    
    print(f"\nFound {len(signal_correct_indices)} correctly predicted signal events (threshold={prediction_threshold})")
    
    # Find incorrectly predicted signal events (false negatives)
    signal_incorrect_mask = (signal_data['truth_info'] == 1) & (signal_data['predictions'] < 0.5)
    signal_incorrect_indices = np.where(signal_incorrect_mask)[0]
    
    print(f"Found {len(signal_incorrect_indices)} incorrectly predicted signal events (false negatives)")

    # --- DEBUG: Subset data for faster testing if requested ---
    if debug_subset is not None:
        print(f"\n--- DEBUG MODE: Subsetting data to a max of {debug_subset} events each ---")
        # Shuffle to get a random sample, not just the first N events
        np.random.shuffle(signal_correct_indices)
        np.random.shuffle(signal_incorrect_indices)
        signal_correct_indices = signal_correct_indices[:debug_subset]
        signal_incorrect_indices = signal_incorrect_indices[:debug_subset]
        print(f"  Using {len(signal_correct_indices)} correct and {len(signal_incorrect_indices)} incorrect events for this run.")
    
    if len(signal_incorrect_indices) == 0:
        print("Error: No incorrectly predicted signal events found!")
        return None
    
    # Extract features
    correct_features = flatten_features(
        signal_data['track'][signal_correct_indices],
        signal_data['cluster'][signal_correct_indices], 
        signal_data['hlv'][signal_correct_indices]
    )
    
    incorrect_features = flatten_features(
        signal_data['track'][signal_incorrect_indices],
        signal_data['cluster'][signal_incorrect_indices],
        signal_data['hlv'][signal_incorrect_indices]
    )
    
    print(f"\nFeature vector dimensions:")
    print(f"  Correct signal features: {correct_features.shape}")
    print(f"  Incorrect signal features: {incorrect_features.shape}")

    # --- BATCH PROCESSING TO MANAGE MEMORY ---
    print(f"\nProcessing in batches of {batch_size} to manage memory...")
    num_correct_events = len(signal_correct_indices)
    all_neighbors_list = []
    all_distances_list = []

    for i in range(0, num_correct_events, batch_size):
        batch_end = min(i + batch_size, num_correct_events)
        print(f"  Processing batch {i//batch_size + 1}/{(num_correct_events + batch_size - 1)//batch_size} (events {i}-{batch_end-1})...")

        # Select the batch of correct features and indices
        correct_features_batch = correct_features[i:batch_end]
        correct_indices_batch = signal_correct_indices[i:batch_end]

        # Compute distance on ORIGINAL features BEFORE scaling
        start_time = time.time()
        distance_matrix_batch = distance_metric(correct_features_batch, incorrect_features)
        print(f"    Distance matrix for batch computed in {time.time() - start_time:.2f} seconds")

        # Find nearest neighbors for the current batch
        neighbors_list_batch, distances_list_batch = find_signal_to_signal_neighbors(
            distance_matrix_batch,
            correct_indices_batch, signal_incorrect_indices,
            max_neighbors=max_neighbors,
            use_parallel=use_parallel,
            n_jobs=n_jobs
        )
        all_neighbors_list.extend(neighbors_list_batch)
        all_distances_list.extend(distances_list_batch)
    
    print("All batches processed.")
    
    # The rest of the script uses the aggregated lists from the batches
    neighbors_list = all_neighbors_list
    distances_list = all_distances_list
    
    # Calculate mean distances for each correctly predicted signal
    mean_distances = []
    for distances in distances_list:
        if len(distances) > 0:
            mean_distances.append(np.mean(distances))
        else:
            mean_distances.append(np.inf)
    mean_distances = np.array(mean_distances)
    
    # Find top 10 events by mean distance (lowest distance is best)
    top_indices = np.argsort(mean_distances)[:10]
    
    print(f"\n{'='*60}")
    print(f"TOP 10 SIGNAL EVENTS BY MEAN DISTANCE TO MISCLASSIFIED SIGNALS")
    print(f"{'='*60}")
    print(f"Max neighbors per event: {max_neighbors}")
    
    # Prepare data for saving
    save_data = {
        'correct_signal_indices': [],
        'correct_signal_predictions': [],
        'correct_signal_hlv': [],
        'correct_signal_track': [],
        'correct_signal_cluster': [],
        'incorrect_neighbor_indices': [],
        'incorrect_neighbor_predictions': [],
        'incorrect_neighbor_hlv': [],
        'incorrect_neighbor_track': [],
        'incorrect_neighbor_cluster': [],
        'distance_scores': [],
        'mean_distances': []
    }
    
    for rank, idx in enumerate(top_indices):
        signal_idx = signal_correct_indices[idx]
        signal_pred = signal_data['predictions'][signal_idx]
        neighbors = neighbors_list[idx]
        distances = distances_list[idx]
        
        print(f"\n--- Rank {rank+1}: Signal Event Index {signal_idx} ---")
        print(f"Signal prediction: {signal_pred:.4f}")
        print(f"Mean distance to misclassified signals: {mean_distances[idx]:.4f}")
        print(f"Number of neighbors found: {len(neighbors)}")
        
        # Store data
        save_data['correct_signal_indices'].append(signal_idx)
        save_data['correct_signal_predictions'].append(signal_pred)
        save_data['correct_signal_hlv'].append(signal_data['hlv'][signal_idx])
        save_data['correct_signal_track'].append(signal_data['track'][signal_idx])
        save_data['correct_signal_cluster'].append(signal_data['cluster'][signal_idx])
        save_data['mean_distances'].append(mean_distances[idx])
        
        if len(neighbors) > 0:
            neighbor_global_indices = signal_incorrect_indices[neighbors]
            save_data['incorrect_neighbor_indices'].append(neighbor_global_indices)
            save_data['incorrect_neighbor_predictions'].append(signal_data['predictions'][neighbor_global_indices])
            save_data['incorrect_neighbor_hlv'].append(signal_data['hlv'][neighbor_global_indices])
            save_data['incorrect_neighbor_track'].append(signal_data['track'][neighbor_global_indices])
            save_data['incorrect_neighbor_cluster'].append(signal_data['cluster'][neighbor_global_indices])
            save_data['distance_scores'].append(distances)
            
            print(f"Nearest misclassified signal neighbors:")
            for j, (neighbor_idx, dist) in enumerate(zip(neighbors, distances)):
                global_idx = signal_incorrect_indices[neighbor_idx]
                neighbor_pred = signal_data['predictions'][global_idx]
                print(f"  {j+1}. Index {global_idx}, Prediction: {neighbor_pred:.4f}, Distance: {dist:.4f}")
        else:
            # Empty arrays for events with no neighbors
            save_data['incorrect_neighbor_indices'].append(np.array([]))
            save_data['incorrect_neighbor_predictions'].append(np.array([]))
            save_data['incorrect_neighbor_hlv'].append(np.array([]))
            save_data['incorrect_neighbor_track'].append(np.array([]))
            save_data['incorrect_neighbor_cluster'].append(np.array([]))
            save_data['distance_scores'].append(np.array([]))
    
    # Convert lists to arrays
    for key in ['correct_signal_indices', 'correct_signal_predictions', 'mean_distances']:
        save_data[key] = np.array(save_data[key])
    for key in ['correct_signal_hlv', 'correct_signal_track', 'correct_signal_cluster']:
        save_data[key] = np.array(save_data[key])
    
    # Save results
    output_file = os.path.join(output_dir, f'signal_to_signal_top10_thresh{prediction_threshold}.npz')
    np.savez(output_file, **save_data)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create visualization
    create_visualization(save_data, signal_data, output_dir, prediction_threshold)
    
    return save_data

def create_visualization(save_data, signal_data, output_dir, prediction_threshold):
    """Create summary plots for the analysis - matching nearest_neighbors_analysis style."""
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Distribution of distance scores
    plt.subplot(2, 2, 1)
    all_distances = np.concatenate([x for x in save_data['distance_scores'] if len(x) > 0])
    plt.hist(all_distances, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distance Scores')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Correct signal vs incorrect neighbor predictions
    plt.subplot(2, 2, 2)
    correct_preds = save_data['correct_signal_predictions']
    # Get mean predictions for neighbors of each signal
    mean_neighbor_preds = []
    for i in range(len(correct_preds)):
        if len(save_data['incorrect_neighbor_predictions'][i]) > 0:
            mean_neighbor_preds.append(np.mean(save_data['incorrect_neighbor_predictions'][i]))
        else:
            mean_neighbor_preds.append(0)
    mean_neighbor_preds = np.array(mean_neighbor_preds)
    
    plt.scatter(correct_preds, mean_neighbor_preds, alpha=0.7)
    plt.xlabel('Correct Signal Predictions')
    plt.ylabel('Mean Incorrect Neighbor Predictions')
    plt.title('Correct Signal vs Incorrect Neighbor Predictions')
    plt.plot([prediction_threshold, 1], [0, 0.5], 'r--', alpha=0.5, label=f'Threshold={prediction_threshold}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: HLV feature comparison (Jet pT)
    plt.subplot(2, 2, 3)
    signal_pt = save_data['correct_signal_hlv'][:, 0]  # Jet pT for correct signals
    # Flatten all neighbor pT values
    neighbor_pt = []
    for i in range(len(save_data['incorrect_neighbor_hlv'])):
        if len(save_data['incorrect_neighbor_hlv'][i]) > 0:
            neighbor_pt.extend(save_data['incorrect_neighbor_hlv'][i][:, 0])
    neighbor_pt = np.array(neighbor_pt)
    
    plt.hist(signal_pt, bins=30, alpha=0.5, label='Correct Signals', color='blue')
    plt.hist(neighbor_pt, bins=30, alpha=0.5, label='Incorrect Neighbors', color='red')
    plt.xlabel('Jet pT [GeV]')
    plt.ylabel('Frequency')
    plt.title('Jet pT Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Distance score vs signal prediction
    plt.subplot(2, 2, 4)
    plt.scatter(correct_preds, save_data['mean_distances'], alpha=0.7)
    plt.xlabel('Correct Signal Predictions')
    plt.ylabel('Mean Distance')
    plt.title('Signal Confidence vs Mean Distance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'signal_to_signal_analysis_distance_thresh{prediction_threshold}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Summary plots saved to: {plot_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find nearest incorrectly predicted signal events to correctly predicted signal events')
    parser.add_argument('--signal_file', default='lstm_1/output_ditau_all.npz', 
                       help='NPZ file with signal predictions')
    parser.add_argument('--prediction_threshold', type=float, default=0.98,
                       help='Prediction threshold for selecting high-confidence correct signals')
    parser.add_argument('--max_neighbors', type=int, default=10,
                       help='Maximum number of neighbors per event')
    parser.add_argument('--output_dir', default='signal_to_signal_analysis_distance',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--n-jobs', type=int, default=None,
                       help='Number of parallel jobs (default: auto, -1 for all CPUs)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing correct events to save memory')
    parser.add_argument('--debug-subset', type=int, default=1000,
                       help='Run on a small random subset of N events for debugging')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run analysis
    results = analyze_signal_to_signal_lstm(
        signal_file=args.signal_file,
        prediction_threshold=args.prediction_threshold,
        max_neighbors=args.max_neighbors,
        output_dir=args.output_dir,
        use_parallel=not args.no_parallel,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        debug_subset=args.debug_subset
    )
    
    if results is not None:
        print(f"\nAnalysis complete! Check {args.output_dir} for detailed results.")
