#!/usr/bin/env python3

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt

def preprocess_all_data():
    folder_pairs = [
        ('../ADHD_data/ADHD_part1',    '../ADHD_data/ADHD_part1_preprocessed'),
        ('../ADHD_data/ADHD_part2',    '../ADHD_data/ADHD_part2_preprocessed'),
        ('../ADHD_data/Control_part1', '../ADHD_data/Control_part1_preprocessed'),
        ('../ADHD_data/Control_part2', '../ADHD_data/Control_part2_preprocessed')
    ]
    
    fs = 128
    low_cut = 1.0
    high_cut = 40.0
    filter_order = 4
    artifact_threshold = 1000.0
    min_length_for_filter = 3 * (2 * filter_order)

    # Design Butterworth (1-40 Hz)
    b, a = butter(filter_order, [low_cut/(fs/2), high_cut/(fs/2)], btype='band')

    for input_dir, output_dir in folder_pairs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n=== Processing folder: {input_dir} ===")
        
        mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
        
        if not mat_files:
            print(f"  * WARNING: No .mat files found in: {input_dir}")
            continue
        
        for i, mat_file in enumerate(mat_files, start=1):
            in_path = os.path.join(input_dir, mat_file)
            print(f"\n -> Processing file [{i}/{len(mat_files)}]: {mat_file}")
            
            data_struct = loadmat(in_path)
            var_names = [k for k in data_struct.keys() if not k.startswith('__')]
            if not var_names:
                print(f"   * ERROR: No data variable found in {mat_file}")
                continue
            
            raw_data = data_struct[var_names[0]]
            if not isinstance(raw_data, np.ndarray) or raw_data.ndim != 2:
                print(f"   * ERROR: {var_names[0]} is not a 2D array.")
                continue
            
            rows, cols = raw_data.shape
            # FIXED: Transpose if ROWS > COLS (time,channels => channels,time)
            if rows > cols:
                raw_data = raw_data.T
                rows, cols = raw_data.shape
            
            channels = rows
            time_points = cols
            
            # Filter if enough samples
            if time_points < min_length_for_filter:
                print(f"    * WARNING: Data length ({time_points}) < {min_length_for_filter}. Skipping filter.")
                filtered_data = raw_data.copy()
            else:
                filtered_data = np.zeros_like(raw_data)
                for ch in range(channels):
                    filtered_data[ch, :] = filtfilt(b, a, raw_data[ch, :])
            
            # Artifact threshold
            artifact_mask = np.abs(filtered_data) > artifact_threshold
            cleaned_data = filtered_data.copy().astype(float)
            cleaned_data[artifact_mask] = np.nan
            
            # Z-score normalization per channel (ignore NaN)
            normalized_data = np.zeros_like(cleaned_data)
            for ch in range(channels):
                chan_data = cleaned_data[ch, :]
                good_idx = ~np.isnan(chan_data)
                if np.any(good_idx):
                    mu = np.mean(chan_data[good_idx])
                    sigma_val = np.std(chan_data[good_idx])
                    if sigma_val < 1e-10:
                        sigma_val = 1.0
                    normalized_data[ch, good_idx] = (chan_data[good_idx] - mu) / sigma_val
                else:
                    normalized_data[ch, :] = np.nan
            
            preprocessed_data = normalized_data
            base_name, _ = os.path.splitext(mat_file)
            out_file_name = base_name + '_preprocessed.mat'
            out_path = os.path.join(output_dir, out_file_name)
            
            savemat(out_path, {
                'preprocessedData': preprocessed_data,
                'fs': fs
            })
            print(f"    -> Saved: {out_file_name}")
        
        print(f"=== Finished folder: {input_dir} ===")

    print("\n*** All folders processed! ***\n")


if __name__ == "__main__":
    preprocess_all_data()
