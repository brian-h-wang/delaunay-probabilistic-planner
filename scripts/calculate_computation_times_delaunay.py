"""
Calculate the average and std dev Delaunay triangulation computation times for each density and print to console.
"""

import numpy as np
import os

# Directory containing the files
base_dir = "results/3-2025-revision/clusters-lowsafety/"

# Densities and safety thresholds to check
densities = ["lowdensity", "defaultdensity", "highdensity"]
ptargets = ["0.950", "0.997", "0.999"]

# Store results
results = {}

for density in densities:
    all_times = []

    for p in ptargets:
        filename = f"time_counts_{density}_1Hyp_{p}ptarget.npz"
        path = os.path.join(base_dir, filename)

        try:
            data = np.load(path)
            times = data["delaunay"] * 1000.0  # convert from seconds to milliseconds
            all_times.append(times)
        except Exception as e:
            print(f"Failed to load or process {path}: {e}")

    if all_times:
        concatenated = np.concatenate(all_times)
        mean_time = np.mean(concatenated)
        std_time = np.std(concatenated)
        results[density] = (mean_time, std_time)
    else:
        results[density] = (None, None)

# Print results
for density, (mean_time, std_time) in results.items():
    if mean_time is not None:
        print(f"{density}: mean = {mean_time:.6f} ms, std = {std_time:.4f} ms")
    else:
        print(f"{density}: No data found or error in processing.")
