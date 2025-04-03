"""
Calculate percent decrease in global planning times.
"""

import numpy as np
import os

# Base directory where the data is stored
base_dir = "results/3-2025-revision/clusters-lowsafety/"

# Forest densities to evaluate
densities = ["lowdensity", "defaultdensity", "highdensity"]

# Mapping between planner safety thresholds and baseline inflation levels
matchups = {"0.950": "2.00", "0.997": "3.00", "0.999": "4.00"}


# Function to compute percent decrease
def percent_decrease(baseline, planner):
    return 100.0 * (baseline - planner) / baseline


# Store per-density percent decreases
results = {}

for density in densities:
    percent_decreases = []

    for ptarget, nstd in matchups.items():
        planner_file = f"time_counts_{density}_1Hyp_{ptarget}ptarget.npz"
        baseline_file = f"time_counts_{density}_baseline_{nstd}std.npz"
        planner_path = os.path.join(base_dir, planner_file)
        baseline_path = os.path.join(base_dir, baseline_file)

        try:
            planner_data = np.load(planner_path)
            baseline_data = np.load(baseline_path)

            planner_times = planner_data["global_planner"]
            baseline_times = baseline_data["baseline_global_planner"]

            planner_mean = np.mean(planner_times)
            baseline_mean = np.mean(baseline_times)

            pct_decrease = percent_decrease(baseline_mean, planner_mean)
            percent_decreases.append(pct_decrease)

        except Exception as e:
            print(f"Error processing {density}, ptarget {ptarget}: {e}")

    if percent_decreases:
        overall_mean = np.mean(percent_decreases)
        results[density] = overall_mean
    else:
        results[density] = None

# Print final aggregated results
print("Average percent decrease in global planner time per forest density:")
for density, mean_pct in results.items():
    if mean_pct is not None:
        print(f"  {density:<15}: {mean_pct:.2f}%")
    else:
        print(f"  {density:<15}: ERROR")
