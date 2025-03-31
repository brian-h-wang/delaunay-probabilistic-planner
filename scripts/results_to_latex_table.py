"""
Read success/stop/crash rate, and avg time to goal, from results
and print the formatted LaTeX table to the console.
"""

import os
from collections import Counter

# Base path to your results directory
base_path = "results/3-2025-revision/clusters-lowsafety"

# Configurations for each method
density_labels = {
    "lowdensity": r"$\rho = 0.1\ \mathrm{trees}/\mathrm{m}^2$",
    "defaultdensity": r"$\rho = 0.2\ \mathrm{trees}/\mathrm{m}^2$",
    "highdensity": r"$\rho = 0.3\ \mathrm{trees}/\mathrm{m}^2$",
}

densities = ["lowdensity", "defaultdensity", "highdensity"]

baseline_files = ["baseline_0.00std", "baseline_2.00std", "baseline_3.00std", "baseline_4.00std"]

planner_files = ["1Hyp_0.950ptarget", "1Hyp_0.997ptarget", "1Hyp_0.999ptarget"]


def load_results(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            status = parts[2].strip()
            time = float(parts[3].strip())
            results.append((status, time))
    return results


def compute_stats(results):
    total = len(results)
    counter = Counter([r[0] for r in results])
    successes = [time for status, time in results if status == "success"]
    success_rate = counter["success"] / total * 100
    stop_rate = counter["stopped"] / total * 100
    crash_rate = counter["crashed"] / total * 100
    avg_time = sum(successes) / len(successes) if successes else 0.0
    return f"{success_rate:.0f}\\%", f"{stop_rate:.0f}\\%", f"{crash_rate:.0f}\\%", f"{avg_time:.2f}"


# Method display names (in order)
method_labels = [
    r"Baseline ($n_\sigma = 0$)",
    r"Baseline ($n_\sigma = 2$)",
    r"Planner ($p_\text{{target}} = 0.95$)",
    r"Baseline ($n_\sigma = 3$)",
    r"Planner ($p_\text{{target}} = 0.997$)",
    r"Baseline ($n_\sigma = 4$)",
    r"Planner ($p_\text{{target}} = 0.999$)",
]

# Match filenames to method_labels
file_sequence = [
    lambda d: f"{d}_baseline_0.00std.txt",
    lambda d: f"{d}_baseline_2.00std.txt",
    lambda d: f"{d}_1Hyp_0.950ptarget.txt",
    lambda d: f"{d}_baseline_3.00std.txt",
    lambda d: f"{d}_1Hyp_0.997ptarget.txt",
    lambda d: f"{d}_baseline_4.00std.txt",
    lambda d: f"{d}_1Hyp_0.999ptarget.txt",
]

# Start printing LaTeX rows
print(r"\midrule")

for i, label in enumerate(method_labels):
    row_values = [label]
    for density in densities:
        filename = file_sequence[i](density)
        full_path = os.path.join(base_path, filename)
        stats = compute_stats(load_results(full_path))
        row_values.extend(stats)
    latex_row = " & ".join(row_values) + r" \\"
    print(latex_row)
    if "Planner" in label:
        print(r"\midrule")

print(r"\bottomrule")
