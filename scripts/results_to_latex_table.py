"""
Read success/stop/crash rate, and avg time to goal, from results
and print the formatted LaTeX table to the console.
"""

import os
from collections import Counter
from statistics import mean, stdev

base_path = "results/3-2025-revision/clusters-lowsafety"
densities = ["lowdensity", "defaultdensity", "highdensity"]

method_labels = [
    r"Baseline ($n_\sigma = 0$)",
    r"Baseline ($n_\sigma = 2$)",
    r"Planner ($p_\text{{target}} = 0.95$)",
    r"Baseline ($n_\sigma = 3$)",
    r"Planner ($p_\text{{target}} = 0.997$)",
    r"Baseline ($n_\sigma = 4$)",
    r"Planner ($p_\text{{target}} = 0.999$)",
]

file_sequence = [
    lambda d: f"{d}_baseline_0.00std.txt",
    lambda d: f"{d}_baseline_2.00std.txt",
    lambda d: f"{d}_1Hyp_0.950ptarget.txt",
    lambda d: f"{d}_baseline_3.00std.txt",
    lambda d: f"{d}_1Hyp_0.997ptarget.txt",
    lambda d: f"{d}_baseline_4.00std.txt",
    lambda d: f"{d}_1Hyp_0.999ptarget.txt",
]


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
    success_times = [time for status, time in results if status == "success"]
    success_rate = counter["success"] / total * 100
    stop_rate = counter["stopped"] / total * 100
    crash_rate = counter["crashed"] / total * 100
    if success_times:
        avg_time = mean(success_times)
        std_time = stdev(success_times) if len(success_times) > 1 else 0.0
    else:
        avg_time = None
        std_time = None
    return {
        "success_rate": success_rate,
        "stop_rate": stop_rate,
        "crash_rate": crash_rate,
        "avg_time": avg_time,
        "time_std": std_time,
    }


# Load all data
table_data = [[] for _ in method_labels]
for i, label in enumerate(method_labels):
    for density in densities:
        filename = file_sequence[i](density)
        full_path = os.path.join(base_path, filename)
        stats = compute_stats(load_results(full_path))
        table_data[i].append(stats)


# Compute column-wide extrema
def get_column_extrema(data, key, func):
    return [func([row[i][key] for row in data]) for i in range(3)]


max_success = get_column_extrema(table_data, "success_rate", max)
min_crash = get_column_extrema(table_data, "crash_rate", min)
min_avg_times = []
for i in range(3):
    times = [row[i]["avg_time"] for row in table_data if row[i]["avg_time"] is not None]
    min_avg_times.append(min(times) if times else None)

# Print LaTeX table rows
print(r"\midrule")

for i, (label, row) in enumerate(zip(method_labels, table_data)):
    if i == 1:
        print(r"\midrule")
    latex_row = [label]
    for col, stats in enumerate(row):
        # Success/stop/crash
        sr = f"{stats['success_rate']:.0f}\\%"
        stop = f"{stats['stop_rate']:.0f}\\%"
        crash = f"{stats['crash_rate']:.0f}\\%"

        if stats["success_rate"] == max_success[col]:
            sr = r"\textbf{" + sr + "}"
        if stats["crash_rate"] == min_crash[col]:
            crash = r"\textbf{" + crash + "}"

        # Time with std
        if stats["avg_time"] is None:
            time_str = r"\textit{N/A}"
        else:
            avg = stats["avg_time"]
            std = stats["time_std"]
            time_str = f"{avg:.2f} ± {std:.2f}"
            if avg == min_avg_times[col]:
                time_str = r"\textbf{" + time_str + "}"

        latex_row.extend([sr, stop, crash, time_str])

    print(" & ".join(latex_row) + r" \\")
    if "Planner" in label:
        print(r"\midrule")

print(r"\bottomrule")
