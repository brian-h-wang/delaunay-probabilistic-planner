"""
Read success/stop/crash rate, and avg time to goal, from results
and print the formatted LaTeX table to the console.
"""

import os
from collections import Counter
from statistics import mean, stdev
from scipy.stats import chi2_contingency

base_path = "results/3-2025-revision50/clusters-lowsafety"
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
    return {
        "success_rate": counter["success"] / total * 100,
        "stop_rate": counter["stopped"] / total * 100,
        "crash_rate": counter["crashed"] / total * 100,
        "avg_time": mean(success_times) if success_times else None,
        "time_std": stdev(success_times) if len(success_times) > 1 else 0.0,
        "raw_counts": counter,
        "total": total,
    }


# Load results
table_data = [[] for _ in method_labels]
for i, label in enumerate(method_labels):
    for density in densities:
        filename = file_sequence[i](density)
        full_path = os.path.join(base_path, filename)
        stats = compute_stats(load_results(full_path))
        table_data[i].append(stats)


# Compute p-values
p_values_full = {}  # for 3-category outcome
p_values_binary = {}  # for binary outcome
for i in [1, 3, 5]:  # Baseline rows paired with the planner below
    base_total = Counter()
    plan_total = Counter()
    for d in range(3):
        base_total.update(table_data[i][d]["raw_counts"])
        plan_total.update(table_data[i + 1][d]["raw_counts"])

    # Full 3-category test
    contingency_full = [
        [base_total["success"], base_total["stopped"], base_total["crashed"]],
        [plan_total["success"], plan_total["stopped"], plan_total["crashed"]],
    ]
    _, p_full, _, _ = chi2_contingency(contingency_full)
    p_values_full[i] = p_full

    # Binary test: success vs. not success
    base_success = base_total["success"]
    base_not = base_total["stopped"] + base_total["crashed"]
    plan_success = plan_total["success"]
    plan_not = plan_total["stopped"] + plan_total["crashed"]
    contingency_bin = [
        [base_success, base_not],
        [plan_success, plan_not],
    ]
    _, p_bin, _, _ = chi2_contingency(contingency_bin)
    p_values_binary[i] = p_bin

# Print LaTeX table
print(r"\midrule")
for i, (label, row) in enumerate(zip(method_labels, table_data)):
    if i == 1:
        print(r"\midrule")
    latex_row = [label]
    for col, stats in enumerate(row):
        sr = f"{stats['success_rate']:.0f}\\%"
        stop = f"{stats['stop_rate']:.0f}\\%"
        crash = f"{stats['crash_rate']:.0f}\\%"
        if stats["avg_time"] is None:
            time_str = r"\textit{N/A}"
        else:
            time_str = f"{stats['avg_time']:.2f} ± {stats['time_std']:.2f}"
        latex_row.extend([sr, stop, crash, time_str])

    # P-value columns
    if i == 0:
        latex_row.extend(["-", "-"])
    elif i in p_values_full:
        latex_row.extend(
            [
                rf"\multirow{{2}}{{*}}{{${p_values_full[i]:.2g}$}}",
                rf"\multirow{{2}}{{*}}{{${p_values_binary[i]:.2g}$}}",
            ]
        )
    else:
        latex_row.extend(["", ""])

    print(" & ".join(latex_row) + r" \\")
    if "Planner" in label:
        print(r"\midrule")
print(r"\bottomrule")
