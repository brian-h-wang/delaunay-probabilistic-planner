"""
Create bar plots showing the success rate over the navigation experiment.

Usage:
python make_results_bar_plots -i1 path/to/low/safety/results -i2 path/to/high/safety/results

"""

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List


def density_label(rho: float) -> str:
    return r"$\rho$ = %.1f trees/m$^2$" % rho


def read_results_into_dataframe(results_path: Path, n_hyp_list: List[int]) -> pd.DataFrame:
    data_cols = ["world_number", "n_hyp", "weight_dist", "weight_safe", "status", "time"]

    all_data = []

    densities = [(0.1, "lowdensity"), (0.2, "defaultdensity"), (0.3, "highdensity")]

    # Load all data over different planners and forest densities into a pandas dataframe
    for density_val, density_str in densities:
        lines = []

        for n_hyp in n_hyp_list:
            if n_hyp == 1:
                filename = "1Hyp.txt"
            elif n_hyp == 0:
                filename = "baseline.txt"
            else:
                filename = "%dHyp.txt" % n_hyp
            filename = "%s_%s" % (density_str, filename)
            try:
                d = pd.read_csv(results_path / filename, names=data_cols)
                d["density"] = density_label(density_val)
                d["density_value"] = density_val
                all_data.append(d)
            except FileNotFoundError:
                print("Couldn't find file %s" % (results_path / filename))
                pass

    data = pd.concat(all_data, axis=0, ignore_index=True)
    data["status"] = [s.strip() for s in data["status"]]

    # data_success = data.loc[data["status"] == "success"]
    # data_success_or_stop = data.loc[data["status"] != "crashed"]
    return data


# def make_success_rate_plot(ax, data_success, data_success_or_stop, title=""):
#     sns.countplot(x="n_hyp", data=data_success, hue="density", ax=ax, palette="Set2", zorder=4)
#     sns.countplot(x="n_hyp", data=data_success_or_stop, hue="density", ax=ax, palette="Set2", saturation=0.2, zorder=3)
#     # TODO hack to make legend work with successes/stops separate bars
#     ax.legend([density_label(r) for r in [0.1, 0.2, 0.3]], ncol=3, loc="upper center")
#     # ax_success.legend([density_label(r) for r in [0.2, 0.3]], ncol=3, loc="upper center")
#     # ax_success.set_ylim(0, 116)
#     y_min = 0
#     n_forests = 20
#     y_max = n_forests + 2.5  # leave some extra space at the top
#     ax.set_ylim(y_min, y_max)
#     # ax_success.set_xticks(n_hyp_list)
#     if not title:
#         title = "Navigation successes vs. number of hypotheses"
#     ax.set_title(title)
#     ax.set_xlabel("Number of hypotheses")
#     ax.set_ylabel("Number of successes")

#     # Set the xticklabels to clearly indicate 0 hypotheses as the baseline
#     xticks = ax.get_xticklabels()
#     for xt in xticks:
#         if xt.get_text() == "0":
#             xt.set_text("0 (baseline)")
#     ax.set_xticklabels(xticks)

#     # Set y tick labels to integers
#     ax.set_yticks(np.arange(y_min, n_forests + 1, 1))


def make_success_rate_plot(
    ax,
    baseline_data: pd.DataFrame,
    planner_low_safety_data: pd.DataFrame,
    planner_high_safety_data: pd.DataFrame,
    title: str = "",
):
    # Populate the "method" field for all data inputs
    baseline_data["method"] = "Baseline"
    planner_low_safety_data["method"] = "Planner, $p_{target} = 0.95$"
    planner_high_safety_data["method"] = "Planner, $p_{target} = 0.999$"

    # Combine data into one dataframe, to count the number of successes and stops in total
    data_combined = pd.concat([baseline_data, planner_low_safety_data, planner_high_safety_data])
    data_success = data_combined[data_combined["status"] == "success"]
    data_success_or_stop = data_combined[data_combined["status"] != "crashed"]

    sns.countplot(x="method", data=data_success, hue="density", ax=ax, palette="Set2", zorder=4)
    sns.countplot(x="method", data=data_success_or_stop, hue="density", ax=ax, palette="Set2", saturation=0.2, zorder=3)
    # TODO hack to make legend work with successes/stops separate bars
    ax.legend([density_label(r) for r in [0.1, 0.2, 0.3]], ncol=3, loc="upper center")
    # ax.legend(["Baseline", "Planner, $p_{target} = 0.95$", "Planner, $p_{target} = 0.999$"], ncol=3, loc="upper center")
    # ax_success.legend([density_label(r) for r in [0.2, 0.3]], ncol=3, loc="upper center")
    # ax_success.set_ylim(0, 116)
    y_min = 0
    n_forests = 20
    y_max = n_forests + 2.5  # leave some extra space at the top
    ax.set_ylim(y_min, y_max)
    # ax_success.set_xticks(n_hyp_list)
    if not title:
        title = "Navigation successes vs. forest density"
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of successes")

    # Set the xticklabels to clearly indicate 0 hypotheses as the baseline
    xticks = ax.get_xticklabels()
    for xt in xticks:
        if xt.get_text() == "0":
            xt.set_text("0 (baseline)")
    ax.set_xticklabels(xticks)

    # Set y tick labels to integers
    ax.set_yticks(np.arange(y_min, n_forests + 1, 1))


sns.set()
sns.set_style("darkgrid")

plt.rcParams["font.family"] = "serif"

parser = argparse.ArgumentParser()
# Get the baseline results from the low-safety directory
# Both the low- and high-safety results will also contain baseline results, and they are the same in both places
parser.add_argument(
    "--input_dir",
    "-i",
    type=str,
    help="Path to input directory for results.",
)
# parser.add_argument(
#     "--input_dir_low_safety",
#     "-i1",
#     type=str,
#     help="Path to input directory for baseline results, and planner results with low safety threshold.",
# )
# parser.add_argument(
#     "--input_dir_high_safety",
#     "-i2",
#     type=str,
#     help="Path to input directory for planner results with high safety threshold.",
# )
args = parser.parse_args()

# input_dir_low_safety = Path(args.input_dir_low_safety)
# input_dir_high_safety = Path(args.input_dir_high_safety)

input_path = Path(args.input_dir)

results_path_baseline_and_low_safety_uniform = input_path / "uniform-lowsafety"
results_path_high_safety_uniform = input_path / "uniform-highsafety"
results_path_baseline_and_low_safety_clusters = input_path / "clusters-lowsafety"
results_path_high_safety_clusters = input_path / "clusters-highsafety"

# titles = ["Desired safety threshold $p_{target}$ = 0.95", "Desired safety threshold $p_{target}$ = 0.999"]

plt.close("all")
fig: plt.Figure = plt.figure()
# ax_time: plt.Axes = fig.add_subplot(1,2,1)
axes_list = []
# for i, (results_dir, title) in enumerate(zip([input_dir_baseline, input_dir_low_safety, input_dir_high_safety], titles)):
#     ax_success: plt.Axes = fig.add_subplot(1, 2, i + 1)
#     # data_success, data_success_or_stop = read_results_into_dataframes(results_dir)
#     data data_success_or_stop = read_results_into_dataframe(results_dir)
#     data_success, data_success_or_stop = read_results_into_dataframes(results_dir)
#     # make_time_plot(ax_time, data_success)
#     make_success_rate_plot(ax_success, data_success, data_success_or_stop, title)
#     axes_list.append(ax_success)

# Create dataframes for the baseline, low-safety planner, and high-safety planner results
data_baseline_uniform = read_results_into_dataframe(
    results_path=results_path_baseline_and_low_safety_uniform, n_hyp_list=[0]
)
data_low_safety_uniform = read_results_into_dataframe(
    results_path=results_path_baseline_and_low_safety_uniform, n_hyp_list=[1]
)
data_high_safety_uniform = read_results_into_dataframe(results_path=results_path_high_safety_uniform, n_hyp_list=[1])

# Combine the data into a plot
ax_uniform = fig.add_subplot(1, 2, 1)
make_success_rate_plot(ax_uniform, data_baseline_uniform, data_low_safety_uniform, data_high_safety_uniform)
ax_uniform.set_title("Forests with uniform obstacles")

# Repeat for cluster forests
data_baseline_clusters = read_results_into_dataframe(
    results_path=results_path_baseline_and_low_safety_clusters, n_hyp_list=[0]
)
data_low_safety_clusters = read_results_into_dataframe(
    results_path=results_path_baseline_and_low_safety_clusters, n_hyp_list=[1]
)
data_high_safety_clusters = read_results_into_dataframe(results_path=results_path_high_safety_clusters, n_hyp_list=[1])

# Combine the data into a plot
ax_clusters = fig.add_subplot(1, 2, 2)
make_success_rate_plot(ax_clusters, data_baseline_clusters, data_low_safety_clusters, data_high_safety_clusters)
ax_clusters.set_title("Forests with Gaussian obstacle clusters")

# Set figure title and layout
# subtitle = "navigation successes versus number of planner hypotheses, compared to baseline"
title = "Navigation successes per method, for different forest layouts and obstacle density $\\rho$"
fig.suptitle(title, fontweight="bold", fontsize=13)
fig_height = 6
fig_width = fig_height * 2.4
fig.set_size_inches(fig_width, fig_height)
fig.tight_layout()

fig.savefig("navigation_successes_per_method.png", dpi=400)
# plt.show()
