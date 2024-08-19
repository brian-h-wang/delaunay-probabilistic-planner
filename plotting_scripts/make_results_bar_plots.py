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
    baseline_data["method"] = "baseline"
    planner_low_safety_data["method"] = "planner_low_safety"
    planner_high_safety_data["method"] = "planner_high_safety"

    # Combine data into one dataframe, to count the number of successes and stops in total
    data_combined = pd.concat([baseline_data, planner_low_safety_data, planner_high_safety_data])
    data_success = data_combined[data_combined["status"] == "success"]
    data_success_or_stop = data_combined[data_combined["status"] != "crashed"]

    sns.countplot(x="density", data=data_success, hue="method", ax=ax, palette="Set2", zorder=4)
    sns.countplot(x="density", data=data_success_or_stop, hue="method", ax=ax, palette="Set2", saturation=0.2, zorder=3)
    # TODO hack to make legend work with successes/stops separate bars
    # ax.legend([density_label(r) for r in [0.1, 0.2, 0.3]], ncol=3, loc="upper center")
    ax.legend(["Baseline", "Planner, $p_{target} = 0.95$", "Planner, $p_{target} = 0.999$"], ncol=3, loc="upper center")
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
    ax.set_xlabel("Forest density")
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
    "--input_dir_low_safety",
    "-i1",
    type=str,
    help="Path to input directory for baseline results, and planner results with low safety threshold.",
)
parser.add_argument(
    "--input_dir_high_safety",
    "-i2",
    type=str,
    help="Path to input directory for planner results with high safety threshold.",
)
args = parser.parse_args()

input_dir_low_safety = Path(args.input_dir_low_safety)
input_dir_high_safety = Path(args.input_dir_high_safety)

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
data_baseline = read_results_into_dataframe(results_path=input_dir_low_safety, n_hyp_list=[0])
data_low_safety = read_results_into_dataframe(results_path=input_dir_low_safety, n_hyp_list=[1])
data_high_safety = read_results_into_dataframe(results_path=input_dir_high_safety, n_hyp_list=[1])

# Combine the data into a plot
ax = fig.add_subplot(1, 1, 1)
make_success_rate_plot(ax, data_baseline, data_low_safety, data_high_safety)

# Set figure title and layout
# subtitle = "navigation successes versus number of planner hypotheses, compared to baseline"
subtitle = "navigation successes versus forest density"
if "cluster" in str(input_dir_low_safety):
    title = "Gaussian cluster forests: " + subtitle
elif "uniform" in str(input_dir_low_safety):
    title = "Uniformly distributed forests: " + subtitle
else:
    title = "DEFAULT TITLE"
if title:
    fig.suptitle(title, fontweight="bold", fontsize=13)
fig_height = 6
fig_width = fig_height * 2.4
fig.set_size_inches(fig_width, fig_height)
fig.tight_layout()
plt.show()
