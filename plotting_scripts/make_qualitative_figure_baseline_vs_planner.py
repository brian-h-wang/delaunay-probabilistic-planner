"""
Script to create qualitative figure 3

Forest 17 high density
Plot baseline, 1 hyp, 5 hyp all together.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import qualitative_figures_format

# TRAJ_FILE_NAMES = ["1hyp.npz", "5hyp.npz", "baseline.npz"]
# LABELS = [
#     "1 hypothesis\nTime to goal: 36.00s",
#     "5 hypotheses\nTime to goal: 28.07s",
#     "Hybrid A* baseline\nRobot crashed",
# ]

TRAJ_FILE_NAMES = ["1hyp.npz", "1hyp_highsafety.npz", "baseline.npz"]
LABELS = ["Navigation graph planner", "Navigation graph planner, high safety", "Baseline planner"]
STYLES = ["-", "--", (0, (1.5, 0.75))]

all_forest_numbers = [3, 8, 17]

for forest_number in all_forest_numbers:

    input_dir = Path(f"trajectories/10-6-2024-1hyp-trajectories/forest{forest_number}_highdensity_clusters")

    # Load in the trajectories
    trajectories_list = []
    replan_time_stamps_list = []
    not_found_indices = []
    forest_xyr = None
    for i, traj_file_name in enumerate(TRAJ_FILE_NAMES):
        try:
            npz_data = np.load(str(input_dir / traj_file_name))
        except FileNotFoundError:
            print("Could not find file '%s'" % (input_dir / traj_file_name))
            not_found_indices.append(i)
            continue
        robot_txy = npz_data["robot_txy"]
        forest_xyr = npz_data["forest_xyr"]  # forest is overwritten, is same for all trajectories
        replan_time_stamps = npz_data["replan_time_stamps"]
        trajectories_list.append(robot_txy)
        replan_time_stamps_list.append(replan_time_stamps)

    if forest_xyr is None:
        raise FileNotFoundError("No trajectory files found!")

    # Remove entries from the file names and labels lists to handle any missing trajectories
    for i in not_found_indices:
        TRAJ_FILE_NAMES.pop(i)
        LABELS.pop(i)

    # Initialize the figure
    fig = plt.figure(figsize=(16, 4))

    ax = qualitative_figures_format.new_ax(fig, forest_xyr, fontsize=16)

    # Plot the trajectories
    me = 100  # markevery
    for traj_txy, label, style in zip(trajectories_list, LABELS, STYLES):
        x = traj_txy[:, 1]
        y = traj_txy[:, 2]
        lw = 6
        ms = 15
        (lh,) = ax.plot(x, y, linestyle=style, label=label, markevery=me, linewidth=lw, markersize=ms)
        # If robot crashed, add an X at the end
        if "crashed" in label:
            ax.plot(x[-1], y[-1], "x", markersize=12, markeredgewidth=4, color=lh.get_color())

    # Plot start, goal
    def plot_start_and_goal(ax):
        start_position = np.array([0.0, 5.0])
        goal_position = np.array([40, 5], dtype=float)
        cell_vertex_size = 18
        start_color = (0.8, 0.4, 0.4)
        goal_color = start_color
        (start_handle,) = ax.plot(
            start_position[0],
            start_position[1],
            "^",
            c=start_color,
            markersize=cell_vertex_size,
            zorder=11,
        )
        (goal_handle,) = ax.plot(
            goal_position[0], goal_position[1], "*", c=goal_color, markersize=cell_vertex_size, zorder=11
        )

    plot_start_and_goal(ax)

    ax.legend(prop={"size": 20}, ncol=3)

    # plt.tight_layout()
    fig.tight_layout()
    # plt.show()
    output_dir = Path("FiguresOutput/qualitative")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    plt.savefig(output_dir / f"qualitative_forest{forest_number}.png", bbox_inches="tight")
