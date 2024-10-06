"""
Script for plotting navigation graphs from a simulation run.

For each n_hyp in a given n_hyp_list, runs the simulation in a specified forest,
until the planner generates its first navigation graph.

Plot this navigation graph and save to an image, then terminate the simulation.
"""

from pathlib import Path
import random
import numpy as np
from planner.simulation import MultipleHypothesisPlannerSimulation
from planner.forest_generation import PoissonForestUniformRadius, ForestBarriers, PoissonTreeCluster
import copy
import time
from scripts.simulation_defaults import seed, get_default_params, get_default_sensors
from planner.plotting import HighLevelPathDistancePlotter, DistanceGraphPlotter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
from math import floor, ceil

fontsize = 16

# Generate navigation graphs for these n_hypotheses settings:
n_hyp_list = [1]


low_density = 0.1
default_density = 0.2
high_density = 0.3
very_high_density = 0.4
density = high_density

# Specify the forest: seed, density, clusters
# forest_id = 16

all_forest_ids = [5]

for forest_id in all_forest_ids:
    for high_safety in [True, False]:

        use_clusters = True

        # Initialize the simulation

        random.seed(seed)
        np.random.seed(seed)
        print("[INFO] Using base random seed '%d'" % seed)

        x_min, x_max = (-2, 42)
        y_min, y_max = (0, 10)
        start_pose = np.array([0.0, 5.0, 0.0])
        goal_position = np.array([40, 5], dtype=float)

        # Define spaces near the start and goal where no trees will be sampled
        default_params = get_default_params()

        w = default_params.robot_width * 3.0  # no trees will generate within this distance of start and goal
        invalid_spaces = np.array([[start_pose[0], start_pose[1], w], [goal_position[0], goal_position[1], w]])

        # Generate forest environments for experiments
        forests = []

        min_tree_radius = 0.2
        max_tree_radius = 0.5

        # Generate the forest
        i = forest_id
        forest_seed = seed + i
        bounds = [x_min, x_max, y_min, y_max]
        if use_clusters:
            cluster_density = density * 4
            cov = np.diag([1, 1.5]) ** 2

            cluster_means = [np.array([10, 5]), np.array([20, 5]), np.array([30, 5])]

            clusters = [
                PoissonTreeCluster(cluster_density, mean, cov, seed=forest_seed + j)
                for (j, mean) in enumerate(cluster_means)
            ]
        else:
            clusters = None
        forest = PoissonForestUniformRadius(
            bounds=bounds,
            density=density,
            radius_min=min_tree_radius,
            radius_max=max_tree_radius,
            seed=forest_seed,
            invalid_spaces=invalid_spaces,
            clusters=clusters,
        )

        class PlannerSettings(object):

            def __init__(self, n_hypotheses=1, weight_distance=0.5, weight_safety=0.5):
                self.n_hypotheses = n_hypotheses
                self.weight_distance = weight_distance
                self.weight_safety = weight_safety

            def __str__(self):
                if self.n_hypotheses == 1:
                    return "1Hyp"
                elif self.n_hypotheses == 0:
                    return "baseline"
                else:
                    # wd_str = ('%.2f' % self.weight_distance).replace('0.', '')
                    # ws_str = ('%.2f' % self.weight_safety).replace('0.', '')
                    # return "%dHyp_WD%s_WS%s" % (self.n_hypotheses, wd_str, ws_str)
                    return "%dHyp" % self.n_hypotheses

        methods = []
        for n_hyp in n_hyp_list:
            # for weight_s in [0.5, 0.10, 0.90]:
            for weight_s in [0.5]:  # TODO skipping different weights for now
                weight_d = 1.0 - weight_s
                methods.append(PlannerSettings(n_hypotheses=n_hyp, weight_distance=weight_d, weight_safety=weight_s))

        range_sigma_min = 0.05
        range_sigma_max = 0.5
        bearing_std = np.deg2rad(2.5)

        n_attempts = 1

        start_time = time.time()

        fig1: plt.Figure = plt.figure()
        fig2: plt.Figure = plt.figure()
        for fig in [fig1, fig2]:
            # fig.set_size_inches(7.1, 4.8)
            fig.set_size_inches(7.1, 3.6)

        ax_baseline = fig1.add_subplot(1, 1, 1)
        ax_1hyp = fig2.add_subplot(1, 1, 1)
        ax_list = [ax_baseline, ax_1hyp]

        cmap_name = "Set2"
        sns.set_palette(cmap_name)
        cmap = get_cmap(cmap_name)

        n_paths = 3
        path_colors = [cmap.colors[i] for i in range(n_paths)]

        for method, ax in zip(methods, ax_list):
            print("Running sim: %s" % (str(method)))
            # Set up the simulation with multi-hypothesis planner parameters
            params = copy.deepcopy(default_params)
            params.n_shortest_paths = method.n_hypotheses
            params.n_safest_paths = 0
            params.weight_safety = method.weight_safety
            params.weight_distance = method.weight_distance

            rb_sensor, size_sensor = get_default_sensors()

            if high_safety:
                params.safety_probability = 0.999

            sim = MultipleHypothesisPlannerSimulation(
                start_pose=start_pose,
                goal_position=goal_position,
                params=params,
                world=forest,
                range_bearing_sensor=rb_sensor,
                size_sensor=size_sensor,
            )
            sim_start_time = time.time()
            while sim.is_running():
                sim.update()
                # Wait until 2 navigation graphs have been created
                # TODO hardcoded
                if sim.n_replans >= 2:
                    break

            # Get the most recent navigation graph
            # TODO hardcoded getting first 3 paths (5hyp planner only generates 3 paths in this case)
            path_plotter = HighLevelPathDistancePlotter(
                sim.mhp_result,
                ax=ax,
                n_paths=min(n_paths, method.n_hypotheses),
                show_cell_decomposition=True,
                colors=path_colors,
                lw=5,
                no_legend=True,
            )
            # ax.legend(loc="lower right", prop={"size": fontsize})
            graph = sim.mhp_result.distance_graph
            graph_plotter = DistanceGraphPlotter(graph, ax=ax, show_start_and_goal=True)

            # Plot the graph and save to a file

        for ax in ax_list:
            xmin, xmax = (-0.88, 15.81)
            # xmin, xmax = (-2, 20)
            # xmin, xmax = (1.53, 16.47)
            xmin, xmax = (-1.47, 13.47)
            # ymin, ymax = (-2, 12)
            # ymin, ymax = (0.25, 9.75)
            ymin, ymax = (2.25, 7.75)
            # ax.set_title(
            #     f"Navigation graph and planned path, safety threshold {params.safety_probability}", fontsize=fontsize
            # )
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_yticks(np.arange(floor(ymin), ceil(ymax) + 1, 1.0))
            ax.set_xticks(np.arange(floor(xmin), ceil(xmax) + 1, 1.0))
            ax.set_aspect("equal")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid("minor", alpha=0.05, color=(0.2, 0.2, 0.2))

        # plt.show()

        print("Generated figures!")
        fig1.tight_layout()
        fig2.tight_layout()

        output_dir = Path("FiguresOutput")
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        if high_safety:
            fig1.savefig(output_dir / f"nav_graph_1hyp_forest{forest_id}_highsafety.png", bbox_inches="tight")
        else:
            fig1.savefig(output_dir / f"nav_graph_1hyp_forest{forest_id}.png", bbox_inches="tight")
        # fig2.savefig(output_dir / "nav_graph_5hyp.png", bbox_inches="tight")
