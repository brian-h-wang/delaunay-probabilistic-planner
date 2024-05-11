"""
Run the planner on randomly generated Poisson forests, and compare results over multiple trials.

See README.md for usage instructions.

"""

import random
import numpy as np
from planner.simulation import MultipleHypothesisPlannerSimulation, NoUncertaintyBaselineSimulation
from planner.forest_generation import PoissonForestUniformRadius, PoissonTreeCluster
import copy
from datetime import datetime
from pathlib import Path
import time
import argparse
from experiments.simulation_defaults import seed, get_default_params, get_default_sensors
from enum import Enum
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", "-o", required=False, help="Output directory path")
parser.add_argument("--low_density", "-ld", action="store_true")
parser.add_argument("--high_density", "-hd", action="store_true")
parser.add_argument("--high_safety", "-hs", action="store_true")
parser.add_argument("--clusters", "-c", action="store_true")
parser.add_argument("--baselines", "-b", action="store_true", help="Run baseline methods")
parser.add_argument("--speed", "-s", type=float, default=None, help="Maximum robot speed, in m/s.")
parser.add_argument("--n_hyp_list", "-hyp", nargs="+", help="List of number of hypotheses to test", required=False)
parser.add_argument("--debug", "-d", action="store_true", help="Debug mode. Stops sims after 1 second sim time")
args = parser.parse_args()

if args.low_density and args.high_density:
    raise ValueError("Specify either low density or high density, not both")

random.seed(seed)
np.random.seed(seed)
print("[INFO] Using base random seed '%d'" % seed)

if args.clusters:
    print("Adding Gaussian tree clusters")
    use_clusters = True
else:
    use_clusters = False

use_high_safety = args.high_safety

subfolder = ""
if use_clusters:
    subfolder += "clusters"
else:
    subfolder += "uniform"

if use_high_safety:
    subfolder += "-highsafety"
else:
    subfolder += "-lowsafety"

# Set up the output directory
if args.output_dir is None:
    output_dir = Path("results/" + datetime.now().strftime("Experiment_%m-%d-%Y_%H.%M.%S"))
else:
    output_dir = Path(args.output_dir)
output_dir = output_dir / subfolder

if not output_dir.exists():
    output_dir.mkdir(parents=True)


# Set up environment parameters
n_environments = 20
x_min, x_max = (-2, 42)
y_min, y_max = (0, 10)
start_pose = np.array([0.0, 5.0, 0.0])
goal_position = np.array([40, 5], dtype=float)


default_params = get_default_params()

if use_high_safety:
    print("Using high safety setting")
    default_params.safety_probability = 0.999

# Define spaces near the start and goal where no trees will be sampled
w = default_params.robot_width * 3.0  # no trees will generate within this distance of start and goal
invalid_spaces = np.array([[start_pose[0], start_pose[1], w], [goal_position[0], goal_position[1], w]])

# Generate forest environments for experiments
forests = []

low_density = 0.1
default_density = 0.2
high_density = 0.3

if args.low_density:
    density = low_density
    experiment_name = "lowdensity"
elif args.high_density:
    density = high_density
    experiment_name = "highdensity"
else:
    density = default_density
    experiment_name = "defaultdensity"


# Set robot speed
if args.speed is not None:
    default_params.motion_primitive_velocity = args.speed
    print("[INFO] Set robot maximum velocity to %.2f m/s" % default_params.motion_primitive_velocity)

min_tree_radius = 0.2
max_tree_radius = 0.5

# Generate N random forest environments, and run each method in each of the environments
print("[INFO] Generating forest environments...")
for i in range(n_environments):
    print(
        "  Generating forest environment with density %.2f [%d of %d]" % (density, i + 1, n_environments),
        end="\r",
        flush=True,
    )
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
    forests.append(forest)
print("")


class PlannerSettings(object):

    class Planner(Enum):
        BASELINE_NO_UNCERTAINTY = 0
        BASELINE_SAFETY_AWARE_ASTAR = 1
        MULTIPLE_HYPOTHESIS = 2

    def __init__(self, planner_type: Planner, n_hypotheses: int = 1):
        self.planner_type = planner_type
        self.n_hypotheses = n_hypotheses

    def __str__(self):
        if self.planner_type == self.Planner.BASELINE_NO_UNCERTAINTY:
            return "Baseline - No uncertainty"
        elif self.planner_type == self.Planner.BASELINE_SAFETY_AWARE_ASTAR:
            return "Baseline - Safety-aware A* search"
        elif self.planner_type == self.Planner.MULTIPLE_HYPOTHESIS:
            return f"Multiple hypothesis planner with n_hyp={self.n_hypotheses}"
        return ""

    def short_filename(self):
        if self.planner_type == self.Planner.BASELINE_NO_UNCERTAINTY:
            return "baselineNU"
        elif self.planner_type == self.Planner.BASELINE_SAFETY_AWARE_ASTAR:
            return "baselineSAA"
        elif self.planner_type == self.Planner.MULTIPLE_HYPOTHESIS:
            return f"{self.n_hypotheses}hyp"
        return ""


if args.n_hyp_list is not None:
    n_hyp_list = [int(x) for x in args.n_hyp_list]
    print("[INFO] Running comparison with n_hypotheses = " + str(n_hyp_list))
else:
    n_hyp_list = []
    print("[INFO] n_hyp list not provided. Skipping multiple hypothesis planner.")


# Generate the list of planners to run based on the input args

methods: List[PlannerSettings] = []

if args.baselines:
    methods.append(PlannerSettings(planner_type=PlannerSettings.Planner.BASELINE_NO_UNCERTAINTY))
    # methods.append(PlannerSettings(planner_type=PlannerSettings.Planner.BASELINE_SAFETY_AWARE_ASTAR))   # Not implemented

for n_hyp in n_hyp_list:
    weight_s = 0.5
    weight_d = 1.0 - weight_s
    methods.append(PlannerSettings(planner_type=PlannerSettings.Planner.MULTIPLE_HYPOTHESIS, n_hypotheses=n_hyp))

debug_mode = args.debug

start_time = time.time()

for method in methods:
    results_lines = []
    experiment_dir = output_dir
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True)
    results_filename = experiment_dir / ("%s_%s.txt" % (experiment_name, method.short_filename()))
    if Path(results_filename).is_file():
        print("File %s already exists, skipping" % str(results_filename))
        continue
    try:
        for forest_idx, forest_world in enumerate(forests):
            # Set up the simulation with multi-hypothesis planner parameters
            params = copy.deepcopy(default_params)
            params.n_hypotheses = method.n_hypotheses

            rb_sensor, size_sensor = get_default_sensors()

            # Using multiple hypothesis planner
            if method.planner_type == PlannerSettings.Planner.MULTIPLE_HYPOTHESIS:
                sim = MultipleHypothesisPlannerSimulation(
                    start_pose=start_pose,
                    goal_position=goal_position,
                    params=params,
                    world=forest_world,
                    range_bearing_sensor=rb_sensor,
                    size_sensor=size_sensor,
                )
            # Using baseline
            elif method.planner_type == PlannerSettings.Planner.BASELINE_NO_UNCERTAINTY:
                sim = NoUncertaintyBaselineSimulation(
                    start_pose=start_pose,
                    goal_position=goal_position,
                    params=params,
                    world=forest_world,
                    range_bearing_sensor=rb_sensor,
                    size_sensor=size_sensor,
                )
            elif method.planner_type == PlannerSettings.Planner.BASELINE_SAFETY_AWARE_ASTAR:
                raise NotImplementedError("Baseline safety-aware A* not implemented")
            sim_start_time = time.time()
            while sim.is_running():
                sim.update()
                dist = np.linalg.norm(sim.robot_position - sim.goal_position)
                print(
                    "%s | Simulation time: %.3f  |   Distance from goal: %.3f" % (str(method), sim.t, dist),
                    end="\r",
                    flush=True,
                )
                if debug_mode and sim.t > 1.0:
                    break
            time_traveled = sim.t
            results_lines.append("%d, %d, %s, %f\n" % (forest_idx, params.n_hypotheses, sim.status, time_traveled))
            runtime = time.time() - sim_start_time
            print(
                "[INFO] World %d: Method '%s' finished with status '%s'."
                " t_sim=%.2f, t_real=%.2f" % (forest_idx, str(method), sim.status, time_traveled, runtime)
            )

    finally:
        if len(results_lines) >= 1:
            # Write simulation results to file
            print("Writing results to file: %s" % results_filename)
            with open(results_filename, "w") as results_file:
                results_file.writelines(results_lines)
            print("")
print("Total experiment runtime: %.2f seconds" % (time.time() - start_time))
