"""
Run the planner on randomly generated Poisson forests, and compare results over multiple trials.

See README.md for usage instructions.

"""

import random
import numpy as np
from planner.simulation import MultipleHypothesisPlannerSimulation, BaselineSimulation
from planner.forest_generation import PoissonForestUniformRadius, PoissonTreeCluster
import copy
from datetime import datetime
from pathlib import Path
import time
import argparse
from experiments.simulation_defaults import seed, get_default_params, get_default_sensors


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", "-o", required=False, help="Output directory path")
parser.add_argument("--low_density", "-ld", action="store_true")
parser.add_argument("--high_density", "-hd", action="store_true")
parser.add_argument("--high_safety", "-hs", action="store_true")
parser.add_argument("--clusters", "-c", action="store_true")
parser.add_argument("--speed", "-s", type=float, default=None,
                    help="Maximum robot speed, in m/s.")
parser.add_argument("--n_hyp_list", "-hyp", nargs='+', help="List of number of hypotheses to test",
                    required=False)
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

subfolder = ''
if use_clusters:
    subfolder += 'clusters'
else:
    subfolder += 'uniform'

if use_high_safety:
    subfolder += '-highsafety'
else:
    subfolder += '-lowsafety'

# Set up the output directory
if args.output_dir is None:
    output_dir = Path('results/' + datetime.now().strftime("Experiment_%m-%d-%Y_%H.%M.%S"))
else:
    output_dir = Path(args.output_dir)
output_dir = output_dir / subfolder

if not output_dir.exists():
    output_dir.mkdir(parents=True)


# Set up environment parameters
n_environments = 20
x_min, x_max = (-2, 42)
y_min, y_max = (0, 10)
start_pose = np.array([0., 5., 0.])
goal_position = np.array([40, 5], dtype=float)


default_params = get_default_params()

if use_high_safety:
    print("Using high safety setting")
    default_params.safety_probability = 0.999

# Define spaces near the start and goal where no trees will be sampled
w = default_params.robot_width * 3.  # no trees will generate within this distance of start and goal
invalid_spaces = np.array([[start_pose[0], start_pose[1], w],
                           [goal_position[0], goal_position[1], w]])

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
    print("  Generating forest environment with density %.2f [%d of %d]" % (density, i+1, n_environments),
          end='\r', flush=True)
    forest_seed = seed+i
    bounds = [x_min, x_max, y_min, y_max]
    if use_clusters:
        cluster_density = density*4
        cov = np.diag([1, 1.5])**2

        cluster_means = [np.array([10, 5]),
                         np.array([20, 5]),
                         np.array([30, 5])]

        clusters = [PoissonTreeCluster(cluster_density, mean, cov, seed=forest_seed+j)
                    for (j,mean) in enumerate(cluster_means)]
    else:
        clusters = None
    forest = PoissonForestUniformRadius(bounds=bounds,
                                              density=density,
                                              radius_min=min_tree_radius,
                                              radius_max=max_tree_radius,
                                              seed=forest_seed,
                                              invalid_spaces=invalid_spaces,
                                        clusters=clusters)
    forests.append(forest)
print("")

class PlannerSettings(object):

    def __init__(self, n_hypotheses=1):
        self.n_hypotheses = n_hypotheses

    def __str__(self):
        if self.n_hypotheses == 1:
            return "1Hyp"
        elif self.n_hypotheses == 0:
            return "baseline"
        else:
            return "%dHyp" % self.n_hypotheses


if args.n_hyp_list is not None:
    n_hyp_list = [int(x) for x in args.n_hyp_list]
else:
    n_hyp_list = [0, 2, 10, 20]
print("[INFO] Running comparison with n_hypotheses = " + str(n_hyp_list))

methods = []
for n_hyp in n_hyp_list:
    weight_s = 0.5
    weight_d = 1.0 - weight_s
    methods.append(PlannerSettings(n_hypotheses=n_hyp))

start_time = time.time()

for method in methods:
    results_lines = []
    experiment_dir = output_dir
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True)
    results_filename = experiment_dir / ("%s_%s.txt" % (experiment_name, str(method)))
    if Path(results_filename).is_file():
        print("File %s already exists, skipping" % str(results_filename))
        continue
    try:
        for forest_idx, forest_world in enumerate(forests):
            # Set up the simulation with multi-hypothesis planner parameters
            params = copy.deepcopy(default_params)
            # TODO remove n_shortest_paths input
            params.n_hypotheses = method.n_hypotheses

            rb_sensor, size_sensor = get_default_sensors()

            # Using multiple hypothesis planner
            if method.n_hypotheses > 0:
                sim = MultipleHypothesisPlannerSimulation(start_pose=start_pose,
                                                          goal_position=goal_position,
                                                          params=params, world=forest_world,
                                                          range_bearing_sensor=rb_sensor,
                                                          size_sensor=size_sensor)
            # Using baseline
            else:
                sim = BaselineSimulation(start_pose=start_pose, goal_position=goal_position,
                                         params=params, world=forest_world,
                                         range_bearing_sensor=rb_sensor,
                                         size_sensor=size_sensor)
            sim_start_time = time.time()
            while sim.is_running():
                sim.update()
                dist = np.linalg.norm(sim.robot_position - sim.goal_position)
                if method.n_hypotheses == 0:
                    method_str = "Baseline"
                else:
                    method_str = "%d hyp" % (params.n_hypotheses)
                print("%s | Simulation time: %.3f  |   Distance from goal: %.3f" % (method_str, sim.t, dist),
                      end='\r', flush=True)
            time_traveled = sim.t
            results_lines.append("%d, %d, %s, %f\n" %
                                 (forest_idx,
                                  params.n_hypotheses,
                                  sim.status, time_traveled))
            runtime = time.time() - sim_start_time
            print("[INFO] World %d: Method '%s' finished with status '%s'."
                  " t_sim=%.2f, t_real=%.2f" %
                  (forest_idx, str(method), sim.status, time_traveled, runtime))

    finally:
        if len(results_lines) >= 1:
            # Write simulation results to file
            print("Writing results to file: %s" % results_filename)
            with open(results_filename, 'w') as results_file:
                results_file.writelines(results_lines)
            print("")
print("Total experiment runtime: %.2f seconds" % (time.time() - start_time))
