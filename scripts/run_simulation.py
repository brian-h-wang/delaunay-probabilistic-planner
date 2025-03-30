"""
Simulate and visualize the planner as a robot moves through a field of obstacles.
"""

DEBUG = True

import numpy as np
import matplotlib.pyplot as plt
from planner.simulation import MultipleHypothesisPlannerSimulation, BaselineSimulation
from planner.forest_generation import PoissonForestUniformRadius
from planner.plotting import SimulationPlotter, SimulationAnimator
from experiments.simulation_defaults import get_default_params, get_default_sensors

## Specify whether the planner visualization should be shown as the planner runs,
#   or whether we should save a video to view after.
save_video = False  # If False, view the planner as it's running (note this may be slow due to overhead from plotting)

## Specify whether the baseline should be used instead of the multiple-hypothesis planner
use_baseline = True

# Parameters for planner
params = get_default_params()
rb_sensor, size_sensor = get_default_sensors()
params.safety_probability = 0.95
if use_baseline:
    params.n_hypotheses = 0
else:
    params.n_hypotheses = 5

# Parameters for forest environment
forest_seed = 12345678
forest_density = 0.3
min_tree_radius = 0.2
max_tree_radius = 0.5

# World bounds and start/goal positions
x_min, x_max = (-2, 42)
y_min, y_max = (0, 10)
start_pose = np.array([0.0, 5.0, 0.0])  # x, y, orientation
goal_position = np.array([40, 5], dtype=float)

# Generate forest environment
# Define spaces near the start and goal where no trees will be sampled
w = params.robot_width * 3.0  # no trees will generate within this distance of start and goal
invalid_spaces = np.array([[start_pose[0], start_pose[1], w], [goal_position[0], goal_position[1], w]])
bounds = [x_min, x_max, y_min, y_max]
forest = PoissonForestUniformRadius(
    bounds=bounds,
    density=forest_density,
    radius_min=min_tree_radius,
    radius_max=max_tree_radius,
    seed=forest_seed,
    invalid_spaces=invalid_spaces,
)

# Initialize the simulation
# Using multiple hypothesis planner
if not use_baseline:
    sim = MultipleHypothesisPlannerSimulation(
        start_pose=start_pose,
        goal_position=goal_position,
        params=params,
        world=forest,
        range_bearing_sensor=rb_sensor,
        size_sensor=size_sensor,
    )
# Using baseline
else:
    sim = BaselineSimulation(
        start_pose=start_pose,
        goal_position=goal_position,
        params=params,
        world=forest,
        range_bearing_sensor=rb_sensor,
        size_sensor=size_sensor,
    )
plt.ion()  # Required for matplotlib not to block execution
if not save_video:
    print("Starting planner simulation...")
    plotter = SimulationPlotter(sim=sim, plot_bounds=bounds, interactive=True)
    frame_count = 0
    while sim.is_running():
        sim.update()
        plotter.update()
        frame_count += 1
        # if frame_count > 30 and DEBUG:
        #     break
else:
    plt.ion()  # Required for
    plotter = SimulationPlotter(sim=sim, plot_bounds=bounds, interactive=False)
    animator = SimulationAnimator(plotter)
    method_str = "baseline" if use_baseline else ("%dhyp" % params.n_hypotheses)
    video_filename = "video_%s.mp4" % method_str
    # Run sim and save video
    print("Creating simulation video...")
    animator.save(video_filename)
