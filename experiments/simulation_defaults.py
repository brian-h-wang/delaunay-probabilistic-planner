"""
Default parameters and sensor models for running simulations
"""

import numpy as np
from planner.simulation import SimulationParams
from planner.sensor_model import QuadraticProportionalNoiseRangeBearingSensor, ProportionalNoiseSizeSensor

seed = 20212021
def get_default_params():
    sim_params = SimulationParams()
    sim_params.max_time = 60.
    sim_params.debug = False
    sim_params.timeout_n_replans = 10
    # sim_params.replan_rate = 1
    # sim_params.detection_rate = 2
    sim_params.replan_rate = 1
    sim_params.detection_rate = 2
    sim_params.sensor_range = 20.
    sim_params.range_short = 5.
    sim_params.range_medium = 15.


    sim_params.planner_robot_bloat = 1.0
    sim_params.astar_far_enough = 3.0
    sim_params.local_astar_n_attempts_reduced_resolution = 0

    sim_params.local_astar_max_n_vertices = -1
    sim_params.local_astar_timeout_n_vertices = 5000

    sim_params.baseline_global_planner_timeout_n_vertices = 10000

    sim_params.safety_probability = 0.95
    return sim_params


p = get_default_params()

# Based on ZED error model from:
# https://support.stereolabs.com/hc/en-us/articles/206953039-How-does-the-ZED-work
range_proportion_min = 0.01
range_proportion_max = 0.09

size_proportion = 0.05


def get_default_sensors():
    # range_bearing_sensor = LinearNoiseRangeBearingSensor(range_sigma_min=0.05, range_sigma_max=0.5,
    #                                                     max_range=default_params.sensor_range,
    #                                                     bearing_sigma=bearing_std,
    #                                                     seed=seed)
    range_bearing_sensor = QuadraticProportionalNoiseRangeBearingSensor(range_proportion_min=range_proportion_min,
                                                                        range_proportion_max=range_proportion_max,
                                                            max_range=p.sensor_range,
                                                            bearing_sigma=np.deg2rad(2.5), seed=seed)
    size_sensor = ProportionalNoiseSizeSensor(size_proportion, seed=seed)
    return range_bearing_sensor, size_sensor
