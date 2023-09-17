"""
Module for path smoothing.

Based on the gradient descent method from:
Dolgov et al., "Practical Search Techniques in Path Planning for Autonomous Driving"
Dolgov et al., "Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments"

Reference on path smoothness term:
https://medium.com/@jaems33/understanding-robot-motion-path-smoothing-5970c8363bc4
"""

from numpy.typing import ArrayLike
import numpy as np
import numba


@numba.njit
def waypoints_to_closest_obstacles(waypoints, obstacle_xy, obstacle_diameter):
    """
    For each waypoint, compute the vector to the edge of the nearest obstacle.

    Parameters
    ----------
    waypoints
        Shape (N, 2)
    obstacle_xy
        Shape (M, 2)
    obstacle_diameter
        Shape (M,)

    Returns
    -------
    to_closest_obstacle
        Shape (N, 2)

    """
    n_waypoints = waypoints.shape[0]
    # Array of vectors from each waypoint to its nearest obstacle
    waypoint_to_obstacle = np.empty_like(waypoints)
    obstacle_radii = obstacle_diameter / 2.
    for wp_idx in range(n_waypoints):
        # Compute distance from this waypoint to all obstacles
        wp_to_obstacles = obstacle_xy - waypoints[wp_idx, :]
        # Find the closest obstacle
        distances_to_obstacles = np.sqrt(wp_to_obstacles[:,0]**2 + wp_to_obstacles[:,1]**2) - obstacle_radii
        # Flip the vectors for any waypoints that are in an obstacle
        #   (otherwise waypoints in an obstacle will be pushed away from the obstacle edge;
        #    i.e. towards the obstacle center)
        in_obstacle = distances_to_obstacles < 0
        wp_to_obstacles[in_obstacle] = -wp_to_obstacles[in_obstacle]
        closest_idx = np.argmin(distances_to_obstacles)

        # Calculate the vector to the closest obstacle
        waypoint_to_obstacle[wp_idx, :] = (wp_to_obstacles[closest_idx] / np.linalg.norm(wp_to_obstacles[closest_idx])) * distances_to_obstacles[closest_idx]

    return waypoint_to_obstacle


@numba.njit
def compute_gradient_obstacles(waypoints, obstacle_xy, obstacle_diameter, distance_max):
    # Compute the vector from each waypoint to its nearest obstacle
    x_to_o = waypoints_to_closest_obstacles(waypoints, obstacle_xy, obstacle_diameter)

    # Compute the gradient
    distances = np.sqrt(x_to_o[:,0]**2 + x_to_o[:,1] ** 2)
    scale = 2 * (distances - distance_max) / distances

    gradient = np.empty_like(x_to_o)
    gradient[:,0] = scale * x_to_o[:,0]
    gradient[:,1] = scale * x_to_o[:,1]

    # Waypoints further than distance_max away from any obstacles are unchanged
    gradient[distances >= distance_max] = 0.

    gradient[0,:] = 0.
    gradient[-1,:] = 0.
    return -gradient


@numba.njit
def compute_gradient_smoothness(waypoints, weight_smoothness=2.0):
    """
    Compute the gradient of the path smoothness cost wrt the waypoint locations.

    Parameters
    ----------
    waypoints
    weight_smoothness

    Returns
    -------

    """
    gradient = np.zeros_like(waypoints)
    n_waypoints = waypoints.shape[0]
    for i in range(1, n_waypoints-1):
        x_prev = waypoints[i-1, :]
        x = waypoints[i, :]
        x_next = waypoints[i+1, :]
        gradient[i,:] = weight_smoothness * (x_prev - 2*x + x_next)
    gradient[0,:] = 0
    gradient[-1,:] = 0

    return -gradient

@numba.njit
def smooth_path_gradient_descent(xy: ArrayLike, obs_xy, obs_diameter,
                                 weight_obstacles: float, weight_smoothness: float,
                                  n_iters: int, distance_max: float, lr: float):
    for _ in range(n_iters):
        gradient_obs = compute_gradient_obstacles(xy, obs_xy, obs_diameter,
                                                  distance_max=distance_max)
        gradient_smoothness = compute_gradient_smoothness(xy)
        xy -= lr * (weight_obstacles * gradient_obs + weight_smoothness * gradient_smoothness )
    return xy


class PathSmoother(object):

    def __init__(self, weight_obstacles: float = 0.4, weight_smoothness: float = 6,
                  n_iters: int = 500, distance_max = 2.0, learning_rate: float = 0.001):
        self.weight_obstacles = weight_obstacles
        # smoothness weight is normalized by dividing by average distnace between waypoints
        self.weight_smoothness = weight_smoothness
        self.n_iters = n_iters
        self.distance_max = distance_max
        self.learning_rate = learning_rate

    def smooth(self, xy: ArrayLike, obstacles: ArrayLike, obstacle_filter_distance=20.):
        obs_xy = obstacles[:,0:2]
        obs_diameter = obstacles[:,2]

        # Compute the average distance between waypoints,
        # use this as a normalization factor for the smoothness weight
        avg_between_wp = np.mean(np.linalg.norm(np.diff(xy, axis=0), axis=1), axis=0)
        w_sm_normalized = self.weight_smoothness / avg_between_wp

        # Filter out any obstacles further than filter_distance away from the first waypoint,
        #   these are unlikely to affect the path smoothing
        obs_distances = np.linalg.norm(obs_xy - xy[0,:], axis=1) - obs_diameter/2.
        in_range = obs_distances < obstacle_filter_distance
        if np.sum(in_range) == 0:
            # no obstacles within the maximum range, so return original path
            return xy
        obs_xy = obs_xy[in_range, :]
        obs_diameter = obs_diameter[in_range]
        return smooth_path_gradient_descent(xy=xy, obs_xy=obs_xy, obs_diameter=obs_diameter,
                                            weight_obstacles=self.weight_obstacles,
                                            weight_smoothness=w_sm_normalized,
                                            n_iters=self.n_iters, distance_max=self.distance_max,
                                            lr=self.learning_rate)