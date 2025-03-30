"""
simulation.py
Brian Wang, bhw45@cornell.edu

Main module for simulating the robot and planner (including the baseline A* planner and multiple-hypothesis planner).

"""

import time
import math
import numpy as np
from typing import Tuple, Optional
from enum import Enum
from planner.motion_planning import feedback_lin
from planner.slam import ObstacleSLAM
from planner.navigation_utils import (
    NavigationPath,
    UncertainObstacleEdgeEvaluator,
    NavigationGraphEdgeEvaluator,
    CircularObstacleEdgeEvaluator,
    BloatedObstacleEdgeEvaluator,
    NavigationBoundary,
)
from planner.hybrid_astar import HybridAStarPlanner
from planner.astar_2d import AStar2DPlanner
from planner.multiple_hypothesis_planning import MultipleHypothesisPlanner
from planner.utils import check_collision, fix_angle, find_unoccluded_obstacles
from planner.sensor_model import SizeSensorModel, RangeBearingSensorModel
from planner.probabilistic_obstacles import SafetyProbability
from planner.path_smoothing import PathSmoother
from planner.time_counter import TimeCounter, StubTimeCounter

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)


class SimulationWorld(object):
    """
    A simulation world containing circular obstacles.
    """

    def __init__(self, xlim, ylim, obstacles=None):
        if obstacles is None:
            obstacles = np.empty((0, 3))
        else:
            obstacles = np.array(obstacles)
        assert len(xlim) == 2 and xlim[0] < xlim[1]
        assert len(ylim) == 2 and ylim[0] < ylim[1]
        assert obstacles.shape[1] == 3, "Obstacles should be an N by 3 array, rows are [x, y, diameter]"
        self.xlim = tuple(xlim)
        self.ylim = tuple(ylim)
        self.obstacles = obstacles

    @property
    def bounds(self):
        # Returns the world boundaries as [xmin, xmax, ymin, ymax]
        return [self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]]


class SimulationStatus(Enum):
    RUNNING = 1
    SUCCESS = 2
    STOPPED = 3
    CRASHED = 4
    TIMEOUT = 5

    def __str__(self):
        return self.name.lower()


class SimulationParams(object):

    def __init__(self):
        self.robot_width: float = 0.5
        self.planner_robot_bloat: float = (
            1.1  # robot width is bloated by this factor in the planner, to add extra buffer to obstacles
        )
        self.sim_rate: float = 100
        self.controller_rate: float = 15
        self.detection_rate: float = 5
        self.replan_rate: float = 1
        self.close_enough: float = 0.2  # Distance from goal that counts as having reached the goal
        self.close_enough_waypoint: float = 0.20  # close-enough for waypoint following
        self.local_goal_close_enough: float = 0.5  # how close A* needs to get to the local goal
        self.min_plan_ahead_distance: float = 3.0  # local goal must be at least this far from robot
        #  prevents planned path being very short if replan time is fast
        self.sensor_range: float = 10.0
        self.sensor_field_of_view: float = math.radians(110)

        self.boundary_obstacle_diameter = 1.0

        # Sensor range during the initial 360 degree sensor measurements
        self.sensor_range_init: float = 4.0

        # Desired probability of safe navigation
        # The higher this is, the more conservatively the robot will behave
        self.safety_probability: float = 0.95

        self.motion_primitive_velocity: float = 2.0  # m/s

        # The low-level waypoint follower controller sets the robot velocity according to the
        #   distance to the closest estimated obstacle.
        # If the closest obstacle is max_distance or further away, the robot travels at max_vel
        # If the closest obstacle is min_distance or nearer, the robot travels at min_vel
        # Otherwise, the velocity is linearly interpolated between max and min vel based on
        #   the distance to the closest obstacle.
        self.controller_max_velocity: float = 5.0  # m/s
        self.controller_max_distance: float = 2.0  # m/s
        self.controller_min_velocity: float = 1.0  # m/s
        self.controller_min_distance: float = 0.5  # m/s

        self.debug = False
        self.max_time: float = -1  # timeout for the simulation. ignored if < 0

        # timeout in terms of number of detections
        # simulation stops early if the robot replans this many times, without finding any path
        self.timeout_n_replans: int = -1

        # Multi-hypothesis parameters
        self.n_hypotheses: int = 1
        # The distance and safety weights should sum to 1
        self.weight_distance: float = 0.5
        self.weight_safety: float = 0.5
        self.safety_normalize_threshold: float = 0.99

        # Hybrid A-star params
        self.local_astar_xy_resolution: float = 0.05
        self.local_astar_angle_resolution: int = 30  # degrees
        self.local_astar_n_motion_primitives: int = 9
        self.local_astar_max_angular_rate: int = 720  # degrees/second

        # delta-time for hybrid A* motion primitives
        # make sure to set dt so that robot traveling at max_robot_velocity changes xy discrete cells
        # dt*velocity >= 1.5 * xy_resolution is a good minimum
        self.local_astar_motion_primitive_dt: float = 0.2
        self.local_astar_n_attempts_reduced_resolution: int = (
            0  # if astar cannot find a path, retry with halved Xy resolution, this many times
        )
        self.local_astar_max_n_vertices: int = (
            -1
        )  # 5000 # if hybrid A* expands this many vertices, will terminate search. default -1 (disabled)
        self.local_astar_timeout_n_vertices: int = -1

        # Params for baseline A* global planner
        self.baseline_global_planner_xy_resolution: float = 0.2
        self.baseline_global_planner_timeout_n_vertices: int = -1

        self.save_obstacles_uncertainty_vs_n_detections: bool = False

        # Range cutoffs for short and medium range obstacles
        # Any obstacles between range_medium and the sensor max range are considered long-range
        self.range_short = 5.0
        self.range_medium = 15.0

        # Minimum safety to consider, when searching for medium/long range paths.
        # Paths under this safety likelihood will be pruned
        self.min_safety_prune = 0.10

        # Whether to simulate occlusions
        self.occlusions = True

        # How many standard deviations should the baseline bloat the obstacle diameters by?
        self.baseline_n_std_devs_bloat = 2  # 0 means no bloat


class Simulation(object):

    def __init__(
        self,
        start_pose,
        goal_position,
        world: SimulationWorld,
        range_bearing_sensor: RangeBearingSensorModel,
        size_sensor: SizeSensorModel,
        params=None,
        time_counter: Optional[TimeCounter] = None,
    ):
        self.world = world
        self.range_bearing_sensor = range_bearing_sensor
        self.size_sensor = size_sensor
        # Use default params if none provided
        if params is None:
            params = SimulationParams()
        self.params: SimulationParams = params

        self._robot_x_history = []
        self._robot_y_history = []
        self._robot_t_history = []
        self._replan_time_stamps = []

        # Use np.array calls to copy the start/goal inputs to new arrays
        self.robot_pose = np.array(start_pose, dtype=float)
        self.goal_position = np.array(goal_position, dtype=float)

        self.cmd = np.array([0.0, 0.0])  # forward velocity and turn rate
        self.t = 0.0

        self.dt = 1.0 / self.params.sim_rate
        self.next_det_time = 0.0
        self.next_replan_time = 0.0
        self.next_controller_time = 0.0

        self.done = False

        # detections_rb gives the detections in terms of range and bearing from the robot
        self._measurements_rb = np.empty((0, 3))
        # detections attribute gives the detections in terms of x-y coordinates, relative to the robot
        self._detections_xy = np.empty((0, 3))
        # detections_global gives detections in global x-y coordinates
        self._detections_xy_global = np.empty((0, 3))
        self._gt_distances_to_measured_landmarks = np.array([])

        # Set up the workspace boundary
        self.boundary = NavigationBoundary(
            bounds=self.world.bounds,
            detection_range=self.params.sensor_range,
            obstacle_diameter=self.params.boundary_obstacle_diameter,
        )

        # Record the number of times detections have been received
        self.n_measurements_received = 0
        self.slam = ObstacleSLAM(
            initial_pose=start_pose,
            range_bearing_sensor=range_bearing_sensor,
            size_sensor=size_sensor,
            save_uncertainty_log=self.params.save_obstacles_uncertainty_vs_n_detections,
        )

        obstacles = self.world.obstacles
        self.n_obstacles = obstacles.shape[0]
        self.obstacle_positions = obstacles[:, 0:2]
        self.obstacle_diameters = obstacles[:, 2]
        self.bloated_obstacles = np.empty((0, 3))

        self.cell_decomposition = None
        self.mhp_result = None
        self.local_planner_result = None
        self.baseline_global_planner_points = None  # 2D A* result, for the baseline planner

        self.waypoints = self.robot_position.reshape((1, 2))
        self.next_waypoint_index = 0

        self.status: SimulationStatus = SimulationStatus.RUNNING

        # Number of replans iterations passed without any path being found.
        self.n_stalled_replans = 0

        if self.params.n_hypotheses > 0:
            print("Planner using safety probability %.4f" % params.safety_probability)
        self.safety_probability = SafetyProbability(params.safety_probability)

        self.n_replans = 0

        self.path_smoother = PathSmoother()

        self._update_trajectory_history()

        # Save the time counter
        if time_counter is None:
            time_counter = StubTimeCounter()
        self.time_counter = time_counter

    def _update_trajectory_history(self):
        self._robot_t_history.append(self.t)
        self._robot_x_history.append(self.robot_position[0])
        self._robot_y_history.append(self.robot_position[1])

    @property
    def robot_position(self):
        return self.robot_pose[0:2]

    @property
    def robot_orientation(self):
        return self.robot_pose[2]

    # measurements_rb, detections_xy, and detections_xy_global are set in the detections update
    # step of the simulation, and should not be set otherwise,
    # or they will be inconsistent with one another

    @property
    def measurements_rb(self):
        """Gives the current range-bearing-size measurements."""
        return self._measurements_rb

    @property
    def gt_distances_to_measured_landmarks(self):
        return self._gt_distances_to_measured_landmarks

    @property
    def detections_xy(self):
        """Gives current measurements, in terms of robot frame x-y coordinates, and diameter."""
        return self._detections_xy

    @property
    def detections_xy_global(self):
        """Gives current measurements, in terms of global frame x-y coordinates, and diameter."""
        return self._detections_xy_global

    def is_running(self):
        # Check if the simulated robot has reached the goal
        return self.status == SimulationStatus.RUNNING

    def success(self):
        # Check if the simulated robot has reached the goal
        return self.status == SimulationStatus.SUCCESS

    def update(self):
        self.t += self.dt

        # Update the ground truth robot pose according to current controls
        cmd_v, cmd_w = self.cmd

        vel_x = cmd_v * math.cos(self.robot_orientation)
        vel_y = cmd_v * math.sin(self.robot_orientation)

        # Update the SLAM estimate using odometry
        # NOTE currently using perfect odometry
        odom_global_frame = np.array([vel_x, vel_y, cmd_w]) * self.dt
        odom_body_frame = np.array([cmd_v, 0, cmd_w]) * self.dt

        self.robot_pose += odom_global_frame

        self._update_trajectory_history()

        # Use new measurements to update estimate of robot and landmark states
        with self.time_counter.count("slam_update"):
            if self.t >= self.next_det_time:
                with self.time_counter.count("update_detections"):
                    # Update SLAM with detections and odometry
                    self.update_detections_range_bearing_noise()
                with self.time_counter.count("update_odometry"):
                    self.slam.update_odometry(odometry=odom_body_frame)
                    # Pass in ground truth distances to landmarks for debugging/plotting later on
                with self.time_counter.count("update_landmarks"):
                    self.slam.update_landmarks(
                        landmark_measurements=self.measurements_rb,
                        gt_distances_to_measured_landmarks=self.gt_distances_to_measured_landmarks,
                    )
                    # Update the boundary, so that boundary obstacles close enough to the robot's current
                    #   location are made visible to the robot
                    self.boundary.update(robot_pose=self.robot_pose)
                self.next_det_time += 1.0 / self.params.detection_rate
            else:
                with self.time_counter.count("update_detections"):
                    pass
                with self.time_counter.count("update_odometry"):
                    # No detections; update with odometry only
                    self.slam.update_odometry(odometry=odom_body_frame)
                with self.time_counter.count("update_landmarks"):
                    pass

        # Plan a new path, at the specified replanning rate
        if self.t >= self.next_replan_time:
            with self.time_counter.count("plan_path"):
                waypoints = self.plan_path(self.robot_pose, self.goal_position)
                self.waypoints = self.smooth_path(waypoints)
            self.next_waypoint_index = 0
            self.next_replan_time += 1.0 / self.params.replan_rate

            # Count how many replans have happened without finding a path
            if self.waypoints.shape[0] == 0:
                self.n_stalled_replans += 1
            else:
                self.n_stalled_replans = 0
            self.n_replans += 1
            self._replan_time_stamps.append(self.t)

        # Check status
        if self.params.max_time > 0 and self.t >= self.params.max_time:
            self.status = SimulationStatus.TIMEOUT
        elif self.params.timeout_n_replans > 0 and self.n_stalled_replans >= self.params.timeout_n_replans:
            self.status = SimulationStatus.STOPPED
        elif check_collision(
            self.robot_position, self.params.robot_width, self.obstacle_positions, self.obstacle_diameters
        ):
            self.status = SimulationStatus.CRASHED
        elif np.linalg.norm(self.robot_position - self.goal_position) <= self.params.close_enough:
            self.status = SimulationStatus.SUCCESS

        # If simulation is still running, compute control command for robot
        if self.status == SimulationStatus.RUNNING:
            self.update_next_waypoint()
            if self.t >= self.next_controller_time:
                self.update_control_command()
                self.next_controller_time += 1.0 / self.params.controller_rate

    def update_next_waypoint(self):
        # This function should be run at sim rate, for fast updating of the next waypoint
        # Determine which waypoint the robot should move to next
        # When robot gets close enough to a waypoint, move on to the next one
        if self.waypoints.shape[0] != 0:
            # Find the next waypoint that is far enough from the robot's position
            while (
                np.linalg.norm(self.waypoints[self.next_waypoint_index, :] - self.robot_position)
                < self.params.close_enough_waypoint
                and self.next_waypoint_index < self.waypoints.shape[0] - 1
            ):  # prevent going past last waypoint
                self.next_waypoint_index += 1

    def update_control_command(self):
        # Determine which waypoint the robot should move to next
        # When robot gets close enough to a waypoint, move on to the next one
        if self.waypoints.shape[0] == 0:
            # No path found
            self.cmd = np.array([0.0, 0.0])
        else:
            # Compute displacement to next waypoint
            waypoint = self.waypoints[self.next_waypoint_index]
            v = waypoint - self.robot_position
            vnorm = np.linalg.norm(v)

            # Calculate the robot velocity, based on the distance to the nearest obstacle
            if self.slam.n_landmarks > 0:
                obstacle_means = self.slam.get_landmarks_position_size_mean()
                robot_position = self.robot_position
                ctr_to_ctr_distances = np.linalg.norm(obstacle_means[:, 0:2] - robot_position, axis=1)
                # account for obstacle and robot radii
                distances = (ctr_to_ctr_distances - obstacle_means[:, 2] / 2.0) - self.params.robot_width / 2.0
                closest_distance = np.min(distances)
            else:
                closest_distance = float("inf")
            if closest_distance < self.params.controller_min_distance:
                v_scale = self.params.controller_min_velocity
            elif closest_distance > self.params.controller_max_distance:
                v_scale = self.params.controller_max_velocity
            else:
                # Linearly interpolate
                distances = [self.params.controller_min_distance, self.params.controller_max_distance]
                vels = [self.params.controller_min_velocity, self.params.controller_max_velocity]
                v_scale = np.interp(closest_distance, distances, vels)

            # Robot slows down as it approaches the final goal
            if np.all(waypoint == self.goal_position):
                if vnorm > v_scale:
                    v = v / vnorm * v_scale
            # For other waypoints, travels at constant velocity
            else:
                # vnorm can be close to zero if robot close to goal;
                #   check to avoid divide by zero
                if vnorm > 1e-8:
                    v = v / vnorm * v_scale
            # Compute forward velocity  and turn rate
            self.cmd = feedback_lin(v[0], v[1], self.robot_orientation)

    def update_detections_range_bearing_noise(self):
        """
        Compute the simulated detections, based on the current robot pose,
        landmark positions, and sensor noise model.

        Simulates measurements according to noise applied to the range and bearing measurements.

        Updates the attributes "measurements_rb" (range-bearing measurements),  (N by 2)
        "gt_distances_to_measured_landmarks" (ground truth distances),          (size N)

        as well as
        "detections" (detections in robot frame),
        and "detections_global" (detections in world frame).
        Both these last two are (N by 3) arrays, each giving detections as measured [x, y, diameter]
        """
        self.n_measurements_received += 1
        robot_position = self.robot_pose[0:2]
        robot_orientation = self.robot_pose[2]
        measurements_list = []
        gt_distances_list = []
        # TODO occlusions checking could be made more efficient by ignoring obstacles outside of sensor range/FoV.
        #      not doing this since for the sims, occlusion checking is fast already (~1ms)
        if self.params.occlusions:
            unoccluded = find_unoccluded_obstacles(robot_position, self.obstacle_positions, self.obstacle_diameters)
        else:
            unoccluded = np.ones(self.n_obstacles, dtype=int)  # all obstacles visible

        for i in range(self.n_obstacles):
            if not unoccluded[i]:
                continue

            # Calculate true range and bearing to this obstacle
            obs_pos = self.obstacle_positions[i, :]
            range_to_obs = np.linalg.norm(robot_position - obs_pos)
            x_rel = obs_pos[0] - robot_position[0]
            y_rel = obs_pos[1] - robot_position[1]
            obs_bearing = math.atan2(y_rel, x_rel) - robot_orientation
            # Convert bearing angle to the range +- pi
            obs_bearing = obs_bearing % (2 * math.pi)
            if obs_bearing > math.pi:
                obs_bearing -= 2 * math.pi

            # Obstacle is detected if within sensor's range and field of view
            if range_to_obs > self.params.sensor_range:
                continue
            # Ignore obstacles outside of sensor FOV, except for first time step
            if abs(obs_bearing) > (self.params.sensor_field_of_view / 2.0) and not (
                self.t < 1 / self.params.detection_rate
            ):
                continue

            # Simulate obstacle range, bearing, and diameter measurements
            r, b = self.range_bearing_sensor.generate_measurement(robot_pose=self.robot_pose, obstacle_position=obs_pos)
            s = self.size_sensor.generate_measurement(
                robot_position=robot_position, obstacle_position=obs_pos, obstacle_diameter=self.obstacle_diameters[i]
            )
            measurement = [r, b, s]
            assert abs(b) < math.pi, "Bearing measurement should be within +- 180 degrees, is %.2f, fixed is %.2f" % (
                math.degrees(b),
                math.degrees(fix_angle(b)),
            )
            measurements_list.append(measurement)

            # Also calculate and save ground truth distances to the measured obstacles for logging
            gt_distances_list.append(range_to_obs)
        n_measurements = len(measurements_list)
        if n_measurements > 0:
            # Save the range, bearing measurements
            measurements = np.array(measurements_list)
            self._measurements_rb = measurements
            self._gt_distances_to_measured_landmarks = np.array(gt_distances_list)

            # Convert the range, bearing measurements to local and global frame XY measurements
            detections_xy = np.empty((n_measurements, 3))
            detections_xy_global = np.empty((n_measurements, 3))
            robot_x = robot_position[0]
            robot_y = robot_position[1]
            for meas_idx in range(n_measurements):
                z_range = measurements[meas_idx, 0]
                z_bearing = measurements[meas_idx, 1]
                local_x = z_range * math.cos(z_bearing)
                local_y = z_range * math.sin(z_bearing)
                detections_xy[meas_idx, 0:2] = local_x, local_y
                angle_global = z_bearing + robot_orientation
                global_x = robot_x + math.cos(angle_global) * z_range
                global_y = robot_y + math.sin(angle_global) * z_range
                detections_xy_global[meas_idx, 0:2] = global_x, global_y

            # Populate size measurements (unaffected by range/bearing vs. XY)
            detections_xy[:, 2] = measurements[:, 2]
            detections_xy_global[:, 2] = measurements[:, 2]
            self._detections_xy = detections_xy
            self._detections_xy_global = detections_xy_global
        else:
            self._measurements_rb = np.empty((0, 3))
            self._gt_distances_to_measured_landmarks = np.empty(0)
            self._detections_xy = np.empty((0, 3))
            self._detections_xy_global = np.empty((0, 3))

    def plan_path(self, start_pose, goal_position):
        """
        Run the local planner to find a list of (x,y) waypoints for the robot to follow.
        In the Simulation.update() method, these waypoints will later be post-processed
        using the path smoother.

        Returns
        -------
        array_like
            N by 2 ndarray.
            Each row is [x, y], a robot position.

        """
        raise NotImplementedError()

    def get_plan_ahead_distance(self):
        """Compute how far ahead the local planner should generate a path."""
        p = self.params
        plan_ahead_distance = p.controller_max_velocity * (1.0 / p.replan_rate) + p.local_goal_close_enough
        plan_ahead_distance = max(plan_ahead_distance, p.min_plan_ahead_distance)
        return plan_ahead_distance

    def smooth_path(self, waypoints):
        """
        Post-process the path using gradient descent, to move the waypoints further away
        from nearby obstacles while also optimizing for smoothness of the path.

        Parameters
        ----------
        waypoints

        Returns
        -------

        """
        # no waypoints provided
        if waypoints.shape[0] == 0:
            return waypoints
        # Get obstacles for smoother
        obstacle_means = self.slam.get_landmarks_position_size_mean()
        boundary_obstacles = self.boundary.get_obstacles_array()
        all_obstacles = np.concatenate([obstacle_means, boundary_obstacles], axis=0)
        # Run the gradient descent smoother
        waypoints = self.path_smoother.smooth(waypoints, all_obstacles)  # obstacles + boundary
        return waypoints

    def get_robot_position_history(self):
        assert (
            len(self._robot_x_history) == len(self._robot_t_history) == len(self._robot_y_history)
        ), "Error! t, x, y histories do not match."
        trajectory_txy = np.empty((len(self._robot_x_history), 3))
        trajectory_txy[:, 0] = self._robot_t_history
        trajectory_txy[:, 1] = self._robot_x_history
        trajectory_txy[:, 2] = self._robot_y_history
        return trajectory_txy

    def get_replan_time_stamps(self):
        return self._replan_time_stamps


class SetCommandSimulation(Simulation):
    """
    Simulation where the robot follows a fixed, predefined velocity/turn rate command
    Currently used for debugging and creating plots only.
    """

    def __init__(
        self,
        start_pose,
        goal_position,
        world: SimulationWorld,
        range_bearing_sensor: RangeBearingSensorModel,
        size_sensor: SizeSensorModel,
        params=None,
    ):
        super().__init__(start_pose, goal_position, world, range_bearing_sensor, size_sensor, params=params)
        self.cmd_v = 0.0
        self.cmd_w = 0.0

    def set_command(self, cmd_v, cmd_w):
        self.cmd_v = cmd_v
        self.cmd_w = cmd_w

    def update_control_command(self):
        self.cmd = (self.cmd_v, self.cmd_w)

    def plan_path(self, start_pose, goal_position):
        return np.empty((0, 2))


class BaselineSimulation(Simulation):
    """
    Baseline: Runs hybrid A* using the SLAM obstacle estimate means.
    The baseline uses 2D A* to generate a path to the global goal, using the obstacle means,
    then uses hybrid A* to plan a path to the point at plan_ahead_distance along this path.
    """

    def plan_path(self, start_pose, goal_position):
        p = self.params
        robot_width_bloated = p.robot_width * p.planner_robot_bloat
        # obstacle_means = self.slam.get_landmarks_position_size_mean()

        # Get mean obstacles and their position and size uncertainties
        obstacle_means, obstacle_variances = self.slam.get_landmarks_position_size_mean_and_variance()

        # Ignore any obstacles beyond the medium-range zone
        for obstacle_idx in range(obstacle_means.shape[0]):
            if np.linalg.norm(obstacle_means[obstacle_idx, 0:2] - self.robot_position) > p.range_medium:
                obstacle_means[obstacle_idx, :] = 0.0
                obstacle_variances[obstacle_idx, :] = 0.0

        # Get boundary obstacles, which have near-zero uncertainty
        boundary_obstacles = self.boundary.get_obstacles_array()
        boundary_variances = np.ones((boundary_obstacles.shape[0], 3)) * 1e-12

        obstacles_with_boundary = np.concatenate([obstacle_means, boundary_obstacles], axis=0)
        variances_with_boundary = np.concatenate([obstacle_variances, boundary_variances], axis=0)

        # global_edge_evaluator = CircularObstacleEdgeEvaluator(obstacles_with_boundary, robot_width=robot_width_bloated)
        global_edge_evaluator = BloatedObstacleEdgeEvaluator(
            obstacle_means=obstacles_with_boundary,
            obstacle_variances=variances_with_boundary,
            n_std_devs=self.params.baseline_n_std_devs_bloat,
            robot_width=robot_width_bloated,
        )

        global_planner = AStar2DPlanner(
            edge_evaluator=global_edge_evaluator,
            xy_resolution=p.baseline_global_planner_xy_resolution,
            timeout_n_vertices=p.baseline_global_planner_timeout_n_vertices,
            close_enough=self.params.close_enough,
            boundary=self.boundary,
        )

        with self.time_counter.count("baseline_global_planner"):
            global_path = global_planner.find_path(start_position=self.robot_position, goal_position=self.goal_position)

        if global_path is None:
            self.baseline_global_planner_points = None
            print("2D A* could not find a path.")
            return np.empty((0, 2))

        plan_ahead_distance = self.get_plan_ahead_distance()
        local_goal, is_global_goal, goal_point_index = global_path.get_local_goal(
            distance=plan_ahead_distance, return_index=True
        )
        if is_global_goal:
            close_enough = p.close_enough
        else:
            close_enough = p.local_goal_close_enough
        self.baseline_global_planner_points = global_path.points[goal_point_index:, :]

        edge_evaluator = CircularObstacleEdgeEvaluator(obstacles_with_boundary, robot_width=robot_width_bloated)

        planner = HybridAStarPlanner(
            edge_evaluator=edge_evaluator,
            xy_resolution=p.local_astar_xy_resolution,
            angle_resolution=p.local_astar_angle_resolution,
            n_motion_primitives=p.local_astar_n_motion_primitives,
            motion_primitive_dt=p.local_astar_motion_primitive_dt,
            motion_primitive_velocity=p.motion_primitive_velocity,
            max_angular_rate=p.local_astar_max_angular_rate,
            far_enough=None,
            n_attempts_reduced_resolution=p.local_astar_n_attempts_reduced_resolution,
            max_n_vertices=p.local_astar_max_n_vertices,
            timeout_n_vertices=p.local_astar_timeout_n_vertices,
            close_enough=close_enough,
        )
        with self.time_counter.count("baseline_local_planner"):
            path = planner.find_path(start_pose, local_goal)
        self.local_planner_result = path
        if path is None:
            # No Hybrid A* result found
            print("2D A* search succeeded but hybrid A* local planner could not find a path.")
            return np.empty((0, 2))

        # Get robot positions from hybrid A* result
        waypoints = path.poses(points_per_motion_primitive=5)[:, 0:2]

        return waypoints


class MultipleHypothesisPlannerSimulation(Simulation):

    def plan_path(self, start_pose, goal_position):
        assert self.params.n_hypotheses >= 1, "MHP simulation is using 0 hypotheses. Use BaselineSimulation instead."
        p = self.params
        robot_width_bloated = p.robot_width * p.planner_robot_bloat
        # If at least 3 landmarks, build a navigation graph.
        if self.slam.n_landmarks >= 3 and self.params.n_hypotheses > 0:
            # Get the current estimates of landmark positions and sizes

            # Build a navigation graph based on current obstacles estimate
            # Increase robot width to account for occ grid bloat
            mhp_planner = MultipleHypothesisPlanner(
                slam=self.slam,
                robot_width=robot_width_bloated,
                safety_probability=self.safety_probability,
                range_short=p.range_short,
                range_medium=p.range_medium,
                boundary=self.boundary,
                time_counter=self.time_counter,
            )

            # Plan high-level path in the graph
            with self.time_counter.count("global_planner"):
                mhp_result = mhp_planner.find_paths(
                    start_position=self.robot_position,
                    goal_position=self.goal_position,
                    n_hypotheses=self.params.n_hypotheses,
                    min_safety_prune=self.params.min_safety_prune,
                )
            self.cell_decomposition = mhp_result.cell_decomposition
            self.mhp_result = mhp_result

            # Take the best path and use it as high-level guidance for Hybrid A*
            high_level_path = mhp_result.best_path

            # No paths found
            if high_level_path is None:
                return np.empty((0, 2))

            # Compute local goal from the high-level path and give as input to the local planner
            plan_ahead_distance = self.get_plan_ahead_distance()
            local_goal, is_global_goal = mhp_result.get_local_goal(distance=plan_ahead_distance)

            # If planning to an intermediate local goal, not the global goal,
            # the close-enough threshold on hybrid A* is more generous
            if is_global_goal:
                close_enough = p.close_enough
            else:
                close_enough = p.local_goal_close_enough
            obstacle_means = self.slam.get_landmarks_position_size_mean()
            boundary_obstacles = self.boundary.get_obstacles_array()
            all_obstacles = np.concatenate([obstacle_means, boundary_obstacles], axis=0)
            edge_evaluator = NavigationGraphEdgeEvaluator(
                path=high_level_path, obstacles=all_obstacles, robot_width=robot_width_bloated
            )
            # For baseline, set to half of the waypoint following "close enough to goal",
            #   to ensure that hybrid A* takes the robot precisely to the goal position
            # For local goal in MHP, set a more generous threshold
            planner = HybridAStarPlanner(
                edge_evaluator=edge_evaluator,
                xy_resolution=p.local_astar_xy_resolution,
                angle_resolution=p.local_astar_angle_resolution,
                n_motion_primitives=p.local_astar_n_motion_primitives,
                motion_primitive_dt=p.local_astar_motion_primitive_dt,
                motion_primitive_velocity=p.motion_primitive_velocity,
                max_angular_rate=p.local_astar_max_angular_rate,
                far_enough=None,
                n_attempts_reduced_resolution=p.local_astar_n_attempts_reduced_resolution,
                max_n_vertices=p.local_astar_max_n_vertices,
                timeout_n_vertices=p.local_astar_timeout_n_vertices,
                close_enough=close_enough,
            )
            with self.time_counter.count("local_planner"):
                path = planner.find_path(start_pose, local_goal)
            self.local_planner_result = path
            if path is None:
                # No Hybrid A* result found
                return np.empty((0, 2))

            # Get robot positions from hybrid A* result
            waypoints = path.poses(points_per_motion_primitive=5)[:, 0:2]
            return waypoints

        else:
            # Not enough obstacles detected to build a navigation graph;
            #   just use the baseline hybrid A* planner
            return super().plan_path(start_pose, self.goal_position)
