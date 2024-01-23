"""
Module for SLAM using obstacle detections.

Acknowledgment:
Original treeSLAM code written by Rachel Zheng, Cornell MAE '21

References:
iSAM example from gtsam Github repo:
- https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/VisualISAM2Example.py
Kaess, Ranganathan, & Dellaert, "iSAM: Incremental Smoothing and Mapping", T-RO 2008
    For reference on Mahalanabois distance data association, and general reference on iSAM
"""

from typing import List, Optional, Tuple
from numpy.typing import ArrayLike
import gtsam
import math
import numpy as np
from numpy import ndarray
import numba
from gtsam.symbol_shorthand import L, X
from scipy.optimize import linear_sum_assignment
from planner.sensor_model import SizeSensorModel, RangeBearingSensorModel
from planner.probabilistic_obstacles import Obstacle2D
from planner.utils import fix_angle

# Noise params
# Note these values are standard deviations (not variance)

# Prior noise on the robot pose. Very small; initial pose can be treated as origin.
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001, 0.00001, 0.00001]))
# Prior noise on newly initialized landmarks
PRIOR_LANDMARK_POSITION_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0]))
# Noise on odometry measurements
# ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.01]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.00001]))
GPS_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.00001]))
# Noise on landmark measurements
# MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 1.0]))

# Data association params

# Minimum number of detections to initialize a new landmarks
MIN_DETECTIONS_THRESHOLD = 4
# Cost threshold; if a detection has cost above this threshold for all previously tracked landmarks,
# it will be treated as a new landmark
# Threshold for Euclidean distance
NEW_LM_COST_THRESHOLD_EUCLIDEAN = 0.5

# Data association matches above this many std devs will not be considered (set to ~infinite cost)
MAX_MAHALANOBIS_DISTANCE = 6.0

# Initial landmark size mean and variance for newly added landmarks
LM_INIT_SIZE_MEAN = 1.0
LM_INIT_SIZE_VAR = 100.0

# Landmarks which are not observed for this many timestamps become inactive
TIME_FOR_INACTIVE_LANDMARK = 10

class LandmarkSizeEstimator(object):
    MIN_SIZE = 0.001

    def __init__(self, init_mean: float, init_var: float):
        self.mean = init_mean
        self.var = init_var

    def update(self, measurement: float, measurement_std: float):
        measurement_noise_var = measurement_std**2
        kalman_gain = self.var / (self.var + measurement_noise_var)
        exp_measurement = self.mean
        self.mean = self.mean + kalman_gain*(measurement - exp_measurement)
        self.var = (1. - kalman_gain) * self.var

        if self.mean < self.MIN_SIZE:
            self.mean = self.MIN_SIZE

    def __str__(self):
        return "Size estimate: Mean %.3f with std dev %.3f" % (self.mean, math.sqrt(self.var))


class UncertaintyLog(object):
    """
    Class for recording how the position and size uncertainty of SLAM landmarks changes with
    respect to the numbers of detections received.
    """

    def __init__(self):
        self.n_detections_dict = {}
        self.pos_uncertainty_history_dict = {}
        self.size_uncertainty_history_dict = {}
        self.range_history_dict = {}

    def add_landmark(self, landmark_number: int):
        self.n_detections_dict[landmark_number] = 0
        self.pos_uncertainty_history_dict[landmark_number] = []
        self.size_uncertainty_history_dict[landmark_number] = []
        self.range_history_dict[landmark_number] = []

    def add_detection(self, landmark_number: int, position_cov: ArrayLike, size_var: float,
                      range_to_landmark: float):
        self.n_detections_dict[landmark_number] += 1
        self.pos_uncertainty_history_dict[landmark_number].append(np.trace(position_cov))
        self.size_uncertainty_history_dict[landmark_number].append(size_var)
        self.range_history_dict[landmark_number].append(range_to_landmark)

    def get_uncertainty_history(self, landmark_number: int):
        return self.pos_uncertainty_history_dict[landmark_number], self.size_uncertainty_history_dict[landmark_number]

    def get_range_history(self, landmark_number: int):
        return self.range_history_dict[landmark_number]

    def landmark_numbers(self):
        return self.n_detections_dict.keys()


class ObstacleSLAM(object):

    def __init__(self, initial_pose, range_bearing_sensor=RangeBearingSensorModel,
                 size_sensor=SizeSensorModel,
                 relinearize_threshold=0.01, relinearize_skip=1,
                 save_uncertainty_log=False,
                 bounds=None,
                 boundary_obstacle_diameter=0.25):
        # self.position_sensor = position_sensor
        self.range_bearing_sensor = range_bearing_sensor
        self.size_sensor = size_sensor
        # Initialize iSAM
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(relinearize_threshold)
        parameters.relinearizeSkip = relinearize_skip
        isam = gtsam.ISAM2(parameters)

        # Initialize the graph
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # Add a prior on the initial pose
        self.current_pose = np.array(initial_pose)
        initial_pose = gtsam.Pose2(initial_pose[0], initial_pose[1], initial_pose[2])
        prior_factor = gtsam.PriorFactorPose2(X(0), initial_pose, PRIOR_NOISE)
        graph.add(prior_factor)
        initial_estimate.insert(X(0), initial_pose)

        isam.update(graph, initial_estimate)

        # Clear the factor graph and initial estimate for the next iteration
        # graph.resize(0)
        # initial_estimate.clear()

        self.graph = graph
        self.isam = isam
        self.prev_pose: gtsam.Pose2 = initial_pose
        self.initial_estimate = initial_estimate
        self.time_step = 0
        self.x_num = {}  # number of detections observed for each tracked landmark
        self.current_estimate: gtsam.Values = self.isam.calculateEstimate()
        self.n_landmarks = 0
        self.landmarks_n_measurements = {}

        # Dict that counts how many times each landmark has been detected
        #   Keys are landmark indices, values are counts for number of detections
        self.landmark_detection_counts = {}

        # Dict for storing factors created for newly observed landmarks
        # Landmarks are only added to the factor graph after they have been detected
        # more than MIN_DETECTIONS_THRESHOLD times.
        # This dict stores the previous detection factors for landmarks, until they are
        # detected this many times and actually added to the factor graph.
        self.new_landmark_factors = {}

        # Dict for storing estimators for landmark size
        self.landmark_size_estimators = {}

        # Record which landmarks are boundary obstacles
        self._is_boundary = {}

        # If optional save_uncertainty_vs_n_detections arg is set,
        # save change in position and size uncertainties over time.
        # Save position uncertainty as the trace of the position uncertainty matrix,
        # and save size uncertainty as the size variance value.
        # Records uncertainty vs. number of detections received, for each landmark.
        if save_uncertainty_log:
            self.uncertainty_log = UncertaintyLog()
        else:
            self.uncertainty_log = None

        # Save the closest distance the robot has been to each landmark
        self.closest_robot_to_landmark_distance = {}

        # Add extra obstacles as workspace boundaries
        if bounds is not None:
            seed = 20212022
            rng = np.random.default_rng(seed)

            x_min, x_max, y_min, y_max = bounds
            assert (x_min < x_max) and (y_min < y_max), \
                "'bounds' must be specified as [xmin xmax ymin ymax], with xmin<xmax and ymin<ymax"

            tree_radius = boundary_obstacle_diameter/2.
            tree_spacing = boundary_obstacle_diameter

            # Amount of random variation to apply in the x-direction (for the left and right walls)
            #   or the y-direction (for the top and bottom walls)
            variation = 0.05

            def make_obstacle(x, y):
                pos_cov = np.eye(2) * 1e-8
                size_var = 1e-8
                return Obstacle2D(pos_mean=[x,y], pos_cov=pos_cov,
                                  size_mean=boundary_obstacle_diameter, size_var=size_var)

            # obstacles = np.empty((0,3))
            # Generate the top and bottom barriers
            for y_coord in [y_min-tree_radius, y_max+tree_radius]:
                # Calculate how many trees should be in this barrier
                n_trees_in_barrier = 1 + math.ceil((x_max - x_min) / tree_spacing)
                # Generate tree center positions along this line
                tree_x = np.linspace(x_min-tree_radius, x_max+tree_radius, n_trees_in_barrier)
                tree_y = rng.uniform(-variation, variation, n_trees_in_barrier) + y_coord

                barrier = np.empty((n_trees_in_barrier, 3))
                # barrier[:,0] = tree_x
                # barrier[:,1] = tree_y
                # barrier[:,2] = tree_radius*2
                # obstacles = np.concatenate([obstacles, barrier], axis=0)
                # self.boundary_obstacles.append(make_obstacle(tree_x, tree_y))
            # Generate the left and right barriers
            for x_coord in [x_min-tree_radius, x_max+tree_radius]:
                n_trees_in_barrier = (1 + math.ceil((y_max - y_min) / tree_spacing))
                tree_x = rng.uniform(-variation, variation, n_trees_in_barrier) + x_coord
                tree_y = np.linspace(y_min - tree_radius, y_max + tree_radius, n_trees_in_barrier)
                # Subtract two because the first and last trees can be removed
                #   (these will overlap with the top/bottom barriers)
                barrier = np.empty((n_trees_in_barrier-2, 3))
                # barrier[:,0] = tree_x[1:-1]
                # barrier[:,1] = tree_y[1:-1]
                # barrier[:,2] = tree_radius*2
                # self.boundary_obstacles.append(make_obstacle(tree_x, tree_y))
                # obstacles = np.concatenate([obstacles, barrier], axis=0)



    def update_odometry(self, odometry):
        """
        Perform an iSAM update step using a new odometry measurement.
        This method updates the current time step of the estimator.

        Parameters
        ----------
        odometry: array_like
            3-element array, containing odometry measurements in the body frame.
            For differential drive robot, should be [forward_vel, 0., angular_vel]
        """
        # Clear the factor graph and initial estimate before this iteration
        self.graph.resize(0)
        self.initial_estimate.clear()

        self.time_step += 1
        k = self.time_step

        prev_estimate: gtsam.Values = self.current_estimate
        prev_pose = prev_estimate.atPose2(X(k-1))

        ### ODOMETRY FACTOR

        # get relative yaw angle, and x/y coordinates
        d_x = odometry[0]
        d_y = odometry[1]  # always 0
        d_yaw = odometry[2]

        d_pose = gtsam.Pose2(d_x, d_y, d_yaw)

        # Add pose factor for this odometry measurement
        odom_factor = gtsam.BetweenFactorPose2(X(k-1), X(k), d_pose, ODOMETRY_NOISE)
        self.graph.add(odom_factor)

        # # Calculate an initial estimate for the updated pose,
        # # by adding the odometry measurement to the previous pose
        # odom_pose = gtsam.Pose2(prev_pose.x() + d_x,
        #                         prev_pose.y() + d_y,
        #                         prev_pose.theta() + d_yaw)

        # Initialize the pose estimate (incorrectly) as the previous pose of the robot
        self.initial_estimate.insert(X(k), prev_pose)

        # Update iSAM estimate
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()

        pose: gtsam.Pose2 = self.current_estimate.atPose2(X(k))
        mean = np.array([pose.x(), pose.y(), pose.theta()])
        self.current_pose = mean


    def update_pose(self, pose):
        """
        Perform an iSAM update step using a new pose measurement.
        This method updates the current time step of the estimator.

        Parameters
        ----------
        pose: array_like
            3-element array, containing a global-frame pose measurement, as [x, y, yaw]

        """
        # Clear the factor graph and initial estimate before this iteration
        self.graph.resize(0)
        self.initial_estimate.clear()

        self.time_step += 1
        k = self.time_step

        prev_estimate: gtsam.Values = self.current_estimate
        prev_pose = prev_estimate.atPose2(X(k-1))

        ### ODOMETRY FACTOR

        # get relative yaw angle, and x/y coordinates

        pose_factor = gtsam.Pose2(pose[0], pose[1], pose[2])

        # Add pose factor for this odometry measurement
        # odom_factor = gtsam.BetweenFactorPose2(X(k-1), X(k), d_pose, ODOMETRY_NOISE)
        prior_factor = gtsam.PriorFactorPose2(X(k), pose_factor, GPS_NOISE)
        self.graph.add(prior_factor)

        # # Calculate an initial estimate for the updated pose,
        # # by adding the odometry measurement to the previous pose
        # odom_pose = gtsam.Pose2(prev_pose.x() + d_x,
        #                         prev_pose.y() + d_y,
        #                         prev_pose.theta() + d_yaw)

        # Initialize the pose estimate (incorrectly) as the previous pose of the robot
        self.initial_estimate.insert(X(k), prev_pose)

        # Update iSAM estimate
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()

        pose: gtsam.Pose2 = self.current_estimate.atPose2(X(k))
        mean = np.array([pose.x(), pose.y(), pose.theta()])
        self.current_pose = mean

    def update_landmarks(self, landmark_measurements=[], gt_distances_to_measured_landmarks=None):
        """
        Perform an incremental iSAM update step, using new landmark measurements.

        Parameters
        ----------
        landmark_measurements: ndarray
            N_det by 3 ndarray, containing range, bearing, and diameter measurements for landmarks.
            May be an empty array, indicating no new detections at this time step.
            In this case iSAM will update based on odometry only.
        gt_distances_to_measured_landmarks: ndarray
            N_det-element array containing the ground truth distance from the robot to each of the
            measured landmarks.
            Used only for logging purposes for analyzing estimators in simulation and making plots;
            not used in actual SLAM calculations.

        Returns
        -------
        None

        """
        landmark_measurements = np.array(landmark_measurements)
        n_measurements = landmark_measurements.shape[0]
        # Clear the factor graph and initial estimate before this iteration
        if n_measurements == 0:
            # No measurements - don't perform update
            return
        self.graph.resize(0)
        self.initial_estimate.clear()

        k = self.time_step
        robot_pose = self.current_estimate.atPose2(X(k))
        robot_pose_np = np.array([robot_pose.x(), robot_pose.y(), robot_pose.theta()])

        ### Associate new detections with estimated landmarks and initialize any new landmarks
        # Perform data association using range and bearing Mahalanobis distance
        assignments, assignment_costs = self.data_association(landmark_measurements,
                                                              robot_pose=robot_pose_np)

        ### Add landmark factors to the factor graph
        # For each detection:
        for meas_idx in range(n_measurements):
            # Get the landmark this detection was assigned to
            assigned_landmark = assignments[meas_idx]

            # Detection not assigned to existing landmark; initialize new landmark tracker
            if assigned_landmark == -1:
                new_lm = self.add_landmark()
                assigned_landmark = new_lm
                assignments[meas_idx] = new_lm
            # Increment n measurements count for this landmark
            self.landmarks_n_measurements[assigned_landmark] += 1

            # Get the estimator for the landmark's size
            size_estimator = self.landmark_size_estimators[assigned_landmark]

            # Increment the detection counts for this landmark
            self.landmark_detection_counts.setdefault(assigned_landmark, 0) # add new landmark to dict
            self.landmark_detection_counts[assigned_landmark] += 1

            # Get range and bearing measurements
            z_range = landmark_measurements[meas_idx, 0]
            z_bearing = gtsam.Rot2(landmark_measurements[meas_idx, 1])
            # Get range and bearing std devs from the sensor model
            # Range std dev increases with distance from the robot (simulating stereo noise),
            #   so these sigmas are not constant.
            range_sigma, bearing_sigma = self.range_bearing_sensor.get_sigmas(z_range)

            # NOTE: gtsam BearingRange factor puts bearing first, so sigmas are bearing-first
            measurement_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([bearing_sigma, range_sigma]))

            measurement_factor = gtsam.BearingRangeFactor2D(X(k), L(assigned_landmark),
                                                            z_bearing, z_range,
                                                            measurement_noise_model)

            # Initialize new landmark
            if not self.current_estimate.exists(L(assigned_landmark)):
                # Convert the range-bearing measurement for this new landmark,
                # into global-frame x- and y- coordinates, to initialize the landmark position
                r = landmark_measurements[meas_idx, 0]  # range
                b = landmark_measurements[meas_idx, 1]  # bearing

                # Using range, bearing, and robot pose, get global frame X-Y
                angle = b + robot_pose.rotation().theta()
                global_x = robot_pose.x() + math.cos(angle) * r
                global_y = robot_pose.y() + math.sin(angle) * r

                # Initialize the landmark at these X- Y coordinates
                initial_landmark = gtsam.Point2(global_x, global_y)
                self.initial_estimate.insert(L(assigned_landmark), initial_landmark)
                # DEPRECATED: This prior seems to have very little effect on the SLAM estimate,
                #             if the prior noise is larger than the measurement noise.
                # Add prior noise on landmark
                # landmark_prior = gtsam.PriorFactorPoint2(L(assigned_landmark), initial_landmark,
                #                                          PRIOR_LANDMARK_POSITION_NOISE)
                # self.graph.add(landmark_prior)

            self.graph.add(measurement_factor)

            # Update size estimator
            # Size measurement noise std dev is a proportion of the landmark's true diameter,
            #   get the sigma from the sensor based on the size measurement
            size_sigma = self.size_sensor.get_sigma(landmark_measurements[meas_idx, 2])
            size_estimator.update(landmark_measurements[meas_idx, 2], measurement_std=size_sigma)

        # Update iSAM estimate
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()

        # Update closest recorded distance from robot to each landmark
        lm_position_means, lm_position_covs = self.get_landmarks_position_estimate()
        robot_position = robot_pose_np[0:2]
        for meas_idx in range(n_measurements):
            assigned_landmark = assignments[meas_idx]
            distance = np.linalg.norm(lm_position_means[assigned_landmark, :] - robot_position)
            if distance < self.closest_robot_to_landmark_distance[assigned_landmark]:
                self.closest_robot_to_landmark_distance[assigned_landmark] = distance

        # Update record of landmark uncertainty vs. detection number/range
        if self.uncertainty_log is not None:
            assert gt_distances_to_measured_landmarks is not None, \
                "If logging uncertainty in simulation, must pass in ground truth landmark distances as argument to slam update."
            n_measurements = landmark_measurements.shape[0]
            for meas_idx in range(n_measurements):
                # Get the landmark this detection was assigned to
                assigned_landmark = assignments[meas_idx]
                size_estimator = self.landmark_size_estimators[assigned_landmark]
                self.uncertainty_log.add_detection(assigned_landmark,
                                                   position_cov=lm_position_covs[assigned_landmark, :, :],
                                                   size_var=size_estimator.var/
                                                                            size_estimator.mean**2,
                                                   range_to_landmark=gt_distances_to_measured_landmarks[meas_idx])

    def data_association(self, landmark_measurements, robot_pose):
        """
        Assign new landmark measurements to previously tracked landmarks.
        Uses matching cost based on Mahalanobis distance.

        Parameters
        ----------
        landmark_measurements: ndarray
            N by 3 array of detected landmark range-bearing measurements.
            Each row is [range, bearing, diameter], a landmark detection
        robot_pose: ndarray
            Shape (3,) array containing the robot pose [x, y, theta]

        Returns
        -------
        assignments: ndarray
            N-element numpy array, where N is the number of detections.
            Array that matches detections to previously tracked landmarks.
            Each element is the index of a previous landmark, or -1 (indicating no match)
        """
        n_measurements = landmark_measurements.shape[0]

        # Initialize all assignments as -1 (not assigned to any previously tracked landmark)
        assignments = np.full(n_measurements, -1, dtype=int)
        assigned_costs = np.zeros_like(assignments, dtype=float)

        # No landmarks being tracked yet; skip data assoc (initialize new landmark for each detection)
        if self.n_landmarks == 0:
            return assignments, assigned_costs

        # Create cost matrix
        costs = np.zeros((n_measurements, self.n_landmarks))
        # for each tree detection
        lm_pos_means, lm_pos_covs = self.get_landmarks_position_estimate()
        landmark_range_bearing_measurements = landmark_measurements[:,0:2]
        lm_size_means, lm_size_vars = self.get_landmarks_size_estimate()
        landmark_size_measurements = landmark_measurements[:,2]
        expected_measurements = self.get_landmarks_expected_range_bearing(robot_pose=robot_pose)
        for det_idx in range(n_measurements):
            # Calculate the sensor noise covariance matrix of this measurement
            z_range = landmark_range_bearing_measurements[det_idx, 0]
            z_size = landmark_size_measurements[det_idx]
            range_sigma, bearing_sigma = self.range_bearing_sensor.get_sigmas(z_range)
            size_sigma = self.size_sensor.get_sigma(z_size)
            rb_measurement_cov = np.diag([range_sigma**2, bearing_sigma**2])
            # Use numba helper function to quickly fill in this row of the cost matrix
            data_assocation_costs_helper(costs, det_idx, robot_pose,
                                         landmark_measurements, expected_measurements,
                                         lm_pos_means, lm_pos_covs, lm_size_means,
                                         rb_measurement_cov, size_sigma)
            # for each existing landmark
            # for lm_idx in range(self.n_landmarks):
            #     ## Mahalanobis distance - range-bearing term
            #     # Calculate difference vector btwn expected and observed range-bearing measurements
            #     d_measurement = (landmark_range_bearing_measurements[det_idx, :] -
            #                      expected_measurements[lm_idx, :])
            #     # Ensure that bearing difference is in the range +-180 degrees
            #     # https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
            #     d_measurement[1] = fix_angle(d_measurement[1])
            #     # Mod difference in radians by 2pi
            #
            #     # Get covariance of the landmark's X-Y position estimate
            #     xy_cov = lm_pos_covs [lm_idx, :, :]
            #
            #     # Project the X-Y position covariance into covariance on the range and bearing
            #     jacobian = bearing_range_jacobian(robot_pose=robot_pose,
            #                                       landmark_position=lm_pos_means[lm_idx, :])
            #     rb_cov = jacobian.dot(xy_cov).dot(jacobian.T)
            #
            #     # Add the sensor noise
            #     mahal_rb_cov = rb_cov + rb_measurement_cov
            #     mahal_dist_rb = d_measurement.dot(np.linalg.inv(mahal_rb_cov)).dot(d_measurement)
            #
            #     ## Mahalanobis distance - size term
            #     # Expected landmark size is the current size estimate mean
            #     d_size = landmark_size_measurements[det_idx] - lm_size_means[lm_idx]
            #     mahal_dist_size = d_size * 1./(size_sigma**2) * d_size
            #
            #     # Calculate the Mahalanobis distance
            #     mahal_dist = math.sqrt(mahal_dist_rb + mahal_dist_size)
            #     # mahal_dist = math.sqrt(mahal_dist_rb)
            #
            #     if mahal_dist > MAX_MAHALANOBIS_DISTANCE:
            #         # Disallow any matches greater than MAX_DIST std devs apart, in Mahal distance
            #         cost = 100000
            #     else:
            #         cost = mahal_dist
            #     costs[det_idx][lm_idx] = cost

        rows, cols = linear_sum_assignment(costs, maximize=False)
        for r, c in zip(rows, cols): # for each detection
            # Do not consider matchings over the maximum threshold
            if math.sqrt(costs[r, c]) < MAX_MAHALANOBIS_DISTANCE:
                assignments[r] = c
                assigned_costs[r] = costs[r, c]

        return assignments, assigned_costs


    # DEPRECATED old data association function for relative XY measurements
    def data_association_position(self, landmark_measurements, mahalanobis=True):
        """
        Assign new landmark measurements to previously tracked landmarks.
        Uses matching cost based on Mahalanobis distance, or Euclidean distance in the world frame.
        Both metrics consider landmark position only and ignore landmark size.

        Parameters
        ----------
        landmark_measurements: ndarray
            N by 3 array of detected landmark detections, *in the global frame*
            Each row is [x_global, y_global, diameter], a landmark detection
        mahalanobis: bool
            If True, use Mahalanobis distance for data association (accounts for uncertainty).
            If False, use Euclidean distance for data association.

        Returns
        -------
        assignments: ndarray
            N-element numpy array, where N is the number of detections.
            Array that matches detections to previously tracked landmarks.
            Each element is the index of a previous landmark, or -1 (indicating no match)
        """
        n_measurements = landmark_measurements.shape[0]

        # Initialize all assignments as -1 (not assigned to any previously tracked landmark)
        assignments = np.full(n_measurements, -1, dtype=int)
        assigned_costs = np.zeros_like(assignments, dtype=float)

        # No landmarks being tracked yet; skip data assoc (initialize new landmark for each detection)
        if self.n_landmarks == 0:
            return assignments, assigned_costs

        # Create cost matrix
        costs = np.zeros((n_measurements, self.n_landmarks))
        # for each tree detection
        lm_pos_means, lm_pos_covs = self.get_landmarks_position_estimate()
        for det_idx in range(n_measurements):
            # for each existing landmark
            for lm_idx in range(self.n_landmarks):
                # mu_expected = landmark_estimates[j]
                landmark_expected_position = self.current_estimate.atPoint2(L(lm_idx))
                if mahalanobis:
                    # Mahalanobis distance, position and size
                    d_position = landmark_measurements[det_idx, 0:2] - landmark_expected_position
                    mahal_cov = lm_pos_covs[lm_idx, :, :]
                    mahal_dist = math.sqrt(d_position.dot(np.linalg.inv(mahal_cov)).dot(d_position))
                    if mahal_dist > MAX_MAHALANOBIS_DISTANCE:
                        # Disallow any matches greater than MAX_DIST std devs apart, in Mahal distance
                        cost = 10000
                    else:
                        cost = mahal_dist
                    costs[det_idx][lm_idx] = cost

                else:
                    # Euclidean distance, position only
                    costs[det_idx][lm_idx] = np.linalg.norm(landmark_measurements[det_idx,0:2]
                                                            - landmark_expected_position)

        rows, cols = linear_sum_assignment(costs, maximize=False)
        for r, c in zip(rows, cols): # for each detection

            # Do not consider matchings over the maximum threshold
            if math.sqrt(costs[r, c]) < MAX_MAHALANOBIS_DISTANCE:
                assignments[r] = c
                assigned_costs[r] = costs[r, c]

        return assignments, assigned_costs


    def get_landmarks_expected_range_bearing(self, robot_pose):
        """
        Get the expected range and bearing measurements to all landmarks.

        Returns
        -------
        array_like
            N by 2 array, where row i contains (r, b) the expected range and bearing measurements
            to landmark i, given the current robot pose.
        """
        # Get mean position of each landmark
        mean_positions = self.get_landmarks_position_size_mean()[:,0:2]

        # Calculate displacement vector from robot position, to each landmark
        robot_position = robot_pose[0:2]
        displacement_vectors = mean_positions - robot_position

        # Calculate ranges to each landmark
        ranges_to_landmarks = np.linalg.norm(mean_positions - robot_position, axis=1)

        # Calculate bearings to each landmark
        global_frame_angles = np.arctan2(displacement_vectors[:,1], displacement_vectors[:,0])
        robot_orientation = robot_pose[2]
        # TODO check that subtracting robot orientation does not cause issues with angles needed to be % 2pi
        #      could fix by converting displacement vector to local frame coordinates /before/ taking arctan2
        bearing_angles = global_frame_angles - robot_orientation

        measurements = np.empty((self.n_landmarks,2))
        measurements[:,0] = ranges_to_landmarks
        measurements[:,1] = bearing_angles
        return measurements

    def get_pose_estimate(self):
        """
        Get the current estimate of the robot pose,
        consisting of a mean and covariance matrix.

        Returns
        -------
        mean: ndarray
            Size (3) ndarray: [x, y, theta]
        cov: ndarray
            Size (3, 3) ndarray; the 3D covariance matrix of the pose estimate.

        """
        estimate: gtsam.Values = self.current_estimate
        pose: gtsam.Pose2 = estimate.atPose2(X(self.time_step))
        mean = np.array([pose.x(), pose.y(), pose.theta()])
        cov = self.isam.marginalCovariance(X(self.time_step))
        return mean, cov

    def get_landmarks_position_estimate(self):
        """
        Get the current estimate of all landmarks

        Returns
        -------
        means: ndarray
            Size (n_landmarks, 2) ndarray.
            Row i contains [x, y] coordinates for landmark i.
        covs: ndarray
            Size (n_landmarks, 2, 2) ndarray.
            Slice [i,:,:] is the 2x2 covariance matrix for landmark i.

        """
        estimate: gtsam.Values = self.current_estimate
        means = np.empty((self.n_landmarks, 2))
        covs = np.empty((self.n_landmarks, 2, 2))
        for i in range(self.n_landmarks):
            means[i,:] = estimate.atPoint2(L(i))
            covs[i,:,:] = self.isam.marginalCovariance(L(i))
        return means, covs

    def get_landmarks_size_estimate(self):
        means = np.empty(self.n_landmarks)
        vars = np.empty(self.n_landmarks)
        for i in range(self.n_landmarks):
            means[i] = self.landmark_size_estimators[i].mean
            vars[i] = self.landmark_size_estimators[i].var
        return means, vars

    def get_landmarks_position_size_mean(self):
        """
        Get the mean estimate for all landmarks, including position and diameter

        Returns
        -------
        ndarray
            Size (n_landmarks, 3) ndarray.
            Row i contains [x, y, diameter] for landmark i

        """
        estimate: gtsam.Values = self.current_estimate
        means = np.empty((self.n_landmarks, 3))
        for i in range(self.n_landmarks):
            means[i,0:2] = estimate.atPoint2(L(i))
            means[i, 2] = self.landmark_size_estimators[i].mean
        return means

    def get_landmarks_estimate(self) -> List[Obstacle2D]:
        """
        Returns the current estimated obstacles as Obstacel2D objects.
        Also includes boundary obstacles, if used.

        Returns
        -------
        List[Obstacle2D]
            List of Obstacle2D objects representing the landmark estimates.

        """
        if self.n_landmarks == 0:
            return []
        landmarks = []
        landmark_pos_means, landmark_pos_covs = self.get_landmarks_position_estimate()
        landmark_size_means, landmark_size_vars = self.get_landmarks_size_estimate()

        for i in range(self.n_landmarks):
            pos_mean = landmark_pos_means[i,:]
            pos_cov = landmark_pos_covs[i,:,:]
            size_mean = landmark_size_means[i]
            size_var = landmark_size_vars[i]
            landmark = Obstacle2D(pos_mean=pos_mean, pos_cov=pos_cov,
                                  size_mean=size_mean, size_var=size_var,
                                  n_measurements=self.landmarks_n_measurements[i])
            landmarks.append(landmark)
        return landmarks

    def get_landmarks_n_measurements(self):
        n_measurements = np.empty(self.n_landmarks)
        for i in range(self.n_landmarks):
            n_measurements[i] = self.landmarks_n_measurements[i]
        return n_measurements

    def add_landmark(self):
        # Create a new landmark for SLAM.
        # Returns the integer index of the newly added landmark.
        idx = self.n_landmarks
        self.landmark_size_estimators[idx] = LandmarkSizeEstimator(init_mean=LM_INIT_SIZE_MEAN,
                                                                 init_var=LM_INIT_SIZE_VAR)
        self.landmarks_n_measurements[idx] = 0
        self._is_boundary[idx] = False
        self.closest_robot_to_landmark_distance[idx] = float('inf')

        if self.uncertainty_log is not None:
            self.uncertainty_log.add_landmark(idx)

        self.n_landmarks += 1
        return idx

    def get_landmark_closest_distance(self, landmark_idx: int):
        """ Given a landmark index, return the closest this landmark has ever been from the robot."""
        return self.closest_robot_to_landmark_distance[landmark_idx]


    def get_obstacles(self):
        """
        Get the SLAM estimated obstacles as a list of Obstacle2D objects

        Returns
        -------
        List[Obstacle2D]

        """
        obstacles = []
        pos_means, pos_covs = self.get_landmarks_position_estimate()
        size_means, size_vars = self.get_landmarks_size_estimate()

        for i in range(pos_means.shape[0]):
            obs = Obstacle2D(pos_mean=pos_means[i,:], pos_cov=pos_covs[i,:,:],
                             size_mean=size_means[i], size_var=size_vars[i])
            obstacles.append(obs)
        return obstacles

@numba.njit()
def bearing_range_jacobian(robot_pose, landmark_position):
    """
    Compute the Jacobian of the range and bearing measurements, with respect to changes in the
    landmark's X- and Y-position.

    This allows us to project the uncertainty in the X- and Y- variables, given by the graph SLAM
    landmark estimates,
    to the range-bearing space, for determining the Mahalanobis distance to use for data association
    with range and bearing measurements.

    Parameters
    ----------
    robot_pose
    landmark_position

    Returns
    -------
    array_like
        2 by 2 Jacobian matrix:
        [ dR/dx  dR/dy ]
        [ dB/dx  dB/dy ]
        Partial derivatives of range R and bearing angle B,
        w.r.t. landmark position (x, y)

    """
    # Get landmarks position
    x_lm = landmark_position[0]
    y_lm = landmark_position[1]

    # Get robot position
    x_r = robot_pose[0]
    y_r = robot_pose[1]

    # Calculate current range of landmark from robot
    robot_to_lm_dist = np.linalg.norm(landmark_position - robot_pose[0:2])

    # Calculate partial derivatives of range measurement
    dR_dx =  (x_lm - x_r) / robot_to_lm_dist
    dR_dy =  (y_lm - y_r) / robot_to_lm_dist

    # Calculate partial derivatives of bearing measurement
    dB_dx = (y_r - y_lm) / robot_to_lm_dist**2
    dB_dy = (x_lm - x_r) / robot_to_lm_dist**2

    # Construct Jacobian
    return np.array([[dR_dx, dR_dy], [dB_dx, dB_dy]])


"""
@numba.njit()
def data_association_helper(landmark_measurements, robot_pose
                            lm_pos_means, lm_pos_covs,
                            lm_size_means, lm_size_vars,
                            expected_measurements):
    n_measurements = landmark_measurements.shape[0]

    # Initialize all assignments as -1 (not assigned to any previously tracked landmark)
    assignments = np.full(n_measurements, -1, dtype=int)
    assigned_costs = np.zeros_like(assignments, dtype=float)

    # No landmarks being tracked yet; skip data assoc (initialize new landmark for each detection)
    if self.n_landmarks == 0:
        return assignments, assigned_costs

    # Create cost matrix
    costs = np.zeros((n_measurements, self.n_landmarks))
    # for each tree detection
    lm_pos_means, lm_pos_covs = self.get_landmarks_position_estimate()
    landmark_range_bearing_measurements = landmark_measurements[:,0:2]
    lm_size_means, lm_size_vars = self.get_landmarks_size_estimate()
    landmark_size_measurements = landmark_measurements[:,2]
    expected_measurements = self.get_landmarks_expected_range_bearing(robot_pose=robot_pose)
    for det_idx in range(n_measurements):
        # Calculate the sensor noise covariance matrix of this measurement
        z_range = landmark_range_bearing_measurements[det_idx, 0]
        z_size = landmark_size_measurements[det_idx]
        range_sigma, bearing_sigma = self.range_bearing_sensor.get_sigmas(z_range)
        size_sigma = self.size_sensor.get_sigma(z_size)
        rb_measurement_cov = np.diag([range_sigma**2, bearing_sigma**2])
        # for each existing landmark
        for lm_idx in range(self.n_landmarks):
            ## Mahalanobis distance - range-bearing term
            # Calculate difference vector btwn expected and observed range-bearing measurements
            d_measurement = (landmark_range_bearing_measurements[det_idx, :] -
                             expected_measurements[lm_idx, :])
            # Ensure that bearing difference is in the range +-180 degrees
            # https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
            # TODO could rework the functions for generating measurements and expected measurements so this loop is not necessary
            d_measurement[1] = fix_angle(d_measurement[1])
            # Mod difference in radians by 2pi

            # Get covariance of the landmark's X-Y position estimate
            xy_cov = lm_pos_covs [lm_idx, :, :]

            # Project the X-Y position covariance into covariance on the range and bearing
            jacobian = bearing_range_jacobian(robot_pose=robot_pose,
                                              landmark_position=lm_pos_means[lm_idx, :])
            rb_cov = jacobian.dot(xy_cov).dot(jacobian.T)

            # Add the sensor noise
            mahal_rb_cov = rb_cov + rb_measurement_cov
            mahal_dist_rb = d_measurement.dot(np.linalg.inv(mahal_rb_cov)).dot(d_measurement)

            ## Mahalanobis distance - size term
            # Expected landmark size is the current size estimate mean
            d_size = landmark_size_measurements[det_idx] - lm_size_means[lm_idx]
            mahal_dist_size = d_size * 1./(size_sigma**2) * d_size

            # Calculate the Mahalanobis distance
            mahal_dist = math.sqrt(mahal_dist_rb + mahal_dist_size)
            # mahal_dist = math.sqrt(mahal_dist_rb)

            if mahal_dist > MAX_MAHALANOBIS_DISTANCE:
                # Disallow any matches greater than MAX_DIST std devs apart, in Mahal distance
                cost = 100000
            else:
                cost = mahal_dist
            costs[det_idx][lm_idx] = cost

    rows, cols = linear_sum_assignment(costs, maximize=False)
    for r, c in zip(rows, cols): # for each detection
        # Do not consider matchings over the maximum threshold
        if math.sqrt(costs[r, c]) < MAX_MAHALANOBIS_DISTANCE:
            assignments[r] = c
            assigned_costs[r] = costs[r, c]


"""

@numba.njit()
# TODO this helper function reduces the runtime of data association by 30-50% with many landmarks.
#      Could implement more of the data association function in numba to speed it up further.
def data_assocation_costs_helper(costs, det_idx: int,
                                 robot_pose,
                                 landmark_measurements, expected_measurements,
                                 lm_pos_means, lm_pos_covs,
                                 lm_size_means,
                                 rb_measurement_cov, size_sigma):
    n_landmarks = costs.shape[1]
    landmark_range_bearing_measurements = landmark_measurements[:,0:2]
    landmark_size_measurements = landmark_measurements[:,2]
    for lm_idx in range(n_landmarks):
        ## Mahalanobis distance - range-bearing term
        # Calculate difference vector btwn expected and observed range-bearing measurements
        d_measurement = (landmark_range_bearing_measurements[det_idx, :] -
                         expected_measurements[lm_idx, :])
        # Ensure that bearing difference is in the range +-180 degrees
        # https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
        d_measurement[1] = fix_angle(d_measurement[1])
        # Mod difference in radians by 2pi

        # Get covariance of the landmark's X-Y position estimate
        xy_cov = lm_pos_covs [lm_idx, :, :]

        # Project the X-Y position covariance into covariance on the range and bearing
        jacobian = bearing_range_jacobian(robot_pose=robot_pose,
                                          landmark_position=lm_pos_means[lm_idx, :])
        rb_cov = jacobian.dot(xy_cov).dot(jacobian.T)

        # Add the sensor noise
        mahal_rb_cov = rb_cov + rb_measurement_cov
        mahal_dist_rb = d_measurement.dot(np.linalg.inv(mahal_rb_cov)).dot(d_measurement)

        ## Mahalanobis distance - size term
        # Expected landmark size is the current size estimate mean
        d_size = landmark_size_measurements[det_idx] - lm_size_means[lm_idx]
        mahal_dist_size = d_size * 1./(size_sigma**2) * d_size

        # Calculate the Mahalanobis distance
        mahal_dist = math.sqrt(mahal_dist_rb + mahal_dist_size)
        # mahal_dist = math.sqrt(mahal_dist_rb)

        if mahal_dist > MAX_MAHALANOBIS_DISTANCE:
            # Disallow any matches greater than MAX_DIST std devs apart, in Mahal distance
            cost = 100000
        else:
            cost = mahal_dist
        costs[det_idx][lm_idx] = cost
