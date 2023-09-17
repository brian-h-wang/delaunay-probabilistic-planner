"""
Module containing models of obstacle sensor noise.
"""

from abc import ABC
import math
import numpy as np
import numpy.typing as npt
from typing import Tuple
from planner.utils import fix_angle

class PositionSensorModel(ABC):

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate_measurement(self, robot_pose: npt.ArrayLike, obstacle_position: npt.ArrayLike,
                             obstacle_diameter: float) -> Tuple[float, float]:
        """

        Parameters
        ----------
        robot_pose
            [x, y, theta]
        obstacle_position
            [x, y]
            True position of the obstacle
        obstacle_diameter: float
            True diameter of the obstacle

        Returns
        -------
        array_like
            Measurement as [rel_x, rel_y],
            consisting of obstacle position relative to the robot,
            given in *robot body frame coordinates* (x forwards, y left)

        """
        raise NotImplementedError()

    def get_covariance(self, range_to_obstacle: float) -> npt.ArrayLike:
        raise NotImplementedError()



class SizeSensorModel(ABC):

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate_measurement(self, robot_position: npt.ArrayLike, obstacle_position: npt.ArrayLike,
                             obstacle_diameter: float) -> float:
        raise NotImplementedError()

    def get_sigma(self, obstacle_diameter: float) -> float:
        raise NotImplementedError()


class LinearNoisePositionSensor(PositionSensorModel):

    def __init__(self, sigma_min: float, sigma_max: float, max_range: float, seed=None):
        """
        Class representing an obstacle position sensor whose noise linearly increases with
        range from the robot.

        Parameters
        ----------
        sigma_min: float
            Standard deviation specifying the minimum amount of noise in the sensor measurements.
            Obstacle detections at range 0 wile have noise std dev of sigma_min.
        sigma_max: float
            Standard deviation specifying the maximum amount of noise in the sensor measurements.
            Obstacle detections at max_range will have noise std dev of sigma_max.
        max_range: float
            Maximum range of the sensor. Obstacle detections at this range will have noise std dev
            equal to sigma_max.
        """
        super().__init__(seed=seed)
        self._max_range = max_range
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._scale_factor = (self._sigma_max - self._sigma_min) / self._max_range

    def generate_measurement(self, robot_pose: npt.ArrayLike, obstacle_position: npt.ArrayLike,
                             obstacle_diameter: float) -> Tuple[float, float]:
        robot_position = robot_pose[0:2]
        robot_orientation = robot_pose[2]
        R = np.array([[math.cos(robot_orientation), math.sin(robot_orientation)],
                      [-math.sin(robot_orientation), math.cos(robot_orientation)]])
        range_to_obs = np.linalg.norm(robot_position - obstacle_position)
        position_std = self._sigma_min + range_to_obs * self._scale_factor
        # Generate noisy x and y measurements
        rel_pos_body_frame = R.dot(obstacle_position - robot_position)
        x = rel_pos_body_frame[0] + self.rng.normal(0.0, position_std)
        y = rel_pos_body_frame[1] + self.rng.normal(0.0, position_std)
        return x, y

    def get_covariance(self, range_to_obstacle: float) -> npt.ArrayLike:
        position_std = self._sigma_min + range_to_obstacle * self._scale_factor
        return position_std**2 * np.eye(2)


class RangeBearingSensorModel(ABC):

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate_measurement(self, robot_pose: npt.ArrayLike,
                             obstacle_position: npt.ArrayLike) -> Tuple[float, float]:
        """

        Parameters
        ----------
        robot_pose
            [x, y, theta]
        obstacle_position
            [x, y]
            True position of the obstacle

        Returns
        -------
        range: float
            Measurement of range to the obstacle.
        bearing: float
            Measurement of bearing to the obstacle, in radians.
            Relative to robot orientation.
        """
        robot_position = robot_pose[0:2]
        robot_orientation = robot_pose[2]
        # Calculate true range and bearing
        range_to_obs = np.linalg.norm(robot_position - obstacle_position)
        x_rel = obstacle_position[0] - robot_position[0]
        y_rel = obstacle_position[1] - robot_position[1]
        bearing = math.atan2(y_rel, x_rel) - robot_orientation
        # Generate noisy range and bearing measurements
        range_sigma, bearing_sigma = self.get_sigmas(range_to_obs)
        r_meas = range_to_obs + self.rng.normal(0.0, range_sigma)
        # Ensure bearing measurement is within +-180 degrees
        b_meas = fix_angle(bearing + self.rng.normal(0.0, bearing_sigma))

        return r_meas, b_meas

    def get_sigmas(self, range_to_obstacle: float) -> Tuple[float, float]:
        # TODO change other sensors to give sigmas instead of covariance
        """

        Parameters
        ----------
        range_to_obstacle: float
            True distance from the robot to the obstacle.
            Needed because sensor noise sigmas may increase with range

        Returns
        -------
        sigma_range: float
            Standard deviation of the range measurement noise.
        sigma_bearing: float
            Standard deviation of the bearing measurement noise
        """
        raise NotImplementedError()



class LinearNoiseRangeBearingSensor(RangeBearingSensorModel):

    def __init__(self, range_sigma_min: float, range_sigma_max: float, max_range: float,
                 bearing_sigma: float, seed=None):
        """
        Class representing a range-bearing sensor whose range noise std dev
        increases linearly with range from the robot.

        Parameters
        ----------
        range_sigma_min: float
            Standard deviation specifying the minimum amount of noise in the range measurements.
            Obstacle detections at range 0 wile have range noise std dev of sigma_min.
        range_sigma_max: float
            Standard deviation specifying the maximum amount of noise in the range measurements.
            Obstacle detections at max_range will have range noise std dev of sigma_max.
        max_range: float
            Maximum range of the sensor. Obstacle detections at this range will have noise std dev
            equal to sigma_max.
        bearing_sigma: float

        """
        super().__init__(seed=seed)
        self._max_range = max_range
        self._range_sigma_min = range_sigma_min
        self._range_sigma_max = range_sigma_max
        self._range_scale_factor = (self._range_sigma_max - self._range_sigma_min) / self._max_range
        self._bearing_sigma = bearing_sigma

    def get_sigmas(self, range_to_obstacle: float):
        # Range sigma increases linearly with range to obstacle
        range_sigma = self._range_sigma_min + range_to_obstacle * self._range_scale_factor
        # Bearing sigma is constant
        return range_sigma, self._bearing_sigma


class QuadraticNoiseRangeBearingSensor(RangeBearingSensorModel):

    def __init__(self, range_sigma_min: float, range_sigma_max: float, max_range: float,
                 bearing_sigma: float, seed=None):
        """
        Class representing a range-bearing sensor whose range noise std dev
        increases quadratically with range from the robot.

        Parameters
        ----------
        range_sigma_min: float
            Standard deviation specifying the minimum amount of noise in the range measurements.
            Obstacle detections at range 0 wile have range noise std dev of sigma_min.
        range_sigma_max: float
            Standard deviation specifying the maximum amount of noise in the range measurements.
            Obstacle detections at max_range will have range noise std dev of sigma_max.
        max_range: float
            Maximum range of the sensor. Obstacle detections at this range will have noise std dev
            equal to sigma_max.
        bearing_sigma: float

        """
        super().__init__(seed=seed)
        self._max_range = max_range
        self._range_sigma_min = range_sigma_min
        self._range_sigma_max = range_sigma_max
        self._range_scale_factor = (self._range_sigma_max - self._range_sigma_min) / (self._max_range**2)
        self._bearing_sigma = bearing_sigma

    def get_sigmas(self, range_to_obstacle: float):
        # Range sigma increases linearly with range to obstacle
        range_sigma = self._range_sigma_min + range_to_obstacle**2 * self._range_scale_factor
        # Bearing sigma is constant
        return range_sigma, self._bearing_sigma


class QuadraticProportionalNoiseRangeBearingSensor(RangeBearingSensorModel):

    def __init__(self, range_proportion_min: float, range_proportion_max: float, max_range: float,
                 bearing_sigma: float, seed=None):
        """
        Class representing a range-bearing sensor whose range noise std dev is proportional to the
        true range to the obstacle.

        The proportion of noise grows quadratically with range.

        Based on percent errors model from:
        https://support.stereolabs.com/hc/en-us/articles/206953039-How-does-the-ZED-work

        Parameters
        ----------
        range_proportion_min: float
            Standard deviation specifying the minimum amount of noise in the range measurements.
            Obstacle detections at range 0 wile have range noise std dev of sigma_min.
        range_proportion_max: float
            Standard deviation specifying the maximum amount of noise in the range measurements.
            Obstacle detections at max_range will have range noise std dev of sigma_max.
        max_range: float
            Maximum range of the sensor. Obstacle detections at this range will have noise std dev
            equal to sigma_max.
        bearing_sigma: float

        """
        super().__init__(seed=seed)
        self._max_range = max_range
        self._range_proportion_min = range_proportion_min
        self._range_proportion_max = range_proportion_max
        self._proportion_scale_factor = (self._range_proportion_max - self._range_proportion_min) / (self._max_range**2)
        self._bearing_sigma = bearing_sigma

    def get_sigmas(self, range_to_obstacle: float):
        # Standard deviation proportion increases quadratically with range
        sigma_proportion = self._range_proportion_min + range_to_obstacle**2 * self._proportion_scale_factor
        range_sigma = range_to_obstacle * sigma_proportion
        # Bearing sigma is constant
        return range_sigma, self._bearing_sigma


class ProportionalNoiseSizeSensor(SizeSensorModel):
    """
    Class representing an object size sensor whose noise is proportional to the size of the detected
    object
    """

    def __init__(self, noise_factor: float, seed=None):
        super().__init__(seed=seed)
        self._noise_factor = noise_factor

    def generate_measurement(self, robot_position: npt.ArrayLike, obstacle_position: npt.ArrayLike,
                             obstacle_diameter: float) -> float:
        size_std = self.get_sigma(obstacle_diameter)
        return obstacle_diameter + self.rng.normal(0.0, size_std)

    # def get_variance(self, obstacle_diameter: float) -> float:
    #     return (self._noise_factor * obstacle_diameter)**2

    def get_sigma(self, obstacle_diameter: float) -> float:
        return self._noise_factor * obstacle_diameter
