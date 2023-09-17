"""
Simulate the probability of safely navigating between a pair of 1D obstacles with uncertain,
Gaussian distributed, 1D positions and size.
"""

from typing import List
import math
import numpy as np
from numpy.typing import ArrayLike
import numba
from scipy.stats import norm
from scipy.optimize import minimize_scalar


class SafetyProbability(object):
    """
    Class for storing a probability of safe navigation, and calculating the associated number
    of standard deviations required for bloating obstacle distributions.
    """

    def __init__(self, probability: float):
        assert 0 <= probability < 1.0, "Safety probability must be greater than or equal to 0, " \
                                       "and less than 1."
        self.probability = probability

        # Solve for the number of standard deviations,
        #  considering only one tail of the normal distribution,
        #  needed to obtain this probability of safety
        def f(x):
            return (probability - norm.cdf(x, 0, 1))**2

        sol = minimize_scalar(f, method='bounded', bounds=[0.0, 10.0])
        self.n_sigmas = sol.x

    def n_sigmas_obstacle_bloat(self, std_width, std_o1, std_o2):
        """
        Return the number of standard deviations to bloat two separate obstacles,
        given that the overall width of the safe space between them should exceed the robot width
        with the desired probability of safety.

        Parameters
        ----------
        std_width: float
            Standard deviation of the Gaussian random variable representing the width of the space
            between two normally distributed obstacles.
        std1: float
        std2: float
            Standard deviations of the two Gaussian random variables representing the positions
            of the two obstacles.

        Returns
        -------
        float
            The number of standard deviations by which to bloat both obstacles.

        """
        return self.n_sigmas * std_width / (std_o1 + std_o2)

class Obstacle1D(object):

    def __init__(self, pos_mean, pos_var, size_mean, size_var):
        self.pos_mean = float(pos_mean)
        self.pos_var = float(pos_var)
        self.size_mean = float(size_mean)  # size is obstacle diameter
        self.size_var = float(size_var)

    @property
    def pos_std(self):
        return math.sqrt(self.pos_var)

    @property
    def size_std(self):
        return math.sqrt(self.size_var)

    @property
    def var(self):
        return self.pos_var + self.size_var

    @property
    def radius_mean(self):
        return self.size_mean / 2.

    @property
    def radius_var(self):
        return self.size_var/4.

    @property
    def radius_std(self):
        return math.sqrt(self.radius_var)

    def pdf(self, x, subtract_size=False):
        if not subtract_size:
            s = 1
        else:
            s = -1
        return norm.pdf(x, self.pos_mean + s*self.size_mean, self.var)

    def sample(self):
        """
        Sample a position and size for this obstacle.

        Returns
        -------
        float, float
            Position and size

        """
        pos = np.random.normal(self.pos_mean, self.pos_std)
        size = np.random.normal(self.size_mean, self.size_std)
        return pos, size


class Obstacle2D(object):
    """
    Class for representing a circular, 2D obstacle, with uncertainty in position and size.
    """

    def __init__(self, pos_mean, pos_cov, size_mean, size_var, n_measurements=None):
        """

        Parameters
        ----------
        pos_mean: ndarray
            A 2 by 1 array containing the obstacle position mean (as [x, y]')
        pos_cov: ndarray
            A 2 by 2 array containing the obstacle position covariance.
        size_mean: float
            Mean diameter of this obstacle
        size_var float
            Variance of the diameter
        n_measurements: int or None
            Number of sensor measurements associated with this landmark estimate.
        """
        self.pos_mean = np.array(pos_mean, dtype=float).reshape((2,1))
        self.pos_cov = np.array(pos_cov, dtype=float).reshape((2,2))

        self.size_mean = float(size_mean)
        self.size_var = float(size_var)

        self.n_measurements = n_measurements

    @property
    def size_std(self):
        return math.sqrt(self.size_var)

    def check_robot_collision(self, robot_position: ArrayLike, robot_width: float, n_sigma: float = 2):
        """
        Check if a robot with a given position and width is in collision with the obstacle estimate,
        within a certain number of std deviations uncertainty bound.

        Returns
        -------
        bool
            True if in collision, False if no collision
        """
        robot_position = np.array(robot_position, dtype=float)
        return _check_collision_with_estimate(robot_position=robot_position, robot_width=robot_width,
                                              obs_pos_mean=self.pos_mean, obs_pos_cov=self.pos_cov,
                                              obs_size_mean=self.size_mean, obs_size_var=self.size_var,
                                              n_sigma=n_sigma)

    def sample(self, n_samples=None):
        """
        Sample a position and size for this obstacle.

        Returns
        -------
        pos: array_like
            Sampled position as array of size 2
        size: float or array_like
            Sampled diameter

        """
        if n_samples is None:
            pos = np.random.multivariate_normal(self.pos_mean.flatten(), self.pos_cov)
            size = np.random.normal(self.size_mean, self.size_std)
            return pos, size
        else:
            positions = np.random.multivariate_normal(self.pos_mean.flatten(), self.pos_cov, n_samples)
            sizes = np.random.normal(self.size_mean, self.size_std, n_samples)
            return positions, sizes

class ProjectedObstaclesPair(object):
    """
    Class representing a pair of 2D obstacles projected to 1D.

    Allows calculating safe navigation probabilities, and midpoints, as with 1D obstacles.
    Also allows transforming calculated points back to the original 2D coordinate frame.

    Attributes
    ----------
    obstacle1: Obstacle1D
    obstacle2: Obstacle1D
        The two obstacles, projected to 1D
    R: ndarray
        2 by 2 array, the rotation matrix used to rotate points from the original 2D coordinate
        frame, to the coordinate frame whose x-axis points from obstacle1's mean to obstacle2's
        mean.
    y: float
        The y-coordinate of the two obstacles, after rotation.
        Used to convert back to 2D coordinates in the original 2D frame.
    """

    # def __init__(self, obstacle1: Obstacle1D, obstacle2: Obstacle1D, R: np.ndarray, y: float):
    def __init__(self, obstacle1: Obstacle2D, obstacle2: Obstacle2D):
        """

        Parameters
        ----------
        obs1: Obstacle1D
        obs2: Obstacle1D
        R: ndarray
            2 by 2 array, the rotation matrix used to rotate points from the original 2D coordinate
            frame, to the coordinate frame whose x-axis points from obstacle1's mean to obstacle2's
            mean.
        y: float
            The y-coordinate of the two obstacles, after rotation
        """
        # self.obstacle1 = obstacle1
        # self.obstacle2 = obstacle2
        # self.R = R
        # self.y = float(y)
        # Find angle of the line from one obstacle's position mean to the other
        angle = math.atan2(obstacle2.pos_mean[1] - obstacle1.pos_mean[1],
                           obstacle2.pos_mean[0] - obstacle1.pos_mean[0])
        c = math.cos(angle)
        s = math.sin(angle)

        # Rotation matrix from original 2D coordinates to the coordinate frame
        # whose x-axis points from position mean of obstacle 1 to mean of obstacle 2
        R = np.array([[c, s], [-s, c]])

        # Rotate the means and covariances
        mu1 = (R.dot(obstacle1.pos_mean))
        sigma1 = R.dot(obstacle1.pos_cov).dot(R.T)

        mu2 = (R.dot(obstacle2.pos_mean))
        sigma2 = R.dot(obstacle2.pos_cov).dot(R.T)

        # After rotation, the two position means will have the same y-coordinate
        assert np.isclose(mu1[1], mu2[1]), "Position means do not have the same y-coordinate"
        y = mu1[1,0]

        # Project the obstacles to 1D and create two Obstacle1D objects
        # We can marginalize the rotated 2D Gaussians to get 1D Gaussians by simply extracting
        # the first value in the mean, and top-left value in the cov matrix.
        self.obstacle1 = Obstacle1D(mu1[0], sigma1[0,0], obstacle1.size_mean, obstacle1.size_var)
        self.obstacle2 = Obstacle1D(mu2[0], sigma2[0,0], obstacle2.size_mean, obstacle2.size_var)
        self.R = R
        self.y = y

    def get_safest_point(self, robot_width: float = 0):
        """
        Calculates the point between the pair of obstacles, where the robot is maximally likely
        to be able to move safely between the obstacles without colliding with either.

        Parameters
        ----------
        robot_width: float
            The width of the robot.
            The safest point calculation is also valid for a point robot (width=0).

        Returns
        -------

        """
        mean1 = self.obstacle1.pos_mean + self.obstacle1.size_mean
        var1 = self.obstacle1.pos_var + self.obstacle1.size_var
        mean2 = self.obstacle2.pos_mean - self.obstacle2.size_mean
        var2 = self.obstacle2.pos_var + self.obstacle2.size_var
        # Swap the points if mean1 is greater than mean2
        #   (along the x-axis)
        # This means the obstacles are close enough together and/or
        #   large enough that their mean edges overlap.
        # This will not affect the safety probability calculation
        #   (which should end up being very close to 0)
        #   but the points need to be swapped for the safest point calculation
        #   Otherwise, this gives a min occupancy point very far away
        #   from both the obstacles.
        if mean1 > mean2:
            mean1, mean2 = mean2, mean1
            var1, var2 = var2, var1
        min_point = min_occupancy_point(mean1=mean1, var1=var1,
                                        mean2=mean2, var2=var2,
                                        robot_width=robot_width)

        # Get distance from mean 1 to the midpoint
        # d = min_point - mean1
        # Then multiply this distance by the unit vector from mean1 to mean2 (in original 2D coords)
        return self.R.T.dot(np.array([min_point, self.y]).reshape((2,1)))

    def get_prob_safe_navigation(self, robot_width):
        """
        Calculate the probability that a robot of given width can safely navigate between
        this pair of obstacles.

        Parameters
        ----------
        robot_width: float

        Returns
        -------
        float
            A probability between 0 and 1.0
        """
        return prob_safe_navigation(self.obstacle1, self.obstacle2, robot_width=robot_width)

    def get_navigation_points(self, robot_width: float, safety_probability: SafetyProbability,
                              point_spacing=None):
        """
        Given this pair of obstacles, return the points that
        Parameters
        ----------
        robot_width: float
            The width of the robot (assumed known).

        Returns
        -------
        array_like
            N by 2
            Currently, N is either 1 or 3, and the navigation points are the expected midpoint of the
            free space between the obstacles, as well as the +-2sigma points corresponding to
            an uncertainty bloat around each obstacle.
            If the +2sigma point of obstacle 1 is to the right of the -2sigma bloat point of obstacle
            2, only the edge midpoint will be returned (this should indicate a <95.5% chance that
            the robot can safely navigate between the obstacles)
            TODO update documentation. now uses safetyprobability class for correct bloat amounts

        """
        # Compute how far apart points should be spaced
        # By default, space out points by the robot width
        if point_spacing is None:
            point_spacing = robot_width

        o1 = self.obstacle1
        o2 = self.obstacle2

        # Calculate the standard deviation for the left and right obstacle edge positions
        left_std = math.sqrt(o1.pos_var + o1.radius_var)
        right_std = math.sqrt(o2.pos_var + o2.radius_var)
        free_space_std = math.sqrt(o1.pos_var + o1.radius_var + o2.pos_var + o2.radius_var)

        n_sigmas_bloat = safety_probability.n_sigmas_obstacle_bloat(free_space_std, left_std, right_std)

        # Calculate the number of standard deviations to bloat the left and right obstacles,
        #   in order to obtain the desired safety probability

        left = (o1.pos_mean + o1.radius_mean) + n_sigmas_bloat*left_std + robot_width/2.
        right = (o2.pos_mean - o2.radius_mean) - n_sigmas_bloat*right_std - robot_width/2.

        # If obstacle 1 bloat point is less than obstacle 2 bloat point, the probability of safely
        #   navigating between the obstacles exceeds the desired safety probability
        if left < right:
            # Calculate how many points should be placed
            difference = right - left
            n_points = int(difference // point_spacing) + 2 # always have two points for left and right
            points = np.empty((n_points, 2))
            points[:,0] = np.linspace(left, right, n_points)
        else:
            points = np.empty((1,2))
            # Calculate mean midpoint of the free space
            points[0,0] = ((o1.pos_mean + o1.radius_mean) + (o2.pos_mean - o2.radius_mean)) / 2.
        # Insert y-value for the projected obstacles
        points[:,1] = self.y
        rot = self.R.T  # rotation - 1D frame to original 2D world coordinates
        return (rot.dot(points.T)).T


def min_occupancy_point(mean1, var1, mean2, var2, robot_width=0):
    """

    Parameters
    ----------
    mean1
    var1
    mean2
    var2

    Returns
    -------
    float

    """

    if var1 == var2:
        # If variances are equal there is just one solution
        return (mean1**2 - mean2**2) / (2*mean1 - 2*mean2)
    else:
        a = var1 - var2
        b = 2*(mean1 * var2 - mean2 * var1) + (2 * robot_width * var1)
        c = (mean2**2 * var1 - mean1**2 * var2 +
             2 * var1 * var2 * math.log(math.sqrt(var2) / math.sqrt(var1)) +
             var1 * (robot_width**2 - 2*robot_width*mean2))
        # If the variances are not equal, there will be two solutions
        # Return the solution which lies between the two obstacles' means
        solution1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
        solution2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2*a)
        if mean1 <= solution1 <= mean2:
            return solution1
        else:
            return solution2


def prob_safe_navigation(obstacle1, obstacle2, robot_width):
    """
    Calculate the probability that a robot with known width can safely move between two obstacles,
    with uncertain (Gaussian distributed) 1D position and size.

    Parameters
    ----------
    obstacle1: Obstacle1D
    obstacle2: Obstacle1D
    robot_width: float

    Returns
    -------
    float
        The probability of a robot with the given width having enough space to navigate between the
        obstacles.

    """
    # print("Obs1 at x=%.3f, size=%.3f, radius_mean=%.3f" % (obstacle1.pos_mean, obstacle1.size_mean, obstacle1.radius_mean))
    # print("Obs2 at x=%.3f, size=%.3f, radius_mean=%.3f" % (obstacle2.pos_mean, obstacle2.size_mean, obstacle2.radius_mean))
    # print("")
    # Swap the obstacles if needed, so obstacle 2's position mean is greater than obstacle 1's
    if obstacle1.pos_mean > obstacle2.pos_mean:
        obstacle2, obstacle1 = obstacle1, obstacle2

    # Pos mean, radius mean affect the navigable space between the obstacles
    navigable_mean = (obstacle2.pos_mean - obstacle2.radius_mean) - (obstacle1.pos_mean + obstacle1.radius_mean)
    # Divide diameter variance by 4 to get radius variance
    navigable_std = math.sqrt(obstacle1.pos_var + obstacle1.radius_var + obstacle2.pos_var + obstacle2.radius_var)
    # Use the CDF to find the probability that the navigable space is greater than the robot width
    safety_probability = 1 - norm.cdf(robot_width, navigable_mean, navigable_std)
    #
    # # DEPRECATED code for checking that safety probabilty is consistent with n sigmas bloat
    # o1 = obstacle1
    # o2 = obstacle2
    # std1 = math.sqrt(o1.pos_var + o1.size_var/4.)
    # std2 = math.sqrt(o2.pos_var + o2.size_var/4.)
    #
    # n_d = 1.690144
    # n_bloat = n_d * navigable_std / (std1 + std2)
    #
    # p1 = o1.pos_mean + o1.radius_mean + n_bloat*std1 + robot_width/2.
    # p2 = o2.pos_mean - o2.radius_mean - n_bloat*std2 - robot_width/2.
    # # if 0.50 < safety_probability < 0.96:
    # if p1 > p2 and safety_probability > 0.9545:
    #     print("robot width is %.3f" % robot_width)
    #     print("prob is %.4f" % (safety_probability))
    #     # print("O1: position (%.3f, %.3f), size (%.3f, %.3f)" % (o1.pos_mean, math.sqrt(o1.pos_var), o1.size_mean, math.sqrt(o1.size_var)))
    #     # print("O2: position (%.3f, %.3f), size (%.3f, %.3f)" % (o2.pos_mean, math.sqrt(o2.pos_var), o2.size_mean, math.sqrt(o2.size_var)))
    #     print("p1 at %.4f" % p1)
    #     print("p2 at %.4f" % p2)
    #     free_space_minus_sigma = navigable_mean - 1.690144*navigable_std
    #     print("Free space minus sigma lower bound: %.4f" % (free_space_minus_sigma))
    #     print()

    return safety_probability


@numba.njit()
def _check_collision_with_estimate(robot_position: ArrayLike, robot_width: float,
                                   obs_pos_mean: ArrayLike, obs_pos_cov: ArrayLike,
                                   obs_size_mean: float, obs_size_var: float,
                                   n_sigma: float):
    """ Check if the robot is in collision with this obstacle, within n-sigma uncertainty. """
    # Find angle from the robot position, to the obstacle center
    angle = math.atan2(obs_pos_mean[1,0] - robot_position[1],
                       obs_pos_mean[0,0] - robot_position[0])

    c = math.cos(angle)
    s = math.sin(angle)

    # Rotation matrix from original 2D coordinates to the coordinate frame
    # whose x-axis points from position mean of obstacle 1 to mean of obstacle 2
    R = np.array([[c, s], [-s, c]])

    # Rotate the robot position
    robot_pos_r = R.dot(robot_position)

    # Rotate the obstacle mean and covariance
    pos_mean_r = R.dot(obs_pos_mean)
    pos_cov_r = R.dot(obs_pos_cov).dot(R.T)

    # Get the position variance along the rotated x-axis (line from robot to the obs estimate)
    pos_var_r = pos_cov_r[0,0]

    # Check if the robot is further from the obstacle than the bloat distance
    robot_distance = pos_mean_r[0,0] - robot_pos_r[0]
    safe_distance = (obs_size_mean/2. + robot_width/2. +
                     n_sigma * math.sqrt(obs_size_var/4. + pos_var_r))
    return robot_distance <= safe_distance

