"""
Module for Poisson forest generation.
"""

import math
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from planner.utils import check_collision
from planner.simulation import SimulationWorld

class Tree(object):

    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def __str__(self):
        return "Tree with position [%.2f, %.2f] and radius %.2f" %\
               (self.position[0], self.position[1], self.radius)

    def __repr__(self):
        return "Tree(position=%s, radius=%f)" % (str(self.position), self.radius)


class TreeCluster(object):

    def __init__(self, n_trees: int, mean=None, cov=None):
        assert n_trees > 0
        self.n_trees = n_trees
        if mean is None:
            mean = np.array([0, 0])
        self.mean = mean
        if cov is None:
            cov = np.eye(2)
        self.cov = cov
        self.rng = None

    def set_rng(self, rng):
        self.rng = rng

    def sample_position(self):
        if self.rng is None:
            self.rng = np.random.default_rng()
        return self.rng.multivariate_normal(self.mean, self.cov)

class PoissonTreeCluster(TreeCluster):

    def __init__(self, density: float, mean, cov, seed=None):
        a = math.sqrt(cov[0,0])
        b = math.sqrt(cov[1,1])
        cluster_area = math.pi*4*a*b
        exp_trees = density * cluster_area
        self.rng = np.random.default_rng(seed)
        n_trees = self.rng.poisson(exp_trees)
        # print("Tree cluster with seed %d, %d trees" % (seed, n_trees))
        super().__init__(n_trees, mean, cov)


class PoissonForest(SimulationWorld):

    def __init__(self, density: float, bounds: List[float], seed=None,
                 invalid_spaces=None, clusters=None, barriers=True):
        """
        Generate a Poisson forest with an expected density of trees,
        according to a homogeneous Poisson process.

        Samples the number of trees in the forest from a Poisson distribution,
        then generate tree positions within the forest using a uniform distribution.

        Parameters
        ----------
        density: float
            Density of trees, specified as number of trees per square meter
        bounds: list of float
            Bounds of the forest, specified as [x_min, x_max, y_min, y_max].
        invalid_spaces: ndarray
            N by 3 ndarray, each row is [x, y, radius]
            Defines circular areas where no trees should be sampled.
            Can be used to avoid putting trees e.g. at start and goal position for navigation.
        """
        assert len(bounds) == 4, "'bounds' argument must have length 4, [xmin xmax ymin ymax]"

        # Based on the density and area, determine expected number of trees in the forest,
        # based on the Poisson distribution
        x_min, x_max, y_min, y_max = bounds
        assert (x_min < x_max) and (y_min < y_max),\
            "'bounds' must be specified as [xmin xmax ymin ymax], with xmin<xmax and ymin<ymax"
        area = (x_max - x_min) * (y_max - y_min)
        exp_trees = density * area

        # Sample Poisson distribution
        self.rng = np.random.default_rng(seed)
        n_trees = self.rng.poisson(exp_trees)


        if clusters is None:
            clusters = []
        else:
            clusters = copy.deepcopy(clusters)

        # for cluster in clusters:
        #     n_trees += cluster.n_trees

        # Sample positions for the trees
        positions = np.empty((n_trees, 2))
        radii = np.empty((n_trees, 1))

        current_cluster: Optional[TreeCluster] = None if len(clusters) == 0 else clusters.pop(0)
        if current_cluster is not None:
            current_cluster.set_rng(self.rng)
        n_trees_in_cluster = 0

        n_trees_sampled = 0
        # Sample trees that don't overlap, using rejection sampling
        while n_trees_sampled < n_trees:
            if current_cluster is not None:
                # Filled up current cluster, move to the next one
                if n_trees_in_cluster == current_cluster.n_trees:
                    if len(clusters) > 0:
                        current_cluster = clusters.pop(0)
                        current_cluster.set_rng(self.rng)
                    else:
                        current_cluster = None
                    n_trees_in_cluster = 0
            # Sample tree from a cluster if provided
            if current_cluster is not None:
                x, y = current_cluster.sample_position()
            else:
                x = self.rng.uniform(x_min, x_max)
                y = self.rng.uniform(y_min, y_max)
            r = self._sample_radius()
            if r < 0:
                print("Warning: Sampled tree radius less than zero")
            # Check if this tree is invalid according to provided checking function, if any
            if invalid_spaces is not None:
                if check_collision(np.array([x, y]), r * 2, invalid_spaces[:, 0:2], invalid_spaces[:, 2]):
                    continue
            # Check if new tree overlaps with any existing trees, discard it if so
            if n_trees_sampled > 0:
                d = np.linalg.norm(positions[0:n_trees_sampled,:] - [x, y], axis=1)
                # for j in range(0, i-1):
                #     if d[j] < (self.radii[j] + r):
                if np.any(d < (radii[0:n_trees_sampled] + r)):
                    continue
            # New tree does not overlap with existing trees; add it to the forest
            positions[n_trees_sampled,0] = x
            positions[n_trees_sampled,1] = y
            radii[n_trees_sampled, 0] = r
            n_trees_sampled += 1
            n_trees_in_cluster += 1

        obstacles = np.column_stack([positions, radii*2])

        super().__init__(xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]),
                         obstacles=obstacles)


    def __len__(self):
        return self.obstacles.shape[0]

    @property
    def n_trees(self):  # included for compatibility
        return len(self)

    def __getitem__(self, item):
        return Tree(position=self.positions[item,:], radius=self.radii[item])

    def concatenate_forest(self, other: SimulationWorld):
        """
        Append the obstacles in another SimulationWorld object to this forest.

        Parameters
        ----------
        other: PoissonForest

        Returns
        -------
        None

        """
        xmin = min(self.xlim[0], other.xlim[0])
        xmax = max(self.xlim[1], other.xlim[1])
        ymin = min(self.ylim[0], other.ylim[0])
        ymax = max(self.ylim[1], other.ylim[1])

        self.xlim = (xmin, xmax)
        self.ylim = (ymin, ymax)
        self.obstacles = np.concatenate([self.obstacles, other.obstacles], axis=0)

    def append_tree(self, x: float, y: float, diameter: float):
        new = np.array([x, y, diameter]).reshape((1,3))
        self.obstacles = np.concatenate([self.obstacles, new], axis=0)

    @property
    def x(self):
        return self.obstacles[:,0]

    @property
    def y(self):
        return self.obstacles[:,1]

    @property
    def r(self):
        return self.obstacles[:,2]/2.

    def plot(self, ax: plt.Axes = None, **kwargs):
        positions = self.obstacles[:,0:2]
        diameters = self.obstacles[:,2]
        if ax is None:
            ax = plt.subplot(1,1,1)
            ax.set_xlim([self.xlim[0], self.xlim[1]])
            ax.set_ylim([self.ylim[0], self.ylim[1]])
            ax.set_aspect('equal')
        for i in range(self.n_trees):
            p = patches.Circle(xy=positions[i,:], radius=diameters[i]/2., **kwargs)
            ax.add_patch(p)

    def _sample_radius(self):
        """
        Function for sampling a tree radius.
        Should be overriden by child classes.

        Returns
        -------
        float
            Radius of a newly sampled tree

        """
        return 0.

class PoissonForestFixedRadius(PoissonForest):
    def __init__(self, density: float, bounds: List[float],
                 radius: float, seed=None, invalid_spaces=None, clusters=None):
        assert radius > 0, "radius must be greater than zero"
        self.constant_radius = radius
        super().__init__(density=density, bounds=bounds, seed=seed, invalid_spaces=invalid_spaces, clusters=clusters)

    def _sample_radius(self):
        return self.constant_radius


class PoissonForestUniformRadius(PoissonForest):

    def __init__(self, density: float, bounds: List[float],
                 radius_min: float, radius_max: float, seed=None, invalid_spaces=None,
                 clusters=None):
        assert radius_min > 0, "minimum radius must be greater than zero"
        assert radius_min <= radius_max, "radius min must be less than or equal to radius max"
        self.radius_min = radius_min
        self.radius_max = radius_max
        super().__init__(density=density, bounds=bounds, seed=seed, invalid_spaces=invalid_spaces,
                         clusters=clusters)

    def _sample_radius(self):
        return self.rng.uniform(low=self.radius_min, high=self.radius_max)


class PoissonForestGaussianRadius(PoissonForest):

    def __init__(self, density: float, bounds: List[float],
                 radius_mean: float, radius_std: float, seed=None,
                 invalid_spaces=None, clusters=None):
        assert radius_mean > 0, "Mean for tree radius Gaussian distribution should be positive"
        self.radius_mean = radius_mean
        self.radius_std = radius_std
        super().__init__(density=density, bounds=bounds, seed=seed, invalid_spaces=invalid_spaces, clusters=clusters)

    def _sample_radius(self):
        return self.rng.normal(loc=self.radius_mean, scale=self.radius_std)


class ForestBarriers(SimulationWorld):

    def __init__(self, bounds: List[float], robot_width, seed=None):
        """
        Generate four "walls" of trees around a bounded space.
        Use these barriers to prevent the robot moving outside of the workspace bounds.

        Parameters
        ----------
        bounds: list of float
            Bounds of the forest, specified as [x_min, x_max, y_min, y_max].
        robot_width: float
            Width of the robot.
            The trees will be generated such that the space between them is half the robot width,
            so that the robot is blocked from moving through.
        """
        assert len(bounds) == 4, "'bounds' argument must have length 4, [xmin xmax ymin ymax]"
        rng = np.random.default_rng(seed)

        # Based on the density and area, determine expected number of trees in the forest,
        # based on the Poisson distribution
        x_min, x_max, y_min, y_max = bounds
        assert (x_min < x_max) and (y_min < y_max), \
            "'bounds' must be specified as [xmin xmax ymin ymax], with xmin<xmax and ymin<ymax"

        # Generate trees in a zig-zag pattern, so that the Delaunay triangulation still works.
        tree_radius = robot_width * 2
        tree_spacing = (2 * tree_radius) + 0.25 * robot_width  # space between successive tree centers

        # Amount of random variation to apply in the x-direction (for the left and right walls)
        #   or the y-direction (for the top and bottom walls)
        variation = 0.05

        obstacles = np.empty((0,3))
        # Generate the top and bottom barriers
        for y_coord in [y_min-tree_radius, y_max+tree_radius]:
            # Calculate how many trees should be in this barrier
            n_trees_in_barrier = 1 + math.ceil((x_max - x_min) / tree_spacing)
            # Generate tree center positions along this line
            tree_x = np.linspace(x_min-tree_radius, x_max+tree_radius, n_trees_in_barrier)
            tree_y = rng.uniform(-variation, variation, n_trees_in_barrier) + y_coord

            barrier = np.empty((n_trees_in_barrier, 3))
            barrier[:,0] = tree_x
            barrier[:,1] = tree_y
            barrier[:,2] = tree_radius*2
            obstacles = np.concatenate([obstacles, barrier], axis=0)
        # Generate the left and right barriers
        for x_coord in [x_min-tree_radius, x_max+tree_radius]:
            n_trees_in_barrier = (1 + math.ceil((y_max - y_min) / tree_spacing))
            tree_x = rng.uniform(-variation, variation, n_trees_in_barrier) + x_coord
            tree_y = np.linspace(y_min - tree_radius, y_max + tree_radius, n_trees_in_barrier)
            # Subtract two because the first and last trees can be removed
            #   (these will overlap with the top/bottom barriers)
            barrier = np.empty((n_trees_in_barrier-2, 3))
            barrier[:,0] = tree_x[1:-1]
            barrier[:,1] = tree_y[1:-1]
            barrier[:,2] = tree_radius*2
            obstacles = np.concatenate([obstacles, barrier], axis=0)

        super().__init__(xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]),
                         obstacles=obstacles)


def make_center_cluster_forest(seed=20212021):
    xmin, xmax = (-2, 12)
    ymin, ymax = (0, 10)
    start_pose = np.array([0., 5., 0.])
    goal_position = np.array([10, 5], dtype=float)

    robot_width = 0.5
    # check_invalid function is used to avoid sampling trees on top of the start and goal
    w = robot_width * 3. # no trees will generate within this distance of start and goal
    invalid_spaces = np.array([[start_pose[0], start_pose[1], w],
                               [goal_position[0], goal_position[1], w],
                               [start_pose[0] + 5, start_pose[1], w]])

    # cluster = TreeCluster(n_trees = 20, mean=[10, 5], cov=np.diag([2, 2]))

    cluster_means = [[5, 5]]
    clusters = [TreeCluster(n_trees = 20, mean=m, cov=np.diag([1, 1])) for m in cluster_means]
    return PoissonForestUniformRadius(bounds = [xmin, xmax, ymin, ymax], density=0.2,
                                      radius_min=0.1, radius_max=0.2, seed=seed,
                                      invalid_spaces=invalid_spaces,
                                      clusters=clusters)

def make_bugtrap_forest(seed=20212021):
    xmin, xmax = (-2, 22)
    ymin, ymax = (0, 10)
    start_pose = np.array([0., 5., 0.])
    goal_position = np.array([20, 5], dtype=float)

    robot_width = 0.5
    # check_invalid function is used to avoid sampling trees on top of the start and goal
    w = robot_width * 3. # no trees will generate within this distance of start and goal
    invalid_spaces = np.array([[start_pose[0], start_pose[1], w],
                               [goal_position[0], goal_position[1], w]])

    cluster_north = TreeCluster(n_trees = 12, mean=[10, 7], cov=np.diag([4, 0.05]))
    cluster_east = TreeCluster(n_trees = 12, mean=[13, 4], cov=np.diag([0.05, 2]))
    cluster_south = TreeCluster(n_trees = 12, mean=[10, 3], cov=np.diag([4, 0.05]))
    return PoissonForestUniformRadius(bounds = [xmin, xmax, ymin, ymax], density=0.15,
                                      radius_min=0.2, radius_max=0.25, seed=seed,
                                      invalid_spaces=invalid_spaces,
                                      clusters=[cluster_north, cluster_east, cluster_south])

class SavedForest(object):
    """ Class for saving a forest to a file with metadata. """

    def __init__(self, forest, seed: int,
                 shortest_path: ArrayLike, shortest_path_length: float, straight_line_distance: float):
        self.forest = forest
        self.seed = seed
        self.shortest_path = shortest_path
        self.shortest_path_length = shortest_path_length
        self.straight_line_distance = straight_line_distance
