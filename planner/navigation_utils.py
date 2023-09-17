"""
Utilities for robot navigation.
"""

from typing import List, Tuple, Optional
import math
import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import Point, LineString
from planner.hybrid_astar import ValidEdgeEvaluator, CircularObstacleEdgeEvaluator
from planner.cell_decomposition import SafetyCellDecomposition, RANGE_MEDIUM
from planner.probabilistic_obstacles import Obstacle2D, SafetyProbability

class NavigationBoundary(object):
    """
    Class used to bound the workspace by adding extra "boundary obstacles".
    These are circular obstacles added in a rectangle around the workspace.

    Use this class to add boundary obstacles without the extra burden of adding them to SLAM
    (where they will add to data association time, and be treated as uncertain
    which may give weird results for planning).
    """

    def __init__(self, bounds: List[float], detection_range: Optional[float],
                 obstacle_diameter=0.25, seed: int = 20212021):
        """

        Parameters
        ----------
        bounds: List[float]
            Limits of the environment, as [x_min, x_max, y_min, y_max]
        detection_range: float or None
            Range at which the robot can detect obstacles.
            If provided, boundary obstacles will be visible to the robot only when within
            the robot's detection range. Use NavigationBoundary.update() to update the list
            of which obstacles are active and visible to the robot.
            If None, all obstacles are active as soon as the boundary is initialized
        obstacle_diameter: float
            Diameter of the boundary obstacles, default 0.25 m
        seed: int
            Random seed for generating the boundary.
            Random seed is needed because a small amount of random variation is added to the
            boundary obstacle positions, so that the Delaunay triangulation works.
        """
        rng = np.random.default_rng(seed)

        x_min, x_max, y_min, y_max = bounds
        assert (x_min < x_max) and (y_min < y_max), \
            "'bounds' must be specified as [xmin xmax ymin ymax], with xmin<xmax and ymin<ymax"

        tree_radius = obstacle_diameter/2.
        tree_spacing = obstacle_diameter

        # Amount of random variation to apply in the x-direction (for the left and right walls)
        #   or the y-direction (for the top and bottom walls)
        variation_max = 0.01

        def make_obstacle(x, y):
            pos_cov = np.eye(2) * 1e-8
            size_var = 1e-8
            return Obstacle2D(pos_mean=[x,y], pos_cov=pos_cov,
                              size_mean=obstacle_diameter, size_var=size_var)

        # Calculate (x,y) coordinates for all obstacles in barriers
        obstacle_positions = np.empty((0,2))
        n_trees_in_top_bottom_barrier = 1 + math.ceil((x_max - x_min) / tree_spacing)
        n_trees_in_left_right_barrier = (1 + math.ceil((y_max - y_min) / tree_spacing))
        # Generate the top and bottom barriers
        for y_coord in [y_min-tree_radius, y_max+tree_radius]:
            # Calculate how many trees should be in this barrier
            n_trees_in_barrier = n_trees_in_top_bottom_barrier
            # Generate tree center positions along this line
            tree_x = np.linspace(x_min-tree_radius, x_max+tree_radius, n_trees_in_barrier)
            variation = rng.uniform(0, variation_max, n_trees_in_barrier)
            if y_coord == y_min - tree_radius:
                variation *= -1
            tree_y = variation + y_coord

            barrier = np.empty((n_trees_in_barrier, 2))
            barrier[:,0] = tree_x
            barrier[:,1] = tree_y
            obstacle_positions = np.concatenate([obstacle_positions, barrier], axis=0)
        # Generate the left and right barriers
        for x_coord in [x_min-tree_radius, x_max+tree_radius]:
            n_trees_in_barrier = n_trees_in_left_right_barrier
            variation = rng.uniform(0, variation_max, n_trees_in_barrier)
            if x_coord == x_min-tree_radius:
                variation *= -1
            tree_x = variation + x_coord
            tree_y = np.linspace(y_min - tree_radius, y_max + tree_radius, n_trees_in_barrier)
            # Subtract two because the first and last trees can be removed
            #   (these will overlap with the top/bottom barriers)
            barrier = np.empty((n_trees_in_barrier-2, 2))
            barrier[:,0] = tree_x[1:-1]
            barrier[:,1] = tree_y[1:-1]
            # self.boundary_obstacles.append(make_obstacle(tree_x, tree_y))
            obstacle_positions = np.concatenate([obstacle_positions, barrier], axis=0)
        n_obstacles = obstacle_positions.shape[0]
        self.obstacle_positions = obstacle_positions

        # Record which obstacle is in which barrier
        # 0 = bottom, 1 = top, 2 = left, 3 = right
        self._which_barrier = np.empty(n_obstacles, dtype=int)
        obs_idx = 0
        for b_idx, n in enumerate([n_trees_in_top_bottom_barrier, n_trees_in_top_bottom_barrier,
                                   n_trees_in_left_right_barrier, n_trees_in_left_right_barrier]):
            self._which_barrier[obs_idx:obs_idx + n] = b_idx
            obs_idx = obs_idx+n

        self.active = {i: False for i in range(n_obstacles)}

        self.detection_range = detection_range
        self.obstacle_diameter = obstacle_diameter
        self.variation = variation_max

        # Save Obstacle2D objects for the active obstacles
        self.obstacles: List[Obstacle2D] = []

        # Activate all obstacles if no detection range set
        if detection_range is None:
            pos_cov = np.eye(2) * 1e-8
            size_var = 1e-8
            for i in range(n_obstacles):
                # Obstacle is not yet active, check if it should be activated
                obs_xy = self.obstacle_positions[i,:]
                self.active[i] = True
                new_obs = Obstacle2D(pos_mean=obs_xy, pos_cov=pos_cov,
                                     size_mean=self.obstacle_diameter, size_var=size_var)
                self.obstacles.append(new_obs)

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


    def update(self, robot_pose):
        assert self.detection_range is not None, \
            "update() should be called only on a NavigationBoundary with detection_range != None"
        robot_position = robot_pose[0:2]
        n_obstacles = self.obstacle_positions.shape[0]
        pos_cov = np.eye(2) * 1e-8
        size_var = 1e-8
        for i in range(n_obstacles):
            if not self.active[i]:
                # Obstacle is not yet active, check if it should be activated
                obs_xy = self.obstacle_positions[i,:]
                if np.linalg.norm(robot_position - obs_xy) < self.detection_range:
                    self.active[i] = True
                    new_obs = Obstacle2D(pos_mean=obs_xy, pos_cov=pos_cov,
                                      size_mean=self.obstacle_diameter, size_var=size_var)
                    self.obstacles.append(new_obs)

    def get_obstacles(self) -> List[Obstacle2D]:
        return self.obstacles

    def get_obstacles_array(self):
        # Return an N by 3 numpy array of the boundary obstacles, with rows [x, y, diameter]
        obs_array = np.empty((len(self.obstacles), 3))
        for i, obs in enumerate(self.obstacles):
            obs_array[i, 0:2] = obs.pos_mean.flatten()
            obs_array[i, 2] = obs.size_mean
        return obs_array


    def xy_in_bounds(self, xy):
        """
        Check if an xy point is in bounds.

        Parameters
        ----------
        xy: array_like
            (x,y) coordinates to check

        Returns
        -------
        bool
            True if point is in bounds, False if not

        """
        x = xy[0]
        y = xy[1]
        return self.x_min < x < self.x_max and self.y_min < y < self.y_max


class NavigationPath(object):
    """
    Class representing a high-level path through uncertain obstacles.
    """

    # TODO I think this could use just the graph vertices as an input; then get the cells & edges in the constructor

    def __init__(self, cells: List[int], edges: List[Tuple[int, int]], graph_vertices: List[int],
                 cell_decomposition: SafetyCellDecomposition,
                 check_valid=True):
        """
        Assumes the cells and edges inputs are valid; i.e. the edges specified form a valid path
        through the cells.

        Parameters
        ----------
        cells: list[int]
            List of length N, indicating the Delaunay triangulation cells this path passes through.
            If the start position is not in any Delaunay triangulation cell, cells[0] should be -1.
            If the goal position is not in the triangulation, cells[-1] should be -1.
            Otherwise, all elements of cells are integers >= 0.
        edges: list[tuple[int, int]]
            List of length N-1, indicating the Delaunay triangulation edges the path passes through.
            Indices to vertices in the cell decomposition.
        graph_vertices: list[int]
            List of navigation graph vertices through which this graph passes.
        cell_decomposition: SafetyCellDecomposition
            The cell decomposition associated with this path
        check_valid: bool
            If True, constructor will check that the provided cells and edges are valid;
            i.e. the specified eges actually lie between the cells in the proper order.
            If False, only the length of the cells and edges inputs will be verified.
        """
        assert len(cells) == len(edges)+1, "Path has %d cells and %d edges; " \
                                           "should have one less edge than cells." % (len(cells), len(edges))

        # Determine which cells the start and goal positions are inside

        # Populate the list of edges
        delaunay = cell_decomposition.delaunay

        self._cells = cells
        self._edges = edges
        self.graph_vertices = graph_vertices

        # Check that the specified cells and edges are valid
        for i in range(len(cells)-1):
            c1 = cells[i]
            c2 = cells[i+1]
            e = edges[i]
            if c1 == c2 == -1:
                raise ValueError("cells list contains -1 twice in a row")
            # If c1 is -1, swap the cells. This check only works if c1 is not -1
            if c1 == -1:
                c2, c1 = c1, c2
            # Check that the two vertices in the edge lie on cell c1
            c1_vertices = list(delaunay.simplices[c1,:])

            e1_idx = c1_vertices.index(e[0])
            e2_idx = c1_vertices.index(e[1])
            # Edge vertex indices are two of 0, 1, or 2.
            # The neighbor cell on the other side of this edge is opposite the third vertex
            neighbor_idx = 3 - e1_idx - e2_idx
            # Check that the cells and edges provided are valid w.r.t. the cell decomposition
            if c2 != delaunay.neighbors[c1, neighbor_idx]:
                import matplotlib.pyplot as plt
                from planner.plotting import CellDecompositionPlotter
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                p = CellDecompositionPlotter(cell_decomposition, show_labels=True, ax=ax)
                plt.show()
                raise ValueError("Edge with vertices (%d, %d) is not shared by cells %d and %d" %
                                 (e[0], e[1], c1, c2))

        # Create a dict storing which cell comes after any given cell in the cell order
        self.next_cell_dict = {}
        for cell_idx, cell in enumerate(cells):
            if cell == -1:
                continue
            if cell_idx + 1 < len(cells):
                self.next_cell_dict[cell] = cells[cell_idx + 1]
            else:
                self.next_cell_dict[cell] = None

        self._cells = cells
        self._edges = edges
        self.cell_decomposition = cell_decomposition

        self._edge_lines = []
        for e1, e2 in self.edges:
            p1 = self.cell_decomposition.delaunay.points[e1, :]
            p2 = self.cell_decomposition.delaunay.points[e2, :]
            self._edge_lines.append(LineString([p1, p2]))
        self._edge_to_goal_dict = {}

    @property
    def cells(self):
        return self._cells

    @property
    def edges(self):
        return self._edges

    def __str__(self):
        cells_str = "Cells: %s " % str(self.cells)
        edges_str = "Edges: %s " % str(self.edges)
        return "Navigation path through: " + cells_str + edges_str

    def __eq__(self, other):
        return self.cells == other.cells

    def get_polygon(self):
        """
        Compute the 2D polygon enclosing the cells which this path consists of.

        Returns
        -------
        ndarray
            N by 2 array of the points forming the enclosing polygon

        """
        # Algorithm:
        # (note - in delaunay simplices, vertices are arranged in counterclockwise order)
        # for each cell:
        #   for each vertex, save the next vertex in counterclockwise order, *unless*
        #   the edge between the two vertices is one of the edges in the path
        #   then iterate over all the vertices, going counterclockwise around the polygon
        #   that encloses the path

        # No cells in the path, or path is outside the cell decomposition - no polygon to show
        if len(self.cells) == 0 or (len(self.cells) == 1 and self.cells[0] == -1):
            return np.zeros((0,2))

        next_ccw_vertex = {}
        polygon_first_vertex = None

        for cell in self.cells:
            if cell == -1:
                continue
            simplex = self.cell_decomposition.delaunay.simplices[cell,:]
            for i in (0,1,2):
                j = (i+1) % 3  # index of the second vertex
                v1, v2 = simplex[i], simplex[j]
                # Exclude this edge from the polygon if it is one of the edges the path goes through,
                # *unless* the edge is between a cell and the space outside the cell decomposition
                neighbor = self.cell_decomposition.delaunay.neighbors[cell, 3-(i+j)]
                if ((v1, v2) in self.edges or (v2, v1) in self.edges) and neighbor != -1:
                    # Neighbor is a cell and edge is part of the path; do not add to dict
                    continue
                else:
                    if polygon_first_vertex is None:
                        polygon_first_vertex = v1
                    next_ccw_vertex[v1] = v2

        n_cells = len(self.cells)
        n_polygon_vertices = 2 + n_cells
        polygon_vertices = np.empty((n_polygon_vertices, 2))
        current_vertex = polygon_first_vertex
        points = self.cell_decomposition.delaunay.points


        for i in range(n_polygon_vertices):
            polygon_vertices[i,:] = points[current_vertex, :]
            current_vertex = next_ccw_vertex[current_vertex]
        return polygon_vertices


class UncertainObstacleEdgeEvaluator(ValidEdgeEvaluator):
    """
    Edge evaluator for uncertain obstacles.

    Checks collisions against circular obstacles estimated by XY position and size,
    by bloating the covariance ellipse by N sigmas towards the robot position.

    This gives a more accurate (less conservative) collision check than bloating the obstacle by
    the worst case of X and Y position uncertainty.
    """

    def __init__(self, obstacles: List[Obstacle2D], robot_width: float,
                 safety_probability: SafetyProbability):
        self.obstacles = obstacles
        self.n_sigma= safety_probability.n_sigmas
        self.robot_width = robot_width
        super().__init__()

    def check_edge(self, position1, position2):
        # First check for obstacle collisions
        obstacle_collision = False
        robot_position = np.array(position2, dtype=float)
        for obs in self.obstacles:
            if obs.check_robot_collision(robot_position, robot_width=self.robot_width,
                                         n_sigma=self.n_sigma):
                obstacle_collision = True
                break
        if obstacle_collision:
            return False, 1.
        else:
            return True, 1.


class NavigationGraphEdgeEvaluator(CircularObstacleEdgeEvaluator):
    """
    Edge evaluator for navigation graphs.

    Checks collisions against the navigation graph obstacles (estimate means),
    and forces edges to follow a specified order of cells in the navigation graph.
    """

    def __init__(self, obstacles: ArrayLike, path: NavigationPath, robot_width: float):
        super().__init__(obstacles, robot_width=robot_width)
        self.path = path
        self.nav_graph = path.cell_decomposition

    def check_edge(self, position1, position2):
        # First check for obstacle collisions
        edge_no_collision, _ = super().check_edge(position1, position2)
        if not edge_no_collision:
            return False, 1.
        # If edge is collision-free, check that it's in the correct Delaunay cell
        c1, c2 = self.nav_graph.find_delaunay_cell(np.array([position1, position2]))
        # Disallow points outside Delaunay cells
        if c1 == -1 or c2 == -1:
            is_valid = False
            cost_scale = 1
        # Both points are in same cell; path is valid
        elif c1 == c2:
            is_valid = True
            cost_scale = 1.0
        # Points are in different cells; check if c2 should come after c1
        else:
            c1_next = self.path.next_cell_dict.setdefault(c1, None)
            is_valid = c1_next == c2
            cost_scale = 1.0
        return is_valid, cost_scale


