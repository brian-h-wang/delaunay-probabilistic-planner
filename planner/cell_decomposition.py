from typing import List, Tuple, Dict, Optional
import math
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from graph_tool import Graph, VertexPropertyMap, EdgePropertyMap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from planner.slam import ObstacleSLAM
from planner.probabilistic_obstacles import Obstacle2D, ProjectedObstaclesPair, SafetyProbability
from planner.utils import check_line_intersect
from planner.time_counter import TimeCounter, StubTimeCounter

# The int value assigned to longer-range zones must be higher than the int values
#  for shorter-range zones.
RANGE_SHORT = 0
RANGE_MEDIUM = 1
RANGE_LONG = 2

EXCLUDE_LONG_RANGE_VERTICES = True


class SafetyCellDecomposition(object):
    """
    A map storing the uncertain positions and sizes of 2D obstacles.
    Used for creating probabilistic graphs and then planning paths.
    """

    def __init__(
        self,
        slam: ObstacleSLAM,
        robot_width: float,
        safety_probability: SafetyProbability,
        range_short=5,
        range_medium=15,
        boundary=None,
        time_counter: Optional[TimeCounter] = None,
    ):
        """

        Parameters
        ----------
        obstacles
        robot_width
        n_measurements_converged: int
            The minimum number of measurements for an obstacle estimate to be treated as converged.
            A converged obstacle has received enough measurements that its estimate is treated
            as fairly certain and unlikely to change significantly.
            Converged and non-converged obstacles are treated differently by the high-level
            path planner.
        range_short: float
        range_medium: float
            Maximum ranges considered as short and medium-range.
        """
        if time_counter is None:
            time_counter = StubTimeCounter()
        self.time_counter = time_counter

        obstacles: List[Obstacle2D] = slam.get_landmarks_estimate()

        # Add the boundary obstacles - known with perfect certainty
        if boundary is not None:
            boundary_obstacles = boundary.get_obstacles()
        else:
            boundary_obstacles = []

        # Create graph where vertices are Delaunay triangulation vertices (i.e. obstacle centers),
        #   with edges between vertices that are in shared triangulation cells.
        # Edges have safety probabilities stored
        graph = Graph(directed=False)
        # edge_cost = graph.new_edge_property("double")
        # n_cells = delaunay.simplices.shape[0]
        vertex_position = graph.new_vertex_property("vector<double>")
        vertex_range = graph.new_vertex_property("int")
        vertex_is_boundary = graph.new_vertex_property("bool")
        vertex_n_measurements = graph.new_vertex_property("int")
        # vertex_converged = graph.new_vertex_property("bool")
        edge_safety = graph.new_edge_property("double", val=0.0)

        obstacles_to_include = []

        for obs_idx, obstacle in enumerate(obstacles):
            distance = slam.get_landmark_closest_distance(obs_idx)
            if distance < range_short:
                range_zone = RANGE_SHORT
            elif distance < range_medium:
                range_zone = RANGE_MEDIUM
            else:
                range_zone = RANGE_LONG
            if EXCLUDE_LONG_RANGE_VERTICES:
                if range_zone != RANGE_LONG:
                    obstacles_to_include.append(obstacle)
                else:
                    continue
            else:
                obstacles_to_include.append(obstacle)
            v = graph.add_vertex()
            vertex_position[v] = obstacle.pos_mean
            vertex_n_measurements[v] = obstacle.n_measurements
            vertex_range[v] = range_zone
            # vertex_converged[v] = obstacle.n_measurements >= n_measurements_converged
            vertex_is_boundary[v] = False

        obstacles = obstacles_to_include

        for obs_idx, obstacle in enumerate(boundary_obstacles):
            v = graph.add_vertex()
            vertex_position[v] = obstacle.pos_mean
            vertex_n_measurements[v] = 0
            # Boundary obstacles treated as known -> treat as short-range, confident obstacles
            vertex_range[v] = RANGE_SHORT
            vertex_is_boundary[v] = True

        ### Create the Delaunay triangulation

        # Combine obstacles and boundary obstacles
        obstacles += boundary_obstacles

        obstacle_positions = np.row_stack([obs.pos_mean.reshape((1, 2)) for obs in obstacles])
        with self.time_counter.count("delaunay"):
            delaunay = Delaunay(obstacle_positions)

        # Get list of (int, int) tuples, indicating connections between the obstacles
        # in the Delaunay triangulation
        delaunay_edges = get_delaunay_edges(delaunay)

        self.navigation_points_dict: Dict[Tuple[int, int], ArrayLike] = {}
        for v1, v2 in delaunay_edges:
            # if EXCLUDE_LONG_RANGE_VERTICES:
            #     if vertex_range[v1] == RANGE_LONG or vertex_range[v2] == RANGE_LONG:
            #         continue
            edge = graph.add_edge(v1, v2)
            # Calculate the probability of safe navigation
            projected_obstacles = ProjectedObstaclesPair(obstacles[v1], obstacles[v2])

            prob_safe = projected_obstacles.get_prob_safe_navigation(robot_width)
            # Save the safety weight of the Delaunay cell edges
            edge_safety[edge] = prob_safe
            nav_points = projected_obstacles.get_navigation_points(
                robot_width=robot_width, safety_probability=safety_probability
            )
            if nav_points.shape[0] == 1 and (vertex_range[v1] == RANGE_SHORT and vertex_range[v2] == RANGE_SHORT):
                nav_points = np.empty((0, 2))
            self.navigation_points_dict[(v1, v2)] = nav_points
            self.navigation_points_dict[(v2, v1)] = nav_points

        self.obstacles = obstacles
        self.obstacle_positions = obstacle_positions
        self.robot_width = robot_width
        self.delaunay = delaunay
        self.delaunay_edges = delaunay_edges

        graph.vp["position"] = vertex_position
        graph.vp["n_measurements"] = vertex_n_measurements
        graph.vp["range_zone"] = vertex_range
        # graph.vp["converged"] = vertex_converged
        graph.ep["safety"] = edge_safety
        self.graph = graph

        self.boundary_cells = None
        self.boundary_edges = None

        self.boundary = boundary
        self.safety_probability = safety_probability

    @property
    def n_cells(self):
        """
        Returns the integer number of Delaunay triangulation cells in the decomposition
        """
        return self.delaunay.simplices.shape[0]

    def get_vertex_positions(self, vertex_idx):
        """
        Returns the 2D position(s) of one or more vertices (obstacles) in the cell decomposition.

        Parameters
        ----------
        vertex_idx: int or array_like
            Index or indices of vertices.

        Returns
        -------
        ndarray
            2D position(s) of queried vertex or vertices.
            2-element array if vertex_idx is an int, and N by 2 array if vertex_idx is array-like.

        """
        return self.obstacle_positions[vertex_idx, :]

    def get_face_vertex_positions(self, v1: int, v2: int, min_safety=0.0):
        """
        Given a pair of vertices (obstacles) in the cell decomposition,
        return the positions of the (possibly multiple) face vertices for the navigation graph,
        which lie between this pair of vertices.

        If the edge between the obstacles is very safe for the robot to pass through,
        there will be multiple possible vertices on this edge - one near each of the obstacles,
        and one more at the midpoint.

        If the edge is unsafe, only the edge geometric midpoint will be returned.

        Parameters
        ----------
        v1: int
        v2: int
            The indices of two obstacles in the cell decomposition.

        Returns
        -------
        array_like
            An array of shape (1, 2) or (3,2),
            containing the (x,y) coordinates of the face vertices between this pair of obstacles.

        """
        return self.navigation_points_dict[(v1, v2)]

    def get_edge_safety(self, v1: int, v2: int):
        """

        Parameters
        ----------
        v1: int
        v2: int

        Returns
        -------
        float
            The probability that the robot can safely traverse this edge.

        """
        edge_safety = self.graph.ep["safety"]
        return edge_safety[self.graph.edge(v1, v2)]

    def get_edge_n_measurements(self, v1: int, v2: int):
        """

        Parameters
        ----------
        v1: int
        v2: int
            Indices of two obstacles (vertices in the Delaunay graph)

        Returns
        -------
        int
            The number of detections this edge has received.
            This is the lesser of the number of detections vertex v1 and v2 have received.

        """
        vertex_n_measurements = self.graph.vp["n_measurements"]
        return max(vertex_n_measurements[v1], vertex_n_measurements[v2])

    def find_delaunay_cell(self, points):
        """
        Given an array of 2D (x, y) points, return the Delaunay cell that each point falls within.

        Parameters
        ----------
        points: ndarray
            N by 2 array of points, as [x, y] coordinates

        Returns
        -------
        cell_labels: ndarray or int
            N-element array.
            cell_labels[i] is the index of the Delauanay triangulation cell that point i falls within.
            This is -1 if point i does not lie within any Delaunay cell.
            cell_labels is an N-element ndarray if N > 1, and is an int if N == 1 (i.e. only one input position given)

        """
        points = np.array(points)
        result = self.delaunay.find_simplex(points)
        if result.size == 1:
            # If only one position input was given, return a scalar
            return result.item()
        else:
            return result

    def get_cell_vertices(self, cell_index):
        """

        Parameters
        ----------
        cell_index: int
            Index of a cell in the decomposition, whose vertices will be returned.

        Returns
        -------
        ndarray
            3 by 2 ndarray, containing the (x, y) coordinates of the three vertices of the cell.
        """
        obstacle_indices = self.delaunay.simplices[cell_index, :]
        return self.obstacle_positions[obstacle_indices, :]

    def get_cell_neighbors(self):
        """
        Return a list of tuples, each tuple containing the two indices of a pair of cells that
        neighbor one another in the cell decomposition,
        as well as containing a 2 by 2 ndarray of the vertex positions for the endpoints of the
        edge between the cells.

        Returns
        -------
        neighbor_cells: List[Tuple[int, int]]
            Contains pairs of indices of cells that neighbor each other in the cell decomposition.
        neighbor_edge_vertices: List[Tuple[int, int]]
            List of the indices of the two vertices (obstacles) that make up the edge between
            each pair of neighboring cells.

        """
        # Precompute cell neighbors, and vertex positions for each edge
        neighbor_cells: List[Tuple[int, int]] = []
        neighbor_edge_vertices: List[Tuple[int, int]] = []

        for c1 in range(self.n_cells):
            for c2, vertex_indices in zip(self.delaunay.neighbors[c1, :], [(1, 2), (0, 2), (0, 1)]):
                # -1 indicates the cell is missing a neighbor (at edge of decomposition); skip these
                if c2 == -1:
                    continue
                # Only add neighbor if c1 < c2, to avoid adding duplicates
                if c1 > c2:
                    continue
                vertices = tuple(self.delaunay.simplices[c1, vertex_indices])
                neighbor_cells.append((c1, c2))
                neighbor_edge_vertices.append(vertices)
        return neighbor_cells, neighbor_edge_vertices

    def get_vertex_range(self, vertex_idx: int):
        vertex_range = self.graph.vp["range_zone"]
        return vertex_range[vertex_idx]

    def _compute_boundary(self):
        """Compute and store the cells and cell edges at the boundary of the decomposition."""
        # Get all edges at the boundary of the cell decomposition,
        # using the cells that have no neighbor at one of their edges
        boundary_cells: List[int] = []
        boundary_edges: List[ArrayLike] = []
        for cell_idx in range(self.n_cells):
            neighbors = self.delaunay.neighbors[cell_idx, :]
            # -1 indicates no neighbor on that edge
            for neighbor_idx, vertex_indices in zip([0, 1, 2], [(1, 2), (0, 2), (0, 1)]):
                if neighbors[neighbor_idx] == -1:
                    boundary_cells.append(cell_idx)
                    boundary_edges.append(tuple(self.delaunay.simplices[cell_idx, vertex_indices]))
        self.boundary_cells = boundary_cells
        self.boundary_edges = boundary_edges

    def get_visible_cells(self, position):
        """
        Find which cell decomposition cells are visible from a given query position.

        Parameters
        ----------
        position: array_like
            2D query position, from which to check visibility of all cells in the decomposition.
            This position should be outside of the cell decomposition.

        Returns
        -------
        visible_cells: list[int]
            Indices of cells which are visible from the query position
            Note it is possible for a single cell to appear twice in this list,
            with two different visible edges.
        visible_cell_edges: list[tuple[int, int]]
            Element i of this list gives the indices of the two vertices, which form the edge
            of cell i which is visible from the query position.

        """
        if self.find_delaunay_cell(position) != -1:
            raise Warning("query position is inside cell decomposition; get_visible_cells behavior is undefined.")

        if self.boundary_cells is None or self.boundary_edges is None:
            self._compute_boundary()
        boundary_cells = self.boundary_cells
        boundary_edges = self.boundary_edges

        visible_cells = []
        visible_cell_edges = []
        # For each boundary edge, check if the line from the query position to the edge midpoint
        #   intersects any other boundary edges.
        # If there are no other edge intersections found, this edge (and cell) are visible.
        for i in range(len(boundary_cells)):
            visible = True
            midpoint = np.mean(self.obstacle_positions[boundary_edges[i], :], axis=0)
            for j in range(len(boundary_cells)):
                if i == j:
                    continue
                edge_points = self.obstacle_positions[boundary_edges[j], :]
                if check_line_intersect(position, midpoint, edge_points[0, :], edge_points[1, :]):
                    visible = False
                    break
            if visible:
                visible_cells.append(boundary_cells[i])
                visible_cell_edges.append(boundary_edges[i])
        return visible_cells, visible_cell_edges

    def check_point_visible(self, position1, position2):
        """
        Check

        Parameters
        ----------
        position: array_like
            2D query position, from which to check visibility of all cells in the decomposition.
            This position should be outside of the cell decomposition.

        Returns
        -------
        bool
            True if the points are visible to one another, False if not

        """
        if self.find_delaunay_cell(position1) != -1 or self.find_delaunay_cell(position2) != -1:
            raise Warning("query position is inside cell decomposition; check visibility behavior is undefined.")

        if self.boundary_cells is None or self.boundary_edges is None:
            self._compute_boundary()
        boundary_edges = self.boundary_edges

        visible_cells = []
        visible_cell_edges = []
        # For each boundary edge, check if the line from the query position to the edge midpoint
        #   intersects any other boundary edges.
        # If there are no other edge intersections found, this edge (and cell) are visible.
        visible = True
        for i in range(len(boundary_edges)):
            boundary_edge = boundary_edges[i]
            edge_points = self.obstacle_positions[boundary_edge, :]
            if check_line_intersect(position1, position2, edge_points[0, :], edge_points[1, :]):
                visible = False
                break
        return visible


def get_delaunay_edges(delaunay: Delaunay):
    """

    Parameters
    ----------
    delaunay: Delaunay
        A Delaunay triangulation

    Returns
    -------
    List[Tuple[int, int]]
        List of (int, int) tuples. Each tuple gives the indices of two vertices in the
        Delaunay triangulation, that have an edge between them.
        In each tuple, the first index is less than the second.

    """
    indptr, indices = delaunay.vertex_neighbor_vertices
    edges = []
    for vertex_idx in range(len(indptr) - 1):
        neighbors = indices[indptr[vertex_idx] : indptr[vertex_idx + 1]]
        for neighbor in neighbors:
            next_edge = (vertex_idx, neighbor)
            if next_edge[0] < next_edge[1]:
                edges.append(next_edge)
    return edges
