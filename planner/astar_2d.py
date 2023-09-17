import math
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Union

from graph_tool import Graph, Vertex, VertexPropertyMap, Edge, EdgePropertyMap
from graph_tool.search import astar_search, AStarVisitor, StopSearch

from planner.hybrid_astar import ValidEdgeEvaluator
from planner.navigation_utils import NavigationBoundary


COST_OCCUPIED_TO_OCCUPIED = 1e9

COST_BUFFER = 1e5

CELL_FREE = 0
CELL_OCCUPIED = 2
CELL_BUFFER = 1
CELL_BARRIER = CELL_OCCUPIED


class AStar2DResult(object):
    """
    Class representing a path from 2D A* search
    """

    def __init__(self, points):
        self.points = np.array(points).reshape((-1, 2))
        self._local_goal_point_index = None

    def get_local_goal(self, distance: float, return_index=False):
        """
        Compute the goal point that the local planner should attempt to reach.

        Arguments
        ---------
        distance: float
            The distance to look ahead of the start position, to get the local goal
        return_index: bool, default False
            If True, also return the index of the local going point within the global planner path

        Returns
        -------
        local_goal: array_like
            A point in 2D space.
        is_global_goal: bool
            If True, the local goal is the same as the robot's global goal
                (i.e. the robot is very close to the global goal)
            If False, the local goal is closer than the global goal
        goal_index: int
            Returned only if return_index arg is True
            The index of the local goal point within the global planner path.

        """
        # Find the first point along the path line segments which is a given distance away from the robot
        remaining_path_distance = distance
        vertex_positions = self.points
        n_vertices = self.points.shape[0]
        for v_idx in range(n_vertices-1):
            # Get the length of the next line segment in the path
            p1 = vertex_positions[v_idx,:]
            p2 = vertex_positions[v_idx+1,:]
            segment = p2 - p1
            segment_length = np.linalg.norm(segment)
            if segment_length > remaining_path_distance:
                goal = p1 + (segment/segment_length) * remaining_path_distance
                is_global_goal = False
                goal_index = v_idx
                break
            else:
                remaining_path_distance -= segment_length
        else:
            # Reached end of path; goal is last vertex in the path (will be the global goal)
            goal = vertex_positions[-1,:]
            is_global_goal = True
            goal_index = n_vertices-1
        if not return_index:
            return goal, is_global_goal
        else:
            return goal, is_global_goal, goal_index

    def get_points_after_local_goal(self):
        """ Returns the 2D points on the global goal, which come after the local goal. """
        if self._local_goal_point_index is None:
            raise ValueError("Must be called after get local goal")


class AStar2DVisitor(AStarVisitor):
    """
    Visitor class for A* search in a 2D occupancy grid.

    Adapted from Hamming hypercube A* example at:
    https://graph-tool.skewed.de/static/doc/search_module.html#graph_tool.search.astar_search
    """

    def __init__(self, graph: Graph, start_position: ArrayLike, goal_position: ArrayLike,
                 x_disc: VertexPropertyMap, y_disc: VertexPropertyMap,
                 xy_resolution: float, close_enough: float,
                 weight: EdgePropertyMap, dist: VertexPropertyMap, cost: VertexPropertyMap,
                 edge_evaluator: ValidEdgeEvaluator, boundary: NavigationBoundary,
                 max_n_vertices: int = -1,
                 timeout_n_vertices: int = -1):
        # Take in goal cell as occ grid coordinates
        # Do all computations in occ grid cell coordinates -
        # makes it easy to compute distances between cells
        self.graph = graph

        self.origin_position = start_position
        self.goal_position = goal_position

        self.edge_evaluator = edge_evaluator
        self.boundary = boundary

        # Discrete x and y coordinates
        # Integers, with [0,0] as the start position
        # Convert to discrete coordinates by multiplying integer coordinates by xy_resolution
        self.x_disc = x_disc
        self.y_disc = y_disc
        self.xy_resolution = xy_resolution
        self.close_enough = close_enough
        self.weight = weight
        self.dist = dist
        self.cost = cost
        self.visited = {}

        # Add the start cell as a vertex
        v0 = graph.add_vertex()
        self.x_disc[v0] = 0
        self.y_disc[v0] = 0
        self.start_vertex = v0

        self.n_vertices_finished = 0

        # search stops if this many vertices explored
        self.max_n_vertices = max_n_vertices

        # search stops if this many vertices explored without getting any closer to the goal
        self.timeout_n_vertices = timeout_n_vertices  #
        self._closest_distance_to_goal = float('inf')
        self._n_vertices_since_closer_to_goal = 0

        self.success = False
        self.goal_vertex = None


    def examine_vertex(self, u):
        # Iterate over the grid cells that neighbor the currently visited cell,
        # including diagonally
        current_cell = (self.x_disc[u], self.y_disc[u])

        x, y = self.discrete_to_continuous(current_cell)

        # for i in range(len(self.state[u])):
        for d_x in [-1, 0, 1]:
            for d_y in [-1, 0, 1]:
                if d_x == d_y == 0:
                    continue
                neighbor_cell = (current_cell[0] + d_x, current_cell[1] + d_y)

                # Check if this new cell is in an obstacle, or outside the workspace bounds

                neighbor_x, neighbor_y = self.discrete_to_continuous(neighbor_cell)

                if not self.boundary.xy_in_bounds((neighbor_x, neighbor_y)):
                    # out of bounds
                    continue
                edge_valid, _ = self.edge_evaluator.check_edge((x, y), (neighbor_x, neighbor_y))
                if not edge_valid:
                    # in collision
                    continue

                if neighbor_cell in self.visited:
                    v = self.visited[neighbor_cell]
                else:
                    v = self.graph.add_vertex()
                    self.visited[neighbor_cell] = v
                    self.x_disc[v] = neighbor_cell[0]
                    self.y_disc[v] = neighbor_cell[1]
                    self.dist[v] = self.cost[v] = float('inf')
                # Check that the graph does not already contain an edge from u to v,
                # if not, create the edge
                for e in u.out_edges():
                    if e.target() == v:
                        break
                # For loop finished- so edge does not exist
                else:
                    e = self.graph.add_edge(u, v)
                    # Edge weight is Euclidean distance between grid cell centers
                    self.weight[e] = math.sqrt((d_x*self.xy_resolution)**2 +
                                               (d_y*self.xy_resolution)**2)
        self.visited[current_cell] = u

    def discrete_to_continuous(self, xy_cell):
        """ Convert discrete cell coordinates to continous x-y coordinates"""
        x_o = self.origin_position[0]
        y_o = self.origin_position[1]

        x_cell = xy_cell[0]
        y_cell = xy_cell[1]

        x = x_o + x_cell * self.xy_resolution
        y = y_o + y_cell * self.xy_resolution
        return x, y

    def edge_relaxed(self, e):
        xy_cell = (self.x_disc[e.target()], self.y_disc[e.target()])
        x, y = self.discrete_to_continuous(xy_cell)
        distance_to_goal = math.sqrt((x - self.goal_position[0])**2 + (y - self.goal_position[1])**2)
        if distance_to_goal <= self.close_enough:
            self.success = True
            self.goal_vertex = e.target()
            raise StopSearch()

    def finish_vertex(self, u):
        # Called when the visitor is done exploring a vertex
        self.n_vertices_finished += 1
        # If the visitor has explored the maximum number of vertices, conclude the search
        xy_cell = (self.x_disc[u], self.y_disc[u])
        x, y = self.discrete_to_continuous(xy_cell)
        distance_to_goal = math.sqrt((x - self.goal_position[0])**2 + (y - self.goal_position[1])**2)
        # Record if we are getting any closer to the goal
        # Search terminates if self.timeout_n_vertices is set, and we explore this many vertices
        #   without getting any closer to the goal
        if distance_to_goal < self._closest_distance_to_goal:
            self._closest_distance_to_goal = distance_to_goal
            self._n_vertices_since_closer_to_goal = 0
        else:
            self._n_vertices_since_closer_to_goal += 1
        if self.max_n_vertices > 0: # max_n_vertices is set to -1 by default (disabled)
            if self.n_vertices_finished >= self.max_n_vertices:
                print("Terminating search: explored max of %d vertices" % self.max_n_vertices)
                raise StopSearch()
        if self.timeout_n_vertices > 0:
            if self._n_vertices_since_closer_to_goal >= self.timeout_n_vertices:
                print("Terminating search: explored %d vertices without getting closer to goal" %
                      self.timeout_n_vertices)
                raise StopSearch()


class AStar2DPlanner(object):

    def __init__(self, edge_evaluator: ValidEdgeEvaluator,
                 xy_resolution: float,  close_enough: float,
                 boundary: NavigationBoundary,
                 max_n_vertices: int = -1,
                 timeout_n_vertices: int = -1):
        self.edge_evaluator = edge_evaluator
        self.xy_resolution = xy_resolution
        self.close_enough = close_enough
        self.boundary = boundary
        self.max_n_vertices = max_n_vertices
        self.timeout_n_vertices = timeout_n_vertices

    def find_path(self, start_position, goal_position):
        graph = Graph()
        # Initialize graph properties for A* search
        weight = graph.new_edge_property("double")
        dist = graph.new_vertex_property("double")
        cost = graph.new_vertex_property("double")

        # Property maps for storing the discrete cell index of each x, y, angle cell
        x_disc = graph.new_vertex_property("int")
        y_disc = graph.new_vertex_property("int")

        def astar_2d_heuristic(v):
            # Heuristic for grid search with diagonal movement allowed
            # Reference:
            # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
            x_o = start_position[0]
            y_o = start_position[1]

            x_cell = x_disc[v]
            y_cell = y_disc[v]

            x_c = x_o + x_cell * self.xy_resolution
            y_c = y_o + y_cell * self.xy_resolution
            x_g = goal_position[0]
            y_g = goal_position[1]

            dx = abs(x_c - x_g)
            dy = abs(y_c - y_g)

            h=  math.sqrt(dx**2 + dy**2)

            # difference in terms of discrete cells
            dr = dy / self.xy_resolution
            dc = dx / self.xy_resolution

            D = self.xy_resolution  # cost of moving straight
            D2 = math.sqrt(2) * D  # cost of moving diagonally

            h = D * (dr + dc) + (D2 - 2* D) * min(dr, dc)

            dx1 = x_c - x_g
            dy1 = y_c - y_g
            dx2 = x_o - x_g
            dy2 = y_o - y_g
            cross = abs(dx1*dy2 - dx2*dy1)

            return h + cross*0.001
            # return math.sqrt((x - x_g)**2 + (y - y_g)**2) * (1 + 1./self.max_n_vertices)

    # Define heuristic function
        visitor = AStar2DVisitor(graph=graph,
                                 start_position=start_position, goal_position=goal_position,
                                 x_disc=x_disc, y_disc=y_disc,
                                 xy_resolution=self.xy_resolution,
                                 close_enough=self.close_enough,
                                 weight=weight, dist=dist, cost=cost,
                                 edge_evaluator=self.edge_evaluator,
                                 boundary=self.boundary,
                                 max_n_vertices=self.max_n_vertices,
                                 timeout_n_vertices=self.timeout_n_vertices)

        dist, pred = astar_search(graph, visitor.start_vertex,
                                  weight=weight, visitor=visitor, dist_map=dist, cost_map=cost,
                                  heuristic=astar_2d_heuristic, implicit=True)

        if visitor.success:

            current_v = visitor.goal_vertex
            path_xy = []
            while True:
                # Get discrete coordinates of the vertex on the path
                x_d = x_disc[current_v]
                y_d = y_disc[current_v]
                x, y = visitor.discrete_to_continuous((x_d, y_d))
                path_xy.append([x,y])

                if current_v == visitor.start_vertex:
                    break
                current_v = pred[current_v]
            path_xy.reverse()


            result = AStar2DResult(points=np.array(path_xy))
        else:
            result = None
        return result



