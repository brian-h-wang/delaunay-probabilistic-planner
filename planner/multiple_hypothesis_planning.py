from typing import List, Tuple, Optional
from numpy.typing import ArrayLike
import numpy as np
import math

from planner.probabilistic_obstacles import Obstacle2D, SafetyProbability
from planner.cell_decomposition import SafetyCellDecomposition
from planner.navigation_graphs import NavigationGraph
from planner.navigation_utils import NavigationPath
from planner.slam import ObstacleSLAM
from planner.time_counter import TimeCounter, StubTimeCounter


class MultipleHypothesisPlanningResult(object):
    """
    Object storing a multiple hypothesis planner result.
    Contains the computed candidate paths and their costs,
    as well as references to the cell decomposition, distance graph, and safety graph.
    """

    def __init__(
        self, paths, costs, distance_costs, safety_costs, cell_decomposition, distance_graph, vertex_excluded_list
    ):
        # Sort the paths in order of cost
        path_indices = [i for i in range(len(paths))]
        path_indices.sort(key=lambda path_idx: costs[path_idx])
        self.sorted_indices = path_indices

        # Save paths, costs, cell decomposition
        # The paths are saved in the order they were generated, but can be retrieved in cost-sorted
        # order using the sorted_paths, sorted_costs, etc. properties.
        self.paths: List[NavigationPath] = paths
        self.costs: List[float] = costs
        self.distance_costs: List[float] = distance_costs
        self.safety_costs: List[float] = safety_costs
        self.shortest_hypotheses = paths
        self.safest_hypotheses = paths
        self.cell_decomposition: SafetyCellDecomposition = cell_decomposition
        self.distance_graph: NavigationGraph = distance_graph
        self.vertex_excluded_list = vertex_excluded_list

    @property
    def best_path(self) -> Optional[NavigationPath]:
        if len(self.paths) == 0:
            return None
        else:
            best_idx = self.sorted_indices[0]
            return self.paths[best_idx]

    @property
    def sorted_paths(self) -> List[NavigationPath]:
        return [self.paths[i] for i in self.sorted_indices]

    @property
    def sorted_costs(self) -> List[float]:
        return [self.costs[i] for i in self.sorted_indices]

    @property
    def sorted_distance_costs(self) -> List[float]:
        return [self.distance_costs[i] for i in self.sorted_indices]

    @property
    def sorted_safety_costs(self) -> List[float]:
        return [self.safety_costs[i] for i in self.sorted_indices]

    def get_local_goal(self, distance: float):
        """
        Compute the goal point that the local planner should attempt to reach.
        To do this, take the best path from this MHP result,
        find the first vertex along this path that is in the medium or long-range (uncertain) zones,
        and use the position of this vertex as the local goal.

        If no medium/long range vertices are in the path, this gives the last vertex in the path.

        Returns
        -------
        local_goal: array_like
            A point in 2D space.
        is_global_goal: bool
            If True, the local goal is the same as the robot's global goal
                (i.e. the robot is very close to the global goal)
            If False, the local goal is closer than the global goal

        """
        path = self.best_path
        if path is None:
            return None
        # Find the first point along the path line segments which is a given distance away from the robot
        remaining_path_distance = distance
        vertices = path.graph_vertices
        vertex_positions = self.distance_graph.graph.vp["position"]
        for v_idx in range(len(vertices) - 1):
            # Get the length of the next line segment in the path
            p1 = vertex_positions[vertices[v_idx]].a
            p2 = vertex_positions[vertices[v_idx + 1]].a
            segment = p2 - p1
            segment_length = np.linalg.norm(segment)
            if segment_length > remaining_path_distance:
                goal = p1 + (segment / segment_length) * remaining_path_distance
                is_global_goal = False
                break
            else:
                remaining_path_distance -= segment_length
        else:
            # Reached end of path; goal is last vertex in the path (will be the global goal)
            goal = vertex_positions[vertices[-1]].a
            is_global_goal = True
        return goal, is_global_goal


class MultipleHypothesisPlanner(object):

    def __init__(
        self,
        robot_width: float,
        slam: ObstacleSLAM,
        safety_probability: SafetyProbability,
        range_short=5,
        range_medium=15,
        boundary=None,
        time_counter: Optional[TimeCounter] = None,
    ):
        """

        Parameters
        ----------
        obstacles: List[Obstacle2D]
        robot_width: float
        """
        obstacles = slam.get_landmarks_estimate()

        if time_counter is None:
            time_counter = StubTimeCounter()

        self.time_counter = time_counter

        with self.time_counter.count("construct_cell_decomposition"):
            self.cell_decomposition = SafetyCellDecomposition(
                slam=slam,
                robot_width=robot_width,
                safety_probability=safety_probability,
                range_short=range_short,
                range_medium=range_medium,
                boundary=boundary,
                time_counter=time_counter,
            )

        if obstacles is None:
            obstacles = []
        self.obstacles = obstacles
        self.safety_goal = safety_probability.probability

    def find_paths(
        self,
        start_position,
        goal_position,
        n_hypotheses=1,
        distance_weight=0.5,
        safety_weight=0.5,
        safety_normalize_threshold=0.99,
        min_safety_prune=0.10,
    ) -> MultipleHypothesisPlanningResult:
        """
        Find multiple hypothesis paths from the start to goal, using given weights on path distance
        and safety, and return the paths sorted in order of cost.

        Parameters
        ----------
        start_position: array_like
        goal_position: array_like
            The 2D start and goal positions.
        n_hypotheses: int
            Number of path hypotheses to compute.
        safety_normalize_threshold: float
            If all paths have at least this probability of being safe, do not normalize safety costs.
            This is because if all paths are very safe, numerical issues can appear.
            (e.g. a path with .999 probability of being safe will be treated as much less safe
            than a path with .9999 probability of being safe, when practically both paths
            should be treated as equal, and both extremely safe)

        Returns
        -------
        MultipleHypothesisPlanningResult

        """
        with self.time_counter.count("construct_navigation_graph"):
            distance_graph = NavigationGraph(
                self.cell_decomposition, start_position, goal_position, time_counter=time_counter
            )

        with self.time_counter.count("navigation_graph_path_search"):
            paths, distance_costs, safety_costs, vertex_excluded_list = (
                distance_graph.find_multiple_range_shortest_paths(
                    max_n_paths=n_hypotheses,
                    safety_threshold=self.safety_goal,
                    min_safety_prune=min_safety_prune,
                    return_excluded_vertices=True,
                )
            )

        # # Normalize costs, unless no paths were found
        if len(paths) > 0:
            max_dc = max(distance_costs)
            # dist costs can be ~zero if robot very close to goal (at that point planner doesn't matter)
            if max_dc > 1e-3:
                normalized_distance_costs = [dc / max_dc for dc in distance_costs]
            else:
                normalized_distance_costs = distance_costs
            max_sc = max(safety_costs)
            # Do *not* normalize safety if all paths are very safe (i.e. max safety cost is very low)
            if max_sc > -math.log(safety_normalize_threshold):
                normalized_safety_costs = [sc / max_sc for sc in safety_costs]
            else:
                normalized_safety_costs = safety_costs

            costs = [
                dc * distance_weight + sc * safety_weight
                for dc, sc in zip(normalized_distance_costs, normalized_safety_costs)
            ]
        else:
            costs = []

        return MultipleHypothesisPlanningResult(
            paths,
            costs,
            distance_costs=distance_costs,
            safety_costs=safety_costs,
            cell_decomposition=self.cell_decomposition,
            distance_graph=distance_graph,
            vertex_excluded_list=vertex_excluded_list,
        )
