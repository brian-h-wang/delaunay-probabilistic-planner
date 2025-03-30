"""
Module for constructing 2D navigation graphs, given a set of estimated obstacles.
"""

from typing import List, Tuple, Dict, Optional
from numpy.typing import ArrayLike
import numpy as np
import math
from abc import ABC
from graph_tool import Graph, Vertex, Edge, VertexPropertyMap, EdgePropertyMap
import matplotlib.pyplot as plt
import heapq

from planner.shortest_path import find_shortest_path
from planner.cell_decomposition import (
    SafetyCellDecomposition,
    RANGE_SHORT,
    RANGE_MEDIUM,
    RANGE_LONG,
    EXCLUDE_LONG_RANGE_VERTICES,
)
from planner.navigation_utils import NavigationPath
from planner.time_counter import TimeCounter, StubTimeCounter

EDGE_VERTICES_NONE = (-1, -1)

DISALLOW_DUPLICATE_PATHS = True


class HypothesisQueue(ABC):
    """
    Tree storing hypotheses of graph edges.

    """

    def __init__(self, graph: Graph, vertex_excluded: VertexPropertyMap):
        """

        Parameters
        ----------
        graph: Graph
            The navigation graph over which to search for navigation hypotheses.
        vertex_excluded: VertexPropertyMap
            Map indicating which vertices should be excluded, over all hypotheses.
            If vertex_excluded[v] is True, then vertex v will be excluded in all hypotheses.

        """
        self.graph = graph
        self.cell_faces_queue = []
        self.vertex_excluded = vertex_excluded
        self.n_added = 0

    def pop(self) -> VertexPropertyMap:
        vertex_excluded = self.graph.new_vertex_property("bool", val=False)
        raise NotImplementedError

    def push(self, cell_face: Tuple[int, int], safety: float, current_vertex_excluded: VertexPropertyMap):
        raise NotImplementedError

    def __len__(self):
        return len(self.cell_faces_queue)


class SimpleQueue(HypothesisQueue):
    """
    Store graph edges with their probabilities.
    Explore graph edges in ascending order of safety.

    Once an edge is deemed "unsafe" in a hypothesis, it is considered unsafe for all
    future hypotheses.

    Note this can result in multiple hypotheses which are identical.
    """

    def __init__(self, graph, vertex_excluded):
        super().__init__(graph, vertex_excluded)
        # Initialize queue of cell faces, sorted by safety

    def pop(self) -> VertexPropertyMap:
        # Get the least safe edge between obstacles from the queue,
        #  and exclude it from the shortest path search.
        safety, _, least_safe_cell_face = heapq.heappop(self.cell_faces_queue)

        graph = self.graph
        vertex_cell_decomp_edge = graph.vp["cell_decomposition_edge"]
        for v in graph.vertices():
            v_cell_face = tuple(vertex_cell_decomp_edge[v].a)
            # TODO try a simple == between the tuples and see if it always works
            if v_cell_face == least_safe_cell_face or (
                v_cell_face[0] == least_safe_cell_face[1] and v_cell_face[1] == least_safe_cell_face[0]
            ):
                self.vertex_excluded[v] = True
        return self.vertex_excluded

    def push(self, cell_face, safety, current_vertex_excluded: VertexPropertyMap):
        heapq.heappush(self.cell_faces_queue, (safety, self.n_added, cell_face))
        self.n_added += 1


class HypothesisTreeQueue(HypothesisQueue):
    """
    Each candidate unsafe edge in the queue is saved along with the hypothesized state of the world
    under which the candidate shortest path that the edge belongs to was generated.

    A candidate hypothesis is added to the queue along with its likelihood, calculated as the
        product of likelihoods that all hypothetically unsafe edges it contains are unsafe.

    The pop() method returns the current highest likelihood hypothesis in the queue.
    """

    def __init__(self, graph, vertex_excluded):
        super().__init__(graph, vertex_excluded)
        self.prev_hypothesis_safety = 1.0

    def pop(self) -> VertexPropertyMap:
        neg_prev_likelihood, _, least_safe_cell_face, previous_vertex_excluded = heapq.heappop(self.cell_faces_queue)
        prev_likelihood = -1 * neg_prev_likelihood
        graph = self.graph
        vertex_cell_decomp_edge = graph.vp["cell_decomposition_edge"]
        vertex_excluded: VertexPropertyMap = previous_vertex_excluded.copy()
        for v in graph.vertices():
            v_cell_face = tuple(vertex_cell_decomp_edge[v].a)
            if v_cell_face == least_safe_cell_face or (
                v_cell_face[0] == least_safe_cell_face[1] and v_cell_face[1] == least_safe_cell_face[0]
            ):
                vertex_excluded[v] = True
        self.prev_hypothesis_safety = 0
        return vertex_excluded, prev_likelihood

    def push(self, cell_face, hypothesis_likelihood, current_vertex_excluded: VertexPropertyMap):
        # Add hypothesis using the negative likelihoods as priorities,
        #   since heapq takes the smallest priority item when calling heappop()
        heapq.heappush(
            self.cell_faces_queue, (-hypothesis_likelihood, self.n_added, cell_face, current_vertex_excluded)
        )
        self.n_added += 1

    def __str__(self):
        entries = []
        for entry in self.cell_faces_queue:
            entries.append("(Edge (%d, %d), safety %.3f)" % (entry[2][0], entry[2][1], entry[0]))
        return str(entries)


class ShortestPathGraph(ABC):

    def __init__(self, cell_decomposition: SafetyCellDecomposition, start_position, goal_position):
        self.cell_decomposition = cell_decomposition

    def find_paths(self, n_paths=1, return_costs=False):
        raise NotImplementedError

    def evaluate_path(self, path: NavigationPath):
        raise NotImplementedError


class NavigationGraph(ShortestPathGraph):

    def __init__(
        self,
        cell_decomposition: SafetyCellDecomposition,
        start_position,
        goal_position,
        time_counter: Optional[TimeCounter] = None,
    ):
        super().__init__(cell_decomposition, start_position, goal_position)

        graph = Graph(directed=False)
        # Vertices in the distance graph correspond to the faces between Delaunay cells,
        #   in the cell decomposition

        # Vertex property for x,y position of vertices
        vertex_positions = graph.new_vertex_property("vector<double>")
        vertex_safeties = graph.new_vertex_property("double")
        # vertex_n_measurements = graph.new_vertex_property("int")
        vertex_range_zone = graph.new_vertex_property("int")

        # Property that records which vertices correspond to Delaunay cell centers
        # vertex_is_cell = graph.new_vertex_property("bool", val=False)
        # Records which vertices correspond to Delaunay cell faces
        # NOTE - this property was needed before, when the graph contained vertices on the
        #        cell faces, and cell centroids. Now, all vertices are face vertices, except
        #        the start and goal. This property is still needed for the k_shortest_paths
        #        function to work properly, could revise that function to remove this property
        vertex_is_face = graph.new_vertex_property("bool", val=True)

        # Also record the cell decomposition edge associated with each vertex
        # A cell decomp edge is a pair of obstacle indices (vertices of the Delaunay triangulation)
        # The start and goal vertices have [-1, -1] saved for this property
        vertex_cell_decomp_edge = graph.new_vertex_property("vector<int>")

        # Edge weights (Euclidean distances)
        edge_weights = graph.new_edge_property("double")
        # Each distance graph edge corresponds to a cell decomposition cell; record the cell indices
        edge_cells = graph.new_edge_property("int")

        # Dict that maps a cell index to the indices of the vertices on the cell's faces
        cell_to_face_vertices_dict: Dict[int, List[int]] = {}
        # Dict that maps a pair of Delaunay vertex indices (i.e. obstacle indices)
        #   to the list of indices to distance graph vertices that lies between these obstacles
        obstacles_to_face_vertices_dict: Dict[Tuple[int, int], List[int]] = {}

        # Get the Euclidean distance along an edge. Called after adding each graph edge
        def calculate_edge_distance(e: Edge):
            p1 = vertex_positions[e.source()].a
            p2 = vertex_positions[e.target()].a
            return np.linalg.norm(p2 - p1)

        # For each simplex (triangle cell) in the cell decomposition,
        for cell_idx in range(cell_decomposition.n_cells):
            # Get the vertices that make up this cell
            # For each pair of vertices, create a distance graph vertex at their midpoint
            face_vertices_lists: List[List[int]] = []
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                v1 = cell_decomposition.delaunay.simplices[cell_idx, i]
                v2 = cell_decomposition.delaunay.simplices[cell_idx, j]
                range1 = cell_decomposition.get_vertex_range(v1)
                range2 = cell_decomposition.get_vertex_range(v2)
                # Skip if either vertex is long-range
                if EXCLUDE_LONG_RANGE_VERTICES:
                    if range1 == RANGE_LONG or range2 == RANGE_LONG:
                        continue
                # Get the vertex on this cell decomposition face
                face_vertices = obstacles_to_face_vertices_dict.setdefault((v1, v2), None)
                if face_vertices is None:
                    # Vertex was not yet created; create it
                    face_vertex_positions = cell_decomposition.get_face_vertex_positions(v1, v2)
                    face_vertices = []
                    for face_v_idx in range(face_vertex_positions.shape[0]):
                        face_vertex = graph.add_vertex()
                        vertex_is_face[face_vertex] = True
                        vertex_positions[face_vertex] = face_vertex_positions[face_v_idx, :]
                        vertex_safeties[face_vertex] = cell_decomposition.get_edge_safety(v1, v2)
                        # Determine if this face vertex is short, medium, or long range
                        vertex_range_zone[face_vertex] = max(range1, range2)
                        # vertex_n_measurements[face_vertex] = cell_decomposition.get_edge_n_measurements(v1, v2)
                        vertex_cell_decomp_edge[face_vertex] = [v1, v2]
                        face_vertices.append(int(face_vertex))
                    # Add to obstacles->vertex dict
                    obstacles_to_face_vertices_dict[(v1, v2)] = face_vertices
                    obstacles_to_face_vertices_dict[(v2, v1)] = face_vertices
                face_vertices_lists.append(face_vertices)

            # current_cell_face_vertices is now a list of length 3,
            # containing a list for all the face vertices on each of the 3 cell edges.
            # Connect the face vertices of this cell to all other face vertices on other cell edges
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                face1_vertices: List[int] = face_vertices_lists[i]
                face2_vertices: List[int] = face_vertices_lists[j]
                for v1 in face1_vertices:
                    for v2 in face2_vertices:
                        edge = graph.add_edge(v1, v2)
                        edge_weights[edge] = calculate_edge_distance(edge)
                        edge_cells[edge] = cell_idx
            cell_to_face_vertices_dict[cell_idx] = [v for vl in face_vertices_lists for v in vl]

        # Create vertices for start and goal, and connect them to the other vertices
        start_vertex = graph.add_vertex()
        vertex_positions[start_vertex] = start_position
        vertex_safeties[start_vertex] = 1.0
        vertex_cell_decomp_edge[start_vertex] = [-1, -1]
        vertex_is_face[start_vertex] = False
        start_cell = cell_decomposition.find_delaunay_cell(start_position)
        self.start_cell = start_cell

        goal_vertex = graph.add_vertex()
        vertex_positions[goal_vertex] = goal_position
        vertex_safeties[goal_vertex] = 1.0
        vertex_cell_decomp_edge[goal_vertex] = [-1, -1]
        vertex_is_face[goal_vertex] = False
        goal_cell = cell_decomposition.find_delaunay_cell(goal_position)
        self.goal_cell = goal_cell

        # Connect start and goal to other vertices in the graph
        for new_vertex, cell in zip([start_vertex, goal_vertex], [start_cell, goal_cell]):
            new_vertex_position = vertex_positions[new_vertex].a
            # Check if start/goal vertex is inside a cell decomposition cell
            if cell != -1:
                # Vertex is in a cell - connect it to the cell's three cell face vertices
                # Get all vertices on the cell's three faces
                vertices_to_connect = cell_to_face_vertices_dict[cell]
                for neighbor_vertex in vertices_to_connect:
                    e = graph.add_edge(new_vertex, neighbor_vertex)
                    edge_weights[e] = calculate_edge_distance(e)
                    edge_cells[e] = cell
            else:
                # Vertex is not in a cell - connect it to vertices on cell edges that are visible
                # ("visible" means there are no other cell decomp edges intersecting the line
                #   segment between the start/goal vertex, and the cell edge's midpoint)
                visible_cells, visible_cell_edges = cell_decomposition.get_visible_cells(new_vertex_position)
                for visible_edge in visible_cell_edges:
                    visible_vertices = obstacles_to_face_vertices_dict[visible_edge]
                    for visible_vertex in visible_vertices:
                        e = graph.add_edge(new_vertex, visible_vertex)
                        edge_weights[e] = calculate_edge_distance(e)
                        edge_cells[e] = cell

        # Also connect the start to the goal if they are in the same cell,
        # or both not in any cell and visible to each other
        if self.start_cell == self.goal_cell:
            if self.start_cell != -1 or (
                self.start_cell == -1 and cell_decomposition.check_point_visible(start_position, goal_position)
            ):
                e = graph.add_edge(start_vertex, goal_vertex)
                edge_weights[e] = calculate_edge_distance(e)
                edge_cells[e] = self.start_cell

        # Save graph and properties as attributes
        graph.vp["position"] = vertex_positions
        graph.vp["safety"] = vertex_safeties
        # graph.vp["n_measurements"] = vertex_n_measurements
        graph.vp["cell_decomposition_edge"] = vertex_cell_decomp_edge
        graph.vp["is_face"] = vertex_is_face
        graph.vp["range_zone"] = vertex_range_zone
        graph.ep["weight"] = edge_weights
        graph.ep["cell"] = edge_cells

        self.graph = graph
        self.start_vertex = start_vertex
        self.goal_vertex = goal_vertex
        self.obstacles_to_face_vertices_dict = obstacles_to_face_vertices_dict

    # TODO this function could be moved to multiple_hypothesis_planning module
    def find_multiple_range_shortest_paths(
        self, max_n_paths: int, safety_threshold=0.95, min_safety_prune=0.10, return_excluded_vertices: bool = False
    ):
        """
        Algorithm for finding path hypotheses, using different behaviors for short,
        medium, and long-range zones.

        In the short range, faces between obstacles are treated as safe/unsafe, without uncertainty.

        Parameters
        ----------
        max_n_paths: int
            Maximum number of hypotheses to consider
        safety_threshold: float
            Safety goal that the path search will try to meet.
            Should be between 0.0 and 1.0
        min_safety_prune: float
            Threshold for minimum path safety.
            Should be set to a low value, e.g. 0.10 (10% safety likelihood)
            In the medium and long range, any graph edges with less than this level of safety
            will be pruned.

        Returns
        -------
        paths: List[NavigationPath]
        distance_costs: List[float]
        safety_costs: List[float]
        vertex_excluded_list: List[VertexPropertyMap]
            Only returned if return_excluded_vertices is True

        """

        graph = self.graph
        vertex_positions = graph.vp["position"]
        vertex_cell_decomp_edge = graph.vp["cell_decomposition_edge"]
        vertex_safeties = graph.vp["safety"]
        vertex_range_zone = graph.vp["range_zone"]
        edge_weights = graph.ep["weight"]
        edge_cells = graph.ep["cell"]
        start = self.start_vertex
        goal = self.goal_vertex

        # Initialize lists for shortest path search results
        paths = []
        distance_costs = []
        safety_costs = []

        vertex_excluded = graph.new_vertex_property("bool", val=False)
        prev_likelihood = 1.0
        for v in graph.vertices():
            # in the short-range zone, exclude graph vertices under the safety goal
            if vertex_range_zone[v] == RANGE_SHORT:
                if vertex_safeties[v] < safety_threshold:
                    vertex_excluded[v] = True
            # in the medium/long-range zones, exclude vertices under the minimum safety
            else:
                if vertex_safeties[v] < min_safety_prune:
                    vertex_excluded[v] = True

        # Initialize hypotheses queue
        hypotheses_queue = HypothesisTreeQueue(graph, vertex_excluded)

        original_vertex_excluded = vertex_excluded.copy()

        vertex_excluded_list = []

        path_idx = 0
        while path_idx < max_n_paths:
            # Get the least safe edge between obstacles from the queue,
            #  and exclude it from the shortest path search.
            if len(hypotheses_queue) > 0:
                vertex_excluded, prev_likelihood = hypotheses_queue.pop()
            elif path_idx != 0:
                # No more edges to try; no more candidate paths can be generated
                break

            path, distance_cost = find_shortest_path(
                graph,
                vertex_positions,
                edge_weights,
                start,
                goal,
                excluded_vertices=vertex_excluded,
                return_safety_cost=False,
            )

            if len(path) == 0:
                # No more paths possible, done with multiple-hypothesis path search
                break

            # Calculate the overall safety cost of this path,
            #  over vertices in short and medium range
            safety_cost = 0

            # Separately, count up the safety cost in the short range only
            #   The planner cannot execute a path that is unsafe in the short range.
            short_range_safety_cost = 0
            # Record the lowest safety value out of all the medium-range graph edges
            for v in path:
                if vertex_range_zone[v] == RANGE_MEDIUM or vertex_range_zone[v] == RANGE_SHORT:
                    cell_face = tuple(vertex_cell_decomp_edge[v].a)
                    safety = vertex_safeties[v]
                    # Calculate likelihood of this hypothesis
                    # Product of the likelihood this new candidate edge is unsafe, with the likelihoods
                    #  of all other unsafe marked edges in this hypothesis
                    likelihood = prev_likelihood * (1.0 - safety)
                    hypotheses_queue.push(cell_face, likelihood, vertex_excluded)

                    safety_cost += -math.log(safety)
                    if vertex_range_zone[v] == RANGE_SHORT:
                        short_range_safety_cost += -math.log(safety)

            # Iterate over the list of graph vertices in the shortest path,
            # and convert it to a list of Delaunay graph cell indices and cell edges
            vertex_indices = path
            cells = []
            edges = []

            for i in range(len(vertex_indices) - 1):
                # For each pair of vertices, get the cell that the edge between them passes through
                v1 = vertex_indices[i]
                v2 = vertex_indices[i + 1]

                e = graph.edge(v1, v2)
                cells.append(edge_cells[e])
                # For each vertex, except the start and goal, get the obstacles indices corresponding
                if i != 0:
                    cell_decomp_edge = vertex_cell_decomp_edge[v1]
                    edges.append(tuple(cell_decomp_edge.a))

            if len(cells) == 0 and len(edges) == 0:  # path is empty; i.e. no path found
                continue

            short_range_path_safety = math.exp(-short_range_safety_cost)
            if short_range_path_safety < safety_threshold:
                # Path is unsafe in the short range, not allowed by planner
                continue

            try:
                new_path = NavigationPath(cells, edges, vertex_indices, cell_decomposition=self.cell_decomposition)
                # Check if this candidate path is already in the set of paths
                path_is_duplicate = False
                if DISALLOW_DUPLICATE_PATHS:
                    for p in paths:
                        if p == new_path:
                            path_is_duplicate = True
                            break
                # Path is not a duplicate, add to the set of paths
                if not path_is_duplicate:
                    paths.append(new_path)
                    distance_costs.append(distance_cost)
                    safety_costs.append(safety_cost)
                path_idx += 1  # Note - duplicates do not get added to the paths set but do count as a hypothesis
                if return_excluded_vertices:
                    ve = vertex_excluded.copy()
                    for v in graph.vertices():
                        if original_vertex_excluded[v]:
                            ve[v] = False
                    vertex_excluded_list.append(ve)
            except AssertionError:
                # For debugging
                print("Warning: navigation path %d is invalid." % (path_idx))
                print("The cells are: " + str(cells))
                print("The vertices are: " + str(vertex_indices))

            # Break, if we found a path that meets the safety goal
            path_safety = math.exp(-safety_cost)

            if path_safety >= safety_threshold:
                break

        if return_excluded_vertices:
            return paths, distance_costs, safety_costs, vertex_excluded_list
        else:
            return paths, distance_costs, safety_costs

    def get_path_xy(self, path: NavigationPath):
        graph_vertices = path.graph_vertices

        xy_list = []
        vertex_positions = self.graph.vp["position"]
        for v in graph_vertices:
            xy_list.append(list(vertex_positions[v].a))

        return np.array(xy_list)

    def plot(self, ax: plt.Axes = None, show_edge_weights=False):
        """DEPRECATED - use plotting module instead."""
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        ax.axis("equal")

        cell_vertex_size = 12
        face_vertex_size = 6

        vertex_color = (0.6, 0.2, 0.2)
        start_color = (0.2, 0.6, 0.2)
        goal_color = (0.2, 0.6, 0.2)
        edge_color = (0.4, 0.4, 0.4)

        graph = self.graph
        vertex_position = graph.vp["position"]
        edge_weights = graph.ep["weight"]

        # Construct arrays of the x- and y- positions of all vertices
        # Cell and face vertices should be plotted with different sizes/colors to distinguish
        face_x = []
        face_y = []
        for v in self.graph.vertices():
            p = vertex_position[v].a
            face_x.append(p[0])
            face_y.append(p[1])

        # Plot the vertices
        ax.plot(face_x, face_y, ".", c=vertex_color, markersize=face_vertex_size, zorder=5)

        # Plot the start and goal
        start_position = vertex_position[self.start_vertex].a
        goal_position = vertex_position[self.goal_vertex].a
        ax.plot(start_position[0], start_position[1], "^", c=start_color, markersize=cell_vertex_size, zorder=7)
        ax.plot(goal_position[0], goal_position[1], "*", c=goal_color, markersize=cell_vertex_size, zorder=7)

        # Plot all edges
        for e in self.graph.edges():
            p1 = vertex_position[e.source()].a
            p2 = vertex_position[e.target()].a
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", c=edge_color, linewidth=1.0, zorder=4)

        # self.cell_decomposition.plot(ax=ax, gray=True, show_labels=False)
        from plotting import CellDecompositionPlotter

        CellDecompositionPlotter(self.cell_decomposition, ax=ax)

        return ax
