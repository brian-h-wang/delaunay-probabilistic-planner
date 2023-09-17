"""
Module implementing Yen's algorithm for finding the K shortest paths in a graph.
"""

from graph_tool import Graph, Vertex, Edge, VertexPropertyMap, EdgePropertyMap
from graph_tool.search import astar_search, AStarVisitor, StopSearch
from graph_tool.draw import graph_draw
import math
import matplotlib
from typing import List

# Heuristic for A*
def h(v, target, pos):
    return math.sqrt(sum((pos[v].a - pos[target].a) ** 2))

def find_shortest_path(graph: Graph, vertex_positions: VertexPropertyMap,
                       edge_weights: EdgePropertyMap, start: Vertex, goal: Vertex,
                       excluded_vertices: VertexPropertyMap = None,
                       excluded_edges: EdgePropertyMap = None,
                       return_safety_cost: bool = False,
                       vertex_safeties: VertexPropertyMap = None):
    """

    Parameters
    ----------
    graph
    vertex_positions
    edge_weights
    start
    goal
    excluded_vertices
    excluded_edges
    return_safety_cost
        If True, returns path, distance cost, and safety cost

    Returns
    -------
    path: List[int]
        List of integer indices to vertices in the graph.
        Empty list if no shortest path found.
    distance: float
        Distance of this path.

    """
    # Create vertex properties needed for A* visitor
    # dist, pred = astar_search(graph, start, edge_weights,
                              # VisitorExample(touch_v, touch_e, goal),
                              # heuristic=lambda v: h(v, goal, vertex_positions))

    if return_safety_cost:
        assert vertex_safeties is not None, "Must provide vertex_safeties arg to return safety costs."

    dist = graph.new_vertex_property("double")
    # Need to copy the weight property map because the visitor will modify the edge weights
    # by setting excluded edges, or edges between excluded vertices, to inf.
    # Don't want to modify the original graph
    edge_weights: EdgePropertyMap = edge_weights.copy()
    visitor = KShortestVisitor(graph, goal, edge_weights, dist,
                               excluded_v=excluded_vertices, excluded_e=excluded_edges)

    dist, pred = astar_search(graph, start, edge_weights,
                              visitor=visitor,
                              heuristic=lambda v: h(v, goal, vertex_positions))

    distance = dist[goal]
    safety_cost = 0
    if distance == float('inf'):
        path = []
        safety_cost = float('inf')
    else:
        path = [int(goal)]
        v = goal
        while v != start:
            if return_safety_cost:
                safety_cost += -math.log(vertex_safeties[v])
            next_v: int = pred[v]
            path.append(next_v)
            v = graph.vertex(next_v)

        path.reverse()

    # distances = [dist[v] for v in path]
    if return_safety_cost:
        return path, distance, safety_cost
    else:
        return path, distance


class KShortestVisitor(AStarVisitor):

    def __init__(self, g, goal, weight, dist, excluded_v=None, excluded_e=None):
        self.g: Graph = g
        self.goal = goal
        self.weight: EdgePropertyMap = weight
        self.dist: VertexPropertyMap = dist

        # self.touched_v = self.g.new_vertex_property("bool")
        # self.touched_e = self.g.new_edge_property("bool")

        if excluded_v is None:
            excluded_v = self.g.new_vertex_property("bool", val=False)
        self.excluded_v: VertexPropertyMap = excluded_v
        if excluded_e is None:
            excluded_e = self.g.new_edge_property("bool", val=False)
        self.excluded_e: EdgePropertyMap = excluded_e

        # Keep track of which excluded vertices have been visited by the A* search
        # Also record the order verticies were visited
        # Used for multi-hypothesis planning

        # These lists are needed for cost2come computation

    # def discover_vertex(self, u):
    #     self.touched_v[u] = True
    #
    # def examine_edge(self, e):
    #     self.touched_e[e] = True

    def examine_vertex(self, u: Vertex):
        # Invoked as a vertex is popped from the queue (i.e. chosen as the lowest cost next vertex)
        for e in u.out_edges():
            if self.excluded_e[e] or self.excluded_v[e.target()] or self.excluded_v[e.source()]:
                self.weight[e] = float('inf')

    def edge_relaxed(self, e):
        if e.target() == self.goal:
            raise StopSearch()

