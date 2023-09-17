"""
Brian Wang, bhw45@cornell.edu
Module for various utility functions, implemented in numba for speed where possible.
"""

from graph_tool import Graph, Vertex, Edge, VertexPropertyMap, EdgePropertyMap
import math
import numba
import numpy as np

#### Line intersection checking
# Source: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    #return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def check_line_intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def _test_line_intersect():
    test_cases = [([0, 0], [5, 5], [5, 0], [0, 5], True),
                  ([0, 5], [3, 5], [7, 4], [8, 7], False),
                  ([-100, -100], [-100, -90], [50, 130], [50, 140], False),
                  ([0, 0], [5,0], [5, -5], [5,10], True)]
    for a1, a2, b1, b2, result in test_cases:
        test_result = check_line_intersect(a1, a2, b1, b2)
        print("Checking intersect between (%s, %s) and (%s, %s)" % (a1, a2, b1, b2))
        print("Result was %s, should be %s" % (test_result, result))
        print("")

### Graph construction

def build_triangle_graph(n_cols, col_distances, row_distances):
    """
    Creates an "upper triangular" graph for testing Yen's Algorithm.
    The graph consists of N columns.
    Column i (from 1 to N) contains i nodes.
    Each node is connected to its neighbor directly to the right of it,
    directly below it, and diagonally to the right and below
    (for each of these neighbors that exists):

    Example layout for n_cols = 3:
    O - O - O
      \ | \ |
        O - O
          \ |
            O

    All edges have cost equal to the Euclidean 2D distance between vertices.

    Parameters
    ----------
    n_cols: int
        Specifies the number of columns and rows of graph vertices.
    col_distances: list, int, or float
        Specifies spacing between the columns.
        If a list, should be a list of ints or float with length (n_cols-1)
        If an int or float, all columns will have the same spacing.
    row_distances: list, int, or float
        Specifies spacing between the rows.
        If a list, should be a list of ints or float with length (n_cols-1)
        If an int or float, all rows will have the same spacing.

    Returns
    -------
    triangle_graph: Graph
    vertex_positions: VertexPropertyMap
    edge_weights: EdgePropertyMap

    """
    if type(col_distances) == int or type(col_distances) == float:
        col_distances = [col_distances for _ in range(n_cols-1)]
    if type(row_distances) == int or type(row_distances) == float:
        row_distances = [row_distances for _ in range(n_cols-1)]

    triangle_graph = Graph(directed=False)
    vertex_positions = triangle_graph.new_vertex_property("vector<double>")

    row_col_dict = {}

    # Check that column and row distances lists have the correct lengths
    assert len(col_distances) == n_cols - 1, "For %d columns, need %d between-column distances" % (n_cols, n_cols-1)
    assert len(row_distances) == n_cols - 1, "For %d columns, need %d between-row distances" % (n_cols, n_cols-1)

    x = 0
    # Add vertices to the example graph
    for col in range(n_cols):
        # The first column should have 1 vertex, second column has 2 vertices, and so on...
        y = 0
        for row in range(col + 1):
            v = triangle_graph.add_vertex()
            vertex_positions[v] = [x,y]
            row_col_dict[row, col] = int(v)
            if row < col:
                y += row_distances[row]
        if col < len(col_distances):
            x += col_distances[col]

    # Add edges to the example graph
    for col in range(n_cols):
        # The first column should have 1 vertex, second column has 2 vertices, and so on...
        for row in range(col + 1):
            v = triangle_graph.vertex(row_col_dict[row, col])
            if col+1 < n_cols:
                # Neighbor 1: to the right, in the next column
                n1 = triangle_graph.vertex(row_col_dict[row, col+1])
                triangle_graph.add_edge(source=v, target=triangle_graph.vertex(n1))
                # Neighbor 2: diagonally down and to the right
                n2 = triangle_graph.vertex(row_col_dict[row+1, col+1])
                triangle_graph.add_edge(source=v, target=triangle_graph.vertex(n2))
            # Neighbor 3: vertically downwards
            if row+1 < col+1:
                n3 = triangle_graph.vertex(row_col_dict[row+1, col])
                triangle_graph.add_edge(source=v, target=triangle_graph.vertex(n3))

    # Add Euclidean distances as edge weights
    edge_weights = triangle_graph.new_edge_property("double")
    for edge in triangle_graph.edges():
        p1 = vertex_positions[edge.source()]
        p2 = vertex_positions[edge.target()]
        edge_weights[edge] = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    return triangle_graph, vertex_positions, edge_weights

@numba.njit()
def check_collision(robot_position, robot_width, obstacle_positions, obstacle_diameters):
    """
    Check if the robot is in collision with any obstacles.

    Parameters
    ----------
    robot_position: ndarray
        [x, y]
    robot_width: float
    obstacle_positions: ndarray
        N by 2 array
    obstacle_diameters: ndarray
        N-element array

    Returns
    -------
    bool
        True if there is a crash, False otherwise
    """
    for i in range(obstacle_positions.shape[0]):
        if (np.linalg.norm(robot_position - obstacle_positions[i,:]) <
                (robot_width/2 + obstacle_diameters[i]/2)):
            return True
    return False

@numba.njit()
def fix_angle(angle: float):
    """ Adjust an angle, in radians, to be within the range +- pi (i.e. +- 180 degrees)"""
    while abs(angle) > math.pi:
        if angle > math.pi:
            angle -= 2*math.pi
        elif angle < math.pi:
            angle += 2*math.pi
    return angle


@numba.njit()
def line_segment_circle_intersect(x1, y1, x2, y2, xc, yc, r):
    """
    Check if a line segment intersects with a circle
    Reference:
    https://math.stackexchange.com/questions/2837/how-to-tell-if-a-line-segment-intersects-with-a-circle/2862#2862

    Parameters
    ----------
    x1: float
    y1: float
    x2: float
    y2: float
        (x1, y1) and (x2, y2) are the two endpoints of the line segment
    xc: float
    yc: float
    r: float
        The center (xc, yc) of the circle, and its radius.

    Returns
    -------
    bool
        True if the line segment and circle intersect, False otherwise.

    """
    h = xc
    k = yc
    a = (x1**2 - 2*x1*x2 + x2**2) + (y1**2 - 2*y1*y2 + y2**2)
    b = (2*x1*x2 - 2*x2**2 - 2*h*x1 + 2*h*x2) + (2*y1*y2 - 2*y2**2 - 2*k*y1 + 2*k*y2)
    c = (x2**2 - 2*h*x2 + h**2) + (y2**2 - 2*k*y2 + k**2) - r**2

    # Solve using the quadratic equation
    # If either solution is a real number between 0 and 1, inclusive, there is an intersection
    x1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
    if not math.isnan(x1):
        if 0 <= x1 <= 1:
            return True
    x2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
    if not math.isnan(x2):
        if 0 <= x2 <= 1:
            return True
    return False


@numba.njit()
def find_unoccluded_obstacles_centroids(robot_position, obstacle_positions, obstacle_diameters):
    """
    DEPRECATED: Use find_unoccluded_obstacles instead
    Finds which obstacles have centroids visible to the robot

    """
    obstacle_radii = obstacle_diameters / 2.
    n_obstacles = obstacle_positions.shape[0]
    unoccluded = [True for _ in range(n_obstacles)]
    x1 = robot_position[0]
    y1 = robot_position[1]
    # Check if each obstacle is occluded
    for obs_idx in range(n_obstacles):
        obs_xy = obstacle_positions[obs_idx, :]
        x2 = obs_xy[0]
        y2 = obs_xy[1]
        # Check if any other obstacle intersects the line segment from the robot,
        #   to this obstacle's centroid
        for other_obs_idx in range(n_obstacles):
            if obs_idx == other_obs_idx:
                continue
            xc = obstacle_positions[other_obs_idx, 0]
            yc = obstacle_positions[other_obs_idx, 1]
            r = obstacle_radii[other_obs_idx]
            if line_segment_circle_intersect(x1, y1, x2, y2, xc, yc, r):
                unoccluded[obs_idx] = False
                break
    return unoccluded


@numba.njit()
def find_unoccluded_obstacles(robot_position, obstacle_positions, obstacle_diameters,
                              n_occlusion_points: int = 20):
    """
    Find obstacles which are visible to the robot.
    An obstacle is considered occluded if it is fully occluded from the robot's view.

    Occlusions are approximated, by checking visibility of a number of points placed around the
    outer circumference of each obstacle. The obstacle is considered visible if *any* of these
    points are visible to the robot.

    Parameters
    ----------
    robot_position:
        x, y position
    obstacle_positions:
        Ndarray of size (N,2)
    obstacle_diameters:
        Ndarray of size (N)
    n_occlusion_points: int
        The occlusion algorithm approximates occlusions by checking visibility of this many points.
        Increasing this value will better approximate full occlusions, but increase computation time.

    Returns
    -------

    """
    obstacle_radii = obstacle_diameters / 2.
    n_obstacles = obstacle_positions.shape[0]
    unoccluded = [False for _ in range(n_obstacles)]
    x1 = robot_position[0]
    y1 = robot_position[1]
    # Check if each obstacle is occluded
    for obs_idx in range(n_obstacles):
        obs_center = obstacle_positions[obs_idx, :]

        # Adjust the endpoint to the last angle manually - numba does not support "endpoint" arg to linspace
        last_angle = np.pi*2 - (np.pi*2/n_occlusion_points)
        angles = np.linspace(0, last_angle, n_occlusion_points)
        r2 = obstacle_radii[obs_idx]

        # Place the occlusion-check points and check visibility to each
        # If any of these points are visible, the obstacle is unoccluded
        for angle in angles:
            x2 = obs_center[0] + math.cos(angle) * r2
            y2 = obs_center[1] + math.sin(angle) * r2
            # Check if any other obstacle intersects the line segment from the robot,
            #   to this obstacle's centroid
            point_is_visible = True
            for other_obs_idx in range(n_obstacles):
                if obs_idx == other_obs_idx:
                    continue
                xc_other = obstacle_positions[other_obs_idx, 0]
                yc_other = obstacle_positions[other_obs_idx, 1]
                r_other = obstacle_radii[other_obs_idx]
                if line_segment_circle_intersect(x1, y1, x2, y2, xc_other, yc_other, r_other):
                    point_is_visible = False
                    break
            if point_is_visible:
                unoccluded[obs_idx] = True
                break
    return unoccluded


