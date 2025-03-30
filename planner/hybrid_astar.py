"""
Module for path planning using Hybrid A* search.

Motion primitives code adapted from Jacopo Banfi's mu_planner repository:
https://github.com/jacoban/mu_planner/blob/master/scripts/motion.py

References:
    - Montemerlo et al. "Junior: The Stanford Entry in the Urban Challenge." JFR 2008.
    - Dolgov et al. "Practical Search Techniques in Path Planning for Autonomous Driving." AAAI 2008.
    - Petereit et al. "Application of Hybrid A* to an Autonomous Mobile Robot for Path Planning in
        Unstructured Outdoor Environments." ROBOTIK 2012.


"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from graph_tool import Graph, Vertex, Edge, VertexPropertyMap, EdgePropertyMap
from graph_tool.search import AStarVisitor, astar_search, StopSearch
from typing import List, Optional
from numpy.typing import ArrayLike
from planner.navigation_utils import ValidEdgeEvaluator


class MotionPrimitives(object):
    """
    Class for storing the set of motion primitives used for a hybrid A* path search.
    """

    def __init__(self, velocity: float, dt: float, angular_rates: List[int], allow_reverse=True):
        """

        Parameters
        ----------
        drive_distance
            The arc distance the robot should travel as it executes a motion primitive
        steering_angles: List[int]
            Change in orientation for each motion primitive, *specified in degrees*
            For example, steering_angles = [-40, 0, 40]
            will calculate motion primitves for the case when the robot drives distance D
            forwards with turning, and drives distance D in an arc while ending up at -40 or +40
            degrees difference from its starting orientation (D is the drive_distance)
        # TODO update documentation with angular_rates
        n_steps: int or None, optional
            If n_steps is None (default), this just stores the robot's end position after
            executing the motion primitives.
            If n_steps is an integer, the motion primitives will be stored as trajectories,
            each with n_steps points
            (use this for plotting the robot's motion along the motion primitive)
        """
        # During each motion primitive, the robot moves forwards by drive_distance,
        # while turning at constant rate
        # self.n_primitives = len(steering_angles)
        self.allow_reverse = allow_reverse
        # self.steering_angles = list(steering_angles)
        self.angular_rates = list(angular_rates)
        self.n_primitives = len(self.angular_rates)
        # self.drive_distance = drive_distance
        # Choose dt and velocity such that the robot moves drive_distance forward
        # self.velocities = [self.drive_distance for _ in self.angular_rates]
        self.velocities = [velocity for _ in self.angular_rates]
        self.dt = dt
        if allow_reverse:
            # Add a primitive that causes the robot to reverse in a straight line
            self.n_primitives += 1
            self.angular_rates.append(0.0)
            self.velocities.append(-velocity)
        # Define an array that stores the robot's final pose after executing each motion primitive
        self.motion_primitives = np.empty((self.n_primitives, 3))
        # for i, (steering_angle, v) in enumerate(zip(self.steering_angles, self.velocities)):
        for i, (v, angular_rate) in enumerate(zip(self.velocities, self.angular_rates)):

            def fun(t, y):  # Calculate change in x, y, angle
                angle = math.radians(y[2])
                return np.array([v * math.cos(angle), v * math.sin(angle), angular_rate])
                # return np.array([v * math.cos(angle), v * math.sin(angle), steering_angle])

            # Integrate the robot kinematics forwards until it arrives
            # Compute the motion primitives as if the robot starts at x=0, y=0, theta=0
            # Then, we'll rotate the primitives into the robot frame when we want to use them
            #   calcualate an updated robot pose
            solution = solve_ivp(fun, (0, dt), np.zeros(3))  # [0,0,0] is robot "initial pose"
            final_pose = solution.y[:, -1]
            self.motion_primitives[i, :] = final_pose

        cost = velocity * dt
        self._costs = [cost for _ in range(self.n_primitives)]
        # Set the reverse motion primitive, if used, to have higher cost than other primitives
        if allow_reverse:
            reverse_cost = cost * 2.0
            self._costs[-1] = reverse_cost

    def get_cost(self, mp_idx):
        # return self.drive_distance
        return self._costs[mp_idx]

    def get_new_configurations(self, x: float, y: float, angle: int):
        """
        During the search, primitives are not recomputed from scratch. They are all stored w.r.t. a given reference frame.
        This function returns the correct new robots' positions from an arbitrary configuration, after having applied each
        primitive in motion_primitives (computed e.g. from the origin)

        NOTE: robot orientation should be specified in degrees

        Given an initial robot pose, calculate the new pose after executing each motion primitive.

        Parameters
        ----------
        x: float
            Robot initial position x-coordinate.
        y: float
            Robot initial position y-coordinate.
        angle: int
            Robot initial orientation angle, in degrees.

        Returns
        -------
        new_configs: array_like
            Array of shape (n_primitives, 3)
            Where new_configs[i, :] is the pose of the robot after executing primitive i
            Note the angle (new_configs[:, 2]) is in degrees.
        """
        assert type(angle) == int, "Angle must be an integer, specified in degrees"
        # Define rotation matrix for rotating the motion primitive into the robot frame
        angle_rad = math.radians(angle)
        R = np.zeros((3, 3), dtype=float)
        R[0, 0] = R[1, 1] = math.cos(angle_rad)
        R[1, 0] = math.sin(angle_rad)
        R[0, 1] = -math.sin(angle_rad)
        R[2, 2] = 1

        # Calculate robot configurations after executing each motion primitive
        new_configs = np.empty((self.n_primitives, 3))
        initial_pose = np.array([x, y, angle])
        for i, motion_primitive in enumerate(self.motion_primitives):
            new_configs[i, :] = R.dot(motion_primitive) + initial_pose

        return new_configs

    def get_motion_primitive_trajectories(self, x: float, y: float, angle: int, n_steps: int = 100):
        """
        Calculate the trajectories followed by the robot when executing each motion primitive,
        from a given starting pose.

        Use this if you want the points the robot travels to along each motion primitive,
        rather than just the endpoint, e.g. for plotting.

        Parameters
        ----------
        x: float
            Robot initial position x-coordinate.
        y: float
            Robot initial position y-coordinate.
        angle: int
            Robot initial orientation angle, in degrees.
        n_steps: int
            Number of steps to include in the trajectory
            More steps gives a finer-grain trajectory.

        Returns
        -------
        trajectories: array_like
            Array of shape (n_primitives, 3, n_steps)
            trajectories[i, :, :] gives the trajectory executed by the robot while following
            motion primitive i.
            rows [i, 0, :], [i, 1, :], and [i, 2, :] are the x, y, and angle history respectively.
        """
        trajectories = np.empty((self.n_primitives, 3, n_steps))
        dt = self.dt
        # v = self.drive_distance
        initial_pose = np.array([x, y, angle])
        for i, (velocity, angular_rate) in enumerate(zip(self.velocities, self.angular_rates)):

            def fun(t, y):  # Calculate change in x, y, angle
                angle = math.radians(y[2])
                return np.array([velocity * math.cos(angle), velocity * math.sin(angle), angular_rate])

            # Integrate the robot kinematics forwards until it arrives
            t_eval = np.linspace(0, dt, n_steps)
            solution = solve_ivp(fun, (0, dt), initial_pose, t_eval=t_eval)
            trajectories[i, :, :] = solution.y
        return trajectories


def euclidean_distance_heuristic(v, goal_position, x_cont, y_cont):
    """
    Calculate the Euclidean distance heuristic, for hybrid A* search

    Parameters
    ----------
    v: Vertex
        The vertex at which to evaluate the heuristic function
    goal_position: array_like
        Goal [x, y] position
    x_cont: VertexPropertyMap
    y_cont: VertexPropertyMap
        Property maps containing the continuous x- and y- coordinates associated with nodes
        in the A* search.

    Returns
    -------
    float
        The heuristic function value

    """
    # Get the current continuous position of the robot
    v_x = x_cont[v]
    v_y = y_cont[v]
    # Calculate Euclidean distance to the goal position
    return np.linalg.norm(goal_position - np.array([v_x, v_y]))


class HybridAStarVisitor(AStarVisitor):

    def __init__(
        self,
        graph: Graph,
        start_pose: ArrayLike,
        goal_position: ArrayLike,
        motion_primitives: MotionPrimitives,
        xy_resolution: float,
        angle_resolution: int,
        close_enough: float,
        far_enough: Optional[float],
        x_disc: VertexPropertyMap,
        y_disc: VertexPropertyMap,
        angle_disc: VertexPropertyMap,
        x_cont: VertexPropertyMap,
        y_cont: VertexPropertyMap,
        angle_cont: VertexPropertyMap,
        weight: EdgePropertyMap,
        dist: VertexPropertyMap,
        cost: VertexPropertyMap,
        edge_evaluator: ValidEdgeEvaluator,
        max_n_vertices: int = -1,
        timeout_n_vertices: int = -1,
    ):
        """

        Parameters
        ----------
        graph
        start_pose: array_like
            Angle should be in *radians*
        goal_position
        motion_primitives
        xy_resolution
        angle_resolution
        close_enough: float
        far_enough: float or None
        max_n_vertices: int
        x_disc
        y_disc
        angle_disc
        x_cont
        y_cont
        angle_cont
        weight
        dist
        cost
        edge_evaluator
        """
        self.graph: Graph = graph
        self.start_pose = start_pose
        self.start_position = start_pose[0:2]
        self.goal_position = goal_position
        self.motion_primitives = motion_primitives
        self.xy_resolution = xy_resolution
        self.angle_resolution = angle_resolution
        self.close_enough = close_enough
        self.far_enough = far_enough

        self.x_disc = x_disc
        self.y_disc = y_disc
        self.angle_disc = angle_disc

        self.x_cont = x_cont
        self.y_cont = y_cont
        self.angle_cont = angle_cont

        # for each edge in the graph, store which motion primitive is used to traverse along the edge
        self.motion_primitive_index = self.graph.new_edge_property("int")

        self.weight = weight
        self.dist = dist
        self.cost = cost

        self.edge_evaluator = edge_evaluator

        self.visited = {}

        self.success = False
        self.goal_vertex = None

        # Add the start cell as a vertex
        v0 = graph.add_vertex()
        xc, yc, ac = self.continuous_to_discrete(start_pose[0], start_pose[1], math.degrees(start_pose[2]))
        self.x_disc[v0] = xc
        self.y_disc[v0] = yc
        self.angle_disc[v0] = ac
        # Store start x, y as continuous point
        self.x_cont[v0] = start_pose[0]
        self.y_cont[v0] = start_pose[1]
        self.angle_cont[v0] = math.degrees(start_pose[2])

        self.start_vertex = v0

        self.n_vertices_finished = 0

        # search stops if this many vertices explored
        self.max_n_vertices = max_n_vertices

        # search stops if this many vertices explored without getting any closer to the goal
        self.timeout_n_vertices = timeout_n_vertices  #
        self._closest_distance_to_goal = float("inf")
        self._n_vertices_since_closer_to_goal = 0

    def continuous_to_discrete(self, x, y, angle):
        x_cell = math.floor(x / self.xy_resolution)
        y_cell = math.floor(y / self.xy_resolution)
        angle_cell = math.floor((angle % 360) / self.angle_resolution)
        return x_cell, y_cell, angle_cell

    def examine_vertex(self, u):
        current_cell = (self.x_disc[u], self.y_disc[u], self.angle_disc[u])

        # Get the current continuous pose of the robot at this cell
        x = self.x_cont[u]
        y = self.y_cont[u]
        # Get angle by multiplying discrete angle cell by angle resolution
        angle = self.angle_cont[u]
        # Calculate the robot's pose after following each of the possible motion primitives
        poses_after_motion = self.motion_primitives.get_new_configurations(x, y, int(angle))
        # Iterate over the motion primitives to find neighboring search vertices of this vertex
        for mp_idx in range(self.motion_primitives.n_primitives):
            neighbor_x, neighbor_y, neighbor_angle = poses_after_motion[mp_idx, :]
            # print("X:")
            # print(neighbor_x)
            # print("Y:")
            # print(neighbor_y)
            # print("THETA:")
            # print(neighbor_angle)
            # Determine which discrete grid cell the robot falls within, at this new pose
            neighbor_cell = self.continuous_to_discrete(neighbor_x, neighbor_y, neighbor_angle)
            # Print a warning if the motion primitive caused the search to stay in the same cell
            # This means the motion primitives should cover more distance.
            if current_cell == neighbor_cell:
                print(
                    "Warning: Search stayed in same discrete cell after motion. Increase motion primitive velocity or time."
                )
                continue

            # Check if edge from this vertex to the potential neighbor is valid
            # Depending on edge_evaluator, checks conditions such as obstacle collisions
            edge_valid, edge_cost_scale = self.edge_evaluator.check_edge([x, y], [neighbor_x, neighbor_y])
            # Edge is invalid (e.g. in collision with obstacle) - do not add this vertex
            if not edge_valid:
                continue

            if neighbor_cell in self.visited:
                v = self.visited[neighbor_cell]
            else:
                # Found a newly visited vertex
                v = self.graph.add_vertex()
                self.visited[neighbor_cell] = v
                # Save discrete and continuous coordinates for this vertex
                self.x_disc[v] = neighbor_cell[0]
                self.y_disc[v] = neighbor_cell[1]
                self.angle_disc[v] = int(neighbor_cell[2])
                self.x_cont[v] = neighbor_x
                self.y_cont[v] = neighbor_y
                self.angle_cont[v] = neighbor_angle
                self.dist[v] = self.cost[v] = float("inf")

            # Check if the graph already contains an edge from this vertex to the neighbor
            for e in u.out_edges():
                if e.target() == v:
                    break
            # For loop finished - so edge does not exist
            else:
                # No collision; create the edge
                e = self.graph.add_edge(u, v)
                # Path cost equals the distance covered executing a motion primitive
                # self.weight[e] = self.motion_primitives.drive_distance
                self.motion_primitive_index[e] = mp_idx
                w = self.motion_primitives.get_cost(mp_idx) * edge_cost_scale
                self.weight[e] = w
                # cells_difference = math.sqrt((neighbor_cell[0] - current_cell[0])**2 +
                #                              (neighbor_cell[1] - current_cell[1])**2)
                # self.weight[e] = cells_difference * self.xy_resolution
        self.visited[current_cell] = u

    def edge_relaxed(self, e):
        # Called when a new shorter path is found to a graph vertex

        # If we are planning in a local horizon (self.far_enough is set),
        #   check if we've found a path that covers enough distance away from the start
        x = self.x_cont[e.target()]
        y = self.y_cont[e.target()]
        if self.far_enough is not None and self.far_enough > 0:
            # if self.dist[e.target()] >= self.far_enough:
            #     self.success = True
            #     self.goal_vertex = e.target()
            #     raise StopSearch()
            distance_from_start = math.sqrt((x - self.start_position[0]) ** 2 + (y - self.start_position[1]) ** 2)
            if distance_from_start >= self.far_enough:
                self.success = True
                self.goal_vertex = e.target()
                raise StopSearch()

        # First check if we've found the goal
        # Check if the continuous coordinate for the edge target is close enough to the goal
        # If yes, search is done
        distance_to_goal = math.sqrt((x - self.goal_position[0]) ** 2 + (y - self.goal_position[1]) ** 2)
        if distance_to_goal < self.close_enough:
            self.success = True
            self.goal_vertex = e.target()
            raise StopSearch()

        # Edge does not lead to the goal - continue with method
        # Update the continuous coordinates of the graph cell
        x_source = self.x_cont[e.source()]
        y_source = self.y_cont[e.source()]
        angle_source = int(self.angle_cont[e.source()])
        mp_poses = self.motion_primitives.get_new_configurations(x_source, y_source, angle_source)

        # Look up the motion primitive associated with this edge
        target_pose = mp_poses[self.motion_primitive_index[e], :]

        target_cell = self.continuous_to_discrete(target_pose[0], target_pose[1], target_pose[2])

        # Update the discrete cell of the edge's target cell
        self.x_disc[e.target()] = target_cell[0]
        self.y_disc[e.target()] = target_cell[1]
        self.angle_disc[e.target()] = target_cell[2]

        # Get the continuous position of the edge target cell
        self.x_cont[e.target()] = target_pose[0]
        self.y_cont[e.target()] = target_pose[1]
        self.angle_cont[e.target()] = target_pose[2]

    def finish_vertex(self, u):
        # Called when the visitor is done exploring a vertex
        self.n_vertices_finished += 1
        # If the visitor has explored the maximum number of vertices, conclude the search
        x = self.x_cont[u]
        y = self.y_cont[u]
        distance_to_goal = math.sqrt((x - self.goal_position[0]) ** 2 + (y - self.goal_position[1]) ** 2)
        # Record if we are getting any closer to the goal
        # Search terminates if self.timeout_n_vertices is set, and we explore this many vertices
        #   without getting any closer to the goal
        if distance_to_goal < self._closest_distance_to_goal:
            self._closest_distance_to_goal = distance_to_goal
            self._n_vertices_since_closer_to_goal = 0
        else:
            self._n_vertices_since_closer_to_goal += 1
        if self.max_n_vertices > 0:  # max_n_vertices is set to -1 by default (disabled)
            if self.n_vertices_finished >= self.max_n_vertices:
                print("Terminating search: explored max of %d vertices" % self.max_n_vertices)
                raise StopSearch()
        if self.timeout_n_vertices > 0:
            if self._n_vertices_since_closer_to_goal >= self.timeout_n_vertices:
                print(
                    "Terminating search: explored %d vertices without getting closer to goal" % self.timeout_n_vertices
                )
                raise StopSearch()


class HybridAStarPlanner(object):

    def __init__(
        self,
        edge_evaluator: ValidEdgeEvaluator,
        xy_resolution: float = 0.1,
        angle_resolution: int = 60,
        close_enough: float = 0.1,
        far_enough: float = None,
        n_motion_primitives: int = 3,
        # motion_primitive_length: float = None,
        # max_steering_angle: int = 40,
        motion_primitive_dt: float = 0.2,
        motion_primitive_velocity: float = 2.0,
        max_angular_rate: int = 90,
        allow_reverse=True,
        n_attempts_reduced_resolution: int = 0,
        max_n_vertices: int = -1,
        timeout_n_vertices: int = -1,
    ):
        """

        Parameters
        ----------
        edge_evaluator: ValidEdgeEvaluator
            Evaluator to be used for checking valid edges during A* search.
            Use the CircularObstacleEdgeEvaluator class for basic Hybrid A-star with circular
            obstacles.
            See the ValidEdgeEvaluator class interface for info on how to implement ne
            edge evaluator subclasses.
        xy_resolution: float
        angle_resolution: int
            Angle discretization resolution, specified in degrees
        close_enough: float
            Robot is considered to have reached the goal when it is within this distnace.
        n_motion_primitives: int
        motion_primitive_length: float
            Distance the robot drives while executing a motion primitive.
            Should be greater than sqrt(2) * xy_resolution, to guarantee that the robot will leave
            its current discretization cell after executing a motion primitive.
        max_steering_angle: int
            Maximum angle the robot's orientation will change while executing one motion primitive.
        n_attempts_reduced_resolution: int
            If the initial planning fails, the planner will re-attempt planning this many times,
            halving the xy resolution each time (and adjusting the drive_distance accordingly).
            E.g. if n_attempts = 2, and xy_resolution = 0.2, the planner will make three attempts
            in total, with resolutions 0.2, 0.1, and 0.05.
            If far_enough parameter is set, each halved resolution will also halve the far_enough
            distance, to prevent spending too long on the path search.
        """
        assert (
            type(n_attempts_reduced_resolution) == int and n_attempts_reduced_resolution >= 0
        ), "n_attempts must be 0 or a positive integer"
        # if n_attempts_reduced_resolution > 0:
        #     assert motion_primitive_length is None, "For multiple-resolution planning attempts, " \
        #                                             "motion primitive length should be " \
        #                                             "chosen automatically (set to None)"
        self.n_attempts_reduced_resolution = n_attempts_reduced_resolution
        self.max_n_vertices = max_n_vertices
        self.timeout_n_vertices = timeout_n_vertices
        self.xy_resolution = xy_resolution
        # Check that angle discretization is valid
        assert 360 % angle_resolution == 0, "360 degrees must be divisible by angle resolution."
        # assert (max_steering_angle % angle_resolution == 0), "Max steering angle must be divisible " \
        #                                                      "by angle resolution."
        self.angle_resolution = angle_resolution  # angle resolution is in degrees
        self.n_angles = 360 // angle_resolution  # number of discrete angles to consider
        self.close_enough = close_enough
        self.far_enough = far_enough

        # Precompute the motion primitives to use for Hybrid A*
        # Angle changes are evenly spaced between - and + max steering angle
        # steering_angles = np.linspace(-max_steering_angle, max_steering_angle,
        #                               n_motion_primitives).astype(int)
        angular_rates = np.linspace(-max_angular_rate, max_angular_rate, n_motion_primitives).astype(int)

        # Show warnings if motion primitives are not valid
        if n_motion_primitives % 2 == 0:
            # If n_motion_primitives is odd, there will not be a theta=0 motion primitive
            # Print a warning to the user
            print("Warning: n_motion_primitives should be odd.")
        if 0 not in angular_rates:
            print("Warning: Angular rate 0 not included in motion primitives.")
        self.angular_rates = angular_rates
        self.allow_reverse = allow_reverse
        self.motion_primitive_velocity = motion_primitive_velocity
        self.motion_primitive_dt = motion_primitive_dt

        self.motion_primitives = MotionPrimitives(
            velocity=motion_primitive_velocity,
            dt=motion_primitive_dt,
            angular_rates=angular_rates,
            allow_reverse=allow_reverse,
        )

        self.edge_evaluator = edge_evaluator

    def find_path(self, start_pose, goal_position):
        """

        Parameters
        ----------
        start_pose: array_like
            Robot pose as [x, y, theta]
            Where theta is orientation *in radians*
            (will be converted to degrees internally in the planer)
        goal_position: array_like
            Goal position as [x, y]

        Returns
        -------
        HybridAStarResult

        """
        result = None
        attempt = 0

        # Halve xy resolution and dt on each attempt
        xy_resolution_attempt = self.xy_resolution
        dt_attempt = self.motion_primitive_dt

        while attempt <= self.n_attempts_reduced_resolution and result is None:
            # Attempt 0: use default motion primitives
            if attempt == 0:
                motion_primitives_attempt = self.motion_primitives
            # Other attempts: use motion primitives with reduced resolution
            else:
                print("Attempt %d: Planning with resolution %.3f" % (attempt + 1, xy_resolution_attempt))
                motion_primitives_attempt = MotionPrimitives(
                    velocity=self.motion_primitive_velocity,
                    dt=dt_attempt,
                    angular_rates=self.angular_rates,
                    allow_reverse=self.allow_reverse,
                )
            # Run the A* search and return a series of XY waypoints that the robot can follow to the goal
            graph = Graph()

            # Initialize graph properties for A* search
            weight = graph.new_edge_property("double")
            dist = graph.new_vertex_property("double")
            cost = graph.new_vertex_property("double")

            # Property maps for storing the discrete cell index of each x, y, angle cell
            x_disc = graph.new_vertex_property("int")
            y_disc = graph.new_vertex_property("int")
            angle_disc = graph.new_vertex_property("int")

            # Property maps for storing the continuous x, y, angle coordinates associated with each cell
            x_cont = graph.new_vertex_property("float")
            y_cont = graph.new_vertex_property("float")
            angle_cont = graph.new_vertex_property("float")

            visitor = HybridAStarVisitor(
                graph=graph,
                start_pose=start_pose,
                goal_position=goal_position,
                motion_primitives=motion_primitives_attempt,
                xy_resolution=xy_resolution_attempt,
                angle_resolution=self.angle_resolution,
                close_enough=self.close_enough,
                far_enough=self.far_enough,
                x_disc=x_disc,
                y_disc=y_disc,
                angle_disc=angle_disc,
                x_cont=x_cont,
                y_cont=y_cont,
                angle_cont=angle_cont,
                weight=weight,
                dist=dist,
                cost=cost,
                edge_evaluator=self.edge_evaluator,
                max_n_vertices=self.max_n_vertices,
                timeout_n_vertices=self.timeout_n_vertices,
            )

            h = lambda v: euclidean_distance_heuristic(v, goal_position=goal_position, x_cont=x_cont, y_cont=y_cont)

            # Run the search
            dist, pred = astar_search(
                graph,
                visitor.start_vertex,
                weight=weight,
                visitor=visitor,
                dist_map=dist,
                cost_map=cost,
                heuristic=h,
                implicit=True,
            )
            # Save graph properties, so they are stored in the HybridAStarResult
            graph.vertex_properties["x_disc"] = x_disc
            graph.vertex_properties["y_disc"] = y_disc
            graph.vertex_properties["angle_disc"] = angle_disc
            graph.vertex_properties["x_cont"] = x_cont
            graph.vertex_properties["y_cont"] = y_cont
            graph.vertex_properties["angle_cont"] = angle_cont
            graph.edge_properties["weight"] = weight
            graph.edge_properties["motion_primitive_index"] = visitor.motion_primitive_index
            if visitor.success:
                # Found a path, create the Result object
                result = HybridAStarResult(
                    graph,
                    dist,
                    pred,
                    start_vertex=visitor.start_vertex,
                    goal_vertex=visitor.goal_vertex,
                    motion_primitives=motion_primitives_attempt,
                )
            attempt += 1
            xy_resolution_attempt /= 2.0
            dt_attempt /= 2.0

        return result


class HybridAStarResult(object):

    def __init__(self, graph, dist, pred, start_vertex, goal_vertex, motion_primitives):
        self.graph: Graph = graph
        self.dist: VertexPropertyMap = dist
        self.pred: VertexPropertyMap = pred
        self.start_vertex: int = start_vertex
        self.goal_vertex: int = goal_vertex
        self.motion_primitives: MotionPrimitives = motion_primitives

    def _get_vertices(self):
        # Get the list of vertices along the path, from start to goal
        vertices = []
        current_vertex = self.goal_vertex
        while current_vertex != self.start_vertex:
            vertices.append(int(current_vertex))
            current_vertex = self.pred[current_vertex]
        vertices.append(int(self.start_vertex))
        vertices.reverse()  # make list go from start -> goal
        return vertices

    def poses(self, points_per_motion_primitive: int = 5):
        """

        Parameters
        ----------
        points_per_motion_primitive

        Returns
        -------
        array_like
            N by 3 array of x, y, theta poses

        """
        vertices = self._get_vertices()

        # Populate waypoints using the motion primitives along the path and output the xy waypoints.

        # initialize pose array
        # there is one motion primitive between each pair of vertices
        n_waypoints = (len(vertices) - 1) * points_per_motion_primitive
        pose_array = np.empty((n_waypoints, 3), dtype=float)

        for vertex_idx in range(len(vertices) - 1):
            v_current = vertices[vertex_idx]
            v_next = vertices[vertex_idx + 1]
            x = self.graph.vp.x_cont[v_current]
            y = self.graph.vp.y_cont[v_current]
            theta = self.graph.vp.angle_cont[v_current]
            # When calculating motion primitives, calculate n_points+1 steps,
            # since the last point will overlap with the next motion primitive start point
            mp_array = self.motion_primitives.get_motion_primitive_trajectories(
                x, y, theta, n_steps=points_per_motion_primitive + 1
            )
            # Remove the last point
            mp_array = mp_array[:, :, 0:-1]
            e = self.graph.edge(v_current, v_next)
            mp_idx = self.graph.ep.motion_primitive_index[e]
            mp_x = mp_array[mp_idx, 0, :]
            mp_y = mp_array[mp_idx, 1, :]
            mp_theta = np.deg2rad(mp_array[mp_idx, 2, :])
            # index of the first waypoint for this motion primitive, in the waypoints array
            wp_start = vertex_idx * points_per_motion_primitive
            pose_array[wp_start : wp_start + points_per_motion_primitive, 0] = mp_x
            pose_array[wp_start : wp_start + points_per_motion_primitive, 1] = mp_y
            pose_array[wp_start : wp_start + points_per_motion_primitive, 2] = mp_theta
        return pose_array

    def get_path_length(self):
        vertices = self._get_vertices()
        # robot moves a fixed distance between each pair of vertices, based on the motion primitives
        return self.motion_primitives.dt * self.motion_primitives.velocities[0] * (len(vertices) - 1)

    def get_search_tree_line_segments(self, points_per_motion_primitive: int = 5) -> ArrayLike:
        # Color edges
        in_path = self.graph.new_edge_property("bool", val=False)

        # Find which edges were on the final path
        current_vertex = self.goal_vertex
        while current_vertex != self.start_vertex:
            edge = self.graph.edge(self.pred[current_vertex], current_vertex)
            in_path[edge] = True
            current_vertex = self.pred[current_vertex]

        line_segments = []
        line_segment_in_path = []

        # Iterate over all vertices in search tree
        for v in self.graph.vertices():
            # Get continuous pose for this node
            vx = self.graph.vp.x_cont[v]
            vy = self.graph.vp.y_cont[v]
            vtheta = self.graph.vp.angle_cont[v]
            # Calculate motion primitives
            mp = self.motion_primitives.get_motion_primitive_trajectories(vx, vy, vtheta, n_steps=5)
            # For each edge out of this node, plot the motion primitive trajectories
            for e in v.out_edges():
                mp_idx = self.graph.ep.motion_primitive_index[e]
                if in_path[e]:
                    line_segment_in_path.append(True)
                else:
                    line_segment_in_path.append(False)
                line_segments.append(mp[mp_idx, 0:2, :].T)
        return line_segments, line_segment_in_path

    def waypoints(self, points_per_motion_primitive: int = 5):
        return self.poses(points_per_motion_primitive)[:, 0:2]

    def plot(self, ax=None, show_search_tree=False, edge_color=(0.1, 0.1, 0.8), path_color=(0.4, 1.0, 0.4)):
        """

        Parameters
        ----------
        ax: plt.Axes
        show_search_tree: bool
            If True, the whole Hybrid A* search tree will be plotted.
            If False, only the final path will be shown.


        Returns
        -------

        """
        if ax is None:
            ax = plt.subplot(1, 1, 1)

        # Color edges
        in_path = self.graph.new_edge_property("bool", val=False)

        # Find which edges were on the final path
        current_vertex = self.goal_vertex
        while current_vertex != self.start_vertex:
            edge = self.graph.edge(self.pred[current_vertex], current_vertex)
            in_path[edge] = True
            current_vertex = self.pred[current_vertex]

        # Iterate over all vertices in search tree
        for v in self.graph.vertices():
            # Get continuous pose for this node
            vx = self.graph.vp.x_cont[v]
            vy = self.graph.vp.y_cont[v]
            vtheta = self.graph.vp.angle_cont[v]
            # Calculate motion primitives
            mp = self.motion_primitives.get_motion_primitive_trajectories(vx, vy, vtheta, n_steps=5)
            # For each edge out of this node, plot the motion primitive trajectories
            for e in v.out_edges():
                mp_idx = self.graph.ep.motion_primitive_index[e]
                if in_path[e]:
                    c = path_color
                    z = 2
                else:
                    c = edge_color
                    z = 1
                if (not show_search_tree and in_path[e]) or show_search_tree:
                    ax.plot(mp[mp_idx, 0, :], mp[mp_idx, 1, :], color=c, zorder=z)

    @property
    def n_examined(self):
        return self.graph.num_vertices()
