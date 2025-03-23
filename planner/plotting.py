"""
Brian H. Wang, bhw45@cornell.edu

Plotting functions
"""

from typing import Optional, List, Tuple
from numpy.typing import ArrayLike

import math
from math import floor, ceil
import numpy as np
import numba
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.cm import get_cmap
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import cmasher as cmr

from planner.simulation import Simulation
from planner.probabilistic_obstacles import Obstacle2D
from planner.cell_decomposition import SafetyCellDecomposition, RANGE_SHORT, RANGE_MEDIUM, RANGE_LONG
from planner.navigation_graphs import NavigationGraph
from planner.navigation_utils import NavigationPath
from planner.multiple_hypothesis_planning import MultipleHypothesisPlanningResult
from planner.hybrid_astar import HybridAStarResult
from graph_tool import VertexPropertyMap

import seaborn as sns

sns.set()
sns.set_style("dark")


def covariance_ellipse(
    mean, cov, ax, n_std=3.0, extra_radius=0.0, facecolor="none", ellipse: patches.Ellipse = None, **kwargs
):
    """
    Adapted from the confidence ellipse example at:
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mean : array-like, shape (2, )
        Mean; the centroid of the ellipse.

    cov: array-like, shape (2, 2)
        Covariance matrix; defines the size and orientation of the ellipse.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    extra_radius : float
        Extra amount that should be added to the ellipse's radius.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if mean.size != cov.shape[0] and mean.size != cov.shape[1]:
        raise ValueError("Mean and covariance must have same number of dimensions.")

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    if ellipse is None:
        ellipse = patches.Ellipse(
            (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs
        )
        new_artist = True
    else:
        ellipse.set_width(ell_radius_x * 2)
        ellipse.set_height(ell_radius_y * 2)
        new_artist = False

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    if new_artist:
        return ax.add_patch(ellipse)
    else:
        return ellipse


class Plotter(object):

    # Palettes
    # Blues: https://coolors.co/012a4a-013a63-01497c-014f86-2a6f97-2c7da0-468faf-61a5c2-89c2d9-a9d6e5
    # Reds: https://coolors.co/641220-85182a-a71e34-bd1f36-c71f37-e43a50-f093a0

    # Colors for the obstacle circles
    # dark_blue = "#02223C"
    dark_blue = "#011728"
    med_blue = "#01497C"
    med_light_blue = "#61A5C2"
    light_blue = "#A9D6E5"

    med_red = "#C71F37"
    light_red = "#E43A50"
    light_pink = "#F3A5B0"

    light_green = "#8AC482"
    med_green = "#519948"

    light_purple = "#D8C5E8"

    def __init__(self, ax=None):
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        self.ax = ax


class PlotRobot(Plotter):

    def __init__(self, ax, position, orientation=0.0, width=1.0, zorder=0, color=(0.4, 0.4, 0.8), linewidth=1.0):
        super().__init__(ax)
        self.radius = width / 2
        self.patch_circle = patches.Circle(position, radius=self.radius, zorder=zorder, color=color)

        rect_height = width / 4.0
        rect_width = (width / 2.0) * 1.4  # how much the line should stick out the front of the robot
        rect_xy = position
        R = np.array([[math.cos(orientation), math.sin(orientation)], [-math.sin(orientation), math.cos(orientation)]])
        rect_xy = position + (R.dot(np.array([0, -rect_height / 2.0]).reshape((2, 1)))).flatten()
        # self.patch_rectangle = patches.Rectangle(rect_xy, width=rect_width, height=rect_height,
        #                                          angle=math.degrees(orientation),
        #                                          zorder=zorder, color=color)
        ax.add_patch(self.patch_circle)
        # ax.add_patch(self.patch_rectangle)
        self.ax = ax
        self.rect_height = rect_height
        self.rect_width = rect_width
        # Show orientation by drawing a line from robot center to robot front
        xs, ys = self._orientation_line(position, orientation)
        (self.line_handle,) = ax.plot(
            xs, ys, "-", linewidth=linewidth, color=self.dark_blue, solid_capstyle="projecting", zorder=zorder + 1
        )

    # def set_position(self, position):
    #     self.patch_circle.set_center(position)
    #     self.patch_rectangle.set_xy()

    def set_pose(self, position, orientation):
        xs, ys = self._orientation_line(position, orientation)
        self.line_handle.set_data(xs, ys)
        # self.set_position(position)
        self.patch_circle.set_center(position)
        R = np.array([[math.cos(orientation), math.sin(orientation)], [-math.sin(orientation), math.cos(orientation)]])
        rect_xy = position + (R.dot(np.array([0, -self.rect_height / 2.0]).reshape((2, 1)))).flatten()
        # self.patch_rectangle.angle = math.degrees(orientation)
        # self.patch_rectangle.set_xy(rect_xy)

    def _orientation_line(self, position, orientation):
        ls = 1.4  # line scale (higher this is the more the line sticks out the front of the robot)
        xs = [position[0], position[0] + ls * self.radius * math.cos(orientation)]
        ys = [position[1], position[1] + ls * self.radius * math.sin(orientation)]
        return xs, ys


class ObstaclePlotter(Plotter):
    """
    Plot a 2D circular obstacle without uncertainty (show mean only)
    """

    def __init__(self, obstacle: Obstacle2D, ax: plt.Axes = None, color=None, label: Optional[str] = None):
        super().__init__(ax)

        ## Plot the mean
        position_mean = obstacle.pos_mean
        radius_mean = obstacle.size_mean / 2.0  # diameter to radius
        if color is None:
            color = self.dark_blue
        circle_mean = patches.Circle(position_mean, radius=radius_mean, color=color, zorder=4)
        ax.add_patch(circle_mean)

        if label is not None:
            label_handle = ax.text(
                position_mean[0], position_mean[1], str(label), ha="center", va="center", c=(0.8, 0.8, 0.9), zorder=10
            )
        else:
            label_handle = None

        self.circle_mean: patches.Circle = circle_mean
        self.label = label_handle

    def update(self, obstacle: Obstacle2D, color=None):
        position_mean = obstacle.pos_mean
        radius_mean = obstacle.size_mean / 2.0  # diameter to radius
        self.circle_mean.set_center(position_mean)
        self.circle_mean.set_radius(radius_mean)
        if self.label is not None:
            self.label.set_position(position_mean)
        if color is not None:
            self.circle_mean.set_color(color)


class ObstacleEstimatePlotter(ObstaclePlotter):
    """
    Plot obstacle mean and position, size uncertainty.
    """

    def __init__(
        self,
        obstacle: Obstacle2D,
        n_std_devs: int = 2,
        ax: plt.Axes = None,
        label=None,
        color=None,
        show_size_uncertainty=True,
    ):
        """

        Parameters
        ----------
        obstacle: Obstacle2D
            Obstacle object containing information on the object position and size estimates.
        n_std_devs: int
            Number of standard deviations of the position and size uncertainty that should be plotted
        ax: plt.Axes
        label: str
            Text label that will be plotted over the obstacle mean.
            Can be used to number different obstacles in the plot.
        """

        super().__init__(obstacle=obstacle, ax=ax, label=label, color=color)

        ## Plot the mean
        #
        position_mean = obstacle.pos_mean
        position_cov = obstacle.pos_cov
        radius_mean = obstacle.size_mean / 2.0  # diameter to radius
        radius_std = math.sqrt(obstacle.size_var) / 2.0
        circle_mean = patches.Circle(position_mean, radius=radius_mean, color=self.dark_blue, zorder=4)

        ## Plot the size uncertainty
        # Show two circles, with radii equal to the mean plus/minus N radius std devs
        if n_std_devs > 0 and show_size_uncertainty:
            circle_small = patches.Circle(
                position_mean,
                radius=radius_mean - n_std_devs * radius_std,
                edgecolor=self.med_light_blue,
                fill=False,
                linestyle="--",
                zorder=5,
            )
            circle_large = patches.Circle(
                position_mean,
                radius=radius_mean + n_std_devs * radius_std,
                edgecolor=self.med_light_blue,
                facecolor=self.light_blue,
                linestyle="--",
                zorder=3,
            )
            ax.add_patch(circle_small)
            ax.add_patch(circle_large)
        else:
            circle_small = circle_large = None

        ## Plot the position uncertainty
        # Show a light colored ellipse encompassing N std devs of the X and Y position uncertainty
        # Also show a dotted line, and lighter colored region, of the ellipse plus radius mean,
        #   plus N radius std devs
        if n_std_devs > 0:
            ellipse_xy = covariance_ellipse(
                position_mean,
                position_cov,
                ax,
                n_std=n_std_devs,
                facecolor=self.light_pink,
                zorder=5,
                edgecolor=self.med_red,
                linewidth=0.5,
                alpha=0.5,
            )
        else:
            ellipse_xy = None

        ellipse_xy_plus_size = None

        self.n_std_devs = n_std_devs
        self.circle_small: patches.Circle = circle_small
        self.circle_large: patches.Circle = circle_large
        self.ellipse_xy: patches.Ellipse = ellipse_xy
        self.ellipse_xy_plus_size: patches.Ellipse = ellipse_xy_plus_size

    def update(self, obstacle: Obstacle2D, color=None):
        super().update(obstacle, color=color)
        position_mean = obstacle.pos_mean
        radius_mean = obstacle.size_mean / 2.0  # diameter to radius
        radius_std = math.sqrt(obstacle.size_var) / 2.0
        if self.circle_small is not None and self.circle_large is not None:
            for circle, r in zip(
                [self.circle_small, self.circle_large],
                [radius_mean - self.n_std_devs * radius_std, radius_mean + self.n_std_devs * radius_std],
            ):
                circle.set_center(position_mean)
                circle.set_radius(r)

        if self.ellipse_xy is not None:
            position_cov = obstacle.pos_cov
            self.ellipse_xy = covariance_ellipse(
                position_mean, position_cov, self.ax, ellipse=self.ellipse_xy, n_std=self.n_std_devs
            )


class BloatedObstacleEstimatePlotter(ObstaclePlotter):
    """
    Plot obstacle mean and position, size uncertainty.
    """

    def __init__(self, obstacle: Obstacle2D, n_std_devs: int, ax: plt.Axes = None, label=None, color=None):
        """

        Parameters
        ----------
        obstacle: Obstacle2D
            Obstacle object containing information on the object position and size estimates.
        n_std_devs: int
            Number of standard deviations of the position and size uncertainty that should be plotted
        ax: plt.Axes
        label: str
            Text label that will be plotted over the obstacle mean.
            Can be used to number different obstacles in the plot.
        """

        # Plot the obstacle
        super().__init__(obstacle=obstacle, ax=ax, label=label, color=color)

        ## Plot the bloated area
        #
        position_mean = obstacle.pos_mean
        position_cov = obstacle.pos_cov
        radius_mean = obstacle.size_mean / 2.0  # diameter to radius
        radius_std = math.sqrt(obstacle.size_var) / 2.0

        # sqrt(x_std^2 + y_std^2) gives the diameter uncertainty, need to divide by 2
        position_std = math.sqrt(position_cov[0, 0] + position_cov[1, 1]) / 2.0
        bloat_radius = radius_mean + n_std_devs * (position_std + radius_std)

        circle_mean = patches.Circle(position_mean, radius=bloat_radius, color=self.dark_blue, zorder=4, alpha=0.5)
        ax.add_patch(circle_mean)


class CellDecompositionPlotter(Plotter):

    color_range_short = Plotter.med_blue
    color_range_medium = Plotter.med_light_blue
    color_range_long = Plotter.light_blue

    def __init__(
        self,
        cell_decomposition: SafetyCellDecomposition,
        gray: bool = False,
        show_labels: bool = False,
        show_uncertainty: bool = False,
        ax: plt.Axes = None,
        show_colorbar=False,
    ):
        super().__init__(ax)

        # Create custom colormap
        #   BELOW INFO DEPRECATED - USING DEFAULT CIVIDIS COLORS FOR NOW
        # Use viridis colormap, except the top third of colors (yellow colors)
        # This creates a colormap where safest edges are green
        # viridis = cm.get_cmap('viridis')
        # self.cmap = ListedColormap(colors=viridis.colors[0:167], name="viridis_partial")
        # self.cmap = cm.get_cmap("cividis")

        # Green (safest) to dark red (unsafest) colormap
        # self.cmap = cmr.get_sub_cmap(cmr.apple, 0.3, 0.8, N=4)
        self.cmap = cmr.get_sub_cmap(cmr.freeze, 0.0, 0.9)

        obstacles = cell_decomposition.obstacles
        delaunay = cell_decomposition.delaunay

        line_handles = []

        # Plot the obstacles
        self.obstacle_plotters = {}  # dict mapping obstacle index to plotter object
        self._show_labels = show_labels
        self._show_uncertainty = show_uncertainty
        self._gray = gray
        for obs_idx in range(len(obstacles)):
            # Get position and convert it to a numpy array
            obstacle = obstacles[obs_idx]
            # TODO plot obstacles in different colors based on range
            self._add_obstacle(obstacle, obs_idx, range_zone=cell_decomposition.get_vertex_range(obs_idx))

        # Show cell numbers

        self.cell_labels = {}
        # TODO cell labels not shown currently
        # if show_labels:
        #     for cell_idx in range(cell_decomposition.n_cells):
        #         self._add_cell_label(cell_decomposition, cell_idx)

        # Plot Delauanay graph edges, colored by safety
        line_segments, colors = self._get_lines_from_cell_decomposition(cell_decomposition)
        lc = LineCollection(line_segments, colors=colors, linewidths=1.6)
        ax.add_collection(lc)
        self.lines = lc

        self.path_polygon = None
        if show_colorbar:
            # colorbar position adjustment reference:
            # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right",
                size="5%",
                pad=0.25,
            )
            colorbar = plt.colorbar(
                cm.ScalarMappable(cmap=self.cmap), cax=cax, label="Cell edge safety probability", ticks=[0, 1]
            )
            colorbar.ax.set_yticklabels([0.0, 1.0])
            colorbar.ax.yaxis.set_label_position("left")
            colorbar.ax.yaxis.set_ticks_position("right")
            self.colorbar = colorbar
        else:
            self.colorbar = None

    def update(self, cell_decomposition: SafetyCellDecomposition):
        # Update obstacles
        for obs_idx, obstacle in enumerate(cell_decomposition.obstacles):
            obstacle_plotter: Optional[ObstaclePlotter] = self.obstacle_plotters.setdefault(obs_idx, None)
            range_zone = cell_decomposition.get_vertex_range(obs_idx)
            if obstacle_plotter is None:
                self._add_obstacle(obstacle, obs_idx, range_zone=range_zone)
            else:
                if not self._gray:
                    if range_zone == RANGE_MEDIUM:
                        obs_color = self.color_range_medium
                    elif range_zone == RANGE_LONG:
                        obs_color = self.color_range_long
                    else:
                        obs_color = self.color_range_short
                else:
                    color = None
                obstacle_plotter.update(obstacle, color=obs_color)

        # Update cell numbers
        if self._show_labels:
            for cell_idx in range(cell_decomposition.n_cells):
                cell_label = self.cell_labels.setdefault(cell_idx, None)
                if cell_label is None:
                    self._add_cell_label(cell_decomposition, cell_idx)
                else:
                    vertices = cell_decomposition.get_cell_vertices(cell_idx)
                    position = np.mean(vertices, axis=0)
                    self.cell_labels[cell_idx].set_position(position)

        # Update cell edge lines
        line_segments, colors = self._get_lines_from_cell_decomposition(cell_decomposition)
        self.lines.set_segments(line_segments)
        self.lines.set_color(colors)

    def plot_navigation_path(self, path: Optional[NavigationPath]):
        if path is None:
            if self.path_polygon is not None:
                self.path_polygon.set_xy(np.empty((0, 2)))
        else:
            polygon_vertices = path.get_polygon()
            if self.path_polygon is None:
                polygon = patches.Polygon(polygon_vertices, alpha=0.7, color=self.light_blue, zorder=0)
                self.ax.add_patch(polygon)
                self.path_polygon = polygon
            else:
                self.path_polygon.set_xy(polygon_vertices)

    def _add_obstacle(self, obstacle, obs_idx, range_zone: int = RANGE_SHORT):
        if self._show_labels:
            obs_label = str(obs_idx)
        else:
            obs_label = None
        if self._gray:
            obs_color = (0.65, 0.65, 0.65)
        else:
            if range_zone == RANGE_MEDIUM:
                obs_color = self.color_range_medium
            elif range_zone == RANGE_LONG:
                obs_color = self.color_range_long
            else:
                obs_color = self.color_range_short
        if self._show_uncertainty:
            self.obstacle_plotters[obs_idx] = ObstacleEstimatePlotter(
                obstacle, ax=self.ax, label=obs_label, color=obs_color
            )
        else:
            self.obstacle_plotters[obs_idx] = ObstaclePlotter(obstacle, ax=self.ax, label=obs_label, color=obs_color)

    def _add_cell_label(self, cell_decomposition, cell_idx):
        vertices = cell_decomposition.get_cell_vertices(cell_idx)
        position = np.mean(vertices, axis=0)
        cell_label_handle = self.ax.text(
            position[0], position[1], str(cell_idx), ha="center", va="center", c=(0.6, 0.6, 0.9), zorder=10
        )
        self.cell_labels[cell_idx] = cell_label_handle

    def _get_lines_from_cell_decomposition(self, cell_decomposition: SafetyCellDecomposition):
        line_segments = []
        colors = []
        for v1, v2 in cell_decomposition.delaunay_edges:
            # for edge in self.delaunay_edges:
            prob_safe = cell_decomposition.get_edge_safety(v1, v2)
            range_zone = min(cell_decomposition.get_vertex_range(v1), cell_decomposition.get_vertex_range(v2))
            RED = [0.9, 0.3, 0.25]
            YELLOW = [0.94, 0.77, 0.06]
            GREEN = [0.56, 0.86, 0.60]
            safety_prob = cell_decomposition.safety_probability.probability
            if self._gray:
                color = [0.75, 0.75, 0.75]
            else:
                if range_zone == RANGE_SHORT:
                    if prob_safe < safety_prob:
                        color = RED
                    else:
                        color = GREEN
                else:
                    if prob_safe < safety_prob:
                        color = YELLOW
                    else:
                        color = GREEN
            line_segments.append(cell_decomposition.delaunay.points[[v1, v2], :])
            colors.append(color)
        return line_segments, colors


class DistanceGraphPlotter(Plotter):

    def __init__(
        self,
        distance_graph: NavigationGraph,
        ax=None,
        vertex_excluded: VertexPropertyMap = None,
        show_start_and_goal=True,
    ):
        super().__init__(ax=ax)

        cell_vertex_size = 12
        face_vertex_size = 4
        x_size = 8

        vertex_color = (0.6, 0.2, 0.2)
        x_color = "#f20fd0"
        # start_color = (0.4, 0.8, 0.4)
        start_color = (0.8, 0.4, 0.4)
        goal_color = start_color
        edge_color = (0.4, 0.4, 0.4)

        graph = distance_graph.graph
        vertex_position = graph.vp["position"]
        edge_weights = graph.ep["weight"]

        # Construct arrays of the x- and y- positions of all vertices
        # Cell and face vertices should be plotted with different sizes/colors to distinguish

        # Plot the vertices
        face_x, face_y = self._get_vertex_xy(distance_graph)
        (vertices_handle,) = self.ax.plot(face_x, face_y, ".", c=vertex_color, markersize=face_vertex_size, zorder=5)

        if vertex_excluded is not None:
            exc_x, exc_y = self._get_vertex_xy(distance_graph, which_vertices=vertex_excluded)
        else:
            exc_x = []
            exc_y = []
        (excluded_handle,) = self.ax.plot(exc_x, exc_y, "x", c=x_color, markersize=x_size, zorder=6, markeredgewidth=2)

        # Plot the start and goal
        if show_start_and_goal:
            start_position = vertex_position[distance_graph.start_vertex].a
            goal_position = vertex_position[distance_graph.goal_vertex].a
            (start_handle,) = self.ax.plot(
                start_position[0],
                start_position[1],
                "^",
                c=start_color,
                markersize=cell_vertex_size,
                zorder=11,
            )
            (goal_handle,) = self.ax.plot(
                goal_position[0], goal_position[1], "*", c=goal_color, markersize=cell_vertex_size, zorder=11
            )
        else:
            start_handle = None
            goal_handle = None

        self.show_start_and_goal = show_start_and_goal

        # Plot all edges as a line collection
        line_segments = self._get_line_segments(distance_graph)
        lc = LineCollection(line_segments, colors=edge_color, linewidths=0.2, zorder=4)
        ax.add_collection(lc)

        self.vertices_handle = vertices_handle
        self.excluded_handle = excluded_handle
        self.start_handle = start_handle
        self.goal_handle = goal_handle
        self.lines = lc

        # self.cell_decomposition.plot(ax=ax, gray=True, show_labels=False)
        # CellDecompositionPlotter(self.cell_decomposition, ax=ax)

    def update(self, distance_graph: NavigationGraph, vertex_excluded: VertexPropertyMap = None):
        # Update vertex locations
        vertex_x, vertex_y = self._get_vertex_xy(distance_graph)
        self.vertices_handle.set_data(vertex_x, vertex_y)
        if vertex_excluded is not None:
            exc_x, exc_y = self._get_vertex_xy(distance_graph, which_vertices=vertex_excluded)
            self.excluded_handle.set_data(exc_x, exc_y)
        # Update start and goal locations
        vertex_position = distance_graph.graph.vp["position"]
        start_position = vertex_position[distance_graph.start_vertex].a
        goal_position = vertex_position[distance_graph.goal_vertex].a
        if self.start_handle is not None:
            self.start_handle.set_data(start_position[0], start_position[1])
        if self.goal_handle is not None:
            self.goal_handle.set_data(goal_position[0], goal_position[1])
        # Update graph edge line segments
        line_segments = self._get_line_segments(distance_graph)
        self.lines.set_segments(line_segments)

    def _get_vertex_xy(self, distance_graph: NavigationGraph, which_vertices=None):
        """

        Parameters
        ----------
        distance_graph
        which_vertices: VertexPropertyMap
            Boolean value vertex property map.
            If this argument is provided, gives the xy coordinates only of those vertices whose
            value in this property map is True

        Returns
        -------
        x: List[float]
        y: List[float]

        """
        graph = distance_graph.graph
        vertex_position = graph.vp["position"]
        vertex_x = []
        vertex_y = []
        for v in graph.vertices():
            if which_vertices is None or which_vertices[v]:
                if not v == distance_graph.start_vertex and not v == distance_graph.goal_vertex:
                    p = vertex_position[v].a
                    vertex_x.append(p[0])
                    vertex_y.append(p[1])
        return vertex_x, vertex_y

    def _get_line_segments(self, distance_graph: NavigationGraph):
        graph = distance_graph.graph
        vertex_position = graph.vp["position"]
        line_segments = []
        for e in graph.edges():
            if not self.show_start_and_goal:
                s = int(e.source())
                t = int(e.target())
                if (
                    s == distance_graph.start_vertex
                    or t == distance_graph.start_vertex
                    or s == distance_graph.goal_vertex
                    or t == distance_graph.goal_vertex
                ):
                    continue
            p1 = vertex_position[e.source()].a
            p2 = vertex_position[e.target()].a
            line_segments.append([p1, p2])
        return line_segments


class LocalPlannerPathPlotter(Plotter):

    def __init__(self, ax=None):
        super().__init__(ax=ax)
        self.waypoints_handle = None
        self.search_tree = None

    def plot_waypoints(self, waypoints):
        if self.waypoints_handle is None:
            self.waypoints_handle = self.ax.plot(
                waypoints[:, 0], waypoints[:, 1], "-", linewidth=1.5, color=self.med_blue, zorder=10, markersize=2.0
            )[0]
        else:
            self.waypoints_handle.set_data(waypoints[:, 0], waypoints[:, 1])

    def plot_search_tree(self, astar_result: HybridAStarResult):
        line_segments, line_segment_in_path = astar_result.get_search_tree_line_segments()
        if self.search_tree is None:
            lc = LineCollection(line_segments, colors=self.med_blue, linewidth=0.5, alpha=0.5)
            self.search_tree = self.ax.add_collection(lc)
        else:
            self.search_tree.set_segments(line_segments)


class BaselineHighLevelPathPlotter(Plotter):
    """
    Class for plotting the high-level path generated by 2D A* search,
    in the baseline planner implementation.
    """

    def __init__(self, points: ArrayLike, ax=None):
        super().__init__(ax=ax)

        lw = 2.0
        (self.path_handle,) = self.ax.plot(
            [], ".-", lw=lw, markersize=4, color=Plotter.med_light_blue, label="2D global planner path", zorder=5
        )

        self.update(points)
        self.ax.legend(loc="upper right")

    def update(self, points: ArrayLike):
        if points is None:
            points = np.empty((0, 2))
        self.path_handle.set_data(points[:, 0], points[:, 1])


class HighLevelPathPlotter(Plotter):

    # Min and max line width
    # Line width depends on the path's cost, relative to the other paths
    lw_min = 3.0
    lw_max = 4.0

    def __init__(
        self,
        mhp_result: MultipleHypothesisPlanningResult,
        ax=None,
        cmap="plasma",
        n_paths: int = 1,
        show_cell_decomposition=True,
        colors=None,
        lw: Optional[float] = None,
        no_legend=False,
    ):
        super().__init__(ax=ax)

        self.cmap = cm.get_cmap(cmap)

        # Plot the cell decomposition
        if show_cell_decomposition:
            self.cell_decomposition_plotter = CellDecompositionPlotter(
                mhp_result.cell_decomposition, show_uncertainty=False, show_labels=False, ax=self.ax
            )
        else:
            self.cell_decomposition_plotter = None

        # plt.colorbar(cm.ScalarMappable(cmap=self.cmap), ax=ax, shrink=0.7)

        self.n_paths = n_paths

        # If linewidth not specified, plot each path in successively increasing width
        if lw is None:
            linewidths = np.linspace(self.lw_min, self.lw_max, n_paths)
        else:
            linewidths = np.full(n_paths, lw)
        # Initialize plotting handle for each path
        if colors is None:
            if n_paths > 1:
                colors = np.random.random((n_paths, 3))
            else:
                colors = np.zeros((1, 3))

        labels = []
        self.path_handles = []
        for path_idx in range(n_paths):
            label = self._get_label(path_idx)
            self.path_handles += ax.plot(
                [],
                lw=linewidths[path_idx],
                color=colors[path_idx],
                label=label,
                zorder=5 + n_paths - path_idx,
                alpha=0.9,
            )

        self._no_legend = no_legend
        if not no_legend:
            self.ax.legend(loc="upper right")

        self.update(mhp_result)

    def _get_label(self, path_idx, dist_cost=None, safety_cost=None):
        labeltxt = self._get_label_text()
        if path_idx == 0:
            label = labeltxt[0].upper() + labeltxt[1:] + " path"
        elif path_idx == 1:
            label = "2nd %s path" % labeltxt
        elif path_idx == 2:
            label = "3rd %s path" % labeltxt
        else:
            label = "%dth %s path" % (path_idx + 1, labeltxt)
        if dist_cost is None or safety_cost is None:
            costs_str = ""
        else:
            costs_str = ": distance %.3f, safety %.3f" % (dist_cost, math.exp(-safety_cost))
        return "%s%s" % (label, costs_str)

    def update(self, mhp_result: MultipleHypothesisPlanningResult):
        if self.cell_decomposition_plotter is not None:
            self.cell_decomposition_plotter.update(mhp_result.cell_decomposition)

        # paths = self._get_sorted_paths(mhp_result)
        paths = mhp_result.sorted_paths
        distance_costs = mhp_result.sorted_distance_costs
        safety_costs = mhp_result.sorted_safety_costs

        # Plot only the top N paths
        if len(paths) > self.n_paths:
            paths = paths[0 : self.n_paths]

        for path_idx in range(self.n_paths):
            handle: plt.Line2D = self.path_handles[path_idx]
            # Check if MHP result contained less than N paths
            if path_idx >= len(paths):
                handle.set_data([], [])
                handle.set_label(self._get_label(path_idx))
            else:
                path = paths[path_idx]
                # Get the vertices of the path
                xy = mhp_result.distance_graph.get_path_xy(path)
                # self.path_handles[path_idx].set_data(xy[:,0], xy[:,1])
                handle.set_data(xy[:, 0], xy[:, 1])

                label = self._get_label(
                    path_idx, dist_cost=distance_costs[path_idx], safety_cost=safety_costs[path_idx]
                )
                handle.set_label(label)
        if not self._no_legend:
            self.ax.legend()  # refresh the legend

    def _get_label_text(self):
        return "best"

    def _get_sorted_paths(self, mhp_result) -> List[NavigationPath]:
        return mhp_result.sorted_paths


# TODO these classes can be removed, no longer generating hypotheses with separate methods
#      The planner used to generated shortest and safest paths separately, so multiple differnet
#      plotter classes were needed
class HighLevelPathDistancePlotter(HighLevelPathPlotter):

    def _get_sorted_paths(self, mhp_result: MultipleHypothesisPlanningResult) -> List[NavigationPath]:
        return mhp_result.sorted_paths

    def _get_label_text(self):
        return "best"


class HighLevelPathSafetyPlotter(HighLevelPathPlotter):

    labeltxt = "safest"

    def _get_sorted_paths(self, mhp_result: MultipleHypothesisPlanningResult) -> List[NavigationPath]:
        return mhp_result.sorted_paths

    def _get_label_text(self):
        return "safest"


class BestHighLevelPathPlotter(HighLevelPathPlotter):

    def _get_sorted_paths(self, mhp_result: MultipleHypothesisPlanningResult) -> List[NavigationPath]:
        path_indices = [i for i in range(len(mhp_result.paths))]
        path_indices.sort(key=lambda path_idx: mhp_result.costs[path_idx])

        sorted_paths = [mhp_result.paths[i] for i in path_indices]

        return sorted_paths


class BloatedObstaclePlotter(Plotter):

    def __init__(self, obstacles, ax=None):
        super().__init__(ax=ax)
        self.circles = []
        self.update(obstacles)

    def update(self, obstacles):
        for i in range(obstacles.shape[0]):
            position = obstacles[i, 0:2]
            radius = obstacles[i, 2] / 2.0
            if i >= len(self.circles):
                circle = patches.Circle(position, radius=radius, color=self.light_green, alpha=0.3, zorder=3)
                self.ax.add_patch(circle)
                self.circles.append(circle)
            else:
                circle = self.circles[i]
                circle.set_center(position)
                circle.set_radius(radius)


class SinglePathPlotter(Plotter):

    # Min and max line width

    def __init__(
        self,
        mhp_result: MultipleHypothesisPlanningResult,
        path_idx: int,
        ax=None,
        cmap="plasma",
        show_cell_decomposition=True,
        colors=None,
    ):
        super().__init__(ax=ax)

        self.cmap = cm.get_cmap(cmap)

        # Plot the cell decomposition
        if show_cell_decomposition:
            self.cell_decomposition_plotter = CellDecompositionPlotter(
                mhp_result.cell_decomposition, show_uncertainty=False, show_labels=False, ax=self.ax
            )
        else:
            self.cell_decomposition_plotter = None

        # plt.colorbar(cm.ScalarMappable(cmap=self.cmap), ax=ax, shrink=0.7)

        # Initialize plotting handle for the path
        linewidth = 4.0
        color = Plotter.med_green

        path = mhp_result.paths[path_idx]
        # Get the vertices of the path
        xy = mhp_result.distance_graph.get_path_xy(path)
        # self.path_handles[path_idx].set_data(xy[:,0], xy[:,1])

        label = "Path"

        (self.path_handle,) = ax.plot(xy[:, 0], xy[:, 1], lw=linewidth, color=color, label=label, zorder=5)

        self.ax.legend(loc="upper right")


class SubplotInterface(object):
    """
    Abstract class for an interface that can be used by any subplot in the plot.
    For example, the subplot showing the ground truth robot position/obstacles,
    the subplot showing the planned path, etc.
    """

    title = "Simulation subplot"
    robot_color = (0.4, 0.4, 0.8)
    obstacle_color = (0.15, 0.15, 0.15)

    def __init__(self, ax: plt.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float], sim: Simulation, title=None):
        self.ax = ax
        #
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        xmin, xmax = xlim
        ymin, ymax = ylim
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")
        # Force integer only tick labels
        self.ax.set_yticks(np.arange(floor(ymin), ceil(ymax), 1.0))
        self.ax.set_xticks(np.arange(floor(xmin), ceil(xmax), 1.0))

        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        self.ax.grid("minor", alpha=0.05, color=(0.2, 0.2, 0.2))

        if title is None:
            # Set ax title to default static member
            title = self.title
        self.ax.set_title(title)

    def update(self, sim: Simulation):
        raise NotImplementedError

    def get_axes(self):
        return self.ax


class GroundTruthSubplot(SubplotInterface):
    """
    Ground truth subplot.
    Shows the robot pose and actual obstacle positions/sizes.
    """

    title = "Ground truth. Simulation time: 0.00"

    def __init__(self, ax: plt.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float], sim: Simulation, title=None):
        super().__init__(ax=ax, xlim=xlim, ylim=ylim, sim=sim, title=title)
        # Initialize plotter for the robot
        robot_width = sim.params.robot_width

        self.plot_robot = PlotRobot(self.ax, position=sim.robot_position, width=robot_width, color=self.robot_color)

        # Plot the obstacles
        for i in range(sim.n_obstacles):
            p = patches.Circle(
                sim.obstacle_positions[i, :], radius=sim.obstacle_diameters[i] / 2, color=self.obstacle_color
            )
            self.ax.add_patch(p)

    def update(self, sim: Simulation):
        self.plot_robot.set_pose(sim.robot_position, sim.robot_orientation)
        self.ax.set_title("Ground truth. Simulation time: %.2f" % sim.t)


class SLAMSubplot(SubplotInterface):
    """
    Estimator subplot.
    Shows the robot position, current estimated obstacle positions, and new detections at each time step.
    """

    title = "Detections and obstacle SLAM estimate"

    def __init__(self, ax: plt.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float], sim: Simulation, title=None):
        super().__init__(ax=ax, xlim=xlim, ylim=ylim, sim=sim, title=title)
        robot_width = sim.params.robot_width

        self.plot_robot = PlotRobot(
            self.ax, position=sim.robot_position, width=robot_width, color=self.robot_color, zorder=100
        )
        self.ax_slam_obstacles = {}
        line_segments = detections_to_lines(sim.detections_xy_global)
        self.ax_slam_detections = LineCollection(line_segments, linewidths=1.2, colors=Plotter.light_green, zorder=8)
        self.ax.add_collection(self.ax_slam_detections)
        self.pose_cov_handle = None

    def update(self, sim: Simulation):
        pose, pose_cov = sim.slam.get_pose_estimate()
        self.plot_robot.set_pose([pose[0], pose[1]], pose[2])
        # TODO update covariance ellipse instead of removing and re-plotting
        if self.pose_cov_handle is not None:
            self.pose_cov_handle.remove()
        self.pose_cov_handle = covariance_ellipse(
            pose, pose_cov, self.ax, n_std=2.0, alpha=0.3, facecolor="blue", edgecolor="blue", zorder=101
        )
        # Get the estimated landmark positions from iSAM and plot them
        obstacles = sim.slam.get_landmarks_estimate()
        for obs_idx, obstacle in enumerate(obstacles):
            handle: Optional[ObstacleEstimatePlotter] = self.ax_slam_obstacles.setdefault(obs_idx, None)
            if handle is None:
                self.ax_slam_obstacles[obs_idx] = ObstacleEstimatePlotter(obstacle, ax=self.ax)
            else:
                handle.update(obstacle)

        detections = sim.detections_xy_global
        line_segments = detections_to_lines(detections)
        self.ax_slam_detections.set_segments(line_segments)

        # Plot boundary obstacles
        boundary_obstacles = sim.boundary.get_obstacles()
        for obs_idx, b_obstacle in enumerate(boundary_obstacles):
            b_obs_idx = -(
                obs_idx + 1
            )  # give boundary obstacles negative indices to avoid conflict with the regular obstacles
            handle: Optional[ObstacleEstimatePlotter] = self.ax_slam_obstacles.setdefault(b_obs_idx, None)
            if handle is None:
                self.ax_slam_obstacles[b_obs_idx] = ObstacleEstimatePlotter(b_obstacle, ax=self.ax, n_std_devs=0)
            else:
                handle.update(b_obstacle)


class BaselinePlannerSubplot(SubplotInterface):
    """
    Subplot for the baseline planner.
    Shows estimated obstacle means, A* global planner path, and local planned path.
    """

    title = "A* baseline: Navigation obstacles and planned path"

    def __init__(
        self,
        ax: plt.Axes,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        sim: Simulation,
        title=None,
        plot_search_tree=False,
    ):
        super().__init__(ax=ax, xlim=xlim, ylim=ylim, sim=sim, title=title)
        self.astar_plotter = None
        # Dictionary of obstacle plotters keyed by obstacle index
        self.baseline_obstacles = {}
        self.baseline_path = None
        self.plot_search_tree = plot_search_tree

    def update(self, sim: Simulation):
        """
        Plot planner results for baseline.
        Shows obstacle means, the 2D A* global planner path, and the hybrid A* path.
        """
        # Update obstacles
        obstacles = sim.slam.get_landmarks_estimate()
        for obs_idx, obstacle in enumerate(obstacles):
            handle: Optional[ObstaclePlotter] = self.baseline_obstacles.setdefault(obs_idx, None)
            if handle is None:
                self.baseline_obstacles[obs_idx] = BloatedObstacleEstimatePlotter(obstacle, n_std_devs=2, ax=self.ax)
            else:
                handle.update(obstacle)

        # Update 2D A* path
        if self.baseline_path is None:
            if sim.baseline_global_planner_points is not None:
                self.baseline_path = BaselineHighLevelPathPlotter(sim.baseline_global_planner_points, ax=self.ax)
        else:
            self.baseline_path.update(sim.baseline_global_planner_points)

        # Update hybrid A* Path
        if sim.local_planner_result is not None:
            if self.astar_plotter is None:
                self.astar_plotter = LocalPlannerPathPlotter(ax=self.ax)
            if not self.plot_search_tree:
                self.astar_plotter.plot_waypoints(sim.waypoints)
            else:
                self.astar_plotter.plot_search_tree(sim.local_planner_result)


class CellDecompositionSubplot(SubplotInterface):
    """
    Cell decomposition subplot
    Shows obstacle estimates, cell decomposition, graph structure, best path selected, and hybrid A* waypoints.
    """

    title = "Multi-hypothesis planner: Cell decomposition and planned path"

    def __init__(
        self,
        ax: plt.Axes,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        sim: Simulation,
        title=None,
        plot_search_tree=False,
        plot_bloated=False,
    ):
        super().__init__(ax=ax, xlim=xlim, ylim=ylim, sim=sim, title=title)
        self.cell_decomp_plot = None
        self.bloated_obstacles = None
        self.astar_plotter = None
        self.path_handle = None
        self.planner_waypoints_handle = None
        self.planner_obstacle_handles = []
        self.plot_search_tree = plot_search_tree
        self.plot_bloated = plot_bloated

    def update(self, sim: Simulation):
        if sim.cell_decomposition is not None:
            cell_decomp = sim.cell_decomposition
            if self.cell_decomp_plot is None:
                cell_decomp_plot = CellDecompositionPlotter(
                    cell_decomp, ax=self.ax, show_labels=False, show_uncertainty=True, show_colorbar=False
                )
                self.cell_decomp_plot = cell_decomp_plot
            else:
                self.cell_decomp_plot.update(cell_decomp)

        mhp_result: Optional[MultipleHypothesisPlanningResult] = sim.mhp_result
        if mhp_result is not None:
            self.cell_decomp_plot.plot_navigation_path(mhp_result.best_path)

        if sim.local_planner_result is not None:
            if self.astar_plotter is None:
                self.astar_plotter = LocalPlannerPathPlotter(ax=self.ax)
            if not self.plot_search_tree:
                self.astar_plotter.plot_waypoints(sim.waypoints)
            else:
                self.astar_plotter.plot_search_tree(sim.local_planner_result)
        if self.plot_bloated:
            if self.bloated_obstacles is None:
                self.bloated_obstacles = BloatedObstaclePlotter(sim.bloated_obstacles, ax=self.ax)
            else:
                self.bloated_obstacles.update(sim.bloated_obstacles)


class MultipleHypothesisSubplot(SubplotInterface):
    """
    Path distances subplot
    Plots the multiple hypothesis paths, colored by distance cost
    """

    title = "Top paths hypothesis"

    def __init__(
        self,
        ax: plt.Axes,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        sim: Simulation,
        title=None,
        n_paths_to_plot: int = 3,
        no_legend=False,
    ):
        if title is None:
            title = "Top %d path hypotheses" % n_paths_to_plot
        super().__init__(ax=ax, xlim=xlim, ylim=ylim, sim=sim, title=title)
        self.n_paths_to_plot = n_paths_to_plot
        self.path_plotter = None
        self.graph_plotter = None

        cmap_name = "Set2"
        cmap = get_cmap(cmap_name)
        colors = [cmap.colors[i] for i in range(n_paths_to_plot)]
        self.path_colors = colors
        self.no_legend = no_legend

    def update(self, sim: Simulation):
        if sim.mhp_result is not None:
            if self.path_plotter is None:
                show_cd = True
                self.path_plotter = HighLevelPathDistancePlotter(
                    sim.mhp_result,
                    ax=self.ax,
                    n_paths=self.n_paths_to_plot,
                    colors=self.path_colors,
                    show_cell_decomposition=show_cd,
                    no_legend=self.no_legend,
                )
            else:
                self.path_plotter.update(sim.mhp_result)
            dg = sim.mhp_result.distance_graph
            if self.graph_plotter is None:
                self.graph_plotter = DistanceGraphPlotter(dg, ax=self.ax)
            else:
                self.graph_plotter.update(dg)


class SimulationPlotter(object):

    def __init__(
        self,
        sim: Simulation,
        plot_bounds,
        fig=None,
        interactive=False,
        minimal=False,
        plot_search_tree=False,
        plot_bloated=False,
        dpi=None,
    ):
        """

        Parameters
        ----------
        plot_bounds
            [xmin, xmax, ymin, ymax]
        interactive: bool
            Set to True if viewing online and set to False if saving a video.
        minimal: bool
            If True, the plotter will omit plotting certain objects, to greatly speed up plotting.
            Useful for quickly visualizing simulations online.
        """
        self.interactive = interactive
        self.minimal = minimal
        self.plot_search_tree = plot_search_tree
        self.plot_bloated = plot_bloated
        is_baseline = sim.params.n_hypotheses == 0  # Plot MHP and baseline differently
        robot_width = sim.params.robot_width

        xmin, xmax, ymin, ymax = plot_bounds
        if fig is None:
            fig: plt.Figure = plt.figure()
            fig.set_size_inches(19.2, 10.8)
            if dpi is not None:
                fig.set_dpi(dpi)
            plt.show()

        subplot_rows = 2
        subplot_cols = 2

        xlim = (xmin, xmax)
        ylim = (ymin, ymax)
        ax_gt: plt.Axes = fig.add_subplot(subplot_rows, subplot_cols, 1)
        self.subplot_gt = GroundTruthSubplot(ax_gt, xlim=xlim, ylim=ylim, sim=sim)

        # All other axes should share x axis limits with the ground truth subplot
        ax_slam = fig.add_subplot(subplot_rows, subplot_cols, 2, sharex=ax_gt)
        self.subplot_slam = SLAMSubplot(ax_slam, xlim=xlim, ylim=ylim, sim=sim)

        self.planner_subplots = []
        if is_baseline:
            ax_base = fig.add_subplot(subplot_rows, subplot_cols, 3, sharex=ax_gt)
            subplot_baseline = BaselinePlannerSubplot(ax_base, xlim=xlim, ylim=ylim, sim=sim)
            self.planner_subplots.append(subplot_baseline)

        else:
            ax_cd = fig.add_subplot(subplot_rows, subplot_cols, 3, sharex=ax_gt)
            subplot_cd = CellDecompositionSubplot(ax_cd, xlim=xlim, ylim=ylim, sim=sim)
            self.planner_subplots.append(subplot_cd)

            ax_paths = fig.add_subplot(subplot_rows, subplot_cols, 4, sharex=ax_gt)
            subplot_paths = MultipleHypothesisSubplot(ax_paths, xlim=xlim, ylim=ylim, sim=sim, no_legend=True)
            self.planner_subplots.append(subplot_paths)

        fig.tight_layout()
        self.fig = fig

        self.sim = sim

        self.next_est_update_time = sim.next_det_time
        self.next_nav_update_time = sim.next_replan_time

        self.is_baseline = is_baseline

    def update(self):
        sim = self.sim

        # Update ground truth at every time step
        self.subplot_gt.update(sim)

        # Check if detections and SLAM should be updated
        if sim.t >= self.next_est_update_time:
            # Do detections update
            self.subplot_slam.update(sim)
            self.next_est_update_time = sim.next_det_time

        if sim.t >= self.next_nav_update_time:
            # Update axes for planner visualization
            for planner_subplot in self.planner_subplots:
                planner_subplot.update(sim)
            self.next_nav_update_time = sim.next_replan_time
        if self.interactive:
            plt.pause(0.0001)

    def is_closed(self):
        return not plt.fignum_exists(self.fig.number)


class SimulationAnimator(object):

    def __init__(self, plotter, max_time=None):
        sim = plotter.sim
        self.plotter = plotter
        self.max_time = max_time

        interval = 1000 / sim.params.sim_rate  # interval between frames in milliseconds

        n_frames_cache = int(sim.params.max_time * sim.params.sim_rate)  # max number of frames in the sim
        self.anim = FuncAnimation(plotter.fig, self.animate, interval=interval, save_count=n_frames_cache, repeat=False)
        plt.draw()
        plt.show()

    def save(self, video_filename):
        try:
            print("Creating animation, saving to file: %s" % video_filename)
            self.anim.save(video_filename)
        except KeyboardInterrupt:
            print("")
            print("Done!")

    def animate(self, i):
        print("[Frame %d]" % i, end="\r", flush=True)
        sim = self.plotter.sim
        if sim.is_running():
            sim.update()
            self.plotter.update()
        else:
            self.anim.event_source.stop()
            plt.close(self.plotter.fig)
            # TODO haven't been able to find another way to end the animation early besides this interrupt...
            #      Happens if sim times out. Video still saves.
            raise KeyboardInterrupt


@numba.njit
def detections_to_lines(detections):
    # Get the line segments forming a square box around detections (x, y, diameter)
    line_segments = []
    for i in range(detections.shape[0]):
        det = detections[i, :]
        x = det[0]
        y = det[1]
        r = det[2] / 2.0
        line_segments.append([(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r), (x - r, y - r)])
    return line_segments
