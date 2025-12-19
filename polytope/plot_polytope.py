"""
3D Polytope Visualization Module

This module provides functions to visualize 3D polytopes from H-representation (Ax <= b), V-representation (vertices/generators), or pycddlib Polyhedron objects.

Dependencies
------------
- numpy
- scipy.spatial.ConvexHull
- matplotlib, mpl_toolkits.mplot3d
- pycddlib (cdd)
- double_description module
"""

import numpy as np
from numpy import array, zeros, ones, hstack
from scipy.spatial import ConvexHull
import cdd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import pylab as pl
from polytope.double_description import *


def plot_hull_in_subplot(hull, ax, color="r", alpha=0.1):
    """
    Plot the faces of a 3D convex hull in the given Axes.

    :param hull: ConvexHull object representing the polytope.
    :type hull: scipy.spatial.ConvexHull
    :param ax: Matplotlib 3D axes to plot on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :param color: Face color.
    :type color: str
    :param alpha: Transparency of the faces.
    :type alpha: float
    :return: The axes with plotted hull.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """
    for s in hull.simplices:
        tri = art3d.Poly3DCollection([hull.points[s]], alpha=alpha)
        tri.set_color(color)
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    return ax


def plot_polygon(pts, ax, color="y", alpha=0.5):
    """
    Plot a polygon from a set of points using a 2D hull projection.

    :param pts: Array of points, shape (n, 3).
    :type pts: np.ndarray
    :param ax: Matplotlib 3D axes to plot on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :param color: Face color.
    :type color: str
    :param alpha: Transparency of the polygon.
    :type alpha: float
    :return: The axes with plotted polygon.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """
    hull2D = ConvexHull(pts[:, :-1])
    ptsSorted = pts[hull2D.vertices]
    x, y, z = ptsSorted[:,0], ptsSorted[:,1], ptsSorted[:,2]
    pol = art3d.Poly3DCollection([list(zip(x, y, z))], alpha=alpha)
    pol.set_color(color)
    pol.set_edgecolor('k')
    ax.add_collection3d(pol)
    return ax


def plot_plane(n, d, scaleFactor, ax=None):
    """
    Plot a plane defined by normal vector `n` and distance `d` over a square grid of size scaleFactor.

    :param n: Normal vector of the plane (3,).
    :type n: np.ndarray
    :param d: Distance from origin.
    :type d: float
    :param scaleFactor: Half-size of the square grid.
    :type scaleFactor: float
    :param ax: Matplotlib 3D axes to plot on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: The axes with plotted plane.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """
    rg = int(scaleFactor)
    xx, yy = np.meshgrid(range(-rg, rg), range(-rg, rg))
    z = (-n[0] * xx - n[1] * yy + d) / n[2]
    ax.plot_surface(xx, yy, z, alpha=0.2)
    return ax


def init_ax(scale):
    """
    Initialize a 3D axes object with a clean display and specified scale.

    :param scale: Maximum absolute coordinate in each axis.
    :type scale: float
    :return: A 3D axes object.
    :rtype: matplotlib.axes._subplots.Axes3DSubplot
    """
    ax = pl.figure().add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim3d(-scale, scale)
    ax.set_ylim3d(-scale, scale)
    ax.set_zlim3d(-scale, scale)
    return ax


def plot_hull(hull, color="r", ax=None):
    """
    Plot a 3D convex hull.

    :param hull: ConvexHull object representing the polytope.
    :type hull: scipy.spatial.ConvexHull
    :param color: Color of the faces.
    :type color: str
    :param ax: Matplotlib 3D axes to plot on.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :return: None
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    plot_hull_in_subplot(hull, ax, color)


def plot_polytope_H_rep(A_in, b_in, color="r", ax=None):
    """
    Plot a polytope from its H-representation (Ax <= b).

    :param A_in: Coefficient matrix of shape (m, n).
    :type A_in: np.ndarray
    :param b_in: Right-hand side vector of length m.
    :type b_in: np.ndarray
    :param color: Color of the polytope faces.
    :type color: str
    :param ax: Optional Matplotlib Axes object.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot or None
    :return: True if plotted, False if empty.
    :rtype: bool
    """
    hull, pts, apts, cd = gen_polytope(A_in, b_in)
    if hull is None:
        print("empty polytope")
        return False
    plot_hull(hull, color, ax=ax)
    return True


def plot_polytope_V_rep(pts, color="r", ax=None):
    """
    Plot a polytope from its vertices (V-representation).

    :param pts: List of 3D points representing vertices.
    :type pts: list of np.ndarray
    :param color: Color of the polytope faces.
    :type color: str
    :param ax: Optional Matplotlib Axes object.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot or None
    :return: None
    """
    pts = [array(el) for el in pts]
    apts = array(pts)
    hull = ConvexHull(apts, qhull_options='Q12')
    plot_hull(hull, color, ax=ax)


def plot_polytope_cdd_polyhedron(H, color="r", ax=None):
    """
    Plot a polytope from a pycddlib Polyhedron object.

    :param H: cdd Polyhedron object.
    :type H: cdd.Polyhedron
    :param color: Color of the polytope faces.
    :type color: str
    :param ax: Optional Matplotlib Axes object.
    :type ax: matplotlib.axes._subplots.Axes3DSubplot or None
    :return: None
    """
    g = cdd.copy_generators(H).array
    pts = [array(g[el][1:]) for el in range(len(g))]
    plot_polytope_V_rep(pts, color, ax)


if __name__ == "__main__":
    num_points = 20
    points = np.random.rand(num_points, 3) * 10

    hull = ConvexHull(points, qhull_options='Q12')
    A, b = ineq_from_hull(hull)

    fig = plt.figure(figsize=(18,6))

    ax1 = fig.add_subplot(131, projection='3d')
    plot_polytope_H_rep(A, b, color='r', ax=ax1)
    ax1.set_title('Polytope from H-rep')

    pts_from_H, H_poly = generators(A, b)
    ax2 = fig.add_subplot(132, projection='3d')
    plot_polytope_V_rep(pts_from_H, color='b', ax=ax2)
    ax2.set_title('Polytope from V-rep (generators)')

    ax3 = fig.add_subplot(133, projection='3d')
    plot_polytope_cdd_polyhedron(H_poly, color='g', ax=ax3)
    ax3.set_title('Polytope from CDD Polyhedron')

    plt.show()
