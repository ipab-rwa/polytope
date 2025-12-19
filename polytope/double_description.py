"""
Polytope generation and H-/V-representation utilities using pycddlib.

This module provides functions to:
- Convert H-representation (Ax <= b) to V-representation (generators) and vice versa.
- Filter points using ConvexHull.
- Convert a set of points to inequalities.
- Generate a polytope object suitable for visualization or further processing.

Dependencies
------------
- numpy
- scipy.spatial.ConvexHull
- pycddlib (cdd)

Notes
-----
- All functions assume 3D points but can work in higher dimensions.
- Uses `cdd.matrix_from_array` and `cdd.polyhedron_from_matrix` from pycddlib >= 3.0.
- Inequalities returned are in the form Ax <= b.
"""

import numpy as np
from numpy import array, zeros, ones, hstack
from scipy.spatial import ConvexHull
import cdd

from random import random as rd, randint as rdi


def generators(A, b, Aeq=None, beq=None):
    """
    Convert H-representation (Ax <= b) to V-representation (generators) using pycddlib.

    Parameters
    ----------
    A : np.ndarray
        Inequality matrix (n_constraints x n_variables).
    b : np.ndarray
        Right-hand side vector.
    Aeq : np.ndarray, optional
        Equality constraints matrix.
    beq : np.ndarray, optional
        Right-hand side for equality constraints.

    Returns
    -------
    generators : list of np.ndarray
        List of generator points.
    H : cdd.Polyhedron
        The polyhedron object in cdd format.
    """
    m = np.hstack([b.reshape(-1,1), -A])
    matcdd = cdd.matrix_from_array(m)
    matcdd.rep_type = cdd.RepType.INEQUALITY

    if Aeq is not None and beq is not None:
        meq = np.hstack([beq.reshape(-1,1), -Aeq])
        matcdd.extend(meq.tolist(), linear=True)

    H = cdd.polyhedron_from_matrix(matcdd)
    g = cdd.copy_generators(H).array
    return [array(g[el][1:]) for el in range(len(g))], H


def filter(pts):
    """
    Filter a list of points to the convex hull using qhull

    Parameters
    ----------
    pts : list of np.ndarray
        Input points.

    Returns
    -------
    list of np.ndarray
        Points on the convex hull.
    """
    hull = ConvexHull(pts, qhull_options='Q12')
    return [pts[i] for i in hull.vertices.tolist()]


def ineq(pts, canonicalize=False):
    """
    Convert a set of points to inequality representation (Ax <= b).

    Parameters
    ----------
    pts : list of np.ndarray
        Input points (V-representation).
    canonicalize : bool, optional
        Whether to canonicalize inequalities (default False).

    Returns
    -------
    A : np.ndarray
        Inequality matrix.
    b : np.ndarray
        Right-hand side vector.
    """
    apts = array(pts)
    m = np.hstack([ones((apts.shape[0],1)), apts])
    matcdd = cdd.matrix_from_array(m)
    matcdd.rep_type = cdd.RepType.GENERATOR
    H = cdd.polyhedron_from_matrix(matcdd)
    bmA = cdd.copy_inequalities(H)
    if canonicalize:
        bmA.canonicalize()
    Ares = zeros((bmA.row_size,bmA.col_size-1))
    bres = zeros(bmA.row_size)
    for i in range(bmA.row_size):
        l = array(bmA[i])
        Ares[i,:] = -l[1:]
        bres[i] = l[0]
    return Ares, bres


def ineq_from_hull(hull):
    """
    Get H-representation (Ax <= b) from a scipy.spatial.ConvexHull object.

    Parameters
    ----------
    hull : ConvexHull
        Input convex hull.

    Returns
    -------
    A : np.ndarray
        Inequality matrix.
    b : np.ndarray
        Right-hand side vector.
    """
    A = hull.equations[:,:-1]
    b = -hull.equations[:,-1]
    return A, b


def gen_polytope(A, b):
    """
    Generate a polytope from H-representation and compute its ConvexHull.

    Parameters
    ----------
    A : np.ndarray
        Inequality matrix.
    b : np.ndarray
        Right-hand side vector.

    Returns
    -------
    hull : ConvexHull or None
        Convex hull of the generators.
    pts : list of np.ndarray or None
        List of generator points.
    apts : np.ndarray or None
        Array of generator points.
    H : cdd.Polyhedron or None
        The polyhedron object in cdd format.
    """
    pts, H = generators(A, b)
    apts = array(pts)
    if len(apts) > 0:
        hull = ConvexHull(apts)
        return hull, pts, apts, H
    return None, None, None, None
