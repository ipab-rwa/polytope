"""
OBJ mesh loading and conversion to half-space inequalities.

This module provides utilities to:
- Load a Wavefront OBJ file (vertices, normals, faces)
- Convert a (convex) triangular mesh into linear inequalities Ax <= b
- Test point inclusion
- Rigidly transform inequalities
- Serialize / deserialize inequalities

Conventions
-----------
- Half-spaces are represented as: A x <= b
- Normals are assumed to be outward-pointing
- OBJ indices are converted from 1-based to 0-based

Limitations
-----------
- The mesh is assumed to be convex
- Face normals are assumed to be provided in the OBJ file
- No robustness handling for degenerate faces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray
import pickle

EPSILON = 0.0


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class ObjectData:
    """Raw data extracted from an OBJ file.

    Attributes
    ----------
    vertices : list of ndarray, shape (3,)
        Vertex positions.
    texcoords : list
        Texture coordinates (unused here).
    normals : list of ndarray, shape (3,)
        Normal vectors.
    faces : list
        Face definitions as OBJ index triplets.
    """

    vertices: List[NDArray[np.float64]]
    texcoords: List
    normals: List[NDArray[np.float64]]
    faces: List


@dataclass
class Inequalities:
    """Half-space representation of a convex polytope.

    Attributes
    ----------
    A : ndarray, shape (m, 3)
        Constraint normals.
    b : ndarray, shape (m,)
        Offsets.
    normals : ndarray, shape (m, 3)
        Copy of constraint normals.
    points : ndarray, shape (m, 4)
        Homogeneous points lying on each supporting plane.
    """

    A: NDArray[np.float64]
    b: NDArray[np.float64]
    normals: NDArray[np.float64]
    points: NDArray[np.float64]


# -----------------------------------------------------------------------------
# OBJ loading
# -----------------------------------------------------------------------------


def _to_float_array(tokens: List[str]) -> NDArray[np.float64]:
    """Convert a list of strings to a float numpy array."""

    return np.asarray(tokens, dtype=np.float64)


def load_obj(filename: str) -> ObjectData:
    """Load a Wavefront OBJ file.

    Parameters
    ----------
    filename : str
        Path to the OBJ file.

    Returns
    -------
    ObjectData
        Parsed OBJ data.
    """

    vertices = []
    texcoords = []
    normals = []
    faces = []

    with open(filename, "r") as file_handle:
        for line in file_handle:
            if not line or line.startswith("#"):
                continue

            tokens = line.strip().split()
            prefix = tokens[0]

            if prefix == "v":
                vertices.append(_to_float_array(tokens[1:4]))
            elif prefix == "vt":
                texcoords.append(tokens[1:])
            elif prefix == "vn":
                normals.append(_to_float_array(tokens[1:4]))
            elif prefix == "f":
                face = []
                for element in tokens[1:]:
                    indices = element.split("/")
                    # Convert OBJ indices from 1-based to 0-based
                    vertex_index = int(indices[0]) - 1
                    normal_index = int(indices[2]) - 1 if len(indices) > 2 else None
                    face.append((vertex_index, normal_index))
                faces.append(face)

    return ObjectData(vertices, texcoords, normals, faces)


# -----------------------------------------------------------------------------
# Geometry utilities
# -----------------------------------------------------------------------------


def find_point_on_plane(
    normal: NDArray[np.float64], offset: float
) -> NDArray[np.float64]:
    """Compute the point on a plane closest to the origin.

    The plane is defined as: normal · x = offset

    Parameters
    ----------
    normal : ndarray, shape (3,)
        Plane normal.
    offset : float
        Plane offset.

    Returns
    -------
    ndarray, shape (3,)
        Closest point to the origin on the plane.
    """

    normal = normal / np.linalg.norm(normal)
    return normal * offset


def plane_inequality(
    point: NDArray[np.float64], normal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Construct a half-space inequality from a plane.

    The plane is defined by:
        normal · (x - point) = 0

    Leading to the inequality:
        normal · x <= normal · point

    Parameters
    ----------
    point : ndarray, shape (3,)
        A point on the plane.
    normal : ndarray, shape (3,)
        Outward normal.

    Returns
    -------
    A : ndarray, shape (3,)
    b : float
    """

    return normal, float(normal.dot(point) + EPSILON)


# -----------------------------------------------------------------------------
# Conversion OBJ -> inequalities
# -----------------------------------------------------------------------------


def mesh_to_inequalities(obj: ObjectData) -> Inequalities:
    """Convert a convex OBJ mesh into linear inequalities.

    Parameters
    ----------
    obj : ObjectData
        Loaded OBJ mesh.

    Returns
    -------
    Inequalities
        Half-space representation Ax <= b.
    """

    number_of_faces = len(obj.faces)

    A = np.empty((number_of_faces, 3))
    b = np.empty(number_of_faces)
    normals = np.empty((number_of_faces, 3))
    points = np.ones((number_of_faces, 4))

    for index, face in enumerate(obj.faces):
        vertex_index, normal_index = face[0]
        vertex = obj.vertices[vertex_index]
        normal = obj.normals[normal_index]

        Ai, bi = plane_inequality(vertex, normal)

        A[index] = Ai
        b[index] = bi
        normals[index] = normal
        points[index, :3] = vertex

    return Inequalities(A=A, b=b, normals=normals, points=points)


# -----------------------------------------------------------------------------
# Queries and transformations
# -----------------------------------------------------------------------------


def is_inside(inequalities: Inequalities, point: NDArray[np.float64]) -> bool:
    """Check whether a point satisfies all inequalities."""

    return np.all(inequalities.A @ point <= inequalities.b)


def transform_inequalities(
    inequalities: Inequalities, transform: NDArray[np.float64]
) -> Inequalities:
    """Apply a rigid homogeneous transform to inequalities.

    Parameters
    ----------
    inequalities : Inequalities
        Original half-spaces.
    transform : ndarray, shape (4, 4)
        Homogeneous transform matrix.

    Returns
    -------
    Inequalities
        Transformed half-spaces.
    """

    number = inequalities.A.shape[0]

    A = np.empty((number, 3))
    b = np.empty(number)
    normals = np.empty((number, 3))
    points = np.empty((number, 4))

    rotation = transform[:3, :3]

    for index in range(number):
        transformed_point = transform @ inequalities.points[index]
        transformed_normal = rotation @ inequalities.normals[index]

        Ai, bi = plane_inequality(transformed_point[:3], transformed_normal)

        A[index] = Ai
        b[index] = bi
        normals[index] = transformed_normal
        points[index] = transformed_point

    return Inequalities(A=A, b=b, normals=normals, points=points)


# -----------------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------------


def save_inequalities(inequalities: Inequalities, filename: str) -> None:
    """Serialize inequalities to disk using pickle."""

    with open(filename, "wb") as file_handle:
        pickle.dump(inequalities, file_handle)


def load_inequalities(filename: str) -> Inequalities:
    """Load inequalities from disk."""

    with open(filename, "rb") as file_handle:
        return pickle.load(file_handle)
