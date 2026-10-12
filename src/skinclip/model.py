"""The data a check or a repair works on: a node skeleton, a skinned figure and its clips."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import linalg

if TYPE_CHECKING:  # intersect imports this module; the settings are only read here
    from .intersect import CheckSettings


@dataclass
class Clip:
    name: str
    seconds: float
    # node name -> path -> (times, values, interpolation)
    channels: dict[str, dict[str, tuple[np.ndarray, np.ndarray, str]]]


@dataclass(frozen=True)
class Skeleton:
    """Every node of the glTF (joints and the nodes between them), with its rest transform and parent (-1: a root)."""

    names: tuple[str, ...]
    parent: np.ndarray  # (N,) node index
    rest_t: np.ndarray  # (N, 3)
    rest_r: np.ndarray  # (N, 4) xyzw
    rest_s: np.ndarray  # (N, 3)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(name) from None


@dataclass
class Figure:
    name: str
    path: Path
    skeleton: Skeleton
    joints: np.ndarray  # skin joint -> node index
    inverse_bind: np.ndarray  # (J, 4, 4)
    positions: np.ndarray  # (V, 3) bind pose
    vertex_joints: np.ndarray  # (V, 4) skin joint indices
    vertex_weights: np.ndarray  # (V, 4)
    triangles: np.ndarray  # (T, 3)
    triangle_label: list[str]  # material:dominant joint, for reports
    welded: np.ndarray  # (T, 3) corner ids with coincident bind-pose vertices merged, so split normals still touch
    crease: np.ndarray  # (T, J) bool: the triangle lies within the crease radius of the joint at rest
    clips: dict[str, Clip] = field(default_factory=dict)


def welded_corners(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """(T, 3) corner ids in which vertices that sit within 0.1 mm of each other in the bind pose (split normals, UV
    seams) are one, so triangles that meet across a seam count as touching."""
    _, weld = np.unique(np.round(positions / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    return weld.reshape(-1)[triangles]


def crease_zones(
    positions: np.ndarray,
    triangles: np.ndarray,
    inverse_bind: np.ndarray,
    joint_names: list[str],
    radius_m: float,
    non_crease_prefixes: tuple[str, ...],
) -> np.ndarray:
    """(T, J): triangle t lies near joint j when the nearest of its three corners and its centroid is within
    `radius_m` of the joint's bind position. Two triangles that share a joint here fold together at it when it
    bends. The columns of joints whose name starts with a `non_crease_prefixes` entry stay False: a skirt chain is
    not a hinge, and a crossing near one is a crossing."""
    joints = linalg.inverse_affine(inverse_bind)[:, :3, 3]
    corners = positions[triangles]
    points = np.concatenate([corners, corners.mean(1, keepdims=True)], 1)  # (T, 4, 3)
    near = np.zeros((len(triangles), len(joints)), bool)
    for start in range(0, len(joints), 8):  # a (T, 4, J, 3) array of the 65-joint figures would be gigabytes
        chunk = joints[start : start + 8]
        near[:, start : start + 8] = np.linalg.norm(points[:, :, None, :] - chunk[None, None], axis=-1).min(1) <= radius_m
    for j, name in enumerate(joint_names):
        if non_crease_prefixes and name.startswith(non_crease_prefixes):
            near[:, j] = False
    return near


def read_figure(name: str, path: Path, settings: CheckSettings) -> Figure:
    """A skinned glTF as a `Figure`: every skinned mesh on the file's one skin, its clips and its crease zones."""
    from . import gltf  # gltf imports Clip from here

    document, data = gltf.read_glb(path)
    nodes = document["nodes"]
    names = [node.get("name", f"node_{i}") for i, node in enumerate(nodes)]
    parent = np.full(len(nodes), -1)
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = i
    if any("matrix" in node for node in nodes):
        raise ValueError(f"{path.name}: a node stores a matrix; this checker reads translation/rotation/scale")
    skeleton = Skeleton(
        tuple(names),
        parent,
        np.array([node.get("translation", [0.0, 0.0, 0.0]) for node in nodes], float),
        np.array([node.get("rotation", [0.0, 0.0, 0.0, 1.0]) for node in nodes], float),
        np.array([node.get("scale", [1.0, 1.0, 1.0]) for node in nodes], float),
    )
    skinned = [node for node in nodes if "mesh" in node and "skin" in node]
    if not skinned or len({node["skin"] for node in skinned}) != 1:
        raise ValueError(f"{path.name}: needs skinned meshes on one skin")
    skin = document["skins"][skinned[0]["skin"]]
    joints = np.array(skin["joints"])
    inverse_bind = gltf.accessor(document, data, skin["inverseBindMatrices"]).astype(float).reshape(-1, 4, 4).transpose(0, 2, 1)
    joint_names = [names[j] for j in joints]
    positions, vertex_joints, weights, triangles, labels = [], [], [], [], []
    offset = 0
    for node in skinned:
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            attributes = primitive["attributes"]
            p = gltf.accessor(document, data, attributes["POSITION"]).astype(float)
            j = gltf.accessor(document, data, attributes["JOINTS_0"]).astype(int)
            w = gltf.accessor(document, data, attributes["WEIGHTS_0"]).astype(float)
            tri = gltf.accessor(document, data, primitive["indices"]).astype(int).reshape(-1, 3)
            material = document["materials"][primitive["material"]]["name"] if "material" in primitive else "mesh"
            for corner in tri:
                totals: dict[int, float] = {}
                for v in corner:
                    for k in range(4):
                        totals[j[v, k]] = totals.get(j[v, k], 0.0) + w[v, k]
                labels.append(f"{material}:{joint_names[max(totals, key=totals.__getitem__)]}")
            positions.append(p)
            vertex_joints.append(j)
            weights.append(w)
            triangles.append(tri + offset)
            offset += len(p)
    stacked = np.concatenate(positions)
    all_triangles = np.concatenate(triangles)
    return Figure(
        name,
        path,
        skeleton,
        joints,
        inverse_bind,
        stacked,
        np.concatenate(vertex_joints),
        np.concatenate(weights),
        all_triangles,
        labels,
        welded_corners(stacked, all_triangles),
        crease_zones(stacked, all_triangles, inverse_bind, joint_names, settings.crease_radius_m, settings.non_crease_prefixes),
        gltf.read_clips(document, data),
    )
