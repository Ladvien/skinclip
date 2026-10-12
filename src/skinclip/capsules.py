"""Body capsules: each bone of a humanoid as a segment with a radius, measured from the mesh that wraps it.

A capsule is the cheapest stand-in for a limb that still answers "is this hand inside that torso, and by how much":
the distance between two segments (Ericson, *Real-Time Collision Detection*, 2005, 5.1.9) less the two radii is a
signed depth that has a gradient, which a mesh-triangle contour does not. `repair.py` pushes limbs out of each other
by that depth; the mesh check (`intersect.py`) stays the judge of whether it was enough.

A capsule runs from a joint to its child along the bone. A bone with no child worth the name (the head, the hand) uses
the centroid rule: its far end is the mirror of the start through the centroid of the vertices it owns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Figure

# Trunk and head are fat and a cap on them only guards against a cape or a wing weighted to a spine joint; a limb that
# measures wider than this is wrapping something that is not the limb.
MAX_TRUNK_RADIUS_M = 0.20
MAX_LIMB_RADIUS_M = 0.12
MIN_VERTICES = 8
SPARSE_RADIUS_M = 0.02
PERCENTILE = 90.0

# (capsule role, start role, end role or None for the centroid rule); `_s` expands to `_l` and `_r`.
_CAPSULE_ROLES = (
    ("pelvis", "pelvis", "spine_01"),
    ("spine_01", "spine_01", "spine_02"),
    ("spine_02", "spine_02", "spine_03"),
    ("spine_03", "spine_03", "neck"),
    ("neck", "neck", "head"),
    ("head", "head", None),
    ("upperarm_s", "upperarm_s", "lowerarm_s"),
    ("lowerarm_s", "lowerarm_s", "hand_s"),
    ("hand_s", "hand_s", None),
    ("thigh_s", "thigh_s", "calf_s"),
    ("calf_s", "calf_s", "foot_s"),
    ("foot_s", "foot_s", "ball_s"),
)
_TRUNK = ("pelvis", "spine_01", "spine_02", "spine_03", "neck", "head")


@dataclass(frozen=True)
class Capsule:
    name: str  # the role of the capsule
    start: str  # joint name
    end: str | None  # joint name; None for the centroid rule
    radius_m: float
    centroid_offset: np.ndarray | None  # the far end less the start, in the start joint's bind frame, for the centroid rule


def capsule_roles() -> list[tuple[str, str, str | None]]:
    """(capsule role, start role, end role) with the sides expanded."""
    out = []
    for name, start, end in _CAPSULE_ROLES:
        if name.endswith("_s"):
            for side in ("l", "r"):
                out.append((name[:-1] + side, start[:-1] + side, None if end is None else end[:-1] + side))
        else:
            out.append((name, start, end))
    return out


def _segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = b - a
    t = np.clip((points - a) @ axis / max(float(axis @ axis), 1e-18), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * axis), axis=1)


def measure(figure: Figure, roles: dict[str, str]) -> dict[str, Capsule]:
    """The capsules of a figure's bones, keyed by role (`upperarm_l`).

    Only vertices a triangle uses count. A vertex belongs to the capsule of the nearest ancestor-or-self of its dominant
    joint that starts one (so fingers count for the hand, eyes and jaw for the head). The radius is the 90th percentile
    of those vertices' distances to the bone segment in the bind pose, never above the cap for the kind of bone; a
    capsule with fewer than 8 vertices is 2 cm. A capsule whose joints the figure lacks is left out."""
    skeleton = figure.skeleton
    slot = {skeleton.names[node]: i for i, node in enumerate(figure.joints)}
    bind = np.linalg.inv(figure.inverse_bind)
    present = [
        (name, start, end)
        for name, start, end in capsule_roles()
        if roles.get(start) in slot and (end is None or roles.get(end) in slot)
    ]
    start_node = {skeleton.index(roles[start]): name for name, start, _ in present}
    used = np.unique(figure.triangles)
    dominant = figure.vertex_joints[used, np.argmax(figure.vertex_weights[used], axis=1)]
    owner = figure.joints[dominant]
    capsule_of: dict[int, str | None] = {}
    for node in np.unique(owner):
        walk = int(node)
        while walk >= 0 and walk not in start_node:
            walk = int(skeleton.parent[walk])
        capsule_of[int(node)] = start_node.get(walk)
    belongs = np.array([capsule_of[int(n)] for n in owner], dtype=object)
    out: dict[str, Capsule] = {}
    for name, start, end in present:
        vertices = figure.positions[used[belongs == name]]
        a = bind[slot[roles[start]], :3, 3]
        offset = None
        if end is None:
            if len(vertices) == 0:
                continue
            b = a + 2.0 * (vertices.mean(0) - a)
            offset = np.linalg.inv(bind[slot[roles[start]], :3, :3]) @ (b - a)
        else:
            b = bind[slot[roles[end]], :3, 3]
        if len(vertices) < MIN_VERTICES:
            radius = SPARSE_RADIUS_M
        else:
            cap = MAX_TRUNK_RADIUS_M if name in _TRUNK else MAX_LIMB_RADIUS_M
            radius = min(float(np.percentile(_segment_distances(vertices, a, b), PERCENTILE)), cap)
        out[name] = Capsule(name, roles[start], None if end is None else roles[end], radius, offset)
    return out


def positions(figure: Figure, worlds: np.ndarray, capsules: dict[str, Capsule]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per frame (F, 3) start and end of every capsule, from `pose.node_worlds` output (F, N, 4, 4). A centroid rule's
    end is the start joint's world matrix applied to the bind offset."""
    skeleton = figure.skeleton
    out = {}
    for name, capsule in capsules.items():
        start = worlds[:, skeleton.index(capsule.start)]
        if capsule.end is None:
            end = np.einsum("fij,j->fi", start[:, :3, :3], capsule.centroid_offset) + start[:, :3, 3]
        else:
            end = worlds[:, skeleton.index(capsule.end), :3, 3]
        out[name] = (start[:, :3, 3], end)
    return out
