"""Test fixtures shipped with the package, so blended's and the Kimodo MCP's tests build the same figure skinclip's do.

`tube_humanoid` writes a skinned glTF: the 14-role humanoid (22 joints) of `humanoid.py` in a T pose, Y-up and +Z
forward at about 1.75 m, every bone a closed 12-sided tube weighted wholly to it. Written with the stdlib and numpy only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from . import humanoid
from .gltf import write_glb

FPS = 30
SIDES = 12
Names = Literal["ue5", "mixamo"]
ClipName = Literal["rest", "hand_through_torso", "thigh_swing"]

# role -> (parent role or None under the root node, world position at rest in metres, tube radius in metres).
# The tube of a bone runs from its joint to its first child's joint; a leaf bone's tube runs 8 cm along its own axis.
_CENTRAL = {
    "pelvis": (None, (0.0, 0.95, 0.0), 0.14),
    "spine_01": ("pelvis", (0.0, 1.03, 0.0), 0.14),
    "spine_02": ("spine_01", (0.0, 1.16, 0.0), 0.14),
    "spine_03": ("spine_02", (0.0, 1.30, 0.0), 0.14),
    "neck": ("spine_03", (0.0, 1.47, 0.0), 0.05),
    "head": ("neck", (0.0, 1.55, 0.0), 0.0),  # a sphere, not a tube
}
_SIDED = {  # x is for the left side; the right side mirrors it
    "clavicle": ("spine_03", (0.06, 1.42, 0.0), 0.04),
    "upperarm": ("clavicle", (0.22, 1.42, 0.0), 0.05),
    "lowerarm": ("upperarm", (0.52, 1.42, 0.0), 0.05),
    "hand": ("lowerarm", (0.79, 1.42, 0.0), 0.04),
    "thigh": ("pelvis", (0.09, 0.93, 0.0), 0.07),
    "calf": ("thigh", (0.09, 0.50, 0.0), 0.07),
    "foot": ("calf", (0.09, 0.08, 0.0), 0.045),
    "ball": ("foot", (0.09, 0.04, 0.12), 0.035),
}
HEAD_RADIUS_M = 0.11
LEAF_M = 0.08


def _rig() -> tuple[list[str], dict[str, tuple[str | None, np.ndarray, float]]]:
    """Roles in parent-before-child order, and role -> (parent role, world position, radius)."""
    rig: dict[str, tuple[str | None, np.ndarray, float]] = {}
    for role, (parent, position, radius) in _CENTRAL.items():
        rig[role] = (parent, np.array(position), radius)
    for role, (parent, position, radius) in _SIDED.items():
        for side, sign in (("l", 1.0), ("r", -1.0)):
            mirrored = np.array([position[0] * sign, position[1], position[2]])
            rig[f"{role}_{side}"] = (f"{parent}_{side}" if parent in _SIDED else parent, mirrored, radius)
    return list(rig), rig


def _tube(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed tube from a to b: two rings of SIDES vertices, a cap centre at each end; outward-facing triangles."""
    axis = (b - a) / np.linalg.norm(b - a)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = 2 * np.pi * np.arange(SIDES) / SIDES
    ring = radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    vertices = np.concatenate([a + ring, b + ring, a[None], b[None]])
    triangles = []
    for k in range(SIDES):
        n = (k + 1) % SIDES
        triangles += [(k, n, SIDES + k), (n, SIDES + n, SIDES + k)]
        triangles += [(2 * SIDES, n, k), (2 * SIDES + 1, SIDES + k, SIDES + n)]
    return vertices, np.array(triangles)


def _sphere(centre: np.ndarray, radius: float, rings: int = 8) -> tuple[np.ndarray, np.ndarray]:
    vertices = [centre + [0.0, radius, 0.0], centre + [0.0, -radius, 0.0]]
    for r in range(1, rings):
        polar = np.pi * r / rings
        for k in range(SIDES):
            azimuth = 2 * np.pi * k / SIDES
            vertices.append(centre + radius * np.array([np.sin(polar) * np.cos(azimuth), np.cos(polar), np.sin(polar) * np.sin(azimuth)]))
    triangles = []
    for k in range(SIDES):
        n = (k + 1) % SIDES
        triangles.append((0, 2 + n, 2 + k))
        last = 2 + (rings - 2) * SIDES
        triangles.append((1, last + k, last + n))
        for r in range(rings - 2):
            a, b = 2 + r * SIDES, 2 + (r + 1) * SIDES
            triangles += [(a + k, a + n, b + k), (a + n, b + n, b + k)]
    return np.array(vertices), np.array(triangles)


def _rotation_x(angle: np.ndarray) -> np.ndarray:
    half = np.asarray(angle, float) / 2.0
    return np.stack([np.sin(half), np.zeros_like(half), np.zeros_like(half), np.cos(half)], -1)


def _rotation_z(angle: np.ndarray) -> np.ndarray:
    half = np.asarray(angle, float) / 2.0
    return np.stack([np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)], -1)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _clip_rotations(clip: ClipName, roles: list[str]) -> tuple[float, dict[str, np.ndarray]]:
    """(seconds, role -> (frames, 4) local rotation keys) of a test clip. Every joint is keyed, as Kimodo's are."""
    seconds = {"rest": 1.0, "hand_through_torso": 2.0, "thigh_swing": 1.0}[clip]
    frames = round(seconds * FPS) + 1
    f = np.arange(frames, dtype=float)
    keys = {role: np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1)) for role in roles}
    if clip == "hand_through_torso":
        # The left arm hangs from the T pose down and on across the thigh and hip: about its local Z, 0 to -100 degrees
        # (negative turns the +X arm downward) in the first second, held for the second.
        keys["upperarm_l"] = _rotation_z(-np.radians(100.0) * _smoothstep(f / 30.0))
    elif clip == "thigh_swing":
        swing = np.radians(35.0) * np.sin(2 * np.pi * f / FPS)
        keys["thigh_l"], keys["thigh_r"] = _rotation_x(swing), _rotation_x(-swing)
    return seconds, keys


def tube_humanoid(path: Path, names: Names, clip: ClipName) -> None:
    """Writes the tube humanoid with one animation, `clip`, to `path`.

    rest: one second of identity. hand_through_torso: two seconds in which the left arm swings from the T pose down
    across the hip (a hand through the thigh and torso from about frame 25 on). thigh_swing: a one-second loop, the
    thighs 35 degrees either way, out of phase (a skirt ring around the legs is crossed by a knee)."""
    mapping = humanoid.CONVENTIONS[names]
    roles, rig = _rig()
    node_index = {role: 2 + i for i, role in enumerate(roles)}  # node 0 is the root, node 1 the skinned mesh
    children: dict[int, list[int]] = {0: []}
    nodes: list[dict] = [{"name": "root"}, {"name": "body", "mesh": 0, "skin": 0}]
    for role in roles:
        parent, position, _ = rig[role]
        parent_position = rig[parent][1] if parent else np.zeros(3)
        nodes.append({"name": mapping[role], "translation": [float(x) for x in position - parent_position]})
        children.setdefault(node_index[parent] if parent else 0, []).append(node_index[role])
    for index, kids in children.items():
        nodes[index]["children"] = kids

    positions, joints, triangles, offset = [], [], [], 0
    kids_of: dict[str, list[str]] = {}
    for role in roles:
        parent = rig[role][0]
        if parent:
            kids_of.setdefault(parent, []).append(role)
    for slot, role in enumerate(roles):
        _, position, radius = rig[role]
        if role == "head":
            vertices, tris = _sphere(position + [0.0, HEAD_RADIUS_M, 0.0], HEAD_RADIUS_M)
        else:
            if role in ("pelvis",):
                start = position + [0.0, -0.07, 0.0]  # the hips reach below the pelvis joint
            else:
                start = position
            if kids_of.get(role):
                # the bone's own chain child: the one in its chain (the clavicle's is the upperarm, the foot's the ball)
                end = rig[kids_of[role][0]][1]
            else:
                direction = position - rig[rig[role][0]][1]  # a leaf: along the bone's own axis
                end = position + direction / np.linalg.norm(direction) * LEAF_M
            vertices, tris = _tube(start, end, radius)
        positions.append(vertices)
        joints.append(np.full(len(vertices), slot))
        triangles.append(tris + offset)
        offset += len(vertices)
    position_array = np.concatenate(positions).astype("<f4")
    vertex_count = len(position_array)
    joint_array = np.zeros((vertex_count, 4), "<u2")
    joint_array[:, 0] = np.concatenate(joints)
    weight_array = np.zeros((vertex_count, 4), "<f4")
    weight_array[:, 0] = 1.0
    index_array = np.concatenate(triangles).astype("<u4").reshape(-1)

    inverse_bind = np.stack(
        [np.linalg.inv(np.block([[np.eye(3), rig[r][1][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])).T for r in roles]
    ).astype("<f4")

    seconds, keys = _clip_rotations(clip, roles)
    frames = round(seconds * FPS) + 1
    times = (np.arange(frames) / FPS).astype("<f4")
    pelvis_translation = np.tile(np.array(nodes[node_index["pelvis"]]["translation"]), (frames, 1)).astype("<f4")

    binary = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add(array: np.ndarray, component: int, kind: str, **extra: object) -> int:
        binary.extend(b"\0" * (-len(binary) % 4))
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": array.nbytes})
        binary.extend(array.tobytes())
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[kind]
        accessors.append({"bufferView": len(views) - 1, "componentType": component, "count": array.size // width, "type": kind, **extra})
        return len(accessors) - 1

    position_accessor = add(
        position_array, 5126, "VEC3", min=[float(x) for x in position_array.min(0)], max=[float(x) for x in position_array.max(0)]
    )
    joint_accessor = add(joint_array, 5123, "VEC4")
    weight_accessor = add(weight_array, 5126, "VEC4")
    index_accessor = add(index_array, 5125, "SCALAR")
    ibm_accessor = add(inverse_bind.reshape(-1), 5126, "MAT4")
    time_accessor = add(times, 5126, "SCALAR", min=[float(times[0])], max=[float(times[-1])])
    channels, samplers = [], []
    for role in roles:
        samplers.append({"input": time_accessor, "output": add(keys[role].astype("<f4"), 5126, "VEC4"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node_index[role], "path": "rotation"}})
    samplers.append({"input": time_accessor, "output": add(pelvis_translation, 5126, "VEC3"), "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": node_index["pelvis"], "path": "translation"}})

    document = {
        "asset": {"generator": "skinclip.testing.tube_humanoid", "version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [
            {"name": "body", "primitives": [{"attributes": {"POSITION": position_accessor, "JOINTS_0": joint_accessor, "WEIGHTS_0": weight_accessor}, "indices": index_accessor, "material": 0}]}
        ],
        "materials": [{"name": "M_Body", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.7, 0.6, 1.0]}}],
        "skins": [{"joints": [node_index[r] for r in roles], "inverseBindMatrices": ibm_accessor}],
        "animations": [{"name": clip, "channels": channels, "samplers": samplers}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    write_glb(document, bytes(binary), path)
