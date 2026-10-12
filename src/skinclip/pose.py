"""Posing a skinned figure the way a glTF renderer does: sampled clips, node world matrices, linear blend skinning."""

from __future__ import annotations

import numpy as np

from .model import Clip, Figure, Skeleton
from .quat import slerp, trs_matrix


def sample(channel: tuple[np.ndarray, np.ndarray, str], at: np.ndarray, path: str) -> np.ndarray:
    times, values, interpolation = channel
    if interpolation == "CUBICSPLINE":
        values = values.reshape(len(times), 3, -1)[:, 1]  # the key values; frames are sampled at the keys' rate
    if len(times) == 1:
        return np.repeat(values[:1], len(at), 0)
    i = np.clip(np.searchsorted(times, at, side="right") - 1, 0, len(times) - 2)
    if interpolation == "STEP":
        return values[np.where(at >= times[i + 1], i + 1, i)]
    u = np.clip((at - times[i]) / np.maximum(times[i + 1] - times[i], 1e-9), 0.0, 1.0)
    if path == "rotation":
        return slerp(values[i], values[i + 1], u)
    return values[i] + (values[i + 1] - values[i]) * u[:, None]


def node_worlds(skeleton: Skeleton, layers: list[tuple[Clip, np.ndarray]]) -> np.ndarray:
    """(F, N, 4, 4) world matrix of every node. Later layers override earlier ones on the nodes they animate."""
    frames = len(layers[0][1])
    n = len(skeleton.names)
    t = np.repeat(skeleton.rest_t[None], frames, 0)
    r = np.repeat(skeleton.rest_r[None], frames, 0)
    s = np.repeat(skeleton.rest_s[None], frames, 0)
    index = {name: i for i, name in enumerate(skeleton.names)}
    for clip, at in layers:
        for node, paths in clip.channels.items():
            i = index.get(node)
            if i is None:
                continue
            for path, channel in paths.items():
                {"translation": t, "rotation": r, "scale": s}[path][:, i] = sample(channel, at, path)
    local = trs_matrix(t, r, s)
    world = np.empty_like(local)
    done = np.zeros(n, bool)

    def resolve(i: int) -> None:
        if done[i]:
            return
        p = skeleton.parent[i]
        if p >= 0:
            resolve(p)
            world[:, i] = world[:, p] @ local[:, i]
        else:
            world[:, i] = local[:, i]
        done[i] = True

    for i in range(n):
        resolve(i)
    return world


def joint_matrices(figure: Figure, layers: list[tuple[Clip, np.ndarray]]) -> np.ndarray:
    """(F, J, 4, 4) skinning matrices: each joint's world matrix times its inverse bind matrix."""
    return node_worlds(figure.skeleton, layers)[:, figure.joints] @ figure.inverse_bind[None]


def skin(figure: Figure, matrices: np.ndarray) -> np.ndarray:
    """Vertex positions for one frame's (J, 4, 4) skinning matrices, as Bevy's linear blend skinning places them."""
    blended = np.einsum("vk,vkij->vij", figure.vertex_weights, matrices[figure.vertex_joints])
    return np.einsum("vij,vj->vi", blended[:, :3, :3], figure.positions) + blended[:, :3, 3]


def clip_from_frames(name: str, rotations: dict[str, np.ndarray], root: str, root_translation: np.ndarray, fps: float = 30.0) -> Clip:
    """A clip from per-joint rotation keys (frames, 4) and the root joint's translation (frames, 3), at `fps` from zero."""
    times = np.arange(len(root_translation)) / fps
    channels: dict[str, dict[str, tuple[np.ndarray, np.ndarray, str]]] = {
        joint: {"rotation": (times, np.asarray(keys, float), "LINEAR")} for joint, keys in rotations.items()
    }
    channels.setdefault(root, {})["translation"] = (times, np.asarray(root_translation, float), "LINEAR")
    return Clip(name, float(times[-1]), channels)
