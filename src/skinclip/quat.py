"""Quaternions in glTF order (x, y, z, w), vectorised over leading axes, numpy only.

scipy's `Rotation` is not available in Blender's Python, so the few operations skinclip needs live here. Every
function takes arrays shaped (..., 4) (or (..., 3) for vectors) and broadcasts over the leading axes.
"""

from __future__ import annotations

import numpy as np

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Hamilton product a * b: b's rotation first, then a's."""
    ax, ay, az, aw = (a[..., k] for k in range(4))
    bx, by, bz, bw = (b[..., k] for k in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vector v turned by the unit quaternion q."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def exp_map(rotvec: np.ndarray) -> np.ndarray:
    """The unit quaternion of a rotation vector (axis times angle, radians)."""
    rotvec = np.asarray(rotvec, float)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = angle / 2.0
    # sin(half) / angle, with its series 1/2 - angle^2 / 48 near zero
    scale = np.where(angle < 1e-8, 0.5 - angle * angle / 48.0, np.sin(half) / np.where(angle < 1e-8, 1.0, angle))
    return np.concatenate([rotvec * scale, np.cos(half)], axis=-1)


def arc(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The shortest-arc rotation taking unit vector u onto unit vector v."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    d = np.sum(u * v, axis=-1, keepdims=True)
    q = np.concatenate([np.cross(u, v), 1.0 + d], axis=-1)
    # Antiparallel: any axis perpendicular to u turns it half way round.
    flip = (1.0 + d)[..., 0] < 1e-9
    if np.any(flip):
        helper = np.where((np.abs(u[..., 0]) < 0.9)[..., None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        axis = np.cross(u, helper)
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        q = np.where(flip[..., None], np.concatenate([axis, np.zeros_like(d)], axis=-1), q)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_matrix(m: np.ndarray) -> np.ndarray:
    """The unit quaternion (w >= 0) of 3x3 rotation matrices (..., 3, 3), by Shepperd's method: the largest of the
    four diagonal combinations is the one whose division is safe (Shepperd, "Quaternion from rotation matrix", 1978)."""
    m = np.asarray(m, float)
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    cases = np.stack(
        [
            np.stack([1 + m00 - m11 - m22, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
            np.stack([m[..., 0, 1] + m[..., 1, 0], 1 - m00 + m11 - m22, m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
            np.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m00 - m11 + m22, m[..., 1, 0] - m[..., 0, 1]], -1),
            np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1], 1 + m00 + m11 + m22], -1),
        ],
        axis=-2,
    )  # (..., 4 cases, 4)
    pivot = np.stack([1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22, 1 + m00 + m11 + m22], -1)
    pick = np.argmax(pivot, axis=-1)
    q = np.take_along_axis(cases, pick[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., 3:4] < 0, -q, q)


def twist(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The twist of q about a unit axis in its swing-twist decomposition q = swing * twist (identity when the swing
    is a half turn and leaves no twist to read)."""
    axis = np.asarray(axis, float)
    along = np.sum(q[..., :3] * axis, axis=-1, keepdims=True)
    t = np.concatenate([axis * along, q[..., 3:4]], axis=-1)
    norm = np.linalg.norm(t, axis=-1, keepdims=True)
    return np.where(norm < 1e-9, IDENTITY, t / np.where(norm < 1e-9, 1.0, norm))


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrices (..., 3, 3) of quaternions (..., 4); the input is normalised first."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-arc spherical interpolation of unit quaternions (..., 4). `t` is shaped like the leading axes of `a`
    (or with a trailing 1) and may be a scalar."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a * b, -1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    t = np.asarray(t, float)
    if t.ndim == a.ndim - 1:
        t = t[..., None]
    near = sin < 1e-6
    wa = np.where(near, 1 - t, np.sin((1 - t) * theta) / np.where(near, 1.0, sin))
    wb = np.where(near, t, np.sin(t * theta) / np.where(near, 1.0, sin))
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def trs_matrix(t: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """4x4 matrices (..., 4, 4) of translation t, rotation r (quaternion) and scale s."""
    m = np.zeros(t.shape[:-1] + (4, 4))
    m[..., :3, :3] = quat_matrix(r) * s[..., None, :]
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m
