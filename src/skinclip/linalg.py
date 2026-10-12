"""The few small dense linear-algebra operations skinclip needs, without LAPACK.

numpy's `linalg.solve` and `linalg.inv` go through multi-threaded OpenBLAS/LAPACK, and on a loaded machine a 24 x 24 solve
measured 85 ms (11 microseconds with one thread): the threads spin against each other. Everything skinclip inverts is a
stack of affine 4 x 4 or rotation 3 x 3 matrices, or a tiny damped normal-equation system, so these do it with plain
array arithmetic.
"""

from __future__ import annotations

import numpy as np


def inverse3(m: np.ndarray) -> np.ndarray:
    """Inverses of 3 x 3 matrices (..., 3, 3) by the adjugate; a singular matrix raises `np.linalg.LinAlgError`."""
    m = np.asarray(m, float)
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    cofactors = np.stack(
        [
            np.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
            np.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
            np.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1),
        ],
        -2,
    )
    det = a * cofactors[..., 0, 0] + b * cofactors[..., 1, 0] + c * cofactors[..., 2, 0]
    if np.any(np.abs(det) < 1e-300):
        raise np.linalg.LinAlgError("singular matrix")
    return cofactors / det[..., None, None]


def inverse_affine(m: np.ndarray) -> np.ndarray:
    """Inverses of affine matrices (..., 4, 4) whose last row is (0, 0, 0, 1)."""
    m = np.asarray(m, float)
    linear = inverse3(m[..., :3, :3])
    out = np.zeros_like(m)
    out[..., :3, :3] = linear
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", linear, m[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The x of a x = b for a small square system (n of a few dozen) by Gauss-Jordan elimination with partial pivoting; a
    singular system raises `np.linalg.LinAlgError`."""
    n = len(b)
    work = np.concatenate([np.asarray(a, float), np.asarray(b, float)[:, None]], axis=1)
    for i in range(n):
        pivot = i + int(np.argmax(np.abs(work[i:, i])))
        if abs(work[pivot, i]) < 1e-300:
            raise np.linalg.LinAlgError("singular matrix")
        if pivot != i:
            work[[i, pivot]] = work[[pivot, i]]
        work[i] /= work[i, i]
        column = work[:, i].copy()
        column[i] = 0.0
        work -= column[:, None] * work[i]
    return work[:, n]
