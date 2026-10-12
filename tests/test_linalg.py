from __future__ import annotations

import numpy as np
import pytest

from skinclip import linalg, quat


def test_affine_inverses_match_numpy_for_rotations_with_scale():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(7, 4))
    m = np.zeros((7, 4, 4))
    m[:, :3, :3] = quat.quat_matrix(q) * rng.uniform(0.5, 2.0, size=(7, 1, 3))
    m[:, :3, 3] = rng.normal(size=(7, 3))
    m[:, 3, 3] = 1.0
    assert np.allclose(linalg.inverse_affine(m), np.linalg.inv(m), atol=1e-10)
    assert np.allclose(linalg.inverse_affine(m) @ m, np.eye(4), atol=1e-10)


def test_solve_matches_numpy_on_a_damped_normal_system_and_pivots():
    rng = np.random.default_rng(2)
    jac = rng.normal(size=(24, 40))
    hessian = jac @ jac.T
    a = hessian + 0.1 * np.diag(np.diag(hessian))
    b = rng.normal(size=24)
    assert np.allclose(linalg.solve(a, b), np.linalg.solve(a, b), atol=1e-9)
    swapped = np.array([[0.0, 1.0], [1.0, 0.0]])  # a zero on the diagonal needs the pivot row
    assert np.allclose(linalg.solve(swapped, np.array([2.0, 3.0])), [3.0, 2.0])


def test_singular_matrices_are_refused():
    with pytest.raises(np.linalg.LinAlgError):
        linalg.solve(np.zeros((3, 3)), np.ones(3))
    with pytest.raises(np.linalg.LinAlgError):
        linalg.inverse3(np.zeros((3, 3)))
