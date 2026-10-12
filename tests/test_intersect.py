"""The checker obeys the relations a contour length must keep (Chen et al. 1998; Xie et al., JSS 2011): it does not
change when the whole pose moves rigidly or its triangles are listed in another order, it scales with the pose, it
never grows when the rest gap or the crease radius widens, a fold at a joint is not a crossing, cloth resting on cloth
is contact, and a beam pushed through a wall cuts a loop exactly its own perimeter long."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from skinclip.intersect import CheckSettings, candidate_pairs, contour_lengths, pose_cut
from skinclip.model import Figure, Skeleton, crease_zones, welded_corners

SETTINGS = CheckSettings()


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = [t for a, b, c, d in faces for t in ((a, b, c), (a, c, d))]
    return v, np.array(tris)


def _figure(parts: list[tuple[str, np.ndarray, np.ndarray]]) -> Figure:
    """A figure of one joint at the origin and rigid parts, so poses are plain vertex moves."""
    positions, triangles, labels, offset = [], [], [], 0
    for name, v, t in parts:
        positions.append(v)
        triangles.append(t + offset)
        labels += [f"{name}:root"] * len(t)
        offset += len(v)
    p, t = np.concatenate(positions), np.concatenate(triangles)
    n = len(p)
    skeleton = Skeleton(("root",), np.array([-1]), np.zeros((1, 3)), np.array([[0.0, 0, 0, 1]]), np.ones((1, 3)))
    return Figure(
        name="test", path=Path("test.glb"), skeleton=skeleton, joints=np.array([0]), inverse_bind=np.eye(4)[None], positions=p,
        vertex_joints=np.zeros((n, 4), int), vertex_weights=np.tile([1.0, 0, 0, 0], (n, 1)), triangles=t, triangle_label=labels,
        welded=welded_corners(p, t), crease=crease_zones(p, t, np.eye(4)[None], ["root"], 0.15, ()),
    )


WALL = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
BEAM = _box((2.0, 0.3, 0.3), (3.0, 0.7, 0.7))  # a 0.4 m square beam a metre clear of the wall
BEAM_PERIMETER = 1.6


def _beam_moved(figure: Figure, dx: float) -> np.ndarray:
    moved = figure.positions.copy()
    moved[len(WALL[0]):, 0] += dx
    return moved


@pytest.fixture(scope="module")
def wall_and_beam() -> Figure:
    return _figure([("Wall", *WALL), ("Beam", *BEAM)])


def test_a_beam_pushed_through_a_wall_cuts_a_loop_its_own_perimeter_long(wall_and_beam):
    assert pose_cut(wall_and_beam, wall_and_beam.positions, SETTINGS).contour_m == 0.0
    through = pose_cut(wall_and_beam, _beam_moved(wall_and_beam, -1.5), SETTINGS)  # beam spans x 0.5..1.5, crossing x = 1
    assert through.contour_m == pytest.approx(BEAM_PERIMETER, abs=1e-9)
    assert set(through.parts) == {("Beam:root", "Wall:root")}


def test_crossing_grows_from_nothing_as_the_beam_goes_in(wall_and_beam):
    depths = [pose_cut(wall_and_beam, _beam_moved(wall_and_beam, dx), SETTINGS).contour_m for dx in (-0.9, -1.0 + 1e-6, -1.1, -1.5, -1.9)]
    assert depths[0] == 0.0  # still clear of the wall
    assert all(d == pytest.approx(BEAM_PERIMETER, abs=1e-9) for d in depths[2:])  # one face crossed: the same loop


def test_parts_that_overlap_at_rest_may_keep_overlapping():
    # The beam already runs through the wall in the bind pose, like a mannequin's joint balls in its limbs.
    figure = _figure([("Wall", *WALL), ("Beam", BEAM[0] - [1.5, 0, 0], BEAM[1])])
    for dx in (0.0, 0.2, -0.2):
        moved = figure.positions.copy()
        moved[len(WALL[0]):, 0] += dx
        assert pose_cut(figure, moved, SETTINGS).contour_m == 0.0


def test_a_fold_at_a_joint_is_a_crease():
    # Two thin boxes on two joints, 10 cm apart at rest, that fold 150 degrees about the joint between them: box B
    # then runs 30 degrees off box A's axis, back over it, and they cross only within 8 cm of the joint.
    a, b = _box((0.0, -0.02, -0.02), (0.35, 0.02, 0.02)), _box((0.45, -0.02, -0.02), (0.8, 0.02, 0.02))
    p, t = np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1] + len(a[0])])
    inverse_bind = np.stack([np.eye(4), np.eye(4)])
    inverse_bind[1, 0, 3] = -0.4
    n = len(p)
    joint_of = np.zeros((n, 4), int)
    joint_of[len(a[0]):, 0] = 1
    skeleton = Skeleton(("j0", "j1"), np.array([-1, 0]), np.zeros((2, 3)), np.tile([0.0, 0, 0, 1], (2, 1)), np.ones((2, 3)))
    figure = Figure(
        name="test", path=Path("test.glb"), skeleton=skeleton, joints=np.array([0, 1]), inverse_bind=inverse_bind, positions=p,
        vertex_joints=joint_of, vertex_weights=np.tile([1.0, 0, 0, 0], (n, 1)), triangles=t,
        triangle_label=["A:j0"] * len(a[1]) + ["B:j1"] * len(b[1]), welded=welded_corners(p, t),
        crease=crease_zones(p, t, inverse_bind, ["j0", "j1"], 0.15, ()),
    )
    angle = np.radians(150.0)
    turn = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    folded = p.copy()
    folded[len(a[0]):] = (p[len(a[0]):] - [0.4, 0, 0]) @ turn.T + [0.4, 0, 0]
    assert pose_cut(figure, folded, SETTINGS).contour_m == 0.0
    assert pose_cut(dataclasses.replace(figure, crease=np.zeros_like(figure.crease)), folded, SETTINGS).contour_m > 0.0


def test_a_skirt_joint_is_not_a_crease():
    # The same fold, but joint j1 is a skirt chain: no hinge there, so the fold counts.
    a, b = _box((0.0, -0.02, -0.02), (0.35, 0.02, 0.02)), _box((0.45, -0.02, -0.02), (0.8, 0.02, 0.02))
    p, t = np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1] + len(a[0])])
    inverse_bind = np.stack([np.eye(4), np.eye(4)])
    inverse_bind[1, 0, 3] = -0.4
    creases = crease_zones(p, t, inverse_bind, ["j0", "skirt_00_a"], 0.15, ("skirt_",))
    assert creases[:, 0].any()
    assert not creases[:, 1].any()


def test_cloth_resting_on_cloth_is_contact(wall_and_beam):
    cloth = _figure([("MI_Robe", *WALL), ("MI_Robe", *BEAM)])
    settings = CheckSettings(cloth_materials=("MI_Robe",))
    assert pose_cut(cloth, _beam_moved(cloth, -1.5), settings).contour_m == 0.0
    assert pose_cut(cloth, _beam_moved(cloth, -1.5), SETTINGS).contour_m == pytest.approx(BEAM_PERIMETER, abs=1e-9)
    assert pose_cut(wall_and_beam, _beam_moved(wall_and_beam, -1.5), settings).contour_m == pytest.approx(BEAM_PERIMETER, abs=1e-9)


# ---- relations on a real frame that crosses (the tube humanoid's hand through its thigh and torso) ----


@pytest.fixture(scope="module")
def pose_that_crosses(crossing_pose):
    figure, vertices = crossing_pose
    assert pose_cut(figure, vertices, SETTINGS).contour_m > SETTINGS.contour_limit_m, "the relations below need a frame that crosses"
    return figure, vertices


def _rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_moving_the_whole_pose_rigidly_changes_nothing(pose_that_crosses):
    figure, vertices = pose_that_crosses
    before = pose_cut(figure, vertices, SETTINGS)
    after = pose_cut(figure, vertices @ _rotation(7).T + [3.0, -1.0, 12.0], SETTINGS)
    assert after.contour_m == pytest.approx(before.contour_m, rel=1e-6)
    assert set(after.parts) == set(before.parts)


def test_scaling_the_pose_scales_the_contour(pose_that_crosses):
    figure, vertices = pose_that_crosses
    assert pose_cut(figure, vertices * 2.0, SETTINGS).contour_m == pytest.approx(2.0 * pose_cut(figure, vertices, SETTINGS).contour_m, rel=1e-6)


def test_listing_triangles_in_another_order_changes_nothing(pose_that_crosses):
    figure, vertices = pose_that_crosses
    order = np.random.default_rng(11).permutation(len(figure.triangles))
    shuffled = dataclasses.replace(
        figure,
        triangles=figure.triangles[order],
        triangle_label=[figure.triangle_label[i] for i in order],
        welded=figure.welded[order],
        crease=figure.crease[order],
    )
    assert pose_cut(shuffled, vertices, SETTINGS).contour_m == pytest.approx(pose_cut(figure, vertices, SETTINGS).contour_m, rel=1e-9)


def test_a_wider_rest_gap_never_finds_more(pose_that_crosses):
    figure, vertices = pose_that_crosses
    contours = [pose_cut(figure, vertices, dataclasses.replace(SETTINGS, rest_gap_m=gap)).contour_m for gap in (0.0, 0.03, 0.06, 0.12, 0.24)]
    assert all(a >= b for a, b in zip(contours, contours[1:]))
    assert contours[0] > contours[-1]


def test_a_wider_crease_never_finds_more(pose_that_crosses):
    figure, vertices = pose_that_crosses
    names = [figure.skeleton.names[j] for j in figure.joints]
    contours = [
        pose_cut(dataclasses.replace(figure, crease=crease_zones(figure.positions, figure.triangles, figure.inverse_bind, names, radius, ())), vertices, SETTINGS).contour_m
        for radius in (0.0, 0.08, 0.15, 0.3)
    ]
    assert all(a >= b for a, b in zip(contours, contours[1:]))


def test_the_grid_misses_no_crossing(pose_that_crosses):
    figure, vertices = pose_that_crosses
    corners = vertices[figure.triangles]
    found = {tuple(p) for p in candidate_pairs(corners, figure.welded).tolist()}
    sample = np.random.default_rng(3).choice(len(corners), 400, replace=False)
    lo, hi = corners.min(1), corners.max(1)
    for i in sample:
        others = np.flatnonzero(np.all((lo[i] <= hi) & (lo <= hi[i]), axis=1) & ~np.isin(figure.welded, figure.welded[i]).any(1))
        crossing = others[contour_lengths(np.repeat(corners[i][None], len(others), 0), corners[others]) > 0]
        assert all((min(i, j), max(i, j)) in found for j in crossing), f"triangle {i}: the broad phase lost a crossing pair"
