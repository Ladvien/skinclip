from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skinclip import capsules, humanoid, pose, repair
from skinclip.intersect import CheckSettings, pose_cut
from skinclip.model import read_figure
from skinclip.testing import tube_humanoid


@pytest.fixture(scope="module")
def tube(hand_through_torso: Path):
    figure = read_figure("tube", hand_through_torso, CheckSettings())
    roles = humanoid.detect(figure.skeleton.names)
    clip = figure.clips["hand_through_torso"]
    rot = {joint: values[1] for joint, paths in clip.channels.items() for path, values in paths.items() if path == "rotation"}
    translation = clip.channels[roles["pelvis"]]["translation"][1]
    return figure, roles, capsules.measure(figure, roles), rot, translation


def test_repair_clears_a_hand_through_the_torso(tube):
    figure, roles, caps, rot, translation = tube
    fixed, report = repair.repair(figure, roles, caps, rot, translation, loop=False)
    assert report.worst_depth_m_before > 0.03
    assert report.worst_depth_m_after <= 0.002
    assert report.frames_repaired > 0
    assert set(report.joints_changed) <= {roles["upperarm_l"], roles["lowerarm_l"]}
    # the frame that crossed worst now passes the mesh check
    settings = CheckSettings()
    clip = figure.clips["hand_through_torso"]
    worst = int(np.argmax([pose_cut(figure, pose.skin(figure, m), settings).contour_m for m in pose.joint_matrices(figure, [(clip, np.arange(61) / 30)])]))
    repaired = pose.clip_from_frames("fixed", fixed, roles["pelvis"], translation)
    matrices = pose.joint_matrices(figure, [(repaired, np.array([worst / 30]))])[0]
    assert pose_cut(figure, pose.skin(figure, matrices), settings).contour_m < settings.contour_limit_m
    for joint, keys in rot.items():
        if joint not in report.joints_changed:
            assert fixed[joint] is keys or np.array_equal(fixed[joint], keys)


def test_a_repair_of_a_loop_ends_where_it_starts(tube):
    figure, roles, caps, rot, translation = tube
    # The same arm, swung down across the hip and back up: a loop whose first and last keys are the same.
    f = np.arange(61, dtype=float)
    half = -np.radians(100.0) * np.sin(np.pi * f / 60.0) ** 2 / 2.0
    looped = dict(rot)
    looped[roles["upperarm_l"]] = np.stack([np.zeros(61), np.zeros(61), np.sin(half), np.cos(half)], -1)
    fixed, report = repair.repair(figure, roles, caps, looped, translation, loop=True)
    assert report.worst_depth_m_before > 0.03
    assert report.worst_depth_m_after <= 0.002
    for joint in (roles["upperarm_l"], roles["lowerarm_l"]):
        assert np.allclose(fixed[joint][-1], fixed[joint][0], atol=1e-6)


def test_a_clip_that_crosses_nothing_comes_back_untouched(tmp_path: Path):
    path = tmp_path / "rest.glb"
    tube_humanoid(path, "ue5", "rest")
    figure = read_figure("tube", path, CheckSettings())
    roles = humanoid.detect(figure.skeleton.names)
    rot = {joint: np.tile([0.0, 0.0, 0.0, 1.0], (31, 1)) for joint in roles.values()}
    translation = np.tile(figure.skeleton.rest_t[figure.skeleton.index(roles["pelvis"])], (31, 1))
    fixed, report = repair.repair(figure, roles, capsules.measure(figure, roles), rot, translation, loop=False)
    assert report.frames_repaired == 0 and report.joints_changed == ()
    assert all(np.array_equal(fixed[j], rot[j]) for j in rot)


def test_a_half_space_keeps_a_hand_in_front_of_a_plane(tube):
    figure, roles, caps, rot, translation = tube
    # A wall just inside the hanging arm's reach: the hand must stay on the outer side of x = 0.2 (the +X normal)
    plane = repair.HalfSpace(roles["spine_03"], (1.0, 0.0, 0.0), 0.2, ("upperarm_l", "lowerarm_l", "hand_l"))
    fixed, report = repair.repair(figure, roles, caps, rot, translation, loop=False, half_spaces=(plane,))
    assert report.worst_depth_m_before > 0.03
    assert report.worst_depth_m_after <= 0.002


def test_humanoid_detects_both_conventions_and_refuses_a_stick(tmp_path: Path):
    for names, convention in (("ue5", humanoid.UE5), ("mixamo", humanoid.MIXAMO)):
        path = tmp_path / f"{names}.glb"
        tube_humanoid(path, names, "rest")
        figure = read_figure("tube", path, CheckSettings())
        assert humanoid.detect(figure.skeleton.names) == convention
        assert humanoid.convention(humanoid.detect(figure.skeleton.names)) == names
    with pytest.raises(ValueError, match="not a humanoid rig: missing"):
        humanoid.detect(["root", "pelvis", "spine_01"])


def test_body_capsules_of_the_tube_humanoid(tube):
    _, roles, caps, _, _ = tube
    assert len(caps) == 18
    assert caps["upperarm_l"].radius_m == pytest.approx(0.05, abs=0.003)
    assert caps["thigh_r"].radius_m == pytest.approx(0.07, abs=0.003)
    assert caps["hand_l"].end is None and caps["hand_l"].centroid_offset is not None


def test_a_hand_keeps_clear_of_an_obstacle_it_cannot_push(tube):
    figure, roles, caps, rot, translation = tube
    frames = len(translation)
    # A post planted where the swinging left forearm and hand would pass, beside the hip: 0.30 m out, 1.0 to 1.4 m up.
    post = (np.tile([0.30, 1.0, 0.0], (frames, 1)), np.tile([0.30, 1.4, 0.0], (frames, 1)), 0.04)
    rest_rot = {joint: np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)) for joint in rot}  # an arm held out in T never meets it
    _, none = repair.repair(figure, roles, caps, rest_rot, translation, loop=False, obstacles={"post": post})
    assert none.frames_repaired == 0
    hanging = dict(rest_rot)
    half = -np.radians(80.0) / 2.0
    hanging[roles["upperarm_l"]] = np.tile([0.0, 0.0, np.sin(half), np.cos(half)], (frames, 1))
    fixed, report = repair.repair(figure, roles, caps, hanging, translation, loop=False, obstacles={"post": post})
    assert report.worst_depth_m_before > 0.02 and report.worst_depth_m_after <= 0.002
    assert set(report.joints_changed) <= {roles["upperarm_l"], roles["lowerarm_l"]}
