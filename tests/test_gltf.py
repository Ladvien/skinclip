from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skinclip import gltf
from skinclip.intersect import CheckSettings
from skinclip.model import read_figure
from skinclip.testing import tube_humanoid


def _keys(frames: int) -> np.ndarray:
    angle = np.linspace(0.0, 1.0, frames)
    q = np.stack([np.sin(angle / 2), np.zeros(frames), np.zeros(frames), np.cos(angle / 2)], -1)
    return q.astype("<f4")


def test_write_rotations_replaces_a_channel_and_adds_one(tube_dir: Path, hand_through_torso: Path):
    out = tube_dir / "written.glb"
    frames = 61
    replaced, added = _keys(frames), _keys(frames)[::-1].copy()
    gltf.write_rotations(hand_through_torso, out, 0, {"upperarm_l": replaced, "root": added})
    document, data = gltf.read_glb(out)
    clip = gltf.read_clips(document, data)["hand_through_torso"]
    assert np.abs(clip.channels["upperarm_l"]["rotation"][1] - replaced).max() < 1e-6
    assert np.abs(clip.channels["root"]["rotation"][1] - added).max() < 1e-6
    # the other joints and the file as a figure are untouched
    original = read_figure("tube", hand_through_torso, CheckSettings())
    again = read_figure("tube", out, CheckSettings())
    assert np.abs(again.clips["hand_through_torso"].channels["pelvis"]["rotation"][1] - original.clips["hand_through_torso"].channels["pelvis"]["rotation"][1]).max() == 0
    assert np.abs(again.positions - original.positions).max() == 0


def test_a_key_count_that_differs_from_the_times_is_refused(tube_dir: Path, hand_through_torso: Path):
    with pytest.raises(ValueError, match="upperarm_l: 5 keys but the animation has 61 times"):
        gltf.write_rotations(hand_through_torso, tube_dir / "bad.glb", 0, {"upperarm_l": _keys(5)})


def test_a_node_that_does_not_exist_is_refused(tube_dir: Path, hand_through_torso: Path):
    with pytest.raises(ValueError, match="no node of that name"):
        gltf.write_rotations(hand_through_torso, tube_dir / "bad.glb", 0, {"tail": _keys(61)})
