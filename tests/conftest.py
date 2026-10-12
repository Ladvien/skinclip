from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skinclip import pose
from skinclip.intersect import CheckSettings
from skinclip.model import Figure, read_figure
from skinclip.testing import tube_humanoid


@pytest.fixture(scope="session")
def tube_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tubes")


@pytest.fixture(scope="session")
def hand_through_torso(tube_dir: Path) -> Path:
    path = tube_dir / "hand_through_torso.glb"
    tube_humanoid(path, "ue5", "hand_through_torso")
    return path


@pytest.fixture(scope="session")
def crossing_pose(hand_through_torso: Path) -> tuple[Figure, np.ndarray]:
    """Frame 45 of the tube humanoid with its hand through its thigh and torso."""
    figure = read_figure("tube", hand_through_torso, CheckSettings())
    clip = figure.clips["hand_through_torso"]
    vertices = pose.skin(figure, pose.joint_matrices(figure, [(clip, np.array([45 / 30]))])[0])
    return figure, vertices
