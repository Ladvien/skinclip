from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skinclip import capsules, cloth, humanoid, pose, quat
from skinclip.intersect import CheckSettings, closest_points
from skinclip.model import Clip, Skeleton, read_figure
from skinclip.testing import tube_humanoid

SPEC = cloth.RingSkirtSpec(count=12, top_height_m=0.90, knee_height_m=0.55, hem_height_m=0.20, blend_height_m=0.94, inset_m=0.0)
FRAMES = 31


def _garment() -> np.ndarray:
    """A ring of vertices 0.18 m from the body's axis, 0.15 to 0.95 m high."""
    angles = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    heights = np.round(np.arange(0.15, 0.951, 0.05), 3)
    return np.array([[0.18 * np.sin(a), h, 0.18 * np.cos(a)] for h in heights for a in angles])


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("cloth") / "thigh_swing.glb"
    tube_humanoid(path, "ue5", "thigh_swing")
    figure = read_figure("tube", path, CheckSettings())
    roles = humanoid.detect(figure.skeleton.names)
    caps = capsules.measure(figure, roles)
    skirt = cloth.ring_skirt_layout(_garment(), SPEC, "y")
    clip = figure.clips["thigh_swing"]
    at = np.arange(FRAMES) / 30.0
    rest_worlds = pose.node_worlds(figure.skeleton, [(Clip("rest", 0.0, {}), np.zeros(1))])[0]
    pelvis = figure.skeleton.index(roles["pelvis"])
    # the skeleton with the chain nodes appended, parented to the pelvis
    nodes = cloth.chain_nodes(skirt, roles["pelvis"], rest_worlds[pelvis])
    names = list(figure.skeleton.names)
    parent, rest_t, rest_r = list(figure.skeleton.parent), list(figure.skeleton.rest_t), list(figure.skeleton.rest_r)
    for name, parent_name, translation, rotation in nodes:
        parent.append(names.index(parent_name))
        names.append(name)
        rest_t.append(translation)
        rest_r.append(rotation)
    skeleton = Skeleton(tuple(names), np.array(parent), np.array(rest_t), np.array(rest_r), np.ones((len(names), 3)))
    worlds = pose.node_worlds(skeleton, [(clip, at)])
    bind = pose.node_worlds(skeleton, [(Clip("rest", 0.0, {}), np.zeros(1))])[0]
    body = capsules.positions(figure, pose.node_worlds(figure.skeleton, [(clip, at)]), caps)
    tracks = {role: (start, end, caps[role].radius_m) for role, (start, end) in body.items()}
    keys = cloth.bake(skeleton, worlds, bind, skirt, tracks, loop=True, ground_m=0.02)
    return figure, roles, caps, skirt, skeleton, clip, at, keys, body


def test_the_layout_has_twelve_chains_on_the_garment():
    skirt = cloth.ring_skirt_layout(_garment(), SPEC, "y")
    assert len(skirt.chains) == 12
    assert skirt.chains[0].name_a == "skirt_00_a" and skirt.chains[11].name_b == "skirt_11_b"
    front = skirt.chains[0]  # theta = 0 is the front, +z in glTF
    assert front.a[2] == pytest.approx(0.18, abs=0.01) and abs(front.a[0]) < 0.01
    assert front.a[1] == pytest.approx(0.90) and front.b[1] == pytest.approx(0.55) and front.hem[1] == pytest.approx(0.20)
    quarter = skirt.chains[3]  # a quarter turn toward +x
    assert quarter.a[0] == pytest.approx(0.18, abs=0.01)


def test_a_blender_ring_has_its_front_on_minus_y():
    vertices = np.array([[0.18 * np.sin(a), -0.18 * np.cos(a), h] for h in np.round(np.arange(0.15, 0.951, 0.05), 3) for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)])
    skirt = cloth.ring_skirt_layout(vertices, SPEC, "z")
    assert skirt.chains[0].a[1] == pytest.approx(-0.18, abs=0.01)
    assert skirt.chains[0].a[2] == pytest.approx(0.90)


def test_a_garment_with_nothing_at_a_height_is_refused():
    with pytest.raises(ValueError, match="no garment at 0.20 m"):
        cloth.ring_skirt_layout(_garment()[_garment()[:, 1] > 0.5], SPEC, "y")


def test_weights_sum_to_one_with_at_most_four_influences():
    skirt = cloth.ring_skirt_layout(_garment(), SPEC, "y")
    vertices = _garment()
    existing = [{"pelvis": 0.5, "thigh_l": 0.3, "spine_01": 0.2} for _ in vertices]
    weights = cloth.ring_skirt_weights(vertices, existing, skirt)
    for vertex, entry, base in zip(vertices, weights, existing):
        assert sum(entry.values()) == pytest.approx(1.0) and len(entry) <= 4 and all(w > 0 for w in entry.values())
        if vertex[1] >= SPEC.blend_height_m:
            assert entry == base  # above the blend height: untouched
        elif vertex[1] < SPEC.top_height_m:
            assert all(name.startswith("skirt_") for name in entry)  # below the top: only chains
    hem = [w for v, w in zip(vertices, weights) if v[1] == pytest.approx(0.20)]
    assert all(name.endswith("_b") for entry in hem for name in entry)


def _chain_points(rig):
    figure, _, _, skirt, skeleton, clip, at, keys, _ = rig
    channels = dict(clip.channels)
    for name, values in keys.items():
        channels[name] = {"rotation": (at, values, "LINEAR")}
    worlds = pose.node_worlds(skeleton, [(Clip("baked", float(at[-1]), channels), at)])
    out = []
    for chain in skirt.chains:
        a, b = skeleton.index(chain.name_a), skeleton.index(chain.name_b)
        top, knee = worlds[:, a, :3, 3], worlds[:, b, :3, 3]
        hem = knee + np.einsum("fij,j->fi", worlds[:, b, :3, :3], chain.hem - chain.b)
        out.append((top, knee, hem))
    return out


def test_the_baked_chains_keep_off_the_thighs_and_calves(rig):
    _, _, caps, _, _, _, _, _, body = rig
    for top, knee, hem in _chain_points(rig):
        for role in ("thigh_l", "thigh_r", "calf_l", "calf_r"):
            start, end = body[role]
            for p, q in ((top, knee), (knee, hem)):
                _, _, c1, c2 = closest_points(p, q, start, end)
                clearance = np.linalg.norm(c1 - c2, axis=1) - caps[role].radius_m
                assert clearance.min() >= -0.005, f"{role}: a chain segment is {clearance.min() * 100:.1f} cm into the capsule"


def test_the_chains_keep_their_lengths_and_the_hem_stays_off_the_floor(rig):
    _, _, _, skirt, _, _, _, _, _ = rig
    for chain, (top, knee, hem) in zip(skirt.chains, _chain_points(rig)):
        assert np.allclose(np.linalg.norm(knee - top, axis=1), np.linalg.norm(chain.b - chain.a), atol=1e-5)
        assert np.allclose(np.linalg.norm(hem - knee, axis=1), np.linalg.norm(chain.hem - chain.b), atol=1e-5)
        assert hem[:, 1].min() >= 0.02 - 1e-6


def test_a_baked_loop_ends_where_it_starts(rig):
    keys = rig[7]
    assert len(keys) == 24
    for name, values in keys.items():
        assert np.allclose(values[-1], values[0], atol=1e-6) or np.allclose(values[-1], -values[0], atol=1e-6), name


def test_the_skirt_actually_moves_under_the_swinging_legs(rig):
    keys = rig[7]
    spread = max(float(np.abs(values - values[0]).max()) for values in keys.values())
    assert spread > 0.02  # the chains are not simply carried rigidly


def test_chain_nodes_are_world_aligned_at_rest(rig):
    _, _, _, _, skeleton, _, _, _, _ = rig
    worlds = pose.node_worlds(skeleton, [(Clip("rest", 0.0, {}), np.zeros(1))])[0]
    for name in ("skirt_00_a", "skirt_05_b"):
        rotation = quat.from_matrix(worlds[skeleton.index(name), :3, :3])
        assert np.allclose(rotation, quat.IDENTITY, atol=1e-6) or np.allclose(rotation, -quat.IDENTITY, atol=1e-6)
