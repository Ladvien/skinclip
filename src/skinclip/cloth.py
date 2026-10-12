"""Ring-skirt cloth chains: bones for a skirt, skin weights onto them, and an animation baked by a small simulation.

A skirt that is skinned by copying the legs' weights goes through the legs it hangs over. A ring of chains does not: the
garment is wrapped by `count` chains of two bones each (a top-to-knee bone `a` and a knee-to-hem bone `b`) that hang
from a pinned anchor joint, and the chains are simulated against the body's capsules and baked to rotation keys, so
the engine only plays bones and nothing runs at game time.

The simulation is position-based dynamics with Verlet integration (Jakobsen, *Advanced Character Physics*, GDC 2001;
for skinned characters Abu Rumman and Fratarcangeli, *Position-Based Skinning for Soft Articulated Characters*,
Computer Graphics Forum 2015, doi:10.1111/cgf.12533): each chain is one pinned particle (the top) and two free ones (the
knee and the hem) joined by rest-length constraints, pulled gently back toward the shape the anchor carries them in, held
to its neighbours by a ring limit, and pushed out of capsules by the segment-segment closest-point test (Ericson,
*Real-Time Collision Detection*, 2005, 5.1.9). The literature search for secondary motion of skirt bones returned the
two cloth-and-skinning references above and no skirt-bone paper, so the layout and the baking of particle positions to
bone rotations are skinclip's own.

Frame convention: `up` is "y" (glTF) or "z" (Blender). The plane basis (u, v) is (+x, +z) for "y" and (+x, -y) for "z"
(the figure faces +z, or -y). A chain angle theta places a point at `centre + r * (sin(theta) u + cos(theta) v)`:
theta = 0 is the front, growing toward +x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import linalg, quat
from .intersect import closest_points
from .model import Skeleton

Up = Literal["y", "z"]

BAND_M = 0.02
WIDE_BAND_M = 0.04
MIN_BAND_VERTICES = 16
MAX_INFLUENCES = 4

FPS = 30.0
SUBSTEPS = 4
ITERATIONS = 4
DAMPING = 0.92  # the share of the last step's velocity a particle keeps
PULL_KNEE = 0.08  # per iteration, toward the shape the anchor carries the chain in
PULL_HEM = 0.05
RING_LIMIT = 1.15  # neighbours' knees (and hems) may be this far apart, times their rest distance
CLEARANCE_M = 0.05  # a chain keeps this far off a capsule's surface (the cloth is thick, and a calf is wider than its capsule)
LOOP_PASSES = 4
SETTLE_FRAMES = 30
MIN_LEVER = 0.25  # a collision near a chain's pinned top moves its knee by at most 4 times the push
COLLIDERS = ("thigh", "calf", "foot", "hand", "lowerarm")


@dataclass(frozen=True, kw_only=True)
class RingSkirtSpec:
    count: int = 12
    top_height_m: float
    knee_height_m: float
    hem_height_m: float
    blend_height_m: float
    inset_m: float
    prefix: str = "skirt_"


@dataclass(frozen=True)
class Chain:
    name_a: str
    name_b: str
    a: np.ndarray  # top, world position at rest
    b: np.ndarray  # knee
    hem: np.ndarray


@dataclass(frozen=True)
class RingSkirt:
    spec: RingSkirtSpec
    up: Up
    chains: tuple[Chain, ...]
    centres: np.ndarray  # (3, 3): the ring's centre at the top, knee and hem heights


def _frame(up: Up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The plane basis u, v and the up axis w, as unit 3-vectors."""
    if up == "y":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    if up == "z":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    raise ValueError(f"up is 'y' or 'z', not {up!r}")


def _angles(du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """The chain angle in [0, 2 pi) of planar offsets (u, v) from a centre."""
    return np.arctan2(du, dv) % (2.0 * np.pi)


def ring_skirt_layout(vertices: np.ndarray, spec: RingSkirtSpec, up: Up) -> RingSkirt:
    """Where the chains of a ring skirt hang: one chain per sector of the garment, its top, knee and hem points on the
    garment's surface (less `inset_m`) at the spec's three heights."""
    u, v, w = _frame(up)
    height = vertices @ w
    pu, pv = vertices @ u, vertices @ v
    count = spec.count
    sector = 2.0 * np.pi / count
    heights = (spec.top_height_m, spec.knee_height_m, spec.hem_height_m)
    radii = np.zeros((3, count))
    centres = np.zeros((3, 3))
    for row, h in enumerate(heights):
        band = np.abs(height - h) <= BAND_M
        if band.sum() < MIN_BAND_VERTICES:
            band = np.abs(height - h) <= WIDE_BAND_M
        if band.sum() < MIN_BAND_VERTICES:
            raise ValueError(f"no garment at {h:.2f} m")
        bu, bv = pu[band], pv[band]
        cu, cv = (bu.min() + bu.max()) / 2.0, (bv.min() + bv.max()) / 2.0
        centres[row] = cu * u + cv * v + h * w
        du, dv = bu - cu, bv - cv
        distance, theta = np.hypot(du, dv), _angles(du, dv)
        found = np.zeros(count, bool)
        for i in range(count):
            offset = (theta - i * sector + np.pi) % (2.0 * np.pi) - np.pi
            inside = np.abs(offset) <= sector / 2.0
            if inside.any():
                radii[row, i] = distance[inside].max()
                found[i] = True
        for i in np.flatnonzero(~found):  # an empty sector takes its neighbours' mean
            near = [radii[row, j % count] for j in (i - 1, i + 1) if found[j % count]]
            radii[row, i] = float(np.mean(near)) if near else float(radii[row][found].mean())
        radii[row] -= spec.inset_m
    chains = []
    for i in range(count):
        theta = i * sector
        direction = np.sin(theta) * u + np.cos(theta) * v
        points = [centres[row] + radii[row, i] * direction for row in range(3)]
        chains.append(Chain(f"{spec.prefix}{i:02d}_a", f"{spec.prefix}{i:02d}_b", points[0], points[1], points[2]))
    return RingSkirt(spec, up, tuple(chains), centres)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ring_skirt_weights(vertices: np.ndarray, existing: list[dict[str, float]], skirt: RingSkirt) -> list[dict[str, float]]:
    """Skin weights of `vertices` with the skirt's chains in them. A vertex below the spec's blend height takes the two
    chains that flank its angle, the top bone near the top and the knee bone toward the hem; between the top and the
    blend height the skirt's weights are mixed with `existing` by a smoothstep (all skirt below the top). The result keeps
    the strongest four influences, normalised. Other vertices keep their `existing` entry."""
    spec = skirt.spec
    u, v, w = _frame(skirt.up)
    height = vertices @ w
    # the ring's centre at each vertex's height, in the plane
    order = [2, 1, 0]  # hem, knee, top: ascending heights
    ascending = np.array([spec.hem_height_m, spec.knee_height_m, spec.top_height_m])
    cu = np.interp(height, ascending, [skirt.centres[i] @ u for i in order])
    cv = np.interp(height, ascending, [skirt.centres[i] @ v for i in order])
    theta = _angles(vertices @ u - cu, vertices @ v - cv)
    count = spec.count
    scaled = theta / (2.0 * np.pi / count)
    k = np.floor(scaled).astype(int) % count
    t = scaled - np.floor(scaled)
    up_weight = np.clip((spec.top_height_m - height) / (spec.top_height_m - spec.knee_height_m), 0.0, 1.0)
    mix = np.where(
        height >= spec.top_height_m,
        _smoothstep((spec.blend_height_m - height) / (spec.blend_height_m - spec.top_height_m)),
        1.0,
    )
    out: list[dict[str, float]] = []
    for i, base in enumerate(existing):
        if height[i] >= spec.blend_height_m:
            out.append(base)
            continue
        k0, k1 = int(k[i]), (int(k[i]) + 1) % count
        a0, a1 = skirt.chains[k0].name_a, skirt.chains[k1].name_a
        b0, b1 = skirt.chains[k0].name_b, skirt.chains[k1].name_b
        s, m, up_w = float(t[i]), float(mix[i]), float(up_weight[i])
        skirt_weights = {a0: (1 - s) * (1 - up_w), a1: s * (1 - up_w), b0: (1 - s) * up_w, b1: s * up_w}
        merged: dict[str, float] = {name: (1.0 - m) * weight for name, weight in base.items()}
        for name, weight in skirt_weights.items():
            merged[name] = merged.get(name, 0.0) + m * weight
        top = sorted(((weight, name) for name, weight in merged.items() if weight > 0.0), reverse=True)[:MAX_INFLUENCES]
        total = sum(weight for weight, _ in top)
        out.append({name: weight / total for weight, name in top})
    return out


def chain_nodes(skirt: RingSkirt, anchor_name: str, anchor_bind_world: np.ndarray) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(name, parent name, rest translation, rest rotation xyzw) of every chain node for glTF splicing, a then b of each
    chain. The rest rotations are world-aligned: the world rotation of every chain node at rest is the identity, so a
    chain's baked keys are plain world-frame swings."""
    anchor_rotation = quat.from_matrix(_orthonormal(anchor_bind_world[:3, :3]))
    anchor_position = anchor_bind_world[:3, 3]
    inverse = quat.conjugate(anchor_rotation)
    nodes = []
    for chain in skirt.chains:
        nodes.append((chain.name_a, anchor_name, quat.rotate(inverse, chain.a - anchor_position), inverse))
        nodes.append((chain.name_b, chain.name_a, chain.b - chain.a, quat.IDENTITY.copy()))
    return nodes


def _orthonormal(m: np.ndarray) -> np.ndarray:
    """The rotation part of a matrix that may carry scale: its columns normalised."""
    return m / np.linalg.norm(m, axis=-2, keepdims=True)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class _Simulation:
    """One ring's particles: A pinned to its anchor's motion, B (knee) and T (hem) free."""

    def __init__(
        self,
        rigid_a: np.ndarray,
        rigid_b: np.ndarray,
        rigid_t: np.ndarray,
        rest_ab: np.ndarray,
        rest_bt: np.ndarray,
        rest_ring_b: np.ndarray,
        rest_ring_t: np.ndarray,
        tracks: list[tuple[np.ndarray, np.ndarray, float]],
        ground: tuple[int, float],
    ) -> None:
        self.rigid_a, self.rigid_b, self.rigid_t = rigid_a, rigid_b, rigid_t  # (F, C, 3)
        self.rest_ab, self.rest_bt = rest_ab, rest_bt
        self.rest_ring_b, self.rest_ring_t = rest_ring_b, rest_ring_t
        self.tracks = tracks  # per capsule (start (F, 3), end (F, 3), radius)
        self.ground = ground
        self.b = rigid_b[0].copy()
        self.t = rigid_t[0].copy()
        self.b_prev, self.t_prev = self.b.copy(), self.t.copy()

    def advance(self, frame: int, to_frame: int) -> None:
        """Steps from the pose at `frame` to the pose at `to_frame` (the same frame holds still)."""
        for sub in range(1, SUBSTEPS + 1):
            alpha = sub / SUBSTEPS
            lerp = lambda x: x[frame] + (x[to_frame] - x[frame]) * alpha  # noqa: E731
            a, rigid_b, rigid_t = lerp(self.rigid_a), lerp(self.rigid_b), lerp(self.rigid_t)
            colliders = [(lerp(start), lerp(end), radius) for start, end, radius in self.tracks]
            old_b, old_t = self.b.copy(), self.t.copy()
            self.b = self.b + (self.b - self.b_prev) * DAMPING
            self.t = self.t + (self.t - self.t_prev) * DAMPING
            self.b_prev, self.t_prev = old_b, old_t
            for _ in range(ITERATIONS):
                self._distances(a)
                self.b += (rigid_b - self.b) * PULL_KNEE
                self.t += (rigid_t - self.t) * PULL_HEM
                self._ring()
                self._collide(a, colliders)
                self._floor()
                self._distances(a)

    def _distances(self, a: np.ndarray) -> None:
        d = self.b - a
        self.b = a + _normalise(d) * self.rest_ab[:, None]
        d = self.t - self.b
        length = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
        correction = d * ((length - self.rest_bt[:, None]) / length) * 0.5
        self.b = self.b + correction
        self.t = self.t - correction

    def _ring(self) -> None:
        for points, rest in ((self.b, self.rest_ring_b), (self.t, self.rest_ring_t)):
            neighbour = np.roll(points, -1, axis=0)
            d = neighbour - points
            length = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
            excess = np.maximum(length - RING_LIMIT * rest[:, None], 0.0)
            correction = d / length * excess * 0.5
            points += correction
            neighbour_index = (np.arange(len(points)) + 1) % len(points)
            np.add.at(points, neighbour_index, -correction)

    def _collide(self, a: np.ndarray, colliders: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Pushes the chains out of every capsule, away from the body: horizontally outward from the ring's centre, over the
        capsule. The shortest way out of a capsule can be inward or sideways, and then the two chains either side of a
        leg go round it opposite ways and the cloth that joins them is stretched across it. Two kinds of segment are
        tested: the chains' own (top-knee, knee-hem) and the ones through the midpoints of neighbouring chains, which stand
        for the cloth stretched between them: a calf can stand between two chains 14 cm apart and still poke through."""
        count = len(a)
        around = np.roll(np.arange(count), -1)
        each = np.arange(count)
        up = self.ground[0]
        centre = a.mean(0)
        for start, end, radius in colliders:
            mid_a, mid_b, mid_t = (a + a[around]) / 2.0, (self.b + self.b[around]) / 2.0, (self.t + self.t[around]) / 2.0
            p1 = np.concatenate([a, self.b, mid_a, mid_b])
            q1 = np.concatenate([self.b, self.t, mid_b, mid_t])
            rows = len(p1)
            s, _, c1, c2 = closest_points(p1, q1, np.broadcast_to(start, (rows, 3)), np.broadcast_to(end, (rows, 3)))
            offset = c1 - c2
            reach = radius + CLEARANCE_M
            hit = np.einsum("pi,pi->p", offset, offset) < reach * reach
            if not hit.any():
                continue
            outward = c1 - centre
            outward[:, up] = 0.0
            outward /= np.maximum(np.linalg.norm(outward, axis=1), 1e-9)[:, None]
            # Slide along `outward` until the distance to the capsule's axis is `reach`: |offset + t outward| = reach.
            along = np.einsum("pi,pi->p", offset, outward)
            across = np.einsum("pi,pi->p", offset, offset) - along * along
            slide = np.maximum(-along + np.sqrt(np.maximum(reach * reach - across, 0.0)), 0.0)
            push = np.where(hit[:, None], outward * slide[:, None], 0.0)
            ab, bt, mab, mbt = (push[i * count : (i + 1) * count] for i in range(4))
            sab, sbt, smab, smbt = (s[i * count : (i + 1) * count, None] for i in range(4))
            # A is pinned: segment A-B moves B by what it takes to move the point at s by the push (s kept off zero), B-T
            # moves B by (1 - s) and T by s, both normalised so the point on the segment moves by the push.
            self.b = self.b + ab / np.maximum(sab, MIN_LEVER)
            norm = (1.0 - sbt) ** 2 + sbt**2
            self.b = self.b + bt * (1.0 - sbt) / norm
            self.t = self.t + bt * sbt / norm
            # Midpoint segments: a neighbouring pair's midpoint moves by the push when both move by it. A's are pinned, so
            # mid(A)-mid(B) moves both B's (the midpoint at s moves by s of their push); mid(B)-mid(T) moves both B's and T's.
            norm = (1.0 - smbt) ** 2 + smbt**2
            for index in (each, around):
                np.add.at(self.b, index, mab / np.maximum(smab, MIN_LEVER) + mbt * (1.0 - smbt) / norm)
                np.add.at(self.t, index, mbt * smbt / norm)

    def _floor(self) -> None:
        axis, level = self.ground
        self.b[:, axis] = np.maximum(self.b[:, axis], level)
        self.t[:, axis] = np.maximum(self.t[:, axis], level)


def bake(
    skeleton: Skeleton,
    worlds: np.ndarray,
    bind_worlds: np.ndarray,
    skirt: RingSkirt,
    capsules_by_role: dict[str, tuple[np.ndarray, np.ndarray, float]],
    loop: bool,
    ground_m: float,
) -> dict[str, np.ndarray]:
    """Local rotation keys (F, 4, xyzw) of the skirt's 2 * count chain joints for a body clip.

    `worlds` is `pose.node_worlds` of the body clip on a skeleton that holds the chain nodes (at rest); `bind_worlds`
    is the (N, 4, 4) bind world of every node. `capsules_by_role` maps a capsule role to its per-frame start and end
    (F, 3) and radius; the thighs, calves, feet, hands and lower arms of both sides are what the chains keep out of.
    A loop is simulated through four passes and the last of them is recorded, then the residual at the seam is spread
    over the clip so its last key equals its first; a clip that plays once settles for 30 frames on its first pose."""
    frames = len(worlds)
    chains = skirt.chains
    count = len(chains)
    index_a = np.array([skeleton.index(c.name_a) for c in chains])
    index_b = np.array([skeleton.index(c.name_b) for c in chains])
    anchor = skeleton.parent[index_a]
    bind_a = np.stack([c.a for c in chains])
    bind_b = np.stack([c.b for c in chains])
    bind_t = np.stack([c.hem for c in chains])

    # the pose each chain's anchor carries it to, per frame: world matrix times the inverse of its bind matrix
    carry = worlds[:, anchor] @ linalg.inverse_affine(bind_worlds[anchor])[None]  # (F, C, 4, 4)

    def carried(points: np.ndarray) -> np.ndarray:
        return np.einsum("fcij,cj->fci", carry[..., :3, :3], points) + carry[..., :3, 3]

    rigid_a, rigid_b, rigid_t = carried(bind_a), carried(bind_b), carried(bind_t)
    ring_rest = lambda p: np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)  # noqa: E731
    axis = {"y": 1, "z": 2}[skirt.up]
    tracks = [
        (start, end, radius)
        for role, (start, end, radius) in capsules_by_role.items()
        if role.rpartition("_")[0] in COLLIDERS
    ]
    sim = _Simulation(
        rigid_a,
        rigid_b,
        rigid_t,
        np.linalg.norm(bind_b - bind_a, axis=1),
        np.linalg.norm(bind_t - bind_b, axis=1),
        ring_rest(bind_b),
        ring_rest(bind_t),
        tracks,
        (axis, ground_m),
    )

    knee = np.zeros((frames, count, 3))
    hem = np.zeros((frames, count, 3))
    if frames < 2:
        knee[:], hem[:] = sim.b, sim.t
    elif loop:
        for _ in range(LOOP_PASSES):
            knee[0], hem[0] = sim.b, sim.t
            for f in range(frames - 1):
                sim.advance(f, f + 1)
                knee[f + 1], hem[f + 1] = sim.b, sim.t
    else:
        for _ in range(SETTLE_FRAMES):
            sim.advance(0, 0)
        knee[0], hem[0] = sim.b, sim.t
        for f in range(frames - 1):
            sim.advance(f, f + 1)
            knee[f + 1], hem[f + 1] = sim.b, sim.t

    # particles to rotations, which holds for any rest orientation of the chain joints
    anchor_rotation = quat.from_matrix(_orthonormal(worlds[:, anchor, :3, :3]))  # (F, C, 4)
    bind_anchor_rotation = quat.from_matrix(_orthonormal(bind_worlds[anchor, :3, :3]))  # (C, 4)
    bind_rotation_a = quat.from_matrix(_orthonormal(bind_worlds[index_a, :3, :3]))
    bind_rotation_b = quat.from_matrix(_orthonormal(bind_worlds[index_b, :3, :3]))
    delta_anchor = quat.multiply(anchor_rotation, quat.conjugate(bind_anchor_rotation)[None])
    d0_ab, d0_bt = bind_b - bind_a, bind_t - bind_b
    swing_a = quat.arc(_normalise(quat.rotate(delta_anchor, d0_ab[None])), _normalise(knee - rigid_a))
    delta_a = quat.multiply(swing_a, delta_anchor)
    world_a = quat.multiply(delta_a, bind_rotation_a[None])
    local_a = quat.multiply(quat.conjugate(anchor_rotation), world_a)
    swing_b = quat.arc(_normalise(quat.rotate(delta_a, d0_bt[None])), _normalise(hem - knee))
    world_b = quat.multiply(quat.multiply(swing_b, delta_a), bind_rotation_b[None])
    local_b = quat.multiply(quat.conjugate(world_a), world_b)

    out: dict[str, np.ndarray] = {}
    for i, chain in enumerate(chains):
        out[chain.name_a] = _continuous(local_a[:, i])
        out[chain.name_b] = _continuous(local_b[:, i])
    if loop and frames > 1:
        for name, keys in out.items():
            residual = quat.multiply(keys[-1], quat.conjugate(keys[0]))
            residual = residual if residual[3] >= 0.0 else -residual
            share = quat.slerp(np.broadcast_to(quat.IDENTITY, (frames, 4)), np.broadcast_to(quat.conjugate(residual), (frames, 4)), np.arange(frames) / (frames - 1))
            out[name] = _continuous(quat.multiply(share, keys))
    return out


def _continuous(keys: np.ndarray) -> np.ndarray:
    """Quaternion keys (F, 4) with the sign chosen so neighbours lie in one hemisphere (interpolation takes the short way)."""
    out = keys.copy()
    for f in range(1, len(out)):
        if float(out[f] @ out[f - 1]) < 0.0:
            out[f] = -out[f]
    return out
