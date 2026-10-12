"""Limb motion repair: turn the arms and legs of a clip off the body, a few degrees at a time, where they pass through it.

Generated and retargeted motion crosses itself most often where a hand or a knee goes through the torso or the other
limb. Each limb is a capsule (`capsules.py`) and a frame is repaired by the smallest rotation of the upper and lower
limb joints that leaves every capsule pair apart: minimise 1e4 * sum(max(depth, 0)^2) + |delta|^2 over the rotation
vectors `delta` with a box on every component, by projected gradient descent with an Armijo backtracking line search
(no scipy: Blender's Python has none). The per-frame answers are smoothed over time (a gaussian, in two passes with a
re-solve of what the smoothing let back in) so the repaired motion does not jitter, and a loop stays a loop.

Penalty methods for interpenetration constraints are standard in cloth and character solvers (Grigorev et al.,
*ContourCraft*, SIGGRAPH 2024, doi:10.1145/3641519.3657408, for the contour as a measure); the segment-segment
distance is Ericson, *Real-Time Collision Detection*, 2005, 5.1.9. The literature search for a self-penetration
solver on retargeted motion found retargeting networks (Aberman et al., *Skeleton-Aware Networks for Deep Motion
Retargeting*, SIGGRAPH 2020, doi:10.1145/3386569.3392462) and no per-frame limb solver to cite, so this one is
skinclip's own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import capsules as capsules_module
from . import pose
from .capsules import Capsule
from .intersect import segment_distance
from .model import Clip, Figure
from .quat import exp_map, multiply, trs_matrix

MARGIN_M = 0.005
PENALTY = 1e4
BOUND_RAD = 0.45
DILATE_FRAMES = 6
FIRST_SIGMA = 2.0
SECOND_SIGMA = 1.0
MAX_ITERATIONS = 60
GRADIENT_STEP = 1e-4
ARMIJO = 1e-4
MIN_STEP = 1e-6
STOP_STEP = 1e-5
ACTIVE_DELTA = 1e-6

_OBSTACLE_TRUNK = ("pelvis", "spine_01", "spine_02", "spine_03", "neck", "head")
# limb -> (role of the chain parent, chain roles); the first two chain joints are the ones rotated
_LIMBS = {
    "arm": ("clavicle", ("upperarm", "lowerarm", "hand")),
    "leg": ("pelvis", ("thigh", "calf", "foot", "ball")),
}
_ARM_MOVERS = ("upperarm", "lowerarm", "hand")
_LEG_MOVERS = ("calf", "foot")


@dataclass(frozen=True)
class HalfSpace:
    """A plane fixed in the bind frame of `frame` that the end of each listed capsule must stay in front of."""

    frame: str  # joint name
    normal: tuple[float, float, float]
    offset_m: float
    capsules: tuple[str, ...]  # capsule roles


@dataclass(frozen=True)
class RepairReport:
    frames_repaired: int
    worst_depth_m_before: float
    worst_depth_m_after: float
    joints_changed: tuple[str, ...]


def _pairs() -> list[tuple[str, str]]:
    """(mover capsule, obstacle capsule) role pairs, each unordered pair once."""
    pairs: list[tuple[str, str]] = []
    for s, o in (("l", "r"), ("r", "l")):
        for mover in (f"lowerarm_{s}", f"hand_{s}"):
            for obstacle in (*_OBSTACLE_TRUNK, f"thigh_{s}", f"thigh_{o}", f"calf_{s}", f"calf_{o}", f"upperarm_{o}", f"lowerarm_{o}", f"hand_{o}"):
                pairs.append((mover, obstacle))
        for obstacle in ("head", f"thigh_{s}", f"thigh_{o}", f"lowerarm_{o}", f"hand_{o}"):
            pairs.append((f"upperarm_{s}", obstacle))
        for mover in (f"calf_{s}", f"foot_{s}"):
            for obstacle in (f"calf_{o}", f"foot_{o}", f"thigh_{o}"):
                pairs.append((mover, obstacle))
    seen: set[frozenset[str]] = set()
    unique = []
    for a, b in pairs:
        if frozenset((a, b)) not in seen:
            seen.add(frozenset((a, b)))
            unique.append((a, b))
    return unique


def _limb_of(role: str) -> tuple[str, str] | None:
    """('arm' | 'leg', side) of a capsule role that moves when its limb is rotated, else None."""
    base, _, side = role.rpartition("_")
    if side in ("l", "r"):
        if base in _ARM_MOVERS:
            return "arm", side
        if base in _LEG_MOVERS:
            return "leg", side
    return None


def _pair_limbs(a: str, b: str) -> set[tuple[str, str]]:
    return {limb for limb in (_limb_of(a), _limb_of(b)) if limb is not None}


class _Solver:
    """Everything a frame's solve needs, from the unrepaired clip."""

    def __init__(
        self,
        figure: Figure,
        roles: dict[str, str],
        capsules: dict[str, Capsule],
        rot: dict[str, np.ndarray],
        root_translation: np.ndarray,
        half_spaces: tuple[HalfSpace, ...],
        fps: float,
    ) -> None:
        self.figure, self.roles, self.capsules, self.rot, self.fps = figure, roles, capsules, rot, fps
        self.root_translation = root_translation
        self.half_spaces = half_spaces
        self.frames = len(root_translation)
        self.pairs = [(a, b) for a, b in _pairs() if a in capsules and b in capsules]
        self.radius = {role: capsule.radius_m for role, capsule in capsules.items()}
        # Pairs that overlap by design in the bind pose are not penetrations to repair.
        rest = pose.node_worlds(figure.skeleton, [(Clip("rest", 0.0, {}), np.zeros(1))])
        rest_positions = capsules_module.positions(figure, rest, capsules)
        self.skip = {pair for pair in self.pairs if self._depth(pair, rest_positions, 0) > 0.0}
        self.pairs = [pair for pair in self.pairs if pair not in self.skip]
        self.worlds = self.node_worlds(rot)
        self.fixed = capsules_module.positions(figure, self.worlds, capsules)
        sk = figure.skeleton
        self.rest_t, self.rest_s = sk.rest_t, sk.rest_s
        self.bind = np.linalg.inv(figure.inverse_bind)
        self.slot = {sk.names[node]: i for i, node in enumerate(figure.joints)}

    def node_worlds(self, rot: dict[str, np.ndarray]) -> np.ndarray:
        clip = pose.clip_from_frames("repair", rot, self.roles["pelvis"], self.root_translation, self.fps)
        return pose.node_worlds(self.figure.skeleton, [(clip, np.arange(self.frames) / self.fps)])

    def _depth(self, pair: tuple[str, str], positions: dict[str, tuple[np.ndarray, np.ndarray]], frame: int) -> float:
        a, b = pair
        s1, e1 = positions[a]
        s2, e2 = positions[b]
        distance = segment_distance(s1[frame : frame + 1], e1[frame : frame + 1], s2[frame : frame + 1], e2[frame : frame + 1])[0]
        return self.radius[a] + self.radius[b] + MARGIN_M - float(distance)

    # ---- depths of a whole clip ----

    def frame_depths(self, rot: dict[str, np.ndarray]) -> np.ndarray:
        """(F,) the deepest penetration of any pair (or half space violation) in each frame, 0 when none."""
        worlds = self.node_worlds(rot)
        positions = capsules_module.positions(self.figure, worlds, self.capsules)
        worst = np.zeros(self.frames)
        if self.pairs:
            starts1 = np.concatenate([positions[a][0] for a, _ in self.pairs])
            ends1 = np.concatenate([positions[a][1] for a, _ in self.pairs])
            starts2 = np.concatenate([positions[b][0] for _, b in self.pairs])
            ends2 = np.concatenate([positions[b][1] for _, b in self.pairs])
            distance = segment_distance(starts1, ends1, starts2, ends2).reshape(len(self.pairs), self.frames)
            radii = np.array([self.radius[a] + self.radius[b] + MARGIN_M for a, b in self.pairs])
            worst = np.maximum(worst, (radii[:, None] - distance).max(0))
        for plane in self.half_spaces:
            for role in plane.capsules:
                if role in positions:
                    worst = np.maximum(worst, self._violation(plane, role, worlds, positions[role][1]))
        return worst

    def _violation(self, plane: HalfSpace, role: str, worlds: np.ndarray, end: np.ndarray) -> np.ndarray:
        """How far the capsule's end, less its radius, is behind the plane, per frame (end shaped (F, 3))."""
        node = self.figure.skeleton.index(plane.frame)
        to_bind = self.bind[self.slot[plane.frame]][None] @ np.linalg.inv(worlds[:, node])
        local = np.einsum("fij,fj->fi", to_bind[:, :3, :3], end) + to_bind[:, :3, 3]
        return plane.offset_m - (local @ np.asarray(plane.normal) - self.radius[role])

    # ---- one frame ----

    def active_limbs(self, frame: int) -> set[tuple[str, str]]:
        """The limbs a penetrating pair at this frame moves."""
        limbs: set[tuple[str, str]] = set()
        for a, b in self.pairs:
            if self._depth((a, b), self.fixed, frame) > 0.0:
                limbs |= _pair_limbs(a, b)
        for plane in self.half_spaces:
            for role in plane.capsules:
                if role in self.fixed and self._violation(plane, role, self.worlds[frame : frame + 1], self.fixed[role][1][frame : frame + 1])[0] > 0.0:
                    limbs |= {limb for limb in (_limb_of(role),) if limb is not None}
        return limbs

    def solve(self, frame: int, limbs: list[tuple[str, str]], start: np.ndarray) -> np.ndarray:
        """The rotation vectors (6 per limb: upper, lower) that clear this frame, from `start` (len(limbs) * 6)."""
        scene = _Scene(self, frame, limbs)
        return _minimise(scene.objective, start, BOUND_RAD)


class _Scene:
    """The objective of one frame over the rotation vectors of the active limbs, batched over candidate vectors."""

    def __init__(self, solver: _Solver, frame: int, limbs: list[tuple[str, str]]) -> None:
        self.solver, self.frame, self.limbs = solver, frame, limbs
        sk = solver.figure.skeleton
        self.chains = []
        for kind, side in limbs:
            parent_role, chain = _LIMBS[kind]
            parent = f"{parent_role}_{side}" if kind == "arm" else parent_role
            joints = [solver.roles[f"{role}_{side}"] for role in chain]
            self.chains.append(
                {
                    "kind": kind,
                    "side": side,
                    "parent": solver.worlds[frame, sk.index(solver.roles[parent])],
                    "joints": joints,
                    "nodes": [sk.index(j) for j in joints],
                    "q": [solver.rot[j][frame] for j in joints],
                }
            )
        moving = set()
        for chain in self.chains:
            moving |= {f"{role}_{chain['side']}" for role in _LIMBS[chain["kind"]][1]}
        self.moving = moving
        involved = [(a, b) for a, b in solver.pairs if a in moving or b in moving]
        self.pairs = involved
        roles = sorted({r for pair in involved for r in pair})
        self.roles = roles
        self.index = {role: i for i, role in enumerate(roles)}
        self.a = np.array([self.index[a] for a, _ in involved], int)
        self.b = np.array([self.index[b] for _, b in involved], int)
        self.margin = np.array([solver.radius[a] + solver.radius[b] + MARGIN_M for a, b in involved])
        self.planes = [(plane, role) for plane in solver.half_spaces for role in plane.capsules if role in solver.capsules and role in self.moving]
        self.to_bind = {}
        for plane, _ in self.planes:
            node = sk.index(plane.frame)
            self.to_bind[plane.frame] = solver.bind[solver.slot[plane.frame]] @ np.linalg.inv(solver.worlds[frame, node])

    def endpoints(self, delta: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Start and end (B, 3) of every capsule role of the scene for a batch of rotation vectors (B, 6 * limbs)."""
        solver = self.solver
        batch = len(delta)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for k, chain in enumerate(self.chains):
            matrix = np.broadcast_to(chain["parent"], (batch, 4, 4))
            world = []
            for j, node in enumerate(chain["nodes"]):
                q = np.broadcast_to(chain["q"][j], (batch, 4))
                if j < 2:
                    q = multiply(q, exp_map(delta[:, 6 * k + 3 * j : 6 * k + 3 * j + 3]))
                local = trs_matrix(np.broadcast_to(solver.rest_t[node], (batch, 3)), q, np.broadcast_to(solver.rest_s[node], (batch, 3)))
                matrix = matrix @ local
                world.append(matrix)
            for j, role in enumerate(_LIMBS[chain["kind"]][1]):
                name = f"{role}_{chain['side']}"
                capsule = solver.capsules.get(name)
                if capsule is None:
                    continue
                start = world[j]
                if capsule.end is None:
                    end = np.einsum("bij,j->bi", start[:, :3, :3], capsule.centroid_offset) + start[:, :3, 3]
                else:
                    end = world[chain["joints"].index(capsule.end)][:, :3, 3]
                out[name] = (start[:, :3, 3], end)
        for role in self.roles:
            if role not in out:
                s, e = solver.fixed[role]
                out[role] = (np.broadcast_to(s[self.frame], (batch, 3)), np.broadcast_to(e[self.frame], (batch, 3)))
        return out

    def objective(self, delta: np.ndarray) -> np.ndarray:
        """(B,) 1e4 * sum of positive depths squared + sum of squared rotation vector lengths, for a batch (B, V)."""
        batch = len(delta)
        positions = self.endpoints(delta)
        total = np.sum(delta * delta, axis=1)
        if len(self.pairs):
            s1 = np.stack([positions[self.roles[i]][0] for i in self.a], 0)  # (P, B, 3)
            e1 = np.stack([positions[self.roles[i]][1] for i in self.a], 0)
            s2 = np.stack([positions[self.roles[i]][0] for i in self.b], 0)
            e2 = np.stack([positions[self.roles[i]][1] for i in self.b], 0)
            pairs = len(self.pairs)
            distance = segment_distance(s1.reshape(-1, 3), e1.reshape(-1, 3), s2.reshape(-1, 3), e2.reshape(-1, 3)).reshape(pairs, batch)
            total = total + PENALTY * np.sum(np.maximum(self.margin[:, None] - distance, 0.0) ** 2, axis=0)
        for plane, role in self.planes:
            to_bind = self.to_bind[plane.frame]
            end = positions[role][1]
            local = end @ to_bind[:3, :3].T + to_bind[:3, 3]
            violation = plane.offset_m - (local @ np.asarray(plane.normal) - self.solver.radius[role])
            total = total + PENALTY * np.maximum(violation, 0.0) ** 2
        return total


def _minimise(objective: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, bound: float) -> np.ndarray:
    """Projected gradient descent on a box [-bound, bound], central-difference gradient (one batched call), step
    halved until the Armijo condition holds, doubled for the next iteration."""
    x = np.clip(x0, -bound, bound)
    value = float(objective(x[None])[0])
    alpha = 1.0
    n = len(x)
    for _ in range(MAX_ITERATIONS):
        probes = np.concatenate([x[None] + GRADIENT_STEP * np.eye(n), x[None] - GRADIENT_STEP * np.eye(n)])
        values = objective(probes)
        gradient = (values[:n] - values[n:]) / (2.0 * GRADIENT_STEP)
        new, new_value = x, value
        while True:
            candidate = np.clip(x - alpha * gradient, -bound, bound)
            candidate_value = float(objective(candidate[None])[0])
            if candidate_value <= value - ARMIJO * float(gradient @ (x - candidate)) or alpha < MIN_STEP:
                new, new_value = candidate, candidate_value
                break
            alpha *= 0.5
        moved = float(np.max(np.abs(new - x)))
        x, value = new, new_value
        alpha = min(2.0 * alpha, 1.0)
        if moved < STOP_STEP:
            break
    return x


def _gaussian(values: np.ndarray, sigma: float, wrap: bool) -> np.ndarray:
    """Each column of (F, V) smoothed over frames by a gaussian of `sigma` frames (radius ceil(4 sigma), normalised);
    wrapped around for a loop, edge-padded otherwise."""
    radius = int(np.ceil(4.0 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="wrap" if wrap else "edge")
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])], axis=1)


def _apply(rot: dict[str, np.ndarray], solver: _Solver, deltas: dict[tuple[str, str], np.ndarray]) -> dict[str, np.ndarray]:
    """`rot` with the first two chain joints of each limb turned by their (F, 6) rotation vectors; other keys are the
    same arrays."""
    out = dict(rot)
    for (kind, side), delta in deltas.items():
        for j, role in enumerate(_LIMBS[kind][1][:2]):
            joint = solver.roles[f"{role}_{side}"]
            out[joint] = multiply(rot[joint], exp_map(delta[:, 3 * j : 3 * j + 3]))
    return out


def repair(
    figure: Figure,
    roles: dict[str, str],
    capsules: dict[str, Capsule],
    rot: dict[str, np.ndarray],
    root_translation: np.ndarray,
    loop: bool,
    half_spaces: tuple[HalfSpace, ...] = (),
    fps: float = 30.0,
) -> tuple[dict[str, np.ndarray], RepairReport]:
    """Turns the limbs of a clip off each other and off the trunk where their capsules overlap.

    `rot` is the clip's local rotation keys (xyzw, (F, 4)) by joint name and `root_translation` the pelvis's keys. The
    result is a copy of `rot` in which only the joints of limbs that needed it differ, with the report."""
    solver = _Solver(figure, roles, capsules, rot, np.asarray(root_translation, float), half_spaces, fps)
    before = solver.frame_depths(rot)
    worst_before = float(max(before.max(), 0.0)) if len(before) else 0.0
    bad = np.flatnonzero(before > 0.0)
    if len(bad) == 0:
        return dict(rot), RepairReport(0, worst_before, worst_before, ())

    frames = solver.frames
    limbs_seen: dict[tuple[str, str], np.ndarray] = {}

    def slot(limb: tuple[str, str]) -> np.ndarray:
        return limbs_seen.setdefault(limb, np.zeros((frames, 6)))

    # 1. Solve every penetrating frame, with the limbs any penetrating frame within DILATE_FRAMES of it moves, so that
    # neighbouring frames share their variables. A clear frame needs no turn of its own: the smoothing below carries
    # its neighbours' turn across.
    active = {int(f): solver.active_limbs(int(f)) for f in bad}
    for f in active:
        limbs = sorted(set().union(*(active[g] for g in active if abs(g - f) <= DILATE_FRAMES)))
        solved = solver.solve(f, limbs, np.zeros(6 * len(limbs)))
        for k, limb in enumerate(limbs):
            slot(limb)[f] = solved[6 * k : 6 * k + 6]

    # 2. Smooth over time, re-solve what the smoothing let back in (an extra turn on the smoothed keys), smooth lightly.
    smoothed = {limb: _gaussian(delta, FIRST_SIGMA, loop) for limb, delta in limbs_seen.items()}
    current = _apply(rot, solver, smoothed)
    again = np.flatnonzero(solver.frame_depths(current) > 0.0)
    if len(again):
        solver_now = _Solver(figure, roles, capsules, current, solver.root_translation, half_spaces, fps)
        for f in again:
            f = int(f)
            limbs = sorted(solver_now.active_limbs(f) | {limb for limb, delta in smoothed.items() if np.abs(delta[f]).max() > ACTIVE_DELTA})
            extra = solver_now.solve(f, limbs, np.zeros(6 * len(limbs)))
            for k, limb in enumerate(limbs):
                if limb not in smoothed:
                    smoothed[limb] = np.zeros((frames, 6))
                smoothed[limb][f] = _compose_vectors(smoothed[limb][f], extra[6 * k : 6 * k + 6])
        smoothed = {limb: _gaussian(delta, SECOND_SIGMA, loop) for limb, delta in smoothed.items()}
    if loop:
        for delta in smoothed.values():
            delta[-1] = delta[0]
    result = _apply(rot, solver, smoothed)
    after = solver.frame_depths(result)
    changed = tuple(
        solver.roles[f"{role}_{side}"]
        for (kind, side), delta in smoothed.items()
        for j, role in enumerate(_LIMBS[kind][1][:2])
        if np.abs(delta[:, 3 * j : 3 * j + 3]).max() > ACTIVE_DELTA
    )
    moved = np.zeros(frames, bool)
    for delta in smoothed.values():
        moved |= np.abs(delta).max(1) > ACTIVE_DELTA
    # joints not in `changed` keep their arrays as given, bit for bit
    final = {joint: (result[joint] if joint in changed else rot[joint]) for joint in rot}
    return final, RepairReport(int(moved.sum()), worst_before, float(max(after.max(), 0.0)), changed)


def _compose_vectors(first: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Rotation vectors (6: upper, lower) of two successive right-multiplied turns, first then extra, composed per joint."""
    out = np.empty(6)
    for j in range(2):
        q = multiply(exp_map(first[3 * j : 3 * j + 3]), exp_map(extra[3 * j : 3 * j + 3]))
        q = q if q[3] >= 0 else -q
        angle = 2.0 * np.arctan2(np.linalg.norm(q[:3]), q[3])
        axis = q[:3] / max(float(np.linalg.norm(q[:3])), 1e-12)
        out[3 * j : 3 * j + 3] = axis * angle
    return out
