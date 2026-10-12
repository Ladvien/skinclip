"""Where a posed, skinned surface cuts through itself, and how long the cut is.

The relation checked is metamorphic: there is no oracle for what a clip's frame 41 should look like, but there is one
for how it must relate to the figure's own bind pose. A surface that is free of intersections at rest must stay free
of them in every pose (the invariant cloth and garment solvers keep, Grigorev et al., *ContourCraft: Learning to
Resolve Intersections in Neural Multi-Garment Simulations*, SIGGRAPH 2024, doi:10.1145/3641519.3657408, which measures
it as the length of the intersection contour; metamorphic testing as in Xie et al., *Testing and validating machine
learning classifiers by metamorphic testing*, JSS 2011, after Chen et al. 1998). So for each frame:

1. The figure is skinned the way a glTF renderer skins it: linear blend skinning, each joint's world matrix times its
   inverse bind matrix, weighted by `WEIGHTS_0` (no renormalisation).
2. Every pair of triangles that cross is found (a hierarchical grid for candidates, Mirtich, *Efficient Algorithms for
   Two-Phase Collision Detection*, 1997; then the interval test of Moeller, *A Fast Triangle-Triangle Intersection
   Test*, JGT 1997, see also Ericson, *Real-Time Collision Detection*, 2005, 5.3), and the length of the segment they
   share is the contour length that pair adds.
3. Pairs whose triangles lie within `rest_gap_m` of each other in the bind pose are left out: they already touch or
   overlap by design (a mannequin's joint balls sit inside its limbs) or are the two sides of one bending crease.
   Every other pair started apart, so a crossing between them is a limb, hand or hem that went through the body.
4. Two more kinds of crossing are left out. Both triangles within `crease_radius_m` of one joint at rest fold together
   at that joint under linear blend skinning (a knee in deep flexion, an armpit, a finger curl), and no weighting
   short of a different mesh avoids it. And cloth on cloth (`cloth_materials`: a sleeve resting on a skirt) is
   contact: linear blend skinning cannot hold loose cloth off itself without a cloth simulation. Skin through cloth,
   and every other pair, still counts.
5. A frame fails when the contour of those pairs is longer than `contour_limit_m`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Clip, Figure
from .pose import joint_matrices, skin

_EPS = 1e-9


@dataclass(frozen=True)
class CheckSettings:
    contour_limit_m: float = 0.04
    # Bind-pose distance under which two triangles may cross: the mannequin's joint balls reach this far into its limbs.
    rest_gap_m: float = 0.06
    # Both triangles within this of one joint at rest fold together at that joint under linear blend skinning (the
    # mannequin's armpit lies 15 cm from its shoulder joint).
    crease_radius_m: float = 0.15
    # Materials of loose cloth: where two of its triangles cross it is contact, not a limb through the body.
    cloth_materials: tuple[str, ...] = ()
    non_crease_prefixes: tuple[str, ...] = ("skirt_",)
    fps: float = 30.0


def _cell_keys(cells: np.ndarray) -> np.ndarray:
    """One int64 per integer cell (21 bits an axis, about 2 million cells across)."""
    c = cells + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def candidate_pairs(corners: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """(P, 2) triangle pairs whose bounding boxes overlap and that share no corner (`shared`: (T, 3) welded ids).

    A hierarchical grid (Mirtich, *Efficient Algorithms for Two-Phase Collision Detection*, 1997): level L has cells
    of median-triangle size times 2^L and holds the triangles that fit one cell, so a wing panel and a fingertip
    never share one flat grid's cells. Triangles of levels L and m <= L meet in level L's grid: the finer one's
    centroid lies within half a level-m cell of the coarser one's box, so the coarser triangle looks in the level-L
    cells its box grown by that much touches (at most 3 by 3 by 3)."""
    count_all = len(corners)
    lo, hi = corners.min(1), corners.max(1)
    centre = (lo + hi) / 2
    base = max(float(np.median((hi - lo).max(1))), 1e-4)
    level = np.ceil(np.log2(np.maximum((hi - lo).max(1), base) / base)).astype(int)
    found = []
    for current in np.unique(level):
        cell = base * 2.0**current
        natives = np.flatnonzero(level == current)
        for finer in np.unique(level[level <= current]):
            members = np.flatnonzero(level == finer)
            point_keys = _cell_keys(np.floor(centre[members] / cell).astype(np.int64))
            order = np.argsort(point_keys)
            point_keys, point_tri = point_keys[order], members[order]
            grow = base * 2.0**finer / 2
            first = np.floor((lo[natives] - grow) / cell).astype(np.int64)
            span = np.floor((hi[natives] + grow) / cell).astype(np.int64) - first + 1
            count = span.prod(1)
            box = np.repeat(np.arange(len(natives)), count)
            k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
            sx, sy = span[box, 0], span[box, 1]
            box_keys = _cell_keys(first[box] + np.stack([k % sx, (k // sx) % sy, k // (sx * sy)], 1))
            start = np.searchsorted(point_keys, box_keys, "left")
            hits = np.searchsorted(point_keys, box_keys, "right") - start
            a = np.repeat(natives[box], hits)
            b = point_tri[np.repeat(start, hits) + (np.arange(hits.sum()) - np.repeat(np.cumsum(hits) - hits, hits))]
            near = np.all((lo[a] <= hi[b]) & (lo[b] <= hi[a]), axis=1) & (a != b)
            a, b = a[near], b[near]
            found.append(np.minimum(a, b).astype(np.int64) * count_all + np.maximum(a, b))
    keys = np.unique(np.concatenate(found)) if found else np.zeros(0, np.int64)
    a, b = keys // count_all, keys % count_all
    touching = (shared[a][:, :, None] == shared[b][:, None, :]).any((1, 2))
    return np.stack([a[~touching], b[~touching]], 1)


def _crossing(tri: np.ndarray, d: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where a triangle crosses the other's plane, as an interval along the planes' common line."""
    lo = np.full(len(tri), np.inf)
    hi = np.full(len(tri), -np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        cross = (da * db <= 0) & (da != db)
        u = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
        point = tri[:, a] + (tri[:, b] - tri[:, a]) * u[:, None]
        s = np.einsum("pi,pi->p", point, direction)
        lo = np.where(cross, np.minimum(lo, s), lo)
        hi = np.where(cross, np.maximum(hi, s), hi)
    return lo, hi


def contour_lengths(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Length of the segment each pair of triangles (P, 3, 3) shares; 0 when they do not cross. Coplanar pairs count 0."""
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
    ok = (la > 1e-12) & (lb > 1e-12)
    na = na / np.where(ok, la, 1.0)[:, None]
    nb = nb / np.where(ok, lb, 1.0)[:, None]
    da = np.einsum("pki,pi->pk", a - b[:, :1], nb)
    db = np.einsum("pki,pi->pk", b - a[:, :1], na)
    da[np.abs(da) < _EPS] = 0.0
    db[np.abs(db) < _EPS] = 0.0
    ok &= ~(np.all(da > 0, 1) | np.all(da < 0, 1) | np.all(db > 0, 1) | np.all(db < 0, 1))
    line = np.cross(na, nb)
    ll = np.linalg.norm(line, axis=1)
    ok &= ll > 1e-9
    line = line / np.where(ok, ll, 1.0)[:, None]
    alo, ahi = _crossing(a, da, line)
    blo, bhi = _crossing(b, db, line)
    length = np.minimum(ahi, bhi) - np.maximum(alo, blo)
    return np.where(ok & np.isfinite(length), np.maximum(length, 0.0), 0.0)


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    u = np.clip(np.einsum("pi,pi->p", p - a, ab) / np.maximum(np.einsum("pi,pi->p", ab, ab), 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + ab * u[:, None]), axis=1)


def _point_triangle(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.maximum(np.einsum("pi,pi->p", n, n), 1e-18)
    inside = np.ones(len(p), bool)
    for x, y in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("pi,pi->p", np.cross(y - x, p - x), n) >= 0
    plane = np.abs(np.einsum("pi,pi->p", p - a, n)) / np.sqrt(nn)
    edges = np.minimum(np.minimum(_point_segment(p, a, b), _point_segment(p, b, c)), _point_segment(p, c, a))
    return np.where(inside, plane, edges)



def closest_points(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The closest points between segments p1-q1 and p2-q2 (Ericson, *Real-Time Collision Detection*, 2005, 5.1.9),
    vectorised over rows: the clamped parameters (s, t) and the points c1 = p1 + s (q1 - p1), c2 = p2 + t (q2 - p2)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("pi,pi->p", d1, d1)
    e = np.einsum("pi,pi->p", d2, d2)
    f = np.einsum("pi,pi->p", d2, r)
    c = np.einsum("pi,pi->p", d1, r)
    b = np.einsum("pi,pi->p", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-18, np.clip((b * f - c * e) / np.where(denom > 1e-18, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-18)
    s = np.where(t < 0, np.clip(-c / np.maximum(a, 1e-18), 0.0, 1.0), np.where(t > 1, np.clip((b - c) / np.maximum(a, 1e-18), 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return s, t, p1 + d1 * s[:, None], p2 + d2 * t[:, None]


def segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Closest distance between segments p1-q1 and p2-q2, vectorised over rows."""
    _, _, c1, c2 = closest_points(p1, q1, p2, q2)
    return np.linalg.norm(c1 - c2, axis=1)


def triangle_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closest distance between each pair of triangles (P, 3, 3); 0 where they cross."""
    best = np.full(len(a), np.inf)
    for k in range(3):
        best = np.minimum(best, _point_triangle(a[:, k], b))
        best = np.minimum(best, _point_triangle(b[:, k], a))
        for m in range(3):
            best = np.minimum(best, segment_distance(a[:, k], a[:, (k + 1) % 3], b[:, m], b[:, (m + 1) % 3]))
    return np.where(contour_lengths(a, b) > 0, 0.0, best)


@dataclass
class FrameCut:
    contour_m: float
    parts: dict[tuple[str, str], float]  # label pair -> contour length


def pose_cut(figure: Figure, vertices: np.ndarray, settings: CheckSettings) -> FrameCut:
    """The contour of crossings, in one pose, between triangles that lay at least `rest_gap_m` apart in the bind pose,
    that do not fold together at a joint and that are not two pieces of loose cloth."""
    corners = vertices[figure.triangles]
    pairs = candidate_pairs(corners, figure.welded)
    length = contour_lengths(corners[pairs[:, 0]], corners[pairs[:, 1]])
    crossing, length = pairs[length > 0], length[length > 0]
    rest = figure.positions[figure.triangles]
    keep = triangle_distances(rest[crossing[:, 0]], rest[crossing[:, 1]]) >= settings.rest_gap_m
    crossing, length = crossing[keep], length[keep]
    keep = ~(figure.crease[crossing[:, 0]] & figure.crease[crossing[:, 1]]).any(1)
    crossing, length = crossing[keep], length[keep]
    keep = np.array([not _cloth_on_cloth(figure.triangle_label[i], figure.triangle_label[j], settings.cloth_materials) for i, j in crossing], bool)
    crossing, length = crossing[keep], length[keep]
    parts: dict[tuple[str, str], float] = {}
    for (i, j), cut in zip(crossing, length):
        key = tuple(sorted((figure.triangle_label[i], figure.triangle_label[j])))
        parts[key] = parts.get(key, 0.0) + float(cut)
    return FrameCut(float(length.sum()), parts)


def _cloth_on_cloth(a: str, b: str, materials: tuple[str, ...]) -> bool:
    return any(a.startswith(f"{m}:") and b.startswith(f"{m}:") for m in materials)


@dataclass
class ClipCheck:
    figure: str
    clip: str
    frames: int
    failing: int
    worst_frame: int
    worst_contour_m: float
    worst_parts: list[tuple[str, str, float]]
    contour_by_frame: list[float]
    ceiling_m: float | None = None  # the tolerated ceiling for this clip, when it is on a list

    @property
    def passed(self) -> bool:
        return self.failing == 0 or (self.ceiling_m is not None and self.worst_contour_m <= self.ceiling_m)


def clip_cuts(figure: Figure, layers: list[Clip], settings: CheckSettings, stop_on_fail: bool = False) -> list[FrameCut]:
    """Every frame of a clip at `settings.fps`, its first frame to its last key. The first layer sets the length; the
    others (a flyer's wings) loop under it. With `stop_on_fail` the frames end at the first one whose contour exceeds
    the limit."""
    at = np.arange(0.0, layers[0].seconds + 1e-6, 1.0 / settings.fps)
    timed = [(layers[0], at)] + [(clip, np.mod(at, clip.seconds) if clip.seconds > 0 else at) for clip in layers[1:]]
    matrices = joint_matrices(figure, timed)
    cuts = []
    for m in matrices:
        cuts.append(pose_cut(figure, skin(figure, m), settings))
        if stop_on_fail and cuts[-1].contour_m > settings.contour_limit_m:
            break
    return cuts


def check_clip(
    figure: Figure,
    name: str,
    layers: list[Clip],
    settings: CheckSettings,
    ceiling_m: float | None = None,
    stop_on_fail: bool = False,
) -> ClipCheck:
    """Checks every frame of a clip. `stop_on_fail` ends at the first failing frame (the result then covers only the
    frames up to it), unless a ceiling applies: a tolerated clip is measured to its worst frame."""
    cuts = clip_cuts(figure, layers, settings, stop_on_fail and ceiling_m is None)
    contour = [cut.contour_m for cut in cuts]
    worst = int(np.argmax(contour))
    parts = sorted(((a, b, round(v, 4)) for (a, b), v in cuts[worst].parts.items()), key=lambda p: -p[2])[:4]
    failing = sum(c > settings.contour_limit_m for c in contour)
    return ClipCheck(figure.name, name, len(cuts), int(failing), worst, round(contour[worst], 4), parts, [round(c, 4) for c in contour], ceiling_m)
