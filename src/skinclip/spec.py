"""The settings of a ring skirt, in a module that imports nothing, so a harness that must not load numpy can name them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RingSkirtSpec:
    count: int = 12
    top_height_m: float
    knee_height_m: float
    hem_height_m: float
    blend_height_m: float
    inset_m: float
    prefix: str = "skirt_"
