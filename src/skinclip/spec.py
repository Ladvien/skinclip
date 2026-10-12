"""The settings of a ring skirt, in a module that imports nothing, so a harness that must not load numpy can name them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RingSkirtSpec:
    """A ring of `count` two-bone chains (`{prefix}NN_a` from the top to the knee, `{prefix}NN_b` from the knee to the hem)
    hanging from one anchor joint, wrapped round a garment. Heights are metres along the up axis; `inset_m` is how far
    inside the garment's outer surface the chain line runs (half the cloth's thickness); between `top_height_m` and
    `blend_height_m` the garment's own weights are blended into the chains' (all chain below the top)."""

    count: int = 12
    top_height_m: float
    knee_height_m: float
    hem_height_m: float
    blend_height_m: float
    inset_m: float
    prefix: str = "skirt_"

    def __post_init__(self) -> None:
        if self.count < 3:
            raise ValueError(f"a ring needs at least 3 chains, not {self.count}")
        if not self.prefix:
            raise ValueError("the chain prefix is empty")
        if not self.hem_height_m < self.knee_height_m < self.top_height_m < self.blend_height_m:
            raise ValueError(
                "heights must rise hem < knee < top < blend, got "
                f"{self.hem_height_m} / {self.knee_height_m} / {self.top_height_m} / {self.blend_height_m}"
            )
