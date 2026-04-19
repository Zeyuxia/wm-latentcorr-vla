from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsWarmupConfig:
    zero_steps: int
    ramp_steps: int
    max_weight: float
    curve: str
