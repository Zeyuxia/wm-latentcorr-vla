from __future__ import annotations

import math

from policy.SmolVLA.latent_config import DynamicsWarmupConfig


class DynamicsWarmup:
    def __init__(self, config: DynamicsWarmupConfig):
        self.config = config

    def weight(self, step: int | float) -> float:
        step_value = float(max(0.0, float(step)))
        if step_value < float(self.config.zero_steps):
            return 0.0

        ramp_steps = int(self.config.ramp_steps)
        if ramp_steps <= 0:
            return float(self.config.max_weight)

        progress = min(1.0, (step_value - float(self.config.zero_steps)) / float(ramp_steps))
        if self.config.curve == "linear":
            return float(self.config.max_weight * progress)
        if self.config.curve == "cosine":
            return float(self.config.max_weight * 0.5 * (1.0 - math.cos(math.pi * progress)))
        raise ValueError(f"Unsupported warmup curve: {self.config.curve}")

