from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_num_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return max(1, groups)


class DynamicsWarmup:
    def __init__(self, zero_steps: int, ramp_steps: int, max_weight: float, curve: str = "cosine"):
        self.zero_steps = int(zero_steps)
        self.ramp_steps = int(ramp_steps)
        self.max_weight = float(max_weight)
        self.curve = str(curve)

    def weight(self, step: float) -> float:
        if step < self.zero_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return self.max_weight
        progress = min(1.0, float(step - self.zero_steps) / float(self.ramp_steps))
        if self.curve == "linear":
            return self.max_weight * progress
        if self.curve == "cosine":
            return self.max_weight * 0.5 * (1.0 - math.cos(math.pi * progress))
        raise ValueError(f"Unsupported warmup curve: {self.curve}")


class LatentProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(out_channels), out_channels),
        )

    def forward(self, z_src: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        z = F.interpolate(z_src, size=target_hw, mode="bilinear", align_corners=False)
        return self.net(z)


class ActionPrefixEncoder(nn.Module):
    def __init__(self, prefix_steps: int, action_dim: int, out_channels: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(prefix_steps * action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * 2),
        )

    def forward(self, actions: torch.Tensor, is_pad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if is_pad is not None:
            actions = actions.masked_fill(is_pad.unsqueeze(-1), 0.0)
        x = actions.reshape(actions.shape[0], -1)
        gamma_beta = self.mlp(x)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma[:, :, None, None], beta[:, :, None, None]


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class AdaLNResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.norm(z)
        h = gamma * h + beta
        h = self.conv1(h)
        h = self.act(h)
        h = self.conv2(h)
        return z + h


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        channels: int,
        prefix_steps: int,
        action_dim: int,
        mlp_hidden_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.encoder = ActionPrefixEncoder(prefix_steps, action_dim, channels, mlp_hidden_dim)
        self.blocks = nn.ModuleList([AdaLNResBlock(channels) for _ in range(num_blocks)])

    def forward(self, z_t: torch.Tensor, action_prefix: torch.Tensor, is_pad: torch.Tensor | None = None) -> torch.Tensor:
        gamma, beta = self.encoder(action_prefix, is_pad=is_pad)
        z = z_t
        for block in self.blocks:
            z = block(z, gamma, beta)
        return z
