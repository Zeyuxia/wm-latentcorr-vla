from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_latent import DynamicsWarmupConfig


def _pick_num_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return max(1, groups)


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DynamicsWarmup:
    def __init__(self, cfg: DynamicsWarmupConfig):
        self.cfg = cfg

    def weight(self, step: float | int) -> float:
        step = float(max(0.0, float(step)))
        if step < self.cfg.zero_steps:
            return 0.0

        if self.cfg.ramp_steps <= 0:
            return float(self.cfg.max_weight)

        p = min(1.0, (step - self.cfg.zero_steps) / float(self.cfg.ramp_steps))
        if self.cfg.curve == "linear":
            return float(self.cfg.max_weight * p)
        if self.cfg.curve == "cosine":
            return float(self.cfg.max_weight * 0.5 * (1.0 - math.cos(math.pi * p)))
        raise ValueError(f"Unsupported warmup curve: {self.cfg.curve}")


class LatentProjector(nn.Module):
    """
    Project ACT visual feature map to WM VAE latent space while preserving 2D topology.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(out_channels), out_channels),
        )

    def forward(self, z_act: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        z = F.interpolate(z_act, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(z)


class ResidualLatentAdapter(nn.Module):
    """
    Light residual adapter that maps WM latent into the shared decoder/readout manifold.
    """

    def __init__(self, channels: int, mid_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(channels), channels),
        )

    def forward(self, z_wm: torch.Tensor) -> torch.Tensor:
        return z_wm + self.net(z_wm)


class ActionPrefixEncoder(nn.Module):
    """
    Encode executed action prefix chunk into AdaLN modulation vectors.
    """

    def __init__(self, prefix_steps: int, action_dim: int, out_channels: int, hidden_dim: int):
        super().__init__()
        in_dim = prefix_steps * action_dim
        self.prefix_steps = prefix_steps
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * 2),
        )

    def forward(self, actions: torch.Tensor, is_pad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # actions: (B, K, A)
        if actions.ndim != 3:
            raise ValueError(f"actions must be (B, K, A), got {actions.shape}")
        if is_pad is not None:
            actions = actions.masked_fill(is_pad.unsqueeze(-1), 0.0)
        x = actions.reshape(actions.shape[0], -1)
        gb = self.mlp(x)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma[:, :, None, None], beta[:, :, None, None]


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
    """
    Predict next latent state with action-conditioned AdaLN residual blocks.
    """

    def __init__(
        self,
        channels: int,
        prefix_steps: int,
        action_dim: int,
        mlp_hidden_dim: int,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.encoder = ActionPrefixEncoder(prefix_steps, action_dim, channels, mlp_hidden_dim)
        self.blocks = nn.ModuleList([AdaLNResBlock(channels) for _ in range(num_blocks)])

    def forward(self, z_t: torch.Tensor, action_prefix: torch.Tensor, is_pad: torch.Tensor | None = None) -> torch.Tensor:
        gamma, beta = self.encoder(action_prefix, is_pad=is_pad)
        z = z_t
        for blk in self.blocks:
            z = blk(z, gamma, beta)
        return z


class LatentActionDecoder(nn.Module):
    """
    Decode latent map to action prefix chunk.
    First step keeps it simple and stable: global pooled latent -> MLP.
    """

    def __init__(self, in_channels: int, hidden_dim: int, prefix_steps: int, action_dim: int):
        super().__init__()
        self.prefix_steps = prefix_steps
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prefix_steps * action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(z.shape[0], self.prefix_steps, self.action_dim)
