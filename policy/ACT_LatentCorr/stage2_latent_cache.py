from __future__ import annotations

import hashlib
import os

import torch


def _tensor_sha1(tensor: torch.Tensor) -> str:
    arr = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def build_stage2_latent_cache_relpath(
    task_name: str,
    episode_id: int,
    start_ts: int,
    prefix_steps: int,
    ddim_steps: int,
    error_action_prefix_raw: torch.Tensor,
) -> str:
    task_dir = task_name.replace("/", "__")
    action_hash = _tensor_sha1(error_action_prefix_raw)
    return os.path.join(
        task_dir,
        f"ps{int(prefix_steps)}_ddim{int(ddim_steps)}",
        f"episode_{int(episode_id):04d}",
        f"start_{int(start_ts):04d}_{action_hash}.pt",
    )
