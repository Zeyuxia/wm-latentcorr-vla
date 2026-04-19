from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SMOLVLA_SRC_DIR = THIS_DIR / "src"
if not SMOLVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"SmolVLA src directory not found: {SMOLVLA_SRC_DIR}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE


def load_episode_instruction(task_name: str, task_config: str, episode_id: int, instruction_type: str) -> str:
    instruction_path = REPO_ROOT / "data" / task_name / task_config / "instructions" / f"episode{int(episode_id)}.json"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Instruction file not found: {instruction_path}")
    with open(instruction_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if instruction_type not in payload:
        raise KeyError(f"Instruction type {instruction_type!r} not found in {instruction_path}")
    values = payload[instruction_type]
    if not isinstance(values, list) or not values:
        raise ValueError(f"Instruction list for {instruction_type!r} is empty in {instruction_path}")
    instruction = str(values[0]).strip()
    if not instruction:
        raise ValueError(f"First instruction for {instruction_type!r} is empty in {instruction_path}")
    return instruction


def build_smolvla_batch(
    policy,
    preprocess,
    image_t: torch.Tensor,
    qpos_raw: torch.Tensor,
    action_chunk_raw: torch.Tensor,
    task_name: str,
    task_config: str,
    episode_id: int,
    instruction_type: str,
) -> dict[str, Any]:
    if image_t.ndim != 4:
        raise ValueError(f"image_t must be (num_cam, C, H, W), got {tuple(image_t.shape)}")
    if int(image_t.shape[0]) != 1:
        raise ValueError(f"Exactly one camera is supported, got num_cam={int(image_t.shape[0])}")
    if qpos_raw.ndim != 1:
        raise ValueError(f"qpos_raw must be (D,), got {tuple(qpos_raw.shape)}")
    if action_chunk_raw.ndim != 2:
        raise ValueError(f"action_chunk_raw must be (T, A), got {tuple(action_chunk_raw.shape)}")

    visual_keys = list(policy.config.image_features.keys())
    if len(visual_keys) != 1:
        raise ValueError(f"Expected exactly one visual feature key, got {visual_keys}")

    instruction = load_episode_instruction(
        task_name=task_name,
        task_config=task_config,
        episode_id=episode_id,
        instruction_type=instruction_type,
    )
    obs = {
        visual_keys[0]: (image_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8"),
        OBS_STATE: qpos_raw.detach().cpu().numpy().astype("float32"),
        ACTION: action_chunk_raw.detach().cpu().numpy().astype("float32"),
    }
    prepared = prepare_observation_for_inference(
        obs,
        device=next(policy.parameters()).device,
        task=instruction,
        robot_type="aloha",
    )
    return preprocess(prepared)


def make_smolvla_processors(policy, pretrained_path: str):
    if not pretrained_path:
        raise ValueError("pretrained_path must be provided for SmolVLA processor construction")
    return make_pre_post_processors(
        policy.config,
        pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(next(policy.parameters()).device)}},
    )


def stack_smolvla_batches(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    keys = set(samples[0].keys())
    for sample in samples[1:]:
        if set(sample.keys()) != keys:
            raise ValueError("All SmolVLA samples must have identical keys for stacking")

    batch: dict[str, Any] = {}
    for key in samples[0]:
        value0 = samples[0][key]
        if isinstance(value0, torch.Tensor):
            batch[key] = torch.cat([sample[key] for sample in samples], dim=0)
        else:
            batch[key] = [sample[key] for sample in samples]
    return batch

