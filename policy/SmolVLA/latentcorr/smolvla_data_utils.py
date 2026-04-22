from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if not SMOLVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"SmolVLA src directory not found: {SMOLVLA_SRC_DIR}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE


@lru_cache(maxsize=None)
def _load_episode_instruction_table(dataset_name: str) -> list[dict[str, Any]]:
    instruction_path = SMOLVLA_ROOT / "data" / dataset_name / "meta" / "episode_instructions.json"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Instruction file not found: {instruction_path}")
    with open(instruction_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Instruction payload must be a non-empty list: {instruction_path}")
    return payload


def _resolve_smolvla_dataset_name(task_name: str, task_config: str) -> str:
    dataset_name = f"robotwin_{task_name}_{task_config}_50_cam_high"
    dataset_root = SMOLVLA_ROOT / "data" / dataset_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"SmolVLA dataset directory not found: {dataset_root}")
    return dataset_name


def load_episode_instruction(task_name: str, task_config: str, episode_id: int, instruction_type: str) -> str:
    if instruction_type != "seen":
        raise ValueError(f"Unsupported instruction_type for SmolVLA dataset: {instruction_type}")

    dataset_name = _resolve_smolvla_dataset_name(task_name=task_name, task_config=task_config)
    instruction_table = _load_episode_instruction_table(dataset_name)
    episode_id = int(episode_id)
    if episode_id < 0 or episode_id >= len(instruction_table):
        raise IndexError(
            f"episode_id={episode_id} out of range for {dataset_name}, num_episodes={len(instruction_table)}"
        )
    record = instruction_table[episode_id]
    if int(record.get("episode_index", -1)) != episode_id:
        raise ValueError(
            f"Episode instruction index mismatch in {dataset_name}: expected {episode_id}, "
            f"got {record.get('episode_index')}"
        )
    values = record.get("instructions")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Instruction list is empty for episode_id={episode_id} in {dataset_name}")
    instruction = str(values[0]).strip()
    if not instruction:
        raise ValueError(f"First instruction is empty for episode_id={episode_id} in {dataset_name}")
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
