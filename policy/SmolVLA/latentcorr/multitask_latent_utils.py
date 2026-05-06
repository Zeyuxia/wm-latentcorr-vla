from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from policy.SmolVLA.latentcorr.latent_dataset_utils import (
    act_processed_images_to_rgb,
    get_episode_length,
    is_lerobot_dataset_dir,
    load_episode_qpos_action_arrays,
    load_lerobot_episode_window,
    resolve_raw_data_dir,
)


@dataclass(frozen=True)
class MultiTaskSpec:
    task_name: str
    dataset_dir: str
    raw_data_dir: str | None
    num_episodes: int
    camera_names: tuple[str, ...]


def _iter_episode_paths(dataset_dir: str, num_episodes: int):
    if is_lerobot_dataset_dir(dataset_dir):
        for episode_id in range(num_episodes):
            try:
                get_episode_length(dataset_dir, int(episode_id))
            except FileNotFoundError:
                continue
            yield episode_id, dataset_dir
        return
    for episode_id in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
        if os.path.exists(path):
            yield episode_id, path


def get_multitask_norm_stats(task_specs: list[MultiTaskSpec]) -> dict[str, np.ndarray]:
    all_qpos = []
    all_action = []
    for spec in task_specs:
        for episode_id, path in _iter_episode_paths(spec.dataset_dir, spec.num_episodes):
            if is_lerobot_dataset_dir(spec.dataset_dir):
                qpos, action = load_episode_qpos_action_arrays(spec.dataset_dir, int(episode_id))
                all_qpos.append(torch.from_numpy(qpos))
                all_action.append(torch.from_numpy(action))
            else:
                with h5py.File(path, "r") as root:
                    all_qpos.append(torch.from_numpy(root["/observations/qpos"][()]))
                    all_action.append(torch.from_numpy(root["/action"][()]))
    if not all_qpos:
        raise ValueError("No episodes found in any multitask dataset_dir")

    max_qpos_len = max(qpos.size(0) for qpos in all_qpos)
    max_action_len = max(action.size(0) for action in all_action)
    padded_qpos = []
    for qpos in all_qpos:
        if qpos.size(0) < max_qpos_len:
            pad = qpos[-1:].repeat(max_qpos_len - qpos.size(0), 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)
    padded_action = []
    for action in all_action:
        if action.size(0) < max_action_len:
            pad = action[-1:].repeat(max_action_len - action.size(0), 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    qpos_tensor = torch.stack(padded_qpos)
    action_tensor = torch.stack(padded_action)
    qpos_mean = qpos_tensor.mean(dim=(0, 1))
    qpos_std = qpos_tensor.std(dim=(0, 1)).clamp_min(1e-2)
    action_mean = action_tensor.mean(dim=(0, 1))
    action_std = action_tensor.std(dim=(0, 1)).clamp_min(1e-2)
    return {
        "qpos_mean": qpos_mean.numpy().astype(np.float32),
        "qpos_std": qpos_std.numpy().astype(np.float32),
        "action_mean": action_mean.numpy().astype(np.float32),
        "action_std": action_std.numpy().astype(np.float32),
    }


class MultiTaskLatentWarmupDataset(Dataset):
    def __init__(
        self,
        task_specs: list[MultiTaskSpec],
        norm_stats: dict[str, np.ndarray],
        act_chunk_size: int,
        prefix_steps: int,
        future_offset: int,
    ):
        self.task_specs = list(task_specs)
        self.stats = norm_stats
        self.act_chunk_size = int(act_chunk_size)
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)
        self.samples: list[tuple[int, int]] = []
        for task_idx, spec in enumerate(task_specs):
            for episode_id, path in _iter_episode_paths(spec.dataset_dir, spec.num_episodes):
                if is_lerobot_dataset_dir(spec.dataset_dir):
                    if int(get_episode_length(spec.dataset_dir, int(episode_id))) >= self.future_offset + 1:
                        self.samples.append((task_idx, episode_id))
                else:
                    with h5py.File(path, "r") as root:
                        if int(root["/action"].shape[0]) >= self.future_offset + 1:
                            self.samples.append((task_idx, episode_id))
        if not self.samples:
            raise ValueError("No valid multitask samples found")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _pad_action_window(actions: np.ndarray, start_ts: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        action_dim = int(actions.shape[1])
        padded = np.zeros((horizon, action_dim), dtype=np.float32)
        is_pad = np.ones((horizon,), dtype=bool)
        if start_ts < actions.shape[0]:
            action_slice = actions[start_ts : start_ts + horizon].astype(np.float32)
            valid_len = int(action_slice.shape[0])
            padded[:valid_len] = action_slice
            is_pad[:valid_len] = False
        return padded, is_pad

    def __getitem__(self, index: int) -> dict:
        task_idx, episode_id = self.samples[index]
        spec = self.task_specs[task_idx]
        if is_lerobot_dataset_dir(spec.dataset_dir):
            episode_len = get_episode_length(spec.dataset_dir, int(episode_id))
            max_start = int(episode_len) - self.future_offset - 1
            if max_start < 0:
                raise ValueError(f"{spec.task_name} episode_{episode_id} too short: {episode_len}")
            start_ts = int(np.random.randint(0, max_start + 1))
            sample = load_lerobot_episode_window(
                dataset_dir=spec.dataset_dir,
                episode_id=int(episode_id),
                camera_names=list(spec.camera_names),
                norm_stats=self.stats,
                act_chunk_size=self.act_chunk_size,
                prefix_steps=self.prefix_steps,
                future_offset=self.future_offset,
                start_ts=start_ts,
            )
            sample["task_idx"] = torch.tensor(task_idx)
            sample["task_name"] = spec.task_name
            sample["raw_data_dir"] = "" if spec.raw_data_dir is None else spec.raw_data_dir
            return sample

        path = os.path.join(spec.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(path, "r") as root:
            actions = root["/action"][()]
            qpos_seq = root["/observations/qpos"][()]
            episode_len = int(actions.shape[0])
            max_start = episode_len - self.future_offset - 1
            if max_start < 0:
                raise ValueError(f"{spec.task_name} episode_{episode_id} too short: {episode_len}")
            start_ts = int(np.random.randint(0, max_start + 1))
            image_t = []
            image_t_future = []
            for camera_name in spec.camera_names:
                image_t.append(root[f"/observations/images/{camera_name}"][start_ts])
                image_t_future.append(root[f"/observations/images/{camera_name}"][start_ts + self.future_offset])

        image_t_np = act_processed_images_to_rgb(np.stack(image_t, axis=0))
        image_t_future_np = act_processed_images_to_rgb(np.stack(image_t_future, axis=0))
        image_t_tensor = torch.from_numpy(image_t_np).permute(0, 3, 1, 2).float() / 255.0
        image_t_future_tensor = torch.from_numpy(image_t_future_np).permute(0, 3, 1, 2).float() / 255.0
        qpos_raw = qpos_seq[start_ts].astype(np.float32)
        qpos_t = (qpos_raw - self.stats["qpos_mean"]) / self.stats["qpos_std"]

        action_chunk_raw, act_is_pad = self._pad_action_window(actions, start_ts, self.act_chunk_size)
        action_prefix_raw = action_chunk_raw[: self.prefix_steps].copy()
        action_future_prefix_raw, is_pad_future = self._pad_action_window(
            actions, start_ts + self.future_offset, self.prefix_steps
        )
        action_chunk = (action_chunk_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action_prefix = (action_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action_future_prefix = (action_future_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
        is_pad_prefix = act_is_pad[: self.prefix_steps].copy()

        return {
            "image_t": image_t_tensor,
            "image_t_future": image_t_future_tensor,
            "qpos_t": torch.from_numpy(qpos_t.astype(np.float32)),
            "qpos_raw": torch.from_numpy(qpos_raw),
            "act_action_chunk": torch.from_numpy(action_chunk.astype(np.float32)),
            "act_action_chunk_raw": torch.from_numpy(action_chunk_raw.astype(np.float32)),
            "act_is_pad": torch.from_numpy(act_is_pad).bool(),
            "action_prefix": torch.from_numpy(action_prefix.astype(np.float32)),
            "action_future_prefix": torch.from_numpy(action_future_prefix.astype(np.float32)),
            "action_prefix_raw": torch.from_numpy(action_prefix_raw.astype(np.float32)),
            "action_future_prefix_raw": torch.from_numpy(action_future_prefix_raw.astype(np.float32)),
            "is_pad_prefix": torch.from_numpy(is_pad_prefix).bool(),
            "is_pad_future_prefix": torch.from_numpy(is_pad_future).bool(),
            "episode_id": torch.tensor(episode_id),
            "start_ts": torch.tensor(start_ts),
            "task_idx": torch.tensor(task_idx),
            "task_name": spec.task_name,
            "raw_data_dir": "" if spec.raw_data_dir is None else spec.raw_data_dir,
        }


def build_multitask_stage1_dataset(
    task_specs: list[MultiTaskSpec],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
) -> tuple[MultiTaskLatentWarmupDataset, dict[str, np.ndarray]]:
    stats = get_multitask_norm_stats(task_specs)
    dataset = MultiTaskLatentWarmupDataset(
        task_specs=task_specs,
        norm_stats=stats,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
    )
    return dataset, stats


def _split_sim_task_name(task_name: str, default_num_episodes: int) -> tuple[str, str, int]:
    task = str(task_name)
    if task.startswith("sim-"):
        task = task[4:]
    parts = task.split("-")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse task_name={task_name!r}")
    task_only = parts[0]
    expert_num = int(default_num_episodes)
    config_parts = parts[1:]
    if config_parts and str(config_parts[-1]).isdigit():
        expert_num = int(config_parts[-1])
        config_parts = config_parts[:-1]
    task_config = "-".join(config_parts)
    if not task_config:
        raise ValueError(f"Cannot parse task_config from task_name={task_name!r}")
    return task_only, task_config, expert_num


def _resolve_smolvla_lerobot_dataset_dir(task_name: str, info: dict) -> str | None:
    source = str(os.environ.get("SMOLVLA_LATENT_DATA_SOURCE", "")).strip().lower()
    if source not in {"smolvla", "smolvla_rgbfix", "lerobot", "lerobot_rgbfix"}:
        return None
    suffix = str(os.environ.get("SMOLVLA_LATENT_DATA_SUFFIX", "")).strip()
    if source in {"smolvla_rgbfix", "lerobot_rgbfix"} and not suffix:
        suffix = "_rgbfix"
    data_root = os.path.realpath(
        os.environ.get("SMOLVLA_LATENT_DATA_ROOT", "/data/zhenyangfan/RoboTwin/policy/SmolVLA/data")
    )
    task_only, task_config, expert_num = _split_sim_task_name(
        task_name,
        default_num_episodes=int(info.get("num_episodes", 50)),
    )
    dataset_name = f"robotwin_{task_only}_{task_config}_{expert_num}_cam_high{suffix}"
    dataset_dir = os.path.realpath(os.path.join(data_root, dataset_name))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"SMOLVLA_LATENT_DATA_SOURCE={source!r} requested {dataset_dir}, but it does not exist."
        )
    return dataset_dir


def resolve_multitask_specs(
    task_names: list[str],
    sim_task_configs: dict,
    raw_data_root_overrides: dict[str, str] | None,
) -> list[MultiTaskSpec]:
    specs: list[MultiTaskSpec] = []
    overrides = {} if raw_data_root_overrides is None else dict(raw_data_root_overrides)
    for task_name in task_names:
        if task_name not in sim_task_configs:
            raise KeyError(f"task_name={task_name} not found in SIM_TASK_CONFIGS")
        info = sim_task_configs[task_name]
        dataset_dir = _resolve_smolvla_lerobot_dataset_dir(task_name, info)
        if dataset_dir is None:
            dataset_dir = info["dataset_dir"]
            if dataset_dir.startswith("./"):
                dataset_dir = os.path.realpath(os.path.join("/data/zhenyangfan/RoboTwin/policy/ACT", dataset_dir[2:]))
            else:
                dataset_dir = os.path.realpath(dataset_dir)
        specs.append(
            MultiTaskSpec(
                task_name=task_name,
                dataset_dir=dataset_dir,
                raw_data_dir=resolve_raw_data_dir(task_name, overrides.get(task_name)),
                num_episodes=int(info["num_episodes"]),
                camera_names=tuple(info["camera_names"]),
            )
        )
    return specs
