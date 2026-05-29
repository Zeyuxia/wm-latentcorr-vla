from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils_latent import resolve_raw_data_dir


@dataclass(frozen=True)
class MultiTaskSpec:
    task_name: str
    dataset_dir: str
    raw_data_dir: str | None
    num_episodes: int
    camera_names: tuple[str, ...]


def build_future_latent_cache_relpath(
    task_name: str,
    episode_id: int,
    start_ts: int,
    future_offset: int,
    prefix_steps: int,
    ddim_steps: int,
) -> str:
    task_dir = task_name.replace("/", "__")
    return os.path.join(
        task_dir,
        f"fo{future_offset}_ps{prefix_steps}_ddim{ddim_steps}",
        f"episode_{int(episode_id):04d}",
        f"start_{int(start_ts):04d}.pt",
    )


def _iter_episode_paths(dataset_dir: str, num_episodes: int):
    for ep in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{ep}.hdf5")
        if os.path.exists(path):
            yield ep, path


def get_multitask_norm_stats(task_specs: list[MultiTaskSpec]) -> dict[str, np.ndarray]:
    all_qpos = []
    all_action = []
    for spec in task_specs:
        for _, path in _iter_episode_paths(spec.dataset_dir, spec.num_episodes):
            with h5py.File(path, "r") as root:
                all_qpos.append(torch.from_numpy(root["/observations/qpos"][()]))
                all_action.append(torch.from_numpy(root["/action"][()]))

    if len(all_qpos) == 0:
        raise ValueError("No episodes found in any multitask dataset_dir")

    max_qpos_len = max(q.size(0) for q in all_qpos)
    max_action_len = max(a.size(0) for a in all_action)

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

    q = torch.stack(padded_qpos)
    a = torch.stack(padded_action)
    q_mean, q_std = q.mean(dim=(0, 1)), q.std(dim=(0, 1)).clamp_min(1e-2)
    a_mean, a_std = a.mean(dim=(0, 1)), a.std(dim=(0, 1)).clamp_min(1e-2)
    return {
        "qpos_mean": q_mean.numpy().astype(np.float32),
        "qpos_std": q_std.numpy().astype(np.float32),
        "action_mean": a_mean.numpy().astype(np.float32),
        "action_std": a_std.numpy().astype(np.float32),
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
        self.task_specs = task_specs
        self.stats = norm_stats
        self.act_chunk_size = act_chunk_size
        self.prefix_steps = prefix_steps
        self.future_offset = future_offset
        self.samples: list[tuple[int, int]] = []
        for task_idx, spec in enumerate(task_specs):
            for ep_id, path in _iter_episode_paths(spec.dataset_dir, spec.num_episodes):
                with h5py.File(path, "r") as root:
                    if int(root["/action"].shape[0]) >= (future_offset + 1):
                        self.samples.append((task_idx, ep_id))

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

    def __getitem__(self, index: int):
        task_idx, ep_id = self.samples[index]
        spec = self.task_specs[task_idx]
        path = os.path.join(spec.dataset_dir, f"episode_{ep_id}.hdf5")
        with h5py.File(path, "r") as root:
            actions = root["/action"][()]
            qpos_seq = root["/observations/qpos"][()]
            ep_len = int(actions.shape[0])
            max_start = ep_len - self.future_offset - 1
            if max_start < 0:
                raise ValueError(f"{spec.task_name} episode_{ep_id} too short: {ep_len}")
            start_ts = np.random.randint(0, max_start + 1)
            image_t = []
            image_t_future = []
            for cam_name in spec.camera_names:
                i0 = root[f"/observations/images/{cam_name}"][start_ts]
                i1 = root[f"/observations/images/{cam_name}"][start_ts + self.future_offset]
                image_t.append(i0)
                image_t_future.append(i1)

        image_t = np.stack(image_t, axis=0)
        image_t_future = np.stack(image_t_future, axis=0)
        image_t = torch.from_numpy(image_t).permute(0, 3, 1, 2).float() / 255.0
        image_t_future = torch.from_numpy(image_t_future).permute(0, 3, 1, 2).float() / 255.0

        qpos_t_np = qpos_seq[start_ts].astype(np.float32)
        qpos_future_np = qpos_seq[start_ts + self.future_offset].astype(np.float32)
        qpos_raw = torch.from_numpy(qpos_t_np)
        qpos_future_raw = torch.from_numpy(qpos_future_np)
        qpos_t = torch.from_numpy(
            ((qpos_t_np - self.stats["qpos_mean"]) / self.stats["qpos_std"]).astype(np.float32)
        )
        qpos_future = torch.from_numpy(
            ((qpos_future_np - self.stats["qpos_mean"]) / self.stats["qpos_std"]).astype(np.float32)
        )

        act_action_chunk_raw, act_is_pad = self._pad_action_window(actions, start_ts, self.act_chunk_size)
        a_prefix_raw = act_action_chunk_raw[: self.prefix_steps].copy()
        is_pad_prefix = act_is_pad[: self.prefix_steps].copy()
        a_future_prefix_raw, is_pad_future = self._pad_action_window(
            actions, start_ts + self.future_offset, self.prefix_steps
        )
        act_action_future_chunk_raw, act_future_is_pad = self._pad_action_window(
            actions, start_ts + self.future_offset, self.act_chunk_size
        )

        act_action_chunk = (act_action_chunk_raw - self.stats["action_mean"]) / self.stats["action_std"]
        act_action_future_chunk = (
            act_action_future_chunk_raw - self.stats["action_mean"]
        ) / self.stats["action_std"]
        a_prefix = (a_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
        a_future_prefix = (a_future_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]

        return {
            "image_t": image_t,
            "image_t_future": image_t_future,
            "qpos_t": qpos_t,
            "qpos_raw": qpos_raw,
            "qpos_future": qpos_future,
            "qpos_future_raw": qpos_future_raw,
            "act_action_chunk": torch.from_numpy(act_action_chunk.astype(np.float32)),
            "act_action_future_chunk": torch.from_numpy(act_action_future_chunk.astype(np.float32)),
            "act_action_chunk_raw": torch.from_numpy(act_action_chunk_raw.astype(np.float32)),
            "act_action_future_chunk_raw": torch.from_numpy(act_action_future_chunk_raw.astype(np.float32)),
            "act_is_pad": torch.from_numpy(act_is_pad).bool(),
            "act_future_is_pad": torch.from_numpy(act_future_is_pad).bool(),
            "action_prefix": torch.from_numpy(a_prefix.astype(np.float32)),
            "action_future_prefix": torch.from_numpy(a_future_prefix.astype(np.float32)),
            "action_prefix_raw": torch.from_numpy(a_prefix_raw.astype(np.float32)),
            "action_future_prefix_raw": torch.from_numpy(a_future_prefix_raw.astype(np.float32)),
            "is_pad_prefix": torch.from_numpy(is_pad_prefix).bool(),
            "is_pad_future_prefix": torch.from_numpy(is_pad_future).bool(),
            "episode_id": torch.tensor(ep_id),
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


def resolve_multitask_specs(task_names: list[str], sim_task_configs: dict, raw_data_root_overrides: dict[str, str] | None = None) -> list[MultiTaskSpec]:
    specs: list[MultiTaskSpec] = []
    raw_data_root_overrides = raw_data_root_overrides or {}
    for task_name in task_names:
        if task_name not in sim_task_configs:
            raise KeyError(f"task_name={task_name} not found in SIM_TASK_CONFIGS")
        info = sim_task_configs[task_name]
        dataset_dir = info["dataset_dir"]
        if dataset_dir.startswith("./"):
            dataset_dir = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "ACT", dataset_dir[2:])
            )
        else:
            dataset_dir = os.path.realpath(dataset_dir)
        raw_override = raw_data_root_overrides.get(task_name)
        raw_data_dir = resolve_raw_data_dir(task_name, raw_override)
        specs.append(
            MultiTaskSpec(
                task_name=task_name,
                dataset_dir=dataset_dir,
                raw_data_dir=raw_data_dir,
                num_episodes=int(info["num_episodes"]),
                camera_names=tuple(info["camera_names"]),
            )
        )
    return specs
