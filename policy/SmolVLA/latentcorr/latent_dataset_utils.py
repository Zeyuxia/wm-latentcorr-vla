from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def get_norm_stats(dataset_dir: str, num_episodes: int) -> dict[str, np.ndarray]:
    all_qpos = []
    all_action = []
    for episode_id in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as root:
            all_qpos.append(torch.from_numpy(root["/observations/qpos"][()]))
            all_action.append(torch.from_numpy(root["/action"][()]))
    if not all_qpos:
        raise ValueError(f"No episodes found in {dataset_dir}")

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


class EpisodicLatentWarmupDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        episode_ids: list[int],
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        act_chunk_size: int,
        prefix_steps: int,
        future_offset: int,
    ):
        self.dataset_dir = dataset_dir
        self.episode_ids = list(episode_ids)
        self.camera_names = list(camera_names)
        self.stats = norm_stats
        self.act_chunk_size = int(act_chunk_size)
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)

    def __len__(self) -> int:
        return len(self.episode_ids)

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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id = int(self.episode_ids[index])
        path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(path, "r") as root:
            actions = root["/action"][()]
            qpos_seq = root["/observations/qpos"][()]
            episode_len = int(actions.shape[0])
            max_start = episode_len - self.future_offset - 1
            if max_start < 0:
                raise ValueError(f"episode_{episode_id} too short for future_offset={self.future_offset}")
            start_ts = int(np.random.randint(0, max_start + 1))

            image_t = []
            image_t_future = []
            for camera_name in self.camera_names:
                image_t.append(root[f"/observations/images/{camera_name}"][start_ts])
                image_t_future.append(root[f"/observations/images/{camera_name}"][start_ts + self.future_offset])

        image_t_tensor = torch.from_numpy(np.stack(image_t, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        image_t_future_tensor = (
            torch.from_numpy(np.stack(image_t_future, axis=0)).permute(0, 3, 1, 2).float() / 255.0
        )

        qpos_raw = qpos_seq[start_ts].astype(np.float32)
        qpos_t = (qpos_raw - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        act_action_chunk_raw, act_is_pad = self._pad_action_window(actions, start_ts, self.act_chunk_size)
        action_prefix_raw = act_action_chunk_raw[: self.prefix_steps].copy()
        action_future_prefix_raw, is_pad_future = self._pad_action_window(
            actions, start_ts + self.future_offset, self.prefix_steps
        )

        act_action_chunk = (act_action_chunk_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action_prefix = (action_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action_future_prefix = (action_future_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
        is_pad_prefix = act_is_pad[: self.prefix_steps].copy()

        return {
            "image_t": image_t_tensor,
            "image_t_future": image_t_future_tensor,
            "qpos_t": torch.from_numpy(qpos_t.astype(np.float32)),
            "qpos_raw": torch.from_numpy(qpos_raw),
            "act_action_chunk": torch.from_numpy(act_action_chunk.astype(np.float32)),
            "act_action_chunk_raw": torch.from_numpy(act_action_chunk_raw.astype(np.float32)),
            "act_is_pad": torch.from_numpy(act_is_pad).bool(),
            "action_prefix": torch.from_numpy(action_prefix.astype(np.float32)),
            "action_future_prefix": torch.from_numpy(action_future_prefix.astype(np.float32)),
            "action_prefix_raw": torch.from_numpy(action_prefix_raw.astype(np.float32)),
            "action_future_prefix_raw": torch.from_numpy(action_future_prefix_raw.astype(np.float32)),
            "is_pad_prefix": torch.from_numpy(is_pad_prefix).bool(),
            "is_pad_future_prefix": torch.from_numpy(is_pad_future).bool(),
            "episode_id": torch.tensor(episode_id),
            "start_ts": torch.tensor(start_ts),
        }


def list_valid_episode_ids(dataset_dir: str, num_episodes: int, future_offset: int) -> list[int]:
    valid_ids = []
    for episode_id in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as root:
            if int(root["/action"].shape[0]) >= future_offset + 1:
                valid_ids.append(episode_id)
    return valid_ids


def load_processed_episode_window(
    dataset_dir: str,
    episode_id: int,
    camera_names: list[str],
    norm_stats: dict[str, np.ndarray],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    start_ts: int,
) -> dict[str, torch.Tensor | int]:
    path = os.path.join(dataset_dir, f"episode_{episode_id}.hdf5")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"processed episode not found: {path}")
    with h5py.File(path, "r") as root:
        actions = root["/action"][()]
        qpos_seq = root["/observations/qpos"][()]
        episode_len = int(actions.shape[0])
        max_start = episode_len - future_offset - 1
        if start_ts < 0 or start_ts > max_start:
            raise ValueError(f"start_ts={start_ts} out of range [0, {max_start}] for episode_{episode_id}")

        image_t = []
        image_t_future = []
        for camera_name in camera_names:
            image_t.append(root[f"/observations/images/{camera_name}"][start_ts])
            image_t_future.append(root[f"/observations/images/{camera_name}"][start_ts + future_offset])

    image_t_tensor = torch.from_numpy(np.stack(image_t, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    image_t_future_tensor = torch.from_numpy(np.stack(image_t_future, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    qpos_raw = qpos_seq[start_ts].astype(np.float32)
    qpos_t = (qpos_raw - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
    act_action_chunk_raw, act_is_pad = EpisodicLatentWarmupDataset._pad_action_window(actions, start_ts, act_chunk_size)
    action_prefix_raw = act_action_chunk_raw[:prefix_steps].copy()
    action_future_prefix_raw, is_pad_future = EpisodicLatentWarmupDataset._pad_action_window(
        actions, start_ts + future_offset, prefix_steps
    )
    action_chunk = (act_action_chunk_raw - norm_stats["action_mean"]) / norm_stats["action_std"]
    action_prefix = (action_prefix_raw - norm_stats["action_mean"]) / norm_stats["action_std"]
    action_future_prefix = (action_future_prefix_raw - norm_stats["action_mean"]) / norm_stats["action_std"]
    is_pad_prefix = act_is_pad[:prefix_steps].copy()

    return {
        "image_t": image_t_tensor,
        "image_t_future": image_t_future_tensor,
        "qpos_t": torch.from_numpy(qpos_t.astype(np.float32)),
        "qpos_raw": torch.from_numpy(qpos_raw.astype(np.float32)),
        "act_action_chunk": torch.from_numpy(action_chunk.astype(np.float32)),
        "act_action_chunk_raw": torch.from_numpy(act_action_chunk_raw.astype(np.float32)),
        "act_is_pad": torch.from_numpy(act_is_pad).bool(),
        "action_prefix": torch.from_numpy(action_prefix.astype(np.float32)),
        "action_future_prefix": torch.from_numpy(action_future_prefix.astype(np.float32)),
        "action_prefix_raw": torch.from_numpy(action_prefix_raw.astype(np.float32)),
        "action_future_prefix_raw": torch.from_numpy(action_future_prefix_raw.astype(np.float32)),
        "is_pad_prefix": torch.from_numpy(is_pad_prefix).bool(),
        "is_pad_future_prefix": torch.from_numpy(is_pad_future).bool(),
        "episode_id": int(episode_id),
        "start_ts": int(start_ts),
        "ep_len": int(episode_len),
    }


def resolve_raw_data_dir(task_name: str, raw_data_dir: str | None) -> str:
    if raw_data_dir is not None:
        path = os.path.realpath(raw_data_dir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"raw_data_dir not found: {path}")
        return path

    task = task_name[4:] if task_name.startswith("sim-") else task_name
    parts = task.split("-")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer raw_data_dir from task_name={task_name}")
    task_name_only = parts[0]
    task_config = "-".join(parts[1:])
    if task_config.rsplit("-", 1)[-1].isdigit():
        task_config = task_config.rsplit("-", 1)[0]

    candidates = [
        os.path.realpath(f"/data/zhenyangfan/RoboTwin/data/{task_name_only}/{task_config}/data"),
        os.path.realpath(f"/data/zhenyangfan/RoboTwin/data_eval/{task_name_only}/{task_config}_seed100k/data"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot resolve raw_data_dir for task_name={task_name}. Tried: {candidates}")


def load_raw_episode(raw_data_dir: str, episode_id: int) -> dict[str, np.ndarray | str | tuple[int, int] | None]:
    path = os.path.join(raw_data_dir, f"episode{int(episode_id)}.hdf5")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"raw episode not found: {path}")
    with h5py.File(path, "r") as root:
        frame0 = bytes(root["observation/head_camera/rgb"][0])
        image0 = cv2.imdecode(np.frombuffer(frame0, np.uint8), cv2.IMREAD_COLOR)
        if image0 is None:
            raise RuntimeError(f"Failed to decode first frame from {path}")
        height_native, width_native = image0.shape[:2]
        return {
            "episode_path": path,
            "left_endpose": root["endpose/left_endpose"][()].astype(np.float32),
            "right_endpose": root["endpose/right_endpose"][()].astype(np.float32),
            "left_gripper": root["endpose/left_gripper"][()].astype(np.float32),
            "right_gripper": root["endpose/right_gripper"][()].astype(np.float32),
            "gt_left_arm": root["joint_action/left_arm"][()].astype(np.float32)
            if ("joint_action" in root and "left_arm" in root["joint_action"])
            else None,
            "gt_right_arm": root["joint_action/right_arm"][()].astype(np.float32)
            if ("joint_action" in root and "right_arm" in root["joint_action"])
            else None,
            "intrinsic_cv": root["observation/head_camera/intrinsic_cv"][0].astype(np.float32),
            "extrinsic_cv": root["observation/head_camera/extrinsic_cv"][0].astype(np.float32),
            "native_resolution": (int(height_native), int(width_native)),
        }

