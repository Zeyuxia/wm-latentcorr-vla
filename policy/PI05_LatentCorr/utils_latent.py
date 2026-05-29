from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _episode_hdf5_path(dataset_dir: str | Path, episode_id: int) -> Path:
    base = Path(dataset_dir).expanduser().resolve()
    direct = base / f"episode_{episode_id}.hdf5"
    nested = base / f"episode_{episode_id}" / f"episode_{episode_id}.hdf5"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"processed episode not found for episode_{episode_id} under {base}")


def _episode_dir(dataset_dir: str | Path, episode_id: int) -> Path:
    base = Path(dataset_dir).expanduser().resolve()
    nested = base / f"episode_{episode_id}"
    if nested.is_dir():
        return nested
    return base


def _resolve_image_dataset(root: h5py.File, camera_name: str) -> h5py.Dataset:
    candidates = [
        f"/observations/images/{camera_name}",
        f"observations/images/{camera_name}",
    ]
    for key in candidates:
        if key in root:
            return root[key]
    available = []
    if "/observations/images" in root:
        available = list(root["/observations/images"].keys())
    elif "observations" in root and "images" in root["observations"]:
        available = list(root["observations"]["images"].keys())
    raise KeyError(f"camera {camera_name!r} not found. available={available}")


def _decode_image(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 1 and arr.dtype.kind in {"S", "O", "V"}:
        frame_bytes = bytes(arr)
        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode jpeg frame")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Unsupported frame shape: {arr.shape}, dtype={arr.dtype}")


def get_norm_stats(dataset_dir: str, num_episodes: int) -> dict[str, np.ndarray]:
    all_qpos = []
    all_action = []
    for ep in range(num_episodes):
        try:
            path = _episode_hdf5_path(dataset_dir, ep)
        except FileNotFoundError:
            continue
        with h5py.File(path, "r") as root:
            all_qpos.append(torch.from_numpy(root["/observations/qpos"][()]))
            all_action.append(torch.from_numpy(root["/action"][()]))

    if len(all_qpos) == 0:
        raise ValueError(f"No episodes found in {dataset_dir}")

    max_qpos_len = max(q.size(0) for q in all_qpos)
    max_action_len = max(a.size(0) for a in all_action)

    padded_qpos = []
    for qpos in all_qpos:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
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
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.stats = norm_stats
        self.act_chunk_size = act_chunk_size
        self.prefix_steps = prefix_steps
        self.future_offset = future_offset

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

    def __getitem__(self, index: int):
        ep_id = self.episode_ids[index]
        path = _episode_hdf5_path(self.dataset_dir, ep_id)
        with h5py.File(path, "r") as root:
            actions = root["/action"][()]
            qpos_seq = root["/observations/qpos"][()]
            ep_len = int(actions.shape[0])
            if ep_len < (self.future_offset + 1):
                raise ValueError(f"episode_{ep_id} too short: {ep_len}")

            max_start = ep_len - self.future_offset - 1
            if max_start < 0:
                raise ValueError(
                    f"episode_{ep_id} too short for future_offset={self.future_offset}: {ep_len}"
                )
            start_ts = np.random.randint(0, max_start + 1)
            image_t = []
            image_t_future = []
            for cam_name in self.camera_names:
                cam_ds = _resolve_image_dataset(root, cam_name)
                i0 = _decode_image(cam_ds[start_ts])
                i1 = _decode_image(cam_ds[start_ts + self.future_offset])
                image_t.append(i0)
                image_t_future.append(i1)

            image_t = np.stack(image_t, axis=0)
            image_t_future = np.stack(image_t_future, axis=0)
            image_t = torch.from_numpy(image_t).permute(0, 3, 1, 2).float() / 255.0
            image_t_future = torch.from_numpy(image_t_future).permute(0, 3, 1, 2).float() / 255.0

            qpos_t = qpos_seq[start_ts]
            qpos_raw = qpos_t.astype(np.float32)
            qpos_t = (qpos_t - self.stats["qpos_mean"]) / self.stats["qpos_std"]
            qpos_t = torch.from_numpy(qpos_t.astype(np.float32))
            qpos_raw = torch.from_numpy(qpos_raw)

            act_action_chunk_raw, act_is_pad = self._pad_action_window(actions, start_ts, self.act_chunk_size)
            a_prefix_raw = act_action_chunk_raw[: self.prefix_steps].copy()
            is_pad = act_is_pad[: self.prefix_steps].copy()

            a_future_prefix_raw, is_pad_future = self._pad_action_window(
                actions,
                start_ts + self.future_offset,
                self.prefix_steps,
            )

            act_action_chunk = (act_action_chunk_raw - self.stats["action_mean"]) / self.stats["action_std"]
            a_prefix = (a_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
            a_future_prefix = (a_future_prefix_raw - self.stats["action_mean"]) / self.stats["action_std"]
            act_action_chunk = torch.from_numpy(act_action_chunk.astype(np.float32))
            a_prefix = torch.from_numpy(a_prefix.astype(np.float32))
            a_future_prefix = torch.from_numpy(a_future_prefix.astype(np.float32))
            act_action_chunk_raw = torch.from_numpy(act_action_chunk_raw.astype(np.float32))
            a_prefix_raw = torch.from_numpy(a_prefix_raw.astype(np.float32))
            a_future_prefix_raw = torch.from_numpy(a_future_prefix_raw.astype(np.float32))
            act_is_pad = torch.from_numpy(act_is_pad).bool()
            is_pad = torch.from_numpy(is_pad).bool()
            is_pad_future = torch.from_numpy(is_pad_future).bool()

        return {
            "image_t": image_t,
            "image_t_future": image_t_future,
            "qpos_t": qpos_t,
            "qpos_raw": qpos_raw,
            "act_action_chunk": act_action_chunk,
            "act_action_chunk_raw": act_action_chunk_raw,
            "act_is_pad": act_is_pad,
            "action_prefix": a_prefix,
            "action_future_prefix": a_future_prefix,
            "action_prefix_raw": a_prefix_raw,
            "action_future_prefix_raw": a_future_prefix_raw,
            "is_pad_prefix": is_pad,
            "is_pad_future_prefix": is_pad_future,
            "episode_id": torch.tensor(ep_id),
            "start_ts": torch.tensor(start_ts),
        }


def list_valid_episode_ids(
    dataset_dir: str,
    num_episodes: int,
    prefix_steps: int,
    future_offset: int,
) -> list[int]:
    del prefix_steps
    valid_ids = []
    for ep in range(num_episodes):
        try:
            path = _episode_hdf5_path(dataset_dir, ep)
        except FileNotFoundError:
            continue
        with h5py.File(path, "r") as root:
            min_len = future_offset + 1
            if int(root["/action"].shape[0]) >= min_len:
                valid_ids.append(ep)
    return valid_ids


def choose_eval_start_indices(
    ep_len: int,
    prefix_steps: int,
    future_offset: int,
    samples_per_episode: int,
) -> list[int]:
    del prefix_steps
    max_start = ep_len - future_offset - 1
    if max_start < 0:
        raise ValueError(f"episode too short for future_offset={future_offset}: {ep_len}")
    if samples_per_episode <= 1 or max_start == 0:
        return [max_start // 2]
    start_ids = np.linspace(0, max_start, num=samples_per_episode, dtype=np.int64).tolist()
    start_ids = sorted({int(x) for x in start_ids})
    return start_ids


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
    path = _episode_hdf5_path(dataset_dir, episode_id)
    with h5py.File(path, "r") as root:
        actions = root["/action"][()]
        qpos_seq = root["/observations/qpos"][()]
        ep_len = int(actions.shape[0])
        max_start = ep_len - future_offset - 1
        if start_ts < 0 or start_ts > max_start:
            raise ValueError(f"start_ts={start_ts} out of range [0, {max_start}] for episode_{episode_id}")

        image_t = []
        image_t_future = []
        for cam_name in camera_names:
            cam_ds = _resolve_image_dataset(root, cam_name)
            i0 = _decode_image(cam_ds[start_ts])
            i1 = _decode_image(cam_ds[start_ts + future_offset])
            image_t.append(i0)
            image_t_future.append(i1)

        image_t = np.stack(image_t, axis=0)
        image_t_future = np.stack(image_t_future, axis=0)
        image_t = torch.from_numpy(image_t).permute(0, 3, 1, 2).float() / 255.0
        image_t_future = torch.from_numpy(image_t_future).permute(0, 3, 1, 2).float() / 255.0

        qpos_raw = qpos_seq[start_ts].astype(np.float32)
        qpos_t = (qpos_raw - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos_t.astype(np.float32))
        qpos_raw = torch.from_numpy(qpos_raw.astype(np.float32))

        act_action_chunk_raw, act_is_pad = EpisodicLatentWarmupDataset._pad_action_window(
            actions, start_ts, act_chunk_size
        )
        a_prefix_raw = act_action_chunk_raw[:prefix_steps].copy()
        is_pad_prefix = act_is_pad[:prefix_steps].copy()
        a_future_prefix_raw, is_pad_future = EpisodicLatentWarmupDataset._pad_action_window(
            actions,
            start_ts + future_offset,
            prefix_steps,
        )
        act_action_chunk = (act_action_chunk_raw - norm_stats["action_mean"]) / norm_stats["action_std"]
        a_prefix = (a_prefix_raw - norm_stats["action_mean"]) / norm_stats["action_std"]
        a_future_prefix = (a_future_prefix_raw - norm_stats["action_mean"]) / norm_stats["action_std"]
        act_action_chunk = torch.from_numpy(act_action_chunk.astype(np.float32))
        a_prefix = torch.from_numpy(a_prefix.astype(np.float32))
        a_future_prefix = torch.from_numpy(a_future_prefix.astype(np.float32))
        act_action_chunk_raw = torch.from_numpy(act_action_chunk_raw.astype(np.float32))
        a_prefix_raw = torch.from_numpy(a_prefix_raw.astype(np.float32))
        a_future_prefix_raw = torch.from_numpy(a_future_prefix_raw.astype(np.float32))
        act_is_pad = torch.from_numpy(act_is_pad).bool()
        is_pad_prefix = torch.from_numpy(is_pad_prefix).bool()
        is_pad_future = torch.from_numpy(is_pad_future).bool()

    return {
        "image_t": image_t,
        "image_t_future": image_t_future,
        "qpos_t": qpos_t,
        "qpos_raw": qpos_raw,
        "act_action_chunk": act_action_chunk,
        "act_action_chunk_raw": act_action_chunk_raw,
        "act_is_pad": act_is_pad,
        "action_prefix": a_prefix,
        "action_future_prefix": a_future_prefix,
        "action_prefix_raw": a_prefix_raw,
        "action_future_prefix_raw": a_future_prefix_raw,
        "is_pad_prefix": is_pad_prefix,
        "is_pad_future": is_pad_future,
        "episode_id": int(episode_id),
        "start_ts": int(start_ts),
        "ep_len": int(ep_len),
    }


def build_stage1_dataloader(
    dataset_dir: str,
    num_episodes: int,
    camera_names: list[str],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, dict[str, np.ndarray]]:
    ds, stats = build_stage1_dataset(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dl, stats


def build_stage1_dataset(
    dataset_dir: str,
    num_episodes: int,
    camera_names: list[str],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
) -> tuple[EpisodicLatentWarmupDataset, dict[str, np.ndarray]]:
    stats = get_norm_stats(dataset_dir, num_episodes)
    valid_ids = list_valid_episode_ids(dataset_dir, num_episodes, prefix_steps, future_offset)
    ds = EpisodicLatentWarmupDataset(
        dataset_dir=dataset_dir,
        episode_ids=valid_ids,
        camera_names=camera_names,
        norm_stats=stats,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
    )
    return ds, stats


def resolve_raw_data_dir(task_name: str, raw_data_dir: str | None = None) -> str:
    if raw_data_dir is not None:
        path = os.path.realpath(raw_data_dir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"raw_data_dir not found: {path}")
        return path

    task = task_name
    if task.startswith("sim-"):
        task = task[len("sim-") :]

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
    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    raise FileNotFoundError(f"Cannot resolve raw_data_dir for task_name={task_name}. Tried: {candidates}")


def load_raw_episode(raw_data_dir: str, episode_id: int) -> dict[str, np.ndarray | str | tuple[int, int] | None]:
    path = os.path.join(raw_data_dir, f"episode{int(episode_id)}.hdf5")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"raw episode not found: {path}")

    with h5py.File(path, "r") as root:
        frame0 = bytes(root["observation/head_camera/rgb"][0])
        img0 = cv2.imdecode(np.frombuffer(frame0, np.uint8), cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError(f"Failed to decode first frame from {path}")
        h_native, w_native = img0.shape[:2]
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
            "native_resolution": (int(h_native), int(w_native)),
        }


def load_episode_prompt(dataset_dir: str | Path, episode_id: int, prompt_mode: str = "first") -> str:
    inst_path = _episode_dir(dataset_dir, episode_id) / "instructions.json"
    if not inst_path.is_file():
        return ""
    try:
        payload = json.loads(inst_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    instructions = payload.get("instructions", [])
    if not instructions:
        return ""
    if str(prompt_mode).strip().lower() == "random":
        return str(np.random.choice(instructions))
    return str(instructions[0])
