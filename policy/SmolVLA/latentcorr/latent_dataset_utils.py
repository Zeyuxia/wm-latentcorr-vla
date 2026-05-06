from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SMOLVLA_ROOT = os.path.realpath(os.path.join(_THIS_DIR, ".."))
_SMOLVLA_SRC_DIR = os.path.join(_SMOLVLA_ROOT, "src")
_DEFAULT_HF_HOME = os.path.join(_SMOLVLA_ROOT, ".hf_home")


def act_processed_images_to_rgb(images: np.ndarray) -> np.ndarray:
    """Return processed ACT hdf5 images as RGB-semantic arrays.

    RoboTwin raw JPEGs were encoded from simulator RGB arrays through OpenCV,
    so OpenCV decode preserves the simulator RGB channel values numerically.
    The processed ACT hdf5 files therefore already carry RGB semantics even
    though they passed through cv2.  Keep the default as identity to avoid
    red/blue channel swaps in SmolVLA, EVAC input frames, and debug renders.
    Set SMOLVLA_ACT_HDF5_COLOR_ORDER=bgr only for a dataset that was truly
    written in OpenCV BGR order.
    """
    if images.ndim < 4 or int(images.shape[-1]) != 3:
        raise ValueError(f"Expected images with trailing channel dimension 3, got {images.shape}")
    color_order = str(os.environ.get("SMOLVLA_ACT_HDF5_COLOR_ORDER", "rgb")).strip().lower()
    if color_order == "bgr":
        return images[..., ::-1].copy()
    if color_order != "rgb":
        raise ValueError(
            f"Unsupported SMOLVLA_ACT_HDF5_COLOR_ORDER={color_order!r}; expected 'rgb' or 'bgr'"
        )
    return np.ascontiguousarray(images)


def _configure_lerobot_cache_env() -> None:
    os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_DEFAULT_HF_HOME, "datasets"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


def is_lerobot_dataset_dir(dataset_dir: str) -> bool:
    root = os.path.realpath(str(dataset_dir))
    return (
        os.path.isfile(os.path.join(root, "meta", "info.json"))
        and os.path.isdir(os.path.join(root, "data"))
        and os.path.isdir(os.path.join(root, "videos"))
    )


@lru_cache(maxsize=64)
def _load_lerobot_info(dataset_dir: str) -> dict:
    info_path = os.path.join(os.path.realpath(str(dataset_dir)), "meta", "info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_parquet_files(root: str) -> list[str]:
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".parquet"):
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


@lru_cache(maxsize=64)
def _load_lerobot_episodes_df(dataset_dir: str):
    import pandas as pd

    root = os.path.realpath(str(dataset_dir))
    paths = _iter_parquet_files(os.path.join(root, "meta", "episodes"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode metadata parquet found under {root}/meta/episodes")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def lerobot_dataset_signature_files(dataset_dir: str) -> list[str]:
    root = os.path.realpath(str(dataset_dir))
    paths = [
        os.path.join(root, "meta", "info.json"),
        os.path.join(root, "meta", "stats.json"),
        os.path.join(root, "meta", "tasks.parquet"),
    ]
    paths.extend(_iter_parquet_files(os.path.join(root, "meta", "episodes")))
    paths.extend(_iter_parquet_files(os.path.join(root, "data")))
    return [path for path in paths if os.path.isfile(path)]


def _get_lerobot_episode_row(dataset_dir: str, episode_id: int):
    episodes_df = _load_lerobot_episodes_df(os.path.realpath(str(dataset_dir)))
    matches = episodes_df[episodes_df["episode_index"].astype(int) == int(episode_id)]
    if matches.empty:
        raise FileNotFoundError(f"episode_index={int(episode_id)} not found in LeRobot dataset {dataset_dir}")
    return matches.iloc[0]


def get_lerobot_episode_length(dataset_dir: str, episode_id: int) -> int:
    row = _get_lerobot_episode_row(os.path.realpath(str(dataset_dir)), int(episode_id))
    return int(row["length"])


def _stack_vector_series(series) -> np.ndarray:
    values = [np.asarray(item, dtype=np.float32).reshape(-1) for item in series.to_list()]
    if not values:
        raise ValueError("Cannot stack an empty vector series")
    return np.stack(values, axis=0).astype(np.float32)


@lru_cache(maxsize=256)
def _load_lerobot_episode_arrays(dataset_dir: str, episode_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    import pandas as pd

    root = os.path.realpath(str(dataset_dir))
    episode_id = int(episode_id)
    info = _load_lerobot_info(root)
    row = _get_lerobot_episode_row(root, episode_id)
    chunk_index = int(row["data/chunk_index"])
    file_index = int(row["data/file_index"])
    data_path = os.path.join(root, info["data_path"].format(chunk_index=chunk_index, file_index=file_index))
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"LeRobot data parquet not found: {data_path}")
    df = pd.read_parquet(data_path)
    ep_df = df[df["episode_index"].astype(int) == episode_id].sort_values("frame_index")
    if ep_df.empty:
        raise FileNotFoundError(f"episode_index={episode_id} not found in {data_path}")
    qpos = _stack_vector_series(ep_df["observation.state"])
    action = _stack_vector_series(ep_df["action"])
    return qpos, action, int(row["dataset_from_index"])


def _load_lerobot_dataset(dataset_dir: str):
    _configure_lerobot_cache_env()
    if _SMOLVLA_SRC_DIR not in sys.path:
        sys.path.insert(0, _SMOLVLA_SRC_DIR)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = os.path.realpath(str(dataset_dir))
    repo_id = os.path.basename(os.path.normpath(root))
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        video_backend=os.environ.get("SMOLVLA_LEROBOT_VIDEO_BACKEND", "pyav"),
    )


@lru_cache(maxsize=16)
def _load_lerobot_dataset_cached(dataset_dir: str):
    return _load_lerobot_dataset(os.path.realpath(str(dataset_dir)))


def _resolve_lerobot_camera_key(dataset, camera_name: str) -> str:
    name = str(camera_name)
    if name in dataset.meta.camera_keys:
        return name
    aliases = {
        "cam_high": "observation.images.camera1",
        "camera1": "observation.images.camera1",
        "head_camera": "observation.images.camera1",
    }
    mapped = aliases.get(name, name)
    if mapped in dataset.meta.camera_keys:
        return mapped
    if len(dataset.meta.camera_keys) == 1:
        return str(dataset.meta.camera_keys[0])
    raise KeyError(f"Cannot map camera_name={camera_name!r}; available keys={dataset.meta.camera_keys}")


def _as_chw_float_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        out = image.detach().float().cpu()
    else:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
        out = torch.from_numpy(arr)
        if int(out.shape[-1]) == 3:
            out = out.permute(2, 0, 1)
        out = out.float()
    if out.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(out.shape)}")
    if int(out.shape[0]) != 3 and int(out.shape[-1]) == 3:
        out = out.permute(2, 0, 1)
    if float(out.max().item()) > 2.0:
        out = out / 255.0
    return torch.clamp(out, 0.0, 1.0).contiguous()


def load_lerobot_episode_window(
    dataset_dir: str,
    episode_id: int,
    camera_names: list[str],
    norm_stats: dict[str, np.ndarray],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    start_ts: int,
) -> dict[str, torch.Tensor | int]:
    root = os.path.realpath(str(dataset_dir))
    episode_id = int(episode_id)
    start_ts = int(start_ts)
    qpos_seq, actions, dataset_from_index = _load_lerobot_episode_arrays(root, episode_id)
    episode_len = int(actions.shape[0])
    max_start = episode_len - int(future_offset) - 1
    if start_ts < 0 or start_ts > max_start:
        raise ValueError(f"start_ts={start_ts} out of range [0, {max_start}] for episode_{episode_id}")

    dataset = _load_lerobot_dataset_cached(root)
    current_item = dataset[int(dataset_from_index) + start_ts]
    future_item = dataset[int(dataset_from_index) + start_ts + int(future_offset)]
    image_t = []
    image_t_future = []
    for camera_name in camera_names:
        key = _resolve_lerobot_camera_key(dataset, str(camera_name))
        image_t.append(_as_chw_float_tensor(current_item[key]))
        image_t_future.append(_as_chw_float_tensor(future_item[key]))

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
        "image_t": torch.stack(image_t, dim=0),
        "image_t_future": torch.stack(image_t_future, dim=0),
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


def get_episode_length(dataset_dir: str, episode_id: int) -> int:
    if is_lerobot_dataset_dir(dataset_dir):
        return get_lerobot_episode_length(os.path.realpath(str(dataset_dir)), int(episode_id))
    path = os.path.join(dataset_dir, f"episode_{int(episode_id)}.hdf5")
    with h5py.File(path, "r") as root:
        return int(root["/action"].shape[0])


def load_episode_action_array(dataset_dir: str, episode_id: int) -> np.ndarray:
    if is_lerobot_dataset_dir(dataset_dir):
        _, action, _ = _load_lerobot_episode_arrays(os.path.realpath(str(dataset_dir)), int(episode_id))
        return np.asarray(action, dtype=np.float32)
    path = os.path.join(dataset_dir, f"episode_{int(episode_id)}.hdf5")
    with h5py.File(path, "r") as root:
        return root["/action"][()].astype(np.float32)


def load_episode_qpos_action_arrays(dataset_dir: str, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    if is_lerobot_dataset_dir(dataset_dir):
        qpos, action, _ = _load_lerobot_episode_arrays(os.path.realpath(str(dataset_dir)), int(episode_id))
        return np.asarray(qpos, dtype=np.float32), np.asarray(action, dtype=np.float32)
    path = os.path.join(dataset_dir, f"episode_{int(episode_id)}.hdf5")
    with h5py.File(path, "r") as root:
        return root["/observations/qpos"][()].astype(np.float32), root["/action"][()].astype(np.float32)


def get_norm_stats(dataset_dir: str, num_episodes: int) -> dict[str, np.ndarray]:
    all_qpos = []
    all_action = []
    if is_lerobot_dataset_dir(dataset_dir):
        for episode_id in range(num_episodes):
            try:
                qpos, action, _ = _load_lerobot_episode_arrays(os.path.realpath(str(dataset_dir)), int(episode_id))
            except FileNotFoundError:
                continue
            all_qpos.append(torch.from_numpy(qpos))
            all_action.append(torch.from_numpy(action))
    else:
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
        if is_lerobot_dataset_dir(self.dataset_dir):
            episode_len = get_episode_length(self.dataset_dir, episode_id)
            max_start = episode_len - self.future_offset - 1
            if max_start < 0:
                raise ValueError(f"episode_{episode_id} too short for future_offset={self.future_offset}")
            start_ts = int(np.random.randint(0, max_start + 1))
            return load_lerobot_episode_window(
                dataset_dir=self.dataset_dir,
                episode_id=episode_id,
                camera_names=self.camera_names,
                norm_stats=self.stats,
                act_chunk_size=self.act_chunk_size,
                prefix_steps=self.prefix_steps,
                future_offset=self.future_offset,
                start_ts=start_ts,
            )
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

        image_t_np = act_processed_images_to_rgb(np.stack(image_t, axis=0))
        image_t_future_np = act_processed_images_to_rgb(np.stack(image_t_future, axis=0))
        image_t_tensor = torch.from_numpy(image_t_np).permute(0, 3, 1, 2).float() / 255.0
        image_t_future_tensor = (
            torch.from_numpy(image_t_future_np).permute(0, 3, 1, 2).float() / 255.0
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
    if is_lerobot_dataset_dir(dataset_dir):
        episodes_df = _load_lerobot_episodes_df(os.path.realpath(str(dataset_dir)))
        for episode_id in range(num_episodes):
            matches = episodes_df[episodes_df["episode_index"].astype(int) == int(episode_id)]
            if matches.empty:
                continue
            if int(matches.iloc[0]["length"]) >= future_offset + 1:
                valid_ids.append(episode_id)
        return valid_ids
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
    if is_lerobot_dataset_dir(dataset_dir):
        return load_lerobot_episode_window(
            dataset_dir=dataset_dir,
            episode_id=episode_id,
            camera_names=camera_names,
            norm_stats=norm_stats,
            act_chunk_size=act_chunk_size,
            prefix_steps=prefix_steps,
            future_offset=future_offset,
            start_ts=start_ts,
        )
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

    image_t_np = act_processed_images_to_rgb(np.stack(image_t, axis=0))
    image_t_future_np = act_processed_images_to_rgb(np.stack(image_t_future, axis=0))
    image_t_tensor = torch.from_numpy(image_t_np).permute(0, 3, 1, 2).float() / 255.0
    image_t_future_tensor = torch.from_numpy(image_t_future_np).permute(0, 3, 1, 2).float() / 255.0
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
