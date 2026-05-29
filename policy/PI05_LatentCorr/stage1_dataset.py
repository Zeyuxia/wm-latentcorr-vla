from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import DEFAULT_ASSETS_BASE_DIR, DEFAULT_CAMERA_MODE, build_train_config


@dataclass(frozen=True)
class QuantileNormStats:
    q01: np.ndarray
    q99: np.ndarray

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)


def load_quantile_norm_stats(norm_stats_path: str | Path) -> tuple[QuantileNormStats, QuantileNormStats]:
    payload = json.loads(Path(norm_stats_path).read_text())["norm_stats"]
    state = payload["state"]
    action = payload["actions"]
    return (
        QuantileNormStats(q01=np.asarray(state["q01"], dtype=np.float32), q99=np.asarray(state["q99"], dtype=np.float32)),
        QuantileNormStats(q01=np.asarray(action["q01"], dtype=np.float32), q99=np.asarray(action["q99"], dtype=np.float32)),
    )


def _decode_jpeg(frame_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode jpeg frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _pad_to_dim(value: np.ndarray, target_dim: int) -> np.ndarray:
    if value.shape[-1] >= target_dim:
        return value[..., :target_dim].astype(np.float32)
    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, target_dim - value.shape[-1])
    return np.pad(value, pad_width, mode="constant").astype(np.float32)


class PI0Stage1Dataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        norm_stats_path: str | Path,
        *,
        prefix_steps: int = 16,
        future_offset: int = 16,
        action_horizon: int = 50,
        model_action_dim: int = 32,
        prompt_mode: str = "random",
        samples_per_epoch: int | None = None,
        train_config_name: str = "pi05_aloha_full_base",
        repo_id: str | None = None,
        assets_base_dir: str | Path = DEFAULT_ASSETS_BASE_DIR,
        camera_mode: str = DEFAULT_CAMERA_MODE,
    ):
        self.processed_dir = Path(processed_dir).expanduser().resolve()
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)
        self.action_horizon = int(action_horizon)
        self.model_action_dim = int(model_action_dim)
        self.prompt_mode = str(prompt_mode)
        self._sample_index: list[tuple[int, Path, Path, int]] = []
        if repo_id is None:
            raise ValueError("repo_id is required so Stage1 uses the same openpi state/action transforms as deploy.")
        self._state_action_transform = self._build_state_action_transform(
            train_config_name=train_config_name,
            repo_id=repo_id,
            assets_base_dir=assets_base_dir,
            camera_mode=camera_mode,
        )

        self.episode_dirs = sorted(path for path in self.processed_dir.glob("episode_*") if path.is_dir())
        if not self.episode_dirs:
            raise FileNotFoundError(f"No processed episodes found under {self.processed_dir}")

        self.index = []
        for episode_dir in self.episode_dirs:
            episode_id = int(episode_dir.name.split("_")[-1])
            hdf5_path = episode_dir / f"episode_{episode_id}.hdf5"
            if not hdf5_path.is_file():
                continue
            with h5py.File(hdf5_path, "r") as root:
                ep_len = int(root["action"].shape[0])
            if ep_len >= (self.future_offset + 1):
                self.index.append((episode_id, episode_dir, hdf5_path))
                max_start = ep_len - self.future_offset - 1
                for start_ts in range(max_start + 1):
                    self._sample_index.append((episode_id, episode_dir, hdf5_path, start_ts))

        if not self.index:
            raise ValueError(f"No valid episodes in {self.processed_dir} for future_offset={self.future_offset}")
        if not self._sample_index:
            raise ValueError(f"No valid frame windows in {self.processed_dir} for future_offset={self.future_offset}")

        self.samples_per_epoch = int(samples_per_epoch or len(self._sample_index))

    def _build_state_action_transform(
        self,
        *,
        train_config_name: str,
        repo_id: str,
        assets_base_dir: str | Path,
        camera_mode: str,
    ):
        import openpi.transforms as openpi_transforms

        cfg = build_train_config(
            train_config_name=train_config_name,
            repo_id=repo_id,
            exp_name="stage1_dataset_transform_probe",
            camera_mode=camera_mode,
            asset_id=repo_id,
            assets_base_dir=assets_base_dir,
        )
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        return openpi_transforms.compose(
            [
                *data_cfg.repack_transforms.inputs,
                *data_cfg.data_transforms.inputs,
                openpi_transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            ]
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _choose_prompt(self, episode_dir: Path) -> str:
        instruction_path = episode_dir / "instructions.json"
        payload = json.loads(instruction_path.read_text())
        instructions = payload.get("instructions", [])
        if not instructions:
            return "open laptop"
        if self.prompt_mode == "first":
            return str(instructions[0])
        return str(random.choice(instructions))

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

    def _transform_sample(
        self,
        *,
        image_chw_uint8: np.ndarray,
        state_raw: np.ndarray,
        actions_raw: np.ndarray | None = None,
    ) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "observation.state": np.asarray(state_raw, dtype=np.float32),
            "observation.images.cam_high": np.asarray(image_chw_uint8, dtype=np.uint8),
            "prompt": "",
        }
        if actions_raw is None:
            sample["action"] = np.zeros((1, np.asarray(state_raw).shape[-1]), dtype=np.float32)
        else:
            sample["action"] = np.asarray(actions_raw, dtype=np.float32)
        return self._state_action_transform(sample)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_idx = int(index) % len(self._sample_index)
        episode_id, episode_dir, hdf5_path, start_ts = self._sample_index[sample_idx]
        prompt = self._choose_prompt(episode_dir)
        with h5py.File(hdf5_path, "r") as root:
            actions_raw = root["action"][()]
            qpos_raw = root["observations/qpos"][()]
            images = root["observations/images/cam_high"]

            image_t = _decode_jpeg(images[start_ts])
            image_t1 = _decode_jpeg(images[start_ts + self.future_offset])
            image_t_chw = np.transpose(image_t, (2, 0, 1))
            image_t1_chw = np.transpose(image_t1, (2, 0, 1))

            qpos_t = qpos_raw[start_ts].astype(np.float32)
            qpos_t1 = qpos_raw[start_ts + self.future_offset].astype(np.float32)

            action_chunk_raw, act_is_pad = self._pad_action_window(actions_raw, start_ts, self.action_horizon)
            is_pad_prefix = act_is_pad[: self.prefix_steps].copy()

            current_transformed = self._transform_sample(
                image_chw_uint8=image_t_chw,
                state_raw=qpos_t,
                actions_raw=action_chunk_raw,
            )
            future_transformed = self._transform_sample(
                image_chw_uint8=image_t1_chw,
                state_raw=qpos_t1,
            )

            qpos_t_norm = np.asarray(current_transformed["state"], dtype=np.float32)
            qpos_t1_norm = np.asarray(future_transformed["state"], dtype=np.float32)
            action_chunk_norm = np.asarray(current_transformed["actions"], dtype=np.float32)
            action_prefix_norm = action_chunk_norm[: self.prefix_steps].copy()

            action_chunk_model = _pad_to_dim(action_chunk_norm, self.model_action_dim)
            action_mask = np.zeros((self.action_horizon, self.model_action_dim), dtype=np.float32)
            valid_steps = (~act_is_pad).astype(np.float32)[:, None]
            action_mask[:, : action_chunk_norm.shape[-1]] = valid_steps

        image_t = torch.from_numpy(image_t).permute(2, 0, 1).float() / 255.0
        image_t1 = torch.from_numpy(image_t1).permute(2, 0, 1).float() / 255.0
        return {
            "image_t": image_t,
            "image_t1": image_t1,
            "qpos_t_norm": torch.from_numpy(qpos_t_norm),
            "qpos_t1_norm": torch.from_numpy(qpos_t1_norm),
            "act_action_chunk": torch.from_numpy(action_chunk_model),
            "act_action_mask": torch.from_numpy(action_mask),
            "action_prefix": torch.from_numpy(action_prefix_norm),
            "is_pad_prefix": torch.from_numpy(is_pad_prefix),
            "episode_id": torch.tensor(episode_id, dtype=torch.long),
            "start_ts": torch.tensor(start_ts, dtype=torch.long),
            "prompt": prompt,
        }
