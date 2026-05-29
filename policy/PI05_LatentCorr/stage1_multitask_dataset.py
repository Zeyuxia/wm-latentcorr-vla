from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import DEFAULT_ASSETS_BASE_DIR, build_train_config
from .multitask_utils import MultiTaskSpec
from .stage1_dataset import _decode_jpeg, _pad_to_dim


def _episode_hdf5_path(processed_dir: str | Path, episode_id: int) -> Path:
    episode_dir = Path(processed_dir).expanduser().resolve() / f"episode_{episode_id}"
    nested = episode_dir / f"episode_{episode_id}.hdf5"
    if nested.is_file():
        return nested
    direct = Path(processed_dir).expanduser().resolve() / f"episode_{episode_id}.hdf5"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"episode_{episode_id}.hdf5 not found under {processed_dir}")


def _camera_names_from_mode(camera_mode: str) -> list[str]:
    if camera_mode == "head_only":
        return ["cam_high"]
    if camera_mode == "tri_view":
        return ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    raise ValueError(f"Unsupported camera_mode: {camera_mode}")


def _compute_multitask_raw_stats(task_specs: list[MultiTaskSpec]) -> dict[str, np.ndarray]:
    all_qpos = []
    all_action = []
    for spec in task_specs:
        processed_dir = Path(spec.processed_dir)
        for episode_dir in sorted(processed_dir.glob("episode_*")):
            ep_id = int(episode_dir.name.split("_")[-1])
            hdf5_path = _episode_hdf5_path(processed_dir, ep_id)
            with h5py.File(hdf5_path, "r") as root:
                all_qpos.append(torch.from_numpy(root["observations/qpos"][()]))
                all_action.append(torch.from_numpy(root["action"][()]))

    if not all_qpos:
        raise ValueError("No episodes found in any multitask processed_dir")

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
    return {
        "qpos_mean": q.mean(dim=(0, 1)).numpy().astype(np.float32),
        "qpos_std": q.std(dim=(0, 1)).clamp_min(1e-2).numpy().astype(np.float32),
        "action_mean": a.mean(dim=(0, 1)).numpy().astype(np.float32),
        "action_std": a.std(dim=(0, 1)).clamp_min(1e-2).numpy().astype(np.float32),
    }


class PI0Stage1MultiTaskDataset(Dataset):
    def __init__(
        self,
        task_specs: list[MultiTaskSpec],
        *,
        prefix_steps: int = 16,
        future_offset: int = 16,
        action_horizon: int = 50,
        model_action_dim: int = 32,
        prompt_mode: str = "random",
        samples_per_epoch: int | None = None,
        train_config_name: str = "pi05_aloha_full_base",
        assets_base_dir: str | Path = DEFAULT_ASSETS_BASE_DIR,
    ):
        self.task_specs = list(task_specs)
        self.prefix_steps = int(prefix_steps)
        self.future_offset = int(future_offset)
        self.action_horizon = int(action_horizon)
        self.model_action_dim = int(model_action_dim)
        self.prompt_mode = str(prompt_mode)
        self.assets_base_dir = assets_base_dir
        self.train_config_name = str(train_config_name)
        self._sample_index: list[tuple[int, int, Path, int]] = []
        self._state_action_transforms: dict[int, Any] = {}

        for task_idx, spec in enumerate(self.task_specs):
            self._state_action_transforms[task_idx] = self._build_state_action_transform(spec)
            processed_dir = Path(spec.processed_dir)
            episode_dirs = sorted(path for path in processed_dir.glob("episode_*") if path.is_dir())
            for episode_dir in episode_dirs:
                episode_id = int(episode_dir.name.split("_")[-1])
                hdf5_path = _episode_hdf5_path(processed_dir, episode_id)
                with h5py.File(hdf5_path, "r") as root:
                    ep_len = int(root["action"].shape[0])
                if ep_len < (self.future_offset + 1):
                    continue
                max_start = ep_len - self.future_offset - 1
                for start_ts in range(max_start + 1):
                    self._sample_index.append((task_idx, episode_id, hdf5_path, start_ts))

        if not self._sample_index:
            raise ValueError("No valid multitask stage1 samples found")
        self.samples_per_epoch = int(samples_per_epoch or len(self._sample_index))

    def _build_state_action_transform(self, spec: MultiTaskSpec):
        import openpi.transforms as openpi_transforms

        cfg = build_train_config(
            train_config_name=self.train_config_name,
            repo_id=spec.repo_id,
            exp_name="stage1_multitask_transform_probe",
            camera_mode=spec.camera_mode,
            asset_id=spec.repo_id,
            assets_base_dir=self.assets_base_dir,
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

    def _choose_prompt(self, episode_dir: Path, default_task_name: str) -> str:
        instruction_path = episode_dir / "instructions.json"
        if not instruction_path.is_file():
            return default_task_name
        payload = json.loads(instruction_path.read_text())
        instructions = payload.get("instructions", [])
        if not instructions:
            return default_task_name
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
        task_idx: int,
        image_chw_uint8: np.ndarray,
        state_raw: np.ndarray,
        actions_raw: np.ndarray | None = None,
        prompt: str = "",
    ) -> dict[str, Any]:
        spec = self.task_specs[task_idx]
        sample: dict[str, Any] = {
            "observation.state": np.asarray(state_raw, dtype=np.float32),
            "observation.images.cam_high": np.asarray(image_chw_uint8, dtype=np.uint8),
            "prompt": str(prompt),
        }
        if actions_raw is None:
            sample["action"] = np.zeros((1, np.asarray(state_raw).shape[-1]), dtype=np.float32)
        else:
            sample["action"] = np.asarray(actions_raw, dtype=np.float32)
        return self._state_action_transforms[task_idx](sample)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_idx = int(index) % len(self._sample_index)
        task_idx, episode_id, hdf5_path, start_ts = self._sample_index[sample_idx]
        spec = self.task_specs[task_idx]
        episode_dir = hdf5_path.parent
        prompt = self._choose_prompt(episode_dir, spec.task_name)

        with h5py.File(hdf5_path, "r") as root:
            actions_raw = root["action"][()]
            qpos_raw = root["observations/qpos"][()]
            image_ds = root["observations/images/cam_high"]

            image_t = _decode_jpeg(image_ds[start_ts])
            image_t1 = _decode_jpeg(image_ds[start_ts + self.future_offset])
            image_t_chw = np.transpose(image_t, (2, 0, 1))
            image_t1_chw = np.transpose(image_t1, (2, 0, 1))

            qpos_t = qpos_raw[start_ts].astype(np.float32)
            qpos_t1 = qpos_raw[start_ts + self.future_offset].astype(np.float32)

            action_chunk_raw, act_is_pad = self._pad_action_window(actions_raw, start_ts, self.action_horizon)
            is_pad_prefix = act_is_pad[: self.prefix_steps].copy()

            current_transformed = self._transform_sample(
                task_idx=task_idx,
                image_chw_uint8=image_t_chw,
                state_raw=qpos_t,
                actions_raw=action_chunk_raw,
                prompt=prompt,
            )
            future_transformed = self._transform_sample(
                task_idx=task_idx,
                image_chw_uint8=image_t1_chw,
                state_raw=qpos_t1,
                prompt=prompt,
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
            "task_idx": torch.tensor(task_idx, dtype=torch.long),
            "task_name": spec.task_name,
            "raw_data_dir": "" if spec.raw_data_dir is None else spec.raw_data_dir,
            "prompt": prompt,
        }


def build_multitask_stage1_dataset(
    *,
    task_specs: list[MultiTaskSpec],
    prefix_steps: int = 16,
    future_offset: int = 16,
    action_horizon: int = 50,
    model_action_dim: int = 32,
    prompt_mode: str = "random",
    samples_per_epoch: int | None = None,
    train_config_name: str = "pi05_aloha_full_base",
    assets_base_dir: str | Path = DEFAULT_ASSETS_BASE_DIR,
) -> tuple[PI0Stage1MultiTaskDataset, dict[str, np.ndarray]]:
    stats = _compute_multitask_raw_stats(task_specs)
    dataset = PI0Stage1MultiTaskDataset(
        task_specs,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
        action_horizon=action_horizon,
        model_action_dim=model_action_dim,
        prompt_mode=prompt_mode,
        samples_per_epoch=samples_per_epoch,
        train_config_name=train_config_name,
        assets_base_dir=assets_base_dir,
    )
    return dataset, stats
