from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from policy.SmolVLA.latentcorr.multitask_latent_utils import MultiTaskSpec, get_multitask_norm_stats
from policy.SmolVLA.latentcorr.stage2_failure_dataset import FailureAwareStage2Dataset


@dataclass(frozen=True)
class MultiTaskFailureDatasetConfig:
    act_chunk_size: int
    prefix_steps: int
    future_offset: int
    sample_phase_window_len: int
    start_margin: int
    failure_table_paths: dict[str, str]
    failure_phase_bins: int
    failure_translation_dir_bins: int
    failure_translation_mag_bins: int
    failure_rotation_dir_bins: int
    failure_rotation_mag_bins: int
    failure_explore_k: int


class MultiTaskFailureDataset(Dataset):
    def __init__(self, task_specs: list[MultiTaskSpec], config: MultiTaskFailureDatasetConfig, mode: str):
        if not task_specs:
            raise ValueError("task_specs must not be empty")
        mode_value = str(mode).strip().lower()
        if mode_value not in {"off", "train", "explore"}:
            raise ValueError(f"Invalid mode={mode!r}, expected one of off/train/explore")

        self.task_specs = list(task_specs)
        self.config = config
        self.mode = mode_value
        self.norm_stats = get_multitask_norm_stats(self.task_specs)
        self.datasets: list[FailureAwareStage2Dataset] = []
        self.index_map: list[tuple[int, int]] = []

        for task_idx, spec in enumerate(self.task_specs):
            dataset = FailureAwareStage2Dataset(
                dataset_dir=spec.dataset_dir,
                episode_ids=list(range(int(spec.num_episodes))),
                camera_names=list(spec.camera_names),
                norm_stats=self.norm_stats,
                act_chunk_size=int(config.act_chunk_size),
                prefix_steps=int(config.prefix_steps),
                future_offset=int(config.future_offset),
                sample_phase_window_len=int(config.sample_phase_window_len),
                start_margin=int(config.start_margin),
                failure_mode=self.mode,
                failure_table_path=self._resolve_failure_table_path(spec.task_name),
                failure_phase_bins=int(config.failure_phase_bins),
                failure_translation_dir_bins=int(config.failure_translation_dir_bins),
                failure_translation_mag_bins=int(config.failure_translation_mag_bins),
                failure_rotation_dir_bins=int(config.failure_rotation_dir_bins),
                failure_rotation_mag_bins=int(config.failure_rotation_mag_bins),
                failure_explore_k=int(config.failure_explore_k),
            )
            self.datasets.append(dataset)
            for local_index in range(len(dataset)):
                self.index_map.append((task_idx, local_index))

        if not self.index_map:
            raise ValueError("No valid multitask failure samples found")

    def _resolve_failure_table_path(self, task_name: str) -> str:
        if task_name not in self.config.failure_table_paths:
            raise KeyError(f"Missing failure table path for task {task_name!r}")
        path = self.config.failure_table_paths[task_name]
        if not path:
            raise ValueError(f"Empty failure table path for task {task_name!r}")
        return path

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict:
        task_idx, local_index = self.index_map[int(index)]
        spec = self.task_specs[task_idx]
        sample = dict(self.datasets[task_idx][local_index])
        sample["task_idx"] = torch.tensor(task_idx, dtype=torch.int64)
        sample["task_name"] = spec.task_name
        sample["raw_data_dir"] = spec.raw_data_dir
        sample["dataset_dir"] = spec.dataset_dir
        return sample


def build_multitask_failure_dataset(
    task_specs: list[MultiTaskSpec],
    config: MultiTaskFailureDatasetConfig,
    mode: str,
) -> tuple[MultiTaskFailureDataset, dict[str, np.ndarray]]:
    dataset = MultiTaskFailureDataset(task_specs=task_specs, config=config, mode=mode)
    return dataset, dataset.norm_stats
