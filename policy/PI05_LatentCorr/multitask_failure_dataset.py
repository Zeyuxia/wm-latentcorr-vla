from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .multitask_utils import MultiTaskSpec
from .stage2_failure_dataset import build_failure_table_dataset


def _camera_names_from_mode(camera_mode: str) -> list[str]:
    if camera_mode == "head_only":
        return ["cam_high"]
    if camera_mode == "tri_view":
        return ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    raise ValueError(f"Unsupported camera_mode: {camera_mode}")


def _count_processed_episodes(processed_dir: str | Path) -> int:
    return len(sorted(Path(processed_dir).expanduser().resolve().glob("episode_*")))


class PI0MultiTaskFailureDataset(Dataset):
    def __init__(
        self,
        *,
        task_specs: list[MultiTaskSpec],
        failure_table_paths: list[str],
        act_chunk_size: int,
        prefix_steps: int,
        future_offset: int,
        sample_phase_window_len: int,
        sample_skip_head_ratio: float | None,
        start_margin: int,
        failure_mode: str,
        failure_phase_bins: int,
        failure_translation_dir_bins: int,
        failure_translation_mag_bins: int,
        failure_rotation_dir_bins: int,
        failure_rotation_mag_bins: int,
        failure_explore_k: int = 1,
        samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        if len(task_specs) != len(failure_table_paths):
            raise ValueError(
                "task_specs and failure_table_paths must have identical lengths: "
                f"{len(task_specs)} vs {len(failure_table_paths)}"
            )

        self.task_specs = list(task_specs)
        self.task_datasets = []
        self.task_norm_stats: dict[str, dict] = {}
        self._index: list[tuple[int, int]] = []

        for task_idx, (spec, failure_table_path) in enumerate(zip(self.task_specs, failure_table_paths, strict=True)):
            num_episodes = _count_processed_episodes(spec.processed_dir)
            dataset, stats = build_failure_table_dataset(
                dataset_dir=spec.processed_dir,
                num_episodes=num_episodes,
                camera_names=_camera_names_from_mode(spec.camera_mode),
                act_chunk_size=act_chunk_size,
                prefix_steps=prefix_steps,
                future_offset=future_offset,
                sample_phase_window_len=sample_phase_window_len,
                sample_skip_head_ratio=sample_skip_head_ratio,
                start_margin=start_margin,
                failure_mode=failure_mode,
                failure_table_path=failure_table_path,
                failure_phase_bins=failure_phase_bins,
                failure_translation_dir_bins=failure_translation_dir_bins,
                failure_translation_mag_bins=failure_translation_mag_bins,
                failure_rotation_dir_bins=failure_rotation_dir_bins,
                failure_rotation_mag_bins=failure_rotation_mag_bins,
                failure_explore_k=failure_explore_k,
            )
            self.task_datasets.append(dataset)
            self.task_norm_stats[spec.task_name] = stats
            self._index.extend((task_idx, i) for i in range(len(dataset)))

        if not self._index:
            raise ValueError("No multitask failure samples found")
        self.samples_per_epoch = int(samples_per_epoch or len(self._index))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        task_idx, local_index = self._index[int(index) % len(self._index)]
        spec = self.task_specs[task_idx]
        sample = dict(self.task_datasets[task_idx][local_index])
        sample["task_idx"] = torch.tensor(task_idx, dtype=torch.long)
        sample["task_name"] = spec.task_name
        sample["processed_dir"] = spec.processed_dir
        sample["raw_data_dir"] = "" if spec.raw_data_dir is None else spec.raw_data_dir
        sample["repo_id"] = spec.repo_id
        return sample

    def set_explore_local_k(self, k_local: int) -> None:
        for dataset in self.task_datasets:
            if hasattr(dataset, "set_explore_local_k"):
                dataset.set_explore_local_k(k_local)

    def set_explore_unit_idx(self, unit_idx: int) -> None:
        for dataset in self.task_datasets:
            if hasattr(dataset, "set_explore_unit_idx"):
                dataset.set_explore_unit_idx(unit_idx)

    def get_explore_num_units(self) -> int:
        return int(sum(len(getattr(dataset, "_explore_units", [])) for dataset in self.task_datasets))

    def get_explore_completed_unit_count(self) -> int:
        return int(sum(int(getattr(dataset, "_explore_completed_unit_count", 0)) for dataset in self.task_datasets))

    def record_explore_trial(self, task_idx: int, unit_idx: int, episode_id: int, start_ts: int) -> None:
        task_idx = int(task_idx)
        if task_idx < 0 or task_idx >= len(self.task_datasets):
            return
        dataset = self.task_datasets[task_idx]
        if hasattr(dataset, "record_explore_trial"):
            dataset.record_explore_trial(unit_idx, episode_id, start_ts)


def build_multitask_failure_table_dataset(
    *,
    task_specs: list[MultiTaskSpec],
    failure_table_paths: list[str],
    act_chunk_size: int,
    prefix_steps: int,
    future_offset: int,
    sample_phase_window_len: int,
    sample_skip_head_ratio: float | None,
    start_margin: int,
    failure_mode: str,
    failure_phase_bins: int,
    failure_translation_dir_bins: int,
    failure_translation_mag_bins: int,
    failure_rotation_dir_bins: int,
    failure_rotation_mag_bins: int,
    failure_explore_k: int = 1,
    samples_per_epoch: int | None = None,
) -> tuple[PI0MultiTaskFailureDataset, dict[str, dict]]:
    dataset = PI0MultiTaskFailureDataset(
        task_specs=task_specs,
        failure_table_paths=failure_table_paths,
        act_chunk_size=act_chunk_size,
        prefix_steps=prefix_steps,
        future_offset=future_offset,
        sample_phase_window_len=sample_phase_window_len,
        sample_skip_head_ratio=sample_skip_head_ratio,
        start_margin=start_margin,
        failure_mode=failure_mode,
        failure_phase_bins=failure_phase_bins,
        failure_translation_dir_bins=failure_translation_dir_bins,
        failure_translation_mag_bins=failure_translation_mag_bins,
        failure_rotation_dir_bins=failure_rotation_dir_bins,
        failure_rotation_mag_bins=failure_rotation_mag_bins,
        failure_explore_k=failure_explore_k,
        samples_per_epoch=samples_per_epoch,
    )
    return dataset, dataset.task_norm_stats
