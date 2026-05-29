from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MultiTaskSpec:
    task_name: str
    processed_dir: str
    repo_id: str
    raw_data_dir: str | None = None
    camera_mode: str = "head_only"


def _normalize_path(path_like: str | Path) -> str:
    return str(Path(path_like).expanduser().resolve())


def resolve_multitask_specs(
    *,
    task_names: list[str],
    processed_dirs: list[str],
    repo_ids: list[str],
    raw_data_dirs: list[str] | None = None,
    camera_mode: str = "head_only",
) -> list[MultiTaskSpec]:
    if not task_names:
        raise ValueError("task_names must not be empty")
    if not (len(task_names) == len(processed_dirs) == len(repo_ids)):
        raise ValueError(
            "task_names, processed_dirs, repo_ids must have identical lengths: "
            f"{len(task_names)} / {len(processed_dirs)} / {len(repo_ids)}"
        )
    raw_data_dirs = raw_data_dirs or [""] * len(task_names)
    if len(raw_data_dirs) != len(task_names):
        raise ValueError(
            f"raw_data_dirs must have length {len(task_names)}, got {len(raw_data_dirs)}"
        )

    specs: list[MultiTaskSpec] = []
    for idx, task_name in enumerate(task_names):
        processed_dir = _normalize_path(processed_dirs[idx])
        if not Path(processed_dir).is_dir():
            raise FileNotFoundError(f"processed_dir not found for {task_name}: {processed_dir}")
        raw_dir = str(raw_data_dirs[idx]).strip()
        specs.append(
            MultiTaskSpec(
                task_name=str(task_name),
                processed_dir=processed_dir,
                repo_id=str(repo_ids[idx]),
                raw_data_dir=(_normalize_path(raw_dir) if raw_dir else None),
                camera_mode=str(camera_mode),
            )
        )
    return specs
