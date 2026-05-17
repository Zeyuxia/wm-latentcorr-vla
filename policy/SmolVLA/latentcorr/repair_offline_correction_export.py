from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARM_AND_GRIPPER_IDXS = np.arange(14, dtype=np.int64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    bad = 0
    if not path.is_file():
        return records, bad
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(item, dict):
                records.append(item)
    return records, bad


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(_jsonable(item), ensure_ascii=False) + "\n")


def _metric_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))[ARM_AND_GRIPPER_IDXS]))


def _resample_by_arclength(points: np.ndarray, n: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError(f"Expected points [T,D], got {points.shape}")
    if n <= 0:
        return np.zeros((0, points.shape[1]), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, n, axis=0).astype(np.float32)

    metric_points = points[:, ARM_AND_GRIPPER_IDXS]
    seg = np.linalg.norm(metric_points[1:] - metric_points[:-1], axis=1)
    cum = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg, dtype=np.float32)])
    total = float(cum[-1])
    if total <= 1e-8:
        return np.repeat(points[-1:], n, axis=0).astype(np.float32)

    # Use open-left/closed-right samples: first target is already after the
    # perturbed current state, last target reaches the clean attach state.
    target_s = np.linspace(total / float(n), total, int(n), dtype=np.float32)
    out = np.empty((int(n), points.shape[1]), dtype=np.float32)
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(target_s, cum, points[:, dim]).astype(np.float32)
    out[:, 6] = np.clip(out[:, 6], 0.0, 1.0)
    out[:, 13] = np.clip(out[:, 13], 0.0, 1.0)
    return out


def repair_vector(vector: np.ndarray, prefix_len: int, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    old = np.asarray(vector, dtype=np.float32)
    if old.ndim != 2 or old.shape[1] != 14:
        raise ValueError(f"Expected joint_action/vector [T,14], got {old.shape}")
    if old.shape[0] < 3:
        raise ValueError(f"Need at least 3 states to repair, got {old.shape[0]}")

    p = int(np.clip(int(prefix_len), 1, old.shape[0] - 1))
    new = old.copy()
    old_first = old[1].copy()
    old_prefix_end = old[p].copy()
    old_tail_first = old[p + 1].copy() if old.shape[0] > p + 1 else old_prefix_end.copy()

    if mode == "shift":
        # Remove the current-state/no-op label by shifting the whole target
        # sequence left by one frame, preserving shape by padding the last row.
        new[1:-1] = old[2:]
        new[-1] = old[-1]
    elif mode == "arc":
        # Reparameterize the correction segment from perturbed state to the
        # clean attach state. This keeps the original start observation, but
        # makes the first label a real recovery step instead of a no-op.
        path_end = min(old.shape[0], p + 2)
        path = np.concatenate([old[0:1], old[1:path_end]], axis=0)
        new[1 : 1 + p] = _resample_by_arclength(path, p)
        if old.shape[0] > p + 2:
            new[1 + p : -1] = old[2 + p :]
            new[-1] = old[-1]
        elif old.shape[0] > 1 + p:
            new[1 + p :] = old[-1]
    else:
        raise ValueError(f"Unsupported repair mode: {mode}")

    stats = {
        "prefix_len": int(p),
        "mode": str(mode),
        "first_l2_before": _metric_l2(old_first, old[0]),
        "first_l2_after": _metric_l2(new[1], new[0]),
        "prefix_end_to_tail0_l2_before": _metric_l2(old_prefix_end, old_tail_first),
        "prefix_end_to_tail0_l2_after": (
            None if new.shape[0] <= p + 1 else _metric_l2(new[p], new[p + 1])
        ),
        "prefix_end_motion_before": _metric_l2(old_prefix_end, old[0]),
        "prefix_end_motion_after": _metric_l2(new[p], new[0]),
    }
    return new.astype(vector.dtype, copy=False), stats


def _copy_instruction(src_instruction: Path, dst_instruction: Path) -> None:
    dst_instruction.parent.mkdir(parents=True, exist_ok=True)
    if src_instruction.is_file():
        shutil.copy2(src_instruction, dst_instruction)
    else:
        dst_instruction.write_text("{}", encoding="utf-8")


def _repair_hdf5(path: Path, prefix_len: int, mode: str, source_config: str, dest_config: str) -> dict[str, Any]:
    with h5py.File(path, "r+") as f:
        if "joint_action/vector" not in f:
            raise KeyError(f"Missing joint_action/vector in {path}")
        vector_ds = f["joint_action/vector"]
        old_dtype = vector_ds.dtype
        vector = vector_ds[()].astype(np.float32)
        repaired, stats = repair_vector(vector, prefix_len=prefix_len, mode=mode)
        repaired = repaired.astype(old_dtype, copy=False)
        vector_ds[...] = repaired

        joint = f["joint_action"]
        if "left_arm" in joint:
            joint["left_arm"][...] = repaired[:, 0:6].astype(joint["left_arm"].dtype, copy=False)
        if "left_gripper" in joint:
            joint["left_gripper"][...] = repaired[:, 6].astype(joint["left_gripper"].dtype, copy=False)
        if "right_arm" in joint:
            joint["right_arm"][...] = repaired[:, 7:13].astype(joint["right_arm"].dtype, copy=False)
        if "right_gripper" in joint:
            joint["right_gripper"][...] = repaired[:, 13].astype(joint["right_gripper"].dtype, copy=False)

        f.attrs["repair_arcfix_version"] = "v1"
        f.attrs["repair_source_task_config"] = str(source_config)
        f.attrs["repair_dest_task_config"] = str(dest_config)
        f.attrs["repair_mode"] = str(mode)
        for key, value in stats.items():
            if value is not None and isinstance(value, (int, float, np.integer, np.floating)):
                f.attrs[f"repair_{key}"] = value
    return stats


def _infer_records_from_data(rank_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for data_path in sorted((rank_root / "data").glob("episode*.hdf5")):
        try:
            episode_id = int(data_path.stem.replace("episode", ""))
        except ValueError:
            episode_id = len(out)
        out.append(
            {
                "rank": int(rank_root.name.replace("rank", "")) if rank_root.name.startswith("rank") else -1,
                "episode_id": int(episode_id),
                "export_path": str(data_path),
                "instruction_path": str(rank_root / "instructions" / f"episode{episode_id}.json"),
            }
        )
    return out


def repair_task(
    *,
    data_root: Path,
    task: str,
    source_config: str,
    dest_config: str,
    mode: str,
    force: bool,
    dry_run: bool,
    max_files: int,
) -> dict[str, Any]:
    src_root = data_root / task / f"{source_config}_shards"
    dst_root = data_root / task / f"{dest_config}_shards"
    if not src_root.is_dir():
        return {"task": task, "status": "missing_source", "source": str(src_root)}
    if dst_root.exists() and not force and not dry_run:
        raise FileExistsError(f"Destination exists: {dst_root}. Use --force to replace this destination only.")
    if dst_root.exists() and force and not dry_run:
        shutil.rmtree(dst_root)
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    repaired_records = 0
    bad_json = 0
    stats_acc: dict[str, list[float]] = {
        "first_l2_before": [],
        "first_l2_after": [],
        "prefix_end_motion_before": [],
        "prefix_end_motion_after": [],
    }

    for rank_root in sorted(path for path in src_root.iterdir() if path.is_dir() and path.name.startswith("rank")):
        records, bad = _read_jsonl(rank_root / "correction_manifest.jsonl")
        bad_json += bad
        if not records:
            records = _infer_records_from_data(rank_root)
        dst_rank = dst_root / rank_root.name
        dst_records: list[dict[str, Any]] = []
        for record in records:
            if max_files > 0 and repaired_records >= max_files:
                break
            episode_id = int(record.get("merged_episode_id", record.get("episode_id", len(dst_records))))
            src_data = Path(record.get("merged_data_path", record.get("export_path", "")))
            if not src_data.is_absolute():
                src_data = rank_root / "data" / f"episode{episode_id}.hdf5"
            src_instruction = Path(record.get("merged_instruction_path", record.get("instruction_path", "")))
            if not src_instruction.is_absolute():
                src_instruction = rank_root / "instructions" / f"episode{episode_id}.json"
            if not src_data.is_file():
                continue

            dst_data = dst_rank / "data" / f"episode{episode_id}.hdf5"
            dst_instruction = dst_rank / "instructions" / f"episode{episode_id}.json"
            prefix_len = int(record.get("correction_prefix_len", 16) or 16)
            if dry_run:
                with h5py.File(src_data, "r") as f:
                    vector = f["joint_action/vector"][()].astype(np.float32)
                    prefix_len = int(f.attrs.get("correction_prefix_len", prefix_len))
                _, stats = repair_vector(vector, prefix_len=prefix_len, mode=mode)
            else:
                dst_data.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_data, dst_data)
                _copy_instruction(src_instruction, dst_instruction)
                stats = _repair_hdf5(
                    dst_data,
                    prefix_len=prefix_len,
                    mode=mode,
                    source_config=source_config,
                    dest_config=dest_config,
                )

            for key in stats_acc:
                value = stats.get(key)
                if value is not None:
                    stats_acc[key].append(float(value))

            new_record = dict(record)
            new_record.update(
                {
                    "task_config": str(dest_config),
                    "export_path": str(dst_data),
                    "instruction_path": str(dst_instruction),
                    "repair_arcfix_version": "v1",
                    "repair_source_task_config": str(source_config),
                    "repair_source_export_path": str(src_data),
                    "repair_mode": str(mode),
                    "repair_stats": stats,
                }
            )
            dst_records.append(new_record)
            repaired_records += 1

        if dst_records and not dry_run:
            _write_jsonl(dst_rank / "correction_manifest.jsonl", dst_records)
        if max_files > 0 and repaired_records >= max_files:
            break

    summary = {
        "task": task,
        "status": "ok",
        "source": str(src_root),
        "destination": str(dst_root),
        "dry_run": bool(dry_run),
        "mode": str(mode),
        "repaired_records": int(repaired_records),
        "bad_json_lines_skipped": int(bad_json),
    }
    for key, values in stats_acc.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size:
            summary[f"{key}_median"] = float(np.median(arr))
            summary[f"{key}_mean"] = float(np.mean(arr))
    return summary


def discover_tasks(data_root: Path, source_config: str) -> list[str]:
    tasks = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and (path / f"{source_config}_shards").is_dir():
            tasks.append(path.name)
    return tasks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Repair offline correction export action labels without re-running explore.")
    parser.add_argument("--data_root", type=Path, default=Path("/data/zhenyangfan/RoboTwin/data"))
    parser.add_argument("--source_task_config", type=str, required=True)
    parser.add_argument("--dest_task_config", type=str, default="")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--mode", choices=["arc", "shift"], default="arc")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--summary_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    data_root = args.data_root.resolve()
    source_config = str(args.source_task_config)
    dest_config = str(args.dest_task_config or f"{source_config}_arcfix_v1")
    tasks = list(args.tasks) if args.tasks else discover_tasks(data_root, source_config)
    if not tasks:
        raise FileNotFoundError(f"No tasks found under {data_root} for {source_config}_shards")
    summaries = [
        repair_task(
            data_root=data_root,
            task=task,
            source_config=source_config,
            dest_config=dest_config,
            mode=str(args.mode),
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            max_files=int(args.max_files),
        )
        for task in tasks
    ]
    payload = {
        "data_root": str(data_root),
        "source_task_config": source_config,
        "dest_task_config": dest_config,
        "mode": str(args.mode),
        "dry_run": bool(args.dry_run),
        "tasks": tasks,
        "summaries": summaries,
    }
    print(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False))
    if args.summary_json is not None and not args.dry_run:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
