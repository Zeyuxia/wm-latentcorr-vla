from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def _collect_trial_files(src_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(src_dir, "failure_trials_live_rank*.json")))


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_meta(src_dir: str) -> dict[str, int]:
    meta_files = sorted(glob.glob(os.path.join(src_dir, "failure_meta_rank*.json")))
    if not meta_files:
        raise RuntimeError(f"No failure_meta_rank*.json found under {src_dir}")
    payload = _load_json(meta_files[0])
    keys = [
        "failure_phase_bins",
        "failure_translation_dir_bins",
        "failure_translation_mag_bins",
        "failure_rotation_dir_bins",
        "failure_rotation_mag_bins",
        "failure_explore_k",
    ]
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError(f"Missing failure meta keys: {missing}")
    return {key: int(payload[key]) for key in keys}


def _build_entries(trials: list[dict], fail_recover_rate_thresh: float) -> list[dict]:
    stats: dict[tuple, dict[str, Any]] = {}
    for trial in trials:
        try:
            key = (
                str(trial["phase_key"]),
                int(trial["phase_instance_idx"]),
                int(trial["phase_bin_id"]),
                str(trial["error_mode"]),
                str(trial.get("active_arm_pattern", "both")),
                int(trial["dir_bin_id"]),
                int(trial["mag_bin_id"]),
            )
        except Exception:
            continue
        if key not in stats:
            stats[key] = {"n": 0, "n_recover": 0}
        stats[key]["n"] += 1
        stats[key]["n_recover"] += int(bool(trial.get("recoverable", False)))

    entries = []
    for key, count in sorted(stats.items()):
        phase_key, phase_instance_idx, phase_bin_id, error_mode, active_arm_pattern, dir_bin_id, mag_bin_id = key
        n_trials = int(count["n"])
        n_recover = int(count["n_recover"])
        recover_rate = float(n_recover) / float(max(1, n_trials))
        if recover_rate > float(fail_recover_rate_thresh):
            continue
        fail_rate = float(1.0 - recover_rate)
        entries.append(
            {
                "phase_key": phase_key,
                "phase_instance_idx": int(phase_instance_idx),
                "phase_bin_id": int(phase_bin_id),
                "error_mode": error_mode,
                "active_arm_pattern": active_arm_pattern,
                "dir_bin_id": int(dir_bin_id),
                "mag_bin_id": int(mag_bin_id),
                "n_trials": n_trials,
                "n_recover": n_recover,
                "recover_rate": recover_rate,
                "fail_rate": fail_rate,
                "weight": float(max(1e-6, fail_rate)),
            }
        )
    return entries


def merge_failure_dir(
    *,
    failure_dir: str,
    out_dir: str = "",
    failure_fail_recover_rate_thresh: float = 0.5,
) -> dict[str, str]:
    src_dir = os.path.abspath(failure_dir)
    dst_dir = os.path.abspath(out_dir) if str(out_dir).strip() else src_dir
    os.makedirs(dst_dir, exist_ok=True)
    trial_files = _collect_trial_files(src_dir)
    if not trial_files:
        raise RuntimeError(f"No per-rank trial files found under {src_dir}")

    all_trials: list[dict] = []
    for path in trial_files:
        payload = _load_json(path)
        if isinstance(payload, list):
            all_trials.extend(payload)

    meta = _infer_meta(src_dir)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for trial in all_trials:
        by_task[str(trial.get("task_name", "unknown"))].append(trial)

    merged_trials_path = os.path.join(dst_dir, "failure_trials_merged.json")
    with open(merged_trials_path, "w", encoding="utf-8") as f:
        json.dump(all_trials, f, indent=2, ensure_ascii=False)

    output_paths: dict[str, str] = {}
    for task_name, task_trials in sorted(by_task.items()):
        entries = _build_entries(task_trials, float(failure_fail_recover_rate_thresh))
        task_dir = os.path.join(dst_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        table = {
            "version": 5,
            "mode": "pi05_explore_merged_live",
            "task_name": task_name,
            **meta,
            "failure_fail_recover_rate_thresh": float(failure_fail_recover_rate_thresh),
            "n_trials": int(len(task_trials)),
            "entries": entries,
        }
        table_path = os.path.join(task_dir, "failure_table.json")
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, ensure_ascii=False)
        output_paths[task_name] = table_path

    manifest_path = os.path.join(dst_dir, "failure_table_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "failure_dir": src_dir,
                "out_dir": dst_dir,
                "failure_trials_merged": merged_trials_path,
                "tasks": output_paths,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--failure-fail-recover-rate-thresh", type=float, default=0.5)
    args = parser.parse_args()
    output_paths = merge_failure_dir(
        failure_dir=args.failure_dir,
        out_dir=args.out_dir,
        failure_fail_recover_rate_thresh=args.failure_fail_recover_rate_thresh,
    )
    print(json.dumps(output_paths, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
