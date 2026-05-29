#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
from typing import Any

from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.ACT_LatentCorr.stage2_failure_dataset import build_multitask_failure_table_dataset
from policy.ACT_LatentCorr.utils_multitask_latent import resolve_multitask_specs


def _iter_leaf_datasets(ds):
    if ds is None:
        return
    if hasattr(ds, "datasets"):
        for sub in ds.datasets:
            yield from _iter_leaf_datasets(sub)
        return
    if hasattr(ds, "dataset"):
        yield from _iter_leaf_datasets(ds.dataset)
        return
    yield ds


def _load_stage2_cfg(run_dir: str) -> dict[str, Any]:
    cfg_path = os.path.join(run_dir, "stage2_config.txt")
    with open(cfg_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    cfg = ast.literal_eval(first)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {cfg_path}")
    return cfg


def _build_multitask_leaves(run_dir: str):
    cfg = _load_stage2_cfg(run_dir)
    task_names = list(cfg.get("multi_task_names") or [])
    if not task_names:
        raise ValueError(f"Run is not multitask or missing multi_task_names: {run_dir}")
    task_specs = resolve_multitask_specs(task_names, SIM_TASK_CONFIGS)
    dataset, _ = build_multitask_failure_table_dataset(
        task_specs=task_specs,
        act_chunk_size=int(cfg["act_chunk_size"]),
        prefix_steps=int(cfg["prefix_steps"]),
        future_offset=int(cfg["future_offset"]),
        sample_phase_window_len=int(cfg.get("act_aligned_sample_pregrasp_phase_window_len", 16)),
        sample_skip_head_ratio=float(cfg.get("failure_sample_skip_head_ratio", 0.0)),
        start_margin=0,
        failure_mode="explore",
        failure_table_paths=[],
        failure_phase_bins=int(cfg["failure_phase_bins"]),
        failure_translation_dir_bins=int(cfg["failure_translation_dir_bins"]),
        failure_translation_mag_bins=int(cfg["failure_translation_mag_bins"]),
        failure_rotation_dir_bins=int(cfg["failure_rotation_dir_bins"]),
        failure_rotation_mag_bins=int(cfg["failure_rotation_mag_bins"]),
        failure_explore_k=int(cfg.get("failure_explore_k", 1)),
    )
    leaves = []
    for leaf in _iter_leaf_datasets(dataset):
        task_name = str(getattr(leaf, "_task_name", "")).strip()
        if task_name:
            leaves.append((task_name, leaf))
    if not leaves:
        raise RuntimeError(f"Failed to build multitask explore leaves for {run_dir}")
    return leaves


def _backfill_one_file(path: str, leaves, write_path: str | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload: {path}")

    stats = collections.Counter()
    matched_task_hist = collections.Counter()
    output = []

    for trial in payload:
        if not isinstance(trial, dict):
            stats["non_dict"] += 1
            output.append(trial)
            continue

        task_name = str(trial.get("task_name", "")).strip()
        if task_name:
            stats["already_labeled"] += 1
            matched_task_hist[task_name] += 1
            output.append(trial)
            continue

        matched = []
        for leaf_task, leaf in leaves:
            can_restore = getattr(leaf, "can_restore_explore_trial", None)
            if callable(can_restore) and bool(can_restore(trial)):
                matched.append(leaf_task)

        if len(matched) == 1:
            patched = dict(trial)
            patched["task_name"] = str(matched[0])
            output.append(patched)
            stats["backfilled"] += 1
            matched_task_hist[str(matched[0])] += 1
        elif len(matched) == 0:
            output.append(trial)
            stats["unmatched"] += 1
        else:
            patched = dict(trial)
            patched["_task_name_candidates"] = list(matched)
            output.append(patched)
            stats["ambiguous"] += 1

    if write_path:
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    return {
        "path": path,
        "write_path": write_path,
        "stats": dict(stats),
        "task_hist": dict(matched_task_hist),
        "total": int(len(payload)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing task_name in multitask explore trial files.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--input_subdir", default="failure_explore")
    parser.add_argument("--output_subdir", default="failure_explore_backfilled")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--summary_path", default="")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    in_dir = os.path.join(run_dir, args.input_subdir)
    out_dir = os.path.join(run_dir, args.output_subdir)

    leaves = _build_multitask_leaves(run_dir)
    paths = sorted(
        os.path.join(in_dir, name)
        for name in os.listdir(in_dir)
        if name.startswith("failure_trials_live_rank") and name.endswith(".json")
    )
    if not paths:
        raise FileNotFoundError(f"No failure_trials_live_rank*.json under {in_dir}")

    results = []
    agg = collections.Counter()
    agg_task_hist = collections.Counter()
    for path in paths:
        write_path = os.path.join(out_dir, os.path.basename(path)) if args.write else None
        result = _backfill_one_file(path, leaves, write_path=write_path)
        results.append(result)
        agg.update(result["stats"])
        agg_task_hist.update(result["task_hist"])
        print(
            os.path.basename(path),
            json.dumps(result["stats"], ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    summary = {
        "run_dir": run_dir,
        "input_subdir": args.input_subdir,
        "output_subdir": (args.output_subdir if args.write else None),
        "files": results,
        "aggregate_stats": dict(agg),
        "aggregate_task_hist": dict(agg_task_hist),
    }
    summary_path = args.summary_path.strip()
    if summary_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["aggregate_stats"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
