from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _stats(values: list[float]) -> dict[str, float | None]:
    vals = sorted(_finite(values))
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    n = len(vals)

    def pct(p: float) -> float:
        if n == 1:
            return vals[0]
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)

    return {
        "n": n,
        "mean": float(sum(vals) / n),
        "p50": float(pct(0.50)),
        "p90": float(pct(0.90)),
        "p95": float(pct(0.95)),
        "max": float(vals[-1]),
    }


def _counter_top(counter: Counter, k: int = 20) -> list[dict[str, Any]]:
    return [{"key": key, "count": int(value)} for key, value in counter.most_common(k)]


def _task_from_name(name: str) -> str:
    text = str(name)
    if text.startswith("sim-"):
        text = text[4:]
    marker = "-demo_clean"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text


def analyze_failure_tables(merged_dir: Path) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    global_failed_thresholds: Counter = Counter()
    global_modes: Counter = Counter()
    global_phase_modes: Counter = Counter()
    for task_dir in sorted(p for p in merged_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        trials_path = task_dir / "failure_trials_merged.json"
        table_path = task_dir / "failure_table.json"
        if not trials_path.exists():
            continue
        trials = _load_json(trials_path)
        valid = [t for t in trials if not bool(t.get("invalid_trial", False))]
        recover = [t for t in valid if bool(t.get("recoverable", False))]
        fail = [t for t in valid if not bool(t.get("recoverable", False))]
        failed_thresholds: Counter = Counter()
        modes: Counter = Counter()
        phase_modes: Counter = Counter()
        phase_bins: Counter = Counter()
        metrics: dict[str, list[float]] = defaultdict(list)
        for item in valid:
            mode = str(item.get("error_mode"))
            phase = str(item.get("phase_key"))
            modes[mode] += 1
            phase_modes[f"{phase}/{mode}"] += 1
            phase_bins[f"{phase}:inst{item.get('phase_instance_idx')}:bin{item.get('phase_bin_id')}"] += 1
            for failed_name in item.get("recover_eval_failed_thresholds") or []:
                failed_thresholds[str(failed_name)] += 1
            for key, value in (item.get("recover_eval_metrics") or {}).items():
                try:
                    metrics[str(key)].append(float(value))
                except Exception:
                    pass
        global_failed_thresholds.update(failed_thresholds)
        global_modes.update(modes)
        global_phase_modes.update(phase_modes)

        entries = []
        if table_path.exists():
            table = _load_json(table_path)
            entries = list(table.get("entries") or [])
        entry_fail_rates = [float(e.get("fail_rate", 0.0)) for e in entries]
        entry_weights = [float(e.get("weight", 0.0)) for e in entries]
        tasks[task] = {
            "n_trials": int(len(trials)),
            "n_valid": int(len(valid)),
            "n_invalid": int(len(trials) - len(valid)),
            "n_recover": int(len(recover)),
            "n_fail": int(len(fail)),
            "valid_recover_rate": float(len(recover) / max(1, len(valid))),
            "valid_fail_rate": float(len(fail) / max(1, len(valid))),
            "failure_table_entries": int(len(entries)),
            "failure_table_fail_rate_stats": _stats(entry_fail_rates),
            "failure_table_weight_stats": _stats(entry_weights),
            "error_mode_counts": dict(modes),
            "phase_mode_top": _counter_top(phase_modes, 20),
            "phase_bin_top": _counter_top(phase_bins, 20),
            "failed_threshold_counts": dict(failed_thresholds),
            "recover_eval_metric_stats": {key: _stats(vals) for key, vals in sorted(metrics.items())},
        }
    return {
        "tasks": tasks,
        "global": {
            "failed_threshold_counts": dict(global_failed_thresholds),
            "error_mode_counts": dict(global_modes),
            "phase_mode_top": _counter_top(global_phase_modes, 30),
        },
    }


def _summarize_window(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "loss",
        "loss_action",
        "loss_action_conditioned",
        "loss_dynamics",
        "clean_action_loss_eval",
        "corr_action_loss_eval",
        "clean_action_abs_mean_norm",
        "corr_source_action_abs_mean_norm",
        "corr_generated_action_abs_mean_norm",
    ]
    out = {key: _stats([float(r[key]) for r in records if r.get(key) is not None]) for key in keys}
    ratios = []
    for record in records:
        clean = record.get("clean_action_loss_eval")
        corr = record.get("corr_action_loss_eval")
        if clean is not None and corr is not None and float(clean) > 1e-8:
            ratios.append(float(corr) / float(clean))
    out["corr_to_clean_action_loss_eval"] = _stats(ratios)
    return out


def analyze_training_run(run_dir: Path) -> dict[str, Any]:
    trace_paths = sorted(run_dir.glob("stage1_correction_trace_rank*.jsonl"))
    records = []
    task_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    phase_counts: Counter = Counter()
    phase_mode_counts: Counter = Counter()
    skip_counts: Counter = Counter()
    for path in trace_paths:
        for record in _iter_jsonl(path):
            records.append(record)
            skip_counts.update(record.get("skip_reasons") or {})
            for unit in record.get("requested_units") or []:
                task_counts[_task_from_name(str(unit.get("task_name")))] += 1
                mode = str(unit.get("error_mode"))
                phase = str(unit.get("phase_key"))
                mode_counts[mode] += 1
                phase_counts[phase] += 1
                phase_mode_counts[f"{phase}/{mode}"] += 1
    records.sort(key=lambda x: (int(x.get("step", 0)), int(x.get("rank", 0))))
    n = len(records)
    first = records[: min(n, 200)]
    last = records[max(0, n - 200) :]
    corr_requested = [float(r.get("corr_requested", 0)) for r in records]
    corr_generated = [float(r.get("corr_generated", 0)) for r in records]
    corr_skipped = [float(r.get("corr_skipped", 0)) for r in records]
    base_batch = [float(r.get("base_batch_size", 0)) for r in records]
    actual_corr_frac = []
    for req, gen, base in zip(corr_requested, corr_generated, base_batch):
        denom = base + gen
        if denom > 0:
            actual_corr_frac.append(gen / denom)
    args_path = run_dir / "train_args.json"
    args = _load_json(args_path) if args_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "train_args": {
            "failure_corr_batch_ratio": args.get("failure_corr_batch_ratio"),
            "dyn_max_weight": args.get("dyn_max_weight"),
            "cond_max_weight": args.get("cond_max_weight"),
            "batch_size": args.get("batch_size"),
            "max_steps": args.get("max_steps"),
        },
        "num_trace_records": int(n),
        "corr_requested_stats": _stats(corr_requested),
        "corr_generated_stats": _stats(corr_generated),
        "corr_skipped_stats": _stats(corr_skipped),
        "actual_corr_fraction_stats": _stats(actual_corr_frac),
        "skip_reason_counts": dict(skip_counts),
        "sampled_task_counts": dict(task_counts),
        "sampled_error_mode_counts": dict(mode_counts),
        "sampled_phase_counts": dict(phase_counts),
        "sampled_phase_mode_top": _counter_top(phase_mode_counts, 30),
        "all_records": _summarize_window(records),
        "first_records": _summarize_window(first),
        "last_records": _summarize_window(last),
    }


def analyze_closed_loop_infos(run_dir: Path, max_files: int | None = None) -> dict[str, Any]:
    paths = []
    debug_root = run_dir / "debug_wm"
    if debug_root.exists():
        for root, _, files in os.walk(debug_root):
            if "closed_loop_info.json" in files:
                paths.append(Path(root) / "closed_loop_info.json")
                if max_files is not None and len(paths) >= max_files:
                    break
    recover_values = []
    first_unrecoverable = []
    failed_thresholds: Counter = Counter()
    mode_counts: Counter = Counter()
    metrics: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        try:
            item = _load_json(path)
        except Exception:
            continue
        last = item.get("recover_eval_last") or {}
        if last.get("recoverable") is not None:
            recover_values.append(1.0 if bool(last.get("recoverable")) else 0.0)
        if item.get("recover_eval_first_unrecoverable_step") is not None:
            first_unrecoverable.append(float(item.get("recover_eval_first_unrecoverable_step")))
        unit = item.get("sampled_unit") or {}
        if unit.get("error_mode") is not None:
            mode_counts[str(unit.get("error_mode"))] += 1
        for name in last.get("failed_thresholds") or []:
            failed_thresholds[str(name)] += 1
        for key, value in (last.get("metrics") or {}).items():
            try:
                metrics[str(key)].append(float(value))
            except Exception:
                pass
    return {
        "num_closed_loop_info": int(len(paths)),
        "recoverable_rate": (None if not recover_values else float(sum(recover_values) / len(recover_values))),
        "recoverable_count": int(sum(recover_values)),
        "unrecoverable_count": int(len(recover_values) - sum(recover_values)),
        "first_unrecoverable_step_stats": _stats(first_unrecoverable),
        "failed_threshold_counts": dict(failed_thresholds),
        "error_mode_counts": dict(mode_counts),
        "recover_eval_metric_stats": {key: _stats(vals) for key, vals in sorted(metrics.items())},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Analyze correction exploration/training signals.")
    parser.add_argument("--merged_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, action="append", default=[])
    parser.add_argument("--closed_loop_max_files", type=int, default=0)
    parser.add_argument("--output_json", type=str, required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    merged_dir = Path(args.merged_dir).resolve()
    out = {
        "merged_dir": str(merged_dir),
        "failure_tables": analyze_failure_tables(merged_dir),
        "training_runs": {},
    }
    max_files = None if int(args.closed_loop_max_files) <= 0 else int(args.closed_loop_max_files)
    for run in args.run_dir:
        run_dir = Path(run).resolve()
        out["training_runs"][run_dir.name] = {
            "trace": analyze_training_run(run_dir),
            "closed_loop": analyze_closed_loop_infos(run_dir, max_files=max_files),
        }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(out, file, indent=2, ensure_ascii=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
