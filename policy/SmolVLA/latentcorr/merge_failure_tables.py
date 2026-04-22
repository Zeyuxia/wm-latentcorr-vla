from __future__ import annotations

import argparse
import glob
import json
import os


def _collect_trial_files(source_dir: str, epoch: int | None) -> list[str]:
    if epoch is None:
        pattern = os.path.join(source_dir, "failure_trials_live_rank*.json")
    else:
        pattern = os.path.join(source_dir, f"failure_trials_epoch_{int(epoch):04d}_rank*.json")
    return sorted(glob.glob(pattern))


def _infer_meta(source_dir: str) -> dict[str, int]:
    meta = {
        "failure_phase_bins": None,
        "failure_translation_dir_bins": None,
        "failure_translation_mag_bins": None,
        "failure_rotation_dir_bins": None,
        "failure_rotation_mag_bins": None,
        "failure_explore_k": None,
    }
    meta_files = sorted(glob.glob(os.path.join(source_dir, "failure_meta_rank*.json")))
    if not meta_files:
        meta_files = sorted(glob.glob(os.path.join(source_dir, "failure_meta.json")))
    for path in meta_files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in meta:
            if key in payload:
                meta[key] = payload[key]
        break
    missing = [key for key, value in meta.items() if value is None]
    if missing:
        raise RuntimeError(
            "Missing required failure meta in input files: "
            + ", ".join(missing)
            + ". Please provide failure_meta[_rankXX].json with these fields."
        )
    return {key: int(value) for key, value in meta.items()}


def _build_entries(trials: list[dict], fail_thresh: float) -> tuple[list[dict], dict]:
    stats: dict[tuple[str, int, int, str, str, int, int], dict[str, int]] = {}
    for trial in trials:
        active_arm_pattern = str(trial.get("active_arm_pattern", "both")).strip().lower()
        if active_arm_pattern == "left_arm":
            active_arm_pattern = "left_only"
        elif active_arm_pattern == "right_arm":
            active_arm_pattern = "right_only"
        if active_arm_pattern not in {"left_only", "right_only", "both"}:
            active_arm_pattern = "both"
        key = (
            str(trial["phase_key"]),
            int(trial["phase_instance_idx"]),
            int(trial["phase_bin_id"]),
            str(trial["error_mode"]),
            active_arm_pattern,
            int(trial["dir_bin_id"]),
            int(trial["mag_bin_id"]),
        )
        invalid_trial = bool(trial.get("invalid_trial", False))
        invalid_reason = str(trial.get("invalid_reason", "")).strip()
        recoverable = bool(trial.get("recoverable", False))
        if key not in stats:
            stats[key] = {"n": 0, "n_valid": 0, "n_recover": 0, "n_invalid_blurry": 0}
        stats[key]["n"] += 1
        if invalid_trial and invalid_reason == "evac_blurry_after_perturb":
            stats[key]["n_invalid_blurry"] += 1
            continue
        if invalid_trial:
            continue
        stats[key]["n_valid"] += 1
        stats[key]["n_recover"] += int(recoverable)

    entries = []
    excluded_blurry_units = []
    for key, count in sorted(stats.items()):
        phase_key, phase_instance_idx, phase_bin_id, error_mode, active_arm_pattern, dir_bin_id, mag_bin_id = key
        n_trials = int(count["n"])
        n_valid = int(count["n_valid"])
        n_recover = int(count["n_recover"])
        n_invalid_blurry = int(count["n_invalid_blurry"])
        blurry_rate = float(n_invalid_blurry) / float(max(1, n_trials))
        if n_invalid_blurry > (0.5 * float(n_trials)):
            excluded_blurry_units.append(
                {
                    "phase_key": str(phase_key),
                    "phase_instance_idx": int(phase_instance_idx),
                    "phase_bin_id": int(phase_bin_id),
                    "error_mode": str(error_mode),
                    "active_arm_pattern": str(active_arm_pattern),
                    "dir_bin_id": int(dir_bin_id),
                    "mag_bin_id": int(mag_bin_id),
                    "n_trials": int(n_trials),
                    "n_valid": int(n_valid),
                    "n_invalid_blurry": int(n_invalid_blurry),
                    "blurry_rate": float(blurry_rate),
                }
            )
            continue
        if n_valid <= 0:
            continue
        recover_rate = float(n_recover) / float(max(1, n_valid))
        if recover_rate > float(fail_thresh):
            continue
        entries.append(
            {
                "phase_key": str(phase_key),
                "phase_instance_idx": int(phase_instance_idx),
                "phase_bin_id": int(phase_bin_id),
                "error_mode": str(error_mode),
                "active_arm_pattern": str(active_arm_pattern),
                "dir_bin_id": int(dir_bin_id),
                "mag_bin_id": int(mag_bin_id),
                "n_trials": int(n_trials),
                "n_valid": int(n_valid),
                "n_recover": int(n_recover),
                "n_invalid_blurry": int(n_invalid_blurry),
                "blurry_rate": float(blurry_rate),
                "recover_rate": float(recover_rate),
                "fail_rate": float(1.0 - recover_rate),
                "weight": float(max(1e-6, 1.0 - recover_rate)),
            }
        )
    summary = {
        "num_units_seen": int(len(stats)),
        "num_units_excluded_blurry_majority": int(len(excluded_blurry_units)),
        "excluded_blurry_units": excluded_blurry_units,
    }
    return entries, summary


def merge_failure_dir(
    failure_dir: str,
    failure_fail_recover_rate_thresh: float,
    out_dir: str,
    epoch: int | None,
) -> str:
    source_dir = os.path.abspath(failure_dir)
    output_dir = os.path.abspath(out_dir) if str(out_dir).strip() else source_dir
    os.makedirs(output_dir, exist_ok=True)

    trial_files = _collect_trial_files(source_dir, epoch)
    if not trial_files:
        raise RuntimeError(f"No per-rank trial files found under: {source_dir}")

    trials: list[dict] = []
    for path in trial_files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            trials.extend(payload)

    meta = _infer_meta(source_dir)
    entries, merge_summary = _build_entries(trials, float(failure_fail_recover_rate_thresh))
    mode = "explore_merged_live" if epoch is None else "explore_merged_epoch"

    merged_trials_path = os.path.join(output_dir, "failure_trials_merged.json")
    with open(merged_trials_path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)

    table = {
        "version": 4,
        "mode": mode,
        "epoch": None if epoch is None else int(epoch),
        "failure_phase_bins": int(meta["failure_phase_bins"]),
        "failure_translation_dir_bins": int(meta["failure_translation_dir_bins"]),
        "failure_translation_mag_bins": int(meta["failure_translation_mag_bins"]),
        "failure_rotation_dir_bins": int(meta["failure_rotation_dir_bins"]),
        "failure_rotation_mag_bins": int(meta["failure_rotation_mag_bins"]),
        "failure_explore_k": int(meta["failure_explore_k"]),
        "failure_fail_recover_rate_thresh": float(failure_fail_recover_rate_thresh),
        "merge_summary": merge_summary,
        "entries": entries,
    }

    full_table_path = os.path.join(output_dir, "failure_table.json")
    with open(full_table_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    for mode_key in ("translation", "rotation", "gripper_close"):
        mode_table = dict(table)
        mode_table["entries"] = [entry for entry in entries if str(entry.get("error_mode", "")) == mode_key]
        with open(os.path.join(output_dir, f"failure_table_{mode_key}.json"), "w", encoding="utf-8") as f:
            json.dump(mode_table, f, indent=2, ensure_ascii=False)

    return full_table_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-rank failure explore outputs into failure tables.")
    parser.add_argument("--failure_dir", type=str, required=True)
    parser.add_argument("--failure_fail_recover_rate_thresh", type=float, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=False, default=None)
    args = parser.parse_args()

    merge_failure_dir(
        failure_dir=args.failure_dir,
        failure_fail_recover_rate_thresh=float(args.failure_fail_recover_rate_thresh),
        out_dir=args.out_dir,
        epoch=args.epoch,
    )


if __name__ == "__main__":
    main()
