from __future__ import annotations

import argparse
import glob
import json
import os


def _collect_trial_files(src_dir: str, epoch: int | None) -> list[str]:
    if epoch is None:
        pattern = os.path.join(src_dir, "failure_trials_live_rank*.json")
    else:
        pattern = os.path.join(src_dir, f"failure_trials_epoch_{int(epoch):04d}_rank*.json")
    return sorted(glob.glob(pattern))


def _infer_meta(src_dir: str) -> dict[str, int]:
    meta = {
        "failure_phase_bins": None,
        "failure_translation_dir_bins": None,
        "failure_translation_mag_bins": None,
        "failure_rotation_dir_bins": None,
        "failure_rotation_mag_bins": None,
        "failure_explore_k": None,
    }
    meta_files = sorted(glob.glob(os.path.join(src_dir, "failure_meta_rank*.json")))
    if not meta_files:
        meta_files = sorted(glob.glob(os.path.join(src_dir, "failure_meta.json")))
    for path in meta_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        for key in list(meta.keys()):
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


def _build_entries(trials: list[dict], fail_thresh: float) -> list[dict]:
    stats: dict[tuple[str, int, int, str, int, int], dict[str, int]] = {}
    for trial in trials:
        try:
            key = (
                str(trial["phase_key"]),
                int(trial["phase_instance_idx"]),
                int(trial["phase_bin_id"]),
                str(trial["error_mode"]),
                int(trial["dir_bin_id"]),
                int(trial["mag_bin_id"]),
            )
            recoverable = bool(trial.get("recoverable", False))
        except Exception:
            continue
        if key not in stats:
            stats[key] = {"n": 0, "n_recover": 0}
        stats[key]["n"] += 1
        stats[key]["n_recover"] += int(recoverable)

    entries = []
    for key, count in sorted(stats.items()):
        phase_key, phase_instance_idx, phase_bin_id, error_mode, dir_bin_id, mag_bin_id = key
        n_trials = int(count["n"])
        n_recover = int(count["n_recover"])
        recover_rate = float(n_recover) / float(max(1, n_trials))
        if recover_rate > float(fail_thresh):
            continue
        entries.append(
            {
                "phase_key": str(phase_key),
                "phase_instance_idx": int(phase_instance_idx),
                "phase_bin_id": int(phase_bin_id),
                "error_mode": str(error_mode),
                "active_arm_pattern": None,
                "dir_bin_id": int(dir_bin_id),
                "mag_bin_id": int(mag_bin_id),
                "n_trials": int(n_trials),
                "n_recover": int(n_recover),
                "recover_rate": float(recover_rate),
                "fail_rate": float(1.0 - recover_rate),
                "weight": float(max(1e-6, 1.0 - recover_rate)),
            }
        )
    return entries


def merge_failure_dir(
    failure_dir: str,
    failure_fail_recover_rate_thresh: float,
    out_dir: str = "",
    epoch: int | None = None,
) -> str:
    src_dir = os.path.abspath(failure_dir)
    dst_dir = os.path.abspath(out_dir) if str(out_dir).strip() else src_dir
    os.makedirs(dst_dir, exist_ok=True)

    trial_files = _collect_trial_files(src_dir, epoch)
    if not trial_files:
        raise RuntimeError(f"No per-rank trial files found under: {src_dir}")

    trials: list[dict] = []
    for path in trial_files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            trials.extend(payload)

    meta = _infer_meta(src_dir)
    entries = _build_entries(trials, float(failure_fail_recover_rate_thresh))
    mode = "explore_merged_live" if epoch is None else "explore_merged_epoch"

    merged_trials_path = os.path.join(dst_dir, "failure_trials_merged.json")
    with open(merged_trials_path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)

    table = {
        "version": 4,
        "mode": mode,
        "epoch": (None if epoch is None else int(epoch)),
        "failure_phase_bins": int(meta["failure_phase_bins"]),
        "failure_translation_dir_bins": int(meta["failure_translation_dir_bins"]),
        "failure_translation_mag_bins": int(meta["failure_translation_mag_bins"]),
        "failure_rotation_dir_bins": int(meta["failure_rotation_dir_bins"]),
        "failure_rotation_mag_bins": int(meta["failure_rotation_mag_bins"]),
        "failure_explore_k": int(meta["failure_explore_k"]),
        "failure_fail_recover_rate_thresh": float(failure_fail_recover_rate_thresh),
        "entries": entries,
    }

    full_table_path = os.path.join(dst_dir, "failure_table.json")
    with open(full_table_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    for mode_key in ("translation", "rotation", "gripper_close"):
        mode_table = dict(table)
        mode_table["entries"] = [entry for entry in entries if str(entry.get("error_mode", "")) == mode_key]
        mode_table_path = os.path.join(dst_dir, f"failure_table_{mode_key}.json")
        with open(mode_table_path, "w", encoding="utf-8") as f:
            json.dump(mode_table, f, indent=2, ensure_ascii=False)

    print(
        f"Merged {len(trial_files)} rank trial files, total trials={len(trials)}, entries={len(entries)}",
        flush=True,
    )
    print(f"Output: {full_table_path}", flush=True)
    return full_table_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-rank failure explore outputs into train-compatible failure tables."
    )
    parser.add_argument("--failure_dir", required=True, help="Directory containing failure_trials_*_rankXX.json")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch id for epoch files; default uses live files")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory; default=failure_dir")
    parser.add_argument(
        "--failure_fail_recover_rate_thresh",
        type=float,
        required=True,
        help="Recover-rate threshold used to keep failure entries.",
    )
    args = parser.parse_args()

    merge_failure_dir(
        failure_dir=args.failure_dir,
        out_dir=args.out_dir,
        epoch=args.epoch,
        failure_fail_recover_rate_thresh=float(args.failure_fail_recover_rate_thresh),
    )


if __name__ == "__main__":
    main()
