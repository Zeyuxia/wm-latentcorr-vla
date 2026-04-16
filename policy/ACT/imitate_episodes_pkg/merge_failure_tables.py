from __future__ import annotations

import argparse
import glob
import json
import os


def _collect_trial_files(src_dir: str, epoch: int | None):
    if epoch is None:
        pat = os.path.join(src_dir, "failure_trials_live_rank*.json")
    else:
        pat = os.path.join(src_dir, f"failure_trials_epoch_{int(epoch):04d}_rank*.json")
    return sorted(glob.glob(pat))


def _infer_meta(src_dir: str):
    meta = {
        "failure_phase_bins": None,
        "failure_translation_dir_bins": None,
        "failure_translation_mag_bins": None,
        "failure_rotation_dir_bins": None,
        "failure_rotation_mag_bins": None,
        "failure_explore_k": None,
    }
    meta_files = sorted(glob.glob(os.path.join(src_dir, "failure_meta_rank*.json")))
    if len(meta_files) == 0:
        meta_files = sorted(glob.glob(os.path.join(src_dir, "failure_meta.json")))
    for p in meta_files:
        try:
            with open(p, "r") as f:
                t = json.load(f)
            for k in list(meta.keys()):
                if k in t:
                    meta[k] = t[k]
            break
        except Exception:
            continue
    missing = [k for k, v in meta.items() if v is None]
    if len(missing) > 0:
        raise RuntimeError(
            "Missing required failure meta in input files: "
            + ", ".join(missing)
            + ". Please provide failure_meta[_rankXX].json with these fields."
        )
    return meta


def _build_entries(trials, fail_thresh: float):
    stats = {}
    for t in trials:
        try:
            key = (
                str(t["phase_key"]),
                int(t["phase_instance_idx"]),
                int(t["phase_bin_id"]),
                str(t["error_mode"]),
                int(t["dir_bin_id"]),
                int(t["mag_bin_id"]),
            )
            recoverable = bool(t.get("recoverable", False))
        except Exception:
            continue
        if key not in stats:
            stats[key] = {"n": 0, "n_recover": 0}
        stats[key]["n"] += 1
        stats[key]["n_recover"] += int(recoverable)

    entries = []
    for key, cnt in sorted(stats.items()):
        phase_key, phase_instance_idx, phase_bin_id, error_mode, dir_bin_id, mag_bin_id = key
        n = int(cnt["n"])
        n_recover = int(cnt["n_recover"])
        recover_rate = float(n_recover) / float(max(1, n))
        if not bool(recover_rate <= float(fail_thresh)):
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
                "n_trials": int(n),
                "n_recover": int(n_recover),
                "recover_rate": float(recover_rate),
                "fail_rate": float(1.0 - recover_rate),
                "weight": float(max(1e-6, 1.0 - recover_rate)),
            }
        )
    return entries


def main():
    ap = argparse.ArgumentParser(description="Merge per-rank failure explore outputs into train-compatible tables.")
    ap.add_argument("--failure_dir", required=True, help="Directory containing failure_trials_*_rankXX.json")
    ap.add_argument("--epoch", type=int, default=None, help="Epoch id for epoch files; default uses live files")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory; default=failure_dir")
    ap.add_argument(
        "--failure_fail_recover_rate_thresh",
        type=float,
        required=True,
        help="Recover-rate threshold used to keep failure entries.",
    )
    args = ap.parse_args()

    src_dir = os.path.abspath(args.failure_dir)
    out_dir = os.path.abspath(args.out_dir) if str(args.out_dir).strip() else src_dir
    os.makedirs(out_dir, exist_ok=True)

    trial_files = _collect_trial_files(src_dir, args.epoch)
    if len(trial_files) == 0:
        raise RuntimeError(f"No per-rank trial files found under: {src_dir}")

    trials = []
    for p in trial_files:
        with open(p, "r") as f:
            arr = json.load(f)
        if isinstance(arr, list):
            trials.extend(arr)

    meta = _infer_meta(src_dir)
    entries = _build_entries(
        trials,
        float(args.failure_fail_recover_rate_thresh),
    )

    mode = "explore_merged_live" if args.epoch is None else "explore_merged_epoch"
    out_trials = os.path.join(out_dir, "failure_trials_merged.json")
    with open(out_trials, "w") as f:
        json.dump(trials, f, indent=2)

    table = {
        "version": 4,
        "mode": mode,
        "epoch": (None if args.epoch is None else int(args.epoch)),
        "failure_phase_bins": int(meta["failure_phase_bins"]),
        "failure_translation_dir_bins": int(meta["failure_translation_dir_bins"]),
        "failure_translation_mag_bins": int(meta["failure_translation_mag_bins"]),
        "failure_rotation_dir_bins": int(meta["failure_rotation_dir_bins"]),
        "failure_rotation_mag_bins": int(meta["failure_rotation_mag_bins"]),
        "failure_explore_k": int(meta["failure_explore_k"]),
        "failure_fail_recover_rate_thresh": float(args.failure_fail_recover_rate_thresh),
        "entries": entries,
    }

    out_all = os.path.join(out_dir, "failure_table.json")
    with open(out_all, "w") as f:
        json.dump(table, f, indent=2)

    for mode_key in ("translation", "rotation", "gripper_close"):
        t_mode = dict(table)
        t_mode["entries"] = [e for e in entries if str(e.get("error_mode", "")) == mode_key]
        out_mode = os.path.join(out_dir, f"failure_table_{mode_key}.json")
        with open(out_mode, "w") as f:
            json.dump(t_mode, f, indent=2)

    print(f"Merged {len(trial_files)} rank trial files, total trials={len(trials)}, entries={len(entries)}")
    print(f"Output: {out_all}")


if __name__ == "__main__":
    main()
