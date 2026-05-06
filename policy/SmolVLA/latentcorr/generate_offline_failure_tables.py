from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.SmolVLA.latentcorr.multitask_latent_utils import resolve_multitask_specs
from policy.SmolVLA.latentcorr.stage2_failure_dataset import build_failure_table_dataset


def _task_dir_name(task_name: str) -> str:
    name = str(task_name).strip()
    if not name.startswith("sim-"):
        return name.replace("/", "_")
    parts = name.split("-")
    if len(parts) >= 2 and parts[1]:
        return str(parts[1])
    return name.replace("/", "_")


def _count_by(entries: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in entries:
        value = str(item.get(key, "unknown"))
        counts[value] = int(counts.get(value, 0) + 1)
    return counts


def _serialize_entries(entries: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for item in entries:
        serialized.append(
            {
                "phase_key": str(item["phase_key"]),
                "phase_instance_idx": int(item["phase_instance_idx"]),
                "phase_bin_id": int(item["phase_bin_id"]),
                "error_mode": str(item["error_mode"]),
                "active_arm_pattern": str(item["active_arm_pattern"]),
                "dir_bin_id": int(item.get("dir_bin_id", -1)),
                "mag_bin_id": int(item.get("mag_bin_id", -1)),
                "weight": float(item.get("weight", 1.0)),
                "n_trials": int(item.get("n_trials", 1)),
                "n_recover": int(item.get("n_recover", 0)),
                "fail_rate": float(item.get("fail_rate", 1.0)),
            }
        )
    return serialized


def generate_offline_failure_tables(
    output_root: str,
    multi_task_names: list[str],
    evac_config: str,
    future_offset: int,
    prefix_steps: int,
    act_chunk_size: int,
    sample_phase_window_len: int,
    start_margin: int,
    failure_phase_bins: int,
    failure_translation_dir_bins: int,
    failure_translation_mag_bins: int,
    failure_rotation_dir_bins: int,
    failure_rotation_mag_bins: int,
    perturb_eef_fail_gain: float,
    perturb_rot_max_deg: float,
    world_model_quality_record: str = "",
    world_model_quality_backend: str = "evac",
) -> str:
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    task_specs = resolve_multitask_specs(list(multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    evac_cfg = OmegaConf.load(evac_config)
    evac_sample_size = tuple(int(x) for x in evac_cfg.data.params.train.params.sample_size)

    manifest_tables: dict[str, str] = {}
    manifest_stats: dict[str, dict] = {}

    for spec in task_specs:
        dataset, _ = build_failure_table_dataset(
            dataset_dir=spec.dataset_dir,
            num_episodes=int(spec.num_episodes),
            camera_names=list(spec.camera_names),
            act_chunk_size=int(act_chunk_size),
            prefix_steps=int(prefix_steps),
            future_offset=int(future_offset),
            sample_phase_window_len=int(sample_phase_window_len),
            start_margin=int(start_margin),
            failure_mode="explore",
            failure_table_path="",
            failure_phase_bins=int(failure_phase_bins),
            failure_translation_dir_bins=int(failure_translation_dir_bins),
            failure_translation_mag_bins=int(failure_translation_mag_bins),
            failure_rotation_dir_bins=int(failure_rotation_dir_bins),
            failure_rotation_mag_bins=int(failure_rotation_mag_bins),
            failure_explore_k=1,
            raw_data_dir=spec.raw_data_dir,
            perturb_eef_fail_gain=float(perturb_eef_fail_gain),
            perturb_rot_max_deg=float(perturb_rot_max_deg),
            evac_sample_size=evac_sample_size,
            world_model_quality_record=str(world_model_quality_record),
            world_model_quality_backend=str(world_model_quality_backend),
        )
        entries = _serialize_entries(list(dataset._explore_units))
        task_dir = out_root / _task_dir_name(spec.task_name)
        task_dir.mkdir(parents=True, exist_ok=True)
        table_path = task_dir / "failure_table.json"
        payload = {
            "version": 1,
            "source": "offline_scan_from_processed_and_raw",
            "task_name": str(spec.task_name),
            "dataset_dir": str(spec.dataset_dir),
            "raw_data_dir": str(spec.raw_data_dir),
            "evac_config": str(Path(evac_config).resolve()),
            "evac_sample_size": [int(evac_sample_size[0]), int(evac_sample_size[1])],
            "failure_phase_bins": int(failure_phase_bins),
            "failure_translation_dir_bins": int(failure_translation_dir_bins),
            "failure_translation_mag_bins": int(failure_translation_mag_bins),
            "failure_rotation_dir_bins": int(failure_rotation_dir_bins),
            "failure_rotation_mag_bins": int(failure_rotation_mag_bins),
            "sample_phase_window_len": int(sample_phase_window_len),
            "start_margin": int(start_margin),
            "future_offset": int(future_offset),
            "prefix_steps": int(prefix_steps),
            "act_chunk_size": int(act_chunk_size),
            "perturb_eef_fail_gain": float(perturb_eef_fail_gain),
            "perturb_rot_max_deg": float(perturb_rot_max_deg),
            "entries": entries,
        }
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        manifest_tables[str(spec.task_name)] = str(table_path)
        manifest_stats[str(spec.task_name)] = {
            "num_entries": int(len(entries)),
            "task_dir": str(task_dir),
            "counts_by_phase": _count_by(entries, "phase_key"),
            "counts_by_active_arm": _count_by(entries, "active_arm_pattern"),
            "counts_by_error_mode": _count_by(entries, "error_mode"),
        }

    manifest = {
        "version": 1,
        "source": "offline_scan_from_processed_and_raw",
        "output_root": str(out_root),
        "evac_config": str(Path(evac_config).resolve()),
        "task_names": list(multi_task_names),
        "task_failure_tables": manifest_tables,
        "task_stats": manifest_stats,
    }
    manifest_path = out_root / "multitask_failure_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return str(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline-scan episodes and directly generate usable failure tables.")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--future_offset", type=int, default=16)
    parser.add_argument("--prefix_steps", type=int, default=16)
    parser.add_argument("--act_chunk_size", type=int, default=50)
    parser.add_argument("--sample_phase_window_len", type=int, default=20)
    parser.add_argument("--start_margin", type=int, default=0)
    parser.add_argument("--failure_phase_bins", type=int, default=3)
    parser.add_argument("--failure_translation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_translation_mag_bins", type=int, default=1)
    parser.add_argument("--failure_rotation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_rotation_mag_bins", type=int, default=1)
    parser.add_argument("--perturb_eef_fail_gain", type=float, default=0.05)
    parser.add_argument("--perturb_rot_max_deg", type=float, default=10.0)
    parser.add_argument("--world_model_quality_record", type=str, default="")
    parser.add_argument("--world_model_quality_backend", type=str, default="evac")
    args = parser.parse_args()

    manifest_path = generate_offline_failure_tables(
        output_root=args.output_root,
        multi_task_names=list(args.multi_task_names),
        evac_config=args.evac_config,
        future_offset=int(args.future_offset),
        prefix_steps=int(args.prefix_steps),
        act_chunk_size=int(args.act_chunk_size),
        sample_phase_window_len=int(args.sample_phase_window_len),
        start_margin=int(args.start_margin),
        failure_phase_bins=int(args.failure_phase_bins),
        failure_translation_dir_bins=int(args.failure_translation_dir_bins),
        failure_translation_mag_bins=int(args.failure_translation_mag_bins),
        failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
        failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
        perturb_eef_fail_gain=float(args.perturb_eef_fail_gain),
        perturb_rot_max_deg=float(args.perturb_rot_max_deg),
        world_model_quality_record=str(args.world_model_quality_record),
        world_model_quality_backend=str(args.world_model_quality_backend),
    )
    print(f"multitask_failure_manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
