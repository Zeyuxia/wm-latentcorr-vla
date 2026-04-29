from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.ACT.util.fk_sapien import SapienFK
from policy.SmolVLA.latentcorr.act_aligned_correction import (
    ACTAlignedCorrectionBuilder,
    build_act_aligned_cfg_from_args,
)
from policy.SmolVLA.latentcorr.correction_evac_rollout import quat_geodesic_deg_wxyz
from policy.SmolVLA.latentcorr.correction_policy_adapter import SampleBoundSmolVLAAdapter
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.failure_manifest_utils import load_failure_table_paths
from policy.SmolVLA.latentcorr.failure_utils import active_arm_pattern_id_to_key, error_mode_id_to_key
from policy.SmolVLA.latentcorr.latent_dataset_utils import load_raw_episode
from policy.SmolVLA.latentcorr.multitask_failure_dataset import (
    MultiTaskFailureDatasetConfig,
    build_multitask_failure_dataset,
)
from policy.SmolVLA.latentcorr.multitask_latent_utils import (
    build_multitask_stage1_dataset,
    resolve_multitask_specs,
)
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentPolicy
from policy.SmolVLA.latentcorr.train_smolvla import (
    add_common_args,
    add_failure_train_args,
    build_base_policy,
    build_bridge_config,
    build_stage1_warmup_config,
    initialize_latent_policy_from_sample,
    load_evac_sample_size,
    parse_task_parts,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("validate generated correction samples")
    add_common_args(parser)
    add_failure_train_args(parser)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--validation_batch_size", type=int, default=1)
    parser.add_argument("--validation_output_json", type=str, required=True)
    return parser


def _to_device_batch(raw_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in raw_batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _denorm_actions(action_norm: torch.Tensor, norm_stats: dict[str, Any]) -> np.ndarray:
    action = action_norm.detach().cpu().float().numpy().astype(np.float32)
    return action * np.asarray(norm_stats["action_std"], dtype=np.float32) + np.asarray(
        norm_stats["action_mean"], dtype=np.float32
    )


def _denorm_qpos(qpos_norm: torch.Tensor, norm_stats: dict[str, Any]) -> np.ndarray:
    qpos = qpos_norm.detach().cpu().float().numpy().astype(np.float32)
    return qpos * np.asarray(norm_stats["qpos_std"], dtype=np.float32) + np.asarray(
        norm_stats["qpos_mean"], dtype=np.float32
    )


def _arm_slices(arm: str) -> tuple[slice, int]:
    if arm == "left_arm":
        return slice(0, 6), 6
    if arm == "right_arm":
        return slice(7, 13), 13
    raise ValueError(f"Unsupported arm: {arm}")


def _inactive_arm(arm: str) -> str:
    return "right_arm" if arm == "left_arm" else "left_arm"


def _pose_error_from_qpos(fk: SapienFK, qpos_a: np.ndarray, qpos_b: np.ndarray, arm: str) -> tuple[float, float]:
    pose_a = fk.forward(qpos_a[0:6], qpos_a[7:13])
    pose_b = fk.forward(qpos_b[0:6], qpos_b[7:13])
    side = "left" if arm == "left_arm" else "right"
    pos_a, quat_a = pose_a[side]
    pos_b, quat_b = pose_b[side]
    return float(np.linalg.norm(pos_a - pos_b)), float(quat_geodesic_deg_wxyz(quat_a, quat_b))


def _take_tail(seq: Any, start_idx: int, n: int, fallback: np.ndarray | float) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,) + np.asarray(fallback).shape, dtype=np.float32)
    if seq is None:
        return np.repeat(np.asarray(fallback, dtype=np.float32)[None, ...], n, axis=0)
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    start = int(np.clip(start_idx, 0, max(0, arr.shape[0])))
    tail = arr[start : start + n]
    if tail.shape[0] >= n:
        return tail.astype(np.float32)
    last = np.asarray(fallback, dtype=np.float32)
    if tail.shape[0] > 0:
        last = tail[-1]
    pad = np.repeat(last[None, ...], n - tail.shape[0], axis=0).astype(np.float32)
    if tail.shape[0] == 0:
        return pad
    return np.concatenate([tail.astype(np.float32), pad], axis=0).astype(np.float32)


def _expected_suffix_from_raw(
    raw_data: dict[str, Any],
    *,
    start_ts: int,
    suffix_len: int,
    active_arm: str,
    original_active_arm: str,
    error_mode: str,
    corr_qpos: np.ndarray,
) -> np.ndarray:
    expected = np.repeat(np.asarray(corr_qpos, dtype=np.float32)[None, :], suffix_len, axis=0)
    if suffix_len <= 0:
        return expected

    follow_abs = int(start_ts)
    gt_left_arm = raw_data.get("gt_left_arm")
    gt_right_arm = raw_data.get("gt_right_arm")
    left_grip = np.asarray(raw_data["left_gripper"], dtype=np.float32).reshape(-1)
    right_grip = np.asarray(raw_data["right_gripper"], dtype=np.float32).reshape(-1)

    use_left_gt = active_arm == "left_arm" or original_active_arm == "both"
    use_right_gt = active_arm == "right_arm" or original_active_arm == "both"
    if use_left_gt:
        expected[:, 0:6] = _take_tail(gt_left_arm, follow_abs, suffix_len, corr_qpos[0:6])
        grip_start = follow_abs + 1 if error_mode == "gripper_close" else follow_abs
        expected[:, 6] = _take_tail(left_grip, grip_start, suffix_len, np.array([corr_qpos[6]], dtype=np.float32))[:, 0]
    if use_right_gt:
        expected[:, 7:13] = _take_tail(gt_right_arm, follow_abs, suffix_len, corr_qpos[7:13])
        grip_start = follow_abs + 1 if error_mode == "gripper_close" else follow_abs
        expected[:, 13] = _take_tail(right_grip, grip_start, suffix_len, np.array([corr_qpos[13]], dtype=np.float32))[:, 0]
    expected[:, 6] = np.clip(expected[:, 6], 0.0, 1.0)
    expected[:, 13] = np.clip(expected[:, 13], 0.0, 1.0)
    return expected.astype(np.float32)


def _safe_stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([x for x in values if np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _summarize(records: list[dict[str, Any]], skipped: Counter[str]) -> dict[str, Any]:
    metric_keys = [
        "prefix_active_q_mse",
        "prefix_active_pos_err_m",
        "prefix_active_rot_err_deg",
        "prefix_active_gripper_err",
        "prefix_inactive_q_delta",
        "prefix_inactive_pos_delta_m",
        "prefix_inactive_rot_delta_deg",
        "prefix_inactive_gripper_delta",
        "suffix_gt_mse_all",
        "suffix_gt_mse_active",
        "suffix_gt_mse_inactive",
        "start_perturb_active_pos_m",
        "start_perturb_active_rot_deg",
    ]
    summary: dict[str, Any] = {
        "n_valid": len(records),
        "n_skipped": int(sum(skipped.values())),
        "skip_reasons": dict(skipped),
        "task_counts": dict(Counter(str(r.get("task_name")) for r in records)),
        "active_arm_counts": dict(Counter(str(r.get("active_arm")) for r in records)),
        "original_active_arm_counts": dict(Counter(str(r.get("original_active_arm")) for r in records)),
        "error_mode_counts": dict(Counter(str(r.get("error_mode")) for r in records)),
    }
    summary["metrics"] = {
        key: _safe_stats([float(record[key]) for record in records if key in record])
        for key in metric_keys
    }
    worst_keys = [
        "prefix_active_pos_err_m",
        "prefix_active_rot_err_deg",
        "prefix_inactive_q_delta",
        "suffix_gt_mse_all",
    ]
    summary["worst"] = {}
    for key in worst_keys:
        ranked = sorted(
            [record for record in records if key in record and np.isfinite(float(record[key]))],
            key=lambda x: float(x[key]),
            reverse=True,
        )[:5]
        summary["worst"][key] = [
            {
                "value": float(item[key]),
                "task_name": item.get("task_name"),
                "episode_id": item.get("episode_id"),
                "start_ts": item.get("start_ts"),
                "active_arm": item.get("active_arm"),
                "original_active_arm": item.get("original_active_arm"),
                "error_mode": item.get("error_mode"),
                "dir_bin_id": item.get("dir_bin_id"),
                "mag_bin_id": item.get("mag_bin_id"),
            }
            for item in ranked
        ]
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Correction Sample Validation ===")
    print(f"valid={summary.get('n_valid')} skipped={summary.get('n_skipped')}")
    print(f"tasks={summary.get('task_counts')}")
    print(f"active_arms={summary.get('active_arm_counts')}")
    print(f"error_modes={summary.get('error_mode_counts')}")
    print("\n-- metrics --")
    for key, stats in summary.get("metrics", {}).items():
        mean = stats.get("mean")
        p95 = stats.get("p95")
        max_value = stats.get("max")
        if mean is None:
            print(f"{key:34s} mean=None")
        else:
            print(f"{key:34s} mean={mean:.6g} p95={p95:.6g} max={max_value:.6g}")


def main() -> None:
    args = build_argparser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    normal_dataset, norm_stats = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
    )
    failure_table_paths = load_failure_table_paths(args.failure_table_paths_json)
    evac_sample_size = load_evac_sample_size(args.evac_config)
    failure_cfg = MultiTaskFailureDatasetConfig(
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
        sample_phase_window_len=int(args.sample_phase_window_len),
        start_margin=int(args.start_margin),
        perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
        perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
        evac_sample_size=evac_sample_size,
        failure_table_paths={str(key): str(value) for key, value in failure_table_paths.items()},
        failure_phase_bins=int(args.failure_phase_bins),
        failure_translation_dir_bins=int(args.failure_translation_dir_bins),
        failure_translation_mag_bins=int(args.failure_translation_mag_bins),
        failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
        failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
        failure_explore_k=int(args.failure_explore_k),
    )
    correction_dataset, _ = build_multitask_failure_dataset(task_specs=task_specs, config=failure_cfg, mode="train")
    loader = DataLoader(
        correction_dataset,
        batch_size=int(args.validation_batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    base_policy, preprocess, postprocess = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_stage1_warmup_config(args),
    )
    model.to(device)
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=device,
        load_rollout_model=True,
    )
    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=normal_dataset[0],
        teacher=teacher,
    )
    model.eval()
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )
    fk = SapienFK(args.urdf_path)

    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    prefix_steps = int(args.prefix_steps)
    act_chunk_size = int(args.act_chunk_size)
    max_samples = int(args.num_samples)
    progress = tqdm(total=max_samples, desc="validate-correction")
    for raw_batch in loader:
        raw_batch = _to_device_batch(raw_batch, device)
        batch_size = int(raw_batch["image_t"].shape[0])
        for sample_index in range(batch_size):
            if len(records) >= max_samples:
                break
            task_name_full = raw_batch["task_name"][sample_index]
            task_only, task_config = parse_task_parts(task_name_full)
            episode_id = int(raw_batch["episode_id"][sample_index].item())
            start_ts = int(raw_batch["start_ts"][sample_index].item())
            raw_dir = str(raw_batch["raw_data_dir"][sample_index])
            raw_key = (raw_dir, episode_id)
            if raw_key not in raw_cache:
                raw_cache[raw_key] = load_raw_episode(raw_dir, episode_id)
            raw_data = raw_cache[raw_key]
            adapter = SampleBoundSmolVLAAdapter(
                latent_policy=model,
                preprocess=preprocess,
                postprocess=postprocess,
                task_name=task_only,
                task_config=task_config,
                episode_id=episode_id,
                instruction_type=args.instruction_type,
            )
            correction = correction_builder.build(
                latent_model=adapter,
                image_t=raw_batch["image_t"][sample_index].to(device),
                qpos_t=raw_batch["qpos_t"][sample_index].to(device),
                raw_data=raw_data,
                norm_stats=norm_stats,
                start_ts=start_ts,
                failure_mode_override="train",
                sampled_phase_id=int(raw_batch["sampled_phase_id"][sample_index].item()),
                sampled_phase_bin_id=int(raw_batch["sampled_phase_bin_id"][sample_index].item()),
                sampled_phase_instance_id=int(raw_batch["sampled_phase_instance_id"][sample_index].item()),
                forced_error_mode_id=int(raw_batch["forced_error_mode_id"][sample_index].item()),
                sampled_active_arm_pattern_id=int(raw_batch["sampled_active_arm_pattern_id"][sample_index].item()),
                original_active_arm_pattern_id=int(raw_batch["original_active_arm_pattern_id"][sample_index].item()),
                forced_dir_bin_id=int(raw_batch["forced_dir_bin_id"][sample_index].item()),
                forced_mag_bin_id=int(raw_batch["forced_mag_bin_id"][sample_index].item()),
                sampled_mode_prob=float(raw_batch["sampled_mode_prob"][sample_index].item()),
                sampled_entry_prob_within_mode=float(raw_batch["sampled_entry_prob_within_mode"][sample_index].item()),
                sampled_unit_prob=float(raw_batch["sampled_unit_prob"][sample_index].item()),
                debug_dir=None,
                precomputed_action_chunk_norm=raw_batch["act_action_chunk"][sample_index],
            )
            if correction is None:
                skip_meta = correction_builder.pop_last_skip_meta() or {}
                skipped[str(skip_meta.get("skip_reason", "unknown_skip"))] += 1
                continue
            if correction["corr_action_chunk_norm"] is None or correction["corr_qpos_norm"] is None:
                skipped["missing_targets"] += 1
                continue

            active_arm = active_arm_pattern_id_to_key(int(raw_batch["sampled_active_arm_pattern_id"][sample_index].item()))
            original_active_arm = active_arm_pattern_id_to_key(
                int(raw_batch["original_active_arm_pattern_id"][sample_index].item())
            )
            error_mode = error_mode_id_to_key(int(raw_batch["forced_error_mode_id"][sample_index].item()))
            inactive = _inactive_arm(active_arm)
            active_slice, active_gripper_idx = _arm_slices(active_arm)
            inactive_slice, inactive_gripper_idx = _arm_slices(inactive)

            corr_action = _denorm_actions(correction["corr_action_chunk_norm"], norm_stats)
            corr_qpos = _denorm_qpos(correction["corr_qpos_norm"], norm_stats)
            clean_qpos = raw_batch["qpos_raw"][sample_index].detach().cpu().float().numpy().astype(np.float32)
            corr_prefix_last = corr_action[min(prefix_steps, corr_action.shape[0]) - 1]
            start_perturb_pos, start_perturb_rot = _pose_error_from_qpos(fk, corr_qpos, clean_qpos, active_arm)
            active_pos_err, active_rot_err = _pose_error_from_qpos(fk, corr_prefix_last, clean_qpos, active_arm)
            inactive_start_pos_delta, inactive_start_rot_delta = _pose_error_from_qpos(
                fk, corr_prefix_last, corr_qpos, inactive
            )

            suffix = corr_action[prefix_steps:act_chunk_size]
            expected_suffix = _expected_suffix_from_raw(
                raw_data,
                start_ts=start_ts,
                suffix_len=int(suffix.shape[0]),
                active_arm=active_arm,
                original_active_arm=original_active_arm,
                error_mode=error_mode,
                corr_qpos=corr_qpos,
            )
            suffix_diff = (
                suffix - expected_suffix if suffix.size and expected_suffix.size else np.zeros((0, 14), dtype=np.float32)
            )

            record = {
                "task_name": str(task_name_full),
                "episode_id": int(episode_id),
                "start_ts": int(start_ts),
                "active_arm": active_arm,
                "original_active_arm": original_active_arm,
                "error_mode": error_mode,
                "dir_bin_id": int(raw_batch["forced_dir_bin_id"][sample_index].item()),
                "mag_bin_id": int(raw_batch["forced_mag_bin_id"][sample_index].item()),
                "prefix_active_q_mse": float(np.mean((corr_prefix_last[active_slice] - clean_qpos[active_slice]) ** 2)),
                "prefix_active_pos_err_m": float(active_pos_err),
                "prefix_active_rot_err_deg": float(active_rot_err),
                "prefix_active_gripper_err": float(abs(corr_prefix_last[active_gripper_idx] - clean_qpos[active_gripper_idx])),
                "prefix_inactive_q_delta": float(
                    np.max(np.abs(corr_action[:prefix_steps, inactive_slice] - corr_qpos[inactive_slice]))
                ),
                "prefix_inactive_pos_delta_m": float(inactive_start_pos_delta),
                "prefix_inactive_rot_delta_deg": float(inactive_start_rot_delta),
                "prefix_inactive_gripper_delta": float(
                    np.max(np.abs(corr_action[:prefix_steps, inactive_gripper_idx] - corr_qpos[inactive_gripper_idx]))
                ),
                "suffix_gt_mse_all": float(np.mean(suffix_diff**2)) if suffix_diff.size else float("nan"),
                "suffix_gt_mse_active": float(np.mean(suffix_diff[:, list(range(active_slice.start, active_slice.stop))] ** 2))
                if suffix_diff.size
                else float("nan"),
                "suffix_gt_mse_inactive": float(
                    np.mean(suffix_diff[:, list(range(inactive_slice.start, inactive_slice.stop))] ** 2)
                )
                if suffix_diff.size
                else float("nan"),
                "start_perturb_active_pos_m": float(start_perturb_pos),
                "start_perturb_active_rot_deg": float(start_perturb_rot),
            }
            records.append(record)
            progress.update(1)
        if len(records) >= max_samples:
            break
    progress.close()

    summary = _summarize(records, skipped)
    payload = {"summary": summary, "records": records, "args": vars(args)}
    output_path = Path(args.validation_output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _print_summary(summary)
    print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
