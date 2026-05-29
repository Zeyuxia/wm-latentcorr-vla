from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .common import DEFAULT_ASSETS_BASE_DIR, prepare_openpi_imports
from .evac_interface import EvacLatentTeacher
from .failure_utils import active_arm_pattern_id_to_key, error_mode_id_to_key, phase_id_to_key
from .merge_failure_tables import merge_failure_dir
from .multitask_failure_dataset import build_multitask_failure_table_dataset
from .multitask_utils import resolve_multitask_specs
from .train_stage1 import init_distributed_if_needed, is_main_process, set_seed, str2bool
from .train_stage1_unified_failure_multitask_latent import build_model, select_head_image
from .train_stage2_latent import PI0CorrectionPolicyAdapter, PI0NormAdapter
from .utils_latent import load_raw_episode


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PI0.5 multitask failure explore")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evac-ckpt", required=True)
    parser.add_argument("--evac-config", required=True)
    parser.add_argument("--train-config-name", default="pi05_aloha_robotwin_singleview_base")
    parser.add_argument("--camera-mode", default="head_only", choices=["head_only", "tri_view"])
    parser.add_argument("--assets-base-dir", default=str(DEFAULT_ASSETS_BASE_DIR))
    parser.add_argument("--pytorch-weight-path", default=None)
    parser.add_argument("--task-checkpoint-dir", default=None)
    parser.add_argument("--task-checkpoint-id", default="latest")
    parser.add_argument("--multi-task-names", nargs="+", required=True)
    parser.add_argument("--processed-dirs", nargs="+", required=True)
    parser.add_argument("--repo-ids", nargs="+", required=True)
    parser.add_argument("--raw-data-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--prefix-steps", type=int, default=16)
    parser.add_argument("--future-offset", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--projector-mid-channels", type=int, default=256)
    parser.add_argument("--predictor-hidden-dim", type=int, default=512)
    parser.add_argument("--predictor-num-blocks", type=int, default=3)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-action-conditioned", type=float, default=0.5)
    parser.add_argument("--lambda-align", type=float, default=0.0)
    parser.add_argument("--beta-dynamics-max", type=float, default=1.0)
    parser.add_argument("--dyn-zero-steps", type=int, default=0)
    parser.add_argument("--dyn-ramp-steps", type=int, default=2000)
    parser.add_argument("--dyn-warmup-curve", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--dyn-schedule-unit", default="step", choices=["step", "epoch"])
    parser.add_argument("--freeze-base-pi0", type=str2bool, default=False)
    parser.add_argument("--use-projector-detach-for-predictor", type=str2bool, default=True)
    parser.add_argument("--detach-act-feature-for-latent", type=str2bool, default=False)
    parser.add_argument("--use-raw-wm-targets", type=str2bool, default=True)
    parser.add_argument("--failure-explore-k", type=int, default=4)
    parser.add_argument("--failure-fail-recover-rate-thresh", type=float, default=0.5)
    parser.add_argument("--failure-phase-bins", type=int, default=3)
    parser.add_argument("--failure-translation-dir-bins", type=int, default=6)
    parser.add_argument("--failure-translation-mag-bins", type=int, default=3)
    parser.add_argument("--failure-rotation-dir-bins", type=int, default=6)
    parser.add_argument("--failure-rotation-mag-bins", type=int, default=3)
    parser.add_argument("--sample-phase-window-len", type=int, default=30)
    parser.add_argument("--sample-skip-head-ratio", type=float, default=0.6)
    parser.add_argument("--explore-debug-max-samples", type=int, default=0)
    parser.add_argument("--urdf-path", required=True)
    parser.add_argument("--curobo-left-yml", default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml")
    parser.add_argument("--curobo-right-yml", default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml")
    parser.add_argument("--max-rollout-steps", type=int, default=1)
    parser.add_argument("--planner-target-mode", default="backward", choices=["forward", "backward"])
    parser.add_argument("--planner-target-lookahead-steps", type=int, default=6)
    parser.add_argument("--planner-orient-weight", type=float, default=0.0573)
    parser.add_argument("--planner-gripper-penalty", type=float, default=1.0)
    parser.add_argument("--planner-nearest-window-radius", type=int, default=12)
    parser.add_argument("--planner-active-joint-delta-thresh", type=float, default=0.01)
    parser.add_argument("--planner-active-gripper-delta-thresh", type=float, default=0.05)
    parser.add_argument("--act-aligned-rollout-exec-steps", type=int, default=16)
    parser.add_argument("--act-aligned-min-dist-fallback-force-correction", type=str2bool, default=True)
    parser.add_argument("--act-aligned-min-dist-recover-ratio", type=float, default=0.75)
    parser.add_argument("--act-aligned-real-error-trigger-enable", type=str2bool, default=True)
    parser.add_argument("--act-aligned-real-error-min-dist-thresh", type=float, default=0.01)
    parser.add_argument("--act-aligned-real-error-min-dist-delta-thresh", type=float, default=0.005)
    parser.add_argument("--act-aligned-debug-recover-eval-rollout", type=str2bool, default=False)
    parser.add_argument("--act-aligned-debug-correction-evac-rollout", type=str2bool, default=False)
    parser.add_argument("--recover-eval-enable", type=str2bool, default=True)
    parser.add_argument("--recover-eval-save-video", type=str2bool, default=False)
    parser.add_argument("--recover-eval-gripper-open-thresh", type=float, default=0.8)
    parser.add_argument("--recover-eval-pos-thresh-m", type=float, default=0.03)
    parser.add_argument("--recover-eval-rot-thresh-deg", type=float, default=10.0)
    parser.add_argument("--recover-eval-nearest-window-radius", type=int, default=16)
    parser.add_argument("--recover-eval-video-bridge-steps", type=int, default=16)
    parser.add_argument("--act-aligned-correction-interp-nearest-enable", type=str2bool, default=False)
    parser.add_argument("--act-aligned-correction-interp-prefix-ratio", type=float, default=0.6)
    parser.add_argument("--act-aligned-correction-planner-prefix-ratio", type=float, default=0.5)
    parser.add_argument("--act-aligned-correction-gripper-close-prefix-ratio", type=float, default=0.32)
    parser.add_argument("--act-aligned-correction-compose-gt-tail-enable", type=str2bool, default=True)
    parser.add_argument("--act-aligned-correction-gripper-switch-ratio", type=float, default=0.5)
    parser.add_argument("--act-aligned-recover-gripper-penalty", type=float, default=0.0)
    parser.add_argument("--act-aligned-enable-perturb", type=str2bool, default=True)
    parser.add_argument("--act-aligned-perturb-prob", type=float, default=1.0)
    parser.add_argument("--act-aligned-perturb-error-mode", default="open_laptop_pregrasp")
    parser.add_argument("--act-aligned-perturb-open-laptop-pregrasp-close-prob", type=float, default=0.5)
    parser.add_argument("--act-aligned-perturb-open-laptop-pregrasp-translation-prob", type=float, default=0.0)
    parser.add_argument("--act-aligned-perturb-open-laptop-pregrasp-rotation-prob", type=float, default=0.0)
    parser.add_argument("--act-aligned-perturb-eef-fail-gain", type=float, default=0.10)
    parser.add_argument("--act-aligned-perturb-rot-max-deg", type=float, default=15.0)
    parser.add_argument("--act-aligned-perturb-mag-random", type=str2bool, default=False)
    parser.add_argument("--act-aligned-perturb-mag-rand-min", type=float, default=1.0)
    parser.add_argument("--act-aligned-perturb-mag-rand-max", type=float, default=1.4)
    parser.add_argument("--act-aligned-perturb-reject-sampling-enable", type=str2bool, default=True)
    parser.add_argument("--act-aligned-perturb-reject-max-trials", type=int, default=4)
    parser.add_argument("--act-aligned-perturb-reject-dir-jitter-eps", type=float, default=0.2)
    parser.add_argument("--act-aligned-perturb-gripper-close-min", type=float, default=0.10)
    parser.add_argument("--act-aligned-perturb-gripper-open-max", type=float, default=0.90)
    parser.add_argument("--act-aligned-perturb-gripper-fast-ratio", type=float, default=0.20)
    parser.add_argument("--act-aligned-sample-pregrasp-phase-window-len", type=int, default=30)
    parser.add_argument("--act-aligned-sample-timeout-sec", type=float, default=30.0)
    return parser


def _rank_suffix(rank: int) -> str:
    return f"_rank{int(rank):02d}"


def _write_live(failure_dir: Path, rank: int, trials: list[dict], args: argparse.Namespace) -> None:
    failure_dir.mkdir(parents=True, exist_ok=True)
    suffix = _rank_suffix(rank)
    (failure_dir / f"failure_trials_live{suffix}.json").write_text(
        json.dumps(trials, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta = {
        "version": 1,
        "mode": "pi05_explore_meta_live",
        "failure_phase_bins": int(args.failure_phase_bins),
        "failure_translation_dir_bins": int(args.failure_translation_dir_bins),
        "failure_translation_mag_bins": int(args.failure_translation_mag_bins),
        "failure_rotation_dir_bins": int(args.failure_rotation_dir_bins),
        "failure_rotation_mag_bins": int(args.failure_rotation_mag_bins),
        "failure_explore_k": int(args.failure_explore_k),
    }
    (failure_dir / f"failure_meta{suffix}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_argparser().parse_args()
    prepare_openpi_imports()
    distributed, rank, world_size, resolved_device = init_distributed_if_needed(args)
    args.device = resolved_device
    if args.teacher_device is None:
        args.teacher_device = resolved_device
    device = torch.device(args.device)
    set_seed(args.seed + rank)

    output_dir = Path(args.output_dir).expanduser().resolve()
    failure_dir = output_dir / "failure_explore"
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    task_specs = resolve_multitask_specs(
        task_names=list(args.multi_task_names),
        processed_dirs=list(args.processed_dirs),
        repo_ids=list(args.repo_ids),
        raw_data_dirs=list(args.raw_data_dirs),
        camera_mode=args.camera_mode,
    )

    model, _, _ = build_model(args, device)
    model.eval()
    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=(args.teacher_device or args.device))
    dataset, norm_stats_by_task = build_multitask_failure_table_dataset(
        task_specs=task_specs,
        failure_table_paths=[""] * len(task_specs),
        act_chunk_size=args.action_horizon,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        sample_phase_window_len=args.sample_phase_window_len,
        sample_skip_head_ratio=args.sample_skip_head_ratio,
        start_margin=0,
        failure_mode="explore",
        failure_phase_bins=args.failure_phase_bins,
        failure_translation_dir_bins=args.failure_translation_dir_bins,
        failure_translation_mag_bins=args.failure_translation_mag_bins,
        failure_rotation_dir_bins=args.failure_rotation_dir_bins,
        failure_rotation_mag_bins=args.failure_rotation_mag_bins,
        failure_explore_k=args.failure_explore_k,
    )
    local_k = int(max(1, (int(args.failure_explore_k) + world_size - 1) // world_size))
    dataset.set_explore_local_k(local_k)
    dataset.set_explore_unit_idx(0)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=args.action_horizon),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )
    pi0_norm_by_task = {
        spec.task_name: PI0NormAdapter(
            repo_id=spec.repo_id,
            train_config_name=args.train_config_name,
            assets_base_dir=args.assets_base_dir,
            model_action_dim=args.model_action_dim,
        ).to(device)
        for spec in task_specs
    }

    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    trials: list[dict] = []
    total_units = int(dataset.get_explore_num_units())
    debug_saved = 0
    global_step = 0
    start_time = time.time()

    try:
        for epoch in range(int(args.num_epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch + 1000)
            completed = int(dataset.get_explore_completed_unit_count())
            if completed >= total_units:
                break
            pbar = tqdm(loader, disable=not is_main_process(rank), desc=f"pi05 explore {epoch}")
            for batch in pbar:
                image_batch = select_head_image(batch["image_t"]).to(device, non_blocking=True)
                qpos_raw_batch = batch["qpos_raw"].to(device, non_blocking=True).float()
                task_names = list(batch["task_name"])
                raw_data_dirs = list(batch["raw_data_dir"])
                task_indices = batch["task_idx"].tolist()
                episode_ids = batch["episode_id"].tolist()
                start_ts_list = batch["start_ts"].tolist()

                for i in range(int(image_batch.shape[0])):
                    task_name = str(task_names[i])
                    ep_id = int(episode_ids[i])
                    start_ts = int(start_ts_list[i])
                    raw_dir = str(raw_data_dirs[i])
                    cache_key = (task_name, ep_id)
                    if cache_key not in raw_cache:
                        raw_cache[cache_key] = load_raw_episode(raw_dir, ep_id)
                    pi0_norm = pi0_norm_by_task[task_name]
                    norm_stats = norm_stats_by_task[task_name]
                    qpos_pi_norm = pi0_norm.normalize_state_raw(qpos_raw_batch[i])[0]
                    corr_policy = PI0CorrectionPolicyAdapter(model, pi0_norm, norm_stats, device)

                    debug_dir = None
                    if int(args.explore_debug_max_samples) > debug_saved:
                        debug_dir = str(output_dir / "explore_debug" / f"rank{rank:02d}" / f"sample_{debug_saved:04d}_{task_name}_ep{ep_id:04d}_t{start_ts:04d}")

                    corr = correction_builder.build(
                        latent_model=corr_policy,
                        image_t=image_batch[i],
                        qpos_t=qpos_pi_norm,
                        raw_data=raw_cache[cache_key],
                        norm_stats=norm_stats,
                        start_ts=start_ts,
                        failure_mode_override="explore",
                        sampled_phase_id=int(batch["sampled_phase_id"][i]),
                        pregrasp_seg_start=int(batch["pregrasp_seg_start"][i]),
                        pregrasp_seg_end=int(batch["pregrasp_seg_end"][i]),
                        sampled_phase_bin_id=int(batch["sampled_phase_bin_id"][i]),
                        sampled_phase_instance_id=int(batch["sampled_phase_instance_id"][i]),
                        forced_error_mode_id=int(batch["forced_error_mode_id"][i]),
                        sampled_active_arm_pattern_id=int(batch["sampled_active_arm_pattern_id"][i]),
                        forced_dir_bin_id=int(batch["forced_dir_bin_id"][i]),
                        forced_mag_bin_id=int(batch["forced_mag_bin_id"][i]),
                        debug_dir=debug_dir,
                    )
                    if debug_dir is not None:
                        debug_saved += 1
                    corr_meta = corr.get("corr_meta") if isinstance(corr, dict) else correction_builder.pop_last_skip_meta()
                    if not isinstance(corr_meta, dict):
                        continue
                    recover_eval_last = corr_meta.get("recover_eval_last", {}) or {}
                    recoverable_raw = recover_eval_last.get("recoverable", None)
                    if recoverable_raw is None:
                        continue

                    forced_error_mode_id = int(batch["forced_error_mode_id"][i])
                    if forced_error_mode_id < 0:
                        continue
                    error_mode = error_mode_id_to_key(forced_error_mode_id)
                    if error_mode not in {"translation", "rotation", "gripper_close"}:
                        continue
                    active_id = int(batch["sampled_active_arm_pattern_id"][i])
                    active_pattern = active_arm_pattern_id_to_key(active_id) if active_id >= 0 else "both"
                    phase_id = int(batch["sampled_phase_id"][i])
                    unit_idx = int(batch["sampled_explore_unit_idx"][i])
                    trial = {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "task_name": task_name,
                        "episode_id": ep_id,
                        "start_ts": start_ts,
                        "phase_key": phase_id_to_key(phase_id),
                        "phase_instance_idx": int(batch["sampled_phase_instance_id"][i]),
                        "phase_bin_id": int(batch["sampled_phase_bin_id"][i]),
                        "error_mode": error_mode,
                        "active_arm_pattern": active_pattern,
                        "dir_bin_id": int(batch["forced_dir_bin_id"][i]),
                        "mag_bin_id": int(batch["forced_mag_bin_id"][i]),
                        "sampled_explore_unit_idx": unit_idx,
                        "recoverable": bool(recoverable_raw),
                        "recover_eval_mode": recover_eval_last.get("mode"),
                        "recover_eval_metric_name": recover_eval_last.get("metric_name"),
                        "recover_eval_metric": recover_eval_last.get("metric"),
                        "recover_eval_threshold": recover_eval_last.get("threshold"),
                    }
                    trials.append(trial)
                    dataset.record_explore_trial(int(task_indices[i]), unit_idx, ep_id, start_ts)
                    global_step += 1

                completed = int(dataset.get_explore_completed_unit_count())
                _write_live(failure_dir, rank, trials, args)
                pbar.set_postfix_str(f"completed={completed}/{total_units} trials={len(trials)}")
                if completed >= total_units:
                    break
    finally:
        _write_live(failure_dir, rank, trials, args)
        if distributed:
            dist.barrier()
        if is_main_process(rank):
            output_paths = merge_failure_dir(
                failure_dir=str(failure_dir),
                out_dir=str(failure_dir),
                failure_fail_recover_rate_thresh=float(args.failure_fail_recover_rate_thresh),
            )
            summary = {
                "finished_at": datetime.datetime.now().isoformat(),
                "elapsed_sec": float(time.time() - start_time),
                "failure_dir": str(failure_dir),
                "tables": output_paths,
            }
            (output_dir / "explore_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
