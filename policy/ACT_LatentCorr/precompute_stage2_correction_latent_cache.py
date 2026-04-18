#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .stage2_failure_dataset import build_failure_table_dataset
from .stage2_latent_cache import build_stage2_latent_cache_relpath
from .train_stage1_latent import _build_act_args, _resolve_dataset_info
from .train_stage2_latent import _FrozenACTChunkPolicy, _load_base_act_from_ckpt, str2bool
from .utils_latent import load_raw_episode, resolve_raw_data_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Precompute stage2 correction latent cache")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--act_aligned_error_policy_ckpt", type=str, default="")
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--camera_names", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--prefix_steps", type=int, default=16)
    parser.add_argument("--act_chunk_size", type=int, default=None)
    parser.add_argument("--future_offset", type=int, default=None)
    parser.add_argument("--max_rollout_steps", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=27)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=1000)
    parser.add_argument("--max_cache_writes", type=int, default=-1)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--curobo_left_yml", type=str, default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml")
    parser.add_argument("--curobo_right_yml", type=str, default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml")
    parser.add_argument("--planner_target_mode", type=str, default="backward", choices=["forward", "backward"])
    parser.add_argument("--planner_target_lookahead_steps", type=int, default=6)
    parser.add_argument("--planner_orient_weight", type=float, default=0.0573)
    parser.add_argument("--planner_gripper_penalty", type=float, default=1.0)
    parser.add_argument("--planner_nearest_window_radius", type=int, default=12)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, default=0.01)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, default=0.05)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, default=16)
    parser.add_argument("--act_aligned_min_dist_fallback_force_correction", type=str2bool, default=True)
    parser.add_argument("--act_aligned_min_dist_recover_ratio", type=float, default=0.75)
    parser.add_argument("--act_aligned_real_error_trigger_enable", type=str2bool, default=True)
    parser.add_argument("--act_aligned_real_error_min_dist_thresh", type=float, default=0.01)
    parser.add_argument("--act_aligned_real_error_min_dist_delta_thresh", type=float, default=0.005)
    parser.add_argument("--act_aligned_debug_recover_eval_rollout", type=str2bool, default=False)
    parser.add_argument("--act_aligned_debug_correction_evac_rollout", type=str2bool, default=False)
    parser.add_argument("--recover_eval_enable", type=str2bool, default=False)
    parser.add_argument("--recover_eval_save_video", type=str2bool, default=False)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, default=0.8)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, default=0.03)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, default=10.0)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, default=16)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, default=16)
    parser.add_argument("--act_aligned_correction_interp_nearest_enable", type=str2bool, default=False)
    parser.add_argument("--act_aligned_correction_interp_prefix_ratio", type=float, default=0.6)
    parser.add_argument("--act_aligned_correction_planner_prefix_ratio", type=float, default=0.5)
    parser.add_argument("--act_aligned_correction_gripper_close_prefix_ratio", type=float, default=0.32)
    parser.add_argument("--act_aligned_correction_compose_gt_tail_enable", type=str2bool, default=True)
    parser.add_argument("--act_aligned_correction_gripper_switch_ratio", type=float, default=0.5)
    parser.add_argument("--act_aligned_recover_gripper_penalty", type=float, default=0.0)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, default=True)
    parser.add_argument("--act_aligned_perturb_prob", type=float, default=1.0)
    parser.add_argument("--act_aligned_perturb_error_mode", type=str, default="open_laptop_pregrasp")
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_close_prob", type=float, default=0.5)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_translation_prob", type=float, default=0.0)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_rotation_prob", type=float, default=0.0)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, default=0.10)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, default=15.0)
    parser.add_argument("--act_aligned_perturb_mag_random", type=str2bool, default=False)
    parser.add_argument("--act_aligned_perturb_mag_rand_min", type=float, default=1.0)
    parser.add_argument("--act_aligned_perturb_mag_rand_max", type=float, default=1.4)
    parser.add_argument("--act_aligned_perturb_reject_sampling_enable", type=str2bool, default=True)
    parser.add_argument("--act_aligned_perturb_reject_max_trials", type=int, default=4)
    parser.add_argument("--act_aligned_perturb_reject_dir_jitter_eps", type=float, default=0.2)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, default=0.10)
    parser.add_argument("--act_aligned_perturb_gripper_open_max", type=float, default=0.90)
    parser.add_argument("--act_aligned_perturb_gripper_fast_ratio", type=float, default=0.20)
    parser.add_argument("--act_aligned_sample_pregrasp_phase_window_len", type=int, default=30)
    parser.add_argument("--act_aligned_sample_timeout_sec", type=float, default=30.0)
    parser.add_argument("--failure_mode", type=str, default="train", choices=["train"])
    parser.add_argument("--failure_table_path", type=str, required=True)
    parser.add_argument("--failure_phase_bins", type=int, default=3)
    parser.add_argument("--failure_translation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_translation_mag_bins", type=int, default=3)
    parser.add_argument("--failure_rotation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_rotation_mag_bins", type=int, default=3)
    parser.add_argument("--failure_explore_k", type=int, default=4)
    parser.add_argument("--failure_sample_skip_head_ratio", type=float, default=0.6)
    return parser


def _build_model(args, camera_names: list[str], act_chunk_size: int, device: str):
    if not args.output_dir:
        args.output_dir = cache_dir
    act_args = _build_act_args(camera_names, argparse.Namespace(**{**vars(args), "act_chunk_size": act_chunk_size}))
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=LatentModelConfig(prefix_steps=args.prefix_steps),
        latent_loss_cfg=LatentLossConfig(),
        warmup_cfg=DynamicsWarmupConfig(),
    ).to(device)
    teacher_bootstrap = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=device,
    )
    return model, teacher_bootstrap


def main() -> None:
    args = build_argparser().parse_args()
    venv_bin_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = venv_bin_dir + os.pathsep + os.environ.get("PATH", "")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cache_dir = os.path.realpath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    dataset_dir, num_episodes, camera_names = _resolve_dataset_info(args.task_name)
    if args.dataset_dir is not None:
        dataset_dir = os.path.realpath(args.dataset_dir)
    if args.num_episodes is not None:
        num_episodes = int(args.num_episodes)
    if args.camera_names is not None:
        camera_names = list(args.camera_names)
    raw_data_dir = resolve_raw_data_dir(args.task_name, args.raw_data_dir)

    source_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    source_ckpt_args = dict(source_ckpt.get("args", {}))
    act_chunk_size = int(args.act_chunk_size or source_ckpt_args.get("act_chunk_size") or args.prefix_steps)
    future_offset = args.future_offset if args.future_offset is not None else args.prefix_steps

    corr_start_margin = int(max(0, args.max_rollout_steps)) * int(max(1, args.act_aligned_rollout_exec_steps))
    corr_dataset, norm_stats = build_failure_table_dataset(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        act_chunk_size=act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=future_offset,
        sample_phase_window_len=args.act_aligned_sample_pregrasp_phase_window_len,
        sample_skip_head_ratio=args.failure_sample_skip_head_ratio,
        start_margin=corr_start_margin,
        failure_mode="train",
        failure_table_path=args.failure_table_path,
        failure_phase_bins=args.failure_phase_bins,
        failure_translation_dir_bins=args.failure_translation_dir_bins,
        failure_translation_mag_bins=args.failure_translation_mag_bins,
        failure_rotation_dir_bins=args.failure_rotation_dir_bins,
        failure_rotation_mag_bins=args.failure_rotation_mag_bins,
        failure_explore_k=args.failure_explore_k,
    )
    if source_ckpt.get("norm_stats") is not None:
        norm_stats = source_ckpt["norm_stats"]
        if hasattr(corr_dataset, "norm_stats"):
            corr_dataset.norm_stats = norm_stats

    dataloader = DataLoader(
        corr_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        drop_last=False,
    )

    print(f"[stage2-cache] dataset_dir={dataset_dir}", flush=True)
    print(f"[stage2-cache] raw_data_dir={raw_data_dir}", flush=True)
    print(f"[stage2-cache] cache_dir={cache_dir}", flush=True)
    print(f"[stage2-cache] stage1_ckpt={args.stage1_ckpt}", flush=True)
    print(
        f"[stage2-cache] act_aligned_error_policy_ckpt={args.act_aligned_error_policy_ckpt or args.stage1_ckpt}",
        flush=True,
    )

    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
    )
    act_args = _build_act_args(camera_names, argparse.Namespace(**{**vars(args), "act_chunk_size": act_chunk_size}))
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=LatentModelConfig(prefix_steps=args.prefix_steps),
        latent_loss_cfg=LatentLossConfig(),
        warmup_cfg=DynamicsWarmupConfig(),
    ).to(args.device)
    first_batch = next(iter(dataloader))
    model.initialize_latent_heads(first_batch["image_t"][:1].to(args.device), teacher)
    missing, unexpected = model.load_state_dict(source_ckpt["model"], strict=False)
    print(
        f"[stage2-cache] loaded stage1 missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    error_policy_model = model
    if args.act_aligned_error_policy_ckpt:
        error_base_act = copy.deepcopy(model.base_act).to(args.device)
        missing, unexpected = _load_base_act_from_ckpt(
            error_base_act,
            args.act_aligned_error_policy_ckpt,
            args.device,
        )
        for param in error_base_act.parameters():
            param.requires_grad = False
        error_base_act.eval()
        error_policy_model = _FrozenACTChunkPolicy(error_base_act)
        print(
            f"[stage2-cache] loaded fixed error policy missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    act_aligned_cfg = build_act_aligned_cfg_from_args(args, max_action_len=act_chunk_size)
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=act_aligned_cfg,
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    raw_cache: dict[int, dict] = {}
    total_seen = 0
    total_generated = 0
    total_skipped_existing = 0
    total_skipped_invalid = 0
    total_written = 0
    total_duplicates = 0
    pbar = tqdm(total=max(1, int(args.num_batches)), desc="stage2-cache", leave=True)
    batch_iter = iter(dataloader)
    start_time = time.time()

    for batch_idx in range(int(args.num_batches)):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        image_t = batch["image_t"].to(args.device, non_blocking=True)
        qpos_t = batch["qpos_t"].to(args.device, non_blocking=True)
        episode_ids = batch["episode_id"].tolist()
        start_ts_list = batch["start_ts"].tolist()
        sampled_phase_ids = batch["sampled_phase_id"].tolist()
        pregrasp_seg_starts = batch["pregrasp_seg_start"].tolist()
        pregrasp_seg_ends = batch["pregrasp_seg_end"].tolist()
        sampled_phase_bin_ids = batch["sampled_phase_bin_id"].tolist()
        sampled_phase_instance_ids = batch["sampled_phase_instance_id"].tolist()
        forced_error_mode_ids = batch["forced_error_mode_id"].tolist()
        sampled_active_arm_pattern_ids = batch["sampled_active_arm_pattern_id"].tolist()
        forced_dir_bin_ids = batch["forced_dir_bin_id"].tolist()
        forced_mag_bin_ids = batch["forced_mag_bin_id"].tolist()
        sampled_mode_probs = batch["sampled_mode_prob"].tolist()
        sampled_entry_probs = batch["sampled_entry_prob_within_mode"].tolist()
        sampled_unit_probs = batch["sampled_unit_prob"].tolist()

        with torch.no_grad():
            correction_policy_pred_chunk_batch = error_policy_model.predict_act_chunk(qpos_t, image_t)

        pending_save_items = []
        for i in range(image_t.shape[0]):
            total_seen += 1
            ep_id = int(episode_ids[i])
            if ep_id not in raw_cache:
                raw_cache[ep_id] = load_raw_episode(raw_data_dir, ep_id)
            raw_data = raw_cache[ep_id]

            with torch.no_grad():
                corr = correction_builder.build(
                    latent_model=error_policy_model,
                    image_t=image_t[i],
                    qpos_t=qpos_t[i],
                    raw_data=raw_data,
                    norm_stats=norm_stats,
                    start_ts=int(start_ts_list[i]),
                    failure_mode_override="train",
                    sampled_phase_id=int(sampled_phase_ids[i]),
                    pregrasp_seg_start=int(pregrasp_seg_starts[i]),
                    pregrasp_seg_end=int(pregrasp_seg_ends[i]),
                    sampled_phase_bin_id=int(sampled_phase_bin_ids[i]),
                    sampled_phase_instance_id=int(sampled_phase_instance_ids[i]),
                    forced_error_mode_id=int(forced_error_mode_ids[i]),
                    sampled_active_arm_pattern_id=int(sampled_active_arm_pattern_ids[i]),
                    forced_dir_bin_id=int(forced_dir_bin_ids[i]),
                    forced_mag_bin_id=int(forced_mag_bin_ids[i]),
                    sampled_mode_prob=float(sampled_mode_probs[i]),
                    sampled_entry_prob_within_mode=float(sampled_entry_probs[i]),
                    sampled_unit_prob=float(sampled_unit_probs[i]),
                    precomputed_action_chunk_norm=correction_policy_pred_chunk_batch[i],
                )
            if corr is None or corr.get("error_action_prefix_raw") is None:
                total_skipped_invalid += 1
                continue

            error_action_prefix_raw = corr["error_action_prefix_raw"].to(args.device)
            cache_relpath = build_stage2_latent_cache_relpath(
                task_name=args.task_name,
                episode_id=ep_id,
                start_ts=int(start_ts_list[i]),
                prefix_steps=args.prefix_steps,
                ddim_steps=args.ddim_steps,
                error_action_prefix_raw=error_action_prefix_raw,
            )
            cache_path = os.path.join(cache_dir, cache_relpath)
            if os.path.exists(cache_path) and not args.overwrite:
                total_skipped_existing += 1
                continue

            pending_save_items.append(
                {
                    "cache_path": cache_path,
                    "image_t": image_t[i, 0].detach().cpu(),
                    "qpos_raw": batch["qpos_raw"][i].detach().cpu(),
                    "action_prefix_raw": error_action_prefix_raw.detach().cpu(),
                    "raw_data": raw_data,
                }
            )
            total_generated += 1

        if pending_save_items:
            latent_batch = teacher.rollout_latent_from_actions_batch(
                curr_image=torch.stack([x["image_t"] for x in pending_save_items], dim=0),
                curr_qpos_raw=torch.stack([x["qpos_raw"] for x in pending_save_items], dim=0),
                action_prefix_raw=torch.stack([x["action_prefix_raw"] for x in pending_save_items], dim=0),
                raw_data=[x["raw_data"] for x in pending_save_items],
                fk=correction_builder.fk,
                ddim_steps=args.ddim_steps,
            ).cpu()
            for idx, item in enumerate(pending_save_items):
                os.makedirs(os.path.dirname(item["cache_path"]), exist_ok=True)
                if os.path.exists(item["cache_path"]) and not args.overwrite:
                    total_duplicates += 1
                    continue
                torch.save(latent_batch[idx].half(), item["cache_path"])
                total_written += 1
                if args.max_cache_writes > 0 and total_written >= int(args.max_cache_writes):
                    break

        pbar.set_postfix(
            seen=total_seen,
            gen=total_generated,
            write=total_written,
            skip=total_skipped_existing,
            invalid=total_skipped_invalid,
        )
        pbar.update(1)
        if args.max_cache_writes > 0 and total_written >= int(args.max_cache_writes):
            break

    pbar.close()
    elapsed = time.time() - start_time
    print(
        "[stage2-cache] done "
        f"seen={total_seen} generated={total_generated} written={total_written} "
        f"skipped_existing={total_skipped_existing} skipped_invalid={total_skipped_invalid} "
        f"duplicate_race={total_duplicates} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(
        "[stage2-cache] note: for best hit rate, use the same "
        f"ACT_ALIGNED_ERROR_POLICY_CKPT during training: {args.act_aligned_error_policy_ckpt or args.stage1_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
