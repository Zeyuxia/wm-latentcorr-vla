#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import os
import pickle
import time
from dataclasses import asdict
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from policy.ACT.constants import SIM_TASK_CONFIGS

from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .stage2_failure_dataset import build_multitask_failure_table_dataset
from .utils_latent import load_raw_episode
from .utils_multitask_latent import build_multitask_stage1_dataset, resolve_multitask_specs
from .wandb_utils import finish_wandb, init_wandb_run, log_wandb, update_wandb_summary


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def is_distributed_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed_if_needed(args: argparse.Namespace) -> tuple[bool, int, int, str]:
    if not is_distributed_env():
        return False, 0, 1, args.device
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    timeout_min = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_MIN", "120"))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=timeout_min),
    )
    return True, rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log_rank(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def cycle_loader(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _as_float_or_none(x) -> float | None:
    try:
        val = float(x)
    except Exception:
        return None
    if val != val:
        return None
    return val


def _finite_tensor(x: torch.Tensor | None) -> bool:
    return x is not None and bool(torch.isfinite(x).all().item())


def _action_raw_from_norm(action_norm: torch.Tensor, norm_stats: dict[str, Any]) -> torch.Tensor:
    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=action_norm.dtype, device=action_norm.device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=action_norm.dtype, device=action_norm.device)
    action_raw = action_norm * action_std.view(1, -1) + action_mean.view(1, -1)
    action_raw = action_raw.clone()
    action_raw[..., 6] = action_raw[..., 6].clamp(0.0, 1.0)
    action_raw[..., 13] = action_raw[..., 13].clamp(0.0, 1.0)
    return action_raw


def _qpos_raw_from_norm(qpos_norm: torch.Tensor, norm_stats: dict[str, Any]) -> torch.Tensor:
    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=qpos_norm.dtype, device=qpos_norm.device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=qpos_norm.dtype, device=qpos_norm.device)
    qpos_raw = qpos_norm * qpos_std + qpos_mean
    qpos_raw = qpos_raw.clone()
    qpos_raw[..., 6] = qpos_raw[..., 6].clamp(0.0, 1.0)
    qpos_raw[..., 13] = qpos_raw[..., 13].clamp(0.0, 1.0)
    return qpos_raw


def _slice_prefix(chunk: torch.Tensor, is_pad: torch.Tensor, start: int, prefix_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros((prefix_steps, chunk.shape[-1]), dtype=chunk.dtype, device=chunk.device)
    out_pad = torch.ones((prefix_steps,), dtype=torch.bool, device=is_pad.device)
    if start < chunk.shape[0]:
        end = min(int(start) + int(prefix_steps), int(chunk.shape[0]))
        n = int(end - int(start))
        if n > 0:
            out[:n] = chunk[int(start):end]
            out_pad[:n] = is_pad[int(start):end]
    return out, out_pad


def _build_act_args(camera_names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    task_name = ",".join(list(args.multi_task_names))
    return {
        "lr": args.lr,
        "lr_backbone": 1e-5,
        "weight_decay": args.weight_decay,
        "backbone": args.backbone,
        "dilation": False,
        "position_embedding": "sine",
        "camera_names": camera_names,
        "enc_layers": 4,
        "dec_layers": 7,
        "dim_feedforward": 3200,
        "hidden_dim": args.hidden_dim,
        "dropout": 0.1,
        "nheads": 8,
        "pre_norm": False,
        "masks": False,
        "chunk_size": args.act_chunk_size,
        "state_dim": args.state_dim,
        "kl_weight": 10,
        "ckpt_dir": args.output_dir,
        "policy_class": "ACT",
        "task_name": task_name,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
    }


def _load_base_act_init(model: ACTLatentStage1, ckpt_path: str, rank: int) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    base_state = {k[len("base_act."):]: v for k, v in state.items() if k.startswith("base_act.")}
    if not base_state:
        base_state = state
    missing, unexpected = model.base_act.load_state_dict(base_state, strict=False)
    if is_main_process(rank):
        print(f"[stage1-unified] initialized base_act from {ckpt_path}", flush=True)
        print(f"[stage1-unified] base_act init missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ACT unified stage-1 with normal+failure batches")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--act_init_ckpt", type=str, default=None)
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--failure_table_paths", nargs="+", required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--curobo_left_yml", type=str, default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml")
    parser.add_argument("--curobo_right_yml", type=str, default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--normal_batch_size", type=int, default=4)
    parser.add_argument("--failure_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--base_act_lr_scale", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--ddim_steps", type=int, default=27)
    parser.add_argument("--prefix_steps", type=int, default=16)
    parser.add_argument("--act_chunk_size", type=int, default=50)
    parser.add_argument("--future_offset", type=int, default=16)
    parser.add_argument("--lambda_action", type=float, default=1.0)
    parser.add_argument("--lambda_action_conditioned", type=float, default=0.5)
    parser.add_argument("--schedule_action_conditioned", type=str2bool, default=True)
    parser.add_argument("--lambda_condition_token", type=float, default=1.0)
    parser.add_argument("--lambda_align", type=float, default=0.0)
    parser.add_argument("--beta_dynamics_max", type=float, default=1.0)
    parser.add_argument("--lambda_wm_action_current", type=float, default=0.0)
    parser.add_argument("--lambda_wm_action_future", type=float, default=0.0)
    parser.add_argument("--lambda_bridge_future", type=float, default=0.0)
    parser.add_argument("--freeze_base_act", type=str2bool, default=False)
    parser.add_argument("--freeze_readout_decoder", type=str2bool, default=True)
    parser.add_argument("--detach_act_feature_for_latent", type=str2bool, default=True)
    parser.add_argument("--use_raw_wm_targets", type=str2bool, default=False)
    parser.add_argument("--dyn_zero_steps", type=int, default=0)
    parser.add_argument("--dyn_ramp_steps", type=int, default=1000)
    parser.add_argument("--dyn_warmup_curve", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--reference_global_batch_size", type=int, default=6)
    parser.add_argument("--predictor_num_blocks", type=int, default=3)
    parser.add_argument("--projector_mid_channels", type=int, default=256)
    parser.add_argument("--wm_adapter_mid_channels", type=int, default=128)
    parser.add_argument("--readout_adapter_mid_channels", type=int, default=128)
    parser.add_argument("--predictor_mlp_hidden", type=int, default=512)
    parser.add_argument("--action_decoder_hidden", type=int, default=512)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--max_rollout_steps", type=int, default=1)
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
    parser.add_argument("--failure_phase_bins", type=int, default=3)
    parser.add_argument("--failure_translation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_translation_mag_bins", type=int, default=3)
    parser.add_argument("--failure_rotation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_rotation_mag_bins", type=int, default=3)
    parser.add_argument("--failure_explore_k", type=int, default=4)
    parser.add_argument("--failure_sample_skip_head_ratio", type=float, default=0.6)
    parser.add_argument(
        "--failure_future_latent_mode",
        type=str,
        default="rollout",
        choices=["vae", "rollout"],
        help="How to construct the future teacher latent for failure samples.",
    )
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="RoboTwin_ACT_LatentCorr")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    return parser


def _prepare_failure_samples(
    *,
    batch: dict[str, Any],
    raw_model: ACTLatentStage1,
    correction_builder: ACTAlignedCorrectionBuilder,
    teacher: EvacLatentTeacher,
    norm_stats: dict[str, Any],
    raw_cache: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, torch.Tensor]], int, float, float]:
    prepared: list[dict[str, torch.Tensor]] = []
    skipped = 0
    t_builder = 0.0
    t_rollout = 0.0
    batch_size = int(batch["image_t"].shape[0])
    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad():
        for i in range(batch_size):
            task_name = str(batch["task_name"][i])
            ep_id = int(batch["episode_id"][i])
            start_ts = int(batch["start_ts"][i])
            raw_dir = str(batch["raw_data_dir"][i])
            cache_key = (task_name, ep_id)
            if cache_key not in raw_cache:
                raw_cache[cache_key] = load_raw_episode(raw_dir, ep_id)
            t0 = time.perf_counter()
            corr = correction_builder.build(
                raw_model,
                image_t=batch["image_t"][i].to(args.device, non_blocking=True),
                qpos_t=batch["qpos_t"][i].to(args.device, non_blocking=True),
                raw_data=raw_cache[cache_key],
                norm_stats=norm_stats,
                start_ts=start_ts,
                failure_mode_override="train",
                sampled_phase_id=int(batch["sampled_phase_id"][i]),
                pregrasp_seg_start=int(batch["pregrasp_seg_start"][i]),
                pregrasp_seg_end=int(batch["pregrasp_seg_end"][i]),
                sampled_phase_bin_id=int(batch["sampled_phase_bin_id"][i]),
                sampled_phase_instance_id=int(batch["sampled_phase_instance_id"][i]),
                forced_error_mode_id=int(batch["forced_error_mode_id"][i]),
                sampled_active_arm_pattern_id=int(batch["sampled_active_arm_pattern_id"][i]),
                forced_dir_bin_id=int(batch["forced_dir_bin_id"][i]),
                forced_mag_bin_id=int(batch["forced_mag_bin_id"][i]),
                sampled_mode_prob=_as_float_or_none(batch["sampled_mode_prob"][i]),
                sampled_entry_prob_within_mode=_as_float_or_none(batch["sampled_entry_prob_within_mode"][i]),
                sampled_unit_prob=_as_float_or_none(batch["sampled_unit_prob"][i]),
            )
            t_builder += time.perf_counter() - t0
            if corr is None:
                skipped += 1
                continue
            corr_image = corr.get("corr_image")
            corr_qpos_norm = corr.get("corr_qpos_norm")
            corr_action = corr.get("corr_action_chunk_norm")
            corr_is_pad = corr.get("corr_is_pad")
            if not (_finite_tensor(corr_image) and _finite_tensor(corr_qpos_norm) and _finite_tensor(corr_action)):
                skipped += 1
                continue
            if corr_is_pad is None or corr_action.ndim != 2 or corr_action.shape[0] < args.prefix_steps:
                skipped += 1
                continue
            corr_image = corr_image.detach().to(args.device)
            corr_qpos_norm = corr_qpos_norm.detach().to(args.device)
            corr_action = corr_action.detach().to(args.device)
            corr_is_pad = corr_is_pad.detach().to(args.device).bool()
            action_prefix = corr_action[: args.prefix_steps].clone()
            is_pad_prefix = corr_is_pad[: args.prefix_steps].clone()
            action_future_prefix, is_pad_future_prefix = _slice_prefix(
                corr_action,
                corr_is_pad,
                args.future_offset,
                args.prefix_steps,
            )
            t1 = time.perf_counter()
            if str(args.failure_future_latent_mode).strip().lower() == "rollout":
                corr_qpos_raw = _qpos_raw_from_norm(corr_qpos_norm, norm_stats)
                corr_action_prefix_raw = _action_raw_from_norm(action_prefix, norm_stats)
                future_latent = teacher.rollout_latent_from_actions(
                    curr_image=corr_image[0],
                    curr_qpos_raw=corr_qpos_raw,
                    action_prefix_raw=corr_action_prefix_raw,
                    raw_data=raw_cache[cache_key],
                    fk=correction_builder.fk,
                    ddim_steps=int(args.ddim_steps),
                )[0]
            else:
                future_latent = teacher.encode_image(corr_image)[0]
            t_rollout += time.perf_counter() - t1
            if not _finite_tensor(future_latent):
                skipped += 1
                continue
            prepared.append(
                {
                    "image_t": corr_image,
                    "image_t1": corr_image,
                    "qpos_t": corr_qpos_norm,
                    "act_action_chunk": corr_action,
                    "act_is_pad": corr_is_pad,
                    "action_prefix": action_prefix,
                    "action_future_prefix": action_future_prefix,
                    "is_pad_prefix": is_pad_prefix,
                    "is_pad_future_prefix": is_pad_future_prefix,
                    "future_teacher_latent": future_latent.detach(),
                }
            )
    if was_training:
        raw_model.train()
    return prepared, skipped, t_builder, t_rollout


def main() -> None:
    args = build_argparser().parse_args()
    distributed, rank, world_size, runtime_device = init_distributed_if_needed(args)
    args.device = runtime_device
    per_rank_batch = int(args.normal_batch_size) + int(args.failure_batch_size)
    current_global_batch_size = per_rank_batch * (world_size if distributed else 1)
    if args.act_chunk_size < args.prefix_steps:
        raise ValueError("act_chunk_size must be >= prefix_steps")
    if args.future_offset != args.prefix_steps and is_main_process(rank):
        print(
            f"[stage1-unified] warning: future_offset={args.future_offset}, prefix_steps={args.prefix_steps}; "
            "new algorithm normally expects both to be 16.",
            flush=True,
        )
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS)
    shared_camera_names = list(task_specs[0].camera_names)
    for spec in task_specs[1:]:
        if list(spec.camera_names) != shared_camera_names:
            raise ValueError("All multitask camera_names must match")

    if is_main_process(rank):
        print("[stage1-unified] tasks=", flush=True)
        for spec in task_specs:
            print(f"  - {spec.task_name} dataset_dir={spec.dataset_dir} raw_data_dir={spec.raw_data_dir}", flush=True)
        print(
            f"[stage1-unified] per_rank normal+failure={args.normal_batch_size}+{args.failure_batch_size} "
            f"global_batch={current_global_batch_size}",
            flush=True,
        )

    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=args.projector_mid_channels,
        wm_adapter_mid_channels=args.wm_adapter_mid_channels,
        readout_adapter_mid_channels=args.readout_adapter_mid_channels,
        predictor_num_blocks=args.predictor_num_blocks,
        predictor_mlp_hidden=args.predictor_mlp_hidden,
        action_decoder_hidden=args.action_decoder_hidden,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        prefix_steps=args.prefix_steps,
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=args.lambda_action,
        lambda_action_conditioned=args.lambda_action_conditioned,
        schedule_action_conditioned=args.schedule_action_conditioned,
        lambda_condition_token=args.lambda_condition_token,
        lambda_align=args.lambda_align,
        beta_dynamics_max=args.beta_dynamics_max,
        lambda_wm_action_current=args.lambda_wm_action_current,
        lambda_wm_action_future=args.lambda_wm_action_future,
        lambda_bridge_future=args.lambda_bridge_future,
        use_projector_detach_for_predictor=True,
        use_projector_detach_for_action_decoder=True,
        detach_act_feature_for_latent=args.detach_act_feature_for_latent,
        use_raw_wm_targets=args.use_raw_wm_targets,
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=args.dyn_zero_steps,
        ramp_steps=args.dyn_ramp_steps,
        max_weight=1.0,
        curve=args.dyn_warmup_curve,
    )

    normal_dataset, norm_stats = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=args.act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
    )
    corr_start_margin = int(max(0, args.max_rollout_steps)) * int(max(1, args.act_aligned_rollout_exec_steps))
    failure_dataset, failure_norm_stats = build_multitask_failure_table_dataset(
        task_specs=task_specs,
        act_chunk_size=args.act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        sample_phase_window_len=args.act_aligned_sample_pregrasp_phase_window_len,
        sample_skip_head_ratio=args.failure_sample_skip_head_ratio,
        start_margin=corr_start_margin,
        failure_mode="train",
        failure_table_paths=list(args.failure_table_paths),
        failure_phase_bins=args.failure_phase_bins,
        failure_translation_dir_bins=args.failure_translation_dir_bins,
        failure_translation_mag_bins=args.failure_translation_mag_bins,
        failure_rotation_dir_bins=args.failure_rotation_dir_bins,
        failure_rotation_mag_bins=args.failure_rotation_mag_bins,
        failure_explore_k=args.failure_explore_k,
    )
    # Use one multitask normalization space for both normal and failure batches.
    failure_dataset.set_norm_stats(norm_stats)
    _ = failure_norm_stats

    normal_sampler = None
    failure_sampler = None
    if distributed:
        normal_sampler = DistributedSampler(normal_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
        failure_sampler = DistributedSampler(failure_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed + 17)
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=max(1, args.normal_batch_size),
        sampler=normal_sampler,
        shuffle=normal_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    failure_loader = DataLoader(
        failure_dataset,
        batch_size=max(1, args.failure_batch_size),
        sampler=failure_sampler,
        shuffle=failure_sampler is None,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=args.device)
    model = ACTLatentStage1(
        act_args=_build_act_args(shared_camera_names, args),
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(args.device)
    init_batch = next(iter(normal_loader))
    model.initialize_latent_heads(init_batch["image_t"][:1].to(args.device), teacher)
    if args.act_init_ckpt and not args.resume_ckpt:
        _load_base_act_init(model, args.act_init_ckpt, rank)
    if args.freeze_base_act:
        model.set_base_act_frozen(True)
    if args.freeze_readout_decoder:
        if model.readout_adapter is not None:
            model.readout_adapter.requires_grad_(False)
        if model.action_decoder is not None:
            model.action_decoder.requires_grad_(False)

    optimizer_groups = []
    base_act_params = []
    latent_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("base_act."):
            base_act_params.append(param)
        else:
            latent_params.append(param)
    if base_act_params:
        optimizer_groups.append({"params": base_act_params, "lr": args.lr * args.base_act_lr_scale})
    if latent_params:
        optimizer_groups.append({"params": latent_params, "lr": args.lr})
    optimizer = AdamW(optimizer_groups, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    sample_count = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                if is_main_process(rank):
                    print(f"[stage1-unified] optimizer state not loaded: {exc}", flush=True)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        sample_count = int(ckpt.get("sample_count", global_step * current_global_batch_size))
        if is_main_process(rank):
            print(f"[stage1-unified] resumed from {args.resume_ckpt}", flush=True)
            print(f"[stage1-unified] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    correction_cfg = build_act_aligned_cfg_from_args(args, max_action_len=args.act_chunk_size)
    correction_cfg.failure_mode = "train"
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=correction_cfg,
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[int(args.device.split(":")[-1])],
            output_device=int(args.device.split(":")[-1]),
            find_unused_parameters=True,
        )
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}

    if is_main_process(rank):
        with open(os.path.join(args.output_dir, "stage1_unified_failure_config.txt"), "w", encoding="utf-8") as f:
            f.write(str(vars(args)))
            f.write("\n")
            f.write(str(asdict(latent_model_cfg)))
            f.write("\n")
            f.write(str(asdict(latent_loss_cfg)))
            f.write("\n")
            f.write(str(asdict(warmup_cfg)))
            f.write("\n")
            for spec in task_specs:
                f.write(f"{spec}\n")
        with open(os.path.join(args.output_dir, "dataset_stats.pkl"), "wb") as f:
            pickle.dump(norm_stats, f)

    wandb_run = init_wandb_run(
        enabled=is_main_process(rank) and args.use_wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or f"stage1_unified_{os.path.basename(args.output_dir)}",
        group=args.wandb_group or "robotwin_multitask_stage1_unified",
        tags=args.wandb_tags or ["stage1", "unified", "failure_table", "base_inference"],
        mode=args.wandb_mode,
        output_dir=args.output_dir,
        config={"stage": "stage1_unified_failure_multitask", "args": vars(args)},
    )

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if normal_sampler is not None:
                normal_sampler.set_epoch(epoch)
            if failure_sampler is not None:
                failure_sampler.set_epoch(epoch)
            model.train()
            raw_model = model.module if isinstance(model, DDP) else model
            if args.freeze_base_act:
                raw_model.base_act.eval()
            normal_iter = cycle_loader(normal_loader)
            failure_iter = cycle_loader(failure_loader)
            steps_per_epoch = len(normal_loader)
            start_time = time.time()
            pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", disable=not is_main_process(rank), leave=True)
            meter = {k: 0.0 for k in ["loss", "action", "action_cond", "cond_token", "align", "dyn", "wm_curr", "wm_future", "bridge", "lambda_cond", "beta", "failure_valid", "failure_skip", "t_builder", "t_rollout"]}

            for _ in pbar:
                normal_batch = next(normal_iter)
                with torch.no_grad():
                    normal_future_latent = teacher.encode_image(normal_batch["image_t_future"][:, 0].to(args.device, non_blocking=True)).detach()
                normal_items = {
                    "image_t": normal_batch["image_t"].to(args.device, non_blocking=True),
                    "image_t1": normal_batch["image_t_future"].to(args.device, non_blocking=True),
                    "qpos_t": normal_batch["qpos_t"].to(args.device, non_blocking=True),
                    "act_action_chunk": normal_batch["act_action_chunk"].to(args.device, non_blocking=True),
                    "act_is_pad": normal_batch["act_is_pad"].to(args.device, non_blocking=True),
                    "action_prefix": normal_batch["action_prefix"].to(args.device, non_blocking=True),
                    "action_future_prefix": normal_batch["action_future_prefix"].to(args.device, non_blocking=True),
                    "is_pad_prefix": normal_batch["is_pad_prefix"].to(args.device, non_blocking=True),
                    "is_pad_future_prefix": normal_batch["is_pad_future_prefix"].to(args.device, non_blocking=True),
                    "future_teacher_latent": normal_future_latent,
                }

                failure_prepared: list[dict[str, torch.Tensor]] = []
                failure_skipped = 0
                t_builder = 0.0
                t_rollout = 0.0
                if args.failure_batch_size > 0:
                    failure_batch = next(failure_iter)
                    failure_prepared, failure_skipped, t_builder, t_rollout = _prepare_failure_samples(
                        batch=failure_batch,
                        raw_model=raw_model,
                        correction_builder=correction_builder,
                        teacher=teacher,
                        norm_stats=norm_stats,
                        raw_cache=raw_cache,
                        args=args,
                    )
                    raw_model.train()
                    if args.freeze_base_act:
                        raw_model.base_act.eval()

                if failure_prepared:
                    failure_items = {
                        key: torch.stack([item[key] for item in failure_prepared], dim=0)
                        for key in normal_items.keys()
                    }
                    train_items = {
                        key: torch.cat([normal_items[key], failure_items[key].to(args.device)], dim=0)
                        for key in normal_items.keys()
                    }
                else:
                    train_items = normal_items

                schedule_step = sample_count / float(max(1, args.reference_global_batch_size))
                out = model(
                    image_t=train_items["image_t"],
                    image_t1=train_items["image_t1"],
                    qpos_t=train_items["qpos_t"],
                    qpos_future_norm=None,
                    act_action_chunk=train_items["act_action_chunk"],
                    act_is_pad=train_items["act_is_pad"],
                    action_prefix=train_items["action_prefix"],
                    action_future_prefix=train_items["action_future_prefix"],
                    is_pad_prefix=train_items["is_pad_prefix"],
                    is_pad_future_prefix=train_items["is_pad_future_prefix"],
                    wm_teacher=teacher,
                    global_step=schedule_step,
                    use_act_head_conditioning=True,
                    future_teacher_latent=train_items["future_teacher_latent"],
                    mode="stage1",
                )
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()

                global_step += 1
                sample_count += current_global_batch_size
                meter["loss"] += float(out.loss.item())
                meter["action"] += float(out.loss_action.item())
                meter["action_cond"] += float(out.loss_action_conditioned.item())
                meter["cond_token"] += float(out.loss_condition_token.item())
                meter["align"] += float(out.loss_align.item())
                meter["dyn"] += float(out.loss_dynamics.item())
                meter["wm_curr"] += float(out.loss_wm_action_current.item())
                meter["wm_future"] += float(out.loss_wm_action_future.item())
                meter["bridge"] += float(out.loss_bridge_future.item())
                meter["lambda_cond"] += float(out.lambda_action_conditioned_eff)
                meter["beta"] += float(out.beta_dynamics)
                meter["failure_valid"] += float(len(failure_prepared))
                meter["failure_skip"] += float(failure_skipped)
                meter["t_builder"] += float(t_builder)
                meter["t_rollout"] += float(t_rollout)

                if is_main_process(rank):
                    pbar.set_postfix(
                        loss=f"{out.loss.item():.4f}",
                        act=f"{out.loss_action.item():.4f}",
                        cond=f"{out.loss_action_conditioned.item():.4f}",
                        ctoken=f"{out.loss_condition_token.item():.4f}",
                        dyn=f"{out.loss_dynamics.item():.4f}",
                        beta=f"{out.beta_dynamics:.3f}",
                        lcond=f"{out.lambda_action_conditioned_eff:.3f}",
                        fvalid=len(failure_prepared),
                    )
                log_wandb(
                    wandb_run,
                    {
                        "global_step": global_step,
                        "schedule_step": schedule_step,
                        "sample_count": sample_count,
                        "train_loss_step": out.loss.item(),
                        "train_action_step": out.loss_action.item(),
                        "train_action_conditioned_step": out.loss_action_conditioned.item(),
                        "train_condition_token_step": out.loss_condition_token.item(),
                        "train_align_step": out.loss_align.item(),
                        "train_dynamics_step": out.loss_dynamics.item(),
                        "beta_dynamics": out.beta_dynamics,
                        "lambda_action_conditioned_eff": out.lambda_action_conditioned_eff,
                        "failure_valid_step": len(failure_prepared),
                        "failure_skipped_step": failure_skipped,
                    },
                    step=global_step,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            n = max(1, steps_per_epoch)
            if distributed:
                keys = list(meter.keys())
                stats = torch.tensor([meter[k] for k in keys] + [float(n)], dtype=torch.float64, device=args.device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                for idx, key in enumerate(keys):
                    meter[key] = float(stats[idx].item())
                n = max(1, int(stats[-1].item()))
            epoch_time = time.time() - start_time
            if is_main_process(rank):
                print(
                    f"[epoch {epoch}] loss={meter['loss']/n:.4f} action={meter['action']/n:.4f} "
                    f"cond={meter['action_cond']/n:.4f} ctoken={meter['cond_token']/n:.4f} "
                    f"align={meter['align']/n:.4f} dyn={meter['dyn']/n:.4f} "
                    f"beta={meter['beta']/n:.3f} lambda_cond={meter['lambda_cond']/n:.3f} "
                    f"failure_valid={meter['failure_valid']/n:.2f} failure_skip={meter['failure_skip']/n:.2f} "
                    f"time={epoch_time:.1f}s builder={meter['t_builder']/n:.2f}s rollout={meter['t_rollout']/n:.2f}s",
                    flush=True,
                )
            log_wandb(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "epoch_loss": meter["loss"] / n,
                    "epoch_action": meter["action"] / n,
                    "epoch_action_conditioned": meter["action_cond"] / n,
                    "epoch_condition_token": meter["cond_token"] / n,
                    "epoch_align": meter["align"] / n,
                    "epoch_dynamics": meter["dyn"] / n,
                    "epoch_beta_dynamics": meter["beta"] / n,
                    "epoch_lambda_action_conditioned_eff": meter["lambda_cond"] / n,
                    "epoch_failure_valid": meter["failure_valid"] / n,
                    "epoch_failure_skipped": meter["failure_skip"] / n,
                    "epoch_time_sec": epoch_time,
                    "global_step": global_step,
                    "sample_count": sample_count,
                },
                step=global_step,
            )

            if is_main_process(rank) and (epoch + 1) % args.save_freq == 0:
                raw_model_save = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "sample_count": sample_count,
                    "schedule_step": sample_count / float(max(1, args.reference_global_batch_size)),
                    "global_batch_size": current_global_batch_size,
                    "world_size": world_size,
                    "model": raw_model_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "norm_stats": norm_stats,
                    "args": vars(args),
                    "multi_task_names": list(args.multi_task_names),
                    "algorithm": "stage1_unified_normal_failure_base_inference",
                }
                save_path = os.path.join(args.output_dir, f"stage1_unified_epoch_{epoch+1:04d}.pt")
                torch.save(ckpt, save_path)
                print(f"[stage1-unified] saved: {save_path}", flush=True)
                update_wandb_summary(wandb_run, {"last_checkpoint": save_path, "last_epoch": epoch + 1})
            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main_process(rank):
                    print(f"[stage1-unified] reached max_steps={args.max_steps}, stopping", flush=True)
                break
    finally:
        finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
