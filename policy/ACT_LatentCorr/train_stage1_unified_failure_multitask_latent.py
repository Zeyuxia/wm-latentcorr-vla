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
from .latent_policy import ACTLatentStage1, Stage1LossOutput
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


def _parse_data_parallel_device_ids(spec: str) -> list[int]:
    items = [x.strip() for x in str(spec).split(",") if x.strip()]
    if not items:
        raise ValueError("data_parallel_device_ids must be a non-empty comma separated list")
    return [int(x) for x in items]


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log_rank(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def sync_cuda_device(device: str) -> None:
    if not torch.cuda.is_available():
        return
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    if dev.index is not None:
        torch.cuda.synchronize(dev.index)
    else:
        torch.cuda.synchronize()


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
    parser.add_argument("--normal_condition_keep_prob", type=float, default=1.0)
    parser.add_argument("--beta_dynamics_max", type=float, default=1.0)
    parser.add_argument("--lambda_wm_action_current", type=float, default=0.0)
    parser.add_argument("--lambda_wm_action_future", type=float, default=0.0)
    parser.add_argument("--lambda_bridge_future", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_max", type=float, default=0.5)
    parser.add_argument("--lambda_pred_max", type=float, default=0.5)
    parser.add_argument("--lambda_latent_max", type=float, default=0.3)
    parser.add_argument("--lambda_token_init", type=float, default=0.1)
    parser.add_argument("--lambda_token_late", type=float, default=0.02)
    parser.add_argument("--latent_loss_type", type=str, default="normalized_mse", choices=["normalized_mse", "cosine", "raw_mse"])
    parser.add_argument("--token_loss_type", type=str, default="mse", choices=["mse"])
    parser.add_argument("--teacher_decay_start_ratio", type=float, default=0.20)
    parser.add_argument("--teacher_decay_end_ratio", type=float, default=0.90)
    parser.add_argument("--pred_warmup_start_ratio", type=float, default=0.10)
    parser.add_argument("--pred_warmup_end_ratio", type=float, default=0.70)
    parser.add_argument("--latent_warmup_end_ratio", type=float, default=0.20)
    parser.add_argument("--token_decay_start_ratio", type=float, default=0.20)
    parser.add_argument("--token_decay_end_ratio", type=float, default=0.90)
    parser.add_argument("--pred_only_finetune_start_ratio", type=float, default=0.90)
    parser.add_argument("--stopgrad_wm_teacher", type=str2bool, default=True)
    parser.add_argument("--stopgrad_teacher_token", type=str2bool, default=True)
    parser.add_argument("--stopgrad_adapter_input_for_token_loss", type=str2bool, default=False)
    parser.add_argument("--freeze_base_act", type=str2bool, default=False)
    parser.add_argument("--freeze_readout_decoder", type=str2bool, default=True)
    parser.add_argument("--detach_act_feature_for_latent", type=str2bool, default=False)
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
    parser.add_argument("--token_adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--token_adapter_num_layers", type=int, default=2)
    parser.add_argument("--token_adapter_dropout", type=float, default=0.1)
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
    parser.add_argument("--balance_failure_tasks", type=str2bool, default=False)
    parser.add_argument("--rebalance_failure_groups", type=str2bool, default=False)
    parser.add_argument(
        "--failure_prepare_owner_rank",
        type=int,
        default=-1,
        help="If >=0, only this rank prepares failure correction samples; other ranks skip failure preparation.",
    )
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="RoboTwin_ACT_LatentCorr")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--use_data_parallel", type=str2bool, default=False)
    parser.add_argument("--data_parallel_device_ids", type=str, default="")
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False)
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


def _broadcast_failure_payload(
    *,
    distributed: bool,
    src_rank: int,
    rank: int,
    prepared: list[dict[str, torch.Tensor]],
    prepared_task_names: list[str],
    skipped: int,
    t_builder: float,
    t_rollout: float,
    device: str,
    reference_items: dict[str, torch.Tensor],
    task_name_vocab: list[str],
) -> tuple[list[dict[str, torch.Tensor]], list[str], int, float, float]:
    if not distributed:
        return prepared, prepared_task_names, skipped, t_builder, t_rollout
    tensor_keys = [
        "image_t",
        "image_t1",
        "qpos_t",
        "act_action_chunk",
        "act_is_pad",
        "action_prefix",
        "action_future_prefix",
        "is_pad_prefix",
        "is_pad_future_prefix",
        "future_teacher_latent",
    ]
    bool_keys = {
        "act_is_pad",
        "is_pad_prefix",
        "is_pad_future_prefix",
    }
    task_to_id = {str(name): idx for idx, name in enumerate(task_name_vocab)}
    meta = torch.empty(2, dtype=torch.int64, device=device)
    time_meta = torch.empty(2, dtype=torch.float64, device=device)
    if rank == src_rank:
        meta[0] = int(len(prepared))
        meta[1] = int(skipped)
        time_meta[0] = float(t_builder)
        time_meta[1] = float(t_rollout)
    dist.broadcast(meta, src=src_rank)
    dist.broadcast(time_meta, src=src_rank)
    recv_count = int(meta[0].item())
    recv_skipped = int(meta[1].item())
    recv_t_builder = float(time_meta[0].item())
    recv_t_rollout = float(time_meta[1].item())

    task_ids = torch.empty(max(1, recv_count), dtype=torch.int64, device=device)
    if rank == src_rank:
        local_ids = [task_to_id[str(name)] for name in prepared_task_names]
        if recv_count > 0:
            task_ids[:recv_count] = torch.as_tensor(local_ids, dtype=torch.int64, device=device)
        else:
            task_ids[0] = -1
    dist.broadcast(task_ids, src=src_rank)
    recv_task_names = [
        str(task_name_vocab[int(task_ids[i].item())])
        for i in range(recv_count)
    ]

    stacked_tensors: dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        ref = reference_items[key]
        if rank == src_rank:
            value = torch.stack([item[key] for item in prepared], dim=0) if recv_count > 0 else ref[:0].clone()
            if key in bool_keys:
                value = value.to(dtype=torch.uint8)
            else:
                value = value.to(device=device, non_blocking=True)
        else:
            shape = (recv_count, *tuple(ref.shape[1:]))
            dtype = torch.uint8 if key in bool_keys else ref.dtype
            value = torch.empty(shape, dtype=dtype, device=device)
        if recv_count > 0:
            dist.broadcast(value, src=src_rank)
        stacked_tensors[key] = value

    recv_prepared: list[dict[str, torch.Tensor]] = []
    for idx in range(recv_count):
        item: dict[str, torch.Tensor] = {}
        for key in tensor_keys:
            value = stacked_tensors[key][idx]
            if key in bool_keys:
                value = value.to(dtype=torch.bool)
            item[key] = value
        recv_prepared.append(item)
    return recv_prepared, recv_task_names, recv_skipped, recv_t_builder, recv_t_rollout


def _build_correction_builder_serialized(
    *,
    args: argparse.Namespace,
    teacher: EvacLatentTeacher,
    distributed: bool,
    rank: int,
    world_size: int,
) -> ACTAlignedCorrectionBuilder | None:
    correction_cfg = build_act_aligned_cfg_from_args(args, max_action_len=args.act_chunk_size)
    correction_cfg.failure_mode = "train"

    owner_rank = int(args.failure_prepare_owner_rank)
    builder: ACTAlignedCorrectionBuilder | None = None
    if distributed and owner_rank >= 0:
        if rank == owner_rank:
            log_rank(rank, f"initializing correction builder on {args.device}")
            builder = ACTAlignedCorrectionBuilder(
                cfg=correction_cfg,
                urdf_path=args.urdf_path,
                curobo_left_yml=args.curobo_left_yml,
                curobo_right_yml=args.curobo_right_yml,
                device=args.device,
                shared_evac_model=teacher.model,
                shared_evac_config=teacher.cfg,
            )
            sync_cuda_device(args.device)
            log_rank(rank, "correction builder ready")
        else:
            log_rank(rank, f"skip correction builder on rank={rank}; owner_rank={owner_rank}")
        dist.barrier()
    else:
        log_rank(rank, f"initializing correction builder on {args.device}")
        builder = ACTAlignedCorrectionBuilder(
            cfg=correction_cfg,
            urdf_path=args.urdf_path,
            curobo_left_yml=args.curobo_left_yml,
            curobo_right_yml=args.curobo_right_yml,
            device=args.device,
            shared_evac_model=teacher.model,
            shared_evac_config=teacher.cfg,
        )
        sync_cuda_device(args.device)
        log_rank(rank, "correction builder ready")
        if distributed:
            dist.barrier()

    if builder is None and not (distributed and owner_rank >= 0 and rank != owner_rank):
        raise RuntimeError(f"rank {rank} failed to initialize correction builder")
    return builder


def main() -> None:
    args = build_argparser().parse_args()
    print("[bootstrap] args parsed", flush=True)
    use_data_parallel = bool(args.use_data_parallel)
    distributed, rank, world_size, runtime_device = init_distributed_if_needed(args)
    print("[bootstrap] distributed init returned", flush=True)
    if use_data_parallel and distributed:
        raise ValueError("use_data_parallel cannot be combined with torchrun distributed mode")
    args.device = runtime_device
    log_rank(rank, f"startup distributed={distributed} world_size={world_size} device={args.device}")
    per_rank_batch = int(args.normal_batch_size) + int(args.failure_batch_size)
    dp_device_ids = _parse_data_parallel_device_ids(args.data_parallel_device_ids) if use_data_parallel else []
    effective_world_size = len(dp_device_ids) if use_data_parallel else (world_size if distributed else 1)
    current_global_batch_size = per_rank_batch * effective_world_size
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
    log_rank(rank, "post output-dir barrier")
    torch.manual_seed(args.seed)
    log_rank(rank, "torch manual_seed complete")
    log_rank(rank, "skip cuda seed setup")

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS)
    task_name_vocab = [str(x) for x in args.multi_task_names]
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
        token_adapter_hidden_dim=args.token_adapter_hidden_dim,
        token_adapter_num_layers=args.token_adapter_num_layers,
        token_adapter_dropout=args.token_adapter_dropout,
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=args.lambda_action,
        normal_condition_keep_prob=args.normal_condition_keep_prob,
        beta_dynamics_max=args.beta_dynamics_max,
        lambda_wm_action_current=args.lambda_wm_action_current,
        lambda_wm_action_future=args.lambda_wm_action_future,
        lambda_bridge_future=args.lambda_bridge_future,
        use_projector_detach_for_predictor=False,
        use_projector_detach_for_action_decoder=True,
        detach_act_feature_for_latent=args.detach_act_feature_for_latent,
        use_raw_wm_targets=args.use_raw_wm_targets,
        lambda_teacher_max=args.lambda_teacher_max,
        lambda_pred_max=args.lambda_pred_max,
        lambda_latent_max=args.lambda_latent_max,
        lambda_token_init=args.lambda_token_init,
        lambda_token_late=args.lambda_token_late,
        latent_loss_type=args.latent_loss_type,
        token_loss_type=args.token_loss_type,
        teacher_decay_start_ratio=args.teacher_decay_start_ratio,
        teacher_decay_end_ratio=args.teacher_decay_end_ratio,
        pred_warmup_start_ratio=args.pred_warmup_start_ratio,
        pred_warmup_end_ratio=args.pred_warmup_end_ratio,
        latent_warmup_end_ratio=args.latent_warmup_end_ratio,
        token_decay_start_ratio=args.token_decay_start_ratio,
        token_decay_end_ratio=args.token_decay_end_ratio,
        pred_only_finetune_start_ratio=args.pred_only_finetune_start_ratio,
        stopgrad_wm_teacher=args.stopgrad_wm_teacher,
        stopgrad_teacher_token=args.stopgrad_teacher_token,
        stopgrad_adapter_input_for_token_loss=args.stopgrad_adapter_input_for_token_loss,
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
    log_rank(rank, f"normal dataset ready len={len(normal_dataset)}")
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
        balance_tasks=args.balance_failure_tasks,
        rebalance_failure_groups=args.rebalance_failure_groups,
    )
    log_rank(rank, f"failure dataset ready len={len(failure_dataset)}")
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
    log_rank(rank, "dataloaders ready")

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

    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=args.device)
    log_rank(rank, "teacher ready")
    model = ACTLatentStage1(
        act_args=_build_act_args(shared_camera_names, args),
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(args.device)
    log_rank(rank, "model instantiated")
    init_batch = next(iter(normal_loader))
    log_rank(rank, "normal init batch fetched")
    model.initialize_latent_heads(init_batch["image_t"][:1].to(args.device), teacher)
    log_rank(rank, "latent heads initialized")
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
    log_rank(rank, "checkpoint load complete")

    if start_epoch >= int(args.num_epochs):
        raise ValueError(
            f"resume start_epoch={start_epoch} is >= num_epochs={args.num_epochs}. "
            "num_epochs is treated as the total training endpoint, not additional epochs."
        )

    correction_builder = _build_correction_builder_serialized(
        args=args,
        teacher=teacher,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    log_rank(rank, "correction builder init phase complete")

    if distributed:
        model = DDP(
            model,
            device_ids=[int(args.device.split(":")[-1])],
            output_device=int(args.device.split(":")[-1]),
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
        log_rank(rank, "ddp wrapper ready")
    elif use_data_parallel:
        if teacher.model is not None:
            teacher.model = teacher.model.to(args.device)
        model = torch.nn.DataParallel(model, device_ids=dp_device_ids, output_device=dp_device_ids[0])
        log_rank(rank, f"data parallel wrapper ready device_ids={dp_device_ids}")
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if normal_sampler is not None:
                normal_sampler.set_epoch(epoch)
            if failure_sampler is not None:
                failure_sampler.set_epoch(epoch)
            model.train()
            raw_model = model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model
            if args.freeze_base_act:
                raw_model.base_act.eval()
            normal_iter = cycle_loader(normal_loader)
            failure_iter = cycle_loader(failure_loader)
            steps_per_epoch = len(normal_loader)
            start_time = time.time()
            pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", disable=not is_main_process(rank), leave=True)
            meter = {
                k: 0.0
                for k in [
                    "loss",
                    "action",
                    "action_cond",
                    "cond_token",
                    "latent",
                    "dyn",
                    "wm_curr",
                    "wm_future",
                    "bridge",
                    "lambda_teacher",
                    "lambda_pred",
                    "lambda_latent",
                    "lambda_token",
                    "beta",
                    "failure_valid",
                    "failure_skip",
                    "t_builder",
                    "t_rollout",
                    "cond_keep_ratio",
                    "action_cond_teacher",
                    "action_cond_pred",
                    "teacher_token_norm",
                    "pred_token_norm",
                    "delta_action",
                    "delta_action_normal",
                    "delta_action_failure",
                    "delta_action_teacher",
                    "delta_action_teacher_normal",
                    "delta_action_teacher_failure",
                ]
            }
            task_meter: dict[str, dict[str, float]] = {}
            steps_done = 0

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
                normal_items["teacher_current_latent"] = teacher.encode_image(
                    normal_batch["image_t"][:, 0].to(args.device, non_blocking=True)
                ).detach()

                failure_prepared: list[dict[str, torch.Tensor]] = []
                failure_skipped = 0
                t_builder = 0.0
                t_rollout = 0.0
                failure_task_names: list[str] = []
                if args.failure_batch_size > 0:
                    failure_batch = next(failure_iter)
                    should_prepare_failure = True
                    if distributed and int(args.failure_prepare_owner_rank) >= 0:
                        should_prepare_failure = rank == int(args.failure_prepare_owner_rank)
                    if should_prepare_failure:
                        failure_prepared, failure_skipped, t_builder, t_rollout = _prepare_failure_samples(
                            batch=failure_batch,
                            raw_model=raw_model,
                            correction_builder=correction_builder,
                            teacher=teacher,
                            norm_stats=norm_stats,
                            raw_cache=raw_cache,
                            args=args,
                        )
                        failure_task_names = [str(failure_batch["task_name"][i]) for i in range(len(failure_prepared))]
                    else:
                        failure_skipped = int(args.failure_batch_size)
                    if distributed and int(args.failure_prepare_owner_rank) >= 0:
                        failure_prepared, failure_task_names, failure_skipped, t_builder, t_rollout = _broadcast_failure_payload(
                            distributed=distributed,
                            src_rank=int(args.failure_prepare_owner_rank),
                            rank=rank,
                            prepared=failure_prepared,
                            prepared_task_names=failure_task_names,
                            skipped=failure_skipped,
                            t_builder=t_builder,
                            t_rollout=t_rollout,
                            device=args.device,
                            reference_items=normal_items,
                            task_name_vocab=task_name_vocab,
                        )
                    raw_model.train()
                    if args.freeze_base_act:
                        raw_model.base_act.eval()

                if failure_prepared:
                    failure_items = {
                        key: torch.stack([item[key] for item in failure_prepared], dim=0)
                        for key in normal_items.keys()
                        if key != "teacher_current_latent"
                    }
                    failure_items["teacher_current_latent"] = teacher.encode_image(failure_items["image_t"][:, 0].to(args.device)).detach()
                    train_items = {
                        key: torch.cat([normal_items[key], failure_items[key].to(args.device)], dim=0)
                        for key in normal_items.keys()
                    }
                else:
                    train_items = normal_items

                normal_task_names = [str(x) for x in list(normal_batch["task_name"])]
                train_task_names = normal_task_names + failure_task_names
                is_failure_sample = torch.tensor(
                    [False] * len(normal_task_names) + [True] * len(failure_task_names),
                    device=args.device,
                    dtype=torch.bool,
                )

                total_expected_samples = max(1, args.num_epochs * steps_per_epoch * current_global_batch_size)
                schedule_step = min(1.0, sample_count / float(total_expected_samples))
                forward_kwargs = dict(
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
                    wm_teacher=None if use_data_parallel else teacher,
                    global_step=schedule_step,
                    use_act_head_conditioning=True,
                    teacher_current_latent=train_items["teacher_current_latent"],
                    future_teacher_latent=train_items["future_teacher_latent"],
                    is_failure_sample=is_failure_sample,
                    mode="stage1",
                    return_dict=use_data_parallel,
                )
                out = model(**forward_kwargs)
                if use_data_parallel:
                    out = Stage1LossOutput(
                        loss=out["loss"],
                        loss_action=out["loss_action"],
                        loss_action_conditioned_teacher=out["loss_action_conditioned_teacher"],
                        loss_action_conditioned_pred=out["loss_action_conditioned_pred"],
                        loss_action_conditioned=out["loss_action_conditioned"],
                        loss_condition_token=out["loss_condition_token"],
                        loss_latent=out["loss_latent"],
                        loss_dynamics=out["loss_dynamics"],
                        loss_wm_action_current=out["loss_wm_action_current"],
                        loss_wm_action_future=out["loss_wm_action_future"],
                        loss_bridge_future=out["loss_bridge_future"],
                        beta_dynamics=float(out["beta_dynamics"].mean().item()),
                        progress=float(out["progress"].mean().item()),
                        lambda_teacher=float(out["lambda_teacher"].mean().item()),
                        lambda_pred=float(out["lambda_pred"].mean().item()),
                        lambda_latent=float(out["lambda_latent"].mean().item()),
                        lambda_token=float(out["lambda_token"].mean().item()),
                        cond_keep_ratio=float(out["cond_keep_ratio"].mean().item()),
                        teacher_token_norm_mean=out["teacher_token_norm_mean"],
                        pred_token_norm_mean=out["pred_token_norm_mean"],
                        delta_action_mean=out["delta_action_mean"],
                        delta_action_normal_mean=out["delta_action_normal_mean"],
                        delta_action_failure_mean=out["delta_action_failure_mean"],
                        delta_action_teacher_mean=out["delta_action_teacher_mean"],
                        delta_action_teacher_normal_mean=out["delta_action_teacher_normal_mean"],
                        delta_action_teacher_failure_mean=out["delta_action_teacher_failure_mean"],
                    )
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()

                global_step += 1
                sample_count += current_global_batch_size
                steps_done += 1
                meter["loss"] += float(out.loss.item())
                meter["action"] += float(out.loss_action.item())
                meter["action_cond"] += float(out.loss_action_conditioned.item())
                meter["cond_token"] += float(out.loss_condition_token.item())
                meter["latent"] += float(out.loss_latent.item())
                meter["dyn"] += float(out.loss_dynamics.item())
                meter["wm_curr"] += float(out.loss_wm_action_current.item())
                meter["wm_future"] += float(out.loss_wm_action_future.item())
                meter["bridge"] += float(out.loss_bridge_future.item())
                meter["lambda_teacher"] += float(out.lambda_teacher)
                meter["lambda_pred"] += float(out.lambda_pred)
                meter["lambda_latent"] += float(out.lambda_latent)
                meter["lambda_token"] += float(out.lambda_token)
                meter["beta"] += float(out.beta_dynamics)
                meter["failure_valid"] += float(len(failure_prepared))
                meter["failure_skip"] += float(failure_skipped)
                meter["t_builder"] += float(t_builder)
                meter["t_rollout"] += float(t_rollout)
                meter["cond_keep_ratio"] += float(out.cond_keep_ratio)
                meter["action_cond_teacher"] += float(out.loss_action_conditioned_teacher.item())
                meter["action_cond_pred"] += float(out.loss_action_conditioned_pred.item())
                meter["teacher_token_norm"] += float(out.teacher_token_norm_mean.item())
                meter["pred_token_norm"] += float(out.pred_token_norm_mean.item())
                meter["delta_action"] += float(out.delta_action_mean.item())
                meter["delta_action_normal"] += float(out.delta_action_normal_mean.item())
                meter["delta_action_failure"] += float(out.delta_action_failure_mean.item())
                meter["delta_action_teacher"] += float(out.delta_action_teacher_mean.item())
                meter["delta_action_teacher_normal"] += float(out.delta_action_teacher_normal_mean.item())
                meter["delta_action_teacher_failure"] += float(out.delta_action_teacher_failure_mean.item())

                step_task_counts: dict[str, dict[str, int]] = {}
                for idx, task_name in enumerate(train_task_names):
                    counts = step_task_counts.setdefault(str(task_name), {"all": 0, "normal": 0, "failure": 0})
                    counts["all"] += 1
                    if idx < len(normal_task_names):
                        counts["normal"] += 1
                    else:
                        counts["failure"] += 1
                for task_name, counts in step_task_counts.items():
                    bucket = task_meter.setdefault(
                        str(task_name),
                        {"samples": 0.0, "normal_samples": 0.0, "failure_samples": 0.0, "failure_valid": 0.0, "failure_skip": 0.0},
                    )
                    bucket["samples"] += float(counts["all"])
                    bucket["normal_samples"] += float(counts["normal"])
                    bucket["failure_samples"] += float(counts["failure"])
                    bucket["failure_valid"] += float(counts["failure"])
                if args.failure_batch_size > 0:
                    raw_failure_names = [str(x) for x in list(failure_batch["task_name"])]
                    skipped_by_task: dict[str, int] = {}
                    for name in raw_failure_names:
                        skipped_by_task[name] = skipped_by_task.get(name, 0) + 1
                    for name in failure_task_names:
                        if skipped_by_task.get(name, 0) > 0:
                            skipped_by_task[name] -= 1
                    for task_name, count in skipped_by_task.items():
                        if count <= 0:
                            continue
                        bucket = task_meter.setdefault(
                            str(task_name),
                            {"samples": 0.0, "normal_samples": 0.0, "failure_samples": 0.0, "failure_valid": 0.0, "failure_skip": 0.0},
                        )
                        bucket["failure_skip"] += float(count)

                if is_main_process(rank):
                    pbar.set_postfix(
                        loss=f"{out.loss.item():.4f}",
                        act=f"{out.loss_action.item():.4f}",
                        cond=f"{out.loss_action_conditioned.item():.4f}",
                        cteach=f"{out.loss_action_conditioned_teacher.item():.4f}",
                        cpred=f"{out.loss_action_conditioned_pred.item():.4f}",
                        ctoken=f"{out.loss_condition_token.item():.4f}",
                        latent=f"{out.loss_latent.item():.4f}",
                        lpred=f"{out.lambda_pred:.3f}",
                        lteach=f"{out.lambda_teacher:.3f}",
                        fvalid=len(failure_prepared),
                        dtch=f"{out.delta_action_teacher_mean.item():.4f}",
                        dact=f"{out.delta_action_mean.item():.4f}",
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
                        "train_action_conditioned_teacher_step": out.loss_action_conditioned_teacher.item(),
                        "train_action_conditioned_pred_step": out.loss_action_conditioned_pred.item(),
                        "train_condition_token_step": out.loss_condition_token.item(),
                        "train_latent_step": out.loss_latent.item(),
                        "train_dynamics_step": out.loss_dynamics.item(),
                        "beta_dynamics": out.beta_dynamics,
                        "lambda_teacher_step": out.lambda_teacher,
                        "lambda_pred_step": out.lambda_pred,
                        "lambda_latent_step": out.lambda_latent,
                        "lambda_token_step": out.lambda_token,
                        "progress_step": out.progress,
                        "failure_valid_step": len(failure_prepared),
                        "failure_skipped_step": failure_skipped,
                        "cond_keep_ratio_step": out.cond_keep_ratio,
                        "teacher_token_norm_step": out.teacher_token_norm_mean.item(),
                        "pred_token_norm_step": out.pred_token_norm_mean.item(),
                        "delta_action_step": out.delta_action_mean.item(),
                        "delta_action_normal_step": out.delta_action_normal_mean.item(),
                        "delta_action_failure_step": out.delta_action_failure_mean.item(),
                        "delta_action_teacher_step": out.delta_action_teacher_mean.item(),
                        "delta_action_teacher_normal_step": out.delta_action_teacher_normal_mean.item(),
                        "delta_action_teacher_failure_step": out.delta_action_teacher_failure_mean.item(),
                    },
                    step=global_step,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            n = max(1, steps_done)
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
                    f"cond={meter['action_cond']/n:.4f} "
                    f"cond_teacher={meter['action_cond_teacher']/n:.4f} "
                    f"cond_pred={meter['action_cond_pred']/n:.4f} "
                    f"ctoken={meter['cond_token']/n:.4f} "
                    f"latent={meter['latent']/n:.4f} dyn={meter['dyn']/n:.4f} "
                    f"beta={meter['beta']/n:.3f} "
                    f"lambda_teacher={meter['lambda_teacher']/n:.3f} "
                    f"lambda_pred={meter['lambda_pred']/n:.3f} "
                    f"lambda_latent={meter['lambda_latent']/n:.3f} "
                    f"lambda_token={meter['lambda_token']/n:.3f} "
                    f"cond_keep={meter['cond_keep_ratio']/n:.3f} "
                    f"teacher_token_norm={meter['teacher_token_norm']/n:.4f} "
                    f"pred_token_norm={meter['pred_token_norm']/n:.4f} "
                    f"delta_action_teacher={meter['delta_action_teacher']/n:.4f} "
                    f"delta_action={meter['delta_action']/n:.4f} "
                    f"delta_action_normal={meter['delta_action_normal']/n:.4f} "
                    f"delta_action_failure={meter['delta_action_failure']/n:.4f} "
                    f"delta_action_teacher_normal={meter['delta_action_teacher_normal']/n:.4f} "
                    f"delta_action_teacher_failure={meter['delta_action_teacher_failure']/n:.4f} "
                    f"failure_valid={meter['failure_valid']/n:.2f} failure_skip={meter['failure_skip']/n:.2f} "
                    f"time={epoch_time:.1f}s builder={meter['t_builder']/n:.2f}s rollout={meter['t_rollout']/n:.2f}s",
                    flush=True,
                )
                if task_meter:
                    for task_name in sorted(task_meter.keys()):
                        stats = task_meter[task_name]
                        print(
                            f"[epoch {epoch}][task {task_name}] "
                            f"samples={stats['samples']:.0f} normal={stats['normal_samples']:.0f} "
                            f"failure={stats['failure_samples']:.0f} valid={stats['failure_valid']:.0f} "
                            f"skip={stats['failure_skip']:.0f}",
                            flush=True,
                        )
            log_wandb(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "epoch_loss": meter["loss"] / n,
                    "epoch_action": meter["action"] / n,
                    "epoch_action_conditioned": meter["action_cond"] / n,
                    "epoch_action_conditioned_teacher": meter["action_cond_teacher"] / n,
                    "epoch_action_conditioned_pred": meter["action_cond_pred"] / n,
                    "epoch_condition_token": meter["cond_token"] / n,
                    "epoch_latent": meter["latent"] / n,
                    "epoch_dynamics": meter["dyn"] / n,
                    "epoch_beta_dynamics": meter["beta"] / n,
                    "epoch_lambda_teacher": meter["lambda_teacher"] / n,
                    "epoch_lambda_pred": meter["lambda_pred"] / n,
                    "epoch_lambda_latent": meter["lambda_latent"] / n,
                    "epoch_lambda_token": meter["lambda_token"] / n,
                    "epoch_cond_keep_ratio": meter["cond_keep_ratio"] / n,
                    "epoch_teacher_token_norm": meter["teacher_token_norm"] / n,
                    "epoch_pred_token_norm": meter["pred_token_norm"] / n,
                    "epoch_delta_action": meter["delta_action"] / n,
                    "epoch_delta_action_normal": meter["delta_action_normal"] / n,
                    "epoch_delta_action_failure": meter["delta_action_failure"] / n,
                    "epoch_delta_action_teacher": meter["delta_action_teacher"] / n,
                    "epoch_delta_action_teacher_normal": meter["delta_action_teacher_normal"] / n,
                    "epoch_delta_action_teacher_failure": meter["delta_action_teacher_failure"] / n,
                    "epoch_failure_valid": meter["failure_valid"] / n,
                    "epoch_failure_skipped": meter["failure_skip"] / n,
                    "epoch_time_sec": epoch_time,
                    "global_step": global_step,
                    "sample_count": sample_count,
                },
                step=global_step,
            )
            if task_meter:
                task_log_payload = {"global_step": global_step, "epoch": epoch + 1}
                for task_name, stats in task_meter.items():
                    safe_name = task_name.replace("/", "_").replace(" ", "_")
                    task_log_payload[f"task/{safe_name}/samples"] = stats["samples"]
                    task_log_payload[f"task/{safe_name}/normal_samples"] = stats["normal_samples"]
                    task_log_payload[f"task/{safe_name}/failure_samples"] = stats["failure_samples"]
                    task_log_payload[f"task/{safe_name}/failure_valid"] = stats["failure_valid"]
                    task_log_payload[f"task/{safe_name}/failure_skip"] = stats["failure_skip"]
                log_wandb(wandb_run, task_log_payload, step=global_step)

            if is_main_process(rank) and (epoch + 1) % args.save_freq == 0:
                raw_model_save = model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model
                ckpt_args = dict(vars(args))
                if raw_model_save._latent_target_hw is not None:
                    ckpt_args["latent_target_hw"] = list(raw_model_save._latent_target_hw)
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
                    "args": ckpt_args,
                    "multi_task_names": list(args.multi_task_names),
                    "algorithm": "stage1_unified_normal_failure_pred_mainline",
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
