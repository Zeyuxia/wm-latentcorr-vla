from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.SmolVLA.latentcorr.act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from policy.SmolVLA.latentcorr.correction_policy_adapter import SampleBoundSmolVLAAdapter
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.failure_manifest_utils import load_failure_table_paths
from policy.SmolVLA.latentcorr.latent_config import DynamicsWarmupConfig, Stage1WarmupConfig
from policy.SmolVLA.latentcorr.latent_dataset_utils import load_raw_episode
from policy.SmolVLA.latentcorr.multitask_failure_dataset import MultiTaskFailureDatasetConfig, build_multitask_failure_dataset
from policy.SmolVLA.latentcorr.multitask_latent_utils import build_multitask_stage1_dataset, resolve_multitask_specs
from policy.SmolVLA.latentcorr.smolvla_data_utils import build_smolvla_batch, make_smolvla_processors, stack_smolvla_batches
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentBridgeConfig, SmolVLALatentPolicy


def parse_task_parts(task_name: str) -> tuple[str, str]:
    if not task_name.startswith("sim-"):
        raise ValueError(f"Expected sim task name, got {task_name}")
    parts = task_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected multitask name format: {task_name}")
    return parts[1], parts[2]


def parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Expected 'true' or 'false', got {value}")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_accelerator() -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])


def compute_retain_weight(
    retain_weight: float,
    retain_weight_final: float,
    retain_decay_start_step: float,
    retain_decay_end_step: float,
    retain_decay_curve: str,
    step_progress: float,
) -> float:
    if retain_decay_end_step <= retain_decay_start_step:
        return retain_weight
    if step_progress <= retain_decay_start_step:
        return retain_weight
    if step_progress >= retain_decay_end_step:
        return retain_weight_final
    ratio = (step_progress - retain_decay_start_step) / max(1e-8, retain_decay_end_step - retain_decay_start_step)
    ratio = max(0.0, min(1.0, ratio))
    if retain_decay_curve == "cosine":
        ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return retain_weight + (retain_weight_final - retain_weight) * ratio


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smolvla_pretrained_path", type=str, required=True)
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--freeze_vision_encoder", type=str, required=True, choices=["true", "false"])
    parser.add_argument("--train_expert_only", type=str, required=True, choices=["true", "false"])
    parser.add_argument("--load_vlm_weights", type=str, required=True, choices=["true", "false"])
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--instruction_type", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--save_freq", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--scheduler_warmup_steps", type=int, required=True)
    parser.add_argument("--scheduler_decay_steps", type=int, required=True)
    parser.add_argument("--scheduler_decay_lr", type=float, required=True)
    parser.add_argument("--future_offset", type=int, required=True)
    parser.add_argument("--prefix_steps", type=int, required=True)
    parser.add_argument("--action_dim", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--adapter_hidden_dim", type=int, required=True)
    parser.add_argument("--predictor_hidden_dim", type=int, required=True)
    parser.add_argument("--dyn_zero_steps", type=int, required=True)
    parser.add_argument("--dyn_ramp_steps", type=int, required=True)
    parser.add_argument("--dyn_max_weight", type=float, required=True)
    parser.add_argument("--dyn_warmup_curve", type=str, required=True, choices=["linear", "cosine"])
    parser.add_argument("--cond_zero_steps", type=int, default=None)
    parser.add_argument("--cond_ramp_steps", type=int, default=None)
    parser.add_argument("--cond_max_weight", type=float, default=None)
    parser.add_argument("--cond_warmup_curve", type=str, default=None, choices=["linear", "cosine"])
    parser.add_argument("--act_chunk_size", type=int, required=True)


def add_stage2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--failure_table_paths_json", type=str, required=True)
    parser.add_argument("--urdf_path", type=str, required=True)
    parser.add_argument("--curobo_left_yml", type=str, required=True)
    parser.add_argument("--curobo_right_yml", type=str, required=True)
    parser.add_argument("--correction_batch_size", type=int, required=True)
    parser.add_argument("--retain_weight", type=float, required=True)
    parser.add_argument("--retain_weight_final", type=float, required=True)
    parser.add_argument("--retain_decay_start_step", type=float, required=True)
    parser.add_argument("--retain_decay_end_step", type=float, required=True)
    parser.add_argument("--retain_decay_curve", type=str, required=True, choices=["linear", "cosine"])
    parser.add_argument("--failure_phase_bins", type=int, required=True)
    parser.add_argument("--failure_translation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_translation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_explore_k", type=int, required=True)
    parser.add_argument("--sample_phase_window_len", type=int, required=True)
    parser.add_argument("--start_margin", type=int, required=True)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, required=True)
    parser.add_argument("--planner_orient_weight", type=float, required=True)
    parser.add_argument("--planner_gripper_penalty", type=float, required=True)
    parser.add_argument("--planner_nearest_window_radius", type=int, required=True)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, required=True)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_save_video", type=str2bool, required=True)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, required=True)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, required=True)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, required=True)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, required=True)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, required=True)


def add_failure_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--failure_mode", type=str, default="off", choices=["off", "train"])
    parser.add_argument("--failure_table_paths_json", type=str, default="")
    parser.add_argument("--failure_corr_batch_ratio", type=float, default=0.0)
    parser.add_argument("--failure_phase_bins", type=int, default=3)
    parser.add_argument("--failure_translation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_translation_mag_bins", type=int, default=1)
    parser.add_argument("--failure_rotation_dir_bins", type=int, default=6)
    parser.add_argument("--failure_rotation_mag_bins", type=int, default=1)
    parser.add_argument("--failure_explore_k", type=int, default=4)
    parser.add_argument("--sample_phase_window_len", type=int, default=20)
    parser.add_argument("--start_margin", type=int, default=16)
    parser.add_argument("--urdf_path", type=str, default="")
    parser.add_argument("--curobo_left_yml", type=str, default="")
    parser.add_argument("--curobo_right_yml", type=str, default="")
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, default=16)
    parser.add_argument("--planner_orient_weight", type=float, default=0.0573)
    parser.add_argument("--planner_gripper_penalty", type=float, default=1.0)
    parser.add_argument("--planner_nearest_window_radius", type=int, default=12)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, default=0.01)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, default=0.05)
    parser.add_argument("--recover_eval_save_video", type=str2bool, default=False)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, default=0.2)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, default=0.04)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, default=8.0)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, default=16)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, default=16)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, default=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, default=0.08)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, default=15.0)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, default=0.10)
    parser.add_argument("--debug_wm_correction", type=str2bool, default=False)
    parser.add_argument("--debug_wm_all_ranks", type=str2bool, default=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("train smolvla latentcorr")
    subparsers = parser.add_subparsers(dest="train_mode", required=True)

    stage1_parser = subparsers.add_parser("stage1")
    add_common_args(stage1_parser)
    add_failure_train_args(stage1_parser)

    stage2_parser = subparsers.add_parser("stage2")
    add_common_args(stage2_parser)
    add_stage2_args(stage2_parser)

    return parser


def build_bridge_config(args: argparse.Namespace) -> SmolVLALatentBridgeConfig:
    return SmolVLALatentBridgeConfig(
        action_dim=int(args.action_dim),
        prefix_steps=int(args.prefix_steps),
        latent_dim=int(args.latent_dim),
        adapter_hidden_dim=int(args.adapter_hidden_dim),
        predictor_hidden_dim=int(args.predictor_hidden_dim),
    )


def build_warmup_config(args: argparse.Namespace) -> DynamicsWarmupConfig:
    return DynamicsWarmupConfig(
        zero_steps=int(args.dyn_zero_steps),
        ramp_steps=int(args.dyn_ramp_steps),
        max_weight=float(args.dyn_max_weight),
        curve=str(args.dyn_warmup_curve),
    )


def build_stage1_warmup_config(args: argparse.Namespace) -> Stage1WarmupConfig:
    dynamics = build_warmup_config(args)
    condition = DynamicsWarmupConfig(
        zero_steps=int(args.cond_zero_steps if args.cond_zero_steps is not None else args.dyn_zero_steps),
        ramp_steps=int(args.cond_ramp_steps if args.cond_ramp_steps is not None else args.dyn_ramp_steps),
        max_weight=float(args.cond_max_weight if args.cond_max_weight is not None else args.dyn_max_weight),
        curve=str(args.cond_warmup_curve if args.cond_warmup_curve is not None else args.dyn_warmup_curve),
    )
    return Stage1WarmupConfig(dynamics=dynamics, condition=condition)


def build_base_policy(args: argparse.Namespace) -> tuple[SmolVLAPolicy, Any]:
    base_policy = SmolVLAPolicy.from_pretrained(args.smolvla_pretrained_path)
    base_policy.config.freeze_vision_encoder = parse_bool_flag(args.freeze_vision_encoder)
    base_policy.config.train_expert_only = parse_bool_flag(args.train_expert_only)
    base_policy.config.load_vlm_weights = parse_bool_flag(args.load_vlm_weights)
    base_policy.model.vlm_with_expert.freeze_vision_encoder = base_policy.config.freeze_vision_encoder
    base_policy.model.vlm_with_expert.train_expert_only = base_policy.config.train_expert_only
    base_policy.model.vlm_with_expert.set_requires_grad()
    base_policy.model.set_requires_grad()
    base_policy.to(torch.device(args.device))
    preprocess, postprocess = make_smolvla_processors(base_policy, args.smolvla_pretrained_path)
    return base_policy, preprocess


def configure_optimizer(model: SmolVLALatentPolicy, args: argparse.Namespace, total_training_steps: int):
    model.base_policy.config.optimizer_lr = float(args.learning_rate)
    model.base_policy.config.optimizer_weight_decay = float(args.weight_decay)
    model.base_policy.config.scheduler_warmup_steps = int(args.scheduler_warmup_steps)
    model.base_policy.config.scheduler_decay_steps = int(args.scheduler_decay_steps)
    model.base_policy.config.scheduler_decay_lr = float(args.scheduler_decay_lr)
    optimizer_cfg = model.base_policy.config.get_optimizer_preset()
    optimizer = optimizer_cfg.build(model.parameters())
    scheduler_cfg = model.base_policy.config.get_scheduler_preset()
    lr_scheduler = scheduler_cfg.build(optimizer, int(total_training_steps))
    return optimizer_cfg, optimizer, lr_scheduler


def log_stage1_tensorboard(writer: SummaryWriter, step: int, output: Any, lr: float) -> None:
    writer.add_scalar("train/loss", float(output.loss.item()), step)
    writer.add_scalar("train/loss_action", float(output.loss_action.item()), step)
    writer.add_scalar("train/loss_action_conditioned", float(output.loss_action_conditioned.item()), step)
    writer.add_scalar("train/loss_dynamics", float(output.loss_dynamics.item()), step)
    writer.add_scalar("train/beta_condition", float(output.beta_condition), step)
    writer.add_scalar("train/beta_dynamics", float(output.beta_dynamics), step)
    writer.add_scalar("train/lr", float(lr), step)


def log_stage2_tensorboard(
    writer: SummaryWriter,
    step: int,
    output: Any,
    lr: float,
    retain_weight: float,
    skipped_batches: int,
) -> None:
    writer.add_scalar("train/loss", float(output.loss.item()), step)
    writer.add_scalar("train/loss_correct", float(output.loss_correct.item()), step)
    writer.add_scalar("train/loss_retain", float(output.loss_retain.item()), step)
    writer.add_scalar("train/loss_dynamics", float(output.loss_dynamics.item()), step)
    writer.add_scalar("train/beta_dynamics", float(output.beta_dynamics), step)
    writer.add_scalar("train/retain_weight", float(retain_weight), step)
    writer.add_scalar("train/lr", float(lr), step)
    writer.add_scalar("train/skipped_batches", float(skipped_batches), step)


def save_train_args(output_dir: Path, args: argparse.Namespace) -> None:
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)


def save_training_checkpoint(
    output_dir: Path,
    prefix: str,
    step: int,
    model: SmolVLALatentPolicy,
    optimizer: Any,
    lr_scheduler: Any,
    extra_state: dict[str, Any],
) -> None:
    checkpoint_path = output_dir / f"{prefix}_step_{step:06d}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "global_step": step,
    }
    payload.update(extra_state)
    torch.save(payload, checkpoint_path)


def resume_training_checkpoint(
    checkpoint_path: str,
    model: SmolVLALatentPolicy,
    optimizer: Any,
    lr_scheduler: Any,
) -> tuple[int, int]:
    if not checkpoint_path:
        return 0, 0
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["global_step"]), int(checkpoint.get("epoch", 0))


def build_smolvla_batch_from_raw_sample(
    latent_policy: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    task_name_full: str,
    image_t: torch.Tensor,
    qpos_raw: torch.Tensor,
    action_chunk_raw: torch.Tensor,
    episode_id: int,
) -> dict[str, torch.Tensor]:
    task_only, task_config = parse_task_parts(task_name_full)
    return build_smolvla_batch(
        policy=latent_policy.base_policy,
        preprocess=preprocess,
        image_t=image_t,
        qpos_raw=qpos_raw,
        action_chunk_raw=action_chunk_raw,
        task_name=task_only,
        task_config=task_config,
        episode_id=episode_id,
        instruction_type=instruction_type,
    )


def initialize_latent_policy_from_sample(
    latent_policy: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    raw_sample: dict[str, Any],
    teacher: EvacLatentTeacher | None = None,
) -> None:
    init_batch = build_smolvla_batch_from_raw_sample(
        latent_policy=latent_policy,
        preprocess=preprocess,
        instruction_type=instruction_type,
        task_name_full=raw_sample["task_name"],
        image_t=raw_sample["image_t"],
        qpos_raw=raw_sample["qpos_raw"],
        action_chunk_raw=raw_sample["act_action_chunk_raw"],
        episode_id=int(raw_sample["episode_id"].item()),
    )
    teacher_latent = None
    if teacher is not None:
        teacher_input = raw_sample["image_t"][0].unsqueeze(0).to(device=latent_policy.device, dtype=torch.float32)
        teacher_latent = teacher.encode_image(teacher_input)
    latent_policy.initialize_from_batch(init_batch, teacher_latent=teacher_latent)


def build_stage1_samples_from_raw_batch(
    raw_batch: dict[str, Any],
    latent_model: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    prefix_steps: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor]]:
    samples = []
    future_images = []
    action_prefix = []
    for batch_index in range(raw_batch["image_t"].shape[0]):
        task_name = raw_batch["task_name"][batch_index]
        task_only, task_config = parse_task_parts(task_name)
        sample = build_smolvla_batch(
            policy=latent_model.base_policy,
            preprocess=preprocess,
            image_t=raw_batch["image_t"][batch_index],
            qpos_raw=raw_batch["qpos_raw"][batch_index],
            action_chunk_raw=raw_batch["act_action_chunk_raw"][batch_index],
            task_name=task_only,
            task_config=task_config,
            episode_id=int(raw_batch["episode_id"][batch_index].item()),
            instruction_type=instruction_type,
        )
        samples.append(sample)
        future_images.append(raw_batch["image_t_future"][batch_index, 0])
        action_prefix.append(sample["action"][:, : int(prefix_steps), :])
    return samples, future_images, action_prefix


def build_stage1_correction_samples(
    correction_raw_batch: dict[str, Any],
    latent_model: SmolVLALatentPolicy,
    preprocess,
    instruction_type: str,
    args: argparse.Namespace,
    correction_builder: ACTAlignedCorrectionBuilder,
    teacher: EvacLatentTeacher,
    norm_stats: dict[str, Any],
    raw_cache: dict[tuple[str, int], dict[str, Any]],
    step_debug_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor], Counter[str]]:
    device = torch.device(args.device)
    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=device)
    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=device)

    samples = []
    future_images = []
    action_prefix = []
    skip_reasons: Counter[str] = Counter()
    for sample_index in range(correction_raw_batch["image_t"].shape[0]):
        task_name_full = correction_raw_batch["task_name"][sample_index]
        task_only, task_config = parse_task_parts(task_name_full)
        episode_id = int(correction_raw_batch["episode_id"][sample_index].item())
        adapter = SampleBoundSmolVLAAdapter(
            latent_policy=latent_model,
            preprocess=preprocess,
            postprocess=postprocess,
            task_name=task_only,
            task_config=task_config,
            episode_id=episode_id,
            instruction_type=instruction_type,
        )
        raw_cache_key = (str(correction_raw_batch["raw_data_dir"][sample_index]), episode_id)
        if raw_cache_key not in raw_cache:
            raw_cache[raw_cache_key] = load_raw_episode(
                correction_raw_batch["raw_data_dir"][sample_index],
                episode_id,
            )
        correction = correction_builder.build(
            latent_model=adapter,
            image_t=correction_raw_batch["image_t"][sample_index].to(device),
            qpos_t=correction_raw_batch["qpos_t"][sample_index].to(device),
            raw_data=raw_cache[raw_cache_key],
            norm_stats=norm_stats,
            start_ts=int(correction_raw_batch["start_ts"][sample_index].item()),
            failure_mode_override="train",
            sampled_phase_id=int(correction_raw_batch["sampled_phase_id"][sample_index].item()),
            sampled_phase_bin_id=int(correction_raw_batch["sampled_phase_bin_id"][sample_index].item()),
            sampled_phase_instance_id=int(correction_raw_batch["sampled_phase_instance_id"][sample_index].item()),
            forced_error_mode_id=int(correction_raw_batch["forced_error_mode_id"][sample_index].item()),
            sampled_active_arm_pattern_id=int(correction_raw_batch["sampled_active_arm_pattern_id"][sample_index].item()),
            forced_dir_bin_id=int(correction_raw_batch["forced_dir_bin_id"][sample_index].item()),
            forced_mag_bin_id=int(correction_raw_batch["forced_mag_bin_id"][sample_index].item()),
            sampled_mode_prob=float(correction_raw_batch["sampled_mode_prob"][sample_index].item()),
            sampled_entry_prob_within_mode=float(correction_raw_batch["sampled_entry_prob_within_mode"][sample_index].item()),
            sampled_unit_prob=float(correction_raw_batch["sampled_unit_prob"][sample_index].item()),
            debug_dir=(None if step_debug_dir is None else str(Path(step_debug_dir) / f"bi{sample_index}")),
            precomputed_action_chunk_norm=correction_raw_batch["act_action_chunk"][sample_index],
        )
        if correction is None:
            skip_meta = correction_builder.pop_last_skip_meta()
            skip_reason = "unknown_skip"
            if isinstance(skip_meta, dict) and skip_meta.get("skip_reason") is not None:
                skip_reason = str(skip_meta["skip_reason"]).strip() or "unknown_skip"
            skip_reasons[skip_reason] += 1
            continue
        if correction["corr_image"] is None or correction["corr_qpos_norm"] is None or correction["corr_action_chunk_norm"] is None:
            skip_reasons["missing_correction_targets"] += 1
            continue

        corr_qpos_raw = correction["corr_qpos_norm"] * qpos_std + qpos_mean
        corr_action_raw = correction["corr_action_chunk_norm"] * action_std.view(1, -1) + action_mean.view(1, -1)
        corr_image = correction["corr_image"]
        if corr_image.ndim != 4 or int(corr_image.shape[0]) != 1:
            raise ValueError(f"Expected single-camera correction image, got {tuple(corr_image.shape)}")
        smolvla_batch = build_smolvla_batch_from_raw_sample(
            latent_policy=latent_model,
            preprocess=preprocess,
            instruction_type=instruction_type,
            task_name_full=task_name_full,
            image_t=corr_image.detach().cpu(),
            qpos_raw=corr_qpos_raw.detach().cpu(),
            action_chunk_raw=corr_action_raw.detach().cpu(),
            episode_id=episode_id,
        )
        samples.append(smolvla_batch)
        future_images.append(corr_image[0].detach().cpu())
        if correction["error_action_prefix_norm"] is not None:
            action_prefix.append(correction["error_action_prefix_norm"][None, ...])
        else:
            action_prefix.append(smolvla_batch["action"][:, : int(args.prefix_steps), :])
    return samples, future_images, action_prefix, skip_reasons


def run_stage1(args: argparse.Namespace) -> None:
    accelerator = build_accelerator()
    is_main = bool(accelerator.is_main_process)
    args.device = str(accelerator.device)
    output_dir = Path(args.output_dir).resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    else:
        tb_writer = None

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    dataset, _ = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    failure_mode = str(args.failure_mode).strip().lower()
    use_failure_corr_loader = bool(failure_mode == "train")
    if use_failure_corr_loader:
        if not str(args.failure_table_paths_json).strip():
            raise ValueError("stage1 failure_mode=train requires --failure_table_paths_json")
        for path_arg, name in (
            (args.urdf_path, "urdf_path"),
            (args.curobo_left_yml, "curobo_left_yml"),
            (args.curobo_right_yml, "curobo_right_yml"),
        ):
            if not str(path_arg).strip():
                raise ValueError(f"stage1 failure_mode=train requires --{name}")
        failure_table_paths = load_failure_table_paths(args.failure_table_paths_json)
        corr_batch_size = int(max(1, round(float(args.batch_size) * float(args.failure_corr_batch_ratio))))
        failure_cfg = MultiTaskFailureDatasetConfig(
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            sample_phase_window_len=int(args.sample_phase_window_len),
            start_margin=int(args.start_margin),
            failure_table_paths={str(key): str(value) for key, value in failure_table_paths.items()},
            failure_phase_bins=int(args.failure_phase_bins),
            failure_translation_dir_bins=int(args.failure_translation_dir_bins),
            failure_translation_mag_bins=int(args.failure_translation_mag_bins),
            failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
            failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
            failure_explore_k=int(args.failure_explore_k),
        )
        correction_dataset, norm_stats = build_multitask_failure_dataset(
            task_specs=task_specs,
            config=failure_cfg,
            mode="train",
        )
        correction_loader = DataLoader(
            correction_dataset,
            batch_size=corr_batch_size,
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    else:
        failure_table_paths = {}
        correction_dataset = None
        correction_loader = None
        norm_stats = dataset.stats if hasattr(dataset, "stats") else None

    base_policy, preprocess = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_stage1_warmup_config(args),
    )
    model.to(torch.device(args.device))
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
        load_rollout_model=use_failure_corr_loader,
    )

    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=dataset[0],
        teacher=teacher,
    )
    correction_builder = None
    if use_failure_corr_loader:
        correction_builder = ACTAlignedCorrectionBuilder(
            cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
            urdf_path=args.urdf_path,
            curobo_left_yml=args.curobo_left_yml,
            curobo_right_yml=args.curobo_right_yml,
            device=args.device,
            shared_evac_model=teacher.model,
            shared_evac_config=teacher.cfg,
        )

    optimizer_cfg, optimizer, lr_scheduler = configure_optimizer(
        model=model,
        args=args,
        total_training_steps=int(args.max_steps),
    )
    if use_failure_corr_loader:
        model, optimizer, dataloader, correction_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, correction_loader, lr_scheduler
        )
    else:
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    latent_model = accelerator.unwrap_model(model)
    if is_main:
        save_train_args(output_dir, args)

    correction_iter = iter(correction_loader) if use_failure_corr_loader else None
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    skip_reason_counter: Counter[str] = Counter()
    debug_wm = bool(getattr(args, "debug_wm_correction", False))
    debug_wm_all_ranks = bool(getattr(args, "debug_wm_all_ranks", False))
    debug_wm_root = (output_dir / "debug_wm") if debug_wm else None
    debug_wm_dir = None
    if debug_wm_root is not None:
        debug_wm_dir = (
            debug_wm_root / f"rank{int(accelerator.process_index):02d}"
            if debug_wm_all_ranks else debug_wm_root
        )
    debug_wm_should_save = bool(debug_wm_dir is not None and (debug_wm_all_ranks or accelerator.is_main_process))
    if debug_wm_should_save:
        debug_wm_dir.mkdir(parents=True, exist_ok=True)
    global_step, epoch = resume_training_checkpoint(
        checkpoint_path=str(args.resume_ckpt),
        model=latent_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    max_steps = int(args.max_steps)
    progress = tqdm(total=max_steps, initial=global_step, desc="stage1", leave=True, disable=not is_main)
    try:
        while global_step < max_steps:
            for raw_batch in dataloader:
                samples, future_images, action_prefix = build_stage1_samples_from_raw_batch(
                    raw_batch=raw_batch,
                    latent_model=latent_model,
                    preprocess=preprocess,
                    instruction_type=args.instruction_type,
                    prefix_steps=int(args.prefix_steps),
                )
                if use_failure_corr_loader:
                    assert correction_iter is not None
                    assert correction_builder is not None
                    assert norm_stats is not None
                    try:
                        correction_raw_batch = next(correction_iter)
                    except StopIteration:
                        correction_iter = iter(correction_loader)
                        correction_raw_batch = next(correction_iter)
                    step_debug_dir = None
                    if debug_wm_should_save:
                        step_debug_dir = str(debug_wm_dir / f"step_{global_step:06d}")
                        Path(step_debug_dir).mkdir(parents=True, exist_ok=True)
                    corr_samples, corr_future_images, corr_action_prefix, corr_skip_reasons = build_stage1_correction_samples(
                        correction_raw_batch=correction_raw_batch,
                        latent_model=latent_model,
                        preprocess=preprocess,
                        instruction_type=args.instruction_type,
                        args=args,
                        correction_builder=correction_builder,
                        teacher=teacher,
                        norm_stats=norm_stats,
                        raw_cache=raw_cache,
                        step_debug_dir=step_debug_dir,
                    )
                    skip_reason_counter.update(corr_skip_reasons)
                    samples.extend(corr_samples)
                    future_images.extend(corr_future_images)
                    action_prefix.extend(corr_action_prefix)
                    if not corr_samples:
                        continue

                batch = stack_smolvla_batches(samples)
                future_image_tensor = torch.stack(future_images, dim=0).to(device=accelerator.device, dtype=torch.float32)
                future_teacher_latent = teacher.encode_image(future_image_tensor)
                action_prefix_tensor = torch.cat(action_prefix, dim=0).to(device=accelerator.device)

                output = model(
                    train_stage="stage1",
                    batch=batch,
                    future_teacher_latent=future_teacher_latent,
                    action_prefix=action_prefix_tensor,
                    global_step=global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(output.loss)
                if optimizer_cfg.grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), optimizer_cfg.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()

                current_lr = float(optimizer.param_groups[0]["lr"])
                if tb_writer is not None:
                    log_stage1_tensorboard(
                        writer=tb_writer,
                        step=global_step,
                        output=output,
                        lr=current_lr,
                    )

                progress.set_postfix(
                    loss=f"{output.loss.item():.4f}",
                    action=f"{output.loss_action.item():.4f}",
                    cond=f"{output.loss_action_conditioned.item():.4f}",
                    dyn=f"{output.loss_dynamics.item():.4f}",
                    beta_cond=f"{output.beta_condition:.4f}",
                    beta=f"{output.beta_dynamics:.4f}",
                    lr=f"{current_lr:.2e}",
                    corr=(len(samples) - int(raw_batch["image_t"].shape[0])),
                    skip=(skip_reason_counter.most_common(1)[0][0] if skip_reason_counter else "none"),
                    step=global_step,
                )
                global_step += 1
                progress.update(1)
                if global_step % int(args.save_freq) == 0 and is_main:
                    save_training_checkpoint(
                        output_dir=output_dir,
                        prefix="stage1",
                        step=global_step,
                        model=latent_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        extra_state={
                            "norm_stats": dataset.stats if hasattr(dataset, "stats") else None,
                            "target_hw": latent_model.target_hw,
                            "failure_table_paths": failure_table_paths,
                            "epoch": epoch,
                            "args": vars(args),
                        },
                    )
                if global_step >= max_steps:
                    break
            epoch += 1
        if global_step % int(args.save_freq) != 0 and is_main:
            save_training_checkpoint(
                output_dir=output_dir,
                prefix="stage1",
                step=global_step,
                model=latent_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                extra_state={
                    "norm_stats": dataset.stats if hasattr(dataset, "stats") else None,
                    "target_hw": latent_model.target_hw,
                    "failure_table_paths": failure_table_paths,
                    "epoch": epoch,
                    "args": vars(args),
                },
            )
    finally:
        progress.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


def run_stage2(args: argparse.Namespace) -> None:
    accelerator = build_accelerator()
    is_main = bool(accelerator.is_main_process)
    args.device = str(accelerator.device)
    output_dir = Path(args.output_dir).resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    else:
        tb_writer = None

    failure_table_paths = load_failure_table_paths(args.failure_table_paths_json)

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
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    failure_cfg = MultiTaskFailureDatasetConfig(
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
        sample_phase_window_len=int(args.sample_phase_window_len),
        start_margin=int(args.start_margin),
        failure_table_paths={str(key): str(value) for key, value in failure_table_paths.items()},
        failure_phase_bins=int(args.failure_phase_bins),
        failure_translation_dir_bins=int(args.failure_translation_dir_bins),
        failure_translation_mag_bins=int(args.failure_translation_mag_bins),
        failure_rotation_dir_bins=int(args.failure_rotation_dir_bins),
        failure_rotation_mag_bins=int(args.failure_rotation_mag_bins),
        failure_explore_k=int(args.failure_explore_k),
    )
    correction_dataset, _ = build_multitask_failure_dataset(task_specs=task_specs, config=failure_cfg, mode="train")
    correction_loader = DataLoader(
        correction_dataset,
        batch_size=int(args.correction_batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    base_policy, preprocess = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_warmup_config(args),
    )
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
        load_rollout_model=True,
    )
    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=normal_dataset[0],
        teacher=teacher,
    )
    stage1_checkpoint = torch.load(args.stage1_ckpt, map_location="cpu")
    model.load_state_dict(stage1_checkpoint["model"], strict=True)
    model.to(torch.device(args.device))
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=args.device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )
    optimizer_cfg, optimizer, lr_scheduler = configure_optimizer(
        model=model,
        args=args,
        total_training_steps=int(args.max_steps),
    )
    model, optimizer, normal_loader, correction_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, normal_loader, correction_loader, lr_scheduler
    )
    latent_model = accelerator.unwrap_model(model)
    if is_main:
        save_train_args(output_dir, args)

    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=args.device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=args.device)
    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=args.device)
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}

    correction_iter = iter(correction_loader)
    global_step, epoch = resume_training_checkpoint(
        checkpoint_path=str(args.resume_ckpt),
        model=latent_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    max_steps = int(args.max_steps)
    skip_reason_counter: Counter[str] = Counter()
    progress = tqdm(total=max_steps, initial=global_step, desc="stage2", leave=True, disable=not is_main)
    try:
        while global_step < max_steps:
            for normal_raw_batch in normal_loader:
                try:
                    correction_raw_batch = next(correction_iter)
                except StopIteration:
                    correction_iter = iter(correction_loader)
                    correction_raw_batch = next(correction_iter)

                normal_samples = []
                for sample_index in range(normal_raw_batch["image_t"].shape[0]):
                    normal_samples.append(
                        build_smolvla_batch_from_raw_sample(
                            latent_policy=latent_model,
                            preprocess=preprocess,
                            instruction_type=args.instruction_type,
                            task_name_full=normal_raw_batch["task_name"][sample_index],
                            image_t=normal_raw_batch["image_t"][sample_index],
                            qpos_raw=normal_raw_batch["qpos_raw"][sample_index],
                            action_chunk_raw=normal_raw_batch["act_action_chunk_raw"][sample_index],
                            episode_id=int(normal_raw_batch["episode_id"][sample_index].item()),
                        )
                    )
                normal_batch = stack_smolvla_batches(normal_samples)

                correction_samples = []
                correction_teacher_latents = []
                rollout_teacher_latents = []
                correction_actions = []
                correction_action_prefixes = []

                for sample_index in range(correction_raw_batch["image_t"].shape[0]):
                    task_name_full = correction_raw_batch["task_name"][sample_index]
                    task_only, task_config = parse_task_parts(task_name_full)
                    episode_id = int(correction_raw_batch["episode_id"][sample_index].item())
                    adapter = SampleBoundSmolVLAAdapter(
                        latent_policy=latent_model,
                        preprocess=preprocess,
                        postprocess=postprocess,
                        task_name=task_only,
                        task_config=task_config,
                        episode_id=episode_id,
                        instruction_type=args.instruction_type,
                    )
                    raw_cache_key = (str(correction_raw_batch["raw_data_dir"][sample_index]), episode_id)
                    if raw_cache_key not in raw_cache:
                        raw_cache[raw_cache_key] = load_raw_episode(
                            correction_raw_batch["raw_data_dir"][sample_index],
                            episode_id,
                        )
                    correction = correction_builder.build(
                        latent_model=adapter,
                        image_t=correction_raw_batch["image_t"][sample_index].to(args.device),
                        qpos_t=correction_raw_batch["qpos_t"][sample_index].to(args.device),
                        raw_data=raw_cache[raw_cache_key],
                        norm_stats=norm_stats,
                        start_ts=int(correction_raw_batch["start_ts"][sample_index].item()),
                        failure_mode_override="train",
                        sampled_phase_id=int(correction_raw_batch["sampled_phase_id"][sample_index].item()),
                        sampled_phase_bin_id=int(correction_raw_batch["sampled_phase_bin_id"][sample_index].item()),
                        sampled_phase_instance_id=int(correction_raw_batch["sampled_phase_instance_id"][sample_index].item()),
                        forced_error_mode_id=int(correction_raw_batch["forced_error_mode_id"][sample_index].item()),
                        sampled_active_arm_pattern_id=int(correction_raw_batch["sampled_active_arm_pattern_id"][sample_index].item()),
                        forced_dir_bin_id=int(correction_raw_batch["forced_dir_bin_id"][sample_index].item()),
                        forced_mag_bin_id=int(correction_raw_batch["forced_mag_bin_id"][sample_index].item()),
                        sampled_mode_prob=float(correction_raw_batch["sampled_mode_prob"][sample_index].item()),
                        sampled_entry_prob_within_mode=float(correction_raw_batch["sampled_entry_prob_within_mode"][sample_index].item()),
                        sampled_unit_prob=float(correction_raw_batch["sampled_unit_prob"][sample_index].item()),
                        precomputed_action_chunk_norm=correction_raw_batch["act_action_chunk"][sample_index],
                    )
                    if correction is None:
                        skip_meta = correction_builder.pop_last_skip_meta()
                        skip_reason = "unknown_skip"
                        if isinstance(skip_meta, dict) and skip_meta.get("skip_reason") is not None:
                            skip_reason = str(skip_meta["skip_reason"]).strip() or "unknown_skip"
                        skip_reason_counter[skip_reason] += 1
                        continue
                    if correction["corr_image"] is None or correction["corr_qpos_norm"] is None or correction["corr_action_chunk_norm"] is None:
                        skip_reason_counter["missing_correction_targets"] += 1
                        continue

                    corr_qpos_raw = correction["corr_qpos_norm"] * qpos_std + qpos_mean
                    corr_action_raw = correction["corr_action_chunk_norm"] * action_std.view(1, -1) + action_mean.view(1, -1)
                    corr_image = correction["corr_image"]
                    if corr_image.ndim != 4 or int(corr_image.shape[0]) != 1:
                        raise ValueError(f"Expected single-camera correction image, got {tuple(corr_image.shape)}")

                    smolvla_batch = build_smolvla_batch_from_raw_sample(
                        latent_policy=latent_model,
                        preprocess=preprocess,
                        instruction_type=args.instruction_type,
                        task_name_full=task_name_full,
                        image_t=corr_image.detach().cpu(),
                        qpos_raw=corr_qpos_raw.detach().cpu(),
                        action_chunk_raw=corr_action_raw.detach().cpu(),
                        episode_id=episode_id,
                    )
                    correction_samples.append(smolvla_batch)
                    correction_actions.append(smolvla_batch["action"])
                    if correction["error_action_prefix_norm"] is not None:
                        correction_action_prefixes.append(correction["error_action_prefix_norm"][None, ...])
                    else:
                        correction_action_prefixes.append(smolvla_batch["action"][:, : int(args.prefix_steps), :])
                    teacher_image = corr_image.to(device=torch.device(args.device), dtype=torch.float32)
                    teacher_latent = teacher.encode_image(teacher_image)
                    correction_teacher_latents.append(teacher_latent)
                    rollout_teacher_latents.append(teacher_latent)

                if not correction_samples:
                    continue

                correction_batch = stack_smolvla_batches(correction_samples)
                correction_actions_tensor = torch.cat(correction_actions, dim=0)
                correction_action_prefix_tensor = torch.cat(correction_action_prefixes, dim=0).to(torch.device(args.device))
                correction_teacher_latent_tensor = torch.cat(correction_teacher_latents, dim=0)
                rollout_teacher_latent_tensor = torch.cat(rollout_teacher_latents, dim=0)

                retain_weight_cur = compute_retain_weight(
                    retain_weight=float(args.retain_weight),
                    retain_weight_final=float(args.retain_weight_final),
                    retain_decay_start_step=float(args.retain_decay_start_step),
                    retain_decay_end_step=float(args.retain_decay_end_step),
                    retain_decay_curve=str(args.retain_decay_curve),
                    step_progress=float(global_step),
                )

                output = model(
                    train_stage="stage2",
                    normal_batch=normal_batch,
                    correction_batch=correction_batch,
                    correction_actions=correction_actions_tensor,
                    correction_action_prefix=correction_action_prefix_tensor,
                    correction_teacher_latent=correction_teacher_latent_tensor,
                    rollout_teacher_latent=rollout_teacher_latent_tensor,
                    global_step=global_step,
                    retain_weight=retain_weight_cur,
                )
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(output.loss)
                if optimizer_cfg.grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), optimizer_cfg.grad_clip_norm)
                optimizer.step()
                lr_scheduler.step()

                current_lr = float(optimizer.param_groups[0]["lr"])
                if tb_writer is not None:
                    log_stage2_tensorboard(
                        writer=tb_writer,
                        step=global_step,
                        output=output,
                        lr=current_lr,
                        retain_weight=retain_weight_cur,
                        skipped_batches=sum(skip_reason_counter.values()),
                    )

                progress.set_postfix(
                    loss=f"{output.loss.item():.4f}",
                    correct=f"{output.loss_correct.item():.4f}",
                    retain=f"{output.loss_retain.item():.4f}",
                    dyn=f"{output.loss_dynamics.item():.4f}",
                    beta=f"{output.beta_dynamics:.4f}",
                    rw=f"{retain_weight_cur:.4f}",
                    lr=f"{current_lr:.2e}",
                    skip=(skip_reason_counter.most_common(1)[0][0] if skip_reason_counter else "none"),
                    step=global_step,
                )
                global_step += 1
                progress.update(1)

                if global_step % int(args.save_freq) == 0 and is_main:
                    save_training_checkpoint(
                        output_dir=output_dir,
                        prefix="stage2",
                        step=global_step,
                        model=latent_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        extra_state={
                            "norm_stats": norm_stats,
                            "target_hw": latent_model.target_hw,
                            "failure_table_paths": failure_table_paths,
                            "epoch": epoch,
                            "args": vars(args),
                        },
                    )
                if global_step >= max_steps:
                    break
            epoch += 1
        if global_step % int(args.save_freq) != 0 and is_main:
            save_training_checkpoint(
                output_dir=output_dir,
                prefix="stage2",
                step=global_step,
                model=latent_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                extra_state={
                    "norm_stats": norm_stats,
                    "target_hw": latent_model.target_hw,
                    "failure_table_paths": failure_table_paths,
                    "epoch": epoch,
                    "args": vars(args),
                },
            )
    finally:
        progress.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


def main() -> None:
    args = build_argparser().parse_args()
    if args.train_mode == "stage1":
        run_stage1(args)
        return
    if args.train_mode == "stage2":
        run_stage2(args)
        return
    raise ValueError(f"Unsupported train mode: {args.train_mode}")


if __name__ == "__main__":
    main()
