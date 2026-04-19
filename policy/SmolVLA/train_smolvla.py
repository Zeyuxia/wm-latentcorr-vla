from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SMOLVLA_SRC_DIR = THIS_DIR / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.SmolVLA.act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from policy.SmolVLA.correction_policy_adapter import SampleBoundSmolVLAAdapter
from policy.SmolVLA.evac_interface import EvacLatentTeacher
from policy.SmolVLA.failure_manifest_utils import load_failure_table_paths
from policy.SmolVLA.latent_config import DynamicsWarmupConfig
from policy.SmolVLA.latent_dataset_utils import load_raw_episode
from policy.SmolVLA.multitask_failure_dataset import MultiTaskFailureDatasetConfig, build_multitask_failure_dataset
from policy.SmolVLA.multitask_latent_utils import build_multitask_stage1_dataset, resolve_multitask_specs
from policy.SmolVLA.smolvla_data_utils import build_smolvla_batch, make_smolvla_processors, stack_smolvla_batches
from policy.SmolVLA.smolvla_latent_policy import SmolVLALatentBridgeConfig, SmolVLALatentPolicy


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


def compute_retain_weight(
    retain_weight: float,
    retain_weight_final: float,
    retain_decay_start_epoch: float,
    retain_decay_end_epoch: float,
    retain_decay_curve: str,
    epoch_progress: float,
) -> float:
    if retain_decay_end_epoch <= retain_decay_start_epoch:
        return retain_weight
    if epoch_progress <= retain_decay_start_epoch:
        return retain_weight
    if epoch_progress >= retain_decay_end_epoch:
        return retain_weight_final
    ratio = (epoch_progress - retain_decay_start_epoch) / max(1e-8, retain_decay_end_epoch - retain_decay_start_epoch)
    ratio = max(0.0, min(1.0, ratio))
    if retain_decay_curve == "cosine":
        ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return retain_weight + (retain_weight_final - retain_weight) * ratio


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smolvla_pretrained_path", type=str, required=True)
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
    parser.add_argument("--num_epochs", type=int, required=True)
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
    parser.add_argument("--retain_decay_start_epoch", type=float, required=True)
    parser.add_argument("--retain_decay_end_epoch", type=float, required=True)
    parser.add_argument("--retain_decay_curve", type=str, required=True, choices=["linear", "cosine"])
    parser.add_argument("--failure_phase_bins", type=int, required=True)
    parser.add_argument("--failure_translation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_translation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_dir_bins", type=int, required=True)
    parser.add_argument("--failure_rotation_mag_bins", type=int, required=True)
    parser.add_argument("--failure_explore_k", type=int, required=True)
    parser.add_argument("--sample_phase_window_len", type=int, required=True)
    parser.add_argument("--sample_skip_head_ratio", type=float, required=True)
    parser.add_argument("--start_margin", type=int, required=True)
    parser.add_argument("--max_rollout_steps", type=int, required=True)
    parser.add_argument("--act_aligned_rollout_exec_steps", type=int, required=True)
    parser.add_argument("--planner_target_mode", type=str, required=True)
    parser.add_argument("--planner_target_lookahead_steps", type=int, required=True)
    parser.add_argument("--planner_orient_weight", type=float, required=True)
    parser.add_argument("--planner_gripper_penalty", type=float, required=True)
    parser.add_argument("--planner_nearest_window_radius", type=int, required=True)
    parser.add_argument("--planner_active_joint_delta_thresh", type=float, required=True)
    parser.add_argument("--planner_active_gripper_delta_thresh", type=float, required=True)
    parser.add_argument("--act_aligned_min_dist_fallback_force_correction", type=str2bool, required=True)
    parser.add_argument("--act_aligned_min_dist_recover_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_real_error_trigger_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_real_error_min_dist_thresh", type=float, required=True)
    parser.add_argument("--act_aligned_real_error_min_dist_delta_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_enable", type=str2bool, required=True)
    parser.add_argument("--recover_eval_save_video", type=str2bool, required=True)
    parser.add_argument("--recover_eval_gripper_open_thresh", type=float, required=True)
    parser.add_argument("--recover_eval_pos_thresh_m", type=float, required=True)
    parser.add_argument("--recover_eval_rot_thresh_deg", type=float, required=True)
    parser.add_argument("--recover_eval_nearest_window_radius", type=int, required=True)
    parser.add_argument("--recover_eval_video_bridge_steps", type=int, required=True)
    parser.add_argument("--act_aligned_correction_interp_nearest_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_correction_interp_prefix_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_correction_planner_prefix_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_correction_gripper_close_prefix_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_correction_compose_gt_tail_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_correction_gripper_switch_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_recover_gripper_penalty", type=float, required=True)
    parser.add_argument("--act_aligned_enable_perturb", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_error_mode", type=str, required=True)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_close_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_translation_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_open_laptop_pregrasp_rotation_prob", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_eef_fail_gain", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_rot_max_deg", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_mag_random", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_mag_rand_min", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_mag_rand_max", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_reject_sampling_enable", type=str2bool, required=True)
    parser.add_argument("--act_aligned_perturb_reject_max_trials", type=int, required=True)
    parser.add_argument("--act_aligned_perturb_reject_dir_jitter_eps", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_close_min", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_open_max", type=float, required=True)
    parser.add_argument("--act_aligned_perturb_gripper_fast_ratio", type=float, required=True)
    parser.add_argument("--act_aligned_sample_pregrasp_phase_window_len", type=int, required=True)
    parser.add_argument("--act_aligned_sample_timeout_sec", type=float, required=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("train smolvla latentcorr")
    subparsers = parser.add_subparsers(dest="train_mode", required=True)

    stage1_parser = subparsers.add_parser("stage1")
    add_common_args(stage1_parser)

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
    preprocess, _ = make_smolvla_processors(base_policy, args.smolvla_pretrained_path)
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


def save_train_args(output_dir: Path, args: argparse.Namespace) -> None:
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)


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
    latent_policy.initialize_from_batch(init_batch)


def run_stage1(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

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
        pin_memory=True,
        drop_last=True,
    )

    base_policy, preprocess = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_warmup_config(args),
    )
    model.to(torch.device(args.device))
    teacher = EvacLatentTeacher(evac_ckpt=args.evac_ckpt, evac_config=args.evac_config, device=args.device)

    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=dataset[0],
    )
    optimizer_cfg, optimizer, lr_scheduler = configure_optimizer(
        model=model,
        args=args,
        total_training_steps=int(args.num_epochs) * max(1, len(dataloader)),
    )
    save_train_args(output_dir, args)

    global_step = 0
    for epoch in range(int(args.num_epochs)):
        progress = tqdm(dataloader, desc=f"stage1 epoch {epoch}", leave=True)
        for raw_batch in progress:
            samples = []
            future_images = []
            action_prefix = []
            for batch_index in range(raw_batch["image_t"].shape[0]):
                task_name = raw_batch["task_name"][batch_index]
                task_only, task_config = parse_task_parts(task_name)
                sample = build_smolvla_batch(
                    policy=model.base_policy,
                    preprocess=preprocess,
                    image_t=raw_batch["image_t"][batch_index],
                    qpos_raw=raw_batch["qpos_raw"][batch_index],
                    action_chunk_raw=raw_batch["act_action_chunk_raw"][batch_index],
                    task_name=task_only,
                    task_config=task_config,
                    episode_id=int(raw_batch["episode_id"][batch_index].item()),
                    instruction_type=args.instruction_type,
                )
                samples.append(sample)
                future_images.append(raw_batch["image_t_future"][batch_index, 0])
                action_prefix.append(sample["action"][:, : int(args.prefix_steps), :])

            batch = stack_smolvla_batches(samples)
            future_image_tensor = torch.stack(future_images, dim=0).to(device=torch.device(args.device), dtype=torch.float32)
            future_teacher_latent = teacher.encode_image(future_image_tensor)
            action_prefix_tensor = torch.cat(action_prefix, dim=0)

            output = model.compute_stage1_loss(
                batch=batch,
                future_teacher_latent=future_teacher_latent,
                action_prefix=action_prefix_tensor,
                global_step=global_step,
            )
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            clip_grad_norm_(model.parameters(), optimizer_cfg.grad_clip_norm)
            optimizer.step()
            lr_scheduler.step()

            progress.set_postfix(
                loss=f"{output.loss.item():.4f}",
                action=f"{output.loss_action.item():.4f}",
                cond=f"{output.loss_action_conditioned.item():.4f}",
                dyn=f"{output.loss_dynamics.item():.4f}",
                beta=f"{output.beta_dynamics:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                step=global_step,
            )
            global_step += 1

        if (epoch + 1) % int(args.save_freq) == 0:
            checkpoint_path = output_dir / f"stage1_epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "norm_stats": dataset.stats if hasattr(dataset, "stats") else None,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "args": vars(args),
                },
                checkpoint_path,
            )


def run_stage2(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

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
        sample_skip_head_ratio=float(args.sample_skip_head_ratio),
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
    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=normal_dataset[0],
    )
    stage1_checkpoint = torch.load(args.stage1_ckpt, map_location="cpu")
    model.load_state_dict(stage1_checkpoint["model"], strict=True)
    model.to(torch.device(args.device))

    teacher = EvacLatentTeacher(evac_ckpt=args.evac_ckpt, evac_config=args.evac_config, device=args.device)
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
        total_training_steps=int(args.num_epochs) * max(1, len(normal_loader)),
    )
    save_train_args(output_dir, args)

    qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=args.device)
    qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=args.device)
    action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=args.device)
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}

    correction_iter = iter(correction_loader)
    global_step = 0
    skip_reason_counter: Counter[str] = Counter()
    for epoch in range(int(args.num_epochs)):
        progress = tqdm(normal_loader, desc=f"stage2 epoch {epoch}", leave=True)
        total_batches = max(1, len(normal_loader))
        for batch_idx, normal_raw_batch in enumerate(progress):
            try:
                correction_raw_batch = next(correction_iter)
            except StopIteration:
                correction_iter = iter(correction_loader)
                correction_raw_batch = next(correction_iter)

            normal_samples = []
            for sample_index in range(normal_raw_batch["image_t"].shape[0]):
                normal_samples.append(
                    build_smolvla_batch_from_raw_sample(
                        latent_policy=model,
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
                    latent_policy=model,
                    preprocess=preprocess,
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
                    pregrasp_seg_start=int(correction_raw_batch["pregrasp_seg_start"][sample_index].item()),
                    pregrasp_seg_end=int(correction_raw_batch["pregrasp_seg_end"][sample_index].item()),
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
                    latent_policy=model,
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

            epoch_progress = epoch + (batch_idx / float(total_batches))
            retain_weight_cur = compute_retain_weight(
                retain_weight=float(args.retain_weight),
                retain_weight_final=float(args.retain_weight_final),
                retain_decay_start_epoch=float(args.retain_decay_start_epoch),
                retain_decay_end_epoch=float(args.retain_decay_end_epoch),
                retain_decay_curve=str(args.retain_decay_curve),
                epoch_progress=epoch_progress,
            )

            output = model.compute_stage2_loss(
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
            output.loss.backward()
            clip_grad_norm_(model.parameters(), optimizer_cfg.grad_clip_norm)
            optimizer.step()
            lr_scheduler.step()

            progress.set_postfix(
                loss=f"{output.loss.item():.4f}",
                correct=f"{output.loss_correct.item():.4f}",
                retain=f"{output.loss_retain.item():.4f}",
                dyn=f"{output.loss_dynamics.item():.4f}",
                beta=f"{output.beta_dynamics:.4f}",
                rw=f"{retain_weight_cur:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                skip=(skip_reason_counter.most_common(1)[0][0] if skip_reason_counter else "none"),
                step=global_step,
            )
            global_step += 1

        if (epoch + 1) % int(args.save_freq) == 0:
            checkpoint_path = output_dir / f"stage2_epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "norm_stats": norm_stats,
                    "failure_table_paths": failure_table_paths,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "args": vars(args),
                },
                checkpoint_path,
            )


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
