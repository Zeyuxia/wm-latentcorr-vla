from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .act_aligned_correction import ACTAlignedCorrectionBuilder, build_act_aligned_cfg_from_args
from .common import DEFAULT_ASSETS_BASE_DIR, prepare_openpi_imports
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .multitask_failure_dataset import build_multitask_failure_table_dataset
from .multitask_utils import resolve_multitask_specs
from .stage1_model import PI0LatentStage1, Stage1LossOutput
from .stage1_multitask_dataset import build_multitask_stage1_dataset
from .train_stage1 import (
    append_jsonl_file,
    cleanup_distributed,
    init_distributed_if_needed,
    is_main_process,
    load_resume_checkpoint,
    resolve_base_pi0_weight_path,
    resolve_resume_path,
    rotate_existing_file,
    save_checkpoint,
    set_seed,
    split_batch_device,
    str2bool,
    write_json_file,
)
from .train_stage2_latent import PI0CorrectionPolicyAdapter, PI0NormAdapter
from .utils_latent import load_episode_prompt, load_raw_episode


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train PI0.5 unified multitask stage1 with normal+failure batches")
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
    parser.add_argument("--raw-data-dirs", nargs="*", default=None)
    parser.add_argument("--failure-table-paths", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal-batch-size", type=int, default=4)
    parser.add_argument("--failure-batch-size", type=int, default=2)
    parser.add_argument("--reference-global-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--prefix-steps", type=int, default=16)
    parser.add_argument("--future-offset", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--projector-mid-channels", type=int, default=256)
    parser.add_argument("--predictor-hidden-dim", type=int, default=512)
    parser.add_argument("--predictor-num-blocks", type=int, default=3)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-action-conditioned", type=float, default=0.5)
    parser.add_argument("--schedule-action-conditioned", type=str2bool, default=True)
    parser.add_argument("--lambda-align", type=float, default=0.0)
    parser.add_argument("--beta-dynamics-max", type=float, default=1.0)
    parser.add_argument("--dyn-zero-steps", type=int, default=0)
    parser.add_argument("--dyn-ramp-steps", type=int, default=2000)
    parser.add_argument("--dyn-warmup-curve", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--dyn-schedule-unit", default="step", choices=["step", "epoch"])
    parser.add_argument("--freeze-base-pi0", type=str2bool, default=False)
    parser.add_argument("--use-act-head-conditioning", type=str2bool, default=True)
    parser.add_argument("--use-projector-detach-for-predictor", type=str2bool, default=True)
    parser.add_argument("--detach-act-feature-for-latent", type=str2bool, default=False)
    parser.add_argument("--use-raw-wm-targets", type=str2bool, default=True)
    parser.add_argument("--prompt-mode", default="random", choices=["random", "first"])
    parser.add_argument("--normal-samples-per-epoch", type=int, default=None)
    parser.add_argument("--failure-samples-per-epoch", type=int, default=None)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--failure-loss-weight", type=float, default=1.0)
    parser.add_argument("--urdf-path", required=True)
    parser.add_argument(
        "--curobo-left-yml",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml",
    )
    parser.add_argument(
        "--curobo-right-yml",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml",
    )
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
    parser.add_argument("--recover-eval-enable", type=str2bool, default=False)
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
    parser.add_argument("--failure-phase-bins", type=int, default=3)
    parser.add_argument("--failure-translation-dir-bins", type=int, default=6)
    parser.add_argument("--failure-translation-mag-bins", type=int, default=3)
    parser.add_argument("--failure-rotation-dir-bins", type=int, default=6)
    parser.add_argument("--failure-rotation-mag-bins", type=int, default=3)
    parser.add_argument("--failure-explore-k", type=int, default=1)
    parser.add_argument("--sample-phase-window-len", type=int, default=30)
    parser.add_argument("--sample-skip-head-ratio", type=float, default=0.6)
    return parser


def select_head_image(image_t: torch.Tensor) -> torch.Tensor:
    if image_t.ndim == 5:
        return image_t[:, 0]
    if image_t.ndim == 4:
        return image_t
    if image_t.ndim == 3:
        return image_t
    raise ValueError(f"Unsupported image_t shape: {tuple(image_t.shape)}")


def build_action_mask_from_is_pad(is_pad: torch.Tensor, model_action_dim: int, action_dim: int = 14) -> torch.Tensor:
    if is_pad.ndim == 1:
        is_pad = is_pad.unsqueeze(0)
    mask = torch.zeros(
        is_pad.shape[0],
        is_pad.shape[1],
        model_action_dim,
        dtype=torch.float32,
        device=is_pad.device,
    )
    mask[..., :action_dim] = (~is_pad).unsqueeze(-1).float()
    return mask


def mean_stage1_outputs(outputs: list[Stage1LossOutput]) -> Stage1LossOutput:
    if not outputs:
        raise ValueError("mean_stage1_outputs expects at least one output")

    def _stack_mean(name: str) -> torch.Tensor:
        return torch.stack([getattr(out, name) for out in outputs]).mean()

    return Stage1LossOutput(
        loss=_stack_mean("loss"),
        loss_action=_stack_mean("loss_action"),
        loss_action_conditioned=_stack_mean("loss_action_conditioned"),
        loss_dynamics=_stack_mean("loss_dynamics"),
        loss_align=_stack_mean("loss_align"),
        beta_dynamics=float(np.mean([out.beta_dynamics for out in outputs])),
        alpha_latent=float(np.mean([out.alpha_latent for out in outputs])),
    )


def schedule_condition_weight(args: argparse.Namespace, raw_model: PI0LatentStage1, schedule_step: float) -> float:
    base = float(args.lambda_action_conditioned)
    if not args.schedule_action_conditioned:
        return base
    return base * float(raw_model.warmup.weight(float(schedule_step)))


def build_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PI0LatentStage1, Path | None, dict[str, str | None]]:
    prepare_openpi_imports()
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training import config as openpi_config

    train_cfg = openpi_config.get_config(args.train_config_name)
    pytorch_weight_path, init_meta = resolve_base_pi0_weight_path(args)
    if pytorch_weight_path is not None:
        base_pi0 = train_cfg.model.load_pytorch(train_cfg, str(pytorch_weight_path))
    else:
        base_pi0 = PI0Pytorch(config=train_cfg.model)
    base_pi0 = base_pi0.to(device)

    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=args.projector_mid_channels,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_num_blocks=args.predictor_num_blocks,
        action_dim=14,
        model_action_dim=args.model_action_dim,
        action_horizon=args.action_horizon,
        prefix_steps=args.prefix_steps,
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=args.lambda_action,
        lambda_action_conditioned=args.lambda_action_conditioned,
        lambda_align=0.0,
        beta_dynamics_max=args.beta_dynamics_max,
        use_projector_detach_for_predictor=args.use_projector_detach_for_predictor,
        detach_act_feature_for_latent=args.detach_act_feature_for_latent,
        use_raw_wm_targets=args.use_raw_wm_targets,
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=args.dyn_zero_steps,
        ramp_steps=args.dyn_ramp_steps,
        max_weight=1.0,
        curve=args.dyn_warmup_curve,
        unit=args.dyn_schedule_unit,
    )
    model = PI0LatentStage1(
        base_pi0=base_pi0,
        tokenizer=PaligemmaTokenizer(max_len=train_cfg.model.max_token_len),
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
        discrete_state_input=bool(train_cfg.model.discrete_state_input),
        freeze_base_pi0=args.freeze_base_pi0,
    ).to(device)
    return model, pytorch_weight_path, init_meta


def prepare_failure_batch(
    *,
    batch: dict[str, Any],
    raw_model: PI0LatentStage1,
    correction_builder: ACTAlignedCorrectionBuilder,
    teacher: EvacLatentTeacher,
    task_norm_stats: dict[str, dict[str, Any]],
    pi0_norm_by_task: dict[str, PI0NormAdapter],
    raw_cache: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, list[str], int]:
    prepared = {
        "image_t": [],
        "image_t1": [],
        "qpos_t_norm": [],
        "qpos_t1_norm": [],
        "act_action_chunk": [],
        "act_action_mask": [],
        "action_prefix": [],
        "is_pad_prefix": [],
        "future_teacher_latent": [],
    }
    prompts: list[str] = []
    skipped = 0
    image_batch = select_head_image(batch["image_t"]).to(device, non_blocking=True)

    for i in range(int(image_batch.shape[0])):
        task_name = str(batch["task_name"][i])
        processed_dir = str(batch["processed_dir"][i])
        raw_data_dir = str(batch["raw_data_dir"][i])
        episode_id = int(batch["episode_id"][i])
        start_ts = int(batch["start_ts"][i])
        norm_stats = task_norm_stats[task_name]
        pi0_norm = pi0_norm_by_task[task_name]
        corr_policy = PI0CorrectionPolicyAdapter(raw_model, pi0_norm, norm_stats, device)
        cache_key = (task_name, episode_id)
        if cache_key not in raw_cache:
            raw_cache[cache_key] = load_raw_episode(raw_data_dir, episode_id)

        corr = correction_builder.build(
            latent_model=corr_policy,
            image_t=image_batch[i],
            qpos_t=batch["qpos_t"][i].to(device, non_blocking=True),
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
            sampled_mode_prob=float(batch["sampled_mode_prob"][i]),
            sampled_entry_prob_within_mode=float(batch["sampled_entry_prob_within_mode"][i]),
            sampled_unit_prob=float(batch["sampled_unit_prob"][i]),
        )
        if corr is None:
            skipped += 1
            continue

        corr_image = corr.get("corr_image")
        corr_qpos_act_norm = corr.get("corr_qpos_norm")
        corr_action_act_norm = corr.get("corr_action_chunk_norm")
        corr_is_pad = corr.get("corr_is_pad")
        if corr_image is None or corr_qpos_act_norm is None or corr_action_act_norm is None or corr_is_pad is None:
            skipped += 1
            continue

        act_qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=device)
        act_qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=device)
        act_action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=device)
        act_action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=device)

        corr_qpos_raw = corr_qpos_act_norm.to(device=device, dtype=torch.float32) * act_qpos_std + act_qpos_mean
        corr_action_raw = (
            corr_action_act_norm.to(device=device, dtype=torch.float32) * act_action_std.view(1, -1)
            + act_action_mean.view(1, -1)
        )

        corr_qpos_pi_norm = pi0_norm.normalize_state_raw(corr_qpos_raw)[0]
        corr_action_pi_norm = pi0_norm.normalize_action_raw(corr_action_raw)[0]
        corr_is_pad = corr_is_pad.to(device=device).bool()
        corr_action_mask = build_action_mask_from_is_pad(
            corr_is_pad,
            model_action_dim=int(raw_model.model_action_dim),
            action_dim=14,
        )[0]
        future_teacher_latent = teacher.encode_image(corr_image.unsqueeze(0)).to(device=device, dtype=torch.float32)[0]

        prepared["image_t"].append(corr_image.to(device=device, dtype=torch.float32))
        prepared["image_t1"].append(corr_image.to(device=device, dtype=torch.float32))
        prepared["qpos_t_norm"].append(corr_qpos_pi_norm)
        prepared["qpos_t1_norm"].append(corr_qpos_pi_norm)
        prepared["act_action_chunk"].append(corr_action_pi_norm)
        prepared["act_action_mask"].append(corr_action_mask)
        prepared["action_prefix"].append(corr_action_pi_norm[: args.prefix_steps])
        prepared["is_pad_prefix"].append(corr_is_pad[: args.prefix_steps])
        prepared["future_teacher_latent"].append(future_teacher_latent)
        prompts.append(load_episode_prompt(processed_dir, episode_id, args.prompt_mode))

    if not prepared["image_t"]:
        return None, [], skipped

    stacked = {key: torch.stack(value, dim=0) for key, value in prepared.items()}
    return stacked, prompts, skipped


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_path = output_dir / "stage1_unified_metrics.jsonl"
    summary_path = output_dir / "stage1_unified_summary.json"
    launch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    distributed, rank, world_size, resolved_device = init_distributed_if_needed(args)
    args.device = resolved_device
    if args.teacher_device is None:
        args.teacher_device = resolved_device
    device = torch.device(args.device)
    set_seed(args.seed)
    prepare_openpi_imports()

    task_specs = resolve_multitask_specs(
        task_names=list(args.multi_task_names),
        processed_dirs=list(args.processed_dirs),
        repo_ids=list(args.repo_ids),
        raw_data_dirs=args.raw_data_dirs,
        camera_mode=args.camera_mode,
    )
    if len(task_specs) != len(args.failure_table_paths):
        raise ValueError("failure_table_paths must have the same length as multi_task_names")

    model, pytorch_weight_path, init_meta = build_model(args, device)
    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=(args.teacher_device or args.device))

    normal_dataset, _ = build_multitask_stage1_dataset(
        task_specs=task_specs,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        action_horizon=args.action_horizon,
        model_action_dim=args.model_action_dim,
        prompt_mode=args.prompt_mode,
        samples_per_epoch=args.normal_samples_per_epoch,
        train_config_name=args.train_config_name,
        assets_base_dir=args.assets_base_dir,
    )
    failure_dataset, failure_norm_stats = build_multitask_failure_table_dataset(
        task_specs=task_specs,
        failure_table_paths=list(args.failure_table_paths),
        act_chunk_size=args.action_horizon,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        sample_phase_window_len=args.sample_phase_window_len,
        sample_skip_head_ratio=args.sample_skip_head_ratio,
        start_margin=0,
        failure_mode="train",
        failure_phase_bins=args.failure_phase_bins,
        failure_translation_dir_bins=args.failure_translation_dir_bins,
        failure_translation_mag_bins=args.failure_translation_mag_bins,
        failure_rotation_dir_bins=args.failure_rotation_dir_bins,
        failure_rotation_mag_bins=args.failure_rotation_mag_bins,
        failure_explore_k=args.failure_explore_k,
        samples_per_epoch=args.failure_samples_per_epoch,
    )

    normal_sampler = None
    failure_sampler = None
    if distributed:
        normal_sampler = DistributedSampler(
            normal_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed
        )
        failure_sampler = DistributedSampler(
            failure_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed + 997
        )

    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.normal_batch_size,
        sampler=normal_sampler,
        shuffle=(normal_sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    failure_loader = DataLoader(
        failure_dataset,
        batch_size=args.failure_batch_size,
        sampler=failure_sampler,
        shuffle=(failure_sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    bootstrap_batch = next(iter(normal_loader))
    bootstrap_prompts, bootstrap_tensors = split_batch_device(bootstrap_batch, device)
    with torch.no_grad():
        model.forward_stage1(
            image_t=bootstrap_tensors["image_t"],
            image_t1=bootstrap_tensors["image_t1"],
            qpos_t_norm=bootstrap_tensors["qpos_t_norm"],
            qpos_t1_norm=bootstrap_tensors["qpos_t1_norm"],
            act_action_chunk=bootstrap_tensors["act_action_chunk"],
            act_action_mask=bootstrap_tensors["act_action_mask"],
            action_prefix=bootstrap_tensors["action_prefix"],
            is_pad_prefix=bootstrap_tensors["is_pad_prefix"].bool(),
            prompts=bootstrap_prompts,
            wm_teacher=teacher,
            global_step=0,
            use_act_head_conditioning=args.use_act_head_conditioning,
            condition_on_current_observation=True,
        )

    if distributed:
        model = DDP(
            model,
            device_ids=[int(args.device.split(":")[-1])],
            output_device=int(args.device.split(":")[-1]),
            find_unused_parameters=True,
        )

    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    resume_path = resolve_resume_path(args, output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        args_payload = vars(args).copy()
        args_payload["resolved_pytorch_weight_path"] = None if pytorch_weight_path is None else str(pytorch_weight_path)
        args_payload["resolved_base_pi0_init_source"] = init_meta["init_source"]
        args_payload["resolved_base_pi0_checkpoint_dir"] = init_meta["resolved_checkpoint_dir"]
        args_payload["world_size"] = world_size
        args_payload["resolved_resume_path"] = None if resume_path is None else str(resume_path)
        rotate_existing_file(metrics_path, launch_id)
        write_json_file(output_dir / "args.json", args_payload)
    if distributed:
        dist.barrier()

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

    reference_global_batch_size = int(args.reference_global_batch_size or (args.normal_batch_size + args.failure_batch_size))
    global_step = 0
    sample_count = 0
    if resume_path is not None:
        global_step, sample_count = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            resume_path=resume_path,
            rank=rank,
            reference_global_batch_size=reference_global_batch_size,
            distributed=distributed,
        )

    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}

    try:
        model.train()
        failure_iter = iter(failure_loader)
        for epoch in range(args.num_epochs):
            if normal_sampler is not None:
                normal_sampler.set_epoch(epoch)
            if failure_sampler is not None:
                failure_sampler.set_epoch(epoch + 997)
            for normal_batch in normal_loader:
                t0 = time.perf_counter()
                raw_model = model.module if isinstance(model, DDP) else model
                if raw_model.latent_loss_cfg.lambda_align != 0.0:
                    raw_model.latent_loss_cfg.lambda_align = 0.0
                if raw_model._freeze_base_pi0:
                    raw_model.base_pi0.eval()

                try:
                    failure_batch = next(failure_iter)
                except StopIteration:
                    failure_iter = iter(failure_loader)
                    failure_batch = next(failure_iter)

                normal_prompts, normal_tensors = split_batch_device(normal_batch, device)
                current_global_batch_size = int(normal_tensors["image_t"].shape[0] + args.failure_batch_size) * world_size
                schedule_step = sample_count / float(max(1, reference_global_batch_size))
                if args.dyn_schedule_unit == "epoch":
                    schedule_step = float(epoch)
                cond_weight = schedule_condition_weight(args, raw_model, schedule_step)

                normal_out = model(
                    image_t=normal_tensors["image_t"],
                    image_t1=normal_tensors["image_t1"],
                    qpos_t_norm=normal_tensors["qpos_t_norm"],
                    qpos_t1_norm=normal_tensors["qpos_t1_norm"],
                    act_action_chunk=normal_tensors["act_action_chunk"],
                    act_action_mask=normal_tensors["act_action_mask"],
                    action_prefix=normal_tensors["action_prefix"],
                    is_pad_prefix=normal_tensors["is_pad_prefix"].bool(),
                    prompts=normal_prompts,
                    wm_teacher=teacher,
                    global_step=schedule_step,
                    use_act_head_conditioning=args.use_act_head_conditioning,
                    condition_on_current_observation=True,
                    lambda_action_conditioned_override=cond_weight,
                )

                failure_prepared, failure_prompts, failure_skipped = prepare_failure_batch(
                    batch=failure_batch,
                    raw_model=raw_model,
                    correction_builder=correction_builder,
                    teacher=teacher,
                    task_norm_stats=failure_norm_stats,
                    pi0_norm_by_task=pi0_norm_by_task,
                    raw_cache=raw_cache,
                    args=args,
                    device=device,
                )

                if failure_prepared is not None:
                    failure_out = model(
                        image_t=failure_prepared["image_t"],
                        image_t1=failure_prepared["image_t1"],
                        qpos_t_norm=failure_prepared["qpos_t_norm"],
                        qpos_t1_norm=failure_prepared["qpos_t1_norm"],
                        act_action_chunk=failure_prepared["act_action_chunk"],
                        act_action_mask=failure_prepared["act_action_mask"],
                        action_prefix=failure_prepared["action_prefix"],
                        is_pad_prefix=failure_prepared["is_pad_prefix"].bool(),
                        prompts=failure_prompts,
                        wm_teacher=teacher,
                        global_step=schedule_step,
                        use_act_head_conditioning=args.use_act_head_conditioning,
                        future_teacher_latent=failure_prepared["future_teacher_latent"],
                        condition_on_current_observation=True,
                        lambda_action_conditioned_override=cond_weight,
                    )
                    total_loss = normal_out.loss + float(args.failure_loss_weight) * failure_out.loss
                else:
                    failure_out = None
                    total_loss = normal_out.loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                global_step += 1
                sample_count += current_global_batch_size
                elapsed = time.perf_counter() - t0

                if is_main_process(rank) and (
                    global_step == 1 or global_step % int(args.log_every) == 0
                ):
                    payload = {
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "sample_count": int(sample_count),
                        "loss": float(total_loss.detach().item()),
                        "loss_normal": float(normal_out.loss.detach().item()),
                        "loss_normal_action": float(normal_out.loss_action.detach().item()),
                        "loss_normal_conditioned": float(normal_out.loss_action_conditioned.detach().item()),
                        "loss_normal_dynamics": float(normal_out.loss_dynamics.detach().item()),
                        "loss_failure": None if failure_out is None else float(failure_out.loss.detach().item()),
                        "loss_failure_action": None if failure_out is None else float(failure_out.loss_action.detach().item()),
                        "loss_failure_conditioned": None if failure_out is None else float(failure_out.loss_action_conditioned.detach().item()),
                        "loss_failure_dynamics": None if failure_out is None else float(failure_out.loss_dynamics.detach().item()),
                        "condition_weight": float(cond_weight),
                        "beta_dynamics": float(normal_out.beta_dynamics),
                        "alpha_latent": float(normal_out.alpha_latent),
                        "failure_skipped": int(failure_skipped),
                        "failure_kept": 0 if failure_prepared is None else int(failure_prepared["image_t"].shape[0]),
                        "step_time_sec": float(elapsed),
                    }
                    append_jsonl_file(metrics_path, payload)
                    print(
                        "[pi05-unified] "
                        f"step={global_step} loss={payload['loss']:.4f} "
                        f"normal={payload['loss_normal']:.4f} "
                        f"failure={payload['loss_failure'] if payload['loss_failure'] is not None else 'nan'} "
                        f"cond_w={payload['condition_weight']:.4f} "
                        f"skip={failure_skipped} "
                        f"time={elapsed:.2f}s",
                        flush=True,
                    )

                if is_main_process(rank) and (
                    global_step == 1 or global_step % int(args.save_every) == 0
                ):
                    save_checkpoint(output_dir, global_step, sample_count, model, optimizer, args)

                if args.max_steps > 0 and global_step >= int(args.max_steps):
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        if is_main_process(rank):
            save_checkpoint(output_dir, global_step, sample_count, model, optimizer, args)
            summary = {
                "step": int(global_step),
                "sample_count": int(sample_count),
                "finished_at": datetime.datetime.now().isoformat(),
            }
            write_json_file(summary_path, summary)
        cleanup_distributed(use_barrier=False)


if __name__ == "__main__":
    main()
