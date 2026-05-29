from __future__ import annotations

import argparse
import json
import math
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
from .evac_interface import EvacLatentTeacher
from .stage1_checkpoint import build_stage1_model_from_checkpoint
from .stage2_failure_dataset import build_failure_table_dataset
from .train_stage1 import cleanup_distributed, init_distributed_if_needed, is_main_process, set_seed, str2bool
from .utils_latent import get_norm_stats, load_episode_prompt, load_raw_episode, resolve_raw_data_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train PI0 LatentCorr Stage 2")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--base-anchor-ckpt", default=None)
    parser.add_argument("--evac-ckpt", required=True)
    parser.add_argument("--evac-config", required=True)
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--train-config-name", default="pi05_aloha_full_base")
    parser.add_argument("--assets-base-dir", default=str(DEFAULT_ASSETS_BASE_DIR))
    parser.add_argument("--camera-mode", default="head_only", choices=["head_only", "tri_view"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--correction-batch-size", type=int, default=2)
    parser.add_argument("--reference-global-batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--prompt-mode", default="first", choices=["first", "random"])
    parser.add_argument("--prefix-steps", type=int, default=16)
    parser.add_argument("--future-offset", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--ddim-steps", type=int, default=27)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight-final", type=float, default=0.1)
    parser.add_argument("--retain-decay-start-epoch", type=float, default=0.0)
    parser.add_argument("--retain-decay-end-epoch", type=float, default=100.0)
    parser.add_argument("--retain-decay-curve", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--bridge-weight", type=float, default=0.0)
    parser.add_argument("--use-pi0-head-correction", type=str2bool, default=True)
    parser.add_argument("--failure-mode", default="train", choices=["off", "train"])
    parser.add_argument("--failure-table-path", default="")
    parser.add_argument("--failure-phase-bins", type=int, default=3)
    parser.add_argument("--failure-translation-dir-bins", type=int, default=6)
    parser.add_argument("--failure-translation-mag-bins", type=int, default=3)
    parser.add_argument("--failure-rotation-dir-bins", type=int, default=6)
    parser.add_argument("--failure-rotation-mag-bins", type=int, default=3)
    parser.add_argument("--failure-explore-k", type=int, default=1)
    parser.add_argument("--sample-phase-window-len", type=int, default=30)
    parser.add_argument("--sample-skip-head-ratio", type=float, default=0.6)
    parser.add_argument("--urdf-path", required=True)
    parser.add_argument(
        "--curobo-left-yml",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml",
    )
    parser.add_argument(
        "--curobo-right-yml",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml",
    )
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
    return parser


class PI0NormAdapter:
    def __init__(self, repo_id: str, train_config_name: str, assets_base_dir: str | Path, model_action_dim: int):
        prepare_openpi_imports()
        from openpi.shared import normalize as _normalize

        stats_root = Path(assets_base_dir).expanduser().resolve() / train_config_name / repo_id
        norm_stats = _normalize.load(stats_root)
        self.state_mean = torch.as_tensor(norm_stats["state"].mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(norm_stats["state"].std, dtype=torch.float32)
        self.action_mean = torch.as_tensor(norm_stats["actions"].mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(norm_stats["actions"].std, dtype=torch.float32)
        self.model_action_dim = int(model_action_dim)

    def to(self, device: torch.device) -> "PI0NormAdapter":
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        self.action_mean = self.action_mean.to(device)
        self.action_std = self.action_std.to(device)
        return self

    def normalize_state_raw(self, qpos_raw: torch.Tensor) -> torch.Tensor:
        if qpos_raw.ndim == 1:
            qpos_raw = qpos_raw.unsqueeze(0)
        return (qpos_raw[..., : self.state_mean.numel()] - self.state_mean) / self.state_std

    def normalize_action_raw(self, action_raw: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if action_raw.ndim == 2:
            action_raw = action_raw.unsqueeze(0)
            squeeze_back = True
        action_norm14 = (
            action_raw[..., : self.action_mean.numel()] - self.action_mean.view(1, 1, -1)
        ) / self.action_std.view(1, 1, -1)
        padded = torch.zeros(
            action_norm14.shape[0],
            action_norm14.shape[1],
            self.model_action_dim,
            dtype=action_norm14.dtype,
            device=action_norm14.device,
        )
        padded[..., : action_norm14.shape[-1]] = action_norm14
        return padded[0] if squeeze_back else padded

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        if action_norm.ndim == 2:
            return action_norm[..., : self.action_mean.numel()] * self.action_std + self.action_mean
        return (
            action_norm[..., : self.action_mean.numel()] * self.action_std.view(1, 1, -1)
            + self.action_mean.view(1, 1, -1)
        )

    def build_action_mask(self, is_pad: torch.Tensor) -> torch.Tensor:
        if is_pad.ndim == 1:
            is_pad = is_pad.unsqueeze(0)
        mask = torch.zeros(
            is_pad.shape[0],
            is_pad.shape[1],
            self.model_action_dim,
            dtype=torch.float32,
            device=is_pad.device,
        )
        mask[..., : self.action_mean.numel()] = (~is_pad).unsqueeze(-1).float()
        return mask


class PI0CorrectionPolicyAdapter:
    def __init__(self, latent_model, pi0_norm: PI0NormAdapter, act_stats: dict[str, np.ndarray], device: torch.device):
        self.latent_model = latent_model
        self.pi0_norm = pi0_norm
        self.device = device
        self.act_action_mean = torch.as_tensor(act_stats["action_mean"], dtype=torch.float32, device=device)
        self.act_action_std = torch.as_tensor(act_stats["action_std"], dtype=torch.float32, device=device)

    @property
    def training(self) -> bool:
        return bool(self.latent_model.training)

    def eval(self):
        self.latent_model.eval()
        return self

    def train(self, mode: bool = True):
        self.latent_model.train(mode)
        return self

    def predict_act_chunk(self, qpos_norm: torch.Tensor, image_t: torch.Tensor) -> torch.Tensor:
        if qpos_norm.ndim == 1:
            qpos_norm = qpos_norm.unsqueeze(0)
        if image_t.ndim == 3:
            image_t = image_t.unsqueeze(0)
        with torch.no_grad():
            pred = self.latent_model.predict_pi0_chunk(
                qpos_norm=qpos_norm.to(self.device),
                image_t=image_t.to(self.device),
                prompts=[""] * qpos_norm.shape[0],
                num_steps=10,
            )
        pred_raw = self.pi0_norm.denormalize_action(pred)
        return (pred_raw - self.act_action_mean.view(1, 1, -1)) / self.act_action_std.view(1, 1, -1)


def current_camera_names(camera_mode: str) -> list[str]:
    if camera_mode == "head_only":
        return ["cam_high"]
    return ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def select_head_image(image_t: torch.Tensor) -> torch.Tensor:
    if image_t.ndim == 5:
        return image_t[:, 0]
    if image_t.ndim == 4:
        return image_t
    raise ValueError(f"Unsupported image_t shape: {tuple(image_t.shape)}")


def compute_retain_weight(args: argparse.Namespace, epoch_progress: float) -> float:
    start = float(args.retain_decay_start_epoch)
    end = float(args.retain_decay_end_epoch)
    initial = float(args.retain_weight)
    final = float(args.retain_weight_final)
    if end <= start:
        return initial
    if epoch_progress <= start:
        return initial
    if epoch_progress >= end:
        return final
    ratio = (epoch_progress - start) / max(1e-8, end - start)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if args.retain_decay_curve == "cosine":
        ratio = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return initial + (final - initial) * ratio


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def save_checkpoint(
    output_dir: Path,
    step: int,
    sample_count: int,
    model: DDP | torch.nn.Module,
    optimizer: AdamW,
    args: argparse.Namespace,
) -> Path:
    raw_model = model.module if isinstance(model, DDP) else model
    payload = {
        "step": int(step),
        "sample_count": int(sample_count),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    ckpt_path = output_dir / f"stage2_step_{step:07d}.pt"
    latest_path = output_dir / "latest.pt"
    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)
    return ckpt_path


def main() -> None:
    args = build_argparser().parse_args()
    distributed, rank, world_size, resolved_device = init_distributed_if_needed(args)
    args.device = resolved_device
    if args.teacher_device is None:
        args.teacher_device = resolved_device
    device = torch.device(args.device)
    set_seed(args.seed)
    prepare_openpi_imports()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "stage2_metrics.jsonl"
    heartbeat_path = output_dir / "heartbeat.json"

    model, ckpt_args, _ = build_stage1_model_from_checkpoint(args.stage1_ckpt, device=device)
    model.train()
    if ckpt_args.get("freeze_base_pi0", False):
        model.base_pi0.eval()

    anchor_model, _, _ = build_stage1_model_from_checkpoint(args.base_anchor_ckpt or args.stage1_ckpt, device=device)
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=args.teacher_device)
    raw_data_dir = resolve_raw_data_dir(args.task_name, args.raw_data_dir)
    camera_names = current_camera_names(args.camera_mode)
    act_stats = get_norm_stats(args.processed_dir, args.num_episodes)
    pi0_norm = PI0NormAdapter(
        repo_id=args.repo_id,
        train_config_name=args.train_config_name,
        assets_base_dir=args.assets_base_dir,
        model_action_dim=int(model.model_action_dim),
    ).to(device)

    base_dataset, _ = build_failure_table_dataset(
        dataset_dir=args.processed_dir,
        num_episodes=args.num_episodes,
        camera_names=camera_names,
        act_chunk_size=args.action_horizon,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        sample_phase_window_len=args.sample_phase_window_len,
        sample_skip_head_ratio=args.sample_skip_head_ratio,
        start_margin=0,
        failure_mode="off",
        failure_table_path="",
        failure_phase_bins=args.failure_phase_bins,
        failure_translation_dir_bins=args.failure_translation_dir_bins,
        failure_translation_mag_bins=args.failure_translation_mag_bins,
        failure_rotation_dir_bins=args.failure_rotation_dir_bins,
        failure_rotation_mag_bins=args.failure_rotation_mag_bins,
        failure_explore_k=1,
    )
    corr_dataset = None
    if args.failure_mode != "off" and args.correction_batch_size > 0:
        corr_dataset, _ = build_failure_table_dataset(
            dataset_dir=args.processed_dir,
            num_episodes=args.num_episodes,
            camera_names=camera_names,
            act_chunk_size=args.action_horizon,
            prefix_steps=args.prefix_steps,
            future_offset=args.future_offset,
            sample_phase_window_len=args.sample_phase_window_len,
            sample_skip_head_ratio=args.sample_skip_head_ratio,
            start_margin=0,
            failure_mode=args.failure_mode,
            failure_table_path=args.failure_table_path,
            failure_phase_bins=args.failure_phase_bins,
            failure_translation_dir_bins=args.failure_translation_dir_bins,
            failure_translation_mag_bins=args.failure_translation_mag_bins,
            failure_rotation_dir_bins=args.failure_rotation_dir_bins,
            failure_rotation_mag_bins=args.failure_rotation_mag_bins,
            failure_explore_k=args.failure_explore_k,
        )

    base_sampler = (
        DistributedSampler(base_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    corr_sampler = (
        DistributedSampler(corr_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if (distributed and corr_dataset is not None)
        else None
    )
    base_loader = DataLoader(
        base_dataset,
        batch_size=args.batch_size,
        sampler=base_sampler,
        shuffle=(base_sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    corr_loader = None
    if corr_dataset is not None:
        corr_loader = DataLoader(
            corr_dataset,
            batch_size=args.correction_batch_size,
            sampler=corr_sampler,
            shuffle=(corr_sampler is None),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    bootstrap = next(iter(base_loader))
    with torch.no_grad():
        model.initialize_latent_heads(select_head_image(bootstrap["image_t"]).to(device), teacher)

    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=args.action_horizon),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    global_step = 0
    sample_count = 0
    raw_cache: dict[int, dict[str, Any]] = {}

    try:
        for epoch in range(args.num_epochs):
            if base_sampler is not None:
                base_sampler.set_epoch(epoch)
            if corr_sampler is not None:
                corr_sampler.set_epoch(epoch + 997)
            corr_iter = iter(corr_loader) if corr_loader is not None else None

            for batch_idx, base_batch in enumerate(base_loader):
                tic = time.perf_counter()
                raw_model = model.module if isinstance(model, DDP) else model
                if ckpt_args.get("freeze_base_pi0", False):
                    raw_model.base_pi0.eval()

                batch_groups: list[tuple[str, dict[str, Any]]] = [("off", base_batch)]
                if corr_iter is not None:
                    try:
                        corr_batch = next(corr_iter)
                    except StopIteration:
                        corr_iter = iter(corr_loader)
                        corr_batch = next(corr_iter)
                    batch_groups.append((args.failure_mode, corr_batch))

                prepared_samples = []
                corr_policy = PI0CorrectionPolicyAdapter(raw_model, pi0_norm, act_stats, device)

                for batch_mode, batch in batch_groups:
                    image_batch = select_head_image(batch["image_t"]).to(device, non_blocking=True)
                    qpos_raw_batch = batch["qpos_raw"].to(device, non_blocking=True).float()
                    act_chunk_raw_batch = batch["act_action_chunk_raw"].to(device, non_blocking=True).float()
                    act_is_pad_batch = batch["act_is_pad"].to(device, non_blocking=True).bool()
                    qpos_norm_batch = pi0_norm.normalize_state_raw(qpos_raw_batch)
                    act_chunk_batch = pi0_norm.normalize_action_raw(act_chunk_raw_batch)
                    action_mask_batch = pi0_norm.build_action_mask(act_is_pad_batch)
                    episode_ids = batch["episode_id"].tolist()
                    start_ts_list = batch["start_ts"].tolist()

                    with torch.no_grad():
                        prompts_batch = [
                            load_episode_prompt(args.processed_dir, int(ep_id), args.prompt_mode)
                            for ep_id in episode_ids
                        ]
                        pred_chunk_batch = raw_model.predict_pi0_chunk(
                            qpos_norm=qpos_norm_batch,
                            image_t=image_batch,
                            prompts=prompts_batch,
                            num_steps=10,
                        )
                        anchor_chunk_batch = anchor_model.predict_pi0_chunk(
                            qpos_norm=qpos_norm_batch,
                            image_t=image_batch,
                            prompts=prompts_batch,
                            num_steps=10,
                        )

                    for i in range(image_batch.shape[0]):
                        episode_id = int(episode_ids[i])
                        if episode_id not in raw_cache:
                            raw_cache[episode_id] = load_raw_episode(raw_data_dir, episode_id)
                        raw_data = raw_cache[episode_id]
                        prompt = load_episode_prompt(args.processed_dir, episode_id, args.prompt_mode)

                        if batch_mode == "off":
                            pred_prefix_norm = pred_chunk_batch[i, : args.prefix_steps].detach()
                            pred_prefix_raw = pi0_norm.denormalize_action(pred_prefix_norm).detach()
                            pred_qpos_err_norm = pi0_norm.normalize_state_raw(pred_prefix_raw[-1]).detach()[0]
                            prepared_samples.append(
                                {
                                    "image_t": image_batch[i].detach(),
                                    "qpos_t_norm": qpos_norm_batch[i].detach(),
                                    "qpos_raw": qpos_raw_batch[i].detach(),
                                    "pi0_action_mask": action_mask_batch[i].detach(),
                                    "base_anchor_chunk": anchor_chunk_batch[i].detach(),
                                    "correction_target_chunk": act_chunk_batch[i].detach(),
                                    "correction_is_pad": act_is_pad_batch[i].detach(),
                                    "correction_target_prefix": act_chunk_batch[i, : args.prefix_steps].detach(),
                                    "correction_is_pad_prefix": act_is_pad_batch[i, : args.prefix_steps].detach(),
                                    "external_action_dev_norm": pred_prefix_norm,
                                    "external_action_dev_raw": pred_prefix_raw,
                                    "external_qpos_err_norm": pred_qpos_err_norm,
                                    "external_action_is_pad_prefix": act_is_pad_batch[i, : args.prefix_steps].detach(),
                                    "raw_data": raw_data,
                                    "prompt": prompt,
                                }
                            )
                            continue

                        corr = correction_builder.build(
                            latent_model=corr_policy,
                            image_t=image_batch[i],
                            qpos_t=qpos_norm_batch[i],
                            raw_data=raw_data,
                            norm_stats=act_stats,
                            start_ts=int(start_ts_list[i]),
                            failure_mode_override=batch_mode,
                            sampled_phase_id=int(batch["sampled_phase_id"][i].item()) if "sampled_phase_id" in batch else -1,
                            pregrasp_seg_start=int(batch["pregrasp_seg_start"][i].item()) if "pregrasp_seg_start" in batch else -1,
                            pregrasp_seg_end=int(batch["pregrasp_seg_end"][i].item()) if "pregrasp_seg_end" in batch else -1,
                            sampled_phase_bin_id=int(batch["sampled_phase_bin_id"][i].item()) if "sampled_phase_bin_id" in batch else -1,
                            sampled_phase_instance_id=int(batch["sampled_phase_instance_id"][i].item()) if "sampled_phase_instance_id" in batch else -1,
                            forced_error_mode_id=int(batch["forced_error_mode_id"][i].item()) if "forced_error_mode_id" in batch else -1,
                            sampled_active_arm_pattern_id=int(batch["sampled_active_arm_pattern_id"][i].item()) if "sampled_active_arm_pattern_id" in batch else -1,
                            forced_dir_bin_id=int(batch["forced_dir_bin_id"][i].item()) if "forced_dir_bin_id" in batch else -1,
                            forced_mag_bin_id=int(batch["forced_mag_bin_id"][i].item()) if "forced_mag_bin_id" in batch else -1,
                            sampled_mode_prob=float(batch["sampled_mode_prob"][i].item()) if "sampled_mode_prob" in batch else float("nan"),
                            sampled_entry_prob_within_mode=float(batch["sampled_entry_prob_within_mode"][i].item()) if "sampled_entry_prob_within_mode" in batch else float("nan"),
                            sampled_unit_prob=float(batch["sampled_unit_prob"][i].item()) if "sampled_unit_prob" in batch else float("nan"),
                        )
                        if corr is None or corr.get("error_action_prefix_raw") is None:
                            continue

                        act_action_mean = torch.as_tensor(act_stats["action_mean"], dtype=torch.float32, device=device)
                        act_action_std = torch.as_tensor(act_stats["action_std"], dtype=torch.float32, device=device)
                        act_qpos_mean = torch.as_tensor(act_stats["qpos_mean"], dtype=torch.float32, device=device)
                        act_qpos_std = torch.as_tensor(act_stats["qpos_std"], dtype=torch.float32, device=device)

                        corr_action_raw = corr["corr_action_chunk_norm"].to(device).float() * act_action_std.view(1, -1) + act_action_mean.view(1, -1)
                        corr_qpos_raw = corr["corr_qpos_norm"].to(device).float() * act_qpos_std + act_qpos_mean
                        error_prefix_raw = corr["error_action_prefix_raw"].to(device).float()
                        error_prefix_pad = corr["error_is_pad_prefix"].to(device).bool() if corr.get("error_is_pad_prefix") is not None else corr["corr_is_pad"][: args.prefix_steps].to(device).bool()

                        prepared_samples.append(
                            {
                                "image_t": image_batch[i].detach(),
                                "qpos_t_norm": qpos_norm_batch[i].detach(),
                                "qpos_raw": qpos_raw_batch[i].detach(),
                                "pi0_action_mask": action_mask_batch[i].detach(),
                                "base_anchor_chunk": anchor_chunk_batch[i].detach(),
                                "correction_target_chunk": pi0_norm.normalize_action_raw(corr_action_raw).detach(),
                                "correction_is_pad": corr["corr_is_pad"].to(device).bool().detach(),
                                "correction_target_prefix": pi0_norm.normalize_action_raw(corr_action_raw).detach()[: args.prefix_steps],
                                "correction_is_pad_prefix": corr["corr_is_pad"].to(device).bool().detach()[: args.prefix_steps],
                                "external_action_dev_norm": pi0_norm.normalize_action_raw(error_prefix_raw).detach(),
                                "external_action_dev_raw": error_prefix_raw.detach(),
                                "external_qpos_err_norm": pi0_norm.normalize_state_raw(corr_qpos_raw).detach()[0],
                                "external_action_is_pad_prefix": error_prefix_pad.detach(),
                                "raw_data": raw_data,
                                "prompt": prompt,
                            }
                        )

                if not prepared_samples:
                    continue

                image_t = torch.stack([x["image_t"] for x in prepared_samples], dim=0)
                qpos_t_norm = torch.stack([x["qpos_t_norm"] for x in prepared_samples], dim=0)
                qpos_raw = torch.stack([x["qpos_raw"] for x in prepared_samples], dim=0)
                pi0_action_mask = torch.stack([x["pi0_action_mask"] for x in prepared_samples], dim=0)
                base_anchor_chunk = torch.stack([x["base_anchor_chunk"] for x in prepared_samples], dim=0)
                correction_target_chunk = torch.stack([x["correction_target_chunk"] for x in prepared_samples], dim=0)
                correction_is_pad = torch.stack([x["correction_is_pad"] for x in prepared_samples], dim=0)
                correction_target_prefix = torch.stack([x["correction_target_prefix"] for x in prepared_samples], dim=0)
                correction_is_pad_prefix = torch.stack([x["correction_is_pad_prefix"] for x in prepared_samples], dim=0)
                external_action_dev_norm = torch.stack(
                    [
                        x["external_action_dev_norm"] if x["external_action_dev_norm"] is not None else x["correction_target_prefix"]
                        for x in prepared_samples
                    ],
                    dim=0,
                )
                external_action_dev_raw = torch.stack(
                    [
                        x["external_action_dev_raw"]
                        if x["external_action_dev_raw"] is not None
                        else pi0_norm.denormalize_action(x["correction_target_prefix"])
                        for x in prepared_samples
                    ],
                    dim=0,
                )
                external_qpos_err_norm = torch.stack(
                    [
                        x["external_qpos_err_norm"] if x["external_qpos_err_norm"] is not None else x["qpos_t_norm"]
                        for x in prepared_samples
                    ],
                    dim=0,
                )
                external_action_is_pad_prefix = torch.stack([x["external_action_is_pad_prefix"] for x in prepared_samples], dim=0)
                prompts = [x["prompt"] for x in prepared_samples]
                raw_data_list = [x["raw_data"] for x in prepared_samples]

                with torch.no_grad():
                    rollout_latent = teacher.rollout_latent_from_actions_batch(
                        curr_image=image_t,
                        curr_qpos_raw=qpos_raw,
                        action_prefix_raw=external_action_dev_raw[..., :14],
                        raw_data=raw_data_list,
                        fk=correction_builder.fk,
                        ddim_steps=args.ddim_steps,
                    ).to(device)

                epoch_progress = epoch + (batch_idx / max(1, len(base_loader)))
                retain_weight_cur = compute_retain_weight(args, epoch_progress)
                out = model(
                    image_t=image_t,
                    qpos_t_norm=qpos_t_norm,
                    qpos_raw=qpos_raw,
                    pi0_action_mask=pi0_action_mask,
                    correction_target_prefix=correction_target_prefix,
                    is_pad_prefix=correction_is_pad_prefix,
                    prompts=prompts,
                    wm_teacher=teacher,
                    raw_data=None,
                    norm_stats=act_stats,
                    global_step=epoch_progress,
                    ddim_steps=args.ddim_steps,
                    retain_weight=retain_weight_cur,
                    bridge_weight=args.bridge_weight,
                    use_pi0_head_correction=args.use_pi0_head_correction,
                    base_anchor_chunk=base_anchor_chunk,
                    correction_target_chunk=correction_target_chunk,
                    correction_is_pad=correction_is_pad,
                    external_action_dev_norm=external_action_dev_norm,
                    external_action_dev_raw=external_action_dev_raw,
                    external_qpos_err_norm=external_qpos_err_norm,
                    external_action_is_pad_prefix=external_action_is_pad_prefix,
                    external_z_wm_rollout=rollout_latent,
                    fk=correction_builder.fk,
                    num_steps=10,
                    mode="stage2",
                )

                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()

                global_step += 1
                local_samples = len(prepared_samples)
                if distributed:
                    count_tensor = torch.tensor([local_samples], device=device, dtype=torch.long)
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                    sample_count += int(count_tensor.item())
                else:
                    sample_count += local_samples

                step_time = time.perf_counter() - tic
                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(out.loss.item()),
                    "loss_correct": float(out.loss_correct.item()),
                    "loss_dynamics": float(out.loss_dynamics.item()),
                    "loss_retain": float(out.loss_retain.item()),
                    "loss_bridge": float(out.loss_bridge.item()),
                    "beta_dynamics": float(out.beta_dynamics),
                    "alpha_latent": float(out.alpha_latent),
                    "retain_weight_current": float(retain_weight_cur),
                    "prepared_samples": int(local_samples),
                    "sample_count": int(sample_count),
                    "step_time_sec": float(step_time),
                }

                if is_main_process(rank) and (global_step % args.log_every == 0):
                    append_jsonl(metrics_path, metrics)
                    write_json(
                        heartbeat_path,
                        {
                            "status": "running",
                            "step": global_step,
                            "epoch": epoch,
                            "sample_count": sample_count,
                            "last_metrics": metrics,
                            "timestamp": time.time(),
                        },
                    )
                    print(
                        f"[stage2] step={global_step} loss={metrics['loss']:.4f} "
                        f"corr={metrics['loss_correct']:.4f} dyn={metrics['loss_dynamics']:.4f} "
                        f"retain={metrics['loss_retain']:.4f} t={metrics['step_time_sec']:.2f}s",
                        flush=True,
                    )

                if is_main_process(rank) and (global_step % args.save_every == 0):
                    ckpt = save_checkpoint(output_dir, global_step, sample_count, model, optimizer, args)
                    print(f"[stage2] saved checkpoint: {ckpt}", flush=True)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        if is_main_process(rank):
            save_checkpoint(output_dir, global_step, sample_count, model, optimizer, args)
            write_json(
                heartbeat_path,
                {
                    "status": "finished",
                    "step": global_step,
                    "sample_count": sample_count,
                    "timestamp": time.time(),
                },
            )
        cleanup_distributed(use_barrier=False)


if __name__ == "__main__":
    main()
