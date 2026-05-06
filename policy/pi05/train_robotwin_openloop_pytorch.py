from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import jax
import torch
import wandb
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel

try:
    from .robotwin_common import (
        DEFAULT_ASSETS_BASE_DIR,
        DEFAULT_CAMERA_MODE,
        DEFAULT_CHECKPOINT_BASE_DIR,
        DEFAULT_SECONDARY_CAMERA,
        build_train_config,
        ensure_norm_stats,
        prepare_openpi_imports,
        timestamp,
    )
except ImportError:
    from policy.pi05.robotwin_common import (
        DEFAULT_ASSETS_BASE_DIR,
        DEFAULT_CAMERA_MODE,
        DEFAULT_CHECKPOINT_BASE_DIR,
        DEFAULT_SECONDARY_CAMERA,
        build_train_config,
        ensure_norm_stats,
        prepare_openpi_imports,
        timestamp,
    )


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train PI0.5 open-loop policy with PyTorch for RoboTwin")
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--train-config-name", default="pi05_aloha_full_base")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--camera-mode", default=DEFAULT_CAMERA_MODE, choices=["head_only", "dual_view", "tri_view"])
    parser.add_argument(
        "--secondary-camera",
        default=DEFAULT_SECONDARY_CAMERA,
        choices=["left_wrist", "right_wrist"],
        help="Used only when camera-mode=dual_view.",
    )
    parser.add_argument("--exp-name", default=f"pi05_openloop_{timestamp()}")
    parser.add_argument("--assets-base-dir", default=str(DEFAULT_ASSETS_BASE_DIR))
    parser.add_argument("--checkpoint-base-dir", default=str(DEFAULT_CHECKPOINT_BASE_DIR))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-period", type=int, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--wandb-enabled", type=str2bool, default=True)
    parser.add_argument("--project-name", default="RoboTwin_PI05")
    parser.add_argument("--pytorch-weight-path", required=True)
    parser.add_argument("--pytorch-training-precision", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--gradient-checkpointing", type=str2bool, default=True)
    parser.add_argument("--find-unused-parameters", type=str2bool, default=True)
    parser.add_argument("--skip-norm-stats", type=str2bool, default=False)
    parser.add_argument("--norm-max-frames", type=int, default=None)
    return parser


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_distributed() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_local_rank()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if is_dist() and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return device


def cleanup_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def dist_barrier(device: torch.device) -> None:
    if not is_dist():
        return
    if device.type == "cuda" and device.index is not None:
        torch.distributed.barrier(device_ids=[device.index])
    else:
        torch.distributed.barrier()


def move_tree_to_device(tree: Any, device: torch.device) -> Any:
    def _move(x: Any) -> Any:
        if not hasattr(x, "to"):
            return x
        target = x.to(device=device, non_blocking=True)
        if torch.is_floating_point(target) and target.dtype == torch.float64:
            target = target.to(torch.float32)
        return target

    return jax.tree.map(
        _move,
        tree,
    )


def latest_step_dir(experiment_dir: Path) -> Path | None:
    candidates = sorted((path for path in experiment_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda p: int(p.name)) if experiment_dir.is_dir() else []
    return candidates[-1] if candidates else None


def resolve_asset_id(cfg: Any) -> str:
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Unable to resolve asset id for norm stats export.")
    return str(asset_id)


def copy_norm_stats(cfg: Any, checkpoint_dir: Path) -> None:
    asset_id = resolve_asset_id(cfg)
    src = cfg.assets_dirs / asset_id / "norm_stats.json"
    if not src.is_file():
        raise FileNotFoundError(f"Norm stats not found: {src}")
    dst = checkpoint_dir / "assets" / asset_id / "norm_stats.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def maybe_init_wandb(cfg: Any, enabled: bool) -> None:
    if not is_main_process():
        wandb.init(mode="disabled")
        return
    if not enabled:
        wandb.init(mode="disabled")
        return
    wandb.init(
        name=cfg.exp_name,
        config=dataclasses.asdict(cfg),
        project=cfg.project_name,
    )


def rank_log(device: torch.device, message: str) -> None:
    print(f"[openloop_pt][rank={get_rank()}][device={device}] {message}", flush=True)


def main_log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def cosine_lr(step: int, *, warmup_steps: int, peak_lr: float, decay_steps: int, decay_lr: float) -> float:
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        alpha = float(step + 1) / float(max(warmup_steps, 1))
        return init_lr + alpha * (peak_lr - init_lr)
    progress = min(max((step - warmup_steps) / float(max(decay_steps - warmup_steps, 1)), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return decay_lr + cosine * (peak_lr - decay_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_training_checkpoint(
    cfg: Any,
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    experiment_dir: Path,
    args: argparse.Namespace,
) -> Path:
    checkpoint_dir = experiment_dir / str(step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_model(raw_model, str(checkpoint_dir / "model.safetensors"))
    copy_norm_stats(cfg, checkpoint_dir)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        checkpoint_dir / "training_state.pt",
    )
    with (experiment_dir / "latest_step.txt").open("w", encoding="utf-8") as f:
        f.write(str(step))
    return checkpoint_dir


def main() -> None:
    args = build_argparser().parse_args()
    prepare_openpi_imports()
    setup_logging()
    device = setup_distributed()
    rank_log(device, "process group ready")

    cfg = build_train_config(
        train_config_name=args.train_config_name,
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        camera_mode=args.camera_mode,
        secondary_camera=args.secondary_camera,
        asset_id=args.asset_id,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        project_name=args.project_name,
        pytorch_weight_path=args.pytorch_weight_path,
        pytorch_training_precision=args.pytorch_training_precision,
    )

    if not args.skip_norm_stats and is_main_process():
        ensure_norm_stats(cfg, max_frames=args.norm_max_frames)
    dist_barrier(device)
    rank_log(device, "norm stats ready")

    if not Path(args.pytorch_weight_path).is_file():
        raise FileNotFoundError(f"PyTorch weight file not found: {args.pytorch_weight_path}")

    from openpi.training import data_loader as openpi_data_loader

    experiment_dir = cfg.checkpoint_dir
    if is_main_process():
        if args.overwrite and experiment_dir.exists() and not args.resume:
            shutil.rmtree(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with (experiment_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
    dist_barrier(device)

    data_loader = openpi_data_loader.create_data_loader(
        cfg,
        shuffle=True,
        num_workers=cfg.num_workers,
        framework="pytorch",
    )
    rank_log(device, "data loader created")
    data_iter = iter(data_loader)
    rank_log(device, "data iterator created")

    start_step = 0
    resume_dir: Path | None = None
    load_weight_path = Path(args.pytorch_weight_path)
    if args.resume:
        resume_dir = latest_step_dir(experiment_dir)
        if resume_dir is None:
            raise FileNotFoundError(f"No numeric checkpoint directories found under: {experiment_dir}")
        state_path = resume_dir / "training_state.pt"
        weight_path = resume_dir / "model.safetensors"
        if not state_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError(f"Incomplete resume checkpoint: {resume_dir}")
        load_weight_path = weight_path

    rank_log(device, f"loading pytorch weights from {load_weight_path}")
    model = cfg.model.load_pytorch(cfg, str(load_weight_path))
    rank_log(device, "pytorch weights loaded")
    model.paligemma_with_expert.to_bfloat16_for_selected_params(args.pytorch_training_precision)
    rank_log(device, f"precision set to {args.pytorch_training_precision}")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        rank_log(device, "gradient checkpointing enabled")
    else:
        model.gradient_checkpointing_disable()
        rank_log(device, "gradient checkpointing disabled")
    model.to(device)
    rank_log(device, "model moved to device")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-8,
        betas=(cfg.optimizer.b1, cfg.optimizer.b2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    if args.resume:
        assert resume_dir is not None
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        rank_log(device, f"resumed from step {start_step}")

    if is_dist():
        rank_log(device, "wrapping with DDP")
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=args.find_unused_parameters,
        )
        rank_log(device, "DDP ready")

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    maybe_init_wandb(cfg, enabled=args.wandb_enabled)

    logging.info("[PI05_RobotWin] starting PyTorch open-loop training")
    logging.info("[PI05_RobotWin] rank=%s world=%s device=%s", get_rank(), os.environ.get("WORLD_SIZE", "1"), device)
    logging.info("[PI05_RobotWin] train_config=%s", args.train_config_name)
    logging.info("[PI05_RobotWin] repo_id=%s", args.repo_id)
    logging.info("[PI05_RobotWin] exp_name=%s", args.exp_name)
    logging.info("[PI05_RobotWin] pytorch_weight_path=%s", Path(args.pytorch_weight_path).resolve())

    running_loss = 0.0
    running_grad_norm = 0.0
    running_steps = 0
    start_time = time.time()
    autocast_enabled = device.type == "cuda" and args.pytorch_training_precision == "bfloat16"

    for step in range(start_step, cfg.num_train_steps):
        if step == start_step:
            rank_log(device, "entering training loop")
        lr = cosine_lr(
            step,
            warmup_steps=cfg.lr_schedule.warmup_steps,
            peak_lr=cfg.lr_schedule.peak_lr,
            decay_steps=cfg.lr_schedule.decay_steps,
            decay_lr=cfg.lr_schedule.decay_lr,
        )
        set_optimizer_lr(optimizer, lr)

        observation, actions = next(data_iter)
        if step == start_step:
            rank_log(device, "first batch fetched")
        observation = move_tree_to_device(observation, device)
        actions = move_tree_to_device(actions, device)
        if step == start_step:
            rank_log(device, "first batch moved to device")

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss_tensor = model(observation, actions)
            loss = loss_tensor.mean()
        if step == start_step:
            rank_log(device, "first forward finished")
        loss.backward()
        if step == start_step:
            rank_log(device, "first backward finished")
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.optimizer.clip_gradient_norm)
        optimizer.step()
        if step == start_step:
            rank_log(device, "first optimizer step finished")

        running_loss += float(loss.detach().item())
        running_grad_norm += float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        running_steps += 1

        completed_step = step + 1
        if completed_step % cfg.log_interval == 0:
            avg_loss = running_loss / max(running_steps, 1)
            avg_grad = running_grad_norm / max(running_steps, 1)
            elapsed = time.time() - start_time
            if is_main_process():
                main_log(
                    "[openloop_pt] step=%d/%d loss=%.6f grad_norm=%.6f lr=%.8f elapsed=%.1fs"
                    % (
                        completed_step,
                        cfg.num_train_steps,
                        avg_loss,
                        avg_grad,
                        lr,
                        elapsed,
                    )
                )
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/grad_norm": avg_grad,
                        "train/lr": lr,
                        "train/step": completed_step,
                    },
                    step=completed_step,
                )
            running_loss = 0.0
            running_grad_norm = 0.0
            running_steps = 0

        should_save = completed_step % cfg.save_interval == 0 or completed_step == cfg.num_train_steps
        if should_save:
            dist_barrier(device)
            if is_main_process():
                checkpoint_dir = save_training_checkpoint(cfg, raw_model, optimizer, completed_step, experiment_dir, args)
                main_log(f"[openloop_pt] saved checkpoint -> {checkpoint_dir}")
            dist_barrier(device)

    if is_main_process():
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
