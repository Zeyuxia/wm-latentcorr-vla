from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .common import DEFAULT_ASSETS_BASE_DIR, prepare_openpi_imports
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .stage1_dataset import PI0Stage1Dataset
from .stage1_model import PI0LatentStage1


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
    parser = argparse.ArgumentParser("Train PI0 Stage-1 latent warmup")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evac-ckpt", required=True)
    parser.add_argument("--evac-config", required=True)
    parser.add_argument("--train-config-name", default="pi05_aloha_full_base")
    parser.add_argument("--camera-mode", default="head_only", choices=["head_only", "tri_view"])
    parser.add_argument("--assets-base-dir", default=str(DEFAULT_ASSETS_BASE_DIR))
    parser.add_argument("--pytorch-weight-path", default=None)
    parser.add_argument(
        "--task-checkpoint-dir",
        default=None,
        help="Task open-loop checkpoint directory or experiment directory containing model.safetensors.",
    )
    parser.add_argument(
        "--task-checkpoint-id",
        default="latest",
        help="Checkpoint id to use when --task-checkpoint-dir points to an experiment directory with numeric children.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reference-global-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=10)
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
    parser.add_argument("--lambda-action-conditioned", type=float, default=1.0)
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
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--use-wandb", type=str2bool, default=True)
    parser.add_argument("--wandb-project", default="RoboTwin_PI0_LatentCorr")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-mode", default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def cleanup_distributed(use_barrier: bool = True) -> None:
    if dist.is_available() and dist.is_initialized():
        if use_barrier:
            try:
                dist.barrier()
            except Exception:
                pass
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def default_norm_stats_path(assets_base_dir: str | Path, train_config_name: str, repo_id: str) -> Path:
    return Path(assets_base_dir).expanduser().resolve() / train_config_name / repo_id / "norm_stats.json"


def resolve_model_safetensors_from_dir(checkpoint_dir: Path, checkpoint_id: str | None = "latest") -> tuple[Path, Path]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    direct_model = checkpoint_dir / "model.safetensors"
    if direct_model.is_file():
        return direct_model.resolve(), checkpoint_dir

    ckpt_str = "" if checkpoint_id is None else str(checkpoint_id).strip()
    numeric_children = sorted(
        (path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ) if checkpoint_dir.is_dir() else []
    if numeric_children:
        resolved_ckpt_dir = numeric_children[-1] if ckpt_str in {"", "latest"} else checkpoint_dir / ckpt_str
        if not resolved_ckpt_dir.is_dir():
            raise FileNotFoundError(f"Task checkpoint id not found under {checkpoint_dir}: {ckpt_str}")
        model_path = resolved_ckpt_dir / "model.safetensors"
        if model_path.is_file():
            return model_path.resolve(), resolved_ckpt_dir.resolve()
        if (resolved_ckpt_dir / "params").is_dir():
            raise FileNotFoundError(
                "Task checkpoint directory contains OpenPI Orbax params but no model.safetensors: "
                f"{resolved_ckpt_dir}. Export this task open-loop checkpoint to a deployable PyTorch checkpoint first."
            )

    if (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(
            "Task checkpoint directory contains OpenPI Orbax params but no model.safetensors: "
            f"{checkpoint_dir}. Export this task open-loop checkpoint to a deployable PyTorch checkpoint first."
        )

    raise FileNotFoundError(f"No model.safetensors found under task checkpoint dir: {checkpoint_dir}")


def resolve_base_pi0_weight_path(args: argparse.Namespace) -> tuple[Path | None, dict[str, str | None]]:
    if args.pytorch_weight_path:
        candidate = Path(args.pytorch_weight_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"PyTorch weight file not found: {candidate}")
        return candidate, {
            "init_source": "explicit_pytorch_weight_path",
            "resolved_checkpoint_dir": None,
        }

    if args.task_checkpoint_dir:
        candidate = Path(args.task_checkpoint_dir).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Task checkpoint path not found: {candidate}")
        if candidate.is_file():
            if candidate.suffix != ".safetensors":
                raise FileNotFoundError(
                    f"Task checkpoint file must be a .safetensors weight file, got: {candidate}"
                )
            return candidate, {
                "init_source": "task_model_safetensors",
                "resolved_checkpoint_dir": None,
            }
        model_path, resolved_ckpt_dir = resolve_model_safetensors_from_dir(candidate, args.task_checkpoint_id)
        return model_path, {
            "init_source": "task_checkpoint_dir",
            "resolved_checkpoint_dir": str(resolved_ckpt_dir),
        }

    local_candidates = [
        Path(__file__).resolve().parent / "lerobot" / "pi05_base" / "model.safetensors",
        Path(__file__).resolve().parent / "lerobot" / "pi0_base" / "model.safetensors",
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.resolve(), {
                "init_source": "fallback_local_base",
                "resolved_checkpoint_dir": None,
            }
    return None, {
        "init_source": "config_init",
        "resolved_checkpoint_dir": None,
    }


def save_checkpoint(
    output_dir: Path,
    step: int,
    sample_count: int,
    model: PI0LatentStage1 | DDP,
    optimizer: AdamW,
    args: argparse.Namespace,
) -> Path:
    raw_model = model.module if isinstance(model, DDP) else model
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"stage1_step_{step:07d}.pt"
    torch.save(
        {
            "step": int(step),
            "sample_count": int(sample_count),
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    latest_path = output_dir / "latest.pt"
    torch.save(
        {
            "step": int(step),
            "sample_count": int(sample_count),
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        latest_path,
    )
    return ckpt_path


def split_batch_device(batch: dict[str, object], device: torch.device) -> tuple[list[str], dict[str, torch.Tensor]]:
    prompts = list(batch["prompt"])
    tensor_batch: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "prompt":
            continue
        tensor_batch[key] = value.to(device, non_blocking=True)
    return prompts, tensor_batch


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def append_jsonl_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")
        f.flush()


def rotate_existing_file(path: Path, launch_id: str) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(f"{path.stem}_{launch_id}.prev{path.suffix}")
    path.replace(backup_path)


def resolve_resume_path(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if not args.resume and not args.resume_path:
        return None
    if args.resume_path:
        path = Path(args.resume_path).expanduser().resolve()
    else:
        path = output_dir / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def broadcast_module_state(module: PI0LatentStage1, rank: int) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def load_resume_checkpoint(
    *,
    model: PI0LatentStage1 | DDP,
    optimizer: AdamW,
    resume_path: Path,
    rank: int,
    reference_global_batch_size: int,
    distributed: bool,
) -> tuple[int, int]:
    raw_model = model.module if isinstance(model, DDP) else model
    meta = [0, 0]

    if distributed:
        if is_main_process(rank):
            checkpoint = torch.load(resume_path, map_location="cpu")
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            ckpt_args = checkpoint.get("args", {})
            meta = [
                int(checkpoint.get("step", 0)),
                int(
                    checkpoint.get(
                        "sample_count",
                        int(checkpoint.get("step", 0)) * int(ckpt_args.get("reference_global_batch_size", reference_global_batch_size)),
                    )
                ),
            ]
            print(f"[stage1] resumed model weights from {resume_path}")
            print("[stage1] optimizer state is reinitialized for distributed resume")
        broadcast_module_state(raw_model, rank)
        dist.broadcast_object_list(meta, src=0)
        return int(meta[0]), int(meta[1])

    checkpoint = torch.load(resume_path, map_location="cpu")
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    ckpt_args = checkpoint.get("args", {})
    step = int(checkpoint.get("step", 0))
    sample_count = int(
        checkpoint.get(
            "sample_count",
            step * int(ckpt_args.get("reference_global_batch_size", reference_global_batch_size)),
        )
    )
    print(f"[stage1] resumed model/optimizer from {resume_path}")
    return step, sample_count


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_path = output_dir / "stage1_metrics.jsonl"
    summary_path = output_dir / "stage1_wandb_summary.json"
    status_path = output_dir / "stage1_wandb_status.json"
    manifest_path = output_dir / "stage1_wandb_manifest.json"
    had_error = False
    launch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_enabled = bool(args.use_wandb and args.wandb_mode != "disabled")
    prepare_openpi_imports()
    distributed, rank, world_size, resolved_device = init_distributed_if_needed(args)
    args.device = resolved_device
    if args.teacher_device is None:
        args.teacher_device = resolved_device
    set_seed(args.seed)

    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training import config as openpi_config
    from policy.ACT_LatentCorr.evac_interface import EvacLatentTeacher

    train_cfg = openpi_config.get_config(args.train_config_name)
    pytorch_weight_path, init_meta = resolve_base_pi0_weight_path(args)
    if pytorch_weight_path is not None:
        print(
            f"[stage1] loading base_pi0 weights from {pytorch_weight_path} "
            f"(source={init_meta['init_source']})"
        )
        if init_meta["init_source"] == "fallback_local_base":
            print("[stage1] warning: using generic local base weights, not a task open-loop checkpoint.")
        if init_meta["resolved_checkpoint_dir"]:
            print(f"[stage1] resolved task checkpoint dir: {init_meta['resolved_checkpoint_dir']}")
        base_pi0 = train_cfg.model.load_pytorch(train_cfg, str(pytorch_weight_path))
    else:
        print("[stage1] no local PyTorch base weights found, initializing PI0 model from config")
        base_pi0 = PI0Pytorch(config=train_cfg.model)

    device = torch.device(args.device)
    base_pi0 = base_pi0.to(device)

    norm_stats_path = default_norm_stats_path(args.assets_base_dir, args.train_config_name, args.repo_id)
    dataset = PI0Stage1Dataset(
        args.processed_dir,
        norm_stats_path,
        prefix_steps=args.prefix_steps,
        future_offset=args.future_offset,
        action_horizon=args.action_horizon,
        model_action_dim=args.model_action_dim,
        prompt_mode=args.prompt_mode,
        samples_per_epoch=args.samples_per_epoch,
        train_config_name=args.train_config_name,
        repo_id=args.repo_id,
        assets_base_dir=args.assets_base_dir,
        camera_mode=args.camera_mode,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

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
        lambda_align=args.lambda_align,
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

    teacher = EvacLatentTeacher(
        args.evac_ckpt,
        args.evac_config,
        device=(args.teacher_device or args.device),
    )

    bootstrap_batch = next(iter(dataloader))
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
        (output_dir / "args.json").write_text(json.dumps(args_payload, indent=2))
        if wandb_enabled:
            rotate_existing_file(metrics_path, launch_id)
            write_json_file(
                manifest_path,
                {
                    "launch_id": launch_id,
                    "project": args.wandb_project,
                    "entity": args.wandb_entity,
                    "run_name": args.wandb_run_name or f"stage1_{output_dir.name}",
                    "group": args.wandb_group or f"{args.train_config_name}_stage1",
                    "tags": args.wandb_tags or ["stage1", args.train_config_name, args.repo_id.replace("/", "_")],
                    "mode": args.wandb_mode,
                    "output_dir": str(output_dir),
                    "config": {
                        "stage": "stage1",
                        "args": args_payload,
                    },
                },
            )
            write_json_file(
                status_path,
                {
                    "state": "running",
                    "launch_id": launch_id,
                    "started_at": datetime.datetime.now().isoformat(),
                    "pid": os.getpid(),
                    "world_size": world_size,
                    "resume_path": None if resume_path is None else str(resume_path),
                },
            )
            print(f"[stage1] metrics -> {metrics_path}")
            print(f"[stage1] wandb manifest -> {manifest_path}")
    if distributed:
        dist.barrier()

    reference_global_batch_size = int(args.reference_global_batch_size or args.batch_size)
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
    try:
        model.train()
        for epoch in range(args.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            meter_loss = 0.0
            meter_action = 0.0
            meter_cond = 0.0
            meter_dyn = 0.0
            meter_align = 0.0
            steps_this_epoch = 0

            for batch in dataloader:
                prompts, tensor_batch = split_batch_device(batch, device)
                local_batch_size = int(tensor_batch["image_t"].shape[0])
                current_global_batch_size = local_batch_size * world_size
                schedule_step = sample_count / float(max(1, reference_global_batch_size))
                if args.dyn_schedule_unit == "epoch":
                    schedule_step = float(epoch)

                loss_out = model(
                    image_t=tensor_batch["image_t"],
                    image_t1=tensor_batch["image_t1"],
                    qpos_t_norm=tensor_batch["qpos_t_norm"],
                    qpos_t1_norm=tensor_batch["qpos_t1_norm"],
                    act_action_chunk=tensor_batch["act_action_chunk"],
                    act_action_mask=tensor_batch["act_action_mask"],
                    action_prefix=tensor_batch["action_prefix"],
                    is_pad_prefix=tensor_batch["is_pad_prefix"].bool(),
                    prompts=prompts,
                    wm_teacher=teacher,
                    global_step=schedule_step,
                    use_act_head_conditioning=args.use_act_head_conditioning,
                    mode="stage1",
                )

                optimizer.zero_grad(set_to_none=True)
                loss_out.loss.backward()
                optimizer.step()

                global_step += 1
                sample_count += current_global_batch_size
                meter_loss += loss_out.loss.item()
                meter_action += loss_out.loss_action.item()
                meter_cond += loss_out.loss_action_conditioned.item()
                meter_dyn += loss_out.loss_dynamics.item()
                meter_align += loss_out.loss_align.item()
                steps_this_epoch += 1

                if is_main_process(rank) and global_step % args.log_every == 0:
                    metric_payload = {
                        "train/loss": loss_out.loss.item(),
                        "train/loss_action": loss_out.loss_action.item(),
                        "train/loss_action_conditioned": loss_out.loss_action_conditioned.item(),
                        "train/loss_dynamics": loss_out.loss_dynamics.item(),
                        "train/loss_align": loss_out.loss_align.item(),
                        "train/beta_dynamics": loss_out.beta_dynamics,
                        "train/alpha_latent": loss_out.alpha_latent,
                        "train/epoch": epoch,
                        "train/sample_count": sample_count,
                        "train/reference_global_batch_size": reference_global_batch_size,
                        "train/world_size": world_size,
                    }
                    print(
                        f"[stage1] epoch={epoch} step={global_step} "
                        f"loss={loss_out.loss.item():.6f} "
                        f"action={loss_out.loss_action.item():.6f} "
                        f"cond={loss_out.loss_action_conditioned.item():.6f} "
                        f"dyn={loss_out.loss_dynamics.item():.6f} "
                        f"align={loss_out.loss_align.item():.6f} "
                        f"beta={loss_out.beta_dynamics:.4f} "
                        f"world={world_size}"
                    )
                    if wandb_enabled:
                        append_jsonl_file(
                            metrics_path,
                            {
                                "type": "metric",
                                "launch_id": launch_id,
                                "timestamp": datetime.datetime.now().isoformat(),
                                "step": global_step,
                                "metrics": metric_payload,
                            },
                        )

                if is_main_process(rank) and global_step % args.save_every == 0:
                    ckpt_path = save_checkpoint(output_dir, global_step, sample_count, model, optimizer, args)
                    print(f"[stage1] saved checkpoint: {ckpt_path}")
                    if wandb_enabled:
                        write_json_file(
                            summary_path,
                            {
                                "last_checkpoint": str(ckpt_path),
                                "latest_step": global_step,
                                "latest_sample_count": sample_count,
                                "updated_at": datetime.datetime.now().isoformat(),
                            },
                        )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            if distributed:
                stats = torch.tensor(
                    [meter_loss, meter_action, meter_cond, meter_dyn, meter_align, float(steps_this_epoch)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                denom = max(1.0, stats[-1].item())
                epoch_loss = stats[0].item() / denom
                epoch_action = stats[1].item() / denom
                epoch_cond = stats[2].item() / denom
                epoch_dyn = stats[3].item() / denom
                epoch_align = stats[4].item() / denom
            else:
                denom = max(1, steps_this_epoch)
                epoch_loss = meter_loss / denom
                epoch_action = meter_action / denom
                epoch_cond = meter_cond / denom
                epoch_dyn = meter_dyn / denom
                epoch_align = meter_align / denom

            if is_main_process(rank):
                epoch_payload = {
                    "epoch/avg_loss": epoch_loss,
                    "epoch/avg_action": epoch_action,
                    "epoch/avg_action_conditioned": epoch_cond,
                    "epoch/avg_dynamics": epoch_dyn,
                    "epoch/avg_align": epoch_align,
                    "epoch/index": epoch,
                    "epoch/sample_count": sample_count,
                }
                print(
                    f"[stage1][epoch={epoch}] avg_loss={epoch_loss:.6f} "
                    f"avg_action={epoch_action:.6f} "
                    f"avg_cond={epoch_cond:.6f} "
                    f"avg_dyn={epoch_dyn:.6f} "
                    f"avg_align={epoch_align:.6f}"
                )
                if wandb_enabled:
                    append_jsonl_file(
                        metrics_path,
                        {
                            "type": "metric",
                            "launch_id": launch_id,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "step": global_step,
                            "metrics": epoch_payload,
                        },
                    )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if is_main_process(rank):
            ckpt_path = save_checkpoint(output_dir, global_step, sample_count, model, optimizer, args)
            print(f"[stage1] finished, checkpoint: {ckpt_path}")
            if wandb_enabled:
                write_json_file(
                    summary_path,
                    {
                        "final_checkpoint": str(ckpt_path),
                        "final_step": global_step,
                        "final_sample_count": sample_count,
                        "updated_at": datetime.datetime.now().isoformat(),
                    },
                )
                write_json_file(
                    status_path,
                    {
                        "state": "completed",
                        "launch_id": launch_id,
                        "finished_at": datetime.datetime.now().isoformat(),
                        "final_checkpoint": str(ckpt_path),
                        "final_step": global_step,
                        "final_sample_count": sample_count,
                    },
                )
    except Exception as exc:
        had_error = True
        error_text = traceback.format_exc()
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / f"stage1_exception_rank{rank}.log"
        error_path.write_text(error_text, encoding="utf-8")
        print(error_text, file=sys.stderr, flush=True)
        if is_main_process(rank):
            if wandb_enabled:
                write_json_file(
                    status_path,
                    {
                        "state": "failed",
                        "launch_id": launch_id,
                        "failed_at": datetime.datetime.now().isoformat(),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "exception_file": str(error_path),
                    },
                )
        raise
    finally:
        cleanup_distributed(use_barrier=not had_error)


if __name__ == "__main__":
    main()
