#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import os
import pickle
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from policy.ACT.constants import SIM_TASK_CONFIGS

from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .utils_latent import load_raw_episode
from .utils_multitask_latent import (
    build_future_latent_cache_relpath,
    build_multitask_stage1_dataset,
    resolve_multitask_specs,
)
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
    device = f"cuda:{local_rank}"
    return True, rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log_rank(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ACT latent correction stage-1 warmup multitask")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--act_init_ckpt", type=str, default=None)
    parser.add_argument("--multi_task_names", nargs="+", required=True)
    parser.add_argument("--future_teacher_source", type=str, default="sim", choices=["real", "sim"])
    parser.add_argument("--future_latent_cache_dir", type=str, default="")
    parser.add_argument("--future_latent_cache_strict", type=str2bool, default=False)
    parser.add_argument("--future_latent_cache_writeback", type=str2bool, default=True)
    parser.add_argument("--urdf_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--base_act_lr_scale", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=27)
    parser.add_argument("--prefix_steps", type=int, default=16)
    parser.add_argument("--act_chunk_size", type=int, default=50)
    parser.add_argument("--future_offset", type=int, default=None)
    parser.add_argument("--lambda_action", type=float, default=1.0)
    parser.add_argument("--lambda_action_conditioned", type=float, default=0.0)
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--beta_dynamics_max", type=float, default=1.0)
    parser.add_argument("--lambda_wm_action_current", type=float, default=0.0)
    parser.add_argument("--lambda_wm_action_future", type=float, default=0.1)
    parser.add_argument("--lambda_bridge_future", type=float, default=0.0)
    parser.add_argument("--freeze_base_act", type=str2bool, default=False)
    parser.add_argument("--freeze_readout_decoder", type=str2bool, default=False)
    parser.add_argument("--detach_act_feature_for_latent", type=str2bool, default=False)
    parser.add_argument("--use_act_head_conditioning", type=str2bool, default=False)
    parser.add_argument("--use_raw_wm_targets", type=str2bool, default=False)
    parser.add_argument("--dyn_zero_steps", type=int, default=5000)
    parser.add_argument("--dyn_ramp_steps", type=int, default=20000)
    parser.add_argument("--dyn_warmup_curve", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--reference_global_batch_size", type=int, default=4)
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
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="RoboTwin_ACT_LatentCorr")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    return parser


def _build_act_args(camera_names: list[str], args: argparse.Namespace) -> dict:
    task_name = ",".join(list(args.multi_task_names)) if getattr(args, "multi_task_names", None) else "sim-open_laptop-demo_clean-50"
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
        "chunk_size": getattr(args, "act_chunk_size", args.prefix_steps),
        "state_dim": args.state_dim,
        "kl_weight": 10,
        "ckpt_dir": args.output_dir,
        "policy_class": "ACT",
        "task_name": task_name,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
    }


def main():
    args = build_argparser().parse_args()
    distributed, rank, world_size, runtime_device = init_distributed_if_needed(args)
    args.device = runtime_device
    current_global_batch_size = args.batch_size * (world_size if distributed else 1)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.act_chunk_size < args.prefix_steps:
        raise ValueError(f"act_chunk_size ({args.act_chunk_size}) must be >= prefix_steps ({args.prefix_steps})")
    future_offset = args.future_offset if args.future_offset is not None else args.prefix_steps

    task_specs = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS)
    shared_camera_names = list(task_specs[0].camera_names)
    for spec in task_specs[1:]:
        if list(spec.camera_names) != shared_camera_names:
            raise ValueError("All multitask camera_names must match for current stage1 pipeline")

    if args.future_teacher_source == "sim" and not args.urdf_path:
        raise ValueError("--urdf_path is required when --future_teacher_source=sim")
    if args.future_latent_cache_dir:
        args.future_latent_cache_dir = os.path.realpath(args.future_latent_cache_dir)
        if is_main_process(rank):
            os.makedirs(args.future_latent_cache_dir, exist_ok=True)

    if is_main_process(rank):
        print("[stage1-multitask] tasks=", flush=True)
        for spec in task_specs:
            print(
                f"  - {spec.task_name} dataset_dir={spec.dataset_dir} raw_data_dir={spec.raw_data_dir} "
                f"num_episodes={spec.num_episodes} cameras={list(spec.camera_names)}",
                flush=True,
            )
        print(
            f"[stage1-multitask] act_chunk_size={args.act_chunk_size}, prefix_steps={args.prefix_steps}, future_offset={future_offset}",
            flush=True,
        )
        print(
            f"[stage1-multitask] distributed={distributed} rank={rank} world_size={world_size} device={args.device}",
            flush=True,
        )
        print(
            f"[stage1-multitask] current_global_batch_size={current_global_batch_size} "
            f"reference_global_batch_size={args.reference_global_batch_size}",
            flush=True,
        )
        if args.future_latent_cache_dir:
            print(
                f"[stage1-multitask] future_latent_cache_dir={args.future_latent_cache_dir} "
                f"strict={args.future_latent_cache_strict} writeback={args.future_latent_cache_writeback}",
                flush=True,
            )
    log_rank(rank, "building latent configs")

    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=args.projector_mid_channels,
        wm_adapter_mid_channels=args.wm_adapter_mid_channels,
        readout_adapter_mid_channels=args.readout_adapter_mid_channels,
        predictor_num_blocks=args.predictor_num_blocks,
        predictor_mlp_hidden=args.predictor_mlp_hidden,
        action_decoder_hidden=args.action_decoder_hidden,
        action_dim=args.action_dim,
        prefix_steps=args.prefix_steps,
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=args.lambda_action,
        lambda_action_conditioned=args.lambda_action_conditioned,
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

    dataset, norm_stats = build_multitask_stage1_dataset(
        task_specs=task_specs,
        act_chunk_size=args.act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=future_offset,
    )
    log_rank(rank, f"dataset ready | samples={len(dataset)}")
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
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    act_args = _build_act_args(shared_camera_names, args)
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(args.device)
    log_rank(rank, "stage1 model built")
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
    )
    log_rank(rank, "evac teacher initialized")
    fk = None
    raw_cache: dict[tuple[str, int], dict] = {}
    if args.future_teacher_source == "sim":
        from policy.ACT.util.fk_sapien import SapienFK

        fk = SapienFK(args.urdf_path)
        log_rank(rank, "sapien fk initialized")

    init_batch = next(iter(dataloader))
    log_rank(rank, "fetched init batch")
    model.initialize_latent_heads(init_batch["image_t"][:1].to(args.device), teacher)
    log_rank(rank, "latent heads initialized")
    if args.act_init_ckpt and not args.resume_ckpt:
        act_ckpt = torch.load(args.act_init_ckpt, map_location="cpu")
        missing, unexpected = model.base_act.load_state_dict(act_ckpt, strict=False)
        if is_main_process(rank):
            print(f"[stage1-multitask] initialized base_act from {args.act_init_ckpt}", flush=True)
            print(f"[stage1-multitask] base_act init missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.freeze_base_act:
        model.base_act.requires_grad_(False)
        model.base_act.eval()
    if args.freeze_readout_decoder:
        if model.readout_adapter is not None:
            model.readout_adapter.requires_grad_(False)
        if model.action_decoder is not None:
            model.action_decoder.requires_grad_(False)

    base_act_params = []
    latent_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("base_act."):
            base_act_params.append(param)
        else:
            latent_params.append(param)
    optimizer_groups = []
    if base_act_params:
        optimizer_groups.append({"params": base_act_params, "lr": args.lr * args.base_act_lr_scale})
    if latent_params:
        optimizer_groups.append({"params": latent_params, "lr": args.lr})
    optimizer = AdamW(optimizer_groups, lr=args.lr, weight_decay=args.weight_decay)
    log_rank(rank, "optimizer initialized")

    global_step = 0
    sample_count = 0
    start_epoch = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", 0))
        sample_count = int(ckpt.get("sample_count", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if is_main_process(rank):
            print(f"[stage1-multitask] resumed from {args.resume_ckpt}", flush=True)
            print(f"[stage1-multitask] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if distributed:
        model = DDP(
            model,
            device_ids=[int(args.device.split(":")[-1])],
            output_device=int(args.device.split(":")[-1]),
            find_unused_parameters=True,
        )
        log_rank(rank, "ddp wrapper initialized")

    if is_main_process(rank):
        with open(os.path.join(args.output_dir, "stage1_multitask_config.txt"), "w", encoding="utf-8") as f:
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
        run_name=args.wandb_run_name or f"stage1_multitask_{os.path.basename(args.output_dir)}",
        group=args.wandb_group or "robotwin_multitask_stage1",
        tags=args.wandb_tags or ["stage1", "multitask", f"teacher_{args.future_teacher_source}"],
        mode=args.wandb_mode,
        output_dir=args.output_dir,
        config={"stage": "stage1_multitask", "args": vars(args)},
    )

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            raw_model_for_mode = model.module if isinstance(model, DDP) else model
            if args.freeze_base_act:
                raw_model_for_mode.base_act.eval()
            start = time.time()
            pbar = tqdm(dataloader, desc=f"epoch {epoch}", leave=True, disable=not is_main_process(rank))
            meter_loss = meter_action = meter_action_cond = meter_align = 0.0
            meter_dyn = meter_wm_curr = meter_wm_future = meter_bridge_future = 0.0
            steps_this_epoch = 0

            for batch in pbar:
                image_t_cpu = batch["image_t"]
                qpos_raw_cpu = batch["qpos_raw"]
                action_prefix_raw_cpu = batch["action_prefix_raw"]
                episode_ids = batch["episode_id"].tolist()
                task_names = list(batch["task_name"])
                raw_data_dirs = list(batch["raw_data_dir"])

                future_teacher_latent = None
                if args.future_teacher_source == "sim":
                    assert fk is not None
                    start_ts_list = batch["start_ts"].tolist()
                    loaded_latents: list[torch.Tensor | None] = [None] * len(episode_ids)
                    missing_indices: list[int] = []

                    if args.future_latent_cache_dir:
                        for i in range(len(episode_ids)):
                            relpath = build_future_latent_cache_relpath(
                                task_name=task_names[i],
                                episode_id=int(episode_ids[i]),
                                start_ts=int(start_ts_list[i]),
                                future_offset=future_offset,
                                prefix_steps=args.prefix_steps,
                                ddim_steps=args.ddim_steps,
                            )
                            cache_path = os.path.join(args.future_latent_cache_dir, relpath)
                            if os.path.exists(cache_path):
                                latent_i = torch.load(cache_path, map_location="cpu")
                                if latent_i.ndim == 4 and latent_i.shape[0] == 1:
                                    latent_i = latent_i[0]
                                loaded_latents[i] = latent_i.float()
                            else:
                                missing_indices.append(i)
                    else:
                        missing_indices = list(range(len(episode_ids)))

                    if missing_indices:
                        if args.future_latent_cache_strict and args.future_latent_cache_dir:
                            missing_desc = [
                                f"{task_names[i]}:ep{int(episode_ids[i])}:t{int(start_ts_list[i])}" for i in missing_indices
                            ]
                            raise FileNotFoundError(
                                "Missing future latent cache entries: " + ", ".join(missing_desc[:8])
                            )

                        raw_data_list = []
                        for i in missing_indices:
                            cache_key = (task_names[i], int(episode_ids[i]))
                            if cache_key not in raw_cache:
                                raw_cache[cache_key] = load_raw_episode(raw_data_dirs[i], int(episode_ids[i]))
                            raw_data_list.append(raw_cache[cache_key])

                        sim_latents_missing = teacher.rollout_latent_from_actions_batch(
                            curr_image=image_t_cpu[missing_indices, 0],
                            curr_qpos_raw=qpos_raw_cpu[missing_indices],
                            action_prefix_raw=action_prefix_raw_cpu[missing_indices],
                            raw_data=raw_data_list,
                            fk=fk,
                            ddim_steps=args.ddim_steps,
                        ).cpu()
                        for local_idx, batch_idx in enumerate(missing_indices):
                            latent_i = sim_latents_missing[local_idx].float()
                            loaded_latents[batch_idx] = latent_i
                            if args.future_latent_cache_dir and args.future_latent_cache_writeback:
                                relpath = build_future_latent_cache_relpath(
                                    task_name=task_names[batch_idx],
                                    episode_id=int(episode_ids[batch_idx]),
                                    start_ts=int(start_ts_list[batch_idx]),
                                    future_offset=future_offset,
                                    prefix_steps=args.prefix_steps,
                                    ddim_steps=args.ddim_steps,
                                )
                                cache_path = os.path.join(args.future_latent_cache_dir, relpath)
                                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                                torch.save(latent_i.half(), cache_path)

                    future_teacher_latent = torch.stack(
                        [latent_i for latent_i in loaded_latents if latent_i is not None],
                        dim=0,
                    ).to(args.device, non_blocking=True)

                image_t = image_t_cpu.to(args.device, non_blocking=True)
                image_t1 = batch["image_t_future"].to(args.device, non_blocking=True)
                qpos_t = batch["qpos_t"].to(args.device, non_blocking=True)
                qpos_future_norm = None
                if args.use_act_head_conditioning:
                    qpos_future_norm = batch["qpos_future"].to(args.device, non_blocking=True)

                act_action_chunk = batch["act_action_chunk"].to(args.device, non_blocking=True)
                act_is_pad = batch["act_is_pad"].to(args.device, non_blocking=True)
                action_prefix = batch["action_prefix"].to(args.device, non_blocking=True)
                action_future_prefix = batch["action_future_prefix"].to(args.device, non_blocking=True)
                is_pad_prefix = batch["is_pad_prefix"].to(args.device, non_blocking=True)
                is_pad_future_prefix = batch["is_pad_future_prefix"].to(args.device, non_blocking=True)

                schedule_step = sample_count / float(max(1, args.reference_global_batch_size))
                out = model(
                    image_t=image_t,
                    image_t1=image_t1,
                    qpos_t=qpos_t,
                    qpos_future_norm=qpos_future_norm,
                    act_action_chunk=act_action_chunk,
                    act_is_pad=act_is_pad,
                    action_prefix=action_prefix,
                    action_future_prefix=action_future_prefix,
                    is_pad_prefix=is_pad_prefix,
                    is_pad_future_prefix=is_pad_future_prefix,
                    wm_teacher=teacher,
                    global_step=schedule_step,
                    use_act_head_conditioning=args.use_act_head_conditioning,
                    future_teacher_latent=future_teacher_latent,
                    mode="stage1",
                )
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()

                global_step += 1
                sample_count += current_global_batch_size
                meter_loss += out.loss.item()
                meter_action += out.loss_action.item()
                meter_action_cond += out.loss_action_conditioned.item()
                meter_align += out.loss_align.item()
                meter_dyn += out.loss_dynamics.item()
                meter_wm_curr += out.loss_wm_action_current.item()
                meter_wm_future += out.loss_wm_action_future.item()
                meter_bridge_future += out.loss_bridge_future.item()
                steps_this_epoch += 1

                if is_main_process(rank):
                    pbar.set_postfix(
                        loss=f"{out.loss.item():.4f}",
                        action=f"{out.loss_action.item():.4f}",
                        action_cond=f"{out.loss_action_conditioned.item():.4f}",
                        align=f"{out.loss_align.item():.4f}",
                        dyn=f"{out.loss_dynamics.item():.4f}",
                        step=global_step,
                        sched=f"{schedule_step:.0f}",
                    )
                log_wandb(
                    wandb_run,
                    {
                        "global_step": global_step,
                        "schedule_step": schedule_step,
                        "sample_count": sample_count,
                        "global_batch_size": current_global_batch_size,
                        "epoch": epoch,
                        "train_loss_step": out.loss.item(),
                        "train_action_step": out.loss_action.item(),
                        "train_action_conditioned_step": out.loss_action_conditioned.item(),
                        "train_align_step": out.loss_align.item(),
                        "train_dynamics_step": out.loss_dynamics.item(),
                    },
                    step=global_step,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

            n = max(1, steps_this_epoch)
            epoch_time = time.time() - start
            if distributed:
                stats_tensor = torch.tensor(
                    [
                        meter_loss,
                        meter_action,
                        meter_action_cond,
                        meter_align,
                        meter_dyn,
                        meter_wm_curr,
                        meter_wm_future,
                        meter_bridge_future,
                        float(steps_this_epoch),
                    ],
                    device=args.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                meter_loss = float(stats_tensor[0].item())
                meter_action = float(stats_tensor[1].item())
                meter_action_cond = float(stats_tensor[2].item())
                meter_align = float(stats_tensor[3].item())
                meter_dyn = float(stats_tensor[4].item())
                meter_wm_curr = float(stats_tensor[5].item())
                meter_wm_future = float(stats_tensor[6].item())
                meter_bridge_future = float(stats_tensor[7].item())
                n = max(1, int(stats_tensor[8].item()))

            if is_main_process(rank):
                print(
                    f"[epoch {epoch}] loss={meter_loss/n:.4f} action={meter_action/n:.4f} "
                    f"action_cond={meter_action_cond/n:.4f} align={meter_align/n:.4f} "
                    f"dyn={meter_dyn/n:.4f} wm_curr={meter_wm_curr/n:.4f} "
                    f"wm_fut={meter_wm_future/n:.4f} bridge_fut={meter_bridge_future/n:.4f} "
                    f"time={epoch_time:.1f}s",
                    flush=True,
                )
            log_wandb(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "epoch_loss": meter_loss / n,
                    "epoch_action": meter_action / n,
                    "epoch_action_conditioned": meter_action_cond / n,
                    "epoch_align": meter_align / n,
                    "epoch_dynamics": meter_dyn / n,
                    "epoch_wm_action_current": meter_wm_curr / n,
                    "epoch_wm_action_future": meter_wm_future / n,
                    "epoch_bridge_future": meter_bridge_future / n,
                    "epoch_time_sec": epoch_time,
                    "global_step": global_step,
                    "schedule_step": sample_count / float(max(1, args.reference_global_batch_size)),
                    "sample_count": sample_count,
                },
                step=global_step,
            )

            if is_main_process(rank) and (epoch + 1) % args.save_freq == 0:
                raw_model = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "sample_count": sample_count,
                    "schedule_step": sample_count / float(max(1, args.reference_global_batch_size)),
                    "global_batch_size": current_global_batch_size,
                    "world_size": world_size,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "norm_stats": norm_stats,
                    "args": vars(args),
                    "multi_task_names": list(args.multi_task_names),
                }
                save_path = os.path.join(args.output_dir, f"stage1_epoch_{epoch+1:04d}.pt")
                torch.save(ckpt, save_path)
                print(f"[stage1-multitask] saved: {save_path}", flush=True)
                update_wandb_summary(
                    wandb_run,
                    {"last_checkpoint": save_path, "last_epoch": epoch + 1, "last_global_step": global_step},
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main_process(rank):
                    print(f"[stage1-multitask] reached max_steps={args.max_steps}, stopping early", flush=True)
                break
    finally:
        finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
