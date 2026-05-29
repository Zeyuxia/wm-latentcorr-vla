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
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from policy.ACT.constants import SIM_TASK_CONFIGS

from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .utils_latent import build_stage1_dataloader, build_stage1_dataset, load_raw_episode, resolve_raw_data_dir
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


def infer_legacy_sample_count(
    ckpt_path: str,
    global_batch_override: int | None = None,
    visited: set[str] | None = None,
) -> tuple[int, int]:
    """
    Recover sample_count from older checkpoints that only stored optimizer-step count.

    The recursive resume chain lets us recover mixed runs such as:
    single-GPU ckpt -> 8-GPU resumed ckpt.
    """
    real_path = os.path.realpath(ckpt_path)
    visited = visited or set()
    if real_path in visited:
        raise RuntimeError(f"Checkpoint resume loop detected while inferring sample count: {real_path}")
    visited.add(real_path)

    ckpt = torch.load(real_path, map_location="cpu")
    step = int(ckpt.get("global_step", 0))
    if "sample_count" in ckpt:
        return int(ckpt["sample_count"]), step

    ckpt_args = ckpt.get("args", {}) or {}
    saved_global_batch = ckpt.get("global_batch_size")
    if saved_global_batch is None:
        if global_batch_override is not None:
            saved_global_batch = int(global_batch_override)
        else:
            saved_world_size = int(ckpt.get("world_size", ckpt_args.get("world_size", 1)))
            saved_batch_size = int(ckpt_args.get("batch_size", 1))
            saved_global_batch = saved_batch_size * max(1, saved_world_size)

    parent_ckpt = ckpt_args.get("resume_ckpt")
    if parent_ckpt and os.path.isfile(parent_ckpt):
        parent_sample_count, parent_step = infer_legacy_sample_count(parent_ckpt, visited=visited)
        delta_steps = max(0, step - parent_step)
        return parent_sample_count + delta_steps * int(saved_global_batch), step

    return step * int(saved_global_batch), step


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ACT latent correction stage-1 warmup")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--evac_ckpt", type=str, required=True)
    parser.add_argument("--evac_config", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--act_init_ckpt", type=str, default=None)
    parser.add_argument("--future_teacher_source", type=str, default="real", choices=["real", "sim"])
    parser.add_argument("--urdf_path", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--camera_names", nargs="+", default=None)
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
    parser.add_argument("--legacy_resume_global_batch_size", type=int, default=-1)

    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--predictor_num_blocks", type=int, default=3)
    parser.add_argument("--projector_mid_channels", type=int, default=256)
    parser.add_argument("--wm_adapter_mid_channels", type=int, default=128)
    parser.add_argument("--readout_adapter_mid_channels", type=int, default=128)
    parser.add_argument("--predictor_mlp_hidden", type=int, default=512)
    parser.add_argument("--action_decoder_hidden", type=int, default=512)
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="RoboTwin_ACT_LatentCorr")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"])
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    return parser


def _resolve_dataset_info(task_name: str):
    if task_name not in SIM_TASK_CONFIGS:
        raise KeyError(f"task_name={task_name} not found in SIM_TASK_CONFIGS")
    info = SIM_TASK_CONFIGS[task_name]
    dataset_dir = info["dataset_dir"]
    # SIM_TASK_CONFIGS stores path relative to original ACT directory.
    if dataset_dir.startswith("./"):
        dataset_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "ACT", dataset_dir[2:]))
    else:
        dataset_dir = os.path.realpath(dataset_dir)
    return dataset_dir, int(info["num_episodes"]), list(info["camera_names"])


def _build_act_args(camera_names: list[str], args: argparse.Namespace) -> dict:
    # Build ACT model config without relying on argparse side effects in original ACT code.
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
        # Keep parser compatibility fields.
        "ckpt_dir": args.output_dir,
        "policy_class": "ACT",
        "task_name": args.task_name,
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

    dataset_dir, num_episodes, camera_names = _resolve_dataset_info(args.task_name)
    if args.dataset_dir is not None:
        dataset_dir = os.path.realpath(args.dataset_dir)
    if args.num_episodes is not None:
        num_episodes = int(args.num_episodes)
    if args.camera_names is not None:
        camera_names = list(args.camera_names)
    if args.act_chunk_size < args.prefix_steps:
        raise ValueError(
            f"act_chunk_size ({args.act_chunk_size}) must be >= prefix_steps ({args.prefix_steps})"
        )
    future_offset = args.future_offset if args.future_offset is not None else args.prefix_steps
    raw_data_dir = None
    raw_data_dir_arg = args.raw_data_dir or None
    use_sim_future_teacher = args.future_teacher_source == "sim"
    if use_sim_future_teacher:
        if not args.urdf_path:
            raise ValueError("--urdf_path is required when --future_teacher_source=sim")
        raw_data_dir = resolve_raw_data_dir(args.task_name, raw_data_dir_arg)
    if is_main_process(rank):
        print(f"[stage1] dataset_dir={dataset_dir}")
        if raw_data_dir is not None:
            print(f"[stage1] raw_data_dir={raw_data_dir}")
        print(f"[stage1] num_episodes={num_episodes}, cameras={camera_names}")
        print(
            f"[stage1] act_chunk_size={args.act_chunk_size}, "
            f"prefix_steps={args.prefix_steps}, future_offset={future_offset}"
        )
        print(f"[stage1] future_teacher_source={args.future_teacher_source}")
        print(f"[stage1] distributed={distributed} rank={rank} world_size={world_size} device={args.device}")
        print(
            f"[stage1] current_global_batch_size={current_global_batch_size} "
            f"reference_global_batch_size={args.reference_global_batch_size}"
        )

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

    dataset, norm_stats = build_stage1_dataset(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        act_chunk_size=args.act_chunk_size,
        prefix_steps=args.prefix_steps,
        future_offset=future_offset,
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
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataloader, _ = build_stage1_dataloader(
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            camera_names=camera_names,
            act_chunk_size=args.act_chunk_size,
            prefix_steps=args.prefix_steps,
            future_offset=future_offset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    act_args = _build_act_args(camera_names, args)
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(args.device)
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=args.device,
    )
    fk = None
    raw_cache: dict[int, dict] = {}
    if use_sim_future_teacher:
        from policy.ACT.util.fk_sapien import SapienFK

        fk = SapienFK(args.urdf_path)

    init_batch = next(iter(dataloader))
    model.initialize_latent_heads(init_batch["image_t"][:1].to(args.device), teacher)
    if args.act_init_ckpt and not args.resume_ckpt:
        act_ckpt = torch.load(args.act_init_ckpt, map_location="cpu")
        missing, unexpected = model.base_act.load_state_dict(act_ckpt, strict=False)
        if is_main_process(rank):
            print(f"[stage1] initialized base_act from {args.act_init_ckpt}")
            print(f"[stage1] base_act init missing={len(missing)} unexpected={len(unexpected)}")
    if args.freeze_base_act:
        model.base_act.requires_grad_(False)
        model.base_act.eval()
        if is_main_process(rank):
            print("[stage1] base_act parameters frozen")
    if args.freeze_readout_decoder:
        if model.readout_adapter is not None:
            model.readout_adapter.requires_grad_(False)
        if model.action_decoder is not None:
            model.action_decoder.requires_grad_(False)
        if is_main_process(rank):
            print("[stage1] readout_adapter/action_decoder frozen")

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
        optimizer_groups.append(
            {
                "params": base_act_params,
                "lr": args.lr * args.base_act_lr_scale,
            }
        )
    if latent_params:
        optimizer_groups.append(
            {
                "params": latent_params,
                "lr": args.lr,
            }
        )
    optimizer = AdamW(optimizer_groups, lr=args.lr, weight_decay=args.weight_decay)
    global_step = 0
    sample_count = 0
    start_epoch = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                if is_main_process(rank):
                    print(f"[stage1] optimizer state not loaded during resume: {exc}")
        global_step = int(ckpt.get("global_step", 0))
        if "sample_count" in ckpt:
            sample_count = int(ckpt["sample_count"])
        else:
            legacy_override = args.legacy_resume_global_batch_size if args.legacy_resume_global_batch_size > 0 else None
            sample_count, _ = infer_legacy_sample_count(args.resume_ckpt, global_batch_override=legacy_override)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if is_main_process(rank):
            print(f"[stage1] resumed from {args.resume_ckpt}")
            print(f"[stage1] start_epoch={start_epoch} global_step={global_step}")
            print(
                f"[stage1] sample_count={sample_count} "
                f"schedule_step={sample_count / float(max(1, args.reference_global_batch_size)):.1f}"
            )
            print(f"[stage1] missing={len(missing)} unexpected={len(unexpected)}")

    if distributed:
        model = DDP(
            model,
            device_ids=[int(args.device.split(":")[-1])],
            output_device=int(args.device.split(":")[-1]),
            find_unused_parameters=True,
        )

    config_path = os.path.join(args.output_dir, "stage1_config.txt")
    if is_main_process(rank):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(str(vars(args)))
            f.write("\n")
            f.write(str(asdict(latent_model_cfg)))
            f.write("\n")
            f.write(str(asdict(latent_loss_cfg)))
            f.write("\n")
            f.write(str(asdict(warmup_cfg)))
        stats_path = os.path.join(args.output_dir, "dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(norm_stats, f)

    wandb_run = init_wandb_run(
        enabled=is_main_process(rank) and args.use_wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or f"stage1_{os.path.basename(args.output_dir)}",
        group=args.wandb_group or f"{args.task_name}_stage1",
        tags=args.wandb_tags or ["stage1", args.task_name, f"teacher_{args.future_teacher_source}"],
        mode=args.wandb_mode,
        output_dir=args.output_dir,
        config={
            "stage": "stage1",
            "args": vars(args),
            "latent_model_cfg": asdict(latent_model_cfg),
            "latent_loss_cfg": asdict(latent_loss_cfg),
            "warmup_cfg": asdict(warmup_cfg),
        },
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
            meter_loss = 0.0
            meter_action = 0.0
            meter_action_cond = 0.0
            meter_align = 0.0
            meter_dyn = 0.0
            meter_wm_curr = 0.0
            meter_wm_future = 0.0
            meter_bridge_future = 0.0
            steps_this_epoch = 0

            for batch in pbar:
                image_t_cpu = batch["image_t"]
                qpos_raw_cpu = batch["qpos_raw"]
                action_prefix_raw_cpu = batch["action_prefix_raw"]
                episode_ids = batch["episode_id"].tolist()

                future_teacher_latent = None
                if use_sim_future_teacher:
                    assert fk is not None
                    sim_latents = []
                    for i in range(len(episode_ids)):
                        ep_id = int(episode_ids[i])
                        if ep_id not in raw_cache:
                            raw_cache[ep_id] = load_raw_episode(raw_data_dir, ep_id)
                        sim_latent = teacher.rollout_latent_from_actions(
                            curr_image=image_t_cpu[i, 0],
                            curr_qpos_raw=qpos_raw_cpu[i],
                            action_prefix_raw=action_prefix_raw_cpu[i],
                            raw_data=raw_cache[ep_id],
                            fk=fk,
                            ddim_steps=args.ddim_steps,
                        )
                        sim_latents.append(sim_latent)
                    future_teacher_latent = torch.cat(sim_latents, dim=0).to(args.device, non_blocking=True)

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
                        wm_curr=f"{out.loss_wm_action_current.item():.4f}",
                        wm_fut=f"{out.loss_wm_action_future.item():.4f}",
                        bridge_fut=f"{out.loss_bridge_future.item():.4f}",
                        beta=f"{out.beta_dynamics:.4f}",
                        alpha=f"{out.alpha_latent:.4f}",
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
                        "train_wm_action_current_step": out.loss_wm_action_current.item(),
                        "train_wm_action_future_step": out.loss_wm_action_future.item(),
                        "train_bridge_future_step": out.loss_bridge_future.item(),
                        "beta_dynamics": out.beta_dynamics,
                        "alpha_latent": out.alpha_latent,
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
                    f"[epoch {epoch}] "
                    f"loss={meter_loss/n:.4f} action={meter_action/n:.4f} "
                    f"action_cond={meter_action_cond/n:.4f} "
                    f"align={meter_align/n:.4f} dyn={meter_dyn/n:.4f} "
                    f"wm_curr={meter_wm_curr/n:.4f} wm_fut={meter_wm_future/n:.4f} "
                    f"bridge_fut={meter_bridge_future/n:.4f} "
                    f"time={epoch_time:.1f}s"
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
                }
                save_path = os.path.join(args.output_dir, f"stage1_epoch_{epoch+1:04d}.pt")
                torch.save(ckpt, save_path)
                print(f"[stage1] saved: {save_path}")
                update_wandb_summary(
                    wandb_run,
                    {
                        "last_checkpoint": save_path,
                        "last_epoch": epoch + 1,
                        "last_global_step": global_step,
                    },
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[stage1] reached max_steps={args.max_steps}, stopping early")
                break
    finally:
        finish_wandb(wandb_run)
        cleanup_distributed()


if __name__ == "__main__":
    main()
