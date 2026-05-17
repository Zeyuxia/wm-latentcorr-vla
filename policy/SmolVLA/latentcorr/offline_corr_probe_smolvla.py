from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.utils.constants import ACTION
from policy.ACT.constants import SIM_TASK_CONFIGS
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.multitask_latent_utils import build_multitask_stage1_dataset, resolve_multitask_specs
from policy.SmolVLA.latentcorr.smolvla_data_utils import stack_smolvla_batches
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentPolicy
from policy.SmolVLA.latentcorr.train_smolvla import (
    OfflineCorrectionExportDataset,
    add_common_args,
    add_failure_train_args,
    build_base_policy,
    build_bridge_config,
    build_stage1_offline_correction_samples,
    build_stage1_samples_from_raw_batch,
    build_stage1_warmup_config,
    initialize_latent_policy_from_sample,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("probe offline correction exports for SmolVLA stage1")
    add_common_args(parser)
    add_failure_train_args(parser)
    parser.add_argument("--probe_task_configs", nargs="+", required=True)
    parser.add_argument("--probe_batches", type=int, default=8)
    parser.add_argument("--probe_corr_batch_size", type=int, default=4)
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--probe_output_json", type=str, default="")
    parser.add_argument("--probe_skip_grad", action="store_true")
    return parser


def _as_device_batch(raw_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw_batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _slice_batch(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=batch[ACTION].device)
    out: dict[str, Any] = {}
    batch_size = int(batch[ACTION].shape[0])
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
            out[key] = value.index_select(0, index_tensor)
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = [value[int(i)] for i in indices]
        else:
            out[key] = value
    return out


def _action_loss_unreduced(model: SmolVLALatentPolicy, batch: dict[str, Any]) -> torch.Tensor:
    images, img_masks, lang_tokens, lang_masks, state = model._policy_inputs_from_batch(batch)
    actions_padded = model.base_policy.prepare_action({ACTION: batch[ACTION]})
    losses = model.base_policy.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions_padded)
    return losses[:, :, : model.base_policy.config.max_action_dim]


def _mean_loss(model: SmolVLALatentPolicy, batch: dict[str, Any], steps: int | None = None) -> torch.Tensor:
    losses = _action_loss_unreduced(model, batch)
    if steps is not None:
        losses = losses[:, : int(steps), :]
    return losses.mean()


def _per_sample_loss(model: SmolVLALatentPolicy, batch: dict[str, Any], steps: int | None = None) -> np.ndarray:
    with torch.no_grad():
        losses = _action_loss_unreduced(model, batch)
        if steps is not None:
            losses = losses[:, : int(steps), :]
        values = losses.mean(dim=(1, 2)).detach().float().cpu().numpy()
    return values.astype(np.float64)


def _zero_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _named_trainable_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _grad_vec(model: torch.nn.Module, params: list[tuple[str, torch.nn.Parameter]], loss: torch.Tensor) -> torch.Tensor:
    _zero_grads(model)
    loss.backward(retain_graph=True)
    chunks = []
    for _, param in params:
        grad = param.grad.detach() if param.grad is not None else torch.zeros_like(param)
        chunks.append(grad.reshape(-1).float().cpu())
    _zero_grads(model)
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    q = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _load_stage1_checkpoint_if_needed(model: SmolVLALatentPolicy, checkpoint_path: str) -> None:
    if not str(checkpoint_path).strip():
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    base_policy, preprocess, _ = build_base_policy(args)
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_stage1_warmup_config(args),
    )
    model.to(device)
    teacher = EvacLatentTeacher(
        evac_ckpt=args.evac_ckpt,
        evac_config=args.evac_config,
        device=device,
        load_rollout_model=False,
    )
    initialize_latent_policy_from_sample(
        latent_policy=model,
        preprocess=preprocess,
        instruction_type=args.instruction_type,
        raw_sample=normal_dataset[0],
        teacher=teacher,
    )
    _load_stage1_checkpoint_if_needed(model, args.stage1_ckpt)
    model.train()
    params = _named_trainable_params(model)

    print(f"normal samples: {len(normal_dataset):,}")
    print(f"trainable params: {sum(param.numel() for _, param in params):,}")
    print(f"probe configs: {args.probe_task_configs}")

    cfg_payloads = []
    for task_config in args.probe_task_configs:
        correction_dataset = OfflineCorrectionExportDataset(
            task_names=[spec.task_name for spec in task_specs],
            norm_stats=norm_stats,
            data_root=str(args.offline_corr_data_root),
            task_config=str(task_config),
            act_chunk_size=int(args.act_chunk_size),
            prefix_steps=int(args.prefix_steps),
            future_offset=int(args.future_offset),
            instruction_type=str(args.instruction_type),
            max_samples_per_task=int(args.offline_corr_max_samples_per_task),
        )
        sampler = correction_dataset.build_balanced_sampler(seed=int(args.seed) + 991)
        corr_loader = DataLoader(
            correction_dataset,
            batch_size=int(args.probe_corr_batch_size),
            sampler=sampler,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        normal_iter = iter(normal_loader)
        corr_iter = iter(corr_loader)
        records = []

        for batch_idx in tqdm(range(int(args.probe_batches)), desc=f"probe {task_config}"):
            try:
                normal_raw = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_loader)
                normal_raw = next(normal_iter)
            try:
                corr_raw = next(corr_iter)
            except StopIteration:
                corr_iter = iter(corr_loader)
                corr_raw = next(corr_iter)

            normal_raw = _as_device_batch(normal_raw, device)
            corr_raw = _as_device_batch(corr_raw, device)
            clean_samples, _, _, _, _ = build_stage1_samples_from_raw_batch(
                raw_batch=normal_raw,
                latent_model=model,
                preprocess=preprocess,
                instruction_type=args.instruction_type,
                prefix_steps=int(args.prefix_steps),
            )
            corr_samples, _, _, _ = build_stage1_offline_correction_samples(
                correction_raw_batch=corr_raw,
                latent_model=model,
                preprocess=preprocess,
                instruction_type=args.instruction_type,
                prefix_steps=int(args.prefix_steps),
            )
            clean_batch = stack_smolvla_batches(clean_samples)
            corr_batch = stack_smolvla_batches(corr_samples)

            clean_per = _per_sample_loss(model, clean_batch)
            corr_per = _per_sample_loss(model, corr_batch)
            clean_prefix_per = _per_sample_loss(model, clean_batch, steps=int(args.prefix_steps))
            corr_prefix_per = _per_sample_loss(model, corr_batch, steps=int(args.prefix_steps))
            outlier_threshold = float(args.stage1_corr_outlier_max_action_loss)

            record: dict[str, Any] = {
                "batch_idx": int(batch_idx),
                "num_clean": int(len(clean_samples)),
                "num_corr": int(len(corr_samples)),
                "clean_action_loss": float(np.mean(clean_per)),
                "corr_action_loss": float(np.mean(corr_per)),
                "clean_prefix_action_loss": float(np.mean(clean_prefix_per)),
                "corr_prefix_action_loss": float(np.mean(corr_prefix_per)),
                "corr_outlier_count_at_threshold": int(np.sum(corr_per > outlier_threshold)),
                "corr_outlier_fraction_at_threshold": float(np.mean(corr_per > outlier_threshold)),
            }
            if not bool(args.probe_skip_grad):
                clean_loss = _mean_loss(model, clean_batch)
                corr_loss = _mean_loss(model, corr_batch)
                clean_prefix_loss = _mean_loss(model, clean_batch, steps=int(args.prefix_steps))
                corr_prefix_loss = _mean_loss(model, corr_batch, steps=int(args.prefix_steps))
                g_clean = _grad_vec(model, params, clean_loss)
                g_corr = _grad_vec(model, params, corr_loss)
                g_clean_prefix = _grad_vec(model, params, clean_prefix_loss)
                g_corr_prefix = _grad_vec(model, params, corr_prefix_loss)
                record.update(
                    {
                        "norm_clean_action": float(g_clean.norm().item()),
                        "norm_corr_action": float(g_corr.norm().item()),
                        "norm_clean_prefix_action": float(g_clean_prefix.norm().item()),
                        "norm_corr_prefix_action": float(g_corr_prefix.norm().item()),
                        "cos_clean_corr_action": _cosine(g_clean, g_corr),
                        "cos_clean_corr_prefix_action": _cosine(g_clean_prefix, g_corr_prefix),
                        "cos_clean_action_corr_prefix_action": _cosine(g_clean, g_corr_prefix),
                    }
                )
            records.append(record)
            del clean_batch, corr_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = {
            "task_config": str(task_config),
            "num_records": int(len(correction_dataset)),
            "clean_action_loss": _summary([r["clean_action_loss"] for r in records]),
            "corr_action_loss": _summary([r["corr_action_loss"] for r in records]),
            "clean_prefix_action_loss": _summary([r["clean_prefix_action_loss"] for r in records]),
            "corr_prefix_action_loss": _summary([r["corr_prefix_action_loss"] for r in records]),
            "corr_outlier_fraction_at_threshold": _summary(
                [r["corr_outlier_fraction_at_threshold"] for r in records]
            ),
        }
        if records and "cos_clean_corr_action" in records[0]:
            summary.update(
                {
                    "cos_clean_corr_action": _summary([r["cos_clean_corr_action"] for r in records]),
                    "cos_clean_corr_prefix_action": _summary([r["cos_clean_corr_prefix_action"] for r in records]),
                    "norm_ratio_corr_over_clean": _summary(
                        [
                            r["norm_corr_action"] / max(1e-12, r["norm_clean_action"])
                            for r in records
                        ]
                    ),
                    "norm_ratio_corr_prefix_over_clean_prefix": _summary(
                        [
                            r["norm_corr_prefix_action"] / max(1e-12, r["norm_clean_prefix_action"])
                            for r in records
                        ]
                    ),
                }
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        cfg_payloads.append({"summary": summary, "records": records})

    payload = {
        "args": vars(args),
        "configs": cfg_payloads,
    }
    if str(args.probe_output_json).strip():
        output_path = Path(args.probe_output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
