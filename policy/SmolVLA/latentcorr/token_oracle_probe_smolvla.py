from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
    add_common_args,
    build_base_policy,
    build_bridge_config,
    build_stage1_samples_from_raw_batch,
    build_stage1_warmup_config,
    initialize_latent_policy_from_sample,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SmolVLA task-wise token predictability and oracle-token action-loss probe")
    add_common_args(p)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--probe_tasks", nargs="+", required=True)
    p.add_argument("--samples_per_task", type=int, default=64)
    p.add_argument("--batch_size_probe", type=int, default=4)
    p.add_argument("--probe_output_json", required=True)
    return p.parse_args()


def _as_device_batch(raw_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw_batch.items()}


def _subset_for_task(dataset, task_name: str, limit: int) -> list[int]:
    requested = str(task_name)
    requested_short = requested
    if requested.startswith("sim-"):
        parts = requested.split("-")
        if len(parts) >= 2:
            requested_short = parts[1]
    out: list[int] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        item_task = str(item.get("task_name"))
        item_short = item_task
        if item_task.startswith("sim-"):
            parts = item_task.split("-")
            if len(parts) >= 2:
                item_short = parts[1]
        if item_task == requested or item_short == requested_short:
            out.append(idx)
            if len(out) >= int(limit):
                break
    return out


def _load_ckpt(model: SmolVLALatentPolicy, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def _sample_action_loss(
    model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    token: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    att_mask: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> torch.Tensor:
    images, img_masks, lang_tokens, lang_masks, state = model._policy_inputs_from_batch(batch)
    actions_padded = model.base_policy.prepare_action({ACTION: batch[ACTION]})
    losses = model.base_policy.model.forward(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions_padded,
        noise=noise,
        time=time,
        external_prefix_tokens=token,
        external_prefix_mask=mask,
        external_prefix_att_mask=att_mask,
    )[:, :, : model.base_policy.config.max_action_dim]
    return losses.mean(dim=(1, 2)).detach().float().cpu()


def _sample_action_chunk(
    model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    token: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    att_mask: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    images, img_masks, lang_tokens, lang_masks, state = model._policy_inputs_from_batch(batch)
    chunk = model.base_policy.model.sample_actions(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=noise,
        external_prefix_tokens=token,
        external_prefix_mask=mask,
        external_prefix_att_mask=att_mask,
    )[:, :, : model.bridge_cfg.action_dim]
    return chunk.detach().float().cpu()


def _per_sample_chunk_mse(chunk: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    steps = min(int(chunk.shape[1]), int(target.shape[1]))
    dims = min(int(chunk.shape[2]), int(target.shape[2]))
    diff = chunk[:, :steps, :dims] - target.detach().float().cpu()[:, :steps, :dims]
    return diff.pow(2).mean(dim=(1, 2)).numpy()


def _per_sample_pair_mse(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    steps = min(int(a.shape[1]), int(b.shape[1]))
    dims = min(int(a.shape[2]), int(b.shape[2]))
    diff = a[:, :steps, :dims] - b[:, :steps, :dims]
    return diff.pow(2).mean(dim=(1, 2)).numpy()


def _cosine_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)


def _summary(xs: list[float]) -> dict[str, float | int]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    q = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(q[1]),
        "q25": float(q[0]),
        "q75": float(q[2]),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _linear_probe_mse(features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    n = int(features.shape[0])
    if n < 4:
        return {"n": n, "mean_mse": float("nan"), "linear_mse": float("nan"), "gain": float("nan")}
    x = features.astype(np.float64)
    y = targets.astype(np.float64)
    split = max(2, int(round(n * 0.7)))
    if split >= n:
        split = n - 1
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-8
    xs_train = (x_train - x_mean) / x_std
    xs_test = (x_test - x_mean) / x_std
    x_aug = np.concatenate([xs_train, np.ones((xs_train.shape[0], 1))], axis=1)
    w = np.linalg.pinv(x_aug) @ y_train
    pred = np.concatenate([xs_test, np.ones((xs_test.shape[0], 1))], axis=1) @ w
    mean_pred = np.broadcast_to(y_train.mean(axis=0, keepdims=True), y_test.shape)
    linear_mse = float(np.mean((pred - y_test) ** 2))
    mean_mse = float(np.mean((mean_pred - y_test) ** 2))
    return {"n": n, "mean_mse": mean_mse, "linear_mse": linear_mse, "gain": float(mean_mse / max(linear_mse, 1e-12))}


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    task_specs_all = resolve_multitask_specs(list(args.multi_task_names), SIM_TASK_CONFIGS, raw_data_root_overrides=None)
    dataset, _ = build_multitask_stage1_dataset(
        task_specs=task_specs_all,
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
    )
    base_policy, preprocess, _ = build_base_policy(args, dataset_sample=dataset[0], dataset_stats=(dataset.stats if hasattr(dataset, "stats") else None))
    model = SmolVLALatentPolicy(
        base_policy=base_policy,
        bridge_cfg=build_bridge_config(args),
        warmup_cfg=build_stage1_warmup_config(args),
    ).to(device)
    teacher = EvacLatentTeacher(args.evac_ckpt, args.evac_config, device=device, load_rollout_model=False)
    initialize_latent_policy_from_sample(model, preprocess, str(args.instruction_type), dataset[0], teacher=teacher)
    _load_ckpt(model, str(args.stage1_ckpt))
    model.eval()

    payload: dict[str, Any] = {"args": vars(args), "tasks": {}}
    for task in args.probe_tasks:
        indices = _subset_for_task(dataset, task, int(args.samples_per_task))
        if not indices:
            payload["tasks"][task] = {"error": "no_samples"}
            continue
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=int(args.batch_size_probe),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        metrics: dict[str, list[float]] = defaultdict(list)
        feats: list[np.ndarray] = []
        teacher_tokens: list[np.ndarray] = []
        pred_tokens: list[np.ndarray] = []
        for raw in tqdm(loader, desc=f"probe {task}"):
            raw = _as_device_batch(raw, device)
            samples, future_images, action_prefixes, _, _ = build_stage1_samples_from_raw_batch(
                raw_batch=raw,
                latent_model=model,
                preprocess=preprocess,
                instruction_type=str(args.instruction_type),
                prefix_steps=int(args.prefix_steps),
            )
            batch = stack_smolvla_batches(samples)
            future = torch.stack(future_images, dim=0).to(device=device, dtype=torch.float32)
            prefix = torch.cat(action_prefixes, dim=0).to(device=device)
            with torch.no_grad():
                teacher_latent = teacher.encode_image(future)
                teacher_shared = model.shared_teacher_latent(teacher_latent)
                target_hw = (teacher_shared.shape[-2], teacher_shared.shape[-1])
                pred_latent = model.predict_next_latent(batch, prefix, target_hw=target_hw)
                rand_latent = torch.randn_like(pred_latent)
                oracle_token, oracle_mask, oracle_att = model.build_condition_token(teacher_shared, scale=1.0)
                pred_token, pred_mask, pred_att = model.build_predicted_condition_token(pred_latent, scale=1.0)
                rand_token, rand_mask, rand_att = model.build_predicted_condition_token(rand_latent, scale=1.0)
                actions_padded = model.base_policy.prepare_action({ACTION: batch[ACTION]})
                shared_noise = model.base_policy.model.sample_noise(actions_padded.shape, actions_padded.device)
                shared_time = model.base_policy.model.sample_time(actions_padded.shape[0], actions_padded.device)
                base_loss = _sample_action_loss(model, batch, noise=shared_noise, time=shared_time)
                pred_loss = _sample_action_loss(
                    model, batch, pred_token, pred_mask, pred_att, noise=shared_noise, time=shared_time
                )
                oracle_loss = _sample_action_loss(
                    model, batch, oracle_token, oracle_mask, oracle_att, noise=shared_noise, time=shared_time
                )
                random_loss = _sample_action_loss(
                    model, batch, rand_token, rand_mask, rand_att, noise=shared_noise, time=shared_time
                )
                shared_sample_noise = model.base_policy.model.sample_noise(actions_padded.shape, actions_padded.device)
                base_chunk = _sample_action_chunk(model, batch, noise=shared_sample_noise)
                pred_chunk = _sample_action_chunk(model, batch, pred_token, pred_mask, pred_att, noise=shared_sample_noise)
                oracle_chunk = _sample_action_chunk(
                    model, batch, oracle_token, oracle_mask, oracle_att, noise=shared_sample_noise
                )
                random_chunk = _sample_action_chunk(
                    model, batch, rand_token, rand_mask, rand_att, noise=shared_sample_noise
                )
                z_proj = model.extract_visual_latent_map(batch, target_hw=target_hw)
                feat = F.adaptive_avg_pool2d(z_proj, 1).flatten(1).detach().float().cpu().numpy()
                t_tok = oracle_token[:, 0, :].detach().float().cpu().numpy()
                p_tok = pred_token[:, 0, :].detach().float().cpu().numpy()
                r_tok = rand_token[:, 0, :].detach().float().cpu().numpy()
            for key, vals in [
                ("base_action_loss", base_loss.numpy()),
                ("pred_action_loss", pred_loss.numpy()),
                ("oracle_action_loss", oracle_loss.numpy()),
                ("random_action_loss", random_loss.numpy()),
            ]:
                metrics[key].extend([float(x) for x in vals])
            metrics["pred_minus_base"].extend([float(x) for x in (pred_loss - base_loss).numpy()])
            metrics["oracle_minus_base"].extend([float(x) for x in (oracle_loss - base_loss).numpy()])
            metrics["random_minus_base"].extend([float(x) for x in (random_loss - base_loss).numpy()])
            metrics["token_mse_pred_teacher"].extend([float(x) for x in np.mean((p_tok - t_tok) ** 2, axis=1)])
            metrics["token_cos_pred_teacher"].extend([float(x) for x in _cosine_np(p_tok, t_tok)])
            metrics["token_cos_random_teacher"].extend([float(x) for x in _cosine_np(r_tok, t_tok)])
            metrics["token_norm_pred"].extend([float(x) for x in np.linalg.norm(p_tok, axis=1)])
            metrics["token_norm_teacher"].extend([float(x) for x in np.linalg.norm(t_tok, axis=1)])
            metrics["token_norm_random"].extend([float(x) for x in np.linalg.norm(r_tok, axis=1)])
            metrics["chunk_mse_base_gt"].extend([float(x) for x in _per_sample_chunk_mse(base_chunk, batch[ACTION])])
            metrics["chunk_mse_pred_gt"].extend([float(x) for x in _per_sample_chunk_mse(pred_chunk, batch[ACTION])])
            metrics["chunk_mse_oracle_gt"].extend([float(x) for x in _per_sample_chunk_mse(oracle_chunk, batch[ACTION])])
            metrics["chunk_mse_random_gt"].extend([float(x) for x in _per_sample_chunk_mse(random_chunk, batch[ACTION])])
            metrics["chunk_mse_pred_base"].extend([float(x) for x in _per_sample_pair_mse(pred_chunk, base_chunk)])
            metrics["chunk_mse_oracle_base"].extend([float(x) for x in _per_sample_pair_mse(oracle_chunk, base_chunk)])
            metrics["chunk_mse_oracle_pred"].extend([float(x) for x in _per_sample_pair_mse(oracle_chunk, pred_chunk)])
            metrics["chunk_mse_random_base"].extend([float(x) for x in _per_sample_pair_mse(random_chunk, base_chunk)])
            feats.append(feat)
            teacher_tokens.append(t_tok)
            pred_tokens.append(p_tok)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        f_all = np.concatenate(feats, axis=0) if feats else np.empty((0, 0))
        t_all = np.concatenate(teacher_tokens, axis=0) if teacher_tokens else np.empty((0, 0))
        p_all = np.concatenate(pred_tokens, axis=0) if pred_tokens else np.empty((0, 0))
        task_payload = {key: _summary(vals) for key, vals in metrics.items()}
        task_payload["token_predictability_linear_probe"] = _linear_probe_mse(f_all, t_all)
        if t_all.size and p_all.size:
            task_payload["pred_token_global_mse"] = float(np.mean((p_all - t_all) ** 2))
        payload["tasks"][task] = task_payload

    out = Path(args.probe_output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
