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
from policy.SmolVLA.latentcorr.act_aligned_correction import (
    ACTAlignedCorrectionBuilder,
    build_act_aligned_cfg_from_args,
)
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.failure_manifest_utils import load_failure_table_paths
from policy.SmolVLA.latentcorr.multitask_failure_dataset import (
    MultiTaskFailureDatasetConfig,
    build_multitask_failure_dataset,
)
from policy.SmolVLA.latentcorr.multitask_latent_utils import (
    build_multitask_stage1_dataset,
    resolve_multitask_specs,
)
from policy.SmolVLA.latentcorr.smolvla_data_utils import stack_smolvla_batches
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentPolicy
from policy.SmolVLA.latentcorr.train_smolvla import (
    add_common_args,
    add_failure_train_args,
    build_base_policy,
    build_bridge_config,
    build_stage1_correction_samples,
    build_stage1_samples_from_raw_batch,
    build_stage1_warmup_config,
    initialize_latent_policy_from_sample,
    load_evac_sample_size,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("probe SmolVLA stage1 gradients")
    add_common_args(parser)
    add_failure_train_args(parser)
    parser.add_argument("--probe_batches", type=int, default=4)
    parser.add_argument("--probe_corr_batch_size", type=int, default=None)
    parser.add_argument("--probe_global_step", type=int, default=1000)
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--probe_output_json", type=str, default="")
    parser.add_argument("--probe_include_full_vectors", type=str, default="false", choices=["true", "false"])
    return parser


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_device_batch(raw_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in raw_batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _action_loss_unreduced(model: SmolVLALatentPolicy, batch: dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
    images, img_masks, lang_tokens, lang_masks, state = model._policy_inputs_from_batch(batch)
    action_batch = {ACTION: actions}
    actions_padded = model.base_policy.prepare_action(action_batch)
    losses = model.base_policy.model.forward(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions_padded,
    )
    return losses[:, :, : model.base_policy.config.max_action_dim]


def _condition_loss_unreduced(
    model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    actions: torch.Tensor,
    future_teacher_latent: torch.Tensor,
) -> torch.Tensor:
    cond_token, cond_mask, cond_att_mask = model.build_condition_token(
        model.teacher_latent_to_map(future_teacher_latent).detach(),
        scale=1.0,
    )
    images, img_masks, lang_tokens, lang_masks, state = model._policy_inputs_from_batch(batch)
    action_batch = {ACTION: actions}
    actions_padded = model.base_policy.prepare_action(action_batch)
    losses = model.base_policy.model.forward(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions_padded,
        external_prefix_tokens=cond_token,
        external_prefix_mask=cond_mask,
        external_prefix_att_mask=cond_att_mask,
    )
    return losses[:, :, : model.base_policy.config.max_action_dim]


def _dynamics_loss_unreduced(
    model: SmolVLALatentPolicy,
    batch: dict[str, Any],
    future_teacher_latent: torch.Tensor,
    action_prefix: torch.Tensor,
) -> torch.Tensor:
    teacher_shared = model.shared_teacher_latent(future_teacher_latent)
    target_hw = (teacher_shared.shape[-2], teacher_shared.shape[-1])
    predicted_latent = model.predict_next_latent(batch, action_prefix, target_hw=target_hw)
    per_elem = F.mse_loss(predicted_latent, teacher_shared.detach(), reduction="none")
    return per_elem.flatten(1).mean(dim=1)


def _zero_model_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _named_trainable_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _grad_vector(
    model: torch.nn.Module,
    params: list[tuple[str, torch.nn.Parameter]],
    loss: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    _zero_model_grads(model)
    loss.backward(retain_graph=True)
    chunks = []
    group_sq: dict[str, float] = {}
    for name, param in params:
        if param.grad is None:
            grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        else:
            grad = param.grad.detach()
        chunks.append(grad.reshape(-1).float().cpu())
        group = _param_group_name(name)
        group_sq[group] = group_sq.get(group, 0.0) + float(torch.sum(grad.detach().float() ** 2).item())
    _zero_model_grads(model)
    vec = torch.cat(chunks, dim=0) if chunks else torch.empty(0)
    group_norms = {key: math.sqrt(value) for key, value in sorted(group_sq.items())}
    return vec, group_norms


def _param_group_name(name: str) -> str:
    if name.startswith("base_policy."):
        if ".action_" in name or name.endswith("action_in_proj.weight") or name.endswith("action_out_proj.weight"):
            return "base_action_proj"
        if ".lm_expert." in name:
            return "base_lm_expert"
        if ".state_proj." in name:
            return "base_state_proj"
        if ".vision_model." in name:
            return "base_vision"
        if ".vlm." in name:
            return "base_vlm"
        return "base_other"
    if name.startswith("projector."):
        return "projector"
    if name.startswith("wm_adapter."):
        return "wm_adapter"
    if name.startswith("predictor."):
        return "predictor"
    if name.startswith("condition_proj."):
        return "condition_proj"
    return "other"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 1e-20:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) <= 1:
        return 0.0 if len(finite) == 1 else float("nan")
    mean = sum(finite) / len(finite)
    return float(math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)))


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    metric_keys = [
        "clean_action_loss",
        "corr_action_loss",
        "clean_cond_loss",
        "corr_cond_loss",
        "clean_dyn_loss",
        "corr_dyn_loss",
        "cos_clean_corr_action",
        "cos_clean_action_clean_dyn",
        "cos_corr_action_corr_dyn",
        "cos_clean_action_corr_dyn",
        "cos_corr_action_clean_dyn",
        "cos_clean_action_clean_cond",
        "cos_corr_action_corr_cond",
        "cos_clean_action_corr_cond",
        "cos_corr_action_clean_cond",
        "norm_clean_action",
        "norm_corr_action",
        "norm_clean_dyn",
        "norm_corr_dyn",
        "norm_clean_cond",
        "norm_corr_cond",
    ]
    summary: dict[str, Any] = {"num_records": len(records)}
    for key in metric_keys:
        values = [float(record[key]) for record in records if key in record]
        summary[key] = {"mean": _mean(values), "std": _std(values)}
    groups = sorted({g for record in records for name in record.get("group_norms", {}).values() for g in name})
    if groups:
        summary["group_norms"] = {}
        loss_names = sorted({name for record in records for name in record.get("group_norms", {})})
        for loss_name in loss_names:
            summary["group_norms"][loss_name] = {}
            for group in groups:
                vals = [
                    float(record["group_norms"][loss_name].get(group, 0.0))
                    for record in records
                    if loss_name in record.get("group_norms", {})
                ]
                summary["group_norms"][loss_name][group] = {"mean": _mean(vals), "std": _std(vals)}
    return summary


def _load_stage1_checkpoint_if_needed(model: SmolVLALatentPolicy, checkpoint_path: str) -> None:
    path = str(checkpoint_path).strip()
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Gradient Probe Summary ===")
    print(f"records: {summary.get('num_records', 0)}")
    for key in (
        "clean_action_loss",
        "corr_action_loss",
        "clean_dyn_loss",
        "corr_dyn_loss",
        "clean_cond_loss",
        "corr_cond_loss",
    ):
        item = summary.get(key, {})
        print(f"{key:28s} mean={item.get('mean', float('nan')):.6g} std={item.get('std', float('nan')):.6g}")
    print("\n-- cosine --")
    for key in (
        "cos_clean_corr_action",
        "cos_clean_action_clean_dyn",
        "cos_corr_action_corr_dyn",
        "cos_clean_action_corr_dyn",
        "cos_corr_action_clean_dyn",
        "cos_clean_action_clean_cond",
        "cos_corr_action_corr_cond",
        "cos_clean_action_corr_cond",
        "cos_corr_action_clean_cond",
    ):
        item = summary.get(key, {})
        print(f"{key:28s} mean={item.get('mean', float('nan')):.6g} std={item.get('std', float('nan')):.6g}")
    print("\n-- grad norm --")
    for key in (
        "norm_clean_action",
        "norm_corr_action",
        "norm_clean_dyn",
        "norm_corr_dyn",
        "norm_clean_cond",
        "norm_corr_cond",
    ):
        item = summary.get(key, {})
        print(f"{key:28s} mean={item.get('mean', float('nan')):.6g} std={item.get('std', float('nan')):.6g}")


def main() -> None:
    args = build_argparser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
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

    if not str(args.failure_table_paths_json).strip():
        raise ValueError("--failure_table_paths_json is required for gradient probe")
    failure_table_paths = load_failure_table_paths(args.failure_table_paths_json)
    corr_batch_size = (
        int(args.probe_corr_batch_size)
        if args.probe_corr_batch_size is not None
        else int(max(1, round(float(args.batch_size) * float(args.failure_corr_batch_ratio))))
    )
    evac_sample_size = load_evac_sample_size(args.evac_config)
    failure_cfg = MultiTaskFailureDatasetConfig(
        act_chunk_size=int(args.act_chunk_size),
        prefix_steps=int(args.prefix_steps),
        future_offset=int(args.future_offset),
        sample_phase_window_len=int(args.sample_phase_window_len),
        start_margin=int(args.start_margin),
        perturb_eef_fail_gain=float(args.act_aligned_perturb_eef_fail_gain),
        perturb_rot_max_deg=float(args.act_aligned_perturb_rot_max_deg),
        evac_sample_size=evac_sample_size,
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
        batch_size=corr_batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    base_policy, preprocess, postprocess = build_base_policy(args)
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
        load_rollout_model=True,
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

    correction_builder = ACTAlignedCorrectionBuilder(
        cfg=build_act_aligned_cfg_from_args(args, max_action_len=int(args.act_chunk_size)),
        urdf_path=args.urdf_path,
        curobo_left_yml=args.curobo_left_yml,
        curobo_right_yml=args.curobo_right_yml,
        device=device,
        shared_evac_model=teacher.model,
        shared_evac_config=teacher.cfg,
    )

    params = _named_trainable_params(model)
    print(f"trainable params: {sum(param.numel() for _, param in params):,}")
    print(f"normal samples: {len(normal_dataset):,}; correction samples index: {len(correction_dataset):,}")
    print(f"probe batches: {int(args.probe_batches)}; clean batch={int(args.batch_size)}; corr batch={corr_batch_size}")

    normal_iter = iter(normal_loader)
    correction_iter = iter(correction_loader)
    raw_cache: dict[tuple[str, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()

    progress = tqdm(range(int(args.probe_batches)), desc="gradient-probe")
    for batch_idx in progress:
        try:
            normal_raw_batch = next(normal_iter)
        except StopIteration:
            normal_iter = iter(normal_loader)
            normal_raw_batch = next(normal_iter)
        try:
            correction_raw_batch = next(correction_iter)
        except StopIteration:
            correction_iter = iter(correction_loader)
            correction_raw_batch = next(correction_iter)

        normal_raw_batch = _as_device_batch(normal_raw_batch, device)
        correction_raw_batch = _as_device_batch(correction_raw_batch, device)

        clean_samples, clean_future_images, clean_action_prefix, _, _ = build_stage1_samples_from_raw_batch(
            raw_batch=normal_raw_batch,
            latent_model=model,
            preprocess=preprocess,
            instruction_type=args.instruction_type,
            prefix_steps=int(args.prefix_steps),
        )
        corr_samples, corr_future_images, corr_action_prefix, corr_skip_reasons, _ = build_stage1_correction_samples(
            correction_raw_batch=correction_raw_batch,
            latent_model=model,
            preprocess=preprocess,
            postprocess=postprocess,
            instruction_type=args.instruction_type,
            args=args,
            correction_builder=correction_builder,
            teacher=teacher,
            norm_stats=norm_stats,
            raw_cache=raw_cache,
            step_debug_dir=None,
        )
        skip_counter.update(corr_skip_reasons)
        if not corr_samples:
            print(f"batch {batch_idx}: all correction samples skipped; reasons={dict(corr_skip_reasons)}")
            continue

        clean_batch = stack_smolvla_batches(clean_samples)
        corr_batch = stack_smolvla_batches(corr_samples)
        clean_future_tensor = torch.stack(clean_future_images, dim=0).to(device=device, dtype=torch.float32)
        corr_future_tensor = torch.stack(corr_future_images, dim=0).to(device=device, dtype=torch.float32)
        clean_future_latent = teacher.encode_image(clean_future_tensor)
        corr_future_latent = teacher.encode_image(corr_future_tensor)
        clean_prefix_tensor = torch.cat(clean_action_prefix, dim=0).to(device=device)
        corr_prefix_tensor = torch.cat(corr_action_prefix, dim=0).to(device=device)

        clean_action_losses = _action_loss_unreduced(model, clean_batch, clean_batch[ACTION]).mean(dim=(1, 2))
        corr_action_losses = _action_loss_unreduced(model, corr_batch, corr_batch[ACTION]).mean(dim=(1, 2))
        clean_cond_losses = _condition_loss_unreduced(
            model, clean_batch, clean_batch[ACTION], clean_future_latent
        ).mean(dim=(1, 2))
        corr_cond_losses = _condition_loss_unreduced(
            model, corr_batch, corr_batch[ACTION], corr_future_latent
        ).mean(dim=(1, 2))
        clean_dyn_losses = _dynamics_loss_unreduced(
            model, clean_batch, clean_future_latent, clean_prefix_tensor
        )
        corr_dyn_losses = _dynamics_loss_unreduced(
            model, corr_batch, corr_future_latent, corr_prefix_tensor
        )

        losses = {
            "clean_action": clean_action_losses.mean(),
            "corr_action": corr_action_losses.mean(),
            "clean_cond": clean_cond_losses.mean(),
            "corr_cond": corr_cond_losses.mean(),
            "clean_dyn": clean_dyn_losses.mean(),
            "corr_dyn": corr_dyn_losses.mean(),
        }
        grads = {}
        group_norms = {}
        for name, loss in losses.items():
            vec, norms = _grad_vector(model, params, loss)
            grads[name] = vec
            group_norms[name] = norms

        record = {
            "batch_idx": int(batch_idx),
            "num_clean": int(len(clean_samples)),
            "num_corr": int(len(corr_samples)),
            "skip_reasons": dict(corr_skip_reasons),
            "clean_action_loss": float(losses["clean_action"].item()),
            "corr_action_loss": float(losses["corr_action"].item()),
            "clean_cond_loss": float(losses["clean_cond"].item()),
            "corr_cond_loss": float(losses["corr_cond"].item()),
            "clean_dyn_loss": float(losses["clean_dyn"].item()),
            "corr_dyn_loss": float(losses["corr_dyn"].item()),
            "norm_clean_action": float(grads["clean_action"].norm().item()),
            "norm_corr_action": float(grads["corr_action"].norm().item()),
            "norm_clean_cond": float(grads["clean_cond"].norm().item()),
            "norm_corr_cond": float(grads["corr_cond"].norm().item()),
            "norm_clean_dyn": float(grads["clean_dyn"].norm().item()),
            "norm_corr_dyn": float(grads["corr_dyn"].norm().item()),
            "cos_clean_corr_action": _cosine(grads["clean_action"], grads["corr_action"]),
            "cos_clean_action_clean_dyn": _cosine(grads["clean_action"], grads["clean_dyn"]),
            "cos_corr_action_corr_dyn": _cosine(grads["corr_action"], grads["corr_dyn"]),
            "cos_clean_action_corr_dyn": _cosine(grads["clean_action"], grads["corr_dyn"]),
            "cos_corr_action_clean_dyn": _cosine(grads["corr_action"], grads["clean_dyn"]),
            "cos_clean_action_clean_cond": _cosine(grads["clean_action"], grads["clean_cond"]),
            "cos_corr_action_corr_cond": _cosine(grads["corr_action"], grads["corr_cond"]),
            "cos_clean_action_corr_cond": _cosine(grads["clean_action"], grads["corr_cond"]),
            "cos_corr_action_clean_cond": _cosine(grads["corr_action"], grads["clean_cond"]),
            "group_norms": group_norms,
        }
        if _parse_bool(args.probe_include_full_vectors):
            record["grad_vectors"] = {key: value.tolist() for key, value in grads.items()}
        records.append(record)
        progress.set_postfix(
            corr=len(corr_samples),
            ca=f"{record['clean_action_loss']:.4f}",
            qa=f"{record['corr_action_loss']:.4f}",
            cos=f"{record['cos_clean_corr_action']:.3f}",
        )

        del clean_batch, corr_batch, clean_future_latent, corr_future_latent, grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = _summarize_records(records)
    payload = {
        "args": vars(args),
        "skip_reasons_total": dict(skip_counter),
        "summary": summary,
        "records": records,
    }
    _print_summary(summary)
    if skip_counter:
        print(f"\nskip reasons: {dict(skip_counter)}")

    output_json = str(args.probe_output_json).strip()
    if output_json:
        output_path = Path(output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
