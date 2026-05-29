from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .common import prepare_openpi_imports
from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .stage1_model import PI0LatentStage1


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def load_stage1_checkpoint_payload(stage1_ckpt: str | Path) -> dict[str, Any]:
    ckpt_path = Path(stage1_ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def build_stage1_model_from_checkpoint(
    stage1_ckpt: str | Path,
    *,
    device: str | torch.device = "cuda:0",
) -> tuple[PI0LatentStage1, dict[str, Any], dict[str, Any]]:
    prepare_openpi_imports()

    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training import config as openpi_config

    payload = load_stage1_checkpoint_payload(stage1_ckpt)
    ckpt_args = dict(payload.get("args", {}))
    if "train_config_name" not in ckpt_args:
        raise KeyError("Stage1 checkpoint args missing train_config_name")

    train_cfg = openpi_config.get_config(ckpt_args["train_config_name"])
    resolved_weight_path = ckpt_args.get("resolved_pytorch_weight_path")
    if resolved_weight_path:
        base_pi0 = train_cfg.model.load_pytorch(train_cfg, str(resolved_weight_path))
    else:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        base_pi0 = PI0Pytorch(config=train_cfg.model)

    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=int(ckpt_args.get("projector_mid_channels", 256)),
        predictor_hidden_dim=int(ckpt_args.get("predictor_hidden_dim", 512)),
        predictor_num_blocks=int(ckpt_args.get("predictor_num_blocks", 3)),
        action_dim=14,
        model_action_dim=int(ckpt_args.get("model_action_dim", 32)),
        action_horizon=int(ckpt_args.get("action_horizon", 50)),
        prefix_steps=int(ckpt_args.get("prefix_steps", 16)),
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=float(ckpt_args.get("lambda_action", 1.0)),
        lambda_action_conditioned=float(ckpt_args.get("lambda_action_conditioned", 1.0)),
        lambda_align=float(ckpt_args.get("lambda_align", 0.0)),
        beta_dynamics_max=float(ckpt_args.get("beta_dynamics_max", 1.0)),
        use_projector_detach_for_predictor=_to_bool(ckpt_args.get("use_projector_detach_for_predictor", True)),
        detach_act_feature_for_latent=_to_bool(ckpt_args.get("detach_act_feature_for_latent", False)),
        use_raw_wm_targets=_to_bool(ckpt_args.get("use_raw_wm_targets", True)),
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=int(ckpt_args.get("dyn_zero_steps", 0)),
        ramp_steps=int(ckpt_args.get("dyn_ramp_steps", 2000)),
        max_weight=1.0,
        curve=str(ckpt_args.get("dyn_warmup_curve", "cosine")),
        unit=str(ckpt_args.get("dyn_schedule_unit", "step")),
    )

    device = torch.device(device)
    model = PI0LatentStage1(
        base_pi0=base_pi0.to(device),
        tokenizer=PaligemmaTokenizer(max_len=train_cfg.model.max_token_len),
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
        discrete_state_input=bool(train_cfg.model.discrete_state_input),
        freeze_base_pi0=_to_bool(ckpt_args.get("freeze_base_pi0", False)),
    ).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    model.eval()
    meta = {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "step": int(payload.get("step", 0)),
        "sample_count": int(payload.get("sample_count", 0)),
    }
    return model, ckpt_args, meta
