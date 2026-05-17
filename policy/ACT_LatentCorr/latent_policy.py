from __future__ import annotations

from dataclasses import dataclass
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.ACT.act_policy import ACTPolicy

from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .latent_modules import (
    ActionConditionedPredictor,
    DynamicsWarmup,
    LatentToTokenAdapter,
    LatentActionDecoder,
    LatentProjector,
    ObservationConditionedFuturePredictor,
    ResidualLatentAdapter,
)


@dataclass
class Stage1LossOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_action_conditioned_teacher: torch.Tensor
    loss_action_conditioned_pred: torch.Tensor
    loss_action_conditioned: torch.Tensor
    loss_condition_token: torch.Tensor
    loss_latent: torch.Tensor
    loss_dynamics: torch.Tensor
    loss_wm_action_current: torch.Tensor
    loss_wm_action_future: torch.Tensor
    loss_bridge_future: torch.Tensor
    beta_dynamics: float
    progress: float
    lambda_teacher: float
    lambda_pred: float
    lambda_latent: float
    lambda_token: float
    cond_keep_ratio: float
    teacher_token_norm_mean: torch.Tensor
    pred_token_norm_mean: torch.Tensor
    delta_action_mean: torch.Tensor
    delta_action_normal_mean: torch.Tensor
    delta_action_failure_mean: torch.Tensor
    delta_action_teacher_mean: torch.Tensor
    delta_action_teacher_normal_mean: torch.Tensor
    delta_action_teacher_failure_mean: torch.Tensor


@dataclass
class Stage2LossOutput:
    loss: torch.Tensor
    loss_correct: torch.Tensor
    loss_dynamics: torch.Tensor
    loss_retain: torch.Tensor
    loss_bridge: torch.Tensor
    beta_dynamics: float
    alpha_latent: float


@dataclass
class Stage2PreparedContext:
    z_proj: torch.Tensor
    act_action_chunk: torch.Tensor
    action_dev_norm: torch.Tensor
    action_dev_raw: torch.Tensor
    qpos_err_norm: torch.Tensor
    z_wm_sim_shared: torch.Tensor
    z_hat_next: torch.Tensor


class ACTLatentStage1(nn.Module):
    """
    Stage-1 latent warmup model.

    This module keeps original ACT network untouched and adds:
    - Projector: ACT feature -> WM latent space
    - Predictor: action-conditioned next latent prediction
    - Action decoder: latent -> action prefix
    """

    def __init__(
        self,
        act_args: dict[str, Any],
        latent_model_cfg: LatentModelConfig,
        latent_loss_cfg: LatentLossConfig,
        warmup_cfg: DynamicsWarmupConfig,
    ):
        super().__init__()
        # Pass a fully constructed Namespace to bypass original ACT argparse parsing.
        self.base_act = ACTPolicy(act_args, RoboTwin_Config=Namespace(**act_args))
        self.latent_model_cfg = latent_model_cfg
        self.latent_loss_cfg = latent_loss_cfg
        self.beta_scheduler = DynamicsWarmup(warmup_cfg)

        self.projector: LatentProjector | None = None
        self.wm_adapter: ResidualLatentAdapter | None = None
        self.readout_adapter: ResidualLatentAdapter | None = None
        self.predictor: ActionConditionedPredictor | None = None
        self.obs_future_predictor: ObservationConditionedFuturePredictor | None = None
        self.token_adapter: LatentToTokenAdapter | None = None
        self.action_decoder: LatentActionDecoder | None = None
        self.act_condition_proj: nn.Linear | None = None
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self._freeze_base_act = False
        self._latent_target_hw: tuple[int, int] | None = None

    def forward(self, *args, mode: str = "stage1", **kwargs):
        if mode == "stage1":
            return self.forward_stage1(*args, **kwargs)
        if mode == "stage2":
            return self.forward_stage2(*args, **kwargs)
        raise ValueError(f"Unsupported forward mode: {mode}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_base_act:
            self.base_act.eval()
        return self

    def set_base_act_frozen(self, frozen: bool = True) -> None:
        self._freeze_base_act = bool(frozen)
        for param in self.base_act.parameters():
            param.requires_grad = not frozen
        if frozen:
            self.base_act.eval()

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        mask = (~is_pad).unsqueeze(-1).float()
        diff = (pred - target).abs() * mask
        return diff.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        mask = (~is_pad).unsqueeze(-1).float().expand_as(pred)
        diff = (pred - target).pow(2) * mask
        return diff.sum() / mask.sum().clamp_min(1.0)

    def _normalize_image_like_act(self, image: torch.Tensor) -> torch.Tensor:
        mean = self._imagenet_mean.to(device=image.device, dtype=image.dtype)
        std = self._imagenet_std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def _extract_act_feature(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract ACT visual map before transformer decoding.
        image: (B, num_cam, 3, H, W) in [0,1]
        """
        model = self.base_act.model
        if model.backbones is None:
            raise RuntimeError("ACT model backbones is None; image-based ACT is required.")

        image = self._normalize_image_like_act(image)
        all_cam_features = []
        for cam_id in range(image.shape[1]):
            features, _ = model.backbones[0](image[:, cam_id])
            features = features[0]
            all_cam_features.append(model.input_proj(features))
        return torch.cat(all_cam_features, dim=3)

    def _lazy_init_heads_from_shapes(
        self,
        z_act: torch.Tensor,
        wm_channels: int,
        target_hw: tuple[int, int],
    ) -> None:
        if self.projector is not None:
            return

        _, c_act, _, _ = z_act.shape
        c_wm = int(wm_channels)
        h_wm, w_wm = int(target_hw[0]), int(target_hw[1])
        self._latent_target_hw = (h_wm, w_wm)
        self.projector = LatentProjector(
            in_channels=c_act,
            out_channels=c_wm,
            mid_channels=self.latent_model_cfg.projector_mid_channels,
        ).to(z_act.device)
        self.wm_adapter = ResidualLatentAdapter(
            channels=c_wm,
            mid_channels=self.latent_model_cfg.wm_adapter_mid_channels,
        ).to(z_act.device)
        self.readout_adapter = ResidualLatentAdapter(
            channels=c_wm,
            mid_channels=self.latent_model_cfg.readout_adapter_mid_channels,
        ).to(z_act.device)
        self.predictor = ActionConditionedPredictor(
            channels=c_wm,
            prefix_steps=self.latent_model_cfg.prefix_steps,
            action_dim=self.latent_model_cfg.action_dim,
            mlp_hidden_dim=self.latent_model_cfg.predictor_mlp_hidden,
            num_blocks=self.latent_model_cfg.predictor_num_blocks,
        ).to(z_act.device)
        self.obs_future_predictor = ObservationConditionedFuturePredictor(
            channels=c_wm,
            state_dim=self.latent_model_cfg.state_dim,
            hidden_dim=self.latent_model_cfg.predictor_mlp_hidden,
            num_blocks=self.latent_model_cfg.predictor_num_blocks,
        ).to(z_act.device)
        self.action_decoder = LatentActionDecoder(
            in_channels=c_wm,
            hidden_dim=self.latent_model_cfg.action_decoder_hidden,
            prefix_steps=self.latent_model_cfg.prefix_steps,
            action_dim=self.latent_model_cfg.action_dim,
        ).to(z_act.device)
        hidden_dim = int(self.base_act.model.transformer.d_model)
        self.token_adapter = LatentToTokenAdapter(
            latent_channels=c_wm,
            token_dim=hidden_dim,
            hidden_dim=self.latent_model_cfg.token_adapter_hidden_dim,
            num_layers=self.latent_model_cfg.token_adapter_num_layers,
            dropout=self.latent_model_cfg.token_adapter_dropout,
        ).to(z_act.device)
        self.act_condition_proj = nn.Linear(c_wm, hidden_dim).to(z_act.device)
        nn.init.zeros_(self.act_condition_proj.bias)

    def _lazy_init_heads(self, z_act: torch.Tensor, z_wm: torch.Tensor):
        self._lazy_init_heads_from_shapes(
            z_act=z_act,
            wm_channels=int(z_wm.shape[1]),
            target_hw=(int(z_wm.shape[-2]), int(z_wm.shape[-1])),
        )

    def initialize_latent_heads(self, image_t: torch.Tensor, wm_teacher) -> None:
        with torch.no_grad():
            z_act = self._extract_act_feature(image_t)
            z_wm = wm_teacher.encode_image(image_t[:, 0])
        self._lazy_init_heads(z_act, z_wm)

    def initialize_latent_heads_from_shapes(self, image_t: torch.Tensor, wm_channels: int, target_hw: tuple[int, int]) -> None:
        with torch.no_grad():
            z_act = self._extract_act_feature(image_t)
        self._lazy_init_heads_from_shapes(z_act, wm_channels=wm_channels, target_hw=target_hw)

    def _shared_wm_latent(self, z_wm: torch.Tensor) -> torch.Tensor:
        assert self.wm_adapter is not None
        return self.wm_adapter(z_wm)

    def _canonical_action_latent(self, z: torch.Tensor) -> torch.Tensor:
        assert self.readout_adapter is not None
        return self.readout_adapter(z)

    def decode_action_latent(self, z: torch.Tensor) -> torch.Tensor:
        assert self.action_decoder is not None
        return self.action_decoder(self._canonical_action_latent(z))

    def _latent_to_act_token(self, z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        assert self.act_condition_proj is not None
        pooled = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        token = self.act_condition_proj(pooled)
        return token * float(scale)

    def _predict_future_latent(self, z_t: torch.Tensor, qpos_t: torch.Tensor) -> torch.Tensor:
        assert self.obs_future_predictor is not None
        return self.obs_future_predictor(z_t, qpos_t)

    def _adapt_latent_to_token(self, z_t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        assert self.token_adapter is not None
        token = self.token_adapter(z_t)
        return token * float(scale)

    @staticmethod
    def _normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        denom = target.pow(2).mean().detach().clamp_min(eps)
        return (pred - target).pow(2).mean() / denom

    @staticmethod
    def _cosine_latent_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()

    def _latent_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = str(self.latent_loss_cfg.latent_loss_type).strip().lower()
        if loss_type == "normalized_mse":
            return self._normalized_mse(pred, target)
        if loss_type == "cosine":
            return self._cosine_latent_loss(pred, target)
        if loss_type == "raw_mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unsupported latent_loss_type={self.latent_loss_cfg.latent_loss_type}")

    def _token_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = str(self.latent_loss_cfg.token_loss_type).strip().lower()
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unsupported token_loss_type={self.latent_loss_cfg.token_loss_type}")

    def _schedule_weight(self, progress: float, start_ratio: float, end_ratio: float, start_value: float, end_value: float) -> float:
        p = float(max(0.0, min(1.0, progress)))
        if p <= start_ratio:
            return float(start_value)
        if p >= end_ratio:
            return float(end_value)
        if end_ratio <= start_ratio:
            return float(end_value)
        r = (p - start_ratio) / float(end_ratio - start_ratio)
        return float(start_value * (1.0 - r) + end_value * r)

    def _compute_stage1_schedule(self, progress: float) -> dict[str, float]:
        cfg = self.latent_loss_cfg
        p = float(max(0.0, min(1.0, progress)))
        if p < cfg.teacher_decay_start_ratio:
            lambda_teacher = float(cfg.lambda_teacher_max)
        elif p < cfg.teacher_decay_end_ratio:
            lambda_teacher = self._schedule_weight(
                p,
                cfg.teacher_decay_start_ratio,
                cfg.teacher_decay_end_ratio,
                cfg.lambda_teacher_max,
                0.0,
            )
        else:
            lambda_teacher = 0.0

        if p < cfg.pred_warmup_start_ratio:
            lambda_pred = 0.0
        elif p < cfg.pred_warmup_end_ratio:
            lambda_pred = self._schedule_weight(
                p,
                cfg.pred_warmup_start_ratio,
                cfg.pred_warmup_end_ratio,
                0.0,
                cfg.lambda_pred_max,
            )
        else:
            lambda_pred = float(cfg.lambda_pred_max)

        if p < cfg.latent_warmup_end_ratio:
            lambda_latent = self._schedule_weight(
                p,
                0.0,
                cfg.latent_warmup_end_ratio,
                0.0,
                cfg.lambda_latent_max,
            )
        else:
            lambda_latent = float(cfg.lambda_latent_max)

        if p < cfg.token_decay_start_ratio:
            lambda_token = float(cfg.lambda_token_init)
        elif p < cfg.token_decay_end_ratio:
            lambda_token = self._schedule_weight(
                p,
                cfg.token_decay_start_ratio,
                cfg.token_decay_end_ratio,
                cfg.lambda_token_init,
                cfg.lambda_token_late,
            )
        else:
            lambda_token = float(cfg.lambda_token_late)
        return {
            "progress": p,
            "lambda_teacher": lambda_teacher,
            "lambda_pred": lambda_pred,
            "lambda_latent": lambda_latent,
            "lambda_token": lambda_token,
        }

    def predict_act_chunk(self, qpos_t: torch.Tensor, image_t: torch.Tensor) -> torch.Tensor:
        return self.base_act(qpos_t, image_t)

    def _build_stage2_targets(
        self,
        act_action_chunk: torch.Tensor,
        act_is_pad: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        correction_target_chunk: torch.Tensor | None,
        correction_is_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_steps = int(self.latent_model_cfg.prefix_steps)
        if correction_target_chunk is None:
            if correction_target_prefix is None or is_pad_prefix is None:
                raise ValueError("Either full correction chunk or correction prefix must be provided.")
            correction_target_chunk = act_action_chunk.detach().clone()
            correction_target_chunk[:, :prefix_steps] = correction_target_prefix
            correction_is_pad = torch.ones_like(act_is_pad, dtype=torch.bool)
            correction_is_pad[:, :prefix_steps] = is_pad_prefix
        elif correction_is_pad is None:
            raise ValueError("correction_is_pad must be provided when correction_target_chunk is provided.")

        if correction_target_prefix is None:
            correction_target_prefix = correction_target_chunk[:, :prefix_steps]
        if is_pad_prefix is None:
            is_pad_prefix = correction_is_pad[:, :prefix_steps]
        return correction_target_chunk, correction_is_pad, correction_target_prefix, is_pad_prefix

    def predict_act_chunk_conditioned(
        self,
        qpos_t: torch.Tensor,
        image_t: torch.Tensor,
        z_condition: torch.Tensor,
        alpha_latent: float = 1.0,
    ) -> torch.Tensor:
        token = self._latent_to_act_token(z_condition, scale=alpha_latent)
        return self.base_act(qpos_t, image_t, external_latent_input=token)

    def predict_act_chunk_predicted_conditioned(
        self,
        qpos_t: torch.Tensor,
        image_t: torch.Tensor,
    ) -> torch.Tensor:
        z_act = self._extract_act_feature(image_t)
        if self.projector is None or self.wm_adapter is None:
            raise RuntimeError("Latent heads are not initialized before pred inference.")
        # Match the spatial latent size learned during initialization without querying WM again.
        if not hasattr(self, "_latent_target_hw") or self._latent_target_hw is None:
            raise RuntimeError("Missing cached latent target resolution for pred inference.")
        z_proj = self.projector(z_act, target_hw=self._latent_target_hw)
        z_pred = self._predict_future_latent(z_proj, qpos_t)
        token = self._adapt_latent_to_token(z_pred, scale=1.0)
        return self.base_act(qpos_t, image_t, external_latent_input=token)

    def forward_stage1(
        self,
        image_t: torch.Tensor,
        image_t1: torch.Tensor,
        qpos_t: torch.Tensor,
        qpos_future_norm: torch.Tensor | None,
        act_action_chunk: torch.Tensor,
        act_is_pad: torch.Tensor,
        action_prefix: torch.Tensor,
        action_future_prefix: torch.Tensor,
        is_pad_prefix: torch.Tensor,
        is_pad_future_prefix: torch.Tensor,
        wm_teacher,
        global_step: int,
        use_act_head_conditioning: bool = False,
        teacher_current_latent: torch.Tensor | None = None,
        future_teacher_latent: torch.Tensor | None = None,
        is_failure_sample: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> Stage1LossOutput:
        if image_t.ndim != 5:
            raise ValueError(f"image_t must be (B,num_cam,3,H,W), got {image_t.shape}")

        # ACT visual feature (B, C_act, H_act, W_act)
        z_act = self._extract_act_feature(image_t)
        z_act_latent = z_act.detach() if self.latent_loss_cfg.detach_act_feature_for_latent else z_act

        # Teacher latent from EVAC final VAE latent, head camera only for stage-1.
        if teacher_current_latent is None:
            if wm_teacher is None:
                raise ValueError("wm_teacher is required when teacher_current_latent is not provided")
            z_wm_t = wm_teacher.encode_image(image_t[:, 0])
        else:
            z_wm_t = teacher_current_latent
        if future_teacher_latent is None:
            if wm_teacher is None:
                raise ValueError("wm_teacher is required when future_teacher_latent is not provided")
            z_wm_t1 = wm_teacher.encode_image(image_t1[:, 0])
        else:
            z_wm_t1 = future_teacher_latent
        self._lazy_init_heads(z_act_latent, z_wm_t)
        assert self.projector is not None
        assert self.wm_adapter is not None
        assert self.readout_adapter is not None
        assert self.predictor is not None
        assert self.obs_future_predictor is not None
        assert self.action_decoder is not None
        assert self.token_adapter is not None

        target_hw = (z_wm_t.shape[-2], z_wm_t.shape[-1])
        z_proj = self.projector(z_act_latent, target_hw=target_hw)
        if self.latent_loss_cfg.use_raw_wm_targets:
            z_wm_shared_t = z_wm_t.detach()
            z_wm_shared_t1 = z_wm_t1.detach()
        else:
            # Freeze EVAC features themselves, but keep the shared-manifold adapter trainable.
            z_wm_shared_t = self._shared_wm_latent(z_wm_t.detach())
            z_wm_shared_t1 = self._shared_wm_latent(z_wm_t1.detach())

        act_loss_dict = self.base_act(qpos_t, image_t, actions=act_action_chunk, is_pad=act_is_pad)
        loss_action = act_loss_dict["loss"]

        z_pred_input = z_proj.detach() if self.latent_loss_cfg.use_projector_detach_for_predictor else z_proj
        z_hat_next = self._predict_future_latent(z_pred_input, qpos_t)
        # `stopgrad_wm_teacher` is already enforced by detaching the EVAC latent before
        # feeding it into `wm_adapter`. Do not detach again here, otherwise `wm_adapter`
        # becomes log-only and DDP sees it as used-by-output but not used-by-loss.
        teacher_latent_target = z_wm_shared_t1
        loss_latent = self._latent_loss(z_hat_next, teacher_latent_target)
        # Stage-1 mainline no longer carries an independent dynamics term.
        loss_dynamics = torch.zeros_like(loss_latent)

        progress = float(max(0.0, min(1.0, float(global_step))))
        weights = self._compute_stage1_schedule(progress)
        loss_action_conditioned = torch.zeros_like(loss_action)
        loss_action_conditioned_teacher = torch.zeros_like(loss_action)
        loss_action_conditioned_pred = torch.zeros_like(loss_action)
        teacher_cond_source = z_wm_shared_t1
        teacher_cond_token_raw = self._latent_to_act_token(teacher_cond_source, scale=1.0)
        pred_cond_source = z_hat_next.detach() if self.latent_loss_cfg.stopgrad_adapter_input_for_token_loss else z_hat_next
        pred_cond_token_raw = self._adapt_latent_to_token(pred_cond_source, scale=1.0)
        teacher_token_target = teacher_cond_token_raw.detach() if self.latent_loss_cfg.stopgrad_teacher_token else teacher_cond_token_raw
        loss_condition_token = self._token_loss(pred_cond_token_raw, teacher_token_target)
        cond_keep_ratio = 1.0
        teacher_token_norm_mean = teacher_cond_token_raw.norm(dim=1).mean()
        pred_token_norm_mean = pred_cond_token_raw.norm(dim=1).mean()
        delta_action_mean = torch.zeros_like(loss_action)
        delta_action_normal_mean = torch.zeros_like(loss_action)
        delta_action_failure_mean = torch.zeros_like(loss_action)
        delta_action_teacher_mean = torch.zeros_like(loss_action)
        delta_action_teacher_normal_mean = torch.zeros_like(loss_action)
        delta_action_teacher_failure_mean = torch.zeros_like(loss_action)
        if use_act_head_conditioning:
            keep_mask = torch.ones(
                (pred_cond_token_raw.shape[0], 1),
                device=pred_cond_token_raw.device,
                dtype=pred_cond_token_raw.dtype,
            )
            if (
                is_failure_sample is not None
                and self.training
                and float(self.latent_loss_cfg.normal_condition_keep_prob) < 1.0
            ):
                keep_prob = float(max(0.0, min(1.0, self.latent_loss_cfg.normal_condition_keep_prob)))
                normal_mask = (~is_failure_sample.bool()).view(-1, 1)
                if keep_prob <= 0.0:
                    keep_mask = torch.where(normal_mask, torch.zeros_like(keep_mask), keep_mask)
                elif keep_prob < 1.0:
                    sampled = (torch.rand_like(keep_mask) < keep_prob).to(pred_cond_token_raw.dtype)
                    keep_mask = torch.where(normal_mask, sampled, keep_mask)
                cond_keep_ratio = float(keep_mask.mean().item())

            teacher_token_for_action = teacher_cond_token_raw * keep_mask
            pred_token_for_action = pred_cond_token_raw * keep_mask
            if weights["lambda_teacher"] > 0.0:
                teacher_loss_dict = self.base_act(
                    qpos_t,
                    image_t,
                    actions=act_action_chunk,
                    is_pad=act_is_pad,
                    external_latent_input=teacher_token_for_action,
                    return_per_sample=True,
                )
                loss_action_conditioned_teacher = teacher_loss_dict["loss"]
            if weights["lambda_pred"] > 0.0:
                pred_loss_dict = self.base_act(
                    qpos_t,
                    image_t,
                    actions=act_action_chunk,
                    is_pad=act_is_pad,
                    external_latent_input=pred_token_for_action,
                    return_per_sample=True,
                )
                loss_action_conditioned_pred = pred_loss_dict["loss"]
            loss_action_conditioned = (
                float(weights["lambda_teacher"]) * loss_action_conditioned_teacher
                + float(weights["lambda_pred"]) * loss_action_conditioned_pred
            )
            with torch.no_grad():
                a_base = self.base_act(qpos_t, image_t)
                a_teacher = self.base_act(
                    qpos_t,
                    image_t,
                    external_latent_input=teacher_token_for_action,
                )
                a_pred = self.base_act(
                    qpos_t,
                    image_t,
                    external_latent_input=pred_token_for_action,
                )
                per_sample_delta_teacher = (a_teacher - a_base).abs().mean(dim=(1, 2))
                per_sample_delta = (a_pred - a_base).abs().mean(dim=(1, 2))
                delta_action_teacher_mean = per_sample_delta_teacher.mean()
                delta_action_mean = per_sample_delta.mean()
                if is_failure_sample is not None:
                    failure_mask = is_failure_sample.bool()
                    if bool((~failure_mask).any().item()):
                        delta_action_teacher_normal_mean = per_sample_delta_teacher[~failure_mask].mean()
                        delta_action_normal_mean = per_sample_delta[~failure_mask].mean()
                    if bool(failure_mask.any().item()):
                        delta_action_teacher_failure_mean = per_sample_delta_teacher[failure_mask].mean()
                        delta_action_failure_mean = per_sample_delta[failure_mask].mean()

        wm_action_current = self.decode_action_latent(z_wm_shared_t)
        loss_wm_action_current = self._masked_l1(wm_action_current, action_prefix, is_pad_prefix)

        wm_action_future = self.decode_action_latent(z_wm_shared_t1)
        loss_wm_action_future = self._masked_l1(wm_action_future, action_future_prefix, is_pad_future_prefix)

        bridge_future = self.decode_action_latent(z_hat_next)
        loss_bridge_future = self._masked_l1(bridge_future, action_future_prefix, is_pad_future_prefix)

        beta_dyn = 0.0
        loss = (
            (self.latent_loss_cfg.lambda_action * loss_action)
            + loss_action_conditioned
            + (float(weights["lambda_token"]) * loss_condition_token)
            + (float(weights["lambda_latent"]) * loss_latent)
        )

        output = Stage1LossOutput(
            loss=loss,
            loss_action=loss_action.detach(),
            loss_action_conditioned_teacher=loss_action_conditioned_teacher.detach(),
            loss_action_conditioned_pred=loss_action_conditioned_pred.detach(),
            loss_action_conditioned=loss_action_conditioned.detach(),
            loss_condition_token=loss_condition_token.detach(),
            loss_latent=loss_latent.detach(),
            loss_dynamics=loss_dynamics.detach(),
            loss_wm_action_current=loss_wm_action_current.detach(),
            loss_wm_action_future=loss_wm_action_future.detach(),
            loss_bridge_future=loss_bridge_future.detach(),
            beta_dynamics=beta_dyn,
            progress=float(weights["progress"]),
            lambda_teacher=float(weights["lambda_teacher"]),
            lambda_pred=float(weights["lambda_pred"]),
            lambda_latent=float(weights["lambda_latent"]),
            lambda_token=float(weights["lambda_token"]),
            cond_keep_ratio=cond_keep_ratio,
            teacher_token_norm_mean=teacher_token_norm_mean.detach(),
            pred_token_norm_mean=pred_token_norm_mean.detach(),
            delta_action_mean=delta_action_mean.detach(),
            delta_action_normal_mean=delta_action_normal_mean.detach(),
            delta_action_failure_mean=delta_action_failure_mean.detach(),
            delta_action_teacher_mean=delta_action_teacher_mean.detach(),
            delta_action_teacher_normal_mean=delta_action_teacher_normal_mean.detach(),
            delta_action_teacher_failure_mean=delta_action_teacher_failure_mean.detach(),
        )
        if not return_dict:
            return output
        return {
            "loss": output.loss,
            "loss_action": output.loss_action,
            "loss_action_conditioned_teacher": output.loss_action_conditioned_teacher,
            "loss_action_conditioned_pred": output.loss_action_conditioned_pred,
            "loss_action_conditioned": output.loss_action_conditioned,
            "loss_condition_token": output.loss_condition_token,
            "loss_latent": output.loss_latent,
            "loss_dynamics": output.loss_dynamics,
            "loss_wm_action_current": output.loss_wm_action_current,
            "loss_wm_action_future": output.loss_wm_action_future,
            "loss_bridge_future": output.loss_bridge_future,
            "beta_dynamics": torch.tensor(float(output.beta_dynamics), device=loss.device),
            "progress": torch.tensor(float(output.progress), device=loss.device),
            "lambda_teacher": torch.tensor(float(output.lambda_teacher), device=loss.device),
            "lambda_pred": torch.tensor(float(output.lambda_pred), device=loss.device),
            "lambda_latent": torch.tensor(float(output.lambda_latent), device=loss.device),
            "lambda_token": torch.tensor(float(output.lambda_token), device=loss.device),
            "cond_keep_ratio": torch.tensor(float(output.cond_keep_ratio), device=loss.device),
            "teacher_token_norm_mean": output.teacher_token_norm_mean,
            "pred_token_norm_mean": output.pred_token_norm_mean,
            "delta_action_mean": output.delta_action_mean,
            "delta_action_normal_mean": output.delta_action_normal_mean,
            "delta_action_failure_mean": output.delta_action_failure_mean,
            "delta_action_teacher_mean": output.delta_action_teacher_mean,
            "delta_action_teacher_normal_mean": output.delta_action_teacher_normal_mean,
            "delta_action_teacher_failure_mean": output.delta_action_teacher_failure_mean,
        }

    def prepare_stage2_context(
        self,
        image_t: torch.Tensor,
        qpos_t: torch.Tensor,
        qpos_raw: torch.Tensor,
        wm_teacher,
        raw_data,
        fk,
        norm_stats: dict[str, Any],
        ddim_steps: int = 27,
        is_pad_prefix: torch.Tensor | None = None,
        external_action_dev_norm: torch.Tensor | None = None,
        external_action_dev_raw: torch.Tensor | None = None,
        external_qpos_err_norm: torch.Tensor | None = None,
        external_action_is_pad_prefix: torch.Tensor | None = None,
        external_z_wm_sim: torch.Tensor | None = None,
    ) -> Stage2PreparedContext:
        if external_z_wm_sim is None and image_t.shape[0] != 1:
            raise ValueError("prepare_stage2_context without precomputed rollout latent expects batch size 1 per call")

        z_act = self._extract_act_feature(image_t)
        z_act_latent = z_act.detach() if self.latent_loss_cfg.detach_act_feature_for_latent else z_act
        if external_z_wm_sim is not None:
            z_wm_ref = external_z_wm_sim.to(device=z_act_latent.device, dtype=torch.float32)
        else:
            z_wm_ref = wm_teacher.encode_image(image_t[:, 0])
        self._lazy_init_heads(z_act_latent, z_wm_ref)
        assert self.projector is not None
        assert self.wm_adapter is not None
        assert self.readout_adapter is not None
        assert self.predictor is not None
        assert self.action_decoder is not None
        assert self.act_condition_proj is not None

        target_hw = (z_wm_ref.shape[-2], z_wm_ref.shape[-1])
        z_proj = self.projector(z_act_latent, target_hw=target_hw)
        act_action_chunk = self.predict_act_chunk(qpos_t, image_t)
        if external_action_dev_norm is not None:
            action_dev_norm = external_action_dev_norm
        else:
            action_dev_norm = act_action_chunk[:, : self.latent_model_cfg.prefix_steps]

        action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=action_dev_norm.dtype, device=action_dev_norm.device)
        action_std = torch.as_tensor(norm_stats["action_std"], dtype=action_dev_norm.dtype, device=action_dev_norm.device)
        if external_action_dev_raw is not None:
            action_dev_raw = external_action_dev_raw
        else:
            action_dev_raw = action_dev_norm.detach() * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
            action_dev_raw = action_dev_raw.clone()
            action_dev_raw[..., 6] = action_dev_raw[..., 6].clamp(0.0, 1.0)
            action_dev_raw[..., 13] = action_dev_raw[..., 13].clamp(0.0, 1.0)
        qpos_mean = torch.as_tensor(
            norm_stats["qpos_mean"],
            dtype=action_dev_raw.dtype,
            device=action_dev_raw.device,
        )
        qpos_std = torch.as_tensor(
            norm_stats["qpos_std"],
            dtype=action_dev_raw.dtype,
            device=action_dev_raw.device,
        )
        if external_qpos_err_norm is not None:
            qpos_err_norm = external_qpos_err_norm
        else:
            qpos_err_norm = (action_dev_raw[:, -1, :] - qpos_mean.view(1, -1)) / qpos_std.view(1, -1)

        if external_z_wm_sim is not None:
            z_wm_sim = external_z_wm_sim.to(device=z_proj.device, dtype=z_proj.dtype)
        else:
            z_wm_sim = wm_teacher.rollout_latent_from_actions(
                curr_image=image_t[0, 0],
                curr_qpos_raw=qpos_raw[0],
                action_prefix_raw=action_dev_raw[0],
                raw_data=raw_data,
                fk=fk,
                ddim_steps=ddim_steps,
            )
            z_wm_sim = z_wm_sim.to(device=z_proj.device, dtype=z_proj.dtype)
        z_wm_sim_shared = self._shared_wm_latent(z_wm_sim.detach())

        predictor_is_pad = external_action_is_pad_prefix
        if predictor_is_pad is None:
            predictor_is_pad = is_pad_prefix
        if predictor_is_pad is None:
            predictor_is_pad = torch.zeros(
                (image_t.shape[0], action_dev_norm.shape[1]),
                dtype=torch.bool,
                device=action_dev_norm.device,
            )
        z_hat_next = self.predictor(z_proj, action_dev_norm.detach(), is_pad=predictor_is_pad)
        return Stage2PreparedContext(
            z_proj=z_proj,
            act_action_chunk=act_action_chunk,
            action_dev_norm=action_dev_norm,
            action_dev_raw=action_dev_raw,
            qpos_err_norm=qpos_err_norm,
            z_wm_sim_shared=z_wm_sim_shared,
            z_hat_next=z_hat_next,
        )

    def compute_stage2_loss(
        self,
        stage2_ctx: Stage2PreparedContext,
        image_t: torch.Tensor,
        qpos_t: torch.Tensor,
        act_action_chunk: torch.Tensor,
        act_is_pad: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        global_step: int,
        retain_weight: float = 0.0,
        bridge_weight: float = 0.0,
        use_act_head_correction: bool = False,
        base_anchor_chunk: torch.Tensor | None = None,
        correction_target_chunk: torch.Tensor | None = None,
        correction_is_pad: torch.Tensor | None = None,
    ) -> Stage2LossOutput:
        z_wm_sim_shared = stage2_ctx.z_wm_sim_shared
        z_hat_next = stage2_ctx.z_hat_next
        loss_dynamics = F.mse_loss(z_hat_next, z_wm_sim_shared.detach())
        alpha_latent = self.beta_scheduler.weight(global_step)
        beta_dyn = alpha_latent * self.latent_loss_cfg.beta_dynamics_max
        correction_target_chunk, correction_is_pad, correction_target_prefix, is_pad_prefix = self._build_stage2_targets(
            act_action_chunk=act_action_chunk,
            act_is_pad=act_is_pad,
            correction_target_prefix=correction_target_prefix,
            is_pad_prefix=is_pad_prefix,
            correction_target_chunk=correction_target_chunk,
            correction_is_pad=correction_is_pad,
        )

        if use_act_head_correction:
            # Train ACT-head correction on teacher latent first. This keeps the
            # correction target semantically tied to the simulated error state
            # instead of immediately forcing the head to consume a still-noisy
            # predicted latent.
            corr_token = self._latent_to_act_token(z_wm_sim_shared.detach(), scale=alpha_latent)
            corr_loss_dict = self.base_act(
                stage2_ctx.qpos_err_norm,
                image_t,
                actions=correction_target_chunk,
                is_pad=correction_is_pad,
                external_latent_input=corr_token,
            )
            loss_correct = corr_loss_dict["loss"]

            loss_retain = torch.zeros_like(loss_correct)
            if retain_weight > 0.0:
                anchor_actions = base_anchor_chunk if base_anchor_chunk is not None else act_action_chunk
                retain_loss_dict = self.base_act(
                    qpos_t,
                    image_t,
                    actions=anchor_actions.detach(),
                    is_pad=act_is_pad,
                )
                loss_retain = retain_loss_dict["loss"]

            loss_bridge = torch.zeros_like(loss_correct)
        else:
            corr_hat = self.decode_action_latent(z_wm_sim_shared.detach())
            loss_correct = self._masked_mse(corr_hat, correction_target_prefix, is_pad_prefix)

            loss_retain = torch.zeros_like(loss_correct)
            if retain_weight > 0.0:
                retain_loss_dict = self.base_act(
                    qpos_t,
                    image_t,
                    actions=act_action_chunk,
                    is_pad=act_is_pad,
                )
                loss_retain = retain_loss_dict["loss"]

            loss_bridge = torch.zeros_like(loss_correct)
            if bridge_weight > 0.0:
                bridge_hat = self.decode_action_latent(z_hat_next)
                loss_bridge = self._masked_mse(bridge_hat, correction_target_prefix, is_pad_prefix)

        loss = loss_correct + (beta_dyn * loss_dynamics) + (retain_weight * loss_retain) + (bridge_weight * loss_bridge)

        return Stage2LossOutput(
            loss=loss,
            loss_correct=loss_correct,
            loss_dynamics=loss_dynamics,
            loss_retain=loss_retain,
            loss_bridge=loss_bridge,
            beta_dynamics=beta_dyn,
            alpha_latent=alpha_latent,
        )

    def compute_stage2_loss_act_only(
        self,
        qpos_t: torch.Tensor,
        image_t: torch.Tensor,
        act_action_chunk: torch.Tensor,
        act_is_pad: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        retain_weight: float = 0.0,
        base_anchor_chunk: torch.Tensor | None = None,
        correction_target_chunk: torch.Tensor | None = None,
        correction_is_pad: torch.Tensor | None = None,
    ) -> Stage2LossOutput:
        correction_target_chunk, correction_is_pad, _, _ = self._build_stage2_targets(
            act_action_chunk=act_action_chunk,
            act_is_pad=act_is_pad,
            correction_target_prefix=correction_target_prefix,
            is_pad_prefix=is_pad_prefix,
            correction_target_chunk=correction_target_chunk,
            correction_is_pad=correction_is_pad,
        )

        corr_loss_dict = self.base_act(
            qpos_t,
            image_t,
            actions=correction_target_chunk,
            is_pad=correction_is_pad,
        )
        loss_correct = corr_loss_dict["loss"]

        loss_retain = torch.zeros_like(loss_correct)
        if retain_weight > 0.0:
            anchor_actions = base_anchor_chunk if base_anchor_chunk is not None else act_action_chunk
            retain_loss_dict = self.base_act(
                qpos_t,
                image_t,
                actions=anchor_actions.detach(),
                is_pad=act_is_pad,
            )
            loss_retain = retain_loss_dict["loss"]

        loss_dynamics = torch.zeros_like(loss_correct)
        loss_bridge = torch.zeros_like(loss_correct)
        loss = loss_correct + (retain_weight * loss_retain)
        return Stage2LossOutput(
            loss=loss,
            loss_correct=loss_correct,
            loss_dynamics=loss_dynamics,
            loss_retain=loss_retain,
            loss_bridge=loss_bridge,
            beta_dynamics=0.0,
            alpha_latent=0.0,
        )

    def forward_stage2(
        self,
        image_t: torch.Tensor,
        qpos_t: torch.Tensor,
        qpos_raw: torch.Tensor,
        act_action_chunk: torch.Tensor,
        act_is_pad: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        wm_teacher,
        raw_data,
        fk,
        norm_stats: dict[str, Any],
        global_step: int,
        ddim_steps: int = 27,
        retain_weight: float = 0.0,
        bridge_weight: float = 0.0,
        use_act_head_correction: bool = False,
        base_anchor_chunk: torch.Tensor | None = None,
        correction_target_chunk: torch.Tensor | None = None,
        correction_is_pad: torch.Tensor | None = None,
        external_action_dev_norm: torch.Tensor | None = None,
        external_action_dev_raw: torch.Tensor | None = None,
        external_qpos_err_norm: torch.Tensor | None = None,
        external_action_is_pad_prefix: torch.Tensor | None = None,
        external_z_wm_sim: torch.Tensor | None = None,
        act_like_loss_only: bool = False,
    ) -> Stage2LossOutput:
        if act_like_loss_only:
            return self.compute_stage2_loss_act_only(
                qpos_t=qpos_t,
                image_t=image_t,
                act_action_chunk=act_action_chunk,
                act_is_pad=act_is_pad,
                correction_target_prefix=correction_target_prefix,
                is_pad_prefix=is_pad_prefix,
                retain_weight=retain_weight,
                base_anchor_chunk=base_anchor_chunk,
                correction_target_chunk=correction_target_chunk,
                correction_is_pad=correction_is_pad,
            )
        stage2_ctx = self.prepare_stage2_context(
            image_t=image_t,
            qpos_t=qpos_t,
            qpos_raw=qpos_raw,
            wm_teacher=wm_teacher,
            raw_data=raw_data,
            fk=fk,
            norm_stats=norm_stats,
            ddim_steps=ddim_steps,
            is_pad_prefix=is_pad_prefix,
            external_action_dev_norm=external_action_dev_norm,
            external_action_dev_raw=external_action_dev_raw,
            external_qpos_err_norm=external_qpos_err_norm,
            external_action_is_pad_prefix=external_action_is_pad_prefix,
            external_z_wm_sim=external_z_wm_sim,
        )
        return self.compute_stage2_loss(
            stage2_ctx=stage2_ctx,
            image_t=image_t,
            qpos_t=qpos_t,
            act_action_chunk=act_action_chunk,
            act_is_pad=act_is_pad,
            correction_target_prefix=correction_target_prefix,
            is_pad_prefix=is_pad_prefix,
            global_step=global_step,
            retain_weight=retain_weight,
            bridge_weight=bridge_weight,
            use_act_head_correction=use_act_head_correction,
            base_anchor_chunk=base_anchor_chunk,
            correction_target_chunk=correction_target_chunk,
            correction_is_pad=correction_is_pad,
        )
