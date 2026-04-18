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
    LatentActionDecoder,
    LatentProjector,
    ResidualLatentAdapter,
)


@dataclass
class Stage1LossOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_action_conditioned: torch.Tensor
    loss_align: torch.Tensor
    loss_dynamics: torch.Tensor
    loss_wm_action_current: torch.Tensor
    loss_wm_action_future: torch.Tensor
    loss_bridge_future: torch.Tensor
    beta_dynamics: float
    alpha_latent: float


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

    def _lazy_init_heads(self, z_act: torch.Tensor, z_wm: torch.Tensor):
        if self.projector is not None:
            return

        _, c_act, _, _ = z_act.shape
        _, c_wm, _, _ = z_wm.shape
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
        self.action_decoder = LatentActionDecoder(
            in_channels=c_wm,
            hidden_dim=self.latent_model_cfg.action_decoder_hidden,
            prefix_steps=self.latent_model_cfg.prefix_steps,
            action_dim=self.latent_model_cfg.action_dim,
        ).to(z_act.device)
        hidden_dim = int(self.base_act.model.transformer.d_model)
        self.act_condition_proj = nn.Linear(c_wm, hidden_dim).to(z_act.device)
        nn.init.zeros_(self.act_condition_proj.weight)
        nn.init.zeros_(self.act_condition_proj.bias)

    def initialize_latent_heads(self, image_t: torch.Tensor, wm_teacher) -> None:
        with torch.no_grad():
            z_act = self._extract_act_feature(image_t)
            z_wm = wm_teacher.encode_image(image_t[:, 0])
        self._lazy_init_heads(z_act, z_wm)

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
        future_teacher_latent: torch.Tensor | None = None,
    ) -> Stage1LossOutput:
        if image_t.ndim != 5:
            raise ValueError(f"image_t must be (B,num_cam,3,H,W), got {image_t.shape}")

        # ACT visual feature (B, C_act, H_act, W_act)
        z_act = self._extract_act_feature(image_t)
        z_act_latent = z_act.detach() if self.latent_loss_cfg.detach_act_feature_for_latent else z_act

        # Teacher latent from EVAC final VAE latent, head camera only for stage-1.
        z_wm_t = wm_teacher.encode_image(image_t[:, 0])
        if future_teacher_latent is None:
            z_wm_t1 = wm_teacher.encode_image(image_t1[:, 0])
        else:
            z_wm_t1 = future_teacher_latent
        self._lazy_init_heads(z_act_latent, z_wm_t)
        assert self.projector is not None
        assert self.wm_adapter is not None
        assert self.readout_adapter is not None
        assert self.predictor is not None
        assert self.action_decoder is not None

        target_hw = (z_wm_t.shape[-2], z_wm_t.shape[-1])
        z_proj = self.projector(z_act_latent, target_hw=target_hw)
        if self.latent_loss_cfg.use_raw_wm_targets:
            z_wm_shared_t = z_wm_t.detach()
            z_wm_shared_t1 = z_wm_t1.detach()
        else:
            z_wm_shared_t = self._shared_wm_latent(z_wm_t.detach())
            z_wm_shared_t1 = self._shared_wm_latent(z_wm_t1.detach())
        loss_align = F.mse_loss(z_proj, z_wm_shared_t.detach())

        act_loss_dict = self.base_act(qpos_t, image_t, actions=act_action_chunk, is_pad=act_is_pad)
        loss_action = act_loss_dict["loss"]

        z_for_pred = z_proj.detach() if self.latent_loss_cfg.use_projector_detach_for_predictor else z_proj
        z_hat_next = self.predictor(z_for_pred, action_prefix, is_pad=is_pad_prefix)
        loss_dynamics = F.mse_loss(z_hat_next, z_wm_shared_t1.detach())

        alpha_latent = self.beta_scheduler.weight(global_step)
        loss_action_conditioned = torch.zeros_like(loss_action)
        if use_act_head_conditioning:
            conditioned_qpos = qpos_future_norm if qpos_future_norm is not None else qpos_t
            cond_token = self._latent_to_act_token(z_hat_next, scale=alpha_latent)
            cond_loss_dict = self.base_act(
                conditioned_qpos,
                image_t,
                actions=act_action_chunk,
                is_pad=act_is_pad,
                external_latent_input=cond_token,
            )
            loss_action_conditioned = cond_loss_dict["loss"]

        wm_action_current = self.decode_action_latent(z_wm_shared_t)
        loss_wm_action_current = self._masked_l1(wm_action_current, action_prefix, is_pad_prefix)

        wm_action_future = self.decode_action_latent(z_wm_shared_t1)
        loss_wm_action_future = self._masked_l1(wm_action_future, action_future_prefix, is_pad_future_prefix)

        bridge_future = self.decode_action_latent(z_hat_next)
        loss_bridge_future = self._masked_l1(bridge_future, action_future_prefix, is_pad_future_prefix)

        beta_dyn = alpha_latent * self.latent_loss_cfg.beta_dynamics_max
        loss = (
            (self.latent_loss_cfg.lambda_action * loss_action)
            + (self.latent_loss_cfg.lambda_action_conditioned * loss_action_conditioned)
            + (self.latent_loss_cfg.lambda_align * loss_align)
            + (self.latent_loss_cfg.lambda_wm_action_current * loss_wm_action_current)
            + (self.latent_loss_cfg.lambda_wm_action_future * loss_wm_action_future)
            + (beta_dyn * loss_dynamics)
            + (beta_dyn * self.latent_loss_cfg.lambda_bridge_future * loss_bridge_future)
        )

        return Stage1LossOutput(
            loss=loss,
            loss_action=loss_action,
            loss_action_conditioned=loss_action_conditioned,
            loss_align=loss_align,
            loss_dynamics=loss_dynamics,
            loss_wm_action_current=loss_wm_action_current,
            loss_wm_action_future=loss_wm_action_future,
            loss_bridge_future=loss_bridge_future,
            beta_dynamics=beta_dyn,
            alpha_latent=alpha_latent,
        )

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
