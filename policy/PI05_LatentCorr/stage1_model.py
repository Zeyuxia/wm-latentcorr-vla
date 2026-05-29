from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .stage1_modules import ActionConditionedPredictor, DynamicsWarmup, LatentProjector


@dataclass
class Stage1LossOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_action_conditioned: torch.Tensor
    loss_dynamics: torch.Tensor
    loss_align: torch.Tensor
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
    pi0_action_chunk: torch.Tensor
    action_dev_norm: torch.Tensor
    action_dev_raw: torch.Tensor
    qpos_err_norm: torch.Tensor
    z_wm_rollout: torch.Tensor
    z_hat_next: torch.Tensor


class PI0LatentStage1(nn.Module):
    def __init__(
        self,
        *,
        base_pi0: nn.Module,
        tokenizer,
        latent_model_cfg: LatentModelConfig,
        latent_loss_cfg: LatentLossConfig,
        warmup_cfg: DynamicsWarmupConfig,
        discrete_state_input: bool = True,
        freeze_base_pi0: bool = False,
    ):
        super().__init__()
        self.base_pi0 = base_pi0
        self.tokenizer = tokenizer
        self.latent_model_cfg = latent_model_cfg
        self.latent_loss_cfg = latent_loss_cfg
        self.warmup_cfg = warmup_cfg
        self.prefix_steps = int(latent_model_cfg.prefix_steps)
        self.action_dim = int(latent_model_cfg.action_dim)
        self.model_action_dim = int(latent_model_cfg.model_action_dim)
        self.action_horizon = int(latent_model_cfg.action_horizon)
        self.discrete_state_input = bool(discrete_state_input)
        self.warmup = DynamicsWarmup(
            warmup_cfg.zero_steps,
            warmup_cfg.ramp_steps,
            warmup_cfg.max_weight,
            warmup_cfg.curve,
        )

        self.projector: LatentProjector | None = None
        self.predictor: ActionConditionedPredictor | None = None
        self.condition_proj: nn.Linear | None = None
        self._freeze_base_pi0 = bool(freeze_base_pi0)
        if self._freeze_base_pi0:
            self.freeze_base_model()

    def forward(self, *args, mode: str = "stage1", **kwargs):
        if mode == "stage1":
            return self.forward_stage1(*args, **kwargs)
        if mode == "stage2":
            return self.forward_stage2(*args, **kwargs)
        raise ValueError(f"Unsupported forward mode: {mode}")

    def freeze_base_model(self) -> None:
        for param in self.base_pi0.parameters():
            param.requires_grad = False
        self.base_pi0.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_base_pi0:
            self.base_pi0.eval()
        return self

    @staticmethod
    def _to_openpi_image(image_t: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, H, W] images in [0, 1] to OpenPI torch-preprocess format [B, C, H, W] in [-1, 1]."""
        return image_t.clamp(0.0, 1.0).mul(2.0).sub(1.0).contiguous()

    def _make_observation(
        self,
        image_t: torch.Tensor,
        qpos_norm: torch.Tensor,
        prompts: Sequence[str],
    ) -> SimpleNamespace:
        batch_size = image_t.shape[0]
        base_image = self._to_openpi_image(image_t)
        zero_image = torch.full_like(base_image, -1.0)

        tokenized_prompt = []
        tokenized_prompt_mask = []
        qpos_np = qpos_norm.detach().cpu().numpy()
        for idx, prompt in enumerate(prompts):
            tokens, mask = self.tokenizer.tokenize(prompt, qpos_np[idx] if self.discrete_state_input else None)
            tokenized_prompt.append(torch.from_numpy(tokens))
            tokenized_prompt_mask.append(torch.from_numpy(mask))

        state = torch.zeros(batch_size, self.model_action_dim, dtype=qpos_norm.dtype, device=qpos_norm.device)
        state[:, : qpos_norm.shape[-1]] = qpos_norm
        return SimpleNamespace(
            images={
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zero_image,
                "right_wrist_0_rgb": zero_image,
            },
            image_masks={
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=qpos_norm.device),
                "left_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=qpos_norm.device),
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=qpos_norm.device),
            },
            state=state,
            tokenized_prompt=torch.stack(tokenized_prompt, dim=0).to(device=qpos_norm.device, dtype=torch.long),
            tokenized_prompt_mask=torch.stack(tokenized_prompt_mask, dim=0).to(device=qpos_norm.device, dtype=torch.bool),
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def _extract_visual_feature(self, image_t: torch.Tensor) -> torch.Tensor:
        batch_size = image_t.shape[0]
        observation = SimpleNamespace(
            images={
                "base_0_rgb": self._to_openpi_image(image_t),
                "left_wrist_0_rgb": torch.full(
                    (batch_size, 3, image_t.shape[2], image_t.shape[3]),
                    -1.0,
                    dtype=torch.float32,
                    device=image_t.device,
                ),
                "right_wrist_0_rgb": torch.full(
                    (batch_size, 3, image_t.shape[2], image_t.shape[3]),
                    -1.0,
                    dtype=torch.float32,
                    device=image_t.device,
                ),
            },
            image_masks={
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=image_t.device),
                "left_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=image_t.device),
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=image_t.device),
            },
            state=torch.zeros(batch_size, self.model_action_dim, dtype=torch.float32, device=image_t.device),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
            token_ar_mask=None,
            token_loss_mask=None,
        )
        images, _, _, _, _ = self.base_pi0._preprocess_observation(observation, train=False)
        base_image = images[0]
        img_tokens = self.base_pi0.paligemma_with_expert.embed_image(base_image)
        token_count = int(img_tokens.shape[1])
        side = int(math.isqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"Expected square image-token grid, got token_count={token_count}")
        return img_tokens.transpose(1, 2).reshape(img_tokens.shape[0], img_tokens.shape[2], side, side).to(torch.float32)

    def _lazy_init_heads(self, z_img: torch.Tensor, z_wm: torch.Tensor) -> None:
        if self.projector is not None:
            return
        token_dim = int(z_img.shape[1])
        wm_channels = int(z_wm.shape[1])
        self.projector = LatentProjector(
            token_dim,
            wm_channels,
            int(self.latent_model_cfg.projector_mid_channels),
        ).to(z_img.device)
        self.predictor = ActionConditionedPredictor(
            wm_channels,
            self.prefix_steps,
            self.action_dim,
            int(self.latent_model_cfg.predictor_hidden_dim),
            int(self.latent_model_cfg.predictor_num_blocks),
        ).to(z_img.device)
        self.condition_proj = nn.Linear(wm_channels, token_dim).to(z_img.device)
        nn.init.zeros_(self.condition_proj.weight)
        nn.init.zeros_(self.condition_proj.bias)

    def _latent_to_condition_token(self, z: torch.Tensor, scale: float) -> torch.Tensor:
        assert self.condition_proj is not None
        pooled = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        return self.condition_proj(pooled) * float(scale)

    def _normalize_prefix_actions(self, action_prefix: torch.Tensor) -> torch.Tensor:
        if action_prefix.shape[-1] == self.action_dim:
            return action_prefix
        if action_prefix.shape[-1] < self.action_dim:
            raise ValueError(
                f"action_prefix last dim must be >= action_dim={self.action_dim}, got {tuple(action_prefix.shape)}"
            )
        return action_prefix[..., : self.action_dim]

    def _compute_pi0_loss(
        self,
        observation: SimpleNamespace,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        latent_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self.base_pi0._preprocess_observation(observation, train=True)
        noise = self.base_pi0.sample_noise(actions.shape, actions.device)
        time = self.base_pi0.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base_pi0.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if latent_token is not None:
            prefix_embs = torch.cat([prefix_embs, latent_token[:, None, :].to(prefix_embs.dtype)], dim=1)
            extra_pad = torch.ones(actions.shape[0], 1, dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device)
            extra_att = torch.zeros(actions.shape[0], 1, dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
            prefix_pad_masks = torch.cat([prefix_pad_masks, extra_pad], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, extra_att], dim=1)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.base_pi0.embed_suffix(state, x_t, time)
        if self.base_pi0.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        att_2d_masks = self.base_pi0._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.base_pi0.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)
        v_t = self.base_pi0.action_out_proj(suffix_out)
        per_token_loss = F.mse_loss(u_t, v_t, reduction="none")
        masked = per_token_loss * action_mask
        return masked.sum() / action_mask.sum().clamp_min(1.0)

    @staticmethod
    def _bool_pad_to_action_mask(is_pad: torch.Tensor, action_dim: int) -> torch.Tensor:
        return (~is_pad).unsqueeze(-1).to(dtype=torch.float32).expand(-1, -1, action_dim)

    def _build_stage2_targets(
        self,
        *,
        pi0_action_chunk: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        correction_target_chunk: torch.Tensor | None,
        correction_is_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_steps = int(self.prefix_steps)
        if correction_target_chunk is None:
            if correction_target_prefix is None or is_pad_prefix is None:
                raise ValueError("Either correction_target_chunk or correction_target_prefix + is_pad_prefix is required.")
            correction_target_chunk = pi0_action_chunk.detach().clone()
            correction_target_chunk[:, :prefix_steps, : correction_target_prefix.shape[-1]] = correction_target_prefix
            correction_is_pad = torch.zeros(
                correction_target_chunk.shape[:2],
                dtype=torch.bool,
                device=correction_target_chunk.device,
            )
            correction_is_pad[:, :prefix_steps] = is_pad_prefix
        elif correction_is_pad is None:
            raise ValueError("correction_is_pad must be provided when correction_target_chunk is provided.")

        if correction_target_prefix is None:
            correction_target_prefix = correction_target_chunk[:, :prefix_steps, : self.action_dim]
        if is_pad_prefix is None:
            is_pad_prefix = correction_is_pad[:, :prefix_steps]
        return correction_target_chunk, correction_is_pad, correction_target_prefix, is_pad_prefix

    @torch.no_grad()
    def initialize_latent_heads(self, image_t: torch.Tensor, wm_teacher) -> None:
        z_img = self._extract_visual_feature(image_t)
        z_img_latent = z_img.detach() if self.latent_loss_cfg.detach_act_feature_for_latent else z_img
        z_wm = wm_teacher.encode_image(image_t).to(device=image_t.device, dtype=torch.float32)
        self._lazy_init_heads(z_img_latent, z_wm)

    @torch.no_grad()
    def project_current_latent(self, image_t: torch.Tensor, wm_teacher) -> torch.Tensor:
        z_img = self._extract_visual_feature(image_t)
        z_img_latent = z_img.detach() if self.latent_loss_cfg.detach_act_feature_for_latent else z_img
        z_wm = wm_teacher.encode_image(image_t).to(device=image_t.device, dtype=torch.float32)
        self._lazy_init_heads(z_img_latent, z_wm)
        assert self.projector is not None
        return self.projector(z_img_latent, target_hw=(z_wm.shape[-2], z_wm.shape[-1]))

    @torch.no_grad()
    def predict_future_latent(
        self,
        *,
        image_t: torch.Tensor,
        action_prefix: torch.Tensor,
        wm_teacher,
        is_pad_prefix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_proj = self.project_current_latent(image_t, wm_teacher)
        assert self.predictor is not None
        action_prefix = self._normalize_prefix_actions(action_prefix)
        predictor_input = z_proj.detach() if self.latent_loss_cfg.use_projector_detach_for_predictor else z_proj
        z_hat_next = self.predictor(predictor_input, action_prefix, is_pad=is_pad_prefix)
        return z_proj, z_hat_next

    @torch.no_grad()
    def latent_to_condition_token(self, z: torch.Tensor, alpha_latent: float = 1.0) -> torch.Tensor:
        return self._latent_to_condition_token(z, scale=alpha_latent)

    @torch.no_grad()
    def predict_pi0_chunk(
        self,
        *,
        qpos_norm: torch.Tensor,
        image_t: torch.Tensor,
        prompts: Sequence[str] | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if prompts is None:
            prompts = [""] * image_t.shape[0]
        observation = self._make_observation(image_t, qpos_norm, prompts)
        return self.base_pi0.sample_actions(image_t.device, observation, num_steps=num_steps)

    @torch.no_grad()
    def predict_pi0_chunk_conditioned(
        self,
        *,
        qpos_norm: torch.Tensor,
        image_t: torch.Tensor,
        latent_z: torch.Tensor,
        alpha_latent: float = 1.0,
        prompts: Sequence[str] | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if prompts is None:
            prompts = [""] * image_t.shape[0]
        observation = self._make_observation(image_t, qpos_norm, prompts)
        latent_token = self._latent_to_condition_token(latent_z, scale=alpha_latent)
        images, img_masks, lang_tokens, lang_masks, state = self.base_pi0._preprocess_observation(observation, train=False)
        noise = self.base_pi0.sample_noise(
            (image_t.shape[0], self.action_horizon, self.model_action_dim),
            image_t.device,
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base_pi0.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_embs = torch.cat([prefix_embs, latent_token[:, None, :].to(prefix_embs.dtype)], dim=1)
        extra_pad = torch.ones(image_t.shape[0], 1, dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device)
        extra_att = torch.zeros(image_t.shape[0], 1, dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
        prefix_pad_masks = torch.cat([prefix_pad_masks, extra_pad], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, extra_att], dim=1)

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.base_pi0._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.base_pi0.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.base_pi0.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / float(num_steps), dtype=torch.float32, device=image_t.device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=image_t.device)
        while time >= -dt / 2:
            expanded_time = time.expand(image_t.shape[0])
            v_t = self.base_pi0.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def forward_stage1(
        self,
        *,
        image_t: torch.Tensor,
        image_t1: torch.Tensor,
        qpos_t_norm: torch.Tensor,
        qpos_t1_norm: torch.Tensor,
        act_action_chunk: torch.Tensor,
        act_action_mask: torch.Tensor,
        action_prefix: torch.Tensor,
        is_pad_prefix: torch.Tensor,
        prompts: Sequence[str],
        wm_teacher,
        global_step: float,
        use_act_head_conditioning: bool = True,
        future_teacher_latent: torch.Tensor | None = None,
        condition_on_current_observation: bool = True,
        lambda_action_conditioned_override: float | None = None,
    ) -> Stage1LossOutput:
        z_img = self._extract_visual_feature(image_t)
        z_img_latent = z_img.detach() if self.latent_loss_cfg.detach_act_feature_for_latent else z_img
        z_wm_t = wm_teacher.encode_image(image_t).to(device=image_t.device, dtype=torch.float32)
        if future_teacher_latent is None:
            z_wm_t1 = wm_teacher.encode_image(image_t1).to(device=image_t.device, dtype=torch.float32)
        else:
            z_wm_t1 = future_teacher_latent.to(device=image_t.device, dtype=torch.float32)
        self._lazy_init_heads(z_img_latent, z_wm_t)
        assert self.projector is not None
        assert self.predictor is not None

        z_proj = self.projector(z_img_latent, target_hw=(z_wm_t.shape[-2], z_wm_t.shape[-1]))
        z_wm_target_t = z_wm_t.detach()
        z_wm_target_t1 = z_wm_t1.detach()
        loss_align = F.mse_loss(z_proj, z_wm_target_t) if self.latent_loss_cfg.lambda_align > 0.0 else z_proj.new_zeros(())

        z_for_pred = z_proj.detach() if self.latent_loss_cfg.use_projector_detach_for_predictor else z_proj
        z_hat_next = self.predictor(
            z_for_pred,
            self._normalize_prefix_actions(action_prefix),
            is_pad=is_pad_prefix,
        )

        beta_scale = self.warmup.weight(float(global_step))
        alpha_latent = beta_scale
        beta_dynamics = beta_scale * float(self.latent_loss_cfg.beta_dynamics_max)

        obs_current = self._make_observation(image_t, qpos_t_norm, prompts)
        conditioned_qpos = qpos_t_norm if condition_on_current_observation else qpos_t1_norm
        obs_conditioned = self._make_observation(image_t, conditioned_qpos, prompts)

        loss_action = self._compute_pi0_loss(obs_current, act_action_chunk, act_action_mask)
        loss_action_conditioned = loss_action.new_zeros(())
        if use_act_head_conditioning:
            cond_token = self._latent_to_condition_token(z_hat_next, scale=alpha_latent)
            loss_action_conditioned = self._compute_pi0_loss(
                obs_conditioned,
                act_action_chunk,
                act_action_mask,
                latent_token=cond_token,
            )
        loss_dynamics = F.mse_loss(z_hat_next, z_wm_target_t1)
        lambda_action_conditioned = (
            float(self.latent_loss_cfg.lambda_action_conditioned)
            if lambda_action_conditioned_override is None
            else float(lambda_action_conditioned_override)
        )
        loss = (
            float(self.latent_loss_cfg.lambda_action) * loss_action
            + lambda_action_conditioned * loss_action_conditioned
            + beta_dynamics * loss_dynamics
        )
        if self.latent_loss_cfg.lambda_align > 0.0:
            loss = loss + float(self.latent_loss_cfg.lambda_align) * loss_align

        return Stage1LossOutput(
            loss=loss,
            loss_action=loss_action,
            loss_action_conditioned=loss_action_conditioned,
            loss_dynamics=loss_dynamics,
            loss_align=loss_align,
            beta_dynamics=beta_dynamics,
            alpha_latent=alpha_latent,
        )

    def prepare_stage2_context(
        self,
        *,
        image_t: torch.Tensor,
        qpos_t_norm: torch.Tensor,
        qpos_raw: torch.Tensor,
        wm_teacher,
        raw_data: dict[str, torch.Tensor | float | int | list | tuple] | None,
        norm_stats: dict[str, torch.Tensor | list | tuple],
        ddim_steps: int = 27,
        is_pad_prefix: torch.Tensor | None = None,
        external_action_dev_norm: torch.Tensor | None = None,
        external_action_dev_raw: torch.Tensor | None = None,
        external_qpos_err_norm: torch.Tensor | None = None,
        external_action_is_pad_prefix: torch.Tensor | None = None,
        external_z_wm_rollout: torch.Tensor | None = None,
        prompts: Sequence[str] | None = None,
        num_steps: int = 10,
        fk=None,
    ) -> Stage2PreparedContext:
        if prompts is None:
            prompts = [""] * image_t.shape[0]
        if external_z_wm_rollout is None and image_t.shape[0] != 1:
            raise ValueError("prepare_stage2_context without precomputed rollout latent expects batch size 1.")

        pi0_action_chunk = self.predict_pi0_chunk(
            qpos_norm=qpos_t_norm,
            image_t=image_t,
            prompts=prompts,
            num_steps=num_steps,
        )
        if external_action_dev_norm is None:
            action_dev_norm = pi0_action_chunk[:, : self.prefix_steps, : self.action_dim]
        else:
            action_dev_norm = external_action_dev_norm.to(device=image_t.device, dtype=torch.float32)

        action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=image_t.device)
        action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=image_t.device)
        qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=image_t.device)
        qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=image_t.device)

        if external_action_dev_raw is None:
            action_dev_raw = action_dev_norm * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
            action_dev_raw = action_dev_raw.clone()
            if action_dev_raw.shape[-1] >= 14:
                action_dev_raw[..., 6] = action_dev_raw[..., 6].clamp(0.0, 1.0)
                action_dev_raw[..., 13] = action_dev_raw[..., 13].clamp(0.0, 1.0)
        else:
            action_dev_raw = external_action_dev_raw.to(device=image_t.device, dtype=torch.float32)

        if external_qpos_err_norm is None:
            qpos_err_norm = (action_dev_raw[:, -1, :] - qpos_mean.view(1, -1)) / qpos_std.view(1, -1)
        else:
            qpos_err_norm = external_qpos_err_norm.to(device=image_t.device, dtype=torch.float32)

        if external_z_wm_rollout is None:
            if raw_data is None or fk is None:
                raise ValueError("raw_data and fk are required when external_z_wm_rollout is not provided.")
            z_wm_rollout = wm_teacher.rollout_latent_from_actions(
                curr_image=image_t[0],
                curr_qpos_raw=qpos_raw[0],
                action_prefix_raw=action_dev_raw[0],
                raw_data=raw_data,
                fk=fk,
                ddim_steps=ddim_steps,
            )
            z_wm_rollout = z_wm_rollout.to(device=image_t.device, dtype=torch.float32)
            if z_wm_rollout.ndim == 3:
                z_wm_rollout = z_wm_rollout.unsqueeze(0)
        else:
            z_wm_rollout = external_z_wm_rollout.to(device=image_t.device, dtype=torch.float32)

        predictor_is_pad = external_action_is_pad_prefix
        if predictor_is_pad is None:
            predictor_is_pad = is_pad_prefix
        z_proj, z_hat_next = self.predict_future_latent(
            image_t=image_t,
            action_prefix=action_dev_norm,
            wm_teacher=wm_teacher,
            is_pad_prefix=predictor_is_pad,
        )
        return Stage2PreparedContext(
            z_proj=z_proj,
            pi0_action_chunk=pi0_action_chunk,
            action_dev_norm=action_dev_norm,
            action_dev_raw=action_dev_raw,
            qpos_err_norm=qpos_err_norm,
            z_wm_rollout=z_wm_rollout,
            z_hat_next=z_hat_next,
        )

    def compute_stage2_loss(
        self,
        *,
        stage2_ctx: Stage2PreparedContext,
        image_t: torch.Tensor,
        qpos_t_norm: torch.Tensor,
        pi0_action_mask: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        prompts: Sequence[str],
        global_step: float,
        retain_weight: float = 0.0,
        bridge_weight: float = 0.0,
        use_pi0_head_correction: bool = True,
        base_anchor_chunk: torch.Tensor | None = None,
        correction_target_chunk: torch.Tensor | None = None,
        correction_is_pad: torch.Tensor | None = None,
    ) -> Stage2LossOutput:
        beta_scale = self.warmup.weight(float(global_step))
        alpha_latent = beta_scale
        beta_dynamics = beta_scale * float(self.latent_loss_cfg.beta_dynamics_max)
        loss_dynamics = F.mse_loss(stage2_ctx.z_hat_next, stage2_ctx.z_wm_rollout.detach())

        (
            correction_target_chunk,
            correction_is_pad,
            correction_target_prefix,
            is_pad_prefix,
        ) = self._build_stage2_targets(
            pi0_action_chunk=stage2_ctx.pi0_action_chunk,
            correction_target_prefix=correction_target_prefix,
            is_pad_prefix=is_pad_prefix,
            correction_target_chunk=correction_target_chunk,
            correction_is_pad=correction_is_pad,
        )

        correction_mask = self._bool_pad_to_action_mask(correction_is_pad, correction_target_chunk.shape[-1])
        obs_error = self._make_observation(image_t, stage2_ctx.qpos_err_norm, prompts)
        obs_current = self._make_observation(image_t, qpos_t_norm, prompts)

        if use_pi0_head_correction:
            teacher_token = self._latent_to_condition_token(stage2_ctx.z_wm_rollout.detach(), scale=alpha_latent)
            loss_correct = self._compute_pi0_loss(
                obs_error,
                correction_target_chunk,
                correction_mask,
                latent_token=teacher_token,
            )
            loss_bridge = loss_correct.new_zeros(())
        else:
            bridge_token = self._latent_to_condition_token(stage2_ctx.z_hat_next, scale=alpha_latent)
            loss_correct = self._compute_pi0_loss(
                obs_error,
                correction_target_chunk,
                correction_mask,
                latent_token=bridge_token,
            )
            loss_bridge = loss_correct.new_zeros(())
            if bridge_weight > 0.0:
                teacher_token = self._latent_to_condition_token(stage2_ctx.z_wm_rollout.detach(), scale=alpha_latent)
                teacher_loss = self._compute_pi0_loss(
                    obs_error,
                    correction_target_chunk,
                    correction_mask,
                    latent_token=teacher_token,
                )
                loss_bridge = teacher_loss

        loss_retain = loss_correct.new_zeros(())
        if retain_weight > 0.0:
            anchor_chunk = base_anchor_chunk if base_anchor_chunk is not None else stage2_ctx.pi0_action_chunk.detach()
            loss_retain = self._compute_pi0_loss(obs_current, anchor_chunk.detach(), pi0_action_mask)

        loss = loss_correct + (beta_dynamics * loss_dynamics) + (retain_weight * loss_retain) + (bridge_weight * loss_bridge)
        return Stage2LossOutput(
            loss=loss,
            loss_correct=loss_correct,
            loss_dynamics=loss_dynamics,
            loss_retain=loss_retain,
            loss_bridge=loss_bridge,
            beta_dynamics=beta_dynamics,
            alpha_latent=alpha_latent,
        )

    def forward_stage2(
        self,
        *,
        image_t: torch.Tensor,
        qpos_t_norm: torch.Tensor,
        qpos_raw: torch.Tensor,
        pi0_action_mask: torch.Tensor,
        correction_target_prefix: torch.Tensor | None,
        is_pad_prefix: torch.Tensor | None,
        prompts: Sequence[str],
        wm_teacher,
        raw_data,
        norm_stats: dict[str, torch.Tensor | list | tuple],
        global_step: float,
        ddim_steps: int = 27,
        retain_weight: float = 0.0,
        bridge_weight: float = 0.0,
        use_pi0_head_correction: bool = True,
        base_anchor_chunk: torch.Tensor | None = None,
        correction_target_chunk: torch.Tensor | None = None,
        correction_is_pad: torch.Tensor | None = None,
        external_action_dev_norm: torch.Tensor | None = None,
        external_action_dev_raw: torch.Tensor | None = None,
        external_qpos_err_norm: torch.Tensor | None = None,
        external_action_is_pad_prefix: torch.Tensor | None = None,
        external_z_wm_rollout: torch.Tensor | None = None,
        num_steps: int = 10,
        fk=None,
    ) -> Stage2LossOutput:
        stage2_ctx = self.prepare_stage2_context(
            image_t=image_t,
            qpos_t_norm=qpos_t_norm,
            qpos_raw=qpos_raw,
            wm_teacher=wm_teacher,
            raw_data=raw_data,
            norm_stats=norm_stats,
            ddim_steps=ddim_steps,
            is_pad_prefix=is_pad_prefix,
            external_action_dev_norm=external_action_dev_norm,
            external_action_dev_raw=external_action_dev_raw,
            external_qpos_err_norm=external_qpos_err_norm,
            external_action_is_pad_prefix=external_action_is_pad_prefix,
            external_z_wm_rollout=external_z_wm_rollout,
            prompts=prompts,
            num_steps=num_steps,
            fk=fk,
        )
        return self.compute_stage2_loss(
            stage2_ctx=stage2_ctx,
            image_t=image_t,
            qpos_t_norm=qpos_t_norm,
            pi0_action_mask=pi0_action_mask,
            correction_target_prefix=correction_target_prefix,
            is_pad_prefix=is_pad_prefix,
            prompts=prompts,
            global_step=global_step,
            retain_weight=retain_weight,
            bridge_weight=bridge_weight,
            use_pi0_head_correction=use_pi0_head_correction,
            base_anchor_chunk=base_anchor_chunk,
            correction_target_chunk=correction_target_chunk,
            correction_is_pad=correction_is_pad,
        )
