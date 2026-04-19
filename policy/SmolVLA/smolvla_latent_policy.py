from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SMOLVLA_SRC_DIR = THIS_DIR / "src"
if not SMOLVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"SmolVLA src directory not found: {SMOLVLA_SRC_DIR}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from policy.SmolVLA.latent_config import DynamicsWarmupConfig
from policy.SmolVLA.latent_warmup import DynamicsWarmup


@dataclass(frozen=True)
class SmolVLALatentBridgeConfig:
    action_dim: int
    prefix_steps: int
    latent_dim: int
    adapter_hidden_dim: int
    predictor_hidden_dim: int


@dataclass
class SmolVLAStage1LossOutput:
    loss: torch.Tensor
    loss_action: torch.Tensor
    loss_action_conditioned: torch.Tensor
    loss_dynamics: torch.Tensor
    beta_dynamics: float


@dataclass
class SmolVLAStage2LossOutput:
    loss: torch.Tensor
    loss_correct: torch.Tensor
    loss_retain: torch.Tensor
    loss_dynamics: torch.Tensor
    beta_dynamics: float


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionConditionedLatentPredictor(nn.Module):
    def __init__(self, latent_dim: int, prefix_steps: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.prefix_steps = int(prefix_steps)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + prefix_steps * action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action_prefix: torch.Tensor) -> torch.Tensor:
        if action_prefix.ndim != 3:
            raise ValueError(f"action_prefix must be (B, K, A), got {tuple(action_prefix.shape)}")
        flat_prefix = action_prefix.reshape(action_prefix.shape[0], -1)
        return self.net(torch.cat([latent, flat_prefix], dim=1))


class SmolVLALatentPolicy(nn.Module):
    def __init__(self, base_policy: SmolVLAPolicy, bridge_cfg: SmolVLALatentBridgeConfig, warmup_cfg: DynamicsWarmupConfig):
        super().__init__()
        self.base_policy = base_policy
        self.bridge_cfg = bridge_cfg
        self.beta_scheduler = DynamicsWarmup(warmup_cfg)
        self.visual_adapter: nn.Linear | None = None
        self.wm_adapter: ResidualMLP | None = None
        self.predictor: ActionConditionedLatentPredictor | None = None
        self.latent_to_token: nn.Linear | None = None

    @property
    def device(self) -> torch.device:
        return next(self.base_policy.parameters()).device

    def _policy_inputs_from_batch(self, batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        images, img_masks = self.base_policy.prepare_images(batch)
        state = self.base_policy.prepare_state(batch)
        if OBS_LANGUAGE_TOKENS not in batch:
            raise KeyError(f"Missing required key: {OBS_LANGUAGE_TOKENS}")
        if OBS_LANGUAGE_ATTENTION_MASK not in batch:
            raise KeyError(f"Missing required key: {OBS_LANGUAGE_ATTENTION_MASK}")
        return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state

    def _ensure_bridge_modules(self, visual_dim: int, hidden_size: int) -> None:
        if self.visual_adapter is not None:
            return
        latent_dim = int(self.bridge_cfg.latent_dim)
        self.visual_adapter = nn.Linear(visual_dim, latent_dim).to(self.device)
        self.wm_adapter = ResidualMLP(latent_dim, int(self.bridge_cfg.adapter_hidden_dim)).to(self.device)
        self.predictor = ActionConditionedLatentPredictor(
            latent_dim=latent_dim,
            prefix_steps=int(self.bridge_cfg.prefix_steps),
            action_dim=int(self.bridge_cfg.action_dim),
            hidden_dim=int(self.bridge_cfg.predictor_hidden_dim),
        ).to(self.device)
        self.latent_to_token = nn.Linear(latent_dim, hidden_size).to(self.device)

    def extract_visual_latent(self, batch: dict) -> torch.Tensor:
        images, img_masks, _, _, _ = self._policy_inputs_from_batch(batch)
        image_embedding, image_mask = self.base_policy.model.extract_primary_visual_embedding(images, img_masks)
        if image_embedding.ndim != 3:
            raise ValueError(f"Expected primary visual embedding to be (B, N, H), got {tuple(image_embedding.shape)}")
        if image_mask.ndim != 2:
            raise ValueError(f"Expected primary visual mask to be (B, N), got {tuple(image_mask.shape)}")
        mask = image_mask.to(dtype=image_embedding.dtype).unsqueeze(-1)
        pooled = (image_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        hidden_size = int(image_embedding.shape[-1])
        self._ensure_bridge_modules(int(pooled.shape[-1]), hidden_size)
        assert self.visual_adapter is not None
        assert self.wm_adapter is not None
        pooled = pooled.to(dtype=self.visual_adapter.weight.dtype)
        return self.wm_adapter(self.visual_adapter(pooled))

    @torch.no_grad()
    def initialize_from_batch(self, batch: dict) -> None:
        _ = self.extract_visual_latent(batch)

    @staticmethod
    def teacher_latent_to_vector(teacher_latent: torch.Tensor) -> torch.Tensor:
        if teacher_latent.ndim != 4:
            raise ValueError(f"Teacher latent must be (B, C, H, W), got {tuple(teacher_latent.shape)}")
        return F.adaptive_avg_pool2d(teacher_latent, output_size=1).flatten(1)

    def build_condition_token(self, latent: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 2:
            raise ValueError(f"latent must be (B, D), got {tuple(latent.shape)}")
        assert self.latent_to_token is not None
        latent = latent.to(dtype=self.latent_to_token.weight.dtype)
        token = self.latent_to_token(latent) * float(scale)
        token = token[:, None, :]
        mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        att_mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        return token, mask, att_mask

    def _action_loss(
        self,
        batch: dict,
        actions: torch.Tensor,
        external_prefix_tokens: torch.Tensor | None = None,
        external_prefix_mask: torch.Tensor | None = None,
        external_prefix_att_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._policy_inputs_from_batch(batch)
        action_batch = {ACTION: actions}
        actions_padded = self.base_policy.prepare_action(action_batch)
        losses = self.base_policy.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions_padded,
            external_prefix_tokens=external_prefix_tokens,
            external_prefix_mask=external_prefix_mask,
            external_prefix_att_mask=external_prefix_att_mask,
        )
        return losses[:, :, : self.base_policy.config.max_action_dim].mean()

    def _require_action_prefix(self, action_prefix: torch.Tensor) -> None:
        if action_prefix.ndim != 3:
            raise ValueError(f"action_prefix must be (B, K, A), got {tuple(action_prefix.shape)}")
        expected_shape = (int(self.bridge_cfg.prefix_steps), int(self.bridge_cfg.action_dim))
        if tuple(action_prefix.shape[1:]) != expected_shape:
            raise ValueError(
                f"action_prefix shape mismatch: expected (*, {expected_shape[0]}, {expected_shape[1]}), got {tuple(action_prefix.shape)}"
            )

    def predict_next_latent(self, batch: dict, action_prefix: torch.Tensor) -> torch.Tensor:
        self._require_action_prefix(action_prefix)
        assert self.predictor is not None
        predictor_dtype = self.predictor.net[0].weight.dtype
        return self.predictor(
            self.extract_visual_latent(batch).to(dtype=predictor_dtype),
            action_prefix.to(dtype=predictor_dtype),
        )

    def compute_stage1_loss(
        self,
        batch: dict,
        future_teacher_latent: torch.Tensor,
        action_prefix: torch.Tensor,
        global_step: int,
    ) -> SmolVLAStage1LossOutput:
        if ACTION not in batch:
            raise KeyError(f"Missing required key: {ACTION}")
        self._require_action_prefix(action_prefix)
        teacher_vector = self.teacher_latent_to_vector(future_teacher_latent)
        visual_latent = self.extract_visual_latent(batch)
        if teacher_vector.shape[1] != visual_latent.shape[1]:
            raise ValueError(f"Teacher latent vector and visual bridge latent dimension mismatch: {teacher_vector.shape[1]} vs {visual_latent.shape[1]}")
        teacher_vector = teacher_vector.to(dtype=visual_latent.dtype)

        beta_dynamics = self.beta_scheduler.weight(global_step)
        predicted_latent = self.predict_next_latent(batch, action_prefix)
        loss_action = self._action_loss(batch=batch, actions=batch[ACTION])
        cond_token, cond_mask, cond_att_mask = self.build_condition_token(teacher_vector, scale=1.0)
        loss_action_conditioned = self._action_loss(
            batch=batch,
            actions=batch[ACTION],
            external_prefix_tokens=cond_token,
            external_prefix_mask=cond_mask,
            external_prefix_att_mask=cond_att_mask,
        )
        loss_dynamics = F.mse_loss(predicted_latent, teacher_vector)
        loss = loss_action + loss_action_conditioned + (beta_dynamics * loss_dynamics)
        return SmolVLAStage1LossOutput(
            loss=loss,
            loss_action=loss_action,
            loss_action_conditioned=loss_action_conditioned,
            loss_dynamics=loss_dynamics,
            beta_dynamics=beta_dynamics,
        )

    def compute_stage2_loss(
        self,
        normal_batch: dict,
        correction_batch: dict,
        correction_actions: torch.Tensor,
        correction_action_prefix: torch.Tensor,
        correction_teacher_latent: torch.Tensor,
        rollout_teacher_latent: torch.Tensor,
        global_step: int,
        retain_weight: float,
    ) -> SmolVLAStage2LossOutput:
        if ACTION not in normal_batch:
            raise KeyError(f"Missing required key in normal_batch: {ACTION}")
        self._require_action_prefix(correction_action_prefix)

        correction_teacher_vector = self.teacher_latent_to_vector(correction_teacher_latent)
        rollout_teacher_vector = self.teacher_latent_to_vector(rollout_teacher_latent)
        if correction_teacher_vector.shape[1] != rollout_teacher_vector.shape[1]:
            raise ValueError(
                "Correction teacher latent and rollout teacher latent dimension mismatch: "
                f"{correction_teacher_vector.shape[1]} vs {rollout_teacher_vector.shape[1]}"
            )

        predicted_latent = self.predict_next_latent(correction_batch, correction_action_prefix)
        correction_teacher_vector = correction_teacher_vector.to(dtype=predicted_latent.dtype)
        rollout_teacher_vector = rollout_teacher_vector.to(dtype=predicted_latent.dtype)
        beta_dynamics = self.beta_scheduler.weight(global_step)
        corr_token, corr_mask, corr_att_mask = self.build_condition_token(correction_teacher_vector, scale=1.0)
        loss_correct = self._action_loss(
            batch=correction_batch,
            actions=correction_actions,
            external_prefix_tokens=corr_token,
            external_prefix_mask=corr_mask,
            external_prefix_att_mask=corr_att_mask,
        )
        loss_retain = self._action_loss(batch=normal_batch, actions=normal_batch[ACTION])
        loss_dynamics = F.mse_loss(predicted_latent, rollout_teacher_vector)
        loss = loss_correct + (float(retain_weight) * loss_retain) + (beta_dynamics * loss_dynamics)
        return SmolVLAStage2LossOutput(
            loss=loss,
            loss_correct=loss_correct,
            loss_retain=loss_retain,
            loss_dynamics=loss_dynamics,
            beta_dynamics=beta_dynamics,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return self.base_policy.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk_conditioned(self, batch: dict, latent: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        token, mask, att_mask = self.build_condition_token(latent, scale=scale)
        images, img_masks, lang_tokens, lang_masks, state = self._policy_inputs_from_batch(batch)
        return self.base_policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            external_prefix_tokens=token,
            external_prefix_mask=mask,
            external_prefix_att_mask=att_mask,
        )[:, :, : self.base_policy.config.action_feature.shape[0]]
