from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if not SMOLVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"SmolVLA src directory not found: {SMOLVLA_SRC_DIR}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from policy.SmolVLA.latentcorr.latent_config import DynamicsWarmupConfig, Stage1WarmupConfig
from policy.SmolVLA.latentcorr.latent_warmup import DynamicsWarmup


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
    loss_condition_token: torch.Tensor
    beta_condition: float
    beta_dynamics: float
    beta_token: float


@dataclass(frozen=True)
class SmolVLAStage1LossWeights:
    token_init: float = 0.1
    token_late: float = 0.02
    token_decay_start_ratio: float = 0.0
    token_decay_end_ratio: float = 1.0
    total_steps: int = 1


@dataclass
class SmolVLAStage2LossOutput:
    loss: torch.Tensor
    loss_correct: torch.Tensor
    loss_retain: torch.Tensor
    loss_dynamics: torch.Tensor
    beta_dynamics: float


def _pick_num_groups(num_channels: int, max_groups: int = 32) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups != 0):
        groups -= 1
    return max(1, groups)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class LatentProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(out_channels), out_channels),
        )

    def forward(self, z_visual: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        z = F.interpolate(z_visual, size=target_hw, mode="bilinear", align_corners=False)
        return self.proj(z)


class ResidualLatentAdapter(nn.Module):
    def __init__(self, channels: int, mid_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
            nn.GroupNorm(_pick_num_groups(channels), channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


class ActionPrefixEncoder(nn.Module):
    def __init__(self, prefix_steps: int, action_dim: int, out_channels: int, hidden_dim: int):
        super().__init__()
        self.prefix_steps = int(prefix_steps)
        self.action_dim = int(action_dim)
        in_dim = self.prefix_steps * self.action_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * 2),
        )

    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 3:
            raise ValueError(f"actions must be (B, K, A), got {tuple(actions.shape)}")
        x = actions.reshape(actions.shape[0], -1)
        gamma_beta = self.mlp(x)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma[:, :, None, None], beta[:, :, None, None]


class AdaLNResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.norm(z)
        h = gamma * h + beta
        h = self.conv1(h)
        h = self.act(h)
        h = self.conv2(h)
        return z + h


class ActionConditionedPredictor(nn.Module):
    def __init__(self, channels: int, prefix_steps: int, action_dim: int, hidden_dim: int, num_blocks: int = 3):
        super().__init__()
        self.encoder = ActionPrefixEncoder(prefix_steps, action_dim, channels, hidden_dim)
        self.blocks = nn.ModuleList([AdaLNResBlock(channels) for _ in range(num_blocks)])

    def forward(self, z_t: torch.Tensor, action_prefix: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.encoder(action_prefix)
        z = z_t
        for block in self.blocks:
            z = block(z, gamma, beta)
        return z


class LatentToTokenAdapter(nn.Module):
    def __init__(self, latent_channels: int, token_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(latent_channels)
        depth = max(1, int(num_layers))
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(token_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, latent_map: torch.Tensor) -> torch.Tensor:
        if latent_map.ndim != 4:
            raise ValueError(f"latent_map must be (B, C, H, W), got {tuple(latent_map.shape)}")
        pooled = F.adaptive_avg_pool2d(latent_map, output_size=1).flatten(1)
        return self.net(pooled)


class SmolVLALatentPolicy(nn.Module):
    def __init__(
        self,
        base_policy: SmolVLAPolicy,
        bridge_cfg: SmolVLALatentBridgeConfig,
        warmup_cfg: DynamicsWarmupConfig | Stage1WarmupConfig,
        loss_weights: SmolVLAStage1LossWeights | None = None,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.bridge_cfg = bridge_cfg
        self.loss_weights = loss_weights or SmolVLAStage1LossWeights()
        if isinstance(warmup_cfg, Stage1WarmupConfig):
            self.beta_dynamics_scheduler = DynamicsWarmup(warmup_cfg.dynamics)
            self.beta_condition_scheduler = DynamicsWarmup(warmup_cfg.condition)
        else:
            self.beta_dynamics_scheduler = DynamicsWarmup(warmup_cfg)
            self.beta_condition_scheduler = DynamicsWarmup(warmup_cfg)
        self.projector: LatentProjector | None = None
        self.wm_adapter: ResidualLatentAdapter | None = None
        self.predictor: ActionConditionedPredictor | None = None
        self.token_adapter: LatentToTokenAdapter | None = None
        self.condition_proj: nn.Linear | None = None
        self._target_hw: tuple[int, int] | None = None

    @staticmethod
    def _linear_schedule(progress: float, start_ratio: float, end_ratio: float, start_value: float, end_value: float) -> float:
        p = float(max(0.0, min(1.0, progress)))
        if p <= float(start_ratio):
            return float(start_value)
        if p >= float(end_ratio) or float(end_ratio) <= float(start_ratio):
            return float(end_value)
        r = (p - float(start_ratio)) / (float(end_ratio) - float(start_ratio))
        return float(start_value) * (1.0 - r) + float(end_value) * r

    def _token_weight(self, global_step: int) -> float:
        cfg = self.loss_weights
        total_steps = max(1.0, float(cfg.total_steps))
        progress = float(global_step) / total_steps
        return self._linear_schedule(
            progress,
            float(cfg.token_decay_start_ratio),
            float(cfg.token_decay_end_ratio),
            float(cfg.token_init),
            float(cfg.token_late),
        )

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

    def _extract_preconnector_visual_map(self, batch: dict) -> tuple[torch.Tensor, int]:
        images, img_masks, _, _, _ = self._policy_inputs_from_batch(batch)
        if len(images) != len(img_masks):
            raise ValueError(f"images/img_masks length mismatch: {len(images)} vs {len(img_masks)}")
        if len(images) == 0:
            raise ValueError("At least one image tensor is required.")

        img = images[0]
        img_mask = img_masks[0]
        if img.ndim != 4:
            raise ValueError(f"Primary image tensor must be (B, C, H, W), got {tuple(img.shape)}")
        if img_mask.ndim != 1:
            raise ValueError(f"Primary image mask must be (B,), got {tuple(img_mask.shape)}")

        vision_model = self.base_policy.model.vlm_with_expert.get_vlm_model().vision_model
        image_hidden_states = vision_model(
            pixel_values=img.to(dtype=vision_model.dtype),
            patch_attention_mask=None,
        ).last_hidden_state
        if image_hidden_states.ndim != 3:
            raise ValueError(
                f"Expected pre-connector visual states to be (B, N, C), got {tuple(image_hidden_states.shape)}"
            )

        batch_size, seq_len, channels = image_hidden_states.shape
        grid_size = int(seq_len**0.5)
        if grid_size * grid_size != seq_len:
            raise ValueError(f"Visual token sequence length is not a square grid: {seq_len}")
        visual_map = image_hidden_states.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        visual_map = visual_map * img_mask[:, None, None, None].to(dtype=visual_map.dtype)
        return visual_map, int(self.base_policy.model.vlm_with_expert.config.text_config.hidden_size)

    def _ensure_bridge_modules(self, visual_channels: int, hidden_size: int) -> None:
        if self.projector is not None:
            return
        latent_dim = int(self.bridge_cfg.latent_dim)
        self.projector = LatentProjector(
            in_channels=visual_channels,
            out_channels=latent_dim,
            mid_channels=int(self.bridge_cfg.adapter_hidden_dim),
        ).to(self.device)
        self.wm_adapter = ResidualLatentAdapter(
            channels=latent_dim,
            mid_channels=int(self.bridge_cfg.adapter_hidden_dim),
        ).to(self.device)
        self.predictor = ActionConditionedPredictor(
            channels=latent_dim,
            prefix_steps=int(self.bridge_cfg.prefix_steps),
            action_dim=int(self.bridge_cfg.action_dim),
            hidden_dim=int(self.bridge_cfg.predictor_hidden_dim),
            num_blocks=3,
        ).to(self.device)
        self.token_adapter = LatentToTokenAdapter(
            latent_channels=latent_dim,
            token_dim=hidden_size,
            hidden_dim=int(self.bridge_cfg.adapter_hidden_dim),
            num_layers=2,
            dropout=0.1,
        ).to(self.device)
        self.condition_proj = nn.Linear(latent_dim, hidden_size).to(self.device)
        nn.init.zeros_(self.condition_proj.bias)

    def extract_visual_latent_map(self, batch: dict, target_hw: tuple[int, int]) -> torch.Tensor:
        visual_map, hidden_size = self._extract_preconnector_visual_map(batch)
        self._ensure_bridge_modules(int(visual_map.shape[1]), hidden_size)
        assert self.projector is not None
        self._target_hw = (int(target_hw[0]), int(target_hw[1]))
        visual_map = visual_map.to(dtype=self.projector.proj[0].weight.dtype)
        return self.projector(visual_map, target_hw=target_hw)

    @staticmethod
    def teacher_latent_to_map(teacher_latent: torch.Tensor) -> torch.Tensor:
        if teacher_latent.ndim != 4:
            raise ValueError(f"Teacher latent must be (B, C, H, W), got {tuple(teacher_latent.shape)}")
        return teacher_latent

    def shared_teacher_latent(self, teacher_latent: torch.Tensor) -> torch.Tensor:
        teacher_map = self.teacher_latent_to_map(teacher_latent)
        if self.projector is None:
            raise RuntimeError("Bridge modules must be initialized before calling shared_teacher_latent.")
        assert self.wm_adapter is not None
        teacher_map = teacher_map.to(dtype=self.wm_adapter.net[0].weight.dtype)
        return self.wm_adapter(teacher_map)

    @torch.no_grad()
    def initialize_from_batch(self, batch: dict, teacher_latent: torch.Tensor | None = None) -> None:
        if teacher_latent is None:
            visual_map, hidden_size = self._extract_preconnector_visual_map(batch)
            self._ensure_bridge_modules(int(visual_map.shape[1]), hidden_size)
            return
        teacher_map = self.teacher_latent_to_map(teacher_latent)
        _ = self.extract_visual_latent_map(batch, target_hw=(teacher_map.shape[-2], teacher_map.shape[-1]))

    @property
    def target_hw(self) -> tuple[int, int] | None:
        return self._target_hw

    def build_condition_token(self, latent_map: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent_map.ndim != 4:
            raise ValueError(f"latent_map must be (B, C, H, W), got {tuple(latent_map.shape)}")
        assert self.condition_proj is not None
        pooled = F.adaptive_avg_pool2d(latent_map, output_size=1).flatten(1)
        pooled = pooled.to(dtype=self.condition_proj.weight.dtype)
        token = self.condition_proj(pooled) * float(scale)
        token = token[:, None, :]
        mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        att_mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        return token, mask, att_mask

    def build_condition_token_raw(self, latent_map: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        token, _mask, _att_mask = self.build_condition_token(latent_map, scale=scale)
        return token

    def build_predicted_condition_token(self, latent_map: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.token_adapter is not None
        latent_map = latent_map.to(dtype=next(self.token_adapter.parameters()).dtype)
        token = self.token_adapter(latent_map) * float(scale)
        token = token[:, None, :]
        mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        att_mask = torch.ones(token.shape[:2], dtype=torch.bool, device=token.device)
        return token, mask, att_mask

    def build_predicted_condition_token_raw(self, latent_map: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        token, _mask, _att_mask = self.build_predicted_condition_token(latent_map, scale=scale)
        return token

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

    def _action_prefix_loss(
        self,
        batch: dict,
        actions: torch.Tensor,
        prefix_steps: int,
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
        )
        steps = min(max(1, int(prefix_steps)), int(losses.shape[1]))
        return losses[:, :steps, : self.base_policy.config.max_action_dim].mean()

    def _require_action_prefix(self, action_prefix: torch.Tensor) -> None:
        if action_prefix.ndim != 3:
            raise ValueError(f"action_prefix must be (B, K, A), got {tuple(action_prefix.shape)}")
        expected_shape = (int(self.bridge_cfg.prefix_steps), int(self.bridge_cfg.action_dim))
        if tuple(action_prefix.shape[1:]) != expected_shape:
            raise ValueError(
                f"action_prefix shape mismatch: expected (*, {expected_shape[0]}, {expected_shape[1]}), got {tuple(action_prefix.shape)}"
            )

    def predict_next_latent(self, batch: dict, action_prefix: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        self._require_action_prefix(action_prefix)
        assert self.predictor is not None
        z_proj = self.extract_visual_latent_map(batch, target_hw=target_hw)
        predictor_dtype = next(self.predictor.parameters()).dtype
        return self.predictor(
            z_proj.to(dtype=predictor_dtype),
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

        teacher_shared = self.shared_teacher_latent(future_teacher_latent)
        target_hw = (teacher_shared.shape[-2], teacher_shared.shape[-1])
        predicted_latent = self.predict_next_latent(batch, action_prefix, target_hw=target_hw)

        beta_dynamics = self.beta_dynamics_scheduler.weight(global_step)
        beta_condition = self.beta_condition_scheduler.weight(global_step)
        beta_token = self._token_weight(global_step)
        loss_action = self._action_loss(batch=batch, actions=batch[ACTION])
        cond_token, cond_mask, cond_att_mask = self.build_predicted_condition_token(
            predicted_latent,
            scale=1.0,
        )
        pred_token_raw = cond_token
        teacher_token_raw = self.build_condition_token_raw(teacher_shared.detach(), scale=1.0)
        loss_condition_token = F.mse_loss(pred_token_raw, teacher_token_raw.detach())
        loss_action_conditioned = self._action_loss(
            batch=batch,
            actions=batch[ACTION],
            external_prefix_tokens=cond_token,
            external_prefix_mask=cond_mask,
            external_prefix_att_mask=cond_att_mask,
        )
        loss_dynamics = F.mse_loss(predicted_latent, teacher_shared)
        loss = (
            loss_action
            + (beta_condition * loss_action_conditioned)
            + (beta_dynamics * loss_dynamics)
            + (beta_token * loss_condition_token)
        )
        return SmolVLAStage1LossOutput(
            loss=loss,
            loss_action=loss_action,
            loss_action_conditioned=loss_action_conditioned,
            loss_dynamics=loss_dynamics,
            loss_condition_token=loss_condition_token,
            beta_condition=beta_condition,
            beta_dynamics=beta_dynamics,
            beta_token=beta_token,
        )

    def forward(self, train_stage: str, **kwargs):
        if train_stage == "stage1":
            return self.compute_stage1_loss(**kwargs)
        if train_stage == "action_prefix":
            return self._action_prefix_loss(**kwargs)
        if train_stage == "stage2":
            return self.compute_stage2_loss(**kwargs)
        raise ValueError(f"Unsupported train_stage: {train_stage}")

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

        correction_shared = self.shared_teacher_latent(correction_teacher_latent)
        rollout_shared = self.shared_teacher_latent(rollout_teacher_latent)
        if correction_shared.shape[1:] != rollout_shared.shape[1:]:
            raise ValueError(
                "Correction teacher latent and rollout teacher latent shape mismatch: "
                f"{tuple(correction_shared.shape[1:])} vs {tuple(rollout_shared.shape[1:])}"
            )

        target_hw = (rollout_shared.shape[-2], rollout_shared.shape[-1])
        predicted_latent = self.predict_next_latent(correction_batch, correction_action_prefix, target_hw=target_hw)
        beta_dynamics = self.beta_dynamics_scheduler.weight(global_step)
        corr_token, corr_mask, corr_att_mask = self.build_predicted_condition_token(
            predicted_latent,
            scale=1.0,
        )
        loss_correct = self._action_loss(
            batch=correction_batch,
            actions=correction_actions,
            external_prefix_tokens=corr_token,
            external_prefix_mask=corr_mask,
            external_prefix_att_mask=corr_att_mask,
        )
        loss_retain = self._action_loss(batch=normal_batch, actions=normal_batch[ACTION])
        loss_dynamics = F.mse_loss(predicted_latent, rollout_shared)
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
    def predict_action_chunk_conditioned(self, batch: dict, latent_map: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        token, mask, att_mask = self.build_condition_token(latent_map, scale=scale)
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
