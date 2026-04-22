from __future__ import annotations

import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
SMOLVLA_ROOT = THIS_DIR.parent
SMOLVLA_SRC_DIR = SMOLVLA_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from lerobot.configs.types import FeatureType
from lerobot.policies.utils import prepare_observation_for_inference
from policy.SmolVLA.latentcorr.smolvla_data_utils import build_smolvla_batch


class SampleBoundSmolVLAAdapter:
    def __init__(
        self,
        latent_policy,
        preprocess,
        postprocess,
        task_name: str,
        task_config: str,
        episode_id: int,
        instruction_type: str,
    ):
        self.latent_policy = latent_policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.task_name = task_name
        self.task_config = task_config
        self.episode_id = int(episode_id)
        self.instruction_type = instruction_type
        self.visual_feature_keys = [
            key
            for key, feature in self.latent_policy.base_policy.config.input_features.items()
            if feature.type == FeatureType.VISUAL
        ]
        self.state_feature_keys = [
            key
            for key, feature in self.latent_policy.base_policy.config.input_features.items()
            if feature.type == FeatureType.STATE
        ]
        if len(self.visual_feature_keys) != 1:
            raise ValueError(f"Expected one visual feature key, got {self.visual_feature_keys}")
        if len(self.state_feature_keys) != 1:
            raise ValueError(f"Expected one state feature key, got {self.state_feature_keys}")

    @property
    def training(self) -> bool:
        return bool(self.latent_policy.training)

    def eval(self):
        self.latent_policy.eval()
        return self

    def train(self, mode: bool = True):
        self.latent_policy.train(mode)
        return self

    def build_batch(
        self,
        image_t: torch.Tensor,
        qpos_raw: torch.Tensor,
        action_chunk_raw: torch.Tensor,
    ) -> dict:
        return build_smolvla_batch(
            policy=self.latent_policy.base_policy,
            preprocess=self.preprocess,
            image_t=image_t,
            qpos_raw=qpos_raw,
            action_chunk_raw=action_chunk_raw,
            task_name=self.task_name,
            task_config=self.task_config,
            episode_id=self.episode_id,
            instruction_type=self.instruction_type,
        )

    def postprocess_action_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        payload = self.postprocess(action_chunk)
        if isinstance(payload, dict):
            from lerobot.utils.constants import ACTION

            if ACTION not in payload:
                raise KeyError(f"SmolVLA postprocess output is missing {ACTION}")
            payload = payload[ACTION]
        if not isinstance(payload, torch.Tensor):
            raise TypeError(f"SmolVLA postprocess must return a Tensor or ACTION dict, got {type(payload)!r}")
        return payload

    def build_eval_batch(
        self,
        image_t: torch.Tensor,
        qpos_raw: torch.Tensor,
        instruction: str | None = None,
    ) -> dict:
        if image_t.ndim != 4:
            raise ValueError(f"image_t must be (num_cam, C, H, W), got {tuple(image_t.shape)}")
        if int(image_t.shape[0]) != 1:
            raise ValueError(f"Exactly one camera is supported, got num_cam={int(image_t.shape[0])}")
        if qpos_raw.ndim != 1:
            raise ValueError(f"qpos_raw must be (D,), got {tuple(qpos_raw.shape)}")

        if instruction is None:
            from policy.SmolVLA.latentcorr.smolvla_data_utils import load_episode_instruction

            instruction = load_episode_instruction(
                task_name=self.task_name,
                task_config=self.task_config,
                episode_id=self.episode_id,
                instruction_type=self.instruction_type,
            )

        obs = {
            self.visual_feature_keys[0]: (
                image_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0
            ).clip(0, 255).astype("uint8"),
            self.state_feature_keys[0]: qpos_raw.detach().cpu().numpy().astype("float32"),
        }
        prepared = prepare_observation_for_inference(
            obs,
            device=next(self.latent_policy.base_policy.parameters()).device,
            task=str(instruction),
            robot_type="aloha",
        )
        return self.preprocess(prepared)

    def predict_act_chunk(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if qpos.ndim != 2 or qpos.shape[0] != 1:
            raise ValueError(f"qpos must be shaped (1, D), got {tuple(qpos.shape)}")
        if image.ndim == 5:
            if image.shape[0] != 1:
                raise ValueError(
                    f"batched image must be shaped (1, num_cam, C, H, W), got {tuple(image.shape)}"
                )
            image = image[0]
        if image.ndim != 4:
            raise ValueError(
                f"image must be shaped (num_cam, C, H, W) or (1, num_cam, C, H, W), got {tuple(image.shape)}"
            )

        action_dim = int(self.latent_policy.bridge_cfg.action_dim)
        chunk_size = int(self.latent_policy.base_policy.config.n_action_steps)
        dummy_action_chunk = torch.zeros(
            (chunk_size, action_dim),
            dtype=torch.float32,
            device=qpos.device,
        )
        batch = self.build_batch(
            image_t=image,
            qpos_raw=qpos[0],
            action_chunk_raw=dummy_action_chunk,
        )
        with torch.no_grad():
            return self.latent_policy.predict_action_chunk(batch)

    def __call__(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.predict_act_chunk(qpos, image)
