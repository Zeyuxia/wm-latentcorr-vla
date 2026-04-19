from __future__ import annotations

import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SMOLVLA_SRC_DIR = THIS_DIR / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SMOLVLA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SMOLVLA_SRC_DIR))

from policy.SmolVLA.smolvla_data_utils import build_smolvla_batch


class SampleBoundSmolVLAAdapter:
    def __init__(
        self,
        latent_policy,
        preprocess,
        task_name: str,
        task_config: str,
        episode_id: int,
        instruction_type: str,
    ):
        self.latent_policy = latent_policy
        self.preprocess = preprocess
        self.task_name = task_name
        self.task_config = task_config
        self.episode_id = int(episode_id)
        self.instruction_type = instruction_type

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
