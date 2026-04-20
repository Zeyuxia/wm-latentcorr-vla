import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
SRC_DIR = FILE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.configs.types import FeatureType
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION
from policy.SmolVLA.deploy_policy import CAMERA_OBS_KEYS, _extract_qpos, _nested_get, _resolve_device
from policy.SmolVLA.latentcorr.smolvla_data_utils import make_smolvla_processors
from policy.SmolVLA.latentcorr.smolvla_latent_policy import SmolVLALatentPolicy
from policy.SmolVLA.latentcorr.train_smolvla import (
    build_base_policy,
    build_bridge_config,
    build_warmup_config,
)


def _load_train_args(bridge_ckpt: str) -> Namespace:
    checkpoint = torch.load(bridge_ckpt, map_location="cpu")
    if "args" not in checkpoint:
        raise KeyError(f"Missing 'args' in bridge checkpoint: {bridge_ckpt}")
    return Namespace(**checkpoint["args"])


class SmolVLABridgeEvalWrapper:
    def __init__(self, policy_config):
        self.model_path = str(policy_config["model_path"])
        self.bridge_ckpt = str(policy_config["bridge_ckpt"])
        self.device = _resolve_device(policy_config.get("device", "cuda"))
        self.camera_sources = policy_config.get("camera_sources", ["head_camera"])
        self.condition_scale = float(policy_config.get("condition_scale", 1.0))
        self.instruction_type = str(policy_config.get("instruction_type", "seen"))
        self.task_name = str(policy_config["task_name"])
        self.task_config = str(policy_config["task_config"])

        self.bridge_args = _load_train_args(self.bridge_ckpt)
        self.bridge_args.smolvla_pretrained_path = self.model_path
        self.bridge_args.device = str(self.device)

        base_policy, _ = build_base_policy(self.bridge_args)
        self.policy = SmolVLALatentPolicy(
            base_policy=base_policy,
            bridge_cfg=build_bridge_config(self.bridge_args),
            warmup_cfg=build_warmup_config(self.bridge_args),
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocess, self.postprocess = make_smolvla_processors(self.policy.base_policy, self.model_path)
        bridge_payload = torch.load(self.bridge_ckpt, map_location="cpu")
        self.bridge_state_dict = bridge_payload["model"]
        self.bridge_loaded = False
        target_hw = bridge_payload.get("target_hw")
        self.target_hw: tuple[int, int] | None = tuple(target_hw) if target_hw is not None else None
        self.visual_feature_keys = [
            key
            for key, feature in self.policy.base_policy.config.input_features.items()
            if feature.type == FeatureType.VISUAL
        ]
        self.state_feature_keys = [
            key
            for key, feature in self.policy.base_policy.config.input_features.items()
            if feature.type == FeatureType.STATE
        ]
        if len(self.visual_feature_keys) != 1:
            raise ValueError(f"Expected one visual feature key, got {self.visual_feature_keys}")
        if len(self.state_feature_keys) != 1:
            raise ValueError(f"Expected one state feature key, got {self.state_feature_keys}")

    def _build_model_input(self, observation, instruction):
        qpos = _extract_qpos(observation)

        if len(self.camera_sources) != 1:
            raise ValueError(f"Expected one camera source, got {self.camera_sources}")
        source_name = self.camera_sources[0]
        if source_name not in CAMERA_OBS_KEYS:
            raise KeyError(f"Unsupported camera source: {source_name}")
        image = _nested_get(observation, CAMERA_OBS_KEYS[source_name])
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        model_obs = {
            self.visual_feature_keys[0]: image,
            self.state_feature_keys[0]: qpos.astype(np.float32),
        }
        prepared_obs = prepare_observation_for_inference(
            model_obs,
            self.device,
            task=str(instruction),
            robot_type="aloha",
        )
        return self.preprocess(prepared_obs)

    def _ensure_bridge_loaded(self, batch):
        if self.bridge_loaded:
            return
        self.policy.initialize_from_batch(batch)
        self.policy.load_state_dict(self.bridge_state_dict, strict=True)
        if self.target_hw is None:
            raise KeyError(f"Missing target_hw in bridge checkpoint: {self.bridge_ckpt}")
        self.policy.eval()
        self.bridge_loaded = True

    def get_action(self, observation, instruction):
        model_input = self._build_model_input(observation, instruction)
        self._ensure_bridge_loaded(model_input)

        with torch.inference_mode():
            raw_action_chunk = self.policy.predict_action_chunk(model_input)
            action_prefix = raw_action_chunk[:, : int(self.bridge_args.prefix_steps), :]
            if self.target_hw is None:
                raise RuntimeError("Bridge target_hw not initialized.")
            predicted_latent = self.policy.predict_next_latent(model_input, action_prefix, target_hw=self.target_hw)
            conditioned_chunk = self.policy.predict_action_chunk_conditioned(
                model_input,
                predicted_latent,
                scale=self.condition_scale,
            )

        action = self.postprocess(conditioned_chunk)
        if isinstance(action, dict) and ACTION in action:
            action = action[ACTION]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().float().numpy()

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 2:
            return action
        if action.ndim == 3 and action.shape[0] == 1:
            return action[0]
        raise ValueError(f"Unexpected bridge action shape: {action.shape}")


def get_model(usr_args):
    return SmolVLABridgeEvalWrapper(usr_args)


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    actions = model.get_action(observation, instruction)
    for action in actions:
        TASK_ENV.take_action(action.tolist())
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    model.policy.base_policy.reset()
