import sys
from pathlib import Path

import cv2
import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
SRC_DIR = FILE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION


CAMERA_OBS_KEYS = {
    "head_camera": ("observation", "head_camera", "rgb"),
}


def _nested_get(data, path):
    value = data
    for key in path:
        value = value[key]
    return value


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _extract_qpos(observation):
    return np.array(
        observation["joint_action"]["left_arm"]
        + [observation["joint_action"]["left_gripper"]]
        + observation["joint_action"]["right_arm"]
        + [observation["joint_action"]["right_gripper"]],
        dtype=np.float32,
    )


class SmolVLAEvalWrapper:
    def __init__(self, policy_config):
        self.model_path = policy_config["model_path"]
        self.device = _resolve_device(policy_config.get("device", "cuda"))
        self.camera_sources = policy_config.get(
            "camera_sources",
            ["head_camera"],
        )

        self.policy = SmolVLAPolicy.from_pretrained(self.model_path)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self.visual_feature_keys = [
            key
            for key, feature in self.policy.config.input_features.items()
            if feature.type == FeatureType.VISUAL
        ]
        self.state_feature_keys = [
            key
            for key, feature in self.policy.config.input_features.items()
            if feature.type == FeatureType.STATE
        ]

        if len(self.visual_feature_keys) == 0:
            raise ValueError("SmolVLA policy has no visual input features.")
        if len(self.state_feature_keys) == 0:
            raise ValueError("SmolVLA policy has no state input features.")

    def _build_model_input(self, observation, instruction):
        qpos = _extract_qpos(observation)

        model_obs = {}
        feature_key = self.visual_feature_keys[0]
        source_name = self.camera_sources[0]
        if source_name not in CAMERA_OBS_KEYS:
            raise KeyError(f"Unsupported camera source: {source_name}")
        image = _nested_get(observation, CAMERA_OBS_KEYS[source_name])
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        model_obs[feature_key] = image

        model_obs[self.state_feature_keys[0]] = qpos
        prepared_obs = prepare_observation_for_inference(
            model_obs,
            self.device,
            task=str(instruction),
            robot_type="aloha",
        )
        return self.preprocess(prepared_obs)

    def get_action(self, observation, instruction):
        model_input = self._build_model_input(observation, instruction)
        with torch.inference_mode():
            action = self.policy.select_action(model_input)
        action = self.postprocess(action)

        if isinstance(action, dict) and ACTION in action:
            action = action[ACTION]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().float().numpy()

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            return action[None, :]
        if action.ndim == 2:
            return action
        raise ValueError(f"Unexpected action shape from SmolVLA: {action.shape}")


def get_model(usr_args):
    return SmolVLAEvalWrapper(usr_args)


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    actions = model.get_action(observation, instruction)

    for action in actions:
        TASK_ENV.take_action(action.tolist())
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    model.policy.reset()
