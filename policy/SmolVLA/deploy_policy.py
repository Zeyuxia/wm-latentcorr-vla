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
        self.model_path = str(policy_config["model_path"])
        self.device = _resolve_device(policy_config.get("device", "cuda"))
        self.camera_sources = policy_config.get(
            "camera_sources",
            ["head_camera"],
        )

        self.policy, processor_path = self._load_policy()
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            processor_path,
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

    def _load_policy(self):
        model_path = Path(self.model_path)
        if model_path.is_dir():
            return SmolVLAPolicy.from_pretrained(str(model_path)), str(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        if model_path.suffix != ".pt":
            raise ValueError(f"Unsupported SmolVLA model path format: {model_path}")

        checkpoint = torch.load(str(model_path), map_location="cpu")
        if "args" not in checkpoint:
            raise KeyError(f"Missing 'args' in SmolVLA checkpoint: {model_path}")
        if "model" not in checkpoint:
            raise KeyError(f"Missing 'model' in SmolVLA checkpoint: {model_path}")
        base_model_path = checkpoint["args"].get("smolvla_pretrained_path")
        if not base_model_path:
            raise KeyError(f"Missing args.smolvla_pretrained_path in SmolVLA checkpoint: {model_path}")
        if not Path(base_model_path).is_dir():
            raise FileNotFoundError(f"Base pretrained_model directory not found: {base_model_path}")

        policy = SmolVLAPolicy.from_pretrained(str(base_model_path))
        base_state = {}
        for key, value in checkpoint["model"].items():
            if key.startswith("base_policy."):
                base_state[key.removeprefix("base_policy.")] = value
        if not base_state:
            raise ValueError(f"No base_policy.* weights found in checkpoint: {model_path}")
        missing, unexpected = policy.load_state_dict(base_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected base policy keys when loading {model_path}: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing base policy keys when loading {model_path}: {missing}")
        return policy, str(base_model_path)

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
