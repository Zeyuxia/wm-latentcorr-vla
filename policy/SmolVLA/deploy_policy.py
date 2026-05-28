import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

FILE_DIR = Path(__file__).resolve().parent
SRC_DIR = FILE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION
from policy.SmolVLA.latentcorr.latent_config import DynamicsWarmupConfig, Stage1WarmupConfig
from policy.SmolVLA.latentcorr.evac_interface import EvacLatentTeacher
from policy.SmolVLA.latentcorr.smolvla_latent_policy import (
    SmolVLALatentBridgeConfig,
    SmolVLALatentPolicy,
    SmolVLAStage1LossWeights,
)
from policy.SmolVLA.latentcorr.smolvla_data_utils import make_smolvla_processors


CAMERA_OBS_KEYS = {
    "head_camera": ("observation", "head_camera", "rgb"),
}

DEFAULT_EVAC_CKPT = (
    "/data/yujieyang/EVAC_new/runs/"
    "evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/"
    "checkpoints/epoch=333-step=10000.ckpt"
)
DEFAULT_EVAC_CONFIG = "/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml"


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


def _stage1_dataset_stats_for_processors(norm_stats):
    if norm_stats is None:
        return None
    if "qpos_mean" not in norm_stats or "action_mean" not in norm_stats:
        return norm_stats
    return {
        "observation.state": {
            "mean": torch.as_tensor(norm_stats["qpos_mean"]),
            "std": torch.as_tensor(norm_stats["qpos_std"]),
        },
        ACTION: {
            "mean": torch.as_tensor(norm_stats["action_mean"]),
            "std": torch.as_tensor(norm_stats["action_std"]),
        },
    }


class SmolVLAEvalWrapper:
    def __init__(self, policy_config):
        self.model_path = str(policy_config["model_path"])
        self.device = _resolve_device(policy_config.get("device", "cuda"))
        self.inference_mode = str(policy_config.get("inference_mode", "pred")).strip().lower()
        self.camera_sources = policy_config.get(
            "camera_sources",
            ["head_camera"],
        )

        (
            self.policy,
            self.latent_policy,
            processor_path,
            dataset_stats,
            self.target_hw,
            self.checkpoint_args,
        ) = self._load_policy()
        if self.latent_policy is not None:
            self.latent_policy.to(self.device)
            self.latent_policy.eval()
        self.policy.to(self.device)
        self.policy.eval()
        self._action_queue = deque()
        self._oracle_future_observations = []
        self._oracle_future_offset = int(policy_config.get("oracle_future_offset", 16))
        self._token_interp_alpha = float(policy_config.get("token_interp_alpha", 1.0))
        self._oracle_step_idx = 0
        self._oracle_teacher = None
        self._oracle_fk = None

        self.preprocess, self.postprocess = make_smolvla_processors(
            self.policy,
            processor_path,
            dataset_stats=dataset_stats,
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
            return SmolVLAPolicy.from_pretrained(str(model_path)), None, str(model_path), None, None, {}
        if not model_path.is_file():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        if model_path.suffix != ".pt":
            raise ValueError(f"Unsupported SmolVLA model path format: {model_path}")

        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if "args" not in checkpoint:
            raise KeyError(f"Missing 'args' in SmolVLA checkpoint: {model_path}")
        if "model" not in checkpoint:
            raise KeyError(f"Missing 'model' in SmolVLA checkpoint: {model_path}")
        base_model_path = checkpoint["args"].get("smolvla_pretrained_path")
        if not base_model_path:
            raise KeyError(f"Missing args.smolvla_pretrained_path in SmolVLA checkpoint: {model_path}")
        if not Path(base_model_path).is_dir():
            raise FileNotFoundError(f"Base pretrained_model directory not found: {base_model_path}")

        config = PreTrainedConfig.from_pretrained(str(base_model_path))
        state_dim = len(checkpoint["norm_stats"]["qpos_mean"]) if "norm_stats" in checkpoint else int(
            checkpoint["args"].get("action_dim", config.input_features["observation.state"].shape[0])
        )
        action_dim = int(checkpoint["args"].get("action_dim", config.output_features[ACTION].shape[0]))
        config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
        config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        }

        policy = SmolVLAPolicy.from_pretrained(str(base_model_path), config=config)
        bridge_cfg = SmolVLALatentBridgeConfig(
            action_dim=action_dim,
            prefix_steps=int(checkpoint["args"].get("prefix_steps", 16)),
            latent_dim=int(checkpoint["args"].get("latent_dim", 4)),
            adapter_hidden_dim=int(checkpoint["args"].get("adapter_hidden_dim", 512)),
            predictor_hidden_dim=int(checkpoint["args"].get("predictor_hidden_dim", 512)),
        )
        warmup_cfg = Stage1WarmupConfig(
            dynamics=DynamicsWarmupConfig(
                zero_steps=int(checkpoint["args"].get("dyn_zero_steps", 0)),
                ramp_steps=int(checkpoint["args"].get("dyn_ramp_steps", 1)),
                max_weight=float(checkpoint["args"].get("dyn_max_weight", 1.0)),
                curve=str(checkpoint["args"].get("dyn_warmup_curve", "cosine")),
            ),
            condition=DynamicsWarmupConfig(
                zero_steps=int(checkpoint["args"].get("cond_zero_steps", 0)),
                ramp_steps=int(checkpoint["args"].get("cond_ramp_steps", 1)),
                max_weight=float(checkpoint["args"].get("cond_max_weight", 1.0)),
                curve=str(checkpoint["args"].get("cond_warmup_curve", "cosine")),
            ),
        )
        loss_weights = SmolVLAStage1LossWeights(
            token_init=float(checkpoint["args"].get("token_loss_weight_init", 0.1)),
            token_late=float(checkpoint["args"].get("token_loss_weight_late", 0.02)),
            token_decay_start_ratio=float(checkpoint["args"].get("token_loss_decay_start_ratio", 0.0)),
            token_decay_end_ratio=float(checkpoint["args"].get("token_loss_decay_end_ratio", 1.0)),
            total_steps=max(1, int(checkpoint["args"].get("max_steps", 1))),
        )
        latent_policy = SmolVLALatentPolicy(
            base_policy=policy,
            bridge_cfg=bridge_cfg,
            warmup_cfg=warmup_cfg,
            loss_weights=loss_weights,
        )

        model_state = checkpoint["model"]
        projector_key = "projector.proj.0.weight"
        condition_key = "condition_proj.weight"
        if projector_key not in model_state or condition_key not in model_state:
            raise KeyError(f"Missing latent bridge keys in checkpoint: {model_path}")
        visual_channels = int(model_state[projector_key].shape[1])
        hidden_size = int(model_state[condition_key].shape[0])
        latent_policy._ensure_bridge_modules(visual_channels=visual_channels, hidden_size=hidden_size)
        missing, unexpected = latent_policy.load_state_dict(model_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Latent policy state mismatch for {model_path}: missing={missing}, unexpected={unexpected}"
            )
        target_hw = checkpoint.get("target_hw")
        if target_hw is not None:
            target_hw = (int(target_hw[0]), int(target_hw[1]))
        return (
            policy,
            latent_policy,
            str(base_model_path),
            _stage1_dataset_stats_for_processors(checkpoint.get("norm_stats")),
            target_hw,
            checkpoint.get("args", {}),
        )

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
            oracle_modes = {"oracle", "future_image_oracle", "wm_rollout_interp", "future_image_interp"}
            if self.latent_policy is not None and self.inference_mode in {"wm_rollout_interp", "future_image_interp"}:
                action = self._select_interp_action(model_input, observation)
            elif self.latent_policy is not None and self.inference_mode in {"oracle", "future_image_oracle"}:
                action = self._select_oracle_action(model_input, observation)
            elif self.latent_policy is not None and self.inference_mode in {"pred", "random"}:
                action = self._select_pred_action(model_input)
            else:
                action = self.policy.select_action(model_input)
            if self.inference_mode in oracle_modes:
                self._oracle_step_idx += 1
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

    def set_oracle_future_observations(self, observations, future_offset: int | None = None):
        self._oracle_future_observations = list(observations or [])
        if future_offset is not None:
            self._oracle_future_offset = int(future_offset)
        self._oracle_step_idx = 0
        self._action_queue.clear()

    def _get_oracle_teacher(self, load_rollout_model: bool):
        if self._oracle_teacher is not None and self._oracle_teacher.load_rollout_model == bool(load_rollout_model):
            return self._oracle_teacher
        evac_ckpt = str(self.checkpoint_args.get("evac_ckpt") or DEFAULT_EVAC_CKPT)
        evac_config = str(self.checkpoint_args.get("evac_config") or DEFAULT_EVAC_CONFIG)
        self._oracle_teacher = EvacLatentTeacher(
            evac_ckpt=evac_ckpt,
            evac_config=evac_config,
            device=self.device,
            load_rollout_model=bool(load_rollout_model),
        )
        return self._oracle_teacher

    def _get_oracle_fk(self):
        if self._oracle_fk is not None:
            return self._oracle_fk
        from policy.ACT.util.fk_sapien import SapienFK

        urdf_path = str(
            self.checkpoint_args.get("urdf_path")
            or "/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
        )
        self._oracle_fk = SapienFK(urdf_path)
        return self._oracle_fk

    def _obs_image_to_evac_tensor(self, observation):
        source_name = self.camera_sources[0]
        if source_name not in CAMERA_OBS_KEYS:
            raise KeyError(f"Unsupported camera source: {source_name}")
        image = _nested_get(observation, CAMERA_OBS_KEYS[source_name])
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Oracle image must be HWC or CHW, got {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            tensor = torch.from_numpy(image).float()
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        if float(tensor.max().item()) > 1.0:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _obs_to_rollout_raw_data(self, observation):
        source_name = self.camera_sources[0]
        image = _nested_get(observation, CAMERA_OBS_KEYS[source_name])
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Oracle rollout image must be HWC or CHW, got {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            height_native, width_native = int(image.shape[1]), int(image.shape[2])
        else:
            height_native, width_native = int(image.shape[0]), int(image.shape[1])
        camera_cfg = observation["observation"][source_name]
        return {
            "intrinsic_cv": np.asarray(camera_cfg["intrinsic_cv"], dtype=np.float32).copy(),
            "extrinsic_cv": np.asarray(camera_cfg["extrinsic_cv"], dtype=np.float32).copy(),
            "native_resolution": (height_native, width_native),
        }

    def _oracle_action_prefix_tensor(self):
        if not self._oracle_future_observations:
            raise RuntimeError("oracle rollout requires expert observations captured by eval_policy.")
        prefix_steps = int(self.latent_policy.bridge_cfg.prefix_steps)
        action_prefix = []
        for offset in range(prefix_steps):
            frame_idx = min(
                max(0, int(self._oracle_step_idx) + int(offset)),
                len(self._oracle_future_observations) - 1,
            )
            action_prefix.append(_extract_qpos(self._oracle_future_observations[frame_idx]))
        return torch.from_numpy(np.stack(action_prefix, axis=0)).float().unsqueeze(0)

    def _predicted_condition_token_from_input(self, model_input):
        if self.target_hw is None:
            raise RuntimeError("Missing target_hw in stage1 checkpoint; cannot run predicted-token inference.")
        base_chunk = self.latent_policy.base_policy.predict_action_chunk(model_input)
        prefix_steps = int(self.latent_policy.bridge_cfg.prefix_steps)
        action_dim = int(self.latent_policy.bridge_cfg.action_dim)
        action_prefix = base_chunk[:, :prefix_steps, :action_dim]
        if int(action_prefix.shape[1]) != prefix_steps:
            raise ValueError(
                f"Base action chunk too short for prefix_steps={prefix_steps}: {tuple(base_chunk.shape)}"
            )
        predicted_latent = self.latent_policy.predict_next_latent(
            model_input,
            action_prefix,
            target_hw=self.target_hw,
        )
        return self.latent_policy.build_predicted_condition_token(predicted_latent, scale=1.0)

    def _oracle_condition_token_from_input(self, model_input, observation, source: str):
        if not self._oracle_future_observations:
            raise RuntimeError("oracle inference requires expert future observations from eval_policy.")
        source = str(source).strip().lower()
        if source == "future_image":
            future_idx = min(
                max(0, int(self._oracle_step_idx) + int(self._oracle_future_offset)),
                len(self._oracle_future_observations) - 1,
            )
            future_image = self._obs_image_to_evac_tensor(self._oracle_future_observations[future_idx])
            teacher_latent = self._get_oracle_teacher(load_rollout_model=False).encode_image(future_image)
        elif source == "wm_rollout":
            current_image = self._obs_image_to_evac_tensor(observation).detach().cpu()
            current_qpos = torch.from_numpy(_extract_qpos(observation)).float().unsqueeze(0)
            action_prefix = self._oracle_action_prefix_tensor()
            raw_data = self._obs_to_rollout_raw_data(observation)
            teacher_latent = self._get_oracle_teacher(load_rollout_model=True).rollout_latent_from_actions(
                curr_image=current_image,
                curr_qpos_raw=current_qpos,
                action_prefix_raw=action_prefix,
                raw_data=raw_data,
                fk=self._get_oracle_fk(),
                ddim_steps=int(self.checkpoint_args.get("stage1_rollout_ddim_steps", 27)),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported oracle token source: {source}")
        teacher_shared = self.latent_policy.shared_teacher_latent(teacher_latent)
        return self.latent_policy.build_condition_token(teacher_shared, scale=1.0)

    def _sample_conditioned_first_action(self, model_input, token, mask, att_mask):
        action_dim = int(self.latent_policy.bridge_cfg.action_dim)
        images, img_masks, lang_tokens, lang_masks, state = self.latent_policy._policy_inputs_from_batch(model_input)
        conditioned_chunk = self.latent_policy.base_policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            external_prefix_tokens=token,
            external_prefix_mask=mask,
            external_prefix_att_mask=att_mask,
        )[:, :, :action_dim]

        n_action_steps = int(getattr(self.policy.config, "n_action_steps", 1))
        for idx in range(1, min(n_action_steps, int(conditioned_chunk.shape[1]))):
            self._action_queue.append(conditioned_chunk[:, idx, :])
        return conditioned_chunk[:, 0, :]

    @torch.no_grad()
    def _select_pred_action(self, model_input):
        if self._action_queue:
            return self._action_queue.popleft()
        if self.latent_policy is None:
            raise RuntimeError("pred inference requires a stage1 latent checkpoint.")
        if self.target_hw is None:
            raise RuntimeError("Missing target_hw in stage1 checkpoint; cannot run pred inference.")

        token, mask, att_mask = self._predicted_condition_token_from_input(model_input)
        if self.inference_mode == "random":
            token = torch.randn_like(token)
        return self._sample_conditioned_first_action(model_input, token, mask, att_mask)

    @torch.no_grad()
    def _select_oracle_action(self, model_input, observation):
        if self._action_queue:
            return self._action_queue.popleft()
        if self.latent_policy is None:
            raise RuntimeError("oracle inference requires a stage1 latent checkpoint.")
        if not self._oracle_future_observations:
            raise RuntimeError("oracle inference requires expert future observations from eval_policy.")

        source = "future_image" if self.inference_mode == "future_image_oracle" else "wm_rollout"
        token, mask, att_mask = self._oracle_condition_token_from_input(model_input, observation, source=source)
        return self._sample_conditioned_first_action(model_input, token, mask, att_mask)

    @torch.no_grad()
    def _select_interp_action(self, model_input, observation):
        if self._action_queue:
            return self._action_queue.popleft()
        if self.latent_policy is None:
            raise RuntimeError("interpolated oracle inference requires a stage1 latent checkpoint.")
        pred_token, pred_mask, pred_att = self._predicted_condition_token_from_input(model_input)
        source = "future_image" if self.inference_mode == "future_image_interp" else "wm_rollout"
        oracle_token, oracle_mask, oracle_att = self._oracle_condition_token_from_input(
            model_input,
            observation,
            source=source,
        )
        alpha = float(self._token_interp_alpha)
        token = pred_token * (1.0 - alpha) + oracle_token * alpha
        # The masks are identical single-token masks; keep the predicted-token
        # mask to preserve the deployable path's prefix placement.
        return self._sample_conditioned_first_action(model_input, token, pred_mask, pred_att)

    def reset(self):
        self._action_queue.clear()
        self._oracle_step_idx = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()


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
    if hasattr(model, "reset"):
        model.reset()
    else:
        model.policy.reset()
