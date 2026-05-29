from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from policy.ACT.util.fk_sapien import SapienFK

from .common import (
    DEFAULT_ASSETS_BASE_DIR,
    DEFAULT_CAMERA_MODE,
    DEFAULT_CHECKPOINT_BASE_DIR,
    build_train_config,
    infer_asset_repo_id_from_checkpoint,
    prepare_openpi_imports,
    resolve_checkpoint_dir,
)
from .stage1_checkpoint import build_stage1_model_from_checkpoint

prepare_openpi_imports()
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _extract_rgb(container: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in container:
            value = container[key]
            if isinstance(value, dict) and "rgb" in value:
                return value["rgb"]
            return value
    raise KeyError(f"Unable to find any camera key from {keys}")


def _extract_state(observation: dict[str, Any]) -> np.ndarray:
    if "joint_action" in observation:
        joint_action = observation["joint_action"]
        if "vector" in joint_action:
            return np.asarray(joint_action["vector"], dtype=np.float32)
        return np.asarray(
            joint_action["left_arm"]
            + [joint_action["left_gripper"]]
            + joint_action["right_arm"]
            + [joint_action["right_gripper"]],
            dtype=np.float32,
        )
    if "qpos" in observation:
        return np.asarray(observation["qpos"], dtype=np.float32)
    raise KeyError("Unsupported observation format: missing joint_action/qpos")


def _to_chw(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        return arr
    if arr.shape[-1] == 3:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _load_checkpoint_norm_stats(checkpoint_dir: Path, asset_repo_id: str) -> dict[str, Any] | None:
    assets_dir = checkpoint_dir / "assets"
    primary_dir = assets_dir / asset_repo_id
    if (primary_dir / "norm_stats.json").exists():
        return _normalize.load(primary_dir)
    matches = sorted(assets_dir.glob("**/norm_stats.json"))
    if not matches:
        return None
    return _normalize.load(matches[0].parent)


def _extract_runtime_obs(observation: dict[str, Any], camera_mode: str) -> dict[str, Any]:
    if "observation" in observation:
        camera_source = observation["observation"]
    else:
        camera_source = observation

    head_value = None
    for key in ("head_camera", "head_cam", "cam_high"):
        if key in camera_source:
            head_value = camera_source[key]
            break
    if head_value is None:
        raise KeyError("Unable to locate head camera in observation")

    if isinstance(head_value, dict) and "rgb" in head_value:
        head_rgb = head_value["rgb"]
        intrinsic = head_value.get("intrinsic_cv")
        extrinsic = head_value.get("extrinsic_cv")
    else:
        head_rgb = head_value
        intrinsic = camera_source.get("intrinsic_cv")
        extrinsic = camera_source.get("extrinsic_cv")

    images = {
        "cam_high": _to_chw(head_rgb),
    }
    if camera_mode == "tri_view":
        images["cam_right_wrist"] = _to_chw(_extract_rgb(camera_source, ("right_camera", "right_cam", "cam_right_wrist")))
        images["cam_left_wrist"] = _to_chw(_extract_rgb(camera_source, ("left_camera", "left_cam", "cam_left_wrist")))
    elif camera_mode != "head_only":
        raise ValueError(f"Unsupported camera_mode: {camera_mode}")

    input_state = _extract_state(observation)
    head_hwc = np.asarray(head_rgb)
    if head_hwc.ndim == 3 and head_hwc.shape[0] == 3:
        native_resolution = (int(head_hwc.shape[1]), int(head_hwc.shape[2]))
    else:
        native_resolution = (int(head_hwc.shape[0]), int(head_hwc.shape[1]))

    extrinsic_np = None
    if extrinsic is not None:
        extrinsic_np = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic_np.shape == (4, 4):
            extrinsic_np = extrinsic_np[:3, :]

    intrinsic_np = None
    if intrinsic is not None:
        intrinsic_np = np.asarray(intrinsic, dtype=np.float32)

    return {
        "images": images,
        "state": input_state,
        "intrinsic_cv": intrinsic_np,
        "extrinsic_cv": extrinsic_np,
        "native_resolution": native_resolution,
    }


def encode_obs(observation: dict[str, Any], camera_mode: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    runtime_obs = _extract_runtime_obs(observation, camera_mode)
    return runtime_obs["images"], runtime_obs["state"]


class PI0OpenLoopModel:
    def __init__(self, usr_args: dict[str, Any]):
        train_config_name = str(usr_args["train_config_name"])
        exp_name = str(usr_args.get("model_name") or usr_args.get("exp_name") or "").strip()
        if not exp_name:
            raise ValueError("PI0_LatentCorr requires --model_name or --exp_name")

        checkpoint_base_dir = usr_args.get("checkpoint_base_dir") or str(DEFAULT_CHECKPOINT_BASE_DIR)
        checkpoint_dir = resolve_checkpoint_dir(
            train_config_name=train_config_name,
            exp_name=exp_name,
            checkpoint_id=usr_args.get("checkpoint_id", "latest"),
            checkpoint_base_dir=checkpoint_base_dir,
        )

        asset_repo_id = usr_args.get("asset_repo_id") or usr_args.get("repo_id")
        if not asset_repo_id:
            asset_repo_id = infer_asset_repo_id_from_checkpoint(checkpoint_dir)
        if not asset_repo_id:
            raise ValueError(
                "Unable to infer asset_repo_id from checkpoint assets. Pass --asset_repo_id explicitly."
            )

        self.camera_mode = str(usr_args.get("camera_mode", DEFAULT_CAMERA_MODE))
        repo_id = str(usr_args.get("repo_id") or asset_repo_id)
        cfg = build_train_config(
            train_config_name=train_config_name,
            repo_id=repo_id,
            exp_name=exp_name,
            camera_mode=self.camera_mode,
            asset_id=str(asset_repo_id),
            assets_base_dir=usr_args.get("assets_base_dir") or str(DEFAULT_ASSETS_BASE_DIR),
            checkpoint_base_dir=checkpoint_base_dir,
        )

        device = str(usr_args.get("device", "cuda:0"))
        checkpoint_norm_stats = _load_checkpoint_norm_stats(checkpoint_dir, str(asset_repo_id))
        self.policy = _policy_config.create_trained_policy(
            cfg,
            checkpoint_dir,
            robotwin_repo_id=str(asset_repo_id),
            pytorch_device=device,
            norm_stats=checkpoint_norm_stats,
        )
        self.train_config_name = train_config_name
        self.exp_name = exp_name
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.asset_repo_id = str(asset_repo_id)
        self.pi0_step = int(usr_args.get("pi0_step", 50))
        self.instruction: str | None = None
        self.observation_window: dict[str, Any] | None = None

        print(
            "[PI0_LatentCorr] loaded open-loop policy | "
            f"config={self.train_config_name} | exp={self.exp_name} | step={self.checkpoint_dir.name} | "
            f"asset_repo_id={self.asset_repo_id} | camera_mode={self.camera_mode} | pi0_step={self.pi0_step}"
        )

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def update_observation_window(self, images: dict[str, np.ndarray], state: np.ndarray) -> None:
        self.observation_window = {
            "state": np.asarray(state, dtype=np.float32),
            "images": {name: np.asarray(image) for name, image in images.items()},
            "prompt": self.instruction,
        }

    def get_action(self) -> np.ndarray:
        if self.observation_window is None:
            raise RuntimeError("update_observation_window must be called before get_action")
        return self.policy.infer(self.observation_window)["actions"]

    def reset_observation_windows(self) -> None:
        self.instruction = None
        self.observation_window = None


class PI0LatentStage1Deploy:
    def __init__(self, usr_args: dict[str, Any]):
        self.usr_args = dict(usr_args)
        self.device = torch.device(str(self.usr_args.get("device", "cuda:0")))
        self.camera_mode = str(self.usr_args.get("camera_mode", DEFAULT_CAMERA_MODE))
        self.inference_mode = str(self.usr_args.get("inference_mode", "bridge")).strip().lower()
        if self.inference_mode not in {"base", "teacher", "bridge"}:
            raise ValueError(f"Unsupported inference_mode={self.inference_mode}")
        self.temporal_agg = _to_bool(self.usr_args.get("temporal_agg", False))
        self.max_timesteps = int(self.usr_args.get("max_timesteps", 3000))
        self.num_steps = int(self.usr_args.get("num_steps", 10))
        self.pi0_step = int(self.usr_args.get("pi0_step", 50))
        self.stage1_ckpt = str(self.usr_args.get("stage1_ckpt") or self.usr_args.get("latent_ckpt_path") or "").strip()
        if not self.stage1_ckpt:
            raise ValueError("Latent deploy requires --stage1_ckpt or --latent_ckpt_path")

        self.model, self.ckpt_args, load_meta = build_stage1_model_from_checkpoint(self.stage1_ckpt, device=self.device)
        print(
            "[PI0_LatentCorr] loaded stage1 checkpoint | "
            f"path={self.stage1_ckpt} | step={load_meta['step']} | mode={self.inference_mode}"
        )
        if load_meta["missing_keys"] or load_meta["unexpected_keys"]:
            print(
                "[PI0_LatentCorr] checkpoint load notes | "
                f"missing={len(load_meta['missing_keys'])} unexpected={len(load_meta['unexpected_keys'])}"
            )

        self.prefix_steps = int(self.ckpt_args.get("prefix_steps", 16))
        self.query_horizon = int(self.ckpt_args.get("action_horizon", 50))
        self.query_frequency = self.query_horizon
        if self.temporal_agg:
            self.query_frequency = 1
        self.state_dim = int(self.ckpt_args.get("model_action_dim", 32))
        self.all_actions: torch.Tensor | None = None
        self.t = 0
        self.instruction: str | None = None
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.query_horizon, self.state_dim],
                device=self.device,
            )

        from policy.ACT_LatentCorr.evac_interface import EvacLatentTeacher

        evac_ckpt = self.usr_args.get("evac_ckpt") or self.ckpt_args.get("evac_ckpt")
        evac_config = self.usr_args.get("evac_config") or self.ckpt_args.get("evac_config")
        if self.inference_mode in {"teacher", "bridge"}:
            if not evac_ckpt or not evac_config:
                raise ValueError("teacher/bridge deploy requires evac_ckpt and evac_config")
            self.teacher = EvacLatentTeacher(evac_ckpt=evac_ckpt, evac_config=evac_config, device=self.device)
        else:
            self.teacher = None

        urdf_path = self.usr_args.get("urdf_path")
        if self.inference_mode in {"teacher", "bridge"}:
            if not urdf_path:
                raise ValueError("teacher/bridge deploy requires urdf_path")
            self.fk = SapienFK(urdf_path)
        else:
            self.fk = None

        repo_id = str(self.usr_args.get("asset_repo_id") or self.usr_args.get("repo_id") or self.ckpt_args.get("repo_id") or "")
        if not repo_id:
            raise ValueError("Latent deploy requires repo_id/asset_repo_id in args or stage1 checkpoint args")
        self.repo_id = repo_id
        assets_base_dir = self.usr_args.get("assets_base_dir") or str(DEFAULT_ASSETS_BASE_DIR)
        stats_root = Path(assets_base_dir).expanduser().resolve() / self.ckpt_args["train_config_name"] / self.repo_id
        norm_stats = _normalize.load(stats_root)
        self.action_mean = torch.as_tensor(norm_stats["actions"].mean, dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(norm_stats["actions"].std, dtype=torch.float32, device=self.device)
        self.qpos_mean = torch.as_tensor(norm_stats["state"].mean, dtype=torch.float32, device=self.device)
        self.qpos_std = torch.as_tensor(norm_stats["state"].std, dtype=torch.float32, device=self.device)

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def _build_prompts(self, batch_size: int) -> Sequence[str]:
        prompt = self.instruction or ""
        return [prompt] * batch_size

    def _normalize_state(self, state_raw: np.ndarray) -> torch.Tensor:
        qpos_raw = torch.from_numpy(np.asarray(state_raw, dtype=np.float32)).unsqueeze(0).to(self.device)
        qpos_norm = (qpos_raw - self.qpos_mean.view(1, -1)) / self.qpos_std.view(1, -1)
        return qpos_norm

    def _denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        return action_norm * self.action_std.view(1, 1, -1) + self.action_mean.view(1, 1, -1)

    def _post_process(self, action_norm: torch.Tensor) -> np.ndarray:
        if action_norm.ndim == 2:
            action_norm = action_norm.unsqueeze(0)
        action = self._denormalize_action(action_norm)
        action = action.detach().cpu().numpy().astype(np.float32)
        if action.shape[-1] >= 14:
            action[..., 6] = np.clip(action[..., 6], 0.0, 1.0)
            action[..., 13] = np.clip(action[..., 13], 0.0, 1.0)
        return action[0]

    def _predict_chunk(self, runtime_obs: dict[str, Any]) -> torch.Tensor:
        image_t = torch.from_numpy(runtime_obs["images"]["cam_high"]).float().unsqueeze(0).to(self.device) / 255.0
        qpos_raw = torch.from_numpy(runtime_obs["state"]).float().unsqueeze(0).to(self.device)
        qpos_t = (qpos_raw - self.qpos_mean.view(1, -1)) / self.qpos_std.view(1, -1)
        prompts = self._build_prompts(image_t.shape[0])

        with torch.no_grad():
            if self.inference_mode == "base":
                return self.model.predict_pi0_chunk(
                    qpos_norm=qpos_t,
                    image_t=image_t,
                    prompts=prompts,
                    num_steps=self.num_steps,
                )

            base_chunk = self.model.predict_pi0_chunk(
                qpos_norm=qpos_t,
                image_t=image_t,
                prompts=prompts,
                num_steps=self.num_steps,
            )
            action_prefix_norm = base_chunk[:, : self.prefix_steps, :14]
            action_prefix_raw = self._denormalize_action(base_chunk[:, : self.prefix_steps])[..., :14]
            qpos_err_norm = (action_prefix_raw[:, -1, :] - self.qpos_mean.view(1, -1)) / self.qpos_std.view(1, -1)

            if runtime_obs["intrinsic_cv"] is None or runtime_obs["extrinsic_cv"] is None:
                raise ValueError(
                    f"inference_mode={self.inference_mode} requires intrinsic_cv and extrinsic_cv in runtime observation"
                )
            raw_data = {
                "intrinsic_cv": runtime_obs["intrinsic_cv"],
                "extrinsic_cv": runtime_obs["extrinsic_cv"],
                "native_resolution": runtime_obs["native_resolution"],
            }
            rollout_latent = self.teacher.rollout_latent_from_actions(
                curr_image=image_t[0],
                curr_qpos_raw=qpos_raw[0],
                action_prefix_raw=action_prefix_raw[0],
                raw_data=raw_data,
                fk=self.fk,
                ddim_steps=int(self.usr_args.get("ddim_steps", 27)),
            ).to(device=self.device, dtype=torch.float32)
            if rollout_latent.ndim == 3:
                rollout_latent = rollout_latent.unsqueeze(0)

            if self.inference_mode == "teacher":
                latent_z = rollout_latent.detach()
            else:
                _, latent_z = self.model.predict_future_latent(
                    image_t=image_t,
                    action_prefix=action_prefix_norm,
                    wm_teacher=self.teacher,
                )
                latent_z = latent_z.detach()

            return self.model.predict_pi0_chunk_conditioned(
                qpos_norm=qpos_err_norm,
                image_t=image_t,
                latent_z=latent_z,
                alpha_latent=1.0,
                prompts=prompts,
                num_steps=self.num_steps,
            )

    def get_action(self, observation: dict[str, Any] | None = None):
        if observation is None:
            return None
        runtime_obs = _extract_runtime_obs(observation, self.camera_mode)
        if self.t % self.query_frequency == 0:
            action_norm = self._predict_chunk(runtime_obs)
            self.all_actions = torch.as_tensor(action_norm, dtype=torch.float32, device=self.device)

        if self.all_actions is None:
            raise RuntimeError("No cached action chunk available")

        if self.temporal_agg:
            self.all_time_actions[[self.t], self.t : self.t + self.query_horizon] = self.all_actions
            actions_for_curr_step = self.all_time_actions[:, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        else:
            raw_action = self.all_actions[:, self.t % self.query_frequency]

        action = self._post_process(raw_action)
        self.t += 1
        return action

    def reset_observation_windows(self) -> None:
        self.instruction = None
        self.all_actions = None
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.query_horizon, self.state_dim],
                device=self.device,
            )


def get_model(usr_args: dict[str, Any]):
    if usr_args.get("stage1_ckpt") or usr_args.get("latent_ckpt_path") or usr_args.get("inference_mode"):
        return PI0LatentStage1Deploy(usr_args)
    return PI0OpenLoopModel(usr_args)


def eval(TASK_ENV, model, observation):
    if isinstance(model, PI0OpenLoopModel):
        if model.observation_window is None:
            model.set_language(TASK_ENV.get_instruction())
        images, input_state = encode_obs(observation, model.camera_mode)
        model.update_observation_window(images, input_state)
        actions = model.get_action()[: model.pi0_step]
        for action in actions:
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            images, input_state = encode_obs(observation, model.camera_mode)
            model.update_observation_window(images, input_state)
        return

    if model.instruction is None:
        model.set_language(TASK_ENV.get_instruction())
    actions = model.get_action(observation)
    for action in actions[: model.pi0_step]:
        TASK_ENV.take_action(action)


def reset_model(model):
    model.reset_observation_windows()
