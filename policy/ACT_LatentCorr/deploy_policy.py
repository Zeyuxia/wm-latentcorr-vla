from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

import cv2
import numpy as np
import torch

from policy.ACT.util.fk_sapien import SapienFK

from .config_latent import DynamicsWarmupConfig, LatentLossConfig, LatentModelConfig
from .evac_interface import EvacLatentTeacher
from .latent_policy import ACTLatentStage1
from .train_stage1_latent import _build_act_args, _resolve_dataset_info
from .utils_latent import build_stage1_dataloader


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _resolve_ckpt_path(usr_args: dict[str, Any]) -> str:
    ckpt_path = usr_args.get("latent_ckpt_path")
    if ckpt_path:
        return os.path.realpath(ckpt_path)
    ckpt_dir = usr_args.get("ckpt_dir")
    ckpt_name = usr_args.get("ckpt_name")
    if ckpt_dir and ckpt_name:
        return os.path.realpath(os.path.join(ckpt_dir, ckpt_name))
    raise ValueError("ACT_LatentCorr requires --latent_ckpt_path or --ckpt_dir + --ckpt_name")


def _normalize_dataset_task_name(task_name: str | None, task_config: str | None, num_episodes: int | None) -> str | None:
    if not task_name:
        return None
    if task_name in {"", "null", "None"}:
        return None
    if task_name in build_stage1_dataloader.__globals__.get("SIM_TASK_CONFIGS", {}):
        return task_name
    if task_name.startswith("sim-"):
        return task_name
    cfg = (task_config or "demo_clean").strip()
    episodes = int(num_episodes) if num_episodes is not None else 50
    return f"sim-{task_name}-{cfg}-{episodes}"


def _build_model_from_ckpt_args(
    ckpt_args: dict[str, Any],
    camera_names: list[str],
    device: str,
    dataset_task_name: str,
) -> ACTLatentStage1:
    model_args = dict(ckpt_args)
    model_args.setdefault("task_name", dataset_task_name)
    latent_model_cfg = LatentModelConfig(
        projector_mid_channels=model_args["projector_mid_channels"],
        wm_adapter_mid_channels=model_args.get("wm_adapter_mid_channels", 128),
        readout_adapter_mid_channels=model_args.get("readout_adapter_mid_channels", 128),
        predictor_num_blocks=model_args["predictor_num_blocks"],
        predictor_mlp_hidden=model_args["predictor_mlp_hidden"],
        action_decoder_hidden=model_args["action_decoder_hidden"],
        action_dim=model_args["action_dim"],
        state_dim=model_args.get("state_dim", model_args["action_dim"]),
        prefix_steps=model_args["prefix_steps"],
        token_adapter_hidden_dim=model_args.get("token_adapter_hidden_dim", model_args.get("predictor_mlp_hidden", 512)),
        token_adapter_num_layers=model_args.get("token_adapter_num_layers", 2),
        token_adapter_dropout=model_args.get("token_adapter_dropout", 0.1),
    )
    latent_loss_cfg = LatentLossConfig(
        lambda_action=model_args.get("lambda_action", 1.0),
        normal_condition_keep_prob=model_args.get("normal_condition_keep_prob", 1.0),
        beta_dynamics_max=model_args.get("beta_dynamics_max", 1.0),
        lambda_wm_action_current=model_args.get("lambda_wm_action_current", 0.0),
        lambda_wm_action_future=model_args.get("lambda_wm_action_future", 0.0),
        lambda_bridge_future=model_args.get("lambda_bridge_future", 0.0),
        use_projector_detach_for_predictor=False,
        use_projector_detach_for_action_decoder=True,
        detach_act_feature_for_latent=model_args.get("detach_act_feature_for_latent", False),
        use_raw_wm_targets=model_args.get("use_raw_wm_targets", False),
        lambda_teacher_max=model_args.get("lambda_teacher_max", 0.5),
        lambda_pred_max=model_args.get("lambda_pred_max", 0.5),
        lambda_latent_max=model_args.get("lambda_latent_max", 0.3),
        lambda_token_init=model_args.get("lambda_token_init", 0.1),
        lambda_token_late=model_args.get("lambda_token_late", 0.02),
        latent_loss_type=model_args.get("latent_loss_type", "normalized_mse"),
        token_loss_type=model_args.get("token_loss_type", "mse"),
        teacher_decay_start_ratio=model_args.get("teacher_decay_start_ratio", 0.20),
        teacher_decay_end_ratio=model_args.get("teacher_decay_end_ratio", 0.90),
        pred_warmup_start_ratio=model_args.get("pred_warmup_start_ratio", 0.10),
        pred_warmup_end_ratio=model_args.get("pred_warmup_end_ratio", 0.70),
        latent_warmup_end_ratio=model_args.get("latent_warmup_end_ratio", 0.20),
        token_decay_start_ratio=model_args.get("token_decay_start_ratio", 0.20),
        token_decay_end_ratio=model_args.get("token_decay_end_ratio", 0.90),
        pred_only_finetune_start_ratio=model_args.get("pred_only_finetune_start_ratio", 0.90),
        stopgrad_wm_teacher=model_args.get("stopgrad_wm_teacher", True),
        stopgrad_teacher_token=model_args.get("stopgrad_teacher_token", True),
        stopgrad_adapter_input_for_token_loss=model_args.get("stopgrad_adapter_input_for_token_loss", False),
    )
    warmup_cfg = DynamicsWarmupConfig(
        zero_steps=model_args.get("dyn_zero_steps", 0),
        ramp_steps=model_args.get("dyn_ramp_steps", 1000),
        max_weight=1.0,
        curve=model_args.get("dyn_warmup_curve", "cosine"),
    )
    act_args = _build_act_args(camera_names, Namespace(**model_args))
    model = ACTLatentStage1(
        act_args=act_args,
        latent_model_cfg=latent_model_cfg,
        latent_loss_cfg=latent_loss_cfg,
        warmup_cfg=warmup_cfg,
    ).to(device)
    return model


def _extract_runtime_obs(observation: dict[str, Any]) -> dict[str, Any]:
    if "observation" in observation:
        head = observation["observation"]["head_camera"]
        rgb = head["rgb"]
        intrinsic = head.get("intrinsic_cv")
        extrinsic = head.get("extrinsic_cv")
        qpos = (
            observation["joint_action"]["left_arm"]
            + [observation["joint_action"]["left_gripper"]]
            + observation["joint_action"]["right_arm"]
            + [observation["joint_action"]["right_gripper"]]
        )
    else:
        rgb = observation["head_cam"]
        intrinsic = observation.get("intrinsic_cv")
        extrinsic = observation.get("extrinsic_cv")
        qpos = observation["qpos"]

    rgb_np = np.asarray(rgb)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 3:
        native_resolution = (int(rgb_np.shape[0]), int(rgb_np.shape[1]))
    elif rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
        native_resolution = (int(rgb_np.shape[1]), int(rgb_np.shape[2]))
    else:
        raise ValueError(f"Unsupported RGB observation shape: {rgb_np.shape}")

    if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
        rgb_chw = rgb_np.astype(np.float32)
        if rgb_chw.max() > 1.0:
            rgb_chw = rgb_chw / 255.0
        if rgb_chw.shape[1:] != (480, 640):
            rgb_hwc = np.moveaxis(rgb_chw, 0, -1)
            rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_LINEAR)
            rgb_chw = np.moveaxis(rgb_hwc, -1, 0)
        else:
            rgb_hwc = np.moveaxis(rgb_chw, 0, -1)
    elif rgb_np.ndim == 3 and rgb_np.shape[-1] == 3:
        rgb_hwc = rgb_np.astype(np.float32)
        if rgb_hwc.max() > 1.0:
            rgb_hwc = rgb_hwc / 255.0
        if rgb_hwc.shape[:2] != (480, 640):
            rgb_hwc = cv2.resize(rgb_hwc, (640, 480), interpolation=cv2.INTER_LINEAR)
        rgb_chw = np.moveaxis(rgb_hwc, -1, 0)

    bgr_hwc = rgb_hwc[..., ::-1].copy()
    bgr_chw = np.moveaxis(bgr_hwc, -1, 0)

    extrinsic_np = None
    if extrinsic is not None:
        extrinsic_np = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic_np.shape == (4, 4):
            extrinsic_np = extrinsic_np[:3, :]

    intrinsic_np = None
    if intrinsic is not None:
        intrinsic_np = np.asarray(intrinsic, dtype=np.float32)

    return {
        "rgb_chw": rgb_chw.astype(np.float32),
        "bgr_chw": bgr_chw.astype(np.float32),
        "qpos_raw": np.asarray(qpos, dtype=np.float32),
        "intrinsic_cv": intrinsic_np,
        "extrinsic_cv": extrinsic_np,
        "native_resolution": native_resolution,
    }


class ACTLatentDeploy:
    def __init__(self, usr_args: dict[str, Any]):
        self.usr_args = dict(usr_args)
        self.device = torch.device(self.usr_args.get("device", "cuda:0"))
        self.inference_mode = str(self.usr_args.get("inference_mode", "teacher")).strip().lower()
        if self.inference_mode not in {"base", "teacher", "bridge", "pred"}:
            raise ValueError(f"Unsupported inference_mode={self.inference_mode}")

        self.ddim_steps = int(self.usr_args.get("ddim_steps", 27))
        self.temporal_agg = _to_bool(self.usr_args.get("temporal_agg", False))
        self._loaded = False

        ckpt_path = _resolve_ckpt_path(self.usr_args)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.ckpt_state = ckpt["model"]
        self.ckpt_args = dict(ckpt["args"])
        self.ckpt_path = ckpt_path

        dataset_task_name = _normalize_dataset_task_name(
            self.ckpt_args.get("task_name"),
            self.usr_args.get("task_config"),
            self.ckpt_args.get("num_episodes"),
        )
        if not dataset_task_name:
            dataset_task_name = _normalize_dataset_task_name(
                self.usr_args.get("task_name"),
                self.usr_args.get("task_config"),
                self.ckpt_args.get("num_episodes"),
            )
        if not dataset_task_name and self.ckpt_args.get("multi_task_names"):
            dataset_task_name = str(self.ckpt_args["multi_task_names"][0])
        if not dataset_task_name:
            raise ValueError("Unable to resolve dataset task name for ACT_LatentCorr deploy")
        dataset_dir, num_episodes, camera_names = _resolve_dataset_info(dataset_task_name)
        if self.usr_args.get("dataset_dir") is not None:
            dataset_dir = os.path.realpath(self.usr_args["dataset_dir"])
        elif self.ckpt_args.get("dataset_dir") is not None:
            dataset_dir = os.path.realpath(self.ckpt_args["dataset_dir"])
        if self.usr_args.get("num_episodes") is not None:
            num_episodes = int(self.usr_args["num_episodes"])
        elif self.ckpt_args.get("num_episodes") is not None:
            num_episodes = int(self.ckpt_args["num_episodes"])
        if self.usr_args.get("camera_names") is not None:
            camera_names = list(self.usr_args["camera_names"])

        self.prefix_steps = int(self.ckpt_args["prefix_steps"])
        self.act_chunk_size = int(self.ckpt_args.get("act_chunk_size", self.prefix_steps))
        norm_stats = ckpt.get("norm_stats")
        if norm_stats is None:
            _, norm_stats = build_stage1_dataloader(
                dataset_dir=dataset_dir,
                num_episodes=num_episodes,
                camera_names=camera_names,
                act_chunk_size=self.act_chunk_size,
                prefix_steps=self.prefix_steps,
                future_offset=int(self.ckpt_args.get("future_offset") or self.prefix_steps),
                batch_size=1,
                num_workers=0,
            )
        self.action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=self.device)
        self.qpos_mean = torch.as_tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=self.device)
        self.qpos_std = torch.as_tensor(norm_stats["qpos_std"], dtype=torch.float32, device=self.device)

        latent_target_hw = self.ckpt_args.get("latent_target_hw")
        if isinstance(latent_target_hw, (list, tuple)) and len(latent_target_hw) == 2:
            self.latent_target_hw = (int(latent_target_hw[0]), int(latent_target_hw[1]))
        else:
            self.latent_target_hw = None
        proj_key = "projector.proj.3.weight"
        self.wm_channels = int(self.ckpt_state[proj_key].shape[0]) if proj_key in self.ckpt_state else None

        # Lightweight deploy modes do not use the planner/FK stack. Avoid
        # initializing SapienFK/curobo there because its CUDA extension JIT can
        # stall eval startup before the first rollout begins.
        self.fk = None
        needs_fk = self.inference_mode not in {"base", "pred"}
        if needs_fk:
            urdf_path = self.usr_args.get("urdf_path")
            if not urdf_path:
                raise ValueError(
                    f"ACT_LatentCorr deploy requires urdf_path for inference_mode={self.inference_mode}"
                )
            self.fk = SapienFK(urdf_path)

        needs_teacher = needs_fk or self.latent_target_hw is None or self.wm_channels is None
        self.teacher = None
        if needs_teacher:
            evac_ckpt = self.usr_args.get("evac_ckpt") or self.ckpt_args.get("evac_ckpt")
            evac_config = self.usr_args.get("evac_config") or self.ckpt_args.get("evac_config")
            if not evac_ckpt or not evac_config:
                raise ValueError("ACT_LatentCorr deploy requires evac_ckpt and evac_config when teacher init is needed")
            self.teacher = EvacLatentTeacher(evac_ckpt=evac_ckpt, evac_config=evac_config, device=self.device)

        self.model = _build_model_from_ckpt_args(self.ckpt_args, camera_names, str(self.device), dataset_task_name)
        self.model.eval()

        self.use_act_head_correction = _to_bool(self.ckpt_args.get("use_act_head_correction", False))
        if self.inference_mode == "base":
            self.query_horizon = self.act_chunk_size
        elif self.inference_mode == "pred":
            self.query_horizon = self.act_chunk_size
        elif self.use_act_head_correction:
            self.query_horizon = self.act_chunk_size
        else:
            self.query_horizon = self.prefix_steps
        self.query_frequency = self.query_horizon
        self.state_dim = int(self.ckpt_args.get("action_dim", 14))
        self.max_timesteps = int(self.usr_args.get("max_timesteps", 3000))
        if self.temporal_agg:
            self.query_frequency = 1
            self.all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.query_horizon, self.state_dim],
                device=self.device,
            )
        self.all_actions: torch.Tensor | None = None
        self.t = 0

        print(
            f"[ACT_LatentCorr] loaded config | mode={self.inference_mode} | "
            f"temporal_agg={self.temporal_agg} | prefix_steps={self.prefix_steps} | act_chunk_size={self.act_chunk_size}"
        )
        print(f"[ACT_LatentCorr] checkpoint={self.ckpt_path}")
        print(f"[ACT_LatentCorr] dataset_task={dataset_task_name}")
        print(f"[ACT_LatentCorr] fk_enabled={self.fk is not None}")
        print(f"[ACT_LatentCorr] teacher_enabled={self.teacher is not None}")

    def _ensure_loaded(self, image_t: torch.Tensor) -> None:
        if self._loaded:
            return
        if self.latent_target_hw is not None and self.wm_channels is not None:
            self.model.initialize_latent_heads_from_shapes(
                image_t,
                wm_channels=self.wm_channels,
                target_hw=self.latent_target_hw,
            )
        else:
            if self.teacher is None:
                raise RuntimeError("Teacher is unavailable for latent-head initialization fallback.")
            self.model.initialize_latent_heads(image_t, self.teacher)
        missing, unexpected = self.model.load_state_dict(self.ckpt_state, strict=False)
        self.model.eval()
        self._loaded = True
        print(f"[ACT_LatentCorr] model initialized | missing={len(missing)} unexpected={len(unexpected)}")

    def _project_current_latent(self, image_t: torch.Tensor) -> torch.Tensor:
        z_act = self.model._extract_act_feature(image_t)
        if self.model._latent_target_hw is None:
            raise RuntimeError("Missing latent target resolution")
        assert self.model.projector is not None
        return self.model.projector(z_act, target_hw=self.model._latent_target_hw)

    def _predict_chunk(self, runtime_obs: dict[str, Any]) -> torch.Tensor:
        image_t = torch.from_numpy(runtime_obs["rgb_chw"]).float().unsqueeze(0).unsqueeze(0).to(self.device)
        qpos_raw = torch.from_numpy(runtime_obs["qpos_raw"]).float().unsqueeze(0).to(self.device)
        qpos_t = (qpos_raw - self.qpos_mean.view(1, -1)) / self.qpos_std.view(1, -1)
        self._ensure_loaded(image_t)

        if self.inference_mode == "base":
            with torch.no_grad():
                return self.model.predict_act_chunk(qpos_t, image_t)

        if self.inference_mode == "pred":
            with torch.no_grad():
                return self.model.predict_act_chunk_predicted_conditioned(
                    qpos_t=qpos_t,
                    image_t=image_t,
                )

        if runtime_obs["intrinsic_cv"] is None or runtime_obs["extrinsic_cv"] is None:
            raise ValueError(
                f"inference_mode={self.inference_mode} requires intrinsic_cv and extrinsic_cv in runtime observation"
            )

        raw_data = {
            "intrinsic_cv": runtime_obs["intrinsic_cv"],
            "extrinsic_cv": runtime_obs["extrinsic_cv"],
            "native_resolution": runtime_obs["native_resolution"],
        }
        is_pad = torch.zeros((1, self.prefix_steps), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            stage2_ctx = self.model.prepare_stage2_context(
                image_t=image_t,
                qpos_t=qpos_t,
                qpos_raw=qpos_raw,
                wm_teacher=self.teacher,
                raw_data=raw_data,
                fk=self.fk,
                norm_stats={
                    "action_mean": self.action_mean.detach().cpu().numpy(),
                    "action_std": self.action_std.detach().cpu().numpy(),
                    "qpos_mean": self.qpos_mean.detach().cpu().numpy(),
                    "qpos_std": self.qpos_std.detach().cpu().numpy(),
                },
                ddim_steps=self.ddim_steps,
                is_pad_prefix=is_pad,
            )
            if self.inference_mode == "teacher":
                if self.use_act_head_correction:
                    return self.model.predict_act_chunk_conditioned(
                        stage2_ctx.qpos_err_norm,
                        image_t,
                        stage2_ctx.z_wm_sim_shared.detach(),
                        alpha_latent=1.0,
                    )
                return self.model.decode_action_latent(stage2_ctx.z_wm_sim_shared.detach())
            if self.use_act_head_correction:
                return self.model.predict_act_chunk_conditioned(
                    stage2_ctx.qpos_err_norm,
                    image_t,
                    stage2_ctx.z_hat_next.detach(),
                    alpha_latent=1.0,
                )
            return self.model.decode_action_latent(stage2_ctx.z_hat_next.detach())

    def _post_process(self, action_norm: torch.Tensor) -> np.ndarray:
        if action_norm.ndim == 2:
            action_norm = action_norm.unsqueeze(0)
        action = action_norm * self.action_std.view(1, 1, -1) + self.action_mean.view(1, 1, -1)
        action = action.detach().cpu().numpy().astype(np.float32)
        action[..., 6] = np.clip(action[..., 6], 0.0, 1.0)
        action[..., 13] = np.clip(action[..., 13], 0.0, 1.0)
        return action[0]

    def get_action(self, observation: dict[str, Any] | None = None):
        if observation is None:
            return None

        runtime_obs = _extract_runtime_obs(observation)
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


def encode_obs(observation: dict[str, Any]) -> dict[str, Any]:
    return _extract_runtime_obs(observation)


def get_model(usr_args):
    return ACTLatentDeploy(usr_args)


def eval(TASK_ENV, model, observation):
    actions = model.get_action(observation)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    if getattr(model, "temporal_agg", False):
        model.all_time_actions = torch.zeros(
            [model.max_timesteps, model.max_timesteps + model.prefix_steps, model.state_dim],
            device=model.device,
        )
    model.t = 0
    model.all_actions = None
