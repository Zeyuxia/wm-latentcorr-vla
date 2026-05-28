from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import os
import sys
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBOTWIN_ROOT = os.path.realpath(os.path.join(_THIS_DIR, "..", "..", ".."))
_SMOLVLA_EVAC_ROOT = os.path.realpath(os.path.join(_THIS_DIR, "..", "evac"))
_SMOLVLA_EVAC_MODULE_ROOT = os.path.join(_SMOLVLA_EVAC_ROOT, "evac")
for _path in (_ROBOTWIN_ROOT, _SMOLVLA_EVAC_ROOT, _SMOLVLA_EVAC_MODULE_ROOT):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

from policy.SmolVLA.latentcorr.correction_step import correction_step
from policy.SmolVLA.latentcorr.correction_cosmos_rollout import build_cosmos_client_from_cfg
from policy.SmolVLA.latentcorr.correction_utils import build_evac_infer_kwargs


def _init_act_correction_modules(
    evac_ckpt: str,
    evac_config: str,
    urdf_path: str,
    curobo_left_yml: str,
    curobo_right_yml: str,
    device: torch.device,
) -> dict[str, Any]:
    import sapien
    from omegaconf import OmegaConf
    from policy.ACT.util.fk_sapien import SapienFK

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))

    # EVAC has its own `utils` package; keep it ahead of ACT's `utils.py`.
    for _path in (_SMOLVLA_EVAC_ROOT, _SMOLVLA_EVAC_MODULE_ROOT):
        if _path in sys.path:
            sys.path.remove(_path)
    sys.path.insert(0, _SMOLVLA_EVAC_ROOT)
    sys.path.insert(0, _SMOLVLA_EVAC_MODULE_ROOT)
    _act_utils = sys.modules.pop("utils", None)
    try:
        from evac.utils.general_utils import load_checkpoints, instantiate_from_config
    finally:
        if _act_utils is not None:
            sys.modules["utils"] = _act_utils

    _env_robot_dir = os.path.join(_ROBOTWIN_ROOT, "envs", "robot")
    _curobo_src_dir = os.path.join(_ROBOTWIN_ROOT, "envs", "curobo", "src")
    for _path in (_ROBOTWIN_ROOT, _env_robot_dir, _curobo_src_dir):
        if _path not in sys.path:
            sys.path.insert(0, _path)

    _prev_cwd = os.getcwd()
    os.chdir(_ROBOTWIN_ROOT)
    try:
        from planner import CuroboPlanner
    finally:
        os.chdir(_prev_cwd)

    evac_cfg = OmegaConf.load(evac_config)
    evac_cfg.model.pretrained_checkpoint = evac_ckpt
    evac_model = instantiate_from_config(evac_cfg.model)
    evac_model = load_checkpoints(evac_model, evac_cfg.model, ignore_mismatched_sizes=False)
    evac_model = evac_model.to(device)
    evac_model.eval()
    for p in evac_model.parameters():
        p.requires_grad = False

    fk = SapienFK(urdf_path)
    root_pose = sapien.Pose([0, -0.65, 0], [0.707, 0, 0, 0.707])
    left_joints = [f"fl_joint{i}" for i in range(1, 7)]
    right_joints = [f"fr_joint{i}" for i in range(1, 7)]
    planner_left = CuroboPlanner(root_pose, left_joints, fk.jnames, yml_path=curobo_left_yml, device=device)
    planner_right = CuroboPlanner(root_pose, right_joints, fk.jnames, yml_path=curobo_right_yml, device=device)
    return {
        "evac_model": evac_model,
        "evac_config": evac_cfg,
        "fk": fk,
        "planner_left": planner_left,
        "planner_right": planner_right,
    }


def _init_act_correction_modules_without_loading_evac(
    shared_evac_model,
    shared_evac_config,
    urdf_path: str,
    curobo_left_yml: str,
    curobo_right_yml: str,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    import sapien
    from policy.ACT.util.fk_sapien import SapienFK

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))

    _env_robot_dir = os.path.join(_ROBOTWIN_ROOT, "envs", "robot")
    _curobo_src_dir = os.path.join(_ROBOTWIN_ROOT, "envs", "curobo", "src")
    for _path in (_ROBOTWIN_ROOT, _env_robot_dir, _curobo_src_dir):
        if _path not in sys.path:
            sys.path.insert(0, _path)

    _prev_cwd = os.getcwd()
    os.chdir(_ROBOTWIN_ROOT)
    try:
        from planner import CuroboPlanner
    finally:
        os.chdir(_prev_cwd)

    fk = SapienFK(urdf_path)
    root_pose = sapien.Pose([0, -0.65, 0], [0.707, 0, 0, 0.707])
    left_joints = [f"fl_joint{i}" for i in range(1, 7)]
    right_joints = [f"fr_joint{i}" for i in range(1, 7)]
    planner_left = CuroboPlanner(root_pose, left_joints, fk.jnames, yml_path=curobo_left_yml, device=device)
    planner_right = CuroboPlanner(root_pose, right_joints, fk.jnames, yml_path=curobo_right_yml, device=device)
    return {
        "evac_model": shared_evac_model,
        "evac_config": shared_evac_config,
        "fk": fk,
        "planner_left": planner_left,
        "planner_right": planner_right,
    }


@dataclass
class ACTAlignedCorrectionConfig:
    max_rollout_steps: int = 1
    correction_force_generate: bool = False
    recover_eval_enable: bool = True
    recover_eval_save_video: bool = False
    save_perturb_rollout_video: bool = False
    save_correction_debug: bool = False
    recover_eval_gripper_open_thresh: float = 0.2
    recover_eval_pos_thresh_m: float = 0.03
    recover_eval_rot_thresh_deg: float = 10.0
    recover_eval_nearest_window_radius: int = 16
    recover_eval_video_bridge_steps: int = 16
    rollout_exec_steps: int = 12
    full_chunk_recovery: bool = False
    chunk_size: int = 50
    max_action_len: int = 50
    orient_weight: float = 0.0573
    gripper_penalty: float = 1.0
    enable_perturb: bool = True
    perturb_eef_fail_gain: float = 0.05
    perturb_rot_max_deg: float = 15.0
    nearest_window_radius: int = 12
    perturb_gripper_close_min: float = 0.10
    perturb_active_joint_delta_thresh: float = 0.01
    perturb_active_gripper_delta_thresh: float = 0.05
    evac_blur_filter_enable: bool = False
    evac_blur_filter_metric: str = "sharpness_ratio"
    evac_blur_filter_min_ratio: float = 0.75
    evac_blur_filter_region: str = "active_gripper_patch"
    evac_blur_filter_patch_pad_px: int = 12
    evac_blur_filter_gripper_axis_m: float = 0.04
    sample_phase_window_len: int = 30
    failure_mode: str = "off"
    failure_phase_bins: int = 3
    failure_translation_dir_bins: int = 6
    failure_translation_mag_bins: int = 1
    failure_rotation_dir_bins: int = 6
    failure_rotation_mag_bins: int = 1
    fail_fast_on_error: bool = False
    evac_use_dual_cache: bool = False
    evac_dc_v_bounds: tuple[int, ...] = ()
    evac_dc_budget: float | None = None
    evac_dc_enc_start: int = 999
    evac_dc_replay_step_noise: bool = False
    evac_dc_hf_metric: bool = False
    evac_dc_v_blur_on_reuse: bool = False
    evac_dc_v_blur_kernel: int = 3
    evac_dc_v_blur_strength: float = 0.15
    world_model_backend: str = "evac"
    sim_use_subprocess: bool = True
    sim_timeout_s: float = 180.0
    cosmos_root: str = "/data/zhenyangfan/cosmos-predict2.5"
    cosmos_python_bin: str = "/data/zhenyangfan/cosmos-predict2.5/.venv/bin/python"
    cosmos_checkpoint_path: str = ""
    cosmos_experiment: str = "robotwin_dualarm_actioncond_2b_256_320"
    cosmos_config_file: str = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
    cosmos_cuda_visible_devices: str = ""
    cosmos_context_parallel_size: int = 1
    cosmos_chunk_size: int = 12
    cosmos_guidance: int = 7
    cosmos_resolution: str = "256,320"
    cosmos_fps_downsample_ratio: int = 1
    cosmos_gripper_scale: float = 1.0
    cosmos_invert_gripper: bool = True
    cosmos_num_steps: int = 35
    cosmos_save_fps: int = 30
    cosmos_num_latent_conditional_frames: int = 1
    cosmos_action_scaler: float = 20.0
    cosmos_action_stats_path: str = ""
    cosmos_action_normalization_clip: float | None = None
    cosmos_use_quat: bool = False
    cosmos_quat_input_order: str = "wxyz"
    cosmos_prompt: str = ""
    cosmos_negative_prompt: str = ""
    cosmos_seed: int = 0
    cosmos_work_dir: str = ""
    cosmos_execution_mode: str = "direct"
    cosmos_startup_timeout_s: float = 600.0
    cosmos_request_timeout_s: float = 900.0
    world_model_compare_mode: bool = False
    world_model_compare_backends: tuple[str, ...] = ()
    world_model_compare_sim: bool = True
    world_model_compare_sim_autorun: bool = False
    world_model_compare_sim_timeout_s: float = 600.0
    world_model_compare_cosmos_autorun: bool = False
    correction_generate_on_unrecoverable: bool = False

    def to_runtime_dict(self) -> dict[str, Any]:
        runtime = self.__dict__.copy()
        runtime["evac_infer_kwargs"] = build_evac_infer_kwargs(runtime)
        return runtime


def _parse_world_model_compare_backends(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = []
        for item in list(value):
            raw_items.extend(str(item).replace(";", ",").split(","))
    backends: list[str] = []
    for item in raw_items:
        backend = str(item).strip().lower()
        if not backend:
            continue
        if backend not in {"evac", "cosmos"}:
            raise ValueError(
                f"Unsupported world-model compare backend={backend!r}; expected evac or cosmos."
            )
        if backend not in backends:
            backends.append(backend)
    return tuple(backends)


def build_act_aligned_cfg_from_args(args, max_action_len: int) -> ACTAlignedCorrectionConfig:
    evac_dc_budget = getattr(args, "evac_dc_budget", None)
    if evac_dc_budget is not None:
        evac_dc_budget = float(evac_dc_budget)
        if evac_dc_budget < 0.0:
            evac_dc_budget = None
    world_model_compare_backends = _parse_world_model_compare_backends(
        getattr(args, "world_model_compare_backends", ())
    )
    if bool(getattr(args, "world_model_compare_mode", False)) and not world_model_compare_backends:
        world_model_compare_backends = ("evac", "cosmos")
    return ACTAlignedCorrectionConfig(
        max_rollout_steps=1,
        correction_force_generate=bool(getattr(args, "correction_force_generate", False)),
        recover_eval_enable=True,
        recover_eval_save_video=bool(getattr(args, "recover_eval_save_video", False)),
        save_perturb_rollout_video=bool(getattr(args, "save_perturb_rollout_video", False)),
        save_correction_debug=bool(getattr(args, "save_correction_debug", False)),
        recover_eval_gripper_open_thresh=float(getattr(args, "recover_eval_gripper_open_thresh", 0.2)),
        recover_eval_pos_thresh_m=float(getattr(args, "recover_eval_pos_thresh_m", 0.03)),
        recover_eval_rot_thresh_deg=float(getattr(args, "recover_eval_rot_thresh_deg", 10.0)),
        recover_eval_nearest_window_radius=int(getattr(args, "recover_eval_nearest_window_radius", 16)),
        recover_eval_video_bridge_steps=int(getattr(args, "recover_eval_video_bridge_steps", 16)),
        rollout_exec_steps=int(getattr(args, "act_aligned_rollout_exec_steps", 12)),
        full_chunk_recovery=bool(getattr(args, "act_aligned_full_chunk_recovery", False)),
        chunk_size=int(getattr(args, "act_chunk_size", 50) or 50),
        max_action_len=int(max_action_len),
        orient_weight=float(getattr(args, "planner_orient_weight", 0.0573)),
        gripper_penalty=float(getattr(args, "planner_gripper_penalty", 1.0)),
        enable_perturb=bool(getattr(args, "act_aligned_enable_perturb", True)),
        perturb_eef_fail_gain=float(getattr(args, "act_aligned_perturb_eef_fail_gain", 0.05)),
        perturb_rot_max_deg=float(getattr(args, "act_aligned_perturb_rot_max_deg", 15.0)),
        nearest_window_radius=int(getattr(args, "planner_nearest_window_radius", 12)),
        perturb_gripper_close_min=float(getattr(args, "act_aligned_perturb_gripper_close_min", 0.10)),
        perturb_active_joint_delta_thresh=float(getattr(args, "planner_active_joint_delta_thresh", 0.01)),
        perturb_active_gripper_delta_thresh=float(getattr(args, "planner_active_gripper_delta_thresh", 0.05)),
        evac_blur_filter_enable=bool(getattr(args, "evac_blur_filter_enable", False)),
        evac_blur_filter_metric=str(getattr(args, "evac_blur_filter_metric", "sharpness_ratio")),
        evac_blur_filter_min_ratio=float(getattr(args, "evac_blur_filter_min_ratio", 0.75)),
        evac_blur_filter_region=str(getattr(args, "evac_blur_filter_region", "active_gripper_patch")),
        evac_blur_filter_patch_pad_px=int(getattr(args, "evac_blur_filter_patch_pad_px", 12)),
        evac_blur_filter_gripper_axis_m=float(getattr(args, "evac_blur_filter_gripper_axis_m", 0.04)),
        sample_phase_window_len=int(getattr(args, "sample_phase_window_len", 30)),
        failure_mode=str(getattr(args, "failure_mode", "off")),
        failure_phase_bins=int(getattr(args, "failure_phase_bins", 3)),
        failure_translation_dir_bins=int(getattr(args, "failure_translation_dir_bins", 6)),
        failure_translation_mag_bins=int(getattr(args, "failure_translation_mag_bins", 1)),
        failure_rotation_dir_bins=int(getattr(args, "failure_rotation_dir_bins", 6)),
        failure_rotation_mag_bins=int(getattr(args, "failure_rotation_mag_bins", 1)),
        fail_fast_on_error=bool(getattr(args, "fail_fast_on_error", False)),
        evac_use_dual_cache=bool(getattr(args, "evac_use_dual_cache", False)),
        evac_dc_v_bounds=tuple(int(x) for x in (getattr(args, "evac_dc_v_bounds", None) or ())),
        evac_dc_budget=evac_dc_budget,
        evac_dc_enc_start=int(getattr(args, "evac_dc_enc_start", 999)),
        evac_dc_replay_step_noise=bool(getattr(args, "evac_dc_replay_step_noise", False)),
        evac_dc_hf_metric=bool(getattr(args, "evac_dc_hf_metric", False)),
        evac_dc_v_blur_on_reuse=bool(getattr(args, "evac_dc_v_blur_on_reuse", False)),
        evac_dc_v_blur_kernel=int(getattr(args, "evac_dc_v_blur_kernel", 3)),
        evac_dc_v_blur_strength=float(getattr(args, "evac_dc_v_blur_strength", 0.15)),
        world_model_backend=str(getattr(args, "world_model_backend", "evac")).strip().lower(),
        sim_use_subprocess=bool(getattr(args, "sim_use_subprocess", True)),
        sim_timeout_s=float(getattr(args, "sim_timeout_s", 180.0)),
        cosmos_root=str(getattr(args, "cosmos_root", "/data/zhenyangfan/cosmos-predict2.5")),
        cosmos_python_bin=str(getattr(args, "cosmos_python_bin", "/data/zhenyangfan/cosmos-predict2.5/.venv/bin/python")),
        cosmos_checkpoint_path=str(getattr(args, "cosmos_checkpoint_path", "")),
        cosmos_experiment=str(getattr(args, "cosmos_experiment", "robotwin_dualarm_actioncond_2b_256_320")),
        cosmos_config_file=str(
            getattr(
                args,
                "cosmos_config_file",
                "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
            )
        ),
        cosmos_cuda_visible_devices=str(getattr(args, "cosmos_cuda_visible_devices", "")),
        cosmos_context_parallel_size=int(getattr(args, "cosmos_context_parallel_size", 1)),
        cosmos_chunk_size=int(getattr(args, "cosmos_chunk_size", 12)),
        cosmos_guidance=int(getattr(args, "cosmos_guidance", 7)),
        cosmos_resolution=str(getattr(args, "cosmos_resolution", "256,320")),
        cosmos_fps_downsample_ratio=int(getattr(args, "cosmos_fps_downsample_ratio", 1)),
        cosmos_gripper_scale=float(getattr(args, "cosmos_gripper_scale", 1.0)),
        cosmos_invert_gripper=bool(getattr(args, "cosmos_invert_gripper", True)),
        cosmos_num_steps=int(getattr(args, "cosmos_num_steps", 35)),
        cosmos_save_fps=int(getattr(args, "cosmos_save_fps", 30)),
        cosmos_num_latent_conditional_frames=int(getattr(args, "cosmos_num_latent_conditional_frames", 1)),
        cosmos_action_scaler=float(getattr(args, "cosmos_action_scaler", 20.0)),
        cosmos_action_stats_path=str(getattr(args, "cosmos_action_stats_path", "")),
        cosmos_action_normalization_clip=(
            None
            if getattr(args, "cosmos_action_normalization_clip", None) in {None, "", "none", "None"}
            else float(getattr(args, "cosmos_action_normalization_clip"))
        ),
        cosmos_use_quat=bool(getattr(args, "cosmos_use_quat", False)),
        cosmos_quat_input_order=str(getattr(args, "cosmos_quat_input_order", "wxyz")),
        cosmos_prompt=str(getattr(args, "cosmos_prompt", "")),
        cosmos_negative_prompt=str(getattr(args, "cosmos_negative_prompt", "")),
        cosmos_seed=int(getattr(args, "cosmos_seed", getattr(args, "seed", 0))),
        cosmos_work_dir=str(getattr(args, "cosmos_work_dir", "")),
        cosmos_execution_mode=str(getattr(args, "cosmos_execution_mode", "direct")),
        cosmos_startup_timeout_s=float(getattr(args, "cosmos_startup_timeout_s", 600.0)),
        cosmos_request_timeout_s=float(getattr(args, "cosmos_request_timeout_s", 900.0)),
        world_model_compare_mode=bool(getattr(args, "world_model_compare_mode", False)),
        world_model_compare_backends=world_model_compare_backends,
        world_model_compare_sim=bool(getattr(args, "world_model_compare_sim", True)),
        world_model_compare_sim_autorun=bool(getattr(args, "world_model_compare_sim_autorun", False)),
        world_model_compare_sim_timeout_s=float(getattr(args, "world_model_compare_sim_timeout_s", 600.0)),
        world_model_compare_cosmos_autorun=bool(getattr(args, "world_model_compare_cosmos_autorun", False)),
        correction_generate_on_unrecoverable=bool(
            getattr(args, "corr_export_dataset", False)
            or getattr(args, "correction_generate_on_unrecoverable", False)
        ),
    )


class _ACTChunkPolicyAdapter:
    def __init__(self, latent_model):
        self.latent_model = latent_model

    @property
    def training(self) -> bool:
        return bool(self.latent_model.training)

    def eval(self):
        self.latent_model.eval()
        return self

    def train(self, mode: bool = True):
        self.latent_model.train(mode)
        return self

    def build_batch(self, image_t, qpos_raw, action_chunk_raw):
        if not hasattr(self.latent_model, "build_batch"):
            raise AttributeError("Underlying latent_model does not implement build_batch")
        return self.latent_model.build_batch(
            image_t=image_t,
            qpos_raw=qpos_raw,
            action_chunk_raw=action_chunk_raw,
        )

    def predict_base_action_chunk(self, batch):
        if not hasattr(self.latent_model, "latent_policy"):
            raise AttributeError("Underlying latent_model does not expose latent_policy")
        return self.latent_model.latent_policy.base_policy.predict_action_chunk(batch)

    def postprocess_action_chunk(self, action_chunk):
        if not hasattr(self.latent_model, "postprocess_action_chunk"):
            raise AttributeError("Underlying latent_model does not implement postprocess_action_chunk")
        return self.latent_model.postprocess_action_chunk(action_chunk)

    def __call__(self, qpos, image):
        return self.latent_model.predict_act_chunk(qpos, image)


class ACTAlignedCorrectionBuilder:
    """Thin adapter that reuses the ACT-style correction generation inside SmolVLA."""

    def __init__(
        self,
        cfg: ACTAlignedCorrectionConfig,
        urdf_path: str,
        curobo_left_yml: str,
        curobo_right_yml: str,
        device: str | torch.device = "cuda:0",
        evac_ckpt: str | None = None,
        evac_config: str | None = None,
        shared_evac_model=None,
        shared_evac_config=None,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        backend = str(getattr(self.cfg, "world_model_backend", "evac")).strip().lower()
        if backend not in {"evac", "cosmos", "sim"}:
            raise ValueError(f"Unsupported world_model_backend={backend!r}; expected 'evac', 'cosmos', or 'sim'.")
        compare_backends = _parse_world_model_compare_backends(
            getattr(self.cfg, "world_model_compare_backends", ())
        )
        compare_enabled = bool(getattr(self.cfg, "world_model_compare_mode", False))
        compare_cosmos_autorun = bool(getattr(self.cfg, "world_model_compare_cosmos_autorun", False))
        need_evac = bool(backend == "evac" or (compare_enabled and "evac" in compare_backends))
        need_cosmos = bool(
            backend == "cosmos"
            or (compare_enabled and "cosmos" in compare_backends and compare_cosmos_autorun)
        )
        if need_evac and shared_evac_model is not None and shared_evac_config is not None:
            self.modules = _init_act_correction_modules_without_loading_evac(
                shared_evac_model=shared_evac_model,
                shared_evac_config=shared_evac_config,
                urdf_path=urdf_path,
                curobo_left_yml=curobo_left_yml,
                curobo_right_yml=curobo_right_yml,
                device=self.device,
            )
        elif need_evac:
            if evac_ckpt is None or evac_config is None:
                raise ValueError("Either shared EVAC model/config or evac_ckpt+evac_config must be provided.")
            self.modules = _init_act_correction_modules(
                evac_ckpt=evac_ckpt,
                evac_config=evac_config,
                urdf_path=urdf_path,
                curobo_left_yml=curobo_left_yml,
                curobo_right_yml=curobo_right_yml,
                device=self.device,
            )
        else:
            # Cosmos can run in a separate worker/direct runner. If EVAC is not
            # needed for this run, keep only FK/planners in the policy process.
            self.modules = _init_act_correction_modules_without_loading_evac(
                shared_evac_model=None,
                shared_evac_config=None,
                urdf_path=urdf_path,
                curobo_left_yml=curobo_left_yml,
                curobo_right_yml=curobo_right_yml,
                device=self.device,
            )
        if need_cosmos:
            if not str(getattr(self.cfg, "cosmos_checkpoint_path", "")).strip():
                raise ValueError("cosmos_checkpoint_path is required when world_model_backend='cosmos'.")
            self.modules["cosmos_client"] = build_cosmos_client_from_cfg(
                self.cfg.to_runtime_dict(),
                device=self.device,
            )
        self.fk = self.modules["fk"]
        self._last_skip_meta: dict[str, Any] | None = None

    def pop_last_skip_meta(self) -> dict[str, Any] | None:
        meta = self._last_skip_meta
        self._last_skip_meta = None
        return meta

    def build(
        self,
        latent_model,
        image_t: torch.Tensor,
        qpos_t: torch.Tensor,
        raw_data: dict[str, Any],
        norm_stats: dict[str, np.ndarray],
        start_ts: int,
        failure_mode_override: str | None = None,
        sampled_phase_id: int | None = None,
        sampled_phase_bin_id: int | None = None,
        sampled_phase_instance_id: int | None = None,
        forced_error_mode_id: int | None = None,
        sampled_active_arm_pattern_id: int | None = None,
        original_active_arm_pattern_id: int | None = None,
        forced_dir_bin_id: int | None = None,
        forced_mag_bin_id: int | None = None,
        sampled_mode_prob: float | None = None,
        sampled_entry_prob_within_mode: float | None = None,
        sampled_unit_prob: float | None = None,
        debug_dir: str | None = None,
        precomputed_action_chunk_norm: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        self._last_skip_meta = None
        adapter = _ACTChunkPolicyAdapter(latent_model)
        runtime_cfg = self.cfg.to_runtime_dict()
        if failure_mode_override is not None:
            runtime_cfg["failure_mode"] = str(failure_mode_override).strip().lower()
        runtime_cfg["correction_force_generate"] = bool(runtime_cfg["failure_mode"] == "train")
        precomputed_action_chunk_raw = None
        if precomputed_action_chunk_norm is not None:
            chunk_norm = precomputed_action_chunk_norm.detach().to(dtype=torch.float32, device="cpu")
            if chunk_norm.ndim == 3:
                if chunk_norm.shape[0] != 1:
                    raise ValueError(
                        f"Expected precomputed_action_chunk_norm batch dim to be 1, got {tuple(chunk_norm.shape)}"
                    )
                chunk_norm = chunk_norm[0]
            if chunk_norm.ndim != 2 or chunk_norm.shape[1] != 14:
                raise ValueError(
                    f"Expected precomputed_action_chunk_norm with shape [T,14], got {tuple(chunk_norm.shape)}"
                )
            action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32)
            action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32)
            precomputed_action_chunk_raw = chunk_norm * action_std.view(1, -1) + action_mean.view(1, -1)
            precomputed_action_chunk_raw = precomputed_action_chunk_raw.numpy()
            precomputed_action_chunk_raw[..., 6] = np.clip(precomputed_action_chunk_raw[..., 6], 0.0, 1.0)
            precomputed_action_chunk_raw[..., 13] = np.clip(precomputed_action_chunk_raw[..., 13], 0.0, 1.0)
        if precomputed_action_chunk_raw is None:
            raise ValueError(
                "ACTAlignedCorrectionBuilder.build requires precomputed_action_chunk_norm. "
                "Explore/correction generation should always start from the dataset GT action chunk."
            )
        try:
            corr = correction_step(
                adapter,
                image_t,
                qpos_t,
                raw_data,
                norm_stats,
                self.modules,
                runtime_cfg,
                self.device,
                debug_dir=debug_dir,
                start_ts=int(start_ts),
                sampled_phase_id=sampled_phase_id,
                sampled_phase_bin_id=sampled_phase_bin_id,
                sampled_phase_instance_id=sampled_phase_instance_id,
                forced_error_mode_id=forced_error_mode_id,
                sampled_active_arm_pattern_id=sampled_active_arm_pattern_id,
                original_active_arm_pattern_id=original_active_arm_pattern_id,
                forced_dir_bin_id=forced_dir_bin_id,
                forced_mag_bin_id=forced_mag_bin_id,
                sampled_mode_prob=sampled_mode_prob,
                sampled_entry_prob_within_mode=sampled_entry_prob_within_mode,
                sampled_unit_prob=sampled_unit_prob,
                precomputed_action_chunk_raw=precomputed_action_chunk_raw,
            )
        except Exception as exc:
            if bool(runtime_cfg.get("fail_fast_on_error", False)):
                raise
            self._last_skip_meta = {
                "correction_generated": False,
                "correction_branch": "error",
                "skip_reason": "builder_exception",
                "skip_error": repr(exc),
                "skip_traceback": traceback.format_exc(),
            }
            return None
        if corr is None:
            self._last_skip_meta = {
                "correction_generated": False,
                "correction_branch": "other",
                "skip_reason": "no_valid_correction",
            }
            return None

        corr_image, corr_qpos_norm, corr_action_norm, corr_is_pad, corr_meta = corr
        runtime_failure_mode = str(runtime_cfg.get("failure_mode", "")).strip().lower()
        if corr_image is None or corr_qpos_norm is None or corr_action_norm is None or corr_is_pad is None:
            if runtime_failure_mode == "explore" and isinstance(corr_meta, dict):
                return {
                    "corr_image": None,
                    "corr_qpos_norm": None,
                    "corr_action_chunk_norm": None,
                    "corr_is_pad": None,
                    "corr_valid_len": 0,
                    "corr_meta": corr_meta,
                    "error_action_prefix_raw": None,
                    "error_action_prefix_norm": None,
                    "error_is_pad_prefix": None,
                }
            self._last_skip_meta = corr_meta if isinstance(corr_meta, dict) else {
                "correction_generated": False,
                "correction_branch": "other",
                "skip_reason": "meta_only_correction_without_targets",
            }
            return None

        valid_len = int((~corr_is_pad).sum().item())
        error_action_prefix_raw = None
        error_action_prefix_norm = None
        error_is_pad_prefix = None
        if isinstance(corr_meta, dict) and corr_meta.get("error_action_prefix_raw") is not None:
            error_action_prefix_raw_np = np.asarray(corr_meta["error_action_prefix_raw"], dtype=np.float32)
            if error_action_prefix_raw_np.ndim != 2 or error_action_prefix_raw_np.shape[1] != 14:
                raise ValueError(
                    f"Expected error_action_prefix_raw with shape [T,14], got {error_action_prefix_raw_np.shape}"
                )
            prefix_steps = int(self.cfg.rollout_exec_steps)
            if hasattr(latent_model, "bridge_cfg") and hasattr(latent_model.bridge_cfg, "prefix_steps"):
                prefix_steps = int(latent_model.bridge_cfg.prefix_steps)
            prefix_steps = int(np.clip(prefix_steps, 1, max(1, self.cfg.max_action_len)))
            prefix_len = int(error_action_prefix_raw_np.shape[0])
            if prefix_len > prefix_steps:
                error_action_prefix_raw_np = error_action_prefix_raw_np[:prefix_steps]
                prefix_len = int(error_action_prefix_raw_np.shape[0])
            error_action_prefix_raw = torch.zeros(
                (prefix_steps, 14), dtype=torch.float32, device=self.device
            )
            error_action_prefix_raw[:prefix_len] = torch.from_numpy(error_action_prefix_raw_np).to(self.device)
            if prefix_len > 0 and prefix_len < prefix_steps:
                error_action_prefix_raw[prefix_len:] = error_action_prefix_raw[prefix_len - 1]

            action_mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32, device=self.device)
            action_std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32, device=self.device)
            error_action_prefix_norm = (error_action_prefix_raw - action_mean.view(1, -1)) / action_std.view(1, -1)
            error_is_pad_prefix = torch.ones((prefix_steps,), dtype=torch.bool, device=self.device)
            error_is_pad_prefix[:prefix_len] = False
        return {
            "corr_image": corr_image,
            "corr_qpos_norm": corr_qpos_norm,
            "corr_action_chunk_norm": corr_action_norm,
            "corr_is_pad": corr_is_pad,
            "corr_valid_len": valid_len,
            "corr_meta": corr_meta,
            "error_action_prefix_raw": error_action_prefix_raw,
            "error_action_prefix_norm": error_action_prefix_norm,
            "error_is_pad_prefix": error_is_pad_prefix,
        }
