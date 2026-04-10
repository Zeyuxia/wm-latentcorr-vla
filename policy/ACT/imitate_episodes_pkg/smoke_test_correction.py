from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ACT_ROOT = os.path.realpath(os.path.join(_THIS_DIR, ".."))
if _ACT_ROOT not in sys.path:
    sys.path.insert(0, _ACT_ROOT)
_EVAC_ROOT = os.path.join(_ACT_ROOT, "evac")
if _EVAC_ROOT not in sys.path:
    sys.path.insert(0, _EVAC_ROOT)

from constants import SIM_TASK_CONFIGS
from imitate_episodes_pkg.correction import correction_step, load_raw_data
from imitate_episodes_pkg.utils import build_evac_infer_kwargs, phase_id_to_key
from imitate_episodes_pkg.training import init_correction, make_policy
from utils import load_data


@contextmanager
def _temporary_argv(argv):
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def _build_args():
    parser = argparse.ArgumentParser(description="Smoke test ACT correction_step on one real sample.")
    parser.add_argument("--task_name", default="sim-open_laptop-demo_clean-50")
    parser.add_argument(
        "--policy_ckpt",
        default="/data/zhenyangfan/RoboTwin/policy/ACT/act_ckpt/act-open_laptop/demo_clean-50/20260331_103042_pregrasp_only/policy_epoch_200_seed_0.ckpt",
    )
    parser.add_argument(
        "--evac_ckpt",
        default="/data/yujieyang/EVAC/runs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/logs/evac_robotwin_mixed50p12_2026-03-16T21-19-22/checkpoints/epoch=3124-step=12500.ckpt",
    )
    parser.add_argument(
        "--evac_config",
        default="/data/zhenyangfan/RoboTwin/policy/ACT/evac/configs/robotwin/train_config.yaml",
    )
    parser.add_argument(
        "--urdf_path",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
    )
    parser.add_argument(
        "--curobo_left_yml",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml",
    )
    parser.add_argument(
        "--curobo_right_yml",
        default="/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml",
    )
    parser.add_argument(
        "--raw_data_dir",
        default="/data/zhenyangfan/RoboTwin/data/open_laptop/demo_clean/data",
    )
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_phase_window_len", type=int, default=30)
    parser.add_argument("--sample_skip_head_ratio", type=float, default=0.25)
    parser.add_argument("--rollout_exec_steps", type=int, default=16)
    parser.add_argument("--max_rollout_steps", type=int, default=1)
    parser.add_argument("--target_mode", default="backward")
    parser.add_argument("--target_lookahead_steps", type=int, default=4)
    parser.add_argument("--correction_force_generate", type=str, default="true")
    parser.add_argument("--debug_correction_evac_rollout", type=str, default="false")
    parser.add_argument("--orient_weight", type=float, default=0.0573)
    parser.add_argument("--gripper_penalty", type=float, default=1.0)
    parser.add_argument("--enable_perturb", type=str, default="true")
    parser.add_argument("--perturb_eef_fail_gain", type=float, default=0.10)
    parser.add_argument("--perturb_rot_max_deg", type=float, default=15.0)
    parser.add_argument("--perturb_mag_random", type=str, default="false")
    parser.add_argument("--perturb_mag_rand_min", type=float, default=1.0)
    parser.add_argument("--perturb_mag_rand_max", type=float, default=1.4)
    parser.add_argument("--perturb_active_joint_delta_thresh", type=float, default=0.01)
    parser.add_argument("--perturb_active_gripper_delta_thresh", type=float, default=0.05)
    parser.add_argument("--export_dir", default="/home/zhenyangfan/act_correction_smoke")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    args = _build_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.export_dir, exist_ok=True)

    task_cfg = SIM_TASK_CONFIGS[args.task_name]
    dataset_dir = os.path.join("/data/zhenyangfan/RoboTwin/policy/ACT", task_cfg["dataset_dir"].lstrip("./"))
    camera_names = task_cfg["camera_names"]
    num_episodes = int(task_cfg["num_episodes"])

    train_loader, _, stats, _, max_action_len = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        args.batch_size,
        args.batch_size,
        raw_data_dir=args.raw_data_dir,
        start_margin=int(args.max_rollout_steps) * int(args.rollout_exec_steps),
        sample_skip_head_ratio=float(args.sample_skip_head_ratio),
        sample_phase_window_len=int(args.sample_phase_window_len),
    )

    data = next(iter(train_loader))
    image_data, qpos_data = data[0], data[1]
    ep_id = int(data[4][0].item())
    start_ts = int(data[5][0].item())
    sampled_phase_id = int(data[6][0].item())
    sampled_phase_key = phase_id_to_key(sampled_phase_id)
    pregrasp_seg_start = int(data[7][0].item())
    pregrasp_seg_end = int(data[8][0].item())

    policy_config = {
        "lr": 4e-5,
        "chunk_size": int(args.chunk_size),
        "num_queries": int(args.chunk_size),
        "kl_weight": int(args.kl_weight),
        "hidden_dim": int(args.hidden_dim),
        "dim_feedforward": int(args.dim_feedforward),
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
    }
    fake_argv = [
        "smoke_test_correction.py",
        "--ckpt_dir",
        args.export_dir,
        "--policy_class",
        "ACT",
        "--task_name",
        args.task_name,
        "--seed",
        "0",
        "--num_epochs",
        "1",
        "--state_dim",
        "14",
    ]
    with _temporary_argv(fake_argv):
        policy = make_policy("ACT", policy_config)
    ckpt = torch.load(args.policy_ckpt, map_location="cpu")
    policy.load_state_dict(ckpt)
    policy = policy.to(device)
    policy.eval()

    corr_args = {
        "evac_ckpt": args.evac_ckpt,
        "evac_config": args.evac_config,
        "urdf_path": args.urdf_path,
        "curobo_left_yml": args.curobo_left_yml,
        "curobo_right_yml": args.curobo_right_yml,
    }
    modules = init_correction(corr_args, device)

    cfg = {
        "max_rollout_steps": int(args.max_rollout_steps),
        "correction_force_generate": _bool(args.correction_force_generate),
        "debug_correction_evac_rollout": _bool(args.debug_correction_evac_rollout),
        "target_mode": str(args.target_mode),
        "target_lookahead_steps": int(args.target_lookahead_steps),
        "chunk_size": int(args.chunk_size),
        "max_action_len": int(max_action_len),
        "rollout_exec_steps": int(args.rollout_exec_steps),
        "orient_weight": float(args.orient_weight),
        "gripper_penalty": float(args.gripper_penalty),
        "evac_infer_kwargs": build_evac_infer_kwargs(vars(args)),
        "enable_perturb": _bool(args.enable_perturb),
        "perturb_eef_fail_gain": float(args.perturb_eef_fail_gain),
        "perturb_rot_max_deg": float(args.perturb_rot_max_deg),
        "perturb_mag_random": _bool(args.perturb_mag_random),
        "perturb_mag_rand_min": float(args.perturb_mag_rand_min),
        "perturb_mag_rand_max": float(args.perturb_mag_rand_max),
        "perturb_active_joint_delta_thresh": float(args.perturb_active_joint_delta_thresh),
        "perturb_active_gripper_delta_thresh": float(args.perturb_active_gripper_delta_thresh),
        "sample_phase_window_len": int(args.sample_phase_window_len),
    }

    raw = load_raw_data(args.raw_data_dir, ep_id)
    debug_dir = os.path.join(args.export_dir, f"episode_{ep_id}_ts_{start_ts}")
    os.makedirs(debug_dir, exist_ok=True)

    corr = correction_step(
        policy,
        image_data[0],
        qpos_data[0],
        raw,
        stats,
        modules,
        cfg,
        device,
        debug_dir=debug_dir,
        start_ts=start_ts,
        sampled_phase_id=sampled_phase_id,
        pregrasp_seg_start=pregrasp_seg_start,
        pregrasp_seg_end=pregrasp_seg_end,
    )

    result = {
        "episode_id": ep_id,
        "start_ts": start_ts,
        "sampled_phase_id": sampled_phase_id,
        "sampled_phase_key": sampled_phase_key,
        "pregrasp_seg_start": pregrasp_seg_start,
        "pregrasp_seg_end": pregrasp_seg_end,
        "debug_dir": debug_dir,
        "triggered": corr is not None,
    }

    if corr is not None:
        corr_image, corr_qpos, corr_action, corr_is_pad, corr_meta = corr
        corr_generated = bool(
            (corr_image is not None) and (corr_qpos is not None)
            and (corr_action is not None) and (corr_is_pad is not None)
        )
        result["correction_generated"] = corr_generated
        result["corr_meta"] = corr_meta
        if corr_generated:
            valid_len = int((~corr_is_pad).sum().item())
            result.update(
                {
                    "corr_image_shape": list(corr_image.shape),
                    "corr_qpos_shape": list(corr_qpos.shape),
                    "corr_action_shape": list(corr_action.shape),
                    "valid_action_len": valid_len,
                    "first_action_raw": (
                        corr_action[0].detach().cpu().numpy() * stats["action_std"] + stats["action_mean"]
                    ).astype(np.float32).tolist(),
                    "last_valid_action_raw": (
                        corr_action[max(0, valid_len - 1)].detach().cpu().numpy() * stats["action_std"] + stats["action_mean"]
                    ).astype(np.float32).tolist(),
                }
            )

    out_path = os.path.join(args.export_dir, "smoke_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"[smoke] result saved to {out_path}")


if __name__ == "__main__":
    main()
