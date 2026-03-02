#!/usr/bin/env python3
"""Closed-loop evaluation using EVAC world model in place of simulator.

Reuses EVAC utility functions from policy/ACT/imitate_episodes.py.
Policy-agnostic: works with any policy implementing get_model/reset_model
in its deploy_policy.py (loaded via deploy_policy.yml --config).

Usage:
    python script/eval_evac.py --config policy/ACT/deploy_policy.yml \
        --overrides --task_name open_laptop --ckpt_dir ... \
        --evac_ckpt ... --evac_config ... --raw_data_dir ...
"""
import sys
import os
import types

# --- Path setup (resolve from script location, not cwd) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_ACT_DIR = os.path.join(_ROOT, 'policy', 'ACT')

for _p in [_ROOT, os.path.join(_ROOT, 'policy'), _ACT_DIR, os.path.join(_ACT_DIR, 'evac')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Mock sim_env to avoid dm_control dependency when importing imitate_episodes
if 'sim_env' not in sys.modules:
    sys.modules['sim_env'] = types.SimpleNamespace(BOX_POSE=[None])

import argparse
import yaml
import glob
import json
import time
import numpy as np
import torch
import cv2
import h5py
from datetime import datetime
from pathlib import Path
from importlib import import_module

# Reuse EVAC utilities from ACT training script (policy-independent functions)
from imitate_episodes import (
    init_correction as init_evac,
    evac_inference,
    find_nearest_traj_point,
)


# ------------------------------------------------------------------ #
#                       Data Loading                                   #
# ------------------------------------------------------------------ #

def load_episode_data(raw_data_dir, episode_id):
    """Load initial observation + expert trajectory for one episode.

    Returns dict with init_image (3,480,640 BGR [0,1] torch), init_qpos (14,),
    expert endpose/gripper trajectories, and camera parameters.
    """
    path = os.path.join(raw_data_dir, f'episode{episode_id}.hdf5')
    with h5py.File(path, 'r') as f:
        # Decode first frame (JPEG → BGR via cv2)
        img0 = cv2.imdecode(
            np.frombuffer(bytes(f['observation/head_camera/rgb'][0]), np.uint8),
            cv2.IMREAD_COLOR)
        h_n, w_n = img0.shape[:2]
        img0 = cv2.resize(img0, (640, 480))
        init_image = torch.from_numpy(
            np.moveaxis(img0, -1, 0).astype(np.float32) / 255.0)

        la = f['joint_action/left_arm'][0].astype(np.float32)
        ra = f['joint_action/right_arm'][0].astype(np.float32)
        lg = float(f['joint_action/left_gripper'][0])
        rg = float(f['joint_action/right_gripper'][0])

        return {
            'init_image': init_image,
            'init_qpos': np.array([*la, lg, *ra, rg], dtype=np.float32),
            'left_endpose': f['endpose/left_endpose'][()].astype(np.float32),
            'right_endpose': f['endpose/right_endpose'][()].astype(np.float32),
            'left_gripper': f['endpose/left_gripper'][()].astype(np.float32),
            'right_gripper': f['endpose/right_gripper'][()].astype(np.float32),
            'intrinsic_cv': f['observation/head_camera/intrinsic_cv'][0].astype(np.float32),
            'extrinsic_cv': f['observation/head_camera/extrinsic_cv'][0].astype(np.float32),
            'native_resolution': (h_n, w_n),
        }


# ------------------------------------------------------------------ #
#                     Closed-Loop Eval Episode                         #
# ------------------------------------------------------------------ #

@torch.no_grad()
def eval_episode(model, reset_fn, modules, ep_data, device, cfg):
    """Run one closed-loop episode with EVAC as the environment.

    Policy queries every query_freq steps (matching simulator behaviour).
    All actions are passed to EVAC at once via model.inference().
    """
    reset_fn(model)
    fk = modules['fk']
    evac_model = modules['evac_model']
    evac_cfg = modules['evac_config']
    query_freq = getattr(model, 'query_frequency',
                         cfg.get('policy_chunk', 1))

    curr_img = ep_data['init_image']          # (3, H, W) BGR [0,1]
    curr_qpos = ep_data['init_qpos'].copy()   # (14,)
    expert_l = ep_data['left_endpose']
    expert_r = ep_data['right_endpose']
    expert_len = len(expert_l)

    trajectory = []
    frames = [curr_img.clone()] if cfg.get('save_video') else None
    live_dir = cfg.get('live_dir')
    if live_dir:
        os.makedirs(live_dir, exist_ok=True)
        _save_frame_png(curr_img, os.path.join(live_dir, 'step_000_init.png'))

    global_step = 0
    done = False

    def _fk(q):
        r = fk.forward(q[:6], q[7:13])
        return r['left'][0], r['left'][1], r['right'][0], r['right'][1]

    while global_step < cfg['max_steps'] and not done:
        n_collect = min(query_freq, cfg['max_steps'] - global_step)
        prev_qpos = curr_qpos.copy()

        # --- Collect actions from policy ---
        actions = []
        for _ in range(n_collect):
            obs = {'head_cam': curr_img.numpy(), 'qpos': curr_qpos.tolist()}
            action = model.get_action(obs)
            if action.ndim > 1:
                action = action[0]
            curr_qpos = np.array(action, dtype=np.float32)
            actions.append(curr_qpos.copy())

        # --- FK: initial state + action outcomes ---
        fk_poses = [_fk(prev_qpos)]  # current state before actions
        grippers = [(prev_qpos[6], prev_qpos[13])]
        for a in actions:
            fk_poses.append(_fk(a))
            grippers.append((a[6], a[13]))

        # --- Evaluate trajectory distance per action ---
        for i, a in enumerate(actions):
            fp = fk_poses[i + 1]
            t_star, dist = find_nearest_traj_point(
                fp[0], fp[1], fp[2], fp[3], expert_l, expert_r,
                orient_weight=cfg['orient_weight'],
                curr_left_grip=a[6], curr_right_grip=a[13],
                left_gripper_traj=ep_data['left_gripper'],
                right_gripper_traj=ep_data['right_gripper'],
                gripper_penalty=cfg['gripper_penalty'])
            trajectory.append({'step': global_step + i,
                               't_star': int(t_star), 'dist': float(dist)})
            if t_star >= expert_len - 13:
                done = True
                break

        if done:
            break

        # --- EVAC: predict via model.inference() ---
        save_dir = cfg.get('debug_dir')
        if save_dir:
            save_dir = os.path.join(save_dir, f'step{global_step}')
        curr_img = evac_inference(
            evac_model, evac_cfg, curr_img,
            fk_poses, grippers, ep_data, device,
            save_dir=save_dir, ddim_steps=27)

        global_step += len(actions)

        if frames is not None:
            frames.append(curr_img.clone())
        if live_dir:
            last_t = trajectory[-1]
            _save_frame_png(curr_img, os.path.join(
                live_dir,
                f'step_{global_step:03d}_t{last_t["t_star"]}_d{last_t["dist"]:.4f}.png'))

    final_t = trajectory[-1]['t_star'] if trajectory else 0
    return {
        'reached_end': final_t >= expert_len - 15,
        'final_t_star': final_t,
        'expert_len': expert_len,
        'avg_dist': float(np.mean([t['dist'] for t in trajectory]))
                    if trajectory else float('inf'),
        'num_steps': len(trajectory),
        'trajectory': trajectory,
        'frames': frames,
    }


# ------------------------------------------------------------------ #
#                        Frame / Video Saving                           #
# ------------------------------------------------------------------ #

def _save_frame_png(img_tensor, path):
    """Save a (3,H,W) BGR [0,1] tensor as PNG. Flush immediately."""
    bgr = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, bgr)


def save_eval_video(frames, path, fps=10):
    """Save list of (3,H,W) BGR [0,1] tensors as mp4 via ffmpeg."""
    import subprocess
    h, w = frames[0].shape[1], frames[0].shape[2]
    proc = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-f', 'rawvideo', '-pixel_format', 'rgb24',
         '-video_size', f'{w}x{h}', '-framerate', str(fps),
         '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
         '-crf', '23', str(path)],
        stdin=subprocess.PIPE)
    for f in frames:
        rgb = (f[[2, 1, 0]].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    proc.wait()


# ------------------------------------------------------------------ #
#                              Main                                    #
# ------------------------------------------------------------------ #

def main(args):
    device = torch.device(args.get('device', 'cuda:0'))

    # --- Load policy (policy-agnostic via dynamic import) ---
    policy_name = args['policy_name']
    policy_mod = import_module(policy_name)
    model = policy_mod.get_model(args)
    reset_fn = policy_mod.reset_model

    # --- Init EVAC modules ---
    modules = init_evac({
        'evac_ckpt': args['evac_ckpt'],
        'evac_config': args['evac_config'],
        'urdf_path': args['urdf_path'],
        'curobo_left_yml': args['curobo_left_yml'],
        'curobo_right_yml': args['curobo_right_yml'],
    }, device)

    # --- Eval config ---
    raw_data_dir = args['raw_data_dir']
    episodes = sorted(glob.glob(os.path.join(raw_data_dir, 'episode*.hdf5')))
    n_episodes = min(args.get('num_episodes', len(episodes)), len(episodes))
    threshold = args.get('success_threshold', 0.05)
    do_video = args.get('save_video', False)
    debug = args.get('debug_evac_eval', False)

    live_frames = args.get('live_frames', True)  # save frames as PNG in real-time

    cfg = {
        'max_steps': args.get('max_steps', 300),
        'orient_weight': args.get('orient_weight', 0.01),
        'gripper_penalty': args.get('gripper_penalty', 1.0),
        'save_video': do_video,
    }

    save_dir = Path(f"eval_result/evac_eval/{args.get('task_name', 'unknown')}"
                    f"/{policy_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Run episodes ---
    print(f"\n\033[34mEVAC Eval | task={args.get('task_name')} | policy={policy_name} | "
          f"episodes={n_episodes} | max_steps={cfg['max_steps']}\033[0m\n")

    results = []
    for idx in range(n_episodes):
        ep_data = load_episode_data(raw_data_dir, idx)
        if debug:
            cfg['debug_dir'] = str(save_dir / f'ep{idx}')
            os.makedirs(cfg['debug_dir'], exist_ok=True)
        else:
            cfg['debug_dir'] = None
        cfg['live_dir'] = str(save_dir / f'ep{idx}_frames') if live_frames else None

        t0 = time.time()
        m = eval_episode(model, reset_fn, modules, ep_data, device, cfg)
        ep_time = time.time() - t0

        success = m['reached_end'] and m['avg_dist'] < threshold
        progress = m['final_t_star'] / max(m['expert_len'], 1)
        results.append({
            'episode': idx, 'success': success,
            'reached_end': m['reached_end'],
            'final_t_star': m['final_t_star'],
            'expert_len': m['expert_len'],
            'progress': round(progress, 4),
            'avg_dist': m['avg_dist'],
            'num_steps': m['num_steps'],
            'time_sec': round(ep_time, 2),
            'trajectory': m['trajectory'],
        })

        if do_video and m['frames']:
            save_eval_video(m['frames'], save_dir / f'ep{idx}.mp4')

        tag = "\033[92mOK\033[0m" if success else "\033[91mFAIL\033[0m"
        print(f"  ep{idx}: {tag}  t*={m['final_t_star']}/{m['expert_len']} ({progress*100:.0f}%)  "
              f"dist={m['avg_dist']:.4f}  steps={m['num_steps']}  time={ep_time:.1f}s")

    # --- Summary ---
    n_suc = sum(r['success'] for r in results)
    n_tot = len(results)
    avg_d = np.mean([r['avg_dist'] for r in results]) if results else 0
    avg_progress = np.mean([r['progress'] for r in results]) if results else 0
    avg_steps = np.mean([r['num_steps'] for r in results]) if results else 0
    total_time = sum(r['time_sec'] for r in results)
    avg_time = total_time / max(n_tot, 1)

    print(f"\n{'=' * 50}")
    print(f"Success: {n_suc}/{n_tot} = {n_suc / max(n_tot, 1) * 100:.1f}%  "
          f"Avg progress: {avg_progress * 100:.1f}%  "
          f"Avg dist: {avg_d:.4f}  Avg steps: {avg_steps:.0f}")
    print(f"Time: {total_time:.1f}s total, {avg_time:.1f}s/ep")

    # Save full results (with per-step trajectory)
    with open(save_dir / 'results.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'success_rate': n_suc / max(n_tot, 1),
                'avg_progress': round(float(avg_progress), 4),
                'avg_dist': float(avg_d),
                'avg_steps': round(float(avg_steps), 1),
                'total_time_sec': round(total_time, 2),
                'avg_time_sec': round(avg_time, 2),
            },
        }, f, indent=2)
    print(f"Saved to {save_dir}")


# ------------------------------------------------------------------ #
#                   Config Parsing (same as eval_policy.py)            #
# ------------------------------------------------------------------ #

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='EVAC closed-loop evaluation')
    parser.add_argument('--config', type=str, required=True,
                        help='Policy deploy config YAML (e.g. policy/ACT/deploy_policy.yml)')
    parser.add_argument('--overrides', nargs=argparse.REMAINDER,
                        help='Key-value pairs to override config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.overrides:
        for i in range(0, len(args.overrides), 2):
            key = args.overrides[i].lstrip('--')
            val = args.overrides[i + 1]
            # Handle bool strings first (eval("false") fails in Python)
            if isinstance(val, str) and val.lower() in ('true', 'false'):
                val = val.lower() == 'true'
            else:
                try:
                    val = eval(val)
                except Exception:
                    pass
            config[key] = val

    return config


if __name__ == '__main__':
    main(parse_args_and_config())
