import os
import sys

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
# Required for deterministic CuBLAS operations
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evac'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import pickle
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, detach_dict  # helper functions
from act_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args.get("eval", False)
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args.get("onscreen_render", False)
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14  # yiheng
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq']
    }

    # if is_eval:
    #     ckpt_names = [f"policy_best.ckpt"]
    #     results = []
    #     for ckpt_name in ckpt_names:
    #         success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
    #         results.append([ckpt_name, success_rate, avg_return])

    #     for ckpt_name, success_rate, avg_return in results:
    #         print(f"{ckpt_name}: {success_rate=} {avg_return=}")
    #     print()
    #     exit()

    enable_wm = args['enable_wm_correction']
    if enable_wm:
        wm_required = ['evac_ckpt', 'evac_config', 'urdf_path', 'curobo_left_yml',
                        'curobo_right_yml', 'raw_data_dir', 'act_init_ckpt',
                        'correction_threshold', 'max_rollout_steps', 'correction_freq',
                        'correction_weight', 'orient_weight', 'gripper_penalty']
        missing = [k for k in wm_required if args.get(k) is None]
        if missing:
            raise ValueError(f"--enable_wm_correction requires these args: {missing}")
    raw_data_dir = args['raw_data_dir'] if enable_wm else None
    train_dataloader, val_dataloader, stats, _, max_action_len = load_data(dataset_dir, num_episodes, camera_names,
                                                                          batch_size_train, batch_size_val,
                                                                          raw_data_dir=raw_data_dir)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    config['enable_wm_correction'] = enable_wm
    if enable_wm:
        config['wm_args'] = args
        config['raw_data_dir'] = raw_data_dir
        config['norm_stats'] = stats
        config['dataset_dir'] = dataset_dir
        config['correction_cfg'] = {
            'threshold': args['correction_threshold'],
            'max_rollout_steps': args['max_rollout_steps'],
            'chunk_size': args['chunk_size'],
            'max_action_len': max_action_len,
            'correction_freq': args['correction_freq'],
            'correction_weight': args['correction_weight'],
            'orient_weight': args['orient_weight'],
            'gripper_penalty': args['gripper_penalty'],
        }
        config['act_init_ckpt'] = args.get('act_init_ckpt')
    config['debug_wm_correction'] = args.get('debug_wm_correction', False)
    train_bc(train_dataloader, val_dataloader, config)
    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # # save best checkpoint
    # ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def init_correction(args, device):
    import sapien
    from omegaconf import OmegaConf
    # EVAC internal modules (e.g. ddpm3d) do `from utils.general_utils import ...`
    # which needs evac/evac/ on sys.path AND `utils` in sys.modules to point to
    # evac's utils package (not ACT's utils.py which is already cached).
    _evac_evac = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evac', 'evac')
    if _evac_evac not in sys.path:
        sys.path.insert(0, _evac_evac)
    # Temporarily swap out ACT's utils module so EVAC can load its own utils package
    _act_utils = sys.modules.pop('utils', None)
    from evac.utils.general_utils import load_checkpoints, instantiate_from_config
    from util.fk_sapien import SapienFK
    _robotwin_root = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    sys.path.insert(0, _robotwin_root)
    sys.path.insert(0, os.path.join(_robotwin_root, 'envs', 'robot'))
    _prev_cwd = os.getcwd()
    os.chdir(_robotwin_root)
    from planner import CuroboPlanner
    os.chdir(_prev_cwd)

    evac_cfg = OmegaConf.load(args['evac_config'])
    evac_cfg.model.pretrained_checkpoint = args['evac_ckpt']
    evac_model = instantiate_from_config(evac_cfg.model)
    evac_model = load_checkpoints(evac_model, evac_cfg.model, ignore_mismatched_sizes=False)
    evac_model = evac_model.to(device)
    evac_model.eval()
    for p in evac_model.parameters():
        p.requires_grad = False

    fk = SapienFK(args['urdf_path'])

    root_pose = sapien.Pose([0, -0.65, 0], [0.707, 0, 0, 0.707])
    left_joints = [f'fl_joint{i}' for i in range(1, 7)]
    right_joints = [f'fr_joint{i}' for i in range(1, 7)]
    planner_l = CuroboPlanner(root_pose, left_joints, fk.jnames, yml_path=args['curobo_left_yml'])
    planner_r = CuroboPlanner(root_pose, right_joints, fk.jnames, yml_path=args['curobo_right_yml'])

    return {
        'evac_model': evac_model, 'evac_config': evac_cfg,
        'fk': fk, 'planner_left': planner_l, 'planner_right': planner_r,
    }


def load_raw_data(raw_data_dir, episode_id):
    import h5py, cv2 as _cv2
    path = os.path.join(raw_data_dir, f'episode{episode_id}.hdf5')
    with h5py.File(path, 'r') as f:
        # decode one frame to get native resolution (intrinsic is calibrated for this size)
        _frame0 = bytes(f['observation/head_camera/rgb'][0])
        _img0 = _cv2.imdecode(np.frombuffer(_frame0, np.uint8), _cv2.IMREAD_COLOR)
        h_native, w_native = _img0.shape[:2]
        return {
            'left_endpose': f['endpose/left_endpose'][()].astype(np.float32),
            'right_endpose': f['endpose/right_endpose'][()].astype(np.float32),
            'left_gripper': f['endpose/left_gripper'][()].astype(np.float32),
            'right_gripper': f['endpose/right_gripper'][()].astype(np.float32),
            'intrinsic_cv': f['observation/head_camera/intrinsic_cv'][0].astype(np.float32),
            'extrinsic_cv': f['observation/head_camera/extrinsic_cv'][0].astype(np.float32),
            'native_resolution': (h_native, w_native),  # intrinsic is calibrated for this
        }


def find_nearest_traj_point(fk_left_pos, fk_left_quat, fk_right_pos, fk_right_quat,
                            left_endpose, right_endpose, orient_weight=0.0,
                            curr_left_grip=None, curr_right_grip=None,
                            left_gripper_traj=None, right_gripper_traj=None,
                            gripper_penalty=0.0):
    """Find nearest trajectory point using position + orientation + gripper distance.
    Quaternions are in wxyz format. orient_weight scales the geodesic
    orientation distance (radians) relative to position distance (meters).
    gripper_penalty penalizes matching to points with different gripper state
    (binarized at 0.5 threshold)."""
    left_d = np.linalg.norm(left_endpose[:, :3] - fk_left_pos, axis=1)
    right_d = np.linalg.norm(right_endpose[:, :3] - fk_right_pos, axis=1)
    dists = (left_d + right_d) / 2
    if orient_weight > 0:
        # quaternion geodesic distance: 2 * arccos(|q1 · q2|)
        left_dot = np.clip(np.abs(np.sum(left_endpose[:, 3:7] * fk_left_quat, axis=1)), 0.0, 1.0)
        right_dot = np.clip(np.abs(np.sum(right_endpose[:, 3:7] * fk_right_quat, axis=1)), 0.0, 1.0)
        left_d_ori = 2.0 * np.arccos(left_dot)
        right_d_ori = 2.0 * np.arccos(right_dot)
        dists += orient_weight * (left_d_ori + right_d_ori) / 2
    if gripper_penalty > 0 and left_gripper_traj is not None:
        # binarize gripper: <=0.5 -> closed(0), >0.5 -> open(1)
        curr_lg = 0.0 if curr_left_grip <= 0.5 else 1.0
        curr_rg = 0.0 if curr_right_grip <= 0.5 else 1.0
        traj_lg = (left_gripper_traj > 0.5).astype(np.float32)
        traj_rg = (right_gripper_traj > 0.5).astype(np.float32)
        lg_mismatch = np.abs(traj_lg - curr_lg)
        rg_mismatch = np.abs(traj_rg - curr_rg)
        dists += gripper_penalty * (lg_mismatch + rg_mismatch) / 2
    t_star = np.argmin(dists)
    return t_star, dists[t_star]


def resample_trajectory(traj, target_len):
    n = len(traj)
    if n == target_len:
        return traj
    if n == 0:
        return np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len)
    result = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        result[i] = traj[lo] * (1 - frac) + traj[hi] * frac
    return result


@torch.no_grad()
def evac_predict(evac_model, evac_cfg, curr_image_act, fk_poses, grippers, raw_data, device,
                debug_dir=None, debug_prefix='', ddim_steps=27, return_offset=0):
    """EVAC prediction via DDIM sampling. ddim_steps controls denoising quality.
    return_offset: which predicted frame to return (0 = first, chunk-1 = last)."""
    import cv2
    import torchvision.transforms as tvt
    from evac.lvdm.data.get_actions import get_actions
    from evac.lvdm.data.statistics import StatisticInfo
    from evac.lvdm.data.domain_table import DomainTable

    chunk = evac_cfg.chunk
    n_previous = evac_cfg.n_previous
    n_total = n_previous + chunk
    sample_size = tuple(evac_cfg.data.params.train.params.sample_size)
    dtype = torch.bfloat16

    # --- image: (3,480,640) [0,1] -> (c,v,t,h,w) [-1,1] ---
    # ACT stores images in BGR (from cv2.imdecode). EVAC expects RGB.
    img = curr_image_act[0].clone()
    img = img[[2, 1, 0], :, :]   # BGR -> RGB
    img_resized = tvt.Resize(sample_size)(img)
    img_normed = tvt.Normalize([0.5]*3, [0.5]*3)(img_resized)
    cond_frames = img_normed.unsqueeze(1).repeat(1, n_previous, 1, 1)
    pred_frames = img_normed.unsqueeze(1).repeat(1, chunk, 1, 1)
    video = torch.cat([cond_frames, pred_frames], dim=1)  # (3, n_total, H, W)
    video = video.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, n_total, H, W)

    # --- actions ---
    n_frames = len(fk_poses)
    all_ends_p = np.zeros((n_frames, 2, 3), dtype=np.float32)
    all_ends_o = np.zeros((n_frames, 2, 4), dtype=np.float32)
    gripper_arr = np.zeros((n_frames, 2), dtype=np.float32)
    for i, ((lp, lq, rp, rq), (lg, rg)) in enumerate(zip(fk_poses, grippers)):
        all_ends_p[i, 0], all_ends_p[i, 1] = lp, rp
        # wxyz→xyzw, canonicalize so w>=0 (match HDF5 sign convention)
        lq_xyzw = np.array([lq[1], lq[2], lq[3], lq[0]])
        rq_xyzw = np.array([rq[1], rq[2], rq[3], rq[0]])
        if lq_xyzw[3] < 0: lq_xyzw = -lq_xyzw
        if rq_xyzw[3] < 0: rq_xyzw = -rq_xyzw
        all_ends_o[i, 0] = lq_xyzw
        all_ends_o[i, 1] = rq_xyzw
        gripper_arr[i] = [lg * 120.0, rg * 120.0]

    abs_act, delta_act = get_actions(
        gripper=gripper_arr, all_ends_p=all_ends_p, all_ends_o=all_ends_o,
        delta_act_sidx=n_previous)
    abs_act = torch.FloatTensor(abs_act)
    delta_act = torch.FloatTensor(delta_act)
    mv = torch.tensor(StatisticInfo['agibotworld']['mean']).unsqueeze(0)
    sv = torch.tensor(StatisticInfo['agibotworld']['std']).unsqueeze(0)
    delta_act[:, :6] = (delta_act[:, :6] - mv[:, :6]) / sv[:, :6]
    delta_act[:, 7:13] = (delta_act[:, 7:13] - mv[:, 6:]) / sv[:, 6:]

    # --- camera ---
    ext_cv = raw_data['extrinsic_cv']
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = ext_cv
    c2w = np.linalg.inv(w2c)
    # Single-view: (1, 4, 4) for get_traj (dim 0 = views)
    c2w_1v = torch.from_numpy(c2w).float().unsqueeze(0)   # (1, 4, 4)
    w2c_1v = torch.from_numpy(w2c).float().unsqueeze(0)   # (1, 4, 4)
    # Time-repeated: (n_total, 4, 4) for batch dict extrinsic
    c2w_t = c2w_1v.repeat(n_total, 1, 1)
    intrinsic_t = torch.from_numpy(raw_data['intrinsic_cv']).float().unsqueeze(0)

    # --- intrinsic scaling (from native resolution to EVAC sample_size) ---
    h_native, w_native = raw_data['native_resolution']
    h_scale = float(sample_size[0]) / float(h_native)
    w_scale = float(sample_size[1]) / float(w_native)
    intrinsic_scaled = intrinsic_t.clone()
    intrinsic_scaled[:, 0, 0] *= w_scale
    intrinsic_scaled[:, 0, 2] *= w_scale
    intrinsic_scaled[:, 1, 1] *= h_scale
    intrinsic_scaled[:, 1, 2] *= h_scale

    # --- trajectory maps (dim 0 of w2c/c2w = num_views, not time) ---
    traj_maps_raw = evac_model.get_traj(sample_size, abs_act[:n_total], w2c_1v, c2w_1v, intrinsic_scaled)
    traj_maps = rearrange(traj_maps_raw, 'c v t h w -> (v t) c h w')
    traj_maps = tvt.Normalize([0.5]*3, [0.5]*3)(traj_maps)
    traj_maps = rearrange(traj_maps, '(v t) c h w -> c v t h w', v=1)
    traj_maps = traj_maps.unsqueeze(0)  # (1, 3, 1, n_total, H, W)

    # --- debug: save EVAC inputs ---
    if debug_dir is not None:
        _dbg_evac = os.path.join(debug_dir, 'evac')
        os.makedirs(_dbg_evac, exist_ok=True)
        # input image (already RGB after BGR->RGB conversion above)
        _inp = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(_dbg_evac, f'{debug_prefix}input.png'), cv2.cvtColor(_inp, cv2.COLOR_RGB2BGR))
        # traj map (first predicted frame, index n_previous)
        _tm_idx = min(n_previous, traj_maps_raw.shape[2] - 1)
        _tm = traj_maps_raw[:, 0, _tm_idx]  # (3, H, W)
        _tm_np = (_tm.cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(_dbg_evac, f'{debug_prefix}traj_t{_tm_idx}.png'), cv2.cvtColor(_tm_np, cv2.COLOR_RGB2BGR))
        # traj map overlay on input
        _inp_r = cv2.resize(_inp, (sample_size[1], sample_size[0]))
        _overlay = cv2.addWeighted(_inp_r, 0.5, _tm_np, 0.5, 0)
        cv2.imwrite(os.path.join(_dbg_evac, f'{debug_prefix}traj_overlay.png'), cv2.cvtColor(_overlay, cv2.COLOR_RGB2BGR))
        # save camera params for inspection
        np.savez(os.path.join(_dbg_evac, f'{debug_prefix}camera.npz'),
                 intrinsic_raw=raw_data['intrinsic_cv'], intrinsic_scaled=intrinsic_scaled[0].numpy(),
                 extrinsic_cv=ext_cv, w2c=w2c, c2w=c2w,
                 native_resolution=np.array([h_native, w_native]),
                 sample_size=np.array(sample_size))
        # save abs_act summary
        np.savetxt(os.path.join(_dbg_evac, f'{debug_prefix}abs_act.csv'), abs_act[:n_total].numpy(), fmt='%.4f', delimiter=',')

    # --- batch dict ---
    fps = 30 * torch.ones((1,)).to(device)
    domain_id = torch.LongTensor([DomainTable['robotwin']]).to(device)
    c2w_batch = c2w_t.unsqueeze(0).unsqueeze(0)  # (1, 1, n_total, 4, 4)
    delta_act_batch = delta_act[:chunk].unsqueeze(0)

    batch = dict(
        video=video.to(dtype=dtype, device=device),
        traj=traj_maps.to(dtype=dtype, device=device),
        delta_action=delta_act_batch.to(dtype=dtype, device=device),
        domain_id=domain_id,
        intrinsic=intrinsic_scaled.to(device=device),
        extrinsic=c2w_batch.to(device=device),
        caption=[""],
        cond_id=torch.tensor([-n_previous - chunk], dtype=torch.int64).to(device),
        fps=fps.to(device),
    )

    # --- get_batch_input: encode + build conditions ---
    with torch.cuda.amp.autocast(dtype=dtype):
        out = evac_model.get_batch_input(
            batch, random_uncond=False,
            return_first_stage_outputs=False,
            return_original_cond=True,
            return_fs=True, return_did=True,
            return_traj=False, return_img_emb=True)
        z, cond, _, fs, did, img_emb = out

        for k in range(len(cond.get('c_crossattn', []))):
            cond['c_crossattn'][k] = cond['c_crossattn'][k].to(dtype=dtype)
        for k in range(len(cond.get('c_concat', []))):
            cond['c_concat'][k] = cond['c_concat'][k].to(dtype=dtype)

        # --- DDIM sampling ---
        _prev_ddim_num_chunk = evac_model.ddim_num_chunk
        _prev_rand_cond = evac_model.rand_cond_frame
        evac_model.ddim_num_chunk = 1
        evac_model.rand_cond_frame = False
        full_z, _ = evac_model.sample_log(
            cond=cond, batch_size=z.shape[0], ddim=True,
            ddim_steps=ddim_steps, causal=True, eta=1.0,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            x0=z.to(dtype), chunk=chunk,
            cat_mask=evac_model.use_cat_mask,
            sparse=evac_model.sparse_memory,
            ddim_dtype=torch.float16,
            fs=fs.long(), domain_id=did.long(),
            timestep_spacing='uniform_trailing',
            guidance_rescale=0.7,
            return_intermediates=False,
        )
        evac_model.ddim_num_chunk = _prev_ddim_num_chunk
        evac_model.rand_cond_frame = _prev_rand_cond

        x_decoded = evac_model.decode_first_stage(full_z.to(z.device))

    # predicted frame at requested offset (0 = first predicted frame)
    frame_idx = n_previous + return_offset
    pred_frame = x_decoded[0, :, frame_idx]  # (3, H, W) in [-1, 1], RGB
    pred_frame = ((pred_frame.float() + 1) / 2).clamp(0, 1)
    pred_np = (pred_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pred_np = cv2.resize(pred_np, (640, 480))

    # --- debug: save EVAC output ---
    if debug_dir is not None:
        _dbg_evac = os.path.join(debug_dir, 'evac')
        cv2.imwrite(os.path.join(_dbg_evac, f'{debug_prefix}pred.png'), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
        # save all decoded frames as grid
        _nf = x_decoded.shape[2]
        _frames = []
        for _fi in range(_nf):
            _f = x_decoded[0, :, _fi]
            _f = ((_f.float() + 1) / 2).clamp(0, 1)
            _frames.append((_f.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        _grid = np.concatenate(_frames, axis=1)  # horizontal concat
        cv2.imwrite(os.path.join(_dbg_evac, f'{debug_prefix}all_frames.png'), cv2.cvtColor(_grid, cv2.COLOR_RGB2BGR))

    # convert back to BGR to match ACT image convention
    pred_np_bgr = pred_np[:, :, ::-1].copy()
    return torch.from_numpy(pred_np_bgr).float().permute(2, 0, 1) / 255.0


@torch.no_grad()
def evac_inference(evac_model, evac_cfg, curr_image, fk_poses, grippers, raw_data, device,
                   save_dir=None, ddim_steps=27):
    """EVAC prediction via model.inference() — matches infer_all.py exactly.

    Args:
        curr_image: (3, H, W) BGR [0,1] tensor (current observation)
        fk_poses: list of (lp, lq_wxyz, rp, rq_wxyz).
                  First entry = current state, rest = action outcomes. Total N entries.
        grippers: list of (lg, rg), same length as fk_poses.
        raw_data: dict with extrinsic_cv, intrinsic_cv, native_resolution
        save_dir: if set, save frames + traj video there
    Returns: last predicted frame as (3, H, W) BGR [0,1] tensor
    """
    import cv2, math, tempfile, shutil
    from evac.lvdm.data.get_actions import get_actions
    from evac.lvdm.data.statistics import StatisticInfo
    import torchvision.transforms as tvt

    chunk = evac_cfg.chunk
    n_prev = evac_cfg.n_previous
    N = len(fk_poses)  # 1 (init state) + N_actions

    # --- Image: BGR→RGB [0,1], resize to native resolution, repeat n_prev ---
    # inference() scales intrinsic by sample_size / image_size, so image must
    # be at the same resolution as the intrinsic calibration (native_resolution).
    h_native, w_native = raw_data['native_resolution']
    img_rgb = curr_image[[2, 1, 0]]  # (3, 480, 640) → RGB
    img_rgb = tvt.Resize((h_native, w_native))(img_rgb)  # → native res
    memories = img_rgb.unsqueeze(1).repeat(1, n_prev, 1, 1)  # (3, n_prev, h, w)

    # --- Actions: same format as infer_all.py h5 variant ---
    all_ends_p = np.zeros((N, 2, 3), dtype=np.float32)
    all_ends_o = np.zeros((N, 2, 4), dtype=np.float32)
    gripper_arr = np.zeros((N, 2), dtype=np.float32)
    for i, ((lp, lq, rp, rq), (lg, rg)) in enumerate(zip(fk_poses, grippers)):
        all_ends_p[i, 0], all_ends_p[i, 1] = lp, rp
        # wxyz→xyzw, canonicalize so w>=0 (match HDF5 sign convention)
        lq_xyzw = np.array([lq[1], lq[2], lq[3], lq[0]])
        rq_xyzw = np.array([rq[1], rq[2], rq[3], rq[0]])
        if lq_xyzw[3] < 0: lq_xyzw = -lq_xyzw
        if rq_xyzw[3] < 0: rq_xyzw = -rq_xyzw
        all_ends_o[i, 0] = lq_xyzw
        all_ends_o[i, 1] = rq_xyzw
        gripper_arr[i] = [lg * 120.0, rg * 120.0]

    slices = [0] * (n_prev - 1) + list(range(N))
    action, delta_action = get_actions(
        gripper=gripper_arr, all_ends_p=all_ends_p, all_ends_o=all_ends_o,
        slices=slices, delta_act_sidx=n_prev)
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    mv = torch.tensor(StatisticInfo['agibotworld']['mean']).unsqueeze(0)
    sv = torch.tensor(StatisticInfo['agibotworld']['std']).unsqueeze(0)
    delta_action[:, :6] = (delta_action[:, :6] - mv[:, :6]) / sv[:, :6]
    delta_action[:, 7:13] = (delta_action[:, 7:13] - mv[:, 6:]) / sv[:, 6:]

    # --- Camera: same format as infer_all.py ---
    ext_cv = raw_data['extrinsic_cv']
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = ext_cv
    c2w = np.linalg.inv(w2c)
    n_act = action.shape[0]
    c2w_t = torch.from_numpy(c2w).float().unsqueeze(0).repeat(n_act, 1, 1)
    w2c_t = torch.from_numpy(w2c).float().unsqueeze(0).repeat(n_act, 1, 1)
    intrinsic = torch.from_numpy(raw_data['intrinsic_cv']).float()

    # --- Run inference ---
    n_valid = N - 1  # predicted frames = number of action steps
    num_chunk = int(math.ceil(float(n_valid) / chunk))
    tmp_dir = None
    if save_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix='evac_')
        target_dir = tmp_dir
    else:
        target_dir = save_dir
    os.makedirs(target_dir, exist_ok=True)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        frames, traj_frames = evac_model.inference(
            evac_cfg, memories, action, delta_action,
            c2w_t, w2c_t, intrinsic,
            target_dir, num_chunk,
            chunk=chunk, n_previous=n_prev, n_valid=n_valid,
            unconditional_guidance_scale=1.0,
            guidance_rescale=0.7,
            ddim_steps=ddim_steps,
            dataset_name="agibotworld",
            saving_video=(save_dir is not None),
            saving_fps=30,
            video_dir=target_dir,
        )
        torch.cuda.empty_cache()

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # frames: (n_valid, H, W, 3) RGB uint8 at sample_size resolution
    last_rgb = cv2.resize(frames[-1], (640, 480))
    last_bgr = last_rgb[:, :, ::-1].copy()
    return torch.from_numpy(last_bgr).float().permute(2, 0, 1) / 255.0


def correction_step(policy_unwrapped, image_data_s, qpos_data_s, raw_data,
                    norm_stats, modules, cfg, device, debug_dir=None):
    fk = modules['fk']
    evac_model = modules['evac_model']
    evac_cfg = modules['evac_config']
    planner_l = modules['planner_left']
    planner_r = modules['planner_right']

    left_ep = raw_data['left_endpose']
    right_ep = raw_data['right_endpose']
    threshold = cfg['threshold']
    max_steps = cfg['max_rollout_steps']
    chunk_size = cfg['chunk_size']
    max_action_len = cfg['max_action_len']
    chunk_evac = evac_cfg.chunk
    n_previous = evac_cfg.n_previous
    orient_weight = cfg.get('orient_weight', 0.0)
    gripper_penalty = cfg.get('gripper_penalty', 0.0)

    left_grip_traj = raw_data['left_gripper']
    right_grip_traj = raw_data['right_gripper']

    qpos_raw = qpos_data_s.cpu().numpy() * norm_stats['qpos_std'] + norm_stats['qpos_mean']
    curr_image = image_data_s.clone()
    curr_qpos_raw = qpos_raw.copy()

    left_q = right_q = None
    left_grip = right_grip = 0.0
    t_star = 0
    min_dist = 0.0
    _dbg_rollout = []  # collect per-step debug info

    for step in range(max_steps):
        left_q, right_q = curr_qpos_raw[0:6], curr_qpos_raw[7:13]
        left_grip, right_grip = curr_qpos_raw[6], curr_qpos_raw[13]
        fk_r = fk.forward(left_q, right_q)
        lp, lq = fk_r['left']
        rp, rq = fk_r['right']
        t_star, min_dist = find_nearest_traj_point(
            lp, lq, rp, rq, left_ep, right_ep, orient_weight,
            curr_left_grip=left_grip, curr_right_grip=right_grip,
            left_gripper_traj=left_grip_traj, right_gripper_traj=right_grip_traj,
            gripper_penalty=gripper_penalty)
        if min_dist >= threshold:
            break
        if t_star >= len(left_ep) - 2:
            return None

        qn = (curr_qpos_raw - norm_stats['qpos_mean']) / norm_stats['qpos_std']
        qt = torch.from_numpy(qn).float().unsqueeze(0).to(device)
        it = curr_image.unsqueeze(0).to(device)
        with torch.no_grad():
            act_chunk = policy_unwrapped(qt, it)
        act_np = act_chunk.squeeze(0).cpu().numpy()

        n_total = n_previous + chunk_evac
        fk_poses, grip_list = [], []
        for _ in range(n_previous):
            fk_poses.append((lp.copy(), lq.copy(), rp.copy(), rq.copy()))
            grip_list.append((left_grip, right_grip))
        for ai in range(min(chunk_evac, len(act_np))):
            ar = act_np[ai] * norm_stats['action_std'] + norm_stats['action_mean']
            fr = fk.forward(ar[0:6], ar[7:13])
            fk_poses.append((fr['left'][0].copy(), fr['left'][1].copy(),
                             fr['right'][0].copy(), fr['right'][1].copy()))
            grip_list.append((ar[6], ar[13]))
        while len(fk_poses) < n_total:
            fk_poses.append(fk_poses[-1])
            grip_list.append(grip_list[-1])

        _evac_dbg_prefix = f'rollout_s{step}_' if debug_dir else ''
        pred = evac_predict(evac_model, evac_cfg, curr_image, fk_poses, grip_list, raw_data, device,
                            debug_dir=debug_dir, debug_prefix=_evac_dbg_prefix)
        new_img = curr_image.clone()
        new_img[0] = pred
        curr_image = new_img
        a0 = act_np[0] * norm_stats['action_std'] + norm_stats['action_mean']
        curr_qpos_raw = a0

        # collect debug info for this rollout step
        if debug_dir is not None:
            _dbg_rollout.append({
                'step': step, 't_star': int(t_star), 'min_dist': float(min_dist),
                'fk_left_pos': lp.tolist(), 'fk_right_pos': rp.tolist(),
                'left_grip': float(left_grip), 'right_grip': float(right_grip),
                'act_chunk_raw_first': (act_np[0] * norm_stats['action_std'] + norm_stats['action_mean']).tolist(),
            })

    if min_dist < threshold:
        # debug: save rollout info even on skip
        if debug_dir is not None:
            import json
            _dbg_corr = os.path.join(debug_dir, 'correction')
            os.makedirs(_dbg_corr, exist_ok=True)
            with open(os.path.join(_dbg_corr, 'skipped.json'), 'w') as _f:
                json.dump({'reason': 'below_threshold', 'min_dist': float(min_dist),
                           'threshold': threshold, 'rollout': _dbg_rollout}, _f, indent=2)
        return None

    import sapien
    target_lp = sapien.Pose(left_ep[t_star, :3], left_ep[t_star, 3:7])
    target_rp = sapien.Pose(right_ep[t_star, :3], right_ep[t_star, 3:7])
    qpos_full = np.zeros(len(fk.jnames), dtype=np.float32)
    qpos_full[fk.fl_idx] = left_q
    qpos_full[fk.fr_idx] = right_q

    try:
        res_l = planner_l.plan_path(qpos_full, target_lp, arms_tag='left')
        res_r = planner_r.plan_path(qpos_full, target_rp, arms_tag='right')
    except Exception:
        return None
    if res_l.get('status') != 'Success' or res_r.get('status') != 'Success':
        return None

    lt = resample_trajectory(res_l['position'], chunk_size)
    rt = resample_trajectory(res_r['position'], chunk_size)
    tl_grip = raw_data['left_gripper'][t_star]
    tr_grip = raw_data['right_gripper'][t_star]

    corr = np.zeros((chunk_size, 14), dtype=np.float32)
    corr[:, 0:6] = lt
    corr[:, 7:13] = rt
    # Gripper: step transition instead of linear interpolation to avoid dropping objects
    # Binarize: <=0.5 -> closed(0), >0.5 -> open(1)
    lg_curr = 0.0 if left_grip <= 0.5 else 1.0
    lg_tgt = 0.0 if tl_grip <= 0.5 else 1.0
    rg_curr = 0.0 if right_grip <= 0.5 else 1.0
    rg_tgt = 0.0 if tr_grip <= 0.5 else 1.0
    if lg_curr == lg_tgt:
        corr[:, 6] = lg_tgt
    else:
        # keep current state for 80% of chunk, snap to target for last 20%
        switch_idx = int(chunk_size * 0.8)
        corr[:switch_idx, 6] = lg_curr
        corr[switch_idx:, 6] = lg_tgt
    if rg_curr == rg_tgt:
        corr[:, 13] = rg_tgt
    else:
        switch_idx = int(chunk_size * 0.8)
        corr[:switch_idx, 13] = rg_curr
        corr[switch_idx:, 13] = rg_tgt
    corr_norm = (corr - norm_stats['action_mean']) / norm_stats['action_std']

    padded = np.zeros((max_action_len, 14), dtype=np.float32)
    padded[:chunk_size] = corr_norm
    is_pad = np.ones(max_action_len, dtype=bool)
    is_pad[:chunk_size] = False

    qn = (curr_qpos_raw - norm_stats['qpos_mean']) / norm_stats['qpos_std']

    # --- debug: save full correction info ---
    if debug_dir is not None:
        import json, cv2
        _dbg_corr = os.path.join(debug_dir, 'correction')
        os.makedirs(_dbg_corr, exist_ok=True)
        # save correction trajectory
        np.savetxt(os.path.join(_dbg_corr, 'corr_action_raw.csv'), corr, fmt='%.6f', delimiter=',')
        # save corrected image (curr_image is BGR, cv2 expects BGR)
        _cimg = (curr_image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(_dbg_corr, 'corrected_image.png'), _cimg)
        # save original image (image_data_s is BGR, cv2 expects BGR)
        _oimg = (image_data_s[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(_dbg_corr, 'original_image.png'), _oimg)
        # save planner results summary
        _plan_info = {
            'rollout': _dbg_rollout,
            't_star': int(t_star), 'min_dist': float(min_dist),
            'threshold': threshold,
            'target_left_pose': left_ep[t_star].tolist(),
            'target_right_pose': right_ep[t_star].tolist(),
            'planner_left_status': res_l.get('status'),
            'planner_right_status': res_r.get('status'),
            'planner_left_len': len(res_l.get('position', [])),
            'planner_right_len': len(res_r.get('position', [])),
            'gripper_left': [float(lg_curr), float(lg_tgt)],
            'gripper_right': [float(rg_curr), float(rg_tgt)],
        }
        with open(os.path.join(_dbg_corr, 'correction_info.json'), 'w') as _f:
            json.dump(_plan_info, _f, indent=2)
        # save correction trajectory plot
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 6))
            labels_l = ['lj1','lj2','lj3','lj4','lj5','lj6','lg']
            labels_r = ['rj1','rj2','rj3','rj4','rj5','rj6','rg']
            for j in range(7):
                axes[0].plot(corr[:, j], label=labels_l[j])
                axes[1].plot(corr[:, 7+j], label=labels_r[j])
            axes[0].set_title('Left arm correction trajectory')
            axes[0].legend(fontsize=7, ncol=4)
            axes[1].set_title('Right arm correction trajectory')
            axes[1].legend(fontsize=7, ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(_dbg_corr, 'corr_trajectory.png'), dpi=100)
            plt.close(fig)
        except Exception:
            pass

    return (
        curr_image.to(device),
        torch.from_numpy(qn.astype(np.float32)).to(device),
        torch.from_numpy(padded).float().to(device),
        torch.from_numpy(is_pad).bool().to(device),
    )


# def eval_bc(config, ckpt_name, save_episode=True):
#     set_seed(1000)
#     ckpt_dir = config["ckpt_dir"]
#     state_dim = config["state_dim"]
#     real_robot = config["real_robot"]
#     policy_class = config["policy_class"]
#     onscreen_render = config["onscreen_render"]
#     policy_config = config["policy_config"]
#     camera_names = config["camera_names"]
#     max_timesteps = config["episode_len"]
#     task_name = config["task_name"]
#     temporal_agg = config["temporal_agg"]
#     onscreen_cam = "angle"
#
#     # load policy and stats
#     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#     policy = make_policy(policy_class, policy_config)
#     loading_status = policy.load_state_dict(torch.load(ckpt_path))
#     print(loading_status)
#     policy.cuda()
#     policy.eval()
#     print(f"Loaded: {ckpt_path}")
#     stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
#     with open(stats_path, "rb") as f:
#         stats = pickle.load(f)
#
#     pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
#     post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
#
#     # load environment
#     if real_robot:
#         from aloha_scripts.robot_utils import move_grippers  # requires aloha
#         from aloha_scripts.real_env import make_real_env  # requires aloha
#
#         env = make_real_env(init_node=True)
#         env_max_reward = 0
#     else:
#         from sim_env import make_sim_env
#
#         env = make_sim_env(task_name)
#         env_max_reward = env.task.max_reward
#
#     query_frequency = policy_config["num_queries"]
#     if temporal_agg:
#         query_frequency = 1
#         num_queries = policy_config["num_queries"]
#
#     max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
#
#     num_rollouts = 50
#     episode_returns = []
#     highest_rewards = []
#     for rollout_id in range(num_rollouts):
#         rollout_id += 0
#         ### set task
#         if "sim_transfer_cube" in task_name:
#             BOX_POSE[0] = sample_box_pose()  # used in sim reset
#         elif "sim_insertion" in task_name:
#             BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
#
#         ts = env.reset()
#
#         ### onscreen render
#         if onscreen_render:
#             ax = plt.subplot()
#             plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
#             plt.ion()
#
#         ### evaluation loop
#         if temporal_agg:
#             all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()
#
#         qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
#         image_list = []  # for visualization
#         qpos_list = []
#         target_qpos_list = []
#         rewards = []
#         with torch.inference_mode():
#             for t in range(max_timesteps):
#                 ### update onscreen render and wait for DT
#                 if onscreen_render:
#                     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
#                     plt_img.set_data(image)
#                     plt.pause(DT)
#
#                 ### process previous timestep to get qpos and image_list
#                 obs = ts.observation
#                 if "images" in obs:
#                     image_list.append(obs["images"])
#                 else:
#                     image_list.append({"main": obs["image"]})
#                 qpos_numpy = np.array(obs["qpos"])
#                 qpos = pre_process(qpos_numpy)
#                 qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
#                 qpos_history[:, t] = qpos
#                 curr_image = get_image(ts, camera_names)
#
#                 ### query policy
#                 if config["policy_class"] == "ACT":
#                     if t % query_frequency == 0:
#                         all_actions = policy(qpos, curr_image)
#                     if temporal_agg:
#                         all_time_actions[[t], t:t + num_queries] = all_actions
#                         actions_for_curr_step = all_time_actions[:, t]
#                         actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
#                         actions_for_curr_step = actions_for_curr_step[actions_populated]
#                         k = 0.01
#                         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
#                         exp_weights = exp_weights / exp_weights.sum()
#                         exp_weights = (torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1))
#                         raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
#                     else:
#                         raw_action = all_actions[:, t % query_frequency]
#                 elif config["policy_class"] == "CNNMLP":
#                     raw_action = policy(qpos, curr_image)
#                 else:
#                     raise NotImplementedError
#
#                 ### post-process actions
#                 raw_action = raw_action.squeeze(0).cpu().numpy()
#                 action = post_process(raw_action)
#                 target_qpos = action
#
#                 ### step the environment
#                 ts = env.step(target_qpos)
#
#                 ### for visualization
#                 qpos_list.append(qpos_numpy)
#                 target_qpos_list.append(target_qpos)
#                 rewards.append(ts.reward)
#
#             plt.close()
#         if real_robot:
#             move_grippers(
#                 [env.puppet_bot_left, env.puppet_bot_right],
#                 [PUPPET_GRIPPER_JOINT_OPEN] * 2,
#                 move_time=0.5,
#             )  # open
#             pass
#
#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards != None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         print(
#             f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
#         )
#
#         if save_episode:
#             save_videos(
#                 image_list,
#                 DT,
#                 video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
#             )
#
#     success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#     avg_return = np.mean(episode_returns)
#     summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
#     for r in range(env_max_reward + 1):
#         more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#         more_or_equal_r_rate = more_or_equal_r / num_rollouts
#         summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"
#
#     # save success rate to txt
#     result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
#     with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
#         f.write(summary_str)
#         f.write(repr(episode_returns))
#         f.write("\n\n")
#         f.write(repr(highest_rewards))
#
#     return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data[0], data[1], data[2], data[3]
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    # Accelerate: prepare model, optimizer, dataloader
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Set seed after Accelerator init so each process gets proper seed offset
    set_seed(seed)

    policy = make_policy(policy_class, policy_config)

    enable_wm = config['enable_wm_correction']
    if enable_wm and config.get('act_init_ckpt'):
        ckpt = torch.load(config['act_init_ckpt'], map_location='cpu')
        policy.load_state_dict(ckpt)
        print(f"Loaded ACT init weights from {config['act_init_ckpt']}")

    optimizer = make_optimizer(policy_class, policy)
    policy, optimizer, train_dataloader = accelerator.prepare(
        policy, optimizer, train_dataloader
    )

    correction_modules = None
    debug_wm = config.get('debug_wm_correction', False)
    debug_wm_dir = os.path.join(ckpt_dir, 'debug_wm') if debug_wm else None
    if debug_wm_dir and accelerator.is_main_process:
        os.makedirs(debug_wm_dir, exist_ok=True)
    # correction statistics tracker
    _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                   'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': []}
    if enable_wm:
        correction_modules = init_correction(config['wm_args'], accelerator.device)
        norm_stats = config['norm_stats']
        correction_cfg = config['correction_cfg']
        raw_data_dir = config['raw_data_dir']

    train_history = []
    validation_history = []
    # min_val_loss = np.inf
    # best_ckpt_info = None

    # TensorBoard (only on main process)
    writer = None
    if accelerator.is_main_process:
        tb_log_dir = os.path.join(ckpt_dir, 'tb_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f'TensorBoard log dir: {tb_log_dir}')
    global_step = 0

    for epoch in tqdm(range(num_epochs), disable=not accelerator.is_main_process):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}")
        # # validation
        # with torch.inference_mode():
        #     policy.eval()
        #     epoch_dicts = []
        #     for batch_idx, data in enumerate(val_dataloader):
        #         forward_dict = forward_pass(data, policy)
        #         epoch_dicts.append(forward_dict)
        #     epoch_summary = compute_dict_mean(epoch_dicts)
        #     validation_history.append(epoch_summary)
        #
        #     epoch_val_loss = epoch_summary["loss"]
        #     if epoch_val_loss < min_val_loss:
        #         min_val_loss = epoch_val_loss
        #         best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        # print(f"Val loss:   {epoch_val_loss:.5f}")
        # summary_string = ""
        # for k, v in epoch_summary.items():
        #     summary_string += f"{k}: {v.item():.3f} "

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]

            # World-model correction
            if enable_wm and correction_modules is not None and global_step % correction_cfg['correction_freq'] == 0:
                try:
                    bs = data[0].shape[0]
                    corr_images, corr_qpos, corr_actions, corr_pads = [], [], [], []
                    corr_mask = []
                    # per-step debug dir (only on main process, only every 100 steps)
                    _step_dbg = None
                    if debug_wm_dir and accelerator.is_main_process:
                        _step_dbg = os.path.join(debug_wm_dir, f'step_{global_step:06d}')
                        os.makedirs(_step_dbg, exist_ok=True)
                    for bi in range(bs):
                        ep_id = data[4][bi].item()
                        raw = load_raw_data(raw_data_dir, ep_id)
                        _bi_dbg = os.path.join(_step_dbg, f'bi{bi}') if _step_dbg and bi == 0 else None
                        if _bi_dbg:
                            os.makedirs(_bi_dbg, exist_ok=True)
                        corr = correction_step(
                            accelerator.unwrap_model(policy),
                            data[0][bi], data[1][bi], raw,
                            norm_stats, correction_modules, correction_cfg,
                            accelerator.device, debug_dir=_bi_dbg)
                        _corr_stats['n_triggered'] += 1
                        if corr is not None:
                            ci, cq, ca, cp = corr
                            corr_images.append(ci)
                            corr_qpos.append(cq)
                            corr_actions.append(ca)
                            corr_pads.append(cp)
                            corr_mask.append(1.0)
                            _corr_stats['n_success'] += 1
                        else:
                            _corr_stats['n_skipped'] += 1
                            # dummy data to keep batch size consistent across ranks
                            corr_images.append(data[0][bi].to(accelerator.device))
                            corr_qpos.append(data[1][bi].to(accelerator.device))
                            corr_actions.append(data[2][bi].to(accelerator.device))
                            corr_pads.append(data[3][bi].to(accelerator.device))
                            corr_mask.append(0.0)
                    # always call policy with bs samples so DDP stays in sync
                    ci_b = torch.stack(corr_images, dim=0)
                    cq_b = torch.stack(corr_qpos, dim=0)
                    ca_b = torch.stack(corr_actions, dim=0)
                    cp_b = torch.stack(corr_pads, dim=0)
                    corr_dict = policy(cq_b, ci_b, ca_b, cp_b)
                    n_corr = sum(corr_mask)
                    if n_corr > 0:
                        # mask out dummy samples: reweight loss
                        mask_t = torch.tensor(corr_mask, device=accelerator.device)
                        # corr_dict['loss'] is a mean over the batch;
                        # we need per-sample then masked mean, but ACT returns scalar loss,
                        # so approximate: scale by (n_valid / bs) to correct for dummy dilution
                        corr_loss = corr_dict['loss'] * (bs / n_corr) * (n_corr / (bs + n_corr))
                        cw = correction_cfg['correction_weight']
                        loss = bs / (bs + n_corr) * loss + cw * corr_loss
                    else:
                        # no valid corrections, but forward was called so DDP is happy
                        loss = loss + 0.0 * corr_dict['loss']
                    if accelerator.is_main_process and writer is not None:
                        writer.add_scalar('train/correction_loss', corr_dict['loss'].item(), global_step)
                        writer.add_scalar('train/n_correction_samples', n_corr, global_step)
                except Exception as exc:
                    _corr_stats['n_error'] += 1
                    if accelerator.is_main_process:
                        import traceback
                        print(f'[WM correction] step {global_step} error: {exc}')
                        traceback.print_exc()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if accelerator.is_main_process and writer is not None:
                writer.add_scalar('train/loss_step', forward_dict['loss'].item(), global_step)
            global_step += 1
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary["loss"]
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        # TensorBoard: log epoch-level metrics
        if accelerator.is_main_process and writer is not None:
            for k, v in epoch_summary.items():
                writer.add_scalar(f'train/{k}_epoch', v.item(), epoch)
            # log correction statistics
            if enable_wm:
                writer.add_scalar('correction/n_triggered', _corr_stats['n_triggered'], epoch)
                writer.add_scalar('correction/n_success', _corr_stats['n_success'], epoch)
                writer.add_scalar('correction/n_skipped', _corr_stats['n_skipped'], epoch)
                writer.add_scalar('correction/n_error', _corr_stats['n_error'], epoch)
                _total = _corr_stats['n_triggered'] or 1
                writer.add_scalar('correction/success_rate', _corr_stats['n_success'] / _total, epoch)
                print(f'  [Correction] triggered={_corr_stats["n_triggered"]} '
                      f'success={_corr_stats["n_success"]} '
                      f'skipped={_corr_stats["n_skipped"]} '
                      f'error={_corr_stats["n_error"]} '
                      f'rate={_corr_stats["n_success"]/_total:.2%}')
                # reset per-epoch
                _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                               'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': []}

        if (epoch + 1) % config['save_freq'] == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            unwrapped_policy = accelerator.unwrap_model(policy)
            torch.save(unwrapped_policy.state_dict(), ckpt_path)
            # plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    if accelerator.is_main_process:
        if writer is not None:
            writer.close()

        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        unwrapped_policy = accelerator.unwrap_model(policy)
        torch.save(unwrapped_policy.state_dict(), ckpt_path)

    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")
    print(f"Training finished: Seed {seed}")

    # # save training curves
    # plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)


# def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
#     # save training curves
#     for key in train_history[0]:
#         plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
#         plt.figure()
#         train_values = [summary[key].item() for summary in train_history]
#         val_values = [summary[key].item() for summary in validation_history]
#         plt.plot(
#             np.linspace(0, num_epochs - 1, len(train_history)),
#             train_values,
#             label="train",
#         )
#         plt.plot(
#             np.linspace(0, num_epochs - 1, len(validation_history)),
#             val_values,
#             label="validation",
#         )
#         # plt.ylim([-0.1, 1])
#         plt.tight_layout()
#         plt.legend()
#         plt.title(key)
#         plt.savefig(plot_path)
#     print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--eval", action="store_true")
    # parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--state_dim", action="store", type=int, help="state dim", required=True)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")

    # World-model correction arguments
    parser.add_argument("--enable_wm_correction", action="store_true",
                        help="Enable world-model based correction training")
    parser.add_argument("--evac_ckpt", type=str,
                        help="Path to EVAC model checkpoint")
    parser.add_argument("--evac_config", type=str,
                        help="Path to EVAC train_config.yaml")
    parser.add_argument("--urdf_path", type=str,
                        help="Path to robot URDF file")
    parser.add_argument("--curobo_left_yml", type=str,
                        help="Path to curobo left arm config yml")
    parser.add_argument("--curobo_right_yml", type=str,
                        help="Path to curobo right arm config yml")
    parser.add_argument("--raw_data_dir", type=str,
                        help="Path to raw episode data directory")
    parser.add_argument("--act_init_ckpt", type=str,
                        help="Path to ACT initial checkpoint")
    parser.add_argument("--correction_threshold", type=float,
                        help="Distance threshold to trigger correction")
    parser.add_argument("--max_rollout_steps", type=int,
                        help="Max rollout steps for correction")
    parser.add_argument("--correction_freq", type=int,
                        help="Apply correction every N steps")
    parser.add_argument("--correction_weight", type=float,
                        help="Weight for correction loss (auto-divided by batch_size)")
    parser.add_argument("--orient_weight", type=float,
                        help="Weight for orientation distance in nearest-point matching (0=position only)")
    parser.add_argument("--gripper_penalty", type=float,
                        help="Penalty for gripper state mismatch in nearest-point matching (0=ignore gripper)")
    parser.add_argument("--debug_wm_correction", action="store_true",
                        help="Enable debug visualization for WM correction (saves images/stats to ckpt_dir/debug_wm)")

    main(vars(parser.parse_args()))