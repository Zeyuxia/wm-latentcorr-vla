from __future__ import annotations

import os
import sys
import time
import pickle
import json
import traceback

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import load_data
from utils import compute_dict_mean, detach_dict
from act_policy import ACTPolicy, CNNMLPPolicy
from imitate_episodes_pkg.correction import (
    _export_correction_sample_as_episode,
    _init_export_episode_id,
    _save_loss_batch_projection,
    correction_step,
    load_raw_data,
)
from imitate_episodes_pkg.utils import (
    build_evac_infer_kwargs,
    phase_id_to_key,
    error_mode_id_to_key,
    active_arm_pattern_id_to_key,
    set_failure_param_bins,
)

def main(args):
    set_seed(int(args["seed"]))
    _rank = int(os.environ.get("RANK", -1))
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[main][rank={_rank} local_rank={_local_rank}] start")
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args.get("onscreen_render", False)
    task_name = args["task_name"]
    multi_task_names_raw = str(args.get("multi_task_names", "")).strip()
    multi_task_weights_raw = str(args.get("multi_task_weights", "")).strip()
    multi_task_names = []
    if multi_task_names_raw:
        multi_task_names = [x.strip() for x in multi_task_names_raw.split(",") if x.strip()]
    use_multi_task = len(multi_task_names) > 0
    batch_size_train = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    def _resolve_task_cfg(_task_name):
        _is_sim = _task_name[:4] == "sim-"
        if _is_sim:
            from constants import SIM_TASK_CONFIGS
            _task_cfg = SIM_TASK_CONFIGS[_task_name]
        else:
            from aloha_scripts.constants import TASK_CONFIGS
            _task_cfg = TASK_CONFIGS[_task_name]
        return _task_cfg, _is_sim

    dataset_dirs = None
    num_episodes_list = None
    multi_task_weights = None
    if use_multi_task:
        resolved = [_resolve_task_cfg(nm) for nm in multi_task_names]
        task_cfgs = [x[0] for x in resolved]
        is_sim = all(x[1] for x in resolved)
        camera_names = task_cfgs[0]["camera_names"]
        for i, tc in enumerate(task_cfgs):
            if tc["camera_names"] != camera_names:
                raise ValueError(
                    f"multi_task camera_names mismatch at task={multi_task_names[i]}: "
                    f"{tc['camera_names']} vs {camera_names}"
                )
        dataset_dirs = [tc["dataset_dir"] for tc in task_cfgs]
        num_episodes_list = [int(tc["num_episodes"]) for tc in task_cfgs]
        episode_len = max(int(tc["episode_len"]) for tc in task_cfgs)
        dataset_dir = dataset_dirs[0]
        num_episodes = num_episodes_list[0]
        if multi_task_weights_raw:
            multi_task_weights = [float(x.strip()) for x in multi_task_weights_raw.split(",") if x.strip()]
            if len(multi_task_weights) != len(multi_task_names):
                raise ValueError(
                    f"multi_task_weights length ({len(multi_task_weights)}) "
                    f"must match multi_task_names length ({len(multi_task_names)})"
                )
    else:
        task_config, is_sim = _resolve_task_cfg(task_name)
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
        "task_name": ((",".join(multi_task_names)) if use_multi_task else task_name),
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq'],
    }

    enable_wm = args['enable_wm_correction']
    start_margin = 0
    sample_skip_head_ratio = float(args['sample_skip_head_ratio'])
    sample_phase_window_len = int(args['sample_phase_window_len'])
    failure_mode = str(args.get('failure_mode', 'off')).strip().lower()
    if failure_mode not in {'off', 'explore', 'train'}:
        raise ValueError(f"Invalid failure_mode={failure_mode!r}, expected off|explore|train")
    failure_table_path = str(args.get('failure_table_path', '')).strip()
    failure_phase_bins = int(max(1, int(args.get('failure_phase_bins', 5))))
    failure_translation_dir_bins = int(max(1, int(args.get('failure_translation_dir_bins', 5))))
    failure_translation_mag_bins = int(max(1, int(args.get('failure_translation_mag_bins', 3))))
    failure_rotation_dir_bins = int(max(1, int(args.get('failure_rotation_dir_bins', 6))))
    failure_rotation_mag_bins = int(max(1, int(args.get('failure_rotation_mag_bins', 3))))
    failure_explore_k = int(max(1, int(args.get('failure_explore_k', 3))))
    failure_corr_batch_ratio = float(max(0.0, args.get('failure_corr_batch_ratio', 0.5)))
    failure_param_cfg = set_failure_param_bins(
        translation_dir_bins=failure_translation_dir_bins,
        translation_mag_bins=failure_translation_mag_bins,
        rotation_dir_bins=failure_rotation_dir_bins,
        rotation_mag_bins=failure_rotation_mag_bins,
    )
    print(f"[main][rank={_rank}] failure_param_bins={failure_param_cfg}")
    # Legacy pregrasp-bias / extra-pregrasp-correction path is retired.
    # Sampling is driven by failure table (explore/train) or plain random sampling (off).
    if not enable_wm:
        sample_skip_head_ratio = 0.0
        if failure_mode in {'explore', 'train'}:
            raise ValueError("failure_mode requires --enable_wm_correction")
    if enable_wm:
        if use_multi_task:
            raise ValueError("enable_wm_correction currently does not support multi-task training")
        wm_required = ['evac_ckpt', 'evac_config', 'urdf_path', 'curobo_left_yml',
                        'curobo_right_yml', 'raw_data_dir', 'act_init_ckpt',
                        'max_rollout_steps',
                        'orient_weight', 'gripper_penalty']
        missing = [k for k in wm_required if args.get(k) is None]
        if missing:
            raise ValueError(f"--enable_wm_correction requires these args: {missing}")
        # Reserve tail horizon based on actual rollout execution steps.
        if args['rollout_exec_steps'] is None:
            raise ValueError("rollout_exec_steps must be explicitly provided when enable_wm_correction=true")
        exec_steps = int(args['rollout_exec_steps'])
        start_margin = int(args['max_rollout_steps']) * int(exec_steps)
    raw_data_dir = args['raw_data_dir'] if enable_wm else None
    print(f"[main][rank={_rank}] before load_data | dataset_dir={dataset_dir} | num_episodes={num_episodes} | start_margin={start_margin}")
    _t_load = time.time()
    use_failure_explore_loader = bool(enable_wm and failure_mode == 'explore')
    use_failure_corr_loader = bool(enable_wm and failure_mode == 'train')
    base_sample_skip_head_ratio = 0.0 if enable_wm else float(sample_skip_head_ratio)

    corr_train_dataloader = None
    if use_failure_explore_loader:
        explore_batch_size = int(max(1, int(batch_size_train)))
        train_dataloader, _, stats, _, max_action_len = load_data(
            dataset_dir,
            num_episodes,
            camera_names,
            explore_batch_size,
            explore_batch_size,
            raw_data_dir=raw_data_dir,
            start_margin=start_margin,
            sample_skip_head_ratio=sample_skip_head_ratio,
            sample_phase_window_len=sample_phase_window_len,
            failure_mode=failure_mode,
            failure_table_path=failure_table_path,
            failure_phase_bins=failure_phase_bins,
            failure_explore_k=failure_explore_k,
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            task_weights=multi_task_weights,
        )
        print(
            f"[main][rank={_rank}] failure explore mode enabled (single explore loader) | "
            f"batch_size={explore_batch_size}"
        )
    else:
        train_dataloader, _, stats, _, max_action_len = load_data(
            dataset_dir,
            num_episodes,
            camera_names,
            batch_size_train,
            batch_size_train,
            raw_data_dir=raw_data_dir,
            start_margin=start_margin,
            sample_skip_head_ratio=base_sample_skip_head_ratio,
            sample_phase_window_len=sample_phase_window_len,
            failure_mode='off',
            failure_table_path='',
            failure_phase_bins=failure_phase_bins,
            failure_explore_k=failure_explore_k,
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            task_weights=multi_task_weights,
        )
    if use_failure_corr_loader:
        corr_batch_size = int(max(1, round(float(batch_size_train) * float(failure_corr_batch_ratio))))
        corr_train_dataloader, _, _, _, _ = load_data(
            dataset_dir,
            num_episodes,
            camera_names,
            corr_batch_size,
            corr_batch_size,
            raw_data_dir=raw_data_dir,
            start_margin=start_margin,
            sample_skip_head_ratio=sample_skip_head_ratio,
            sample_phase_window_len=sample_phase_window_len,
            failure_mode=failure_mode,
            failure_table_path=failure_table_path,
            failure_phase_bins=failure_phase_bins,
            failure_explore_k=failure_explore_k,
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            task_weights=multi_task_weights,
        )
        print(
            f"[main][rank={_rank}] failure-mode correction dataloader enabled | "
            f"mode={failure_mode} | corr_batch_size={corr_batch_size}"
        )
    print(f"[main][rank={_rank}] after load_data | elapsed={time.time() - _t_load:.2f}s | max_action_len={max_action_len}")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    config['enable_wm_correction'] = enable_wm
    config['failure_mode'] = failure_mode
    config['failure_phase_bins'] = failure_phase_bins
    config['failure_translation_dir_bins'] = failure_translation_dir_bins
    config['failure_translation_mag_bins'] = failure_translation_mag_bins
    config['failure_rotation_dir_bins'] = failure_rotation_dir_bins
    config['failure_rotation_mag_bins'] = failure_rotation_mag_bins
    config['failure_table_path'] = failure_table_path
    config['failure_table_dir'] = str(args.get('failure_table_dir', '')).strip()
    config['failure_explore_card_mode'] = str(args.get('failure_explore_card_mode', 'single')).strip().lower()
    config['failure_explore_k'] = int(max(1, int(args.get('failure_explore_k', 3))))
    config['failure_fail_recover_rate_thresh'] = float(args.get('failure_fail_recover_rate_thresh', 0.5))
    if enable_wm:
        config['wm_args'] = args
        config['raw_data_dir'] = raw_data_dir
        config['norm_stats'] = stats
        config['dataset_dir'] = dataset_dir
        rollout_exec_steps = args.get('rollout_exec_steps', None)
        if rollout_exec_steps is None or int(rollout_exec_steps) <= 0:
            rollout_exec_steps = int(args['chunk_size'])
        config['correction_cfg'] = {
            'max_rollout_steps': args['max_rollout_steps'],
            'correction_force_generate': bool(failure_mode == 'train'),
            'debug_correction_evac_rollout': args.get('debug_correction_evac_rollout', False),
            'rollout_exec_steps': int(rollout_exec_steps),
            'recover_eval_enable': bool(args.get('recover_eval_enable', False)),
            'recover_eval_save_video': bool(args.get('recover_eval_save_video', False)),
            'recover_eval_gripper_open_thresh': float(args.get('recover_eval_gripper_open_thresh', 0.8)),
            'recover_eval_pos_thresh_m': float(args.get('recover_eval_pos_thresh_m', 0.03)),
            'recover_eval_rot_thresh_deg': float(args.get('recover_eval_rot_thresh_deg', 10.0)),
            'recover_eval_nearest_window_radius': int(args.get('recover_eval_nearest_window_radius', 16)),
            'sample_phase_window_len': int(args['sample_phase_window_len']),
            'chunk_size': args['chunk_size'],
            'max_action_len': max_action_len,
            'orient_weight': args['orient_weight'],
            'gripper_penalty': args['gripper_penalty'],
            'evac_infer_kwargs': build_evac_infer_kwargs(args),
            # Perturbation is driven by failure_mode only.
            'enable_perturb': bool(failure_mode in {'explore', 'train'}),
            'perturb_eef_fail_gain': args['perturb_eef_fail_gain'],
            'perturb_rot_max_deg': args['perturb_rot_max_deg'],
            'perturb_active_joint_delta_thresh': args['perturb_active_joint_delta_thresh'],
            'perturb_active_gripper_delta_thresh': args['perturb_active_gripper_delta_thresh'],
            'failure_mode': failure_mode,
            'export_correction_dataset': args['export_correction_dataset'],
            'export_correction_dir': args['export_correction_dir'],
        }
        if failure_mode == 'explore':
            config['correction_cfg']['recover_eval_enable'] = True
    config['act_init_ckpt'] = args.get('act_init_ckpt')
    config['debug_wm_correction'] = args.get('debug_wm_correction', False)
    config['debug_loss_batch_projection'] = bool(args.get('debug_loss_batch_projection', False))
    print(f"[main][rank={_rank}] before train_bc | enable_wm={enable_wm}")
    train_bc(train_dataloader, config, corr_train_dataloader=corr_train_dataloader)

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

def init_correction(args, device):
    import sapien
    from omegaconf import OmegaConf
    # EVAC internal modules (e.g. ddpm3d) do `from utils.general_utils import ...`
    # which needs evac/evac/ on sys.path AND `utils` in sys.modules to point to
    # evac's utils package (not ACT's utils.py which is already cached).
    _act_root = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    _evac_evac = os.path.join(_act_root, 'evac', 'evac')
    if _evac_evac not in sys.path:
        sys.path.insert(0, _evac_evac)
    # Temporarily swap out ACT's utils module so EVAC can load its own utils package
    _act_utils = sys.modules.pop('utils', None)
    from evac.utils.general_utils import load_checkpoints, instantiate_from_config
    from util.fk_sapien import SapienFK
    _robotwin_root = os.path.realpath(os.path.join(_act_root, '..', '..'))
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

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data[0], data[1], data[2], data[3]
    return policy(qpos_data, image_data, action_data, is_pad)

def train_bc(train_dataloader, config, corr_train_dataloader=None):
    _r = int(os.environ.get("RANK", -1))
    _lr = int(os.environ.get("LOCAL_RANK", -1))
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    enable_wm = bool(config["enable_wm_correction"])
    failure_mode = str(config.get("failure_mode", "off")).strip().lower()
    failure_explore_mode = bool(failure_mode == "explore")
    failure_explore_card_mode = str(config.get("failure_explore_card_mode", "single")).strip().lower()
    if failure_explore_card_mode not in {"single", "multi"}:
        raise ValueError(
            f"Invalid failure_explore_card_mode={failure_explore_card_mode!r}, expected single|multi"
        )
    failure_explore_no_sync = bool(failure_explore_card_mode == "multi")
    failure_explore_emit_local_table = bool(failure_explore_card_mode == "single")
    failure_explore_k = int(max(1, int(config.get("failure_explore_k", 3))))
    failure_fail_thresh = float(config.get("failure_fail_recover_rate_thresh", 0.5))
    failure_table_dir = str(config.get("failure_table_dir", "")).strip()
    if failure_table_dir == "":
        failure_table_dir = os.path.join(ckpt_dir, "failure_explore")
    failure_live_dump_interval = 20

    # Accelerate: prepare model, optimizer, dataloader
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Set seed after Accelerator init so each process gets proper seed offset
    set_seed(seed)
    failure_explore_k_global = int(failure_explore_k)
    failure_explore_k_local = int(failure_explore_k_global)
    if failure_explore_mode and failure_explore_no_sync:
        ws = int(max(1, int(accelerator.num_processes)))
        # In multi-card explore, treat CLI k as global target and split to local quota.
        failure_explore_k_local = int(max(1, (failure_explore_k_global + ws - 1) // ws))
    if failure_explore_mode:
        print(
            f"[failure_explore] k_global={failure_explore_k_global} "
            f"k_local={failure_explore_k_local} mode={failure_explore_card_mode}"
        )

    print(f"[train_bc][rank={_r} local_rank={_lr}] before make_policy")
    _t_make_policy = time.time()
    policy = make_policy(policy_class, policy_config)
    print(f"[train_bc][rank={_r} local_rank={_lr}] after make_policy | elapsed={time.time() - _t_make_policy:.2f}s")

    if config.get('act_init_ckpt'):
        print(f"[train_bc][rank={_r} local_rank={_lr}] before load_act_init_ckpt")
        _t_load_act = time.time()
        ckpt = torch.load(config['act_init_ckpt'], map_location='cpu')
        policy.load_state_dict(ckpt)
        print(f"[train_bc][rank={_r} local_rank={_lr}] after load_act_init_ckpt | elapsed={time.time() - _t_load_act:.2f}s")
        print(f"Loaded ACT init weights from {config['act_init_ckpt']}")

    print(f"[train_bc][rank={_r} local_rank={_lr}] before make_optimizer")
    _t_make_opt = time.time()
    optimizer = make_optimizer(policy_class, policy)
    print(f"[train_bc][rank={_r} local_rank={_lr}] after make_optimizer | elapsed={time.time() - _t_make_opt:.2f}s")
    print(f"[train_bc][rank={_r} local_rank={_lr}] before accelerator.prepare")
    _t_prepare = time.time()
    if failure_explore_mode:
        policy, optimizer = accelerator.prepare(policy, optimizer)
    elif corr_train_dataloader is None:
        policy, optimizer, train_dataloader = accelerator.prepare(
            policy, optimizer, train_dataloader
        )
    else:
        policy, optimizer, train_dataloader, corr_train_dataloader = accelerator.prepare(
            policy, optimizer, train_dataloader, corr_train_dataloader
        )
    print(f"[train_bc][rank={_r} local_rank={_lr}] after accelerator.prepare | elapsed={time.time() - _t_prepare:.2f}s")

    correction_modules = None
    debug_wm = config.get('debug_wm_correction', False)
    debug_loss_batch_projection = bool(config.get('debug_loss_batch_projection', False))
    debug_wm_dir = os.path.join(ckpt_dir, 'debug_wm') if debug_wm else None
    if debug_wm_dir and accelerator.is_main_process:
        os.makedirs(debug_wm_dir, exist_ok=True)
    # correction statistics tracker
    _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                   'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': [],
                   'n_fallback': 0}
    _failure_trials = []
    _failure_stats = {}
    _exc_log_dir = os.path.join(ckpt_dir, "exception_logs")
    os.makedirs(_exc_log_dir, exist_ok=True)
    _exc_log_path = os.path.join(
        _exc_log_dir,
        f"correction_exceptions_rank{int(accelerator.process_index):02d}.jsonl",
    )

    def _append_correction_exception(exc, epoch_idx, step_idx, batch_idx, explore_unit_idx, tb_text):
        rec = {
            "time_unix": float(time.time()),
            "rank": int(accelerator.process_index),
            "local_rank": int(accelerator.local_process_index),
            "epoch": int(epoch_idx),
            "global_step": int(step_idx),
            "batch_idx": int(batch_idx),
            "failure_mode": str(failure_mode),
            "failure_explore_mode": bool(failure_explore_mode),
            "explore_unit_idx": int(explore_unit_idx),
            "error": str(exc),
            "traceback": str(tb_text),
        }
        try:
            with open(_exc_log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _build_failure_entries(trials):
        stats = {}
        for t in trials:
            try:
                key = (
                    str(t["phase_key"]),
                    int(t["phase_instance_idx"]),
                    int(t["phase_bin_id"]),
                    str(t["error_mode"]),
                    str(t["active_arm_pattern"]),
                    int(t["dir_bin_id"]),
                    int(t["mag_bin_id"]),
                )
                recoverable = bool(t.get("recoverable", False))
            except Exception:
                continue
            if key not in stats:
                stats[key] = {"n": 0, "n_recover": 0}
            stats[key]["n"] += 1
            stats[key]["n_recover"] += int(recoverable)

        entries = []
        for key, cnt in sorted(stats.items()):
            phase_key, phase_instance_idx, phase_bin_id, error_mode, active_arm_pattern, dir_bin_id, mag_bin_id = key
            n = int(cnt["n"])
            if n < failure_explore_k_global:
                continue
            n_recover = int(cnt["n_recover"])
            recover_rate = float(n_recover) / float(max(1, n))
            fail_flag = bool(recover_rate <= float(failure_fail_thresh))
            if not fail_flag:
                continue
            entries.append(
                {
                    "phase_key": str(phase_key),
                    "phase_instance_idx": int(phase_instance_idx),
                    "phase_bin_id": int(phase_bin_id),
                    "error_mode": str(error_mode),
                    "active_arm_pattern": str(active_arm_pattern),
                    "dir_bin_id": int(dir_bin_id),
                    "mag_bin_id": int(mag_bin_id),
                    "n_trials": int(n),
                    "n_recover": int(n_recover),
                    "recover_rate": float(recover_rate),
                    "fail_rate": float(1.0 - recover_rate),
                    "weight": float(max(1e-6, 1.0 - recover_rate)),
                }
            )
        return entries

    def _gather_failure_trials_all_ranks():
        local_trials = list(_failure_trials)
        if failure_explore_no_sync:
            return local_trials
        if not (dist.is_available() and dist.is_initialized()):
            return local_trials
        ws = int(dist.get_world_size())
        gathered = [None for _ in range(ws)]
        dist.all_gather_object(gathered, local_trials)
        merged = []
        for g in gathered:
            if isinstance(g, list):
                merged.extend(g)
        return merged

    def _write_failure_tables(epoch_idx, live=False):
        if not failure_explore_mode:
            return
        all_trials = _gather_failure_trials_all_ranks()
        if (not failure_explore_no_sync) and (not accelerator.is_main_process):
            return
        os.makedirs(failure_table_dir, exist_ok=True)
        rank_suffix = (f"_rank{int(accelerator.process_index):02d}" if failure_explore_no_sync else "")
        trials_path = os.path.join(
            failure_table_dir,
            (
                f"failure_trials_live{rank_suffix}.json"
                if live else f"failure_trials_epoch_{epoch_idx + 1:04d}{rank_suffix}.json"
            ),
        )
        with open(trials_path, "w") as f:
            json.dump(all_trials, f, indent=2)

        if not failure_explore_emit_local_table:
            return

        entries = _build_failure_entries(all_trials)
        table = {
            "version": 3,
            "mode": ("explore_live" if live else "explore"),
            "epoch": int(epoch_idx + 1),
            "failure_phase_bins": int(config.get("failure_phase_bins", 5)),
            "failure_translation_dir_bins": int(config.get("failure_translation_dir_bins", 5)),
            "failure_translation_mag_bins": int(config.get("failure_translation_mag_bins", 3)),
            "failure_rotation_dir_bins": int(config.get("failure_rotation_dir_bins", 6)),
            "failure_rotation_mag_bins": int(config.get("failure_rotation_mag_bins", 3)),
                    "failure_explore_k": int(failure_explore_k_global),
            "failure_fail_recover_rate_thresh": float(failure_fail_thresh),
            "entries": entries,
        }
        table_path = os.path.join(
            failure_table_dir,
            (
                f"failure_table_live{rank_suffix}.json"
                if live else f"failure_table{rank_suffix}.json"
            ),
        )
        with open(table_path, "w") as f:
            json.dump(table, f, indent=2)

        for mode in ("translation", "rotation", "gripper_close"):
            table_mode = dict(table)
            table_mode["entries"] = [e for e in entries if str(e.get("error_mode", "")) == mode]
            mode_path = os.path.join(
                failure_table_dir,
                (
                    f"failure_table_live_{mode}{rank_suffix}.json"
                    if live else f"failure_table_{mode}{rank_suffix}.json"
                ),
            )
            with open(mode_path, "w") as f:
                json.dump(table_mode, f, indent=2)

        if not live:
            print(
                f"[failure_explore] epoch={epoch_idx + 1} trials={len(all_trials)} "
                f"failure_entries={len(entries)} table={table_path}"
            )

    def _iter_leaf_datasets(ds):
        if ds is None:
            return
        if hasattr(ds, "datasets"):
            for _sub in ds.datasets:
                yield from _iter_leaf_datasets(_sub)
            return
        if hasattr(ds, "dataset"):
            yield from _iter_leaf_datasets(ds.dataset)
            return
        yield ds

    def _set_explore_unit_idx_for_loader(loader, unit_idx):
        if loader is None:
            return
        ds = getattr(loader, "dataset", None)
        for leaf in _iter_leaf_datasets(ds):
            if hasattr(leaf, "set_explore_unit_idx"):
                leaf.set_explore_unit_idx(int(unit_idx))

    def _set_explore_local_k_for_loader(loader, k_local):
        if loader is None:
            return
        ds = getattr(loader, "dataset", None)
        for leaf in _iter_leaf_datasets(ds):
            if hasattr(leaf, "set_explore_local_k"):
                leaf.set_explore_local_k(int(k_local))

    def _record_explore_trial_for_loader(loader, unit_idx, episode_id, start_ts):
        if loader is None:
            return
        ds = getattr(loader, "dataset", None)
        for leaf in _iter_leaf_datasets(ds):
            if hasattr(leaf, "record_explore_trial"):
                leaf.record_explore_trial(int(unit_idx), int(episode_id), int(start_ts))

    def _get_explore_num_units(loader):
        ds = getattr(loader, "dataset", None)
        for leaf in _iter_leaf_datasets(ds):
            if hasattr(leaf, "_explore_units"):
                return int(len(getattr(leaf, "_explore_units")))
        return 0

    def _get_current_explore_unit_idx(loader):
        ds = getattr(loader, "dataset", None)
        for leaf in _iter_leaf_datasets(ds):
            if hasattr(leaf, "_explore_curr_unit_idx"):
                return int(getattr(leaf, "_explore_curr_unit_idx"))
        return -1

    if enable_wm:
        print(f"[train_bc][rank={_r} local_rank={_lr}] before init_correction")
        _t_init_corr = time.time()
        correction_modules = init_correction(config['wm_args'], accelerator.device)
        print(f"[train_bc][rank={_r} local_rank={_lr}] after init_correction | elapsed={time.time() - _t_init_corr:.2f}s")
        norm_stats = config['norm_stats']
        correction_cfg = config['correction_cfg']
        raw_data_dir = config['raw_data_dir']
        export_corr = bool(correction_cfg.get('export_correction_dataset', False))
        export_corr_dir = str(correction_cfg.get('export_correction_dir', '')).strip()
        if export_corr and export_corr_dir == '':
            export_corr_dir = os.path.join(ckpt_dir, 'correction_dataset')
        export_next_id = 0
        if export_corr and accelerator.is_main_process:
            os.makedirs(export_corr_dir, exist_ok=True)
            export_next_id = _init_export_episode_id(export_corr_dir)
            print(f"[train_bc] correction export enabled: {export_corr_dir}, next_episode_id={export_next_id}")

    train_history = []

    # TensorBoard (only on main process)
    writer = None
    if accelerator.is_main_process:
        tb_log_dir = os.path.join(ckpt_dir, 'tb_logs')
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f'TensorBoard log dir: {tb_log_dir}')
        if failure_explore_mode:
            os.makedirs(failure_table_dir, exist_ok=True)
    global_step = 0
    explore_num_units = int(_get_explore_num_units(train_dataloader)) if failure_explore_mode else 0
    if failure_explore_mode and explore_num_units > 0:
        _set_explore_unit_idx_for_loader(train_dataloader, 0)
        _set_explore_local_k_for_loader(train_dataloader, int(failure_explore_k_local))

    for epoch in tqdm(range(num_epochs), disable=not accelerator.is_main_process):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}")
        corr_iter = iter(corr_train_dataloader) if corr_train_dataloader is not None else None

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            did_explore_collective = False

            apply_wm = enable_wm and correction_modules is not None
            if not apply_wm:
                forward_dict = forward_pass(data, policy)
                loss = forward_dict["loss"]
            else:
                try:
                    corr_source = data
                    if corr_iter is not None:
                        try:
                            corr_source = next(corr_iter)
                        except StopIteration:
                            corr_iter = iter(corr_train_dataloader)
                            corr_source = next(corr_iter)
                    bs = int(corr_source[0].shape[0])
                    bs_base = int(data[0].shape[0])
                    corr_images, corr_qpos, corr_actions, corr_pads = [], [], [], []
                    corr_mask = []
                    # per-step debug dir (only on main process, only every 100 steps)
                    _step_dbg = None
                    if debug_wm_dir and accelerator.is_main_process:
                        _step_dbg = os.path.join(debug_wm_dir, f'step_{global_step:06d}')
                        os.makedirs(_step_dbg, exist_ok=True)
                    for bi in range(bs):
                        ep_id = corr_source[4][bi].item()
                        raw = load_raw_data(raw_data_dir, ep_id)
                        _bi_dbg = os.path.join(_step_dbg, f'bi{bi}') if _step_dbg else None
                        if _bi_dbg:
                            os.makedirs(_bi_dbg, exist_ok=True)
                        corr = correction_step(
                            accelerator.unwrap_model(policy),
                            corr_source[0][bi], corr_source[1][bi], raw,
                            norm_stats, correction_modules, correction_cfg,
                            accelerator.device, debug_dir=_bi_dbg,
                            start_ts=corr_source[5][bi].item(),
                            sampled_phase_id=corr_source[6][bi].item(),
                            pregrasp_seg_start=corr_source[7][bi].item(),
                            pregrasp_seg_end=corr_source[8][bi].item(),
                            sampled_phase_bin_id=(
                                None if len(corr_source) <= 9 else corr_source[9][bi].item()
                            ),
                            sampled_phase_instance_id=(
                                None if len(corr_source) <= 10 else corr_source[10][bi].item()
                            ),
                            forced_error_mode_id=(
                                None if len(corr_source) <= 11 else corr_source[11][bi].item()
                            ),
                            sampled_active_arm_pattern_id=(
                                None if len(corr_source) <= 12 else corr_source[12][bi].item()
                            ),
                            forced_dir_bin_id=(
                                None if len(corr_source) <= 13 else corr_source[13][bi].item()
                            ),
                            forced_mag_bin_id=(
                                None if len(corr_source) <= 14 else corr_source[14][bi].item()
                            ),
                        )
                        _corr_stats['n_triggered'] += 1
                        ci, cq, ca, cp = None, None, None, None
                        cmeta = {}
                        if isinstance(corr, (tuple, list)):
                            if len(corr) >= 5:
                                ci, cq, ca, cp, cmeta = corr
                            elif len(corr) >= 4:
                                ci, cq, ca, cp = corr[0], corr[1], corr[2], corr[3]
                        elif isinstance(corr, dict):
                            cmeta = corr

                        corr_generated = bool(
                            (ci is not None) and (cq is not None) and (ca is not None) and (cp is not None)
                        )

                        if isinstance(cmeta, dict):
                            if corr_generated and bool(cmeta.get('closed_loop_fallback_used', False)):
                                _corr_stats['n_fallback'] += 1
                            if failure_explore_mode:
                                phase_key = phase_id_to_key(int(corr_source[6][bi].item()))
                                phase_bin_id = int(corr_source[9][bi].item()) if len(corr_source) > 9 else -1
                                phase_instance_idx = int(corr_source[10][bi].item()) if len(corr_source) > 10 else -1
                                forced_mode_id = int(corr_source[11][bi].item()) if len(corr_source) > 11 else -1
                                active_pattern_id = int(corr_source[12][bi].item()) if len(corr_source) > 12 else -1
                                dir_bin_id = int(corr_source[13][bi].item()) if len(corr_source) > 13 else -1
                                mag_bin_id = int(corr_source[14][bi].item()) if len(corr_source) > 14 else -1
                                forced_mode_key = None
                                if forced_mode_id >= 0:
                                    forced_mode_key = error_mode_id_to_key(forced_mode_id)
                                active_pattern_key = "both"
                                if active_pattern_id >= 0:
                                    active_pattern_key = active_arm_pattern_id_to_key(active_pattern_id)
                                recover_eval_last = cmeta.get("recover_eval_last", {})
                                recoverable_raw = recover_eval_last.get("recoverable", None)
                                error_mode_key = str(forced_mode_key or cmeta.get("sampled_error_mode") or "").strip().lower()
                                if (
                                    (recoverable_raw is not None)
                                    and (error_mode_key in {"translation", "rotation", "gripper_close"})
                                ):
                                    sample_unit_idx = int(corr_source[15][bi].item()) if len(corr_source) > 15 else -1
                                    recoverable = bool(recoverable_raw)
                                    trial = {
                                        "epoch": int(epoch),
                                        "global_step": int(global_step),
                                        "episode_id": int(ep_id),
                                        "start_ts": int(corr_source[5][bi].item()),
                                        "phase_key": str(phase_key),
                                        "phase_instance_idx": int(phase_instance_idx),
                                        "phase_bin_id": int(phase_bin_id),
                                        "error_mode": str(error_mode_key),
                                        "active_arm_pattern": str(active_pattern_key),
                                        "dir_bin_id": int(dir_bin_id),
                                        "mag_bin_id": int(mag_bin_id),
                                        "recoverable": bool(recoverable),
                                        "recover_eval_mode": recover_eval_last.get("mode"),
                                        "recover_eval_metric_name": recover_eval_last.get("metric_name"),
                                        "recover_eval_metric": recover_eval_last.get("metric"),
                                        "recover_eval_threshold": recover_eval_last.get("threshold"),
                                    }
                                    key = (
                                        str(trial["phase_key"]),
                                        int(trial["phase_instance_idx"]),
                                        int(trial["phase_bin_id"]),
                                        str(trial["error_mode"]),
                                        str(trial["active_arm_pattern"]),
                                        int(trial["dir_bin_id"]),
                                        int(trial["mag_bin_id"]),
                                    )
                                    if key not in _failure_stats:
                                        _failure_stats[key] = {"n": 0, "n_recover": 0, "seen_samples": set()}
                                    sample_uid = (int(ep_id), int(corr_source[5][bi].item()))
                                    if sample_uid not in _failure_stats[key]["seen_samples"]:
                                        _failure_stats[key]["seen_samples"].add(sample_uid)
                                        _failure_stats[key]["n"] += 1
                                        _failure_stats[key]["n_recover"] += int(recoverable)
                                        _failure_trials.append(trial)
                                        _record_explore_trial_for_loader(
                                            train_dataloader,
                                            int(sample_unit_idx),
                                            int(ep_id),
                                            int(corr_source[5][bi].item()),
                                        )

                        if corr_generated:
                            corr_images.append(ci)
                            corr_qpos.append(cq)
                            corr_actions.append(ca)
                            corr_pads.append(cp)
                            corr_mask.append(1.0)
                            _corr_stats['n_success'] += 1
                            if enable_wm and export_corr and accelerator.is_main_process:
                                _export_correction_sample_as_episode(
                                    export_corr_dir,
                                    export_next_id,
                                    config['camera_names'],
                                    ci,
                                    cq,
                                    ca,
                                    cp,
                                    norm_stats,
                                )
                                export_next_id += 1
                        else:
                            _corr_stats['n_skipped'] += 1
                            # dummy data to keep batch size consistent across ranks
                            corr_images.append(corr_source[0][bi].to(accelerator.device))
                            corr_qpos.append(corr_source[1][bi].to(accelerator.device))
                            corr_actions.append(corr_source[2][bi].to(accelerator.device))
                            corr_pads.append(corr_source[3][bi].to(accelerator.device))
                            corr_mask.append(0.0)

                    if failure_explore_mode:
                        if global_step % failure_live_dump_interval == 0:
                            _write_failure_tables(epoch, live=True)
                        did_explore_collective = True
                        global_step += 1
                        continue

                    # Build a fixed-size mixed batch: [base bs] + [correction bs].
                    ci_b = torch.stack(corr_images, dim=0)
                    cq_b = torch.stack(corr_qpos, dim=0)
                    ca_b = torch.stack(corr_actions, dim=0)
                    cp_b = torch.stack(corr_pads, dim=0)

                    base_images = data[0].to(accelerator.device)
                    base_qpos = data[1].to(accelerator.device)
                    base_actions = data[2].to(accelerator.device)
                    base_pads = data[3].to(accelerator.device)

                    mixed_images = torch.cat([base_images, ci_b], dim=0)
                    mixed_qpos = torch.cat([base_qpos, cq_b], dim=0)
                    mixed_actions = torch.cat([base_actions, ca_b], dim=0)
                    mixed_pads = torch.cat([base_pads, cp_b], dim=0)
                    if debug_loss_batch_projection and _step_dbg is not None:
                        _loss_dbg = os.path.join(_step_dbg, 'loss_batch_projection')
                        os.makedirs(_loss_dbg, exist_ok=True)
                        _raw_cache = {}

                        def _get_raw(ep_id_i):
                            _k = int(ep_id_i)
                            if _k not in _raw_cache:
                                _raw_cache[_k] = load_raw_data(raw_data_dir, _k)
                            return _raw_cache[_k]

                        for i in range(bs_base):
                            ep_i = int(data[4][i].item())
                            st_i = int(data[5][i].item())
                            _sdir = os.path.join(_loss_dbg, f'base_{i:03d}_ep{ep_i}_ts{st_i:04d}')
                            _save_loss_batch_projection(
                                _sdir,
                                mixed_images[i][0],
                                mixed_actions[i],
                                mixed_pads[i],
                                _get_raw(ep_i),
                                correction_modules['fk'],
                                norm_stats,
                                meta={'sample_type': 'base', 'episode_id': ep_i, 'start_ts': st_i},
                            )
                        for j in range(bs):
                            k = bs_base + j
                            ep_j = int(corr_source[4][j].item())
                            st_j = int(corr_source[5][j].item())
                            _sdir = os.path.join(_loss_dbg, f'corr_{j:03d}_ep{ep_j}_ts{st_j:04d}')
                            _save_loss_batch_projection(
                                _sdir,
                                mixed_images[k][0],
                                mixed_actions[k],
                                mixed_pads[k],
                                _get_raw(ep_j),
                                correction_modules['fk'],
                                norm_stats,
                                meta={
                                    'sample_type': 'correction',
                                    'episode_id': ep_j,
                                    'start_ts': st_j,
                                    'correction_generated': bool(corr_mask[j] > 0.5),
                                },
                            )

                    mixed_dict = policy(
                        mixed_qpos,
                        mixed_images,
                        mixed_actions,
                        mixed_pads,
                        return_per_sample=True,
                    )

                    per_sample_loss = mixed_dict['loss_per_sample']
                    mask_t = torch.tensor(corr_mask, device=accelerator.device, dtype=per_sample_loss.dtype)
                    n_corr = int(mask_t.sum().item())

                    base_weight = torch.ones(bs_base, device=accelerator.device, dtype=per_sample_loss.dtype)
                    sample_weight = torch.cat([base_weight, mask_t], dim=0)
                    loss = (per_sample_loss * sample_weight).sum() / (bs_base + n_corr)
                    forward_dict = {"loss": loss}

                    if accelerator.is_main_process and writer is not None:
                        if n_corr > 0:
                            corr_loss = (per_sample_loss[bs_base:] * mask_t).sum() / mask_t.sum()
                        else:
                            corr_loss = torch.tensor(0.0, device=accelerator.device, dtype=per_sample_loss.dtype)
                        writer.add_scalar('train/correction_loss', corr_loss.item(), global_step)
                        writer.add_scalar('train/n_correction_samples', n_corr, global_step)
                except Exception as exc:
                    _corr_stats['n_error'] += 1
                    _tb_text = traceback.format_exc()
                    _append_correction_exception(
                        exc,
                        epoch,
                        global_step,
                        batch_idx,
                        _get_current_explore_unit_idx(train_dataloader),
                        _tb_text,
                    )
                    if accelerator.is_main_process:
                        print(f'[WM correction] step {global_step} error: {exc}')
                        print(_tb_text)
                    if failure_explore_mode and (not did_explore_collective):
                        # Keep collectives aligned across ranks on synced mode.
                        if (not failure_explore_no_sync) and dist.is_available() and dist.is_initialized():
                            ws = int(dist.get_world_size())
                            gathered_uids = [None for _ in range(ws)]
                            dist.all_gather_object(gathered_uids, [])
                        if global_step % failure_live_dump_interval == 0:
                            _write_failure_tables(epoch, live=True)
                        global_step += 1
                        continue
                    forward_dict = forward_pass(data, policy)
                    loss = forward_dict["loss"]

            if failure_explore_mode:
                global_step += 1
                continue

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if accelerator.is_main_process and writer is not None:
                writer.add_scalar('train/loss_step', forward_dict['loss'].item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
        if failure_explore_mode:
            epoch_summary = {}
        else:
            epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
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
                writer.add_scalar('correction/fallback_count', _corr_stats['n_fallback'], epoch)
                _total = _corr_stats['n_triggered'] or 1
                writer.add_scalar('correction/success_rate', _corr_stats['n_success'] / _total, epoch)
                _succ = _corr_stats['n_success'] or 1
                writer.add_scalar('correction/fallback_rate', _corr_stats['n_fallback'] / _succ, epoch)
                print(f'  [Correction] triggered={_corr_stats["n_triggered"]} '
                      f'success={_corr_stats["n_success"]} '
                      f'skipped={_corr_stats["n_skipped"]} '
                      f'error={_corr_stats["n_error"]} '
                      f'fallback={_corr_stats["n_fallback"]} '
                      f'rate={_corr_stats["n_success"]/_total:.2%}')
                # reset per-epoch
                _corr_stats = {'n_triggered': 0, 'n_success': 0, 'n_skipped': 0,
                               'n_plan_fail': 0, 'n_error': 0, 'dists': [], 'steps_used': [],
                               'n_fallback': 0}

        if failure_explore_mode:
            _write_failure_tables(epoch, live=False)

        if (epoch + 1) % config['save_freq'] == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            unwrapped_policy = accelerator.unwrap_model(policy)
            torch.save(unwrapped_policy.state_dict(), ckpt_path)

    if accelerator.is_main_process:
        if writer is not None:
            writer.close()

        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        unwrapped_policy = accelerator.unwrap_model(policy)
        torch.save(unwrapped_policy.state_dict(), ckpt_path)

    print(f"Training finished: Seed {seed}")
