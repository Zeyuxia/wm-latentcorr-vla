#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml


FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "policy") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "policy"))

from envs import CONFIGS_PATH  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "policy/SmolVLA/outputs/stage1/robotwin_multitask_5_cam_high"
    / "20260430_002055-stage1_corr025_all5_cards7654_steps3000"
)
DEFAULT_PARQUET = (
    REPO_ROOT
    / "policy/SmolVLA/data/robotwin_multitask_5_cam_high/data/chunk-000/file-000.parquet"
)
DEFAULT_TASKS = [
    "open_laptop",
    "pick_dual_bottles",
    "put_bottles_dustbin",
    "place_burger_fries",
    "handover_block",
]
TASK_EPISODE_RANGES = {
    "open_laptop": range(0, 50),
    "pick_dual_bottles": range(50, 100),
    "put_bottles_dustbin": range(100, 150),
    "place_burger_fries": range(150, 200),
    "handover_block": range(200, 250),
}
DIM_NAMES = [
    "LJ0",
    "LJ1",
    "LJ2",
    "LJ3",
    "LJ4",
    "LJ5",
    "LGrip",
    "RJ0",
    "RJ1",
    "RJ2",
    "RJ3",
    "RJ4",
    "RJ5",
    "RGrip",
]


def parse_task_name(task_name_full: str | None) -> str | None:
    if not task_name_full:
        return None
    value = str(task_name_full)
    match = re.match(r"^sim-(.*)-demo_clean-\d+$", value)
    if match:
        return match.group(1)
    if value.startswith("sim-"):
        value = value[4:]
    return value.split("-demo_clean-")[0]


def class_decorator(task_name: str):
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except AttributeError as exc:
        raise SystemExit(f"No such task: {task_name}") from exc


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)


def load_task_args(
    task_name: str,
    task_config: str,
    save_path: Path | None = None,
    ray_tracing_denoiser: str | None = None,
) -> dict[str, Any]:
    config_path = REPO_ROOT / "task_config" / f"{task_config}.yml"
    with config_path.open("r", encoding="utf-8") as file:
        args = yaml.load(file.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_data"] = False
    args["collect_data"] = False
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None
    args["eval_mode"] = False
    args["need_topp"] = False
    if save_path is not None:
        args["save_path"] = str(save_path)
    if ray_tracing_denoiser is not None:
        args["ray_tracing_denoiser"] = ray_tracing_denoiser

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as file:
        embodiment_types = yaml.load(file.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment: str) -> str:
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"Missing embodiment file for {embodiment}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest in sorted(run_dir.glob("correction_data_manifest_rank*.jsonl")):
        with manifest.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["task"] = parse_task_name(record.get("task_name"))
                record["manifest_path"] = str(manifest)
                records.append(record)
    return records


def load_original_stats(parquet_path: Path) -> dict[str, dict[str, np.ndarray]]:
    import pandas as pd

    episode_to_task = {
        episode: task for task, episodes in TASK_EPISODE_RANGES.items() for episode in episodes
    }
    df = pd.read_parquet(parquet_path, columns=["episode_index", "action"])
    task_actions: dict[str, list[np.ndarray]] = {task: [] for task in DEFAULT_TASKS}
    for episode, group in df.groupby("episode_index"):
        task = episode_to_task.get(int(episode))
        if task is None:
            continue
        task_actions[task].append(np.stack(group["action"].to_numpy()).astype(np.float64))

    stats: dict[str, dict[str, np.ndarray]] = {}
    for task, chunks in task_actions.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        stats[task] = {
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
        }
    return stats


def compute_hard_excess(record: dict[str, Any], stats: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    task = record.get("task")
    sample_path = Path(record["path"])
    if task not in stats or not sample_path.is_file():
        return {"max_excess": 0.0}
    with np.load(sample_path) as sample:
        action = sample["corr_action_chunk_raw"].astype(np.float64)
    low = np.maximum(stats[task]["min"] - action, 0.0)
    high = np.maximum(action - stats[task]["max"], 0.0)
    excess = np.maximum(low, high)
    flat_idx = int(np.argmax(excess))
    time_idx, dim_idx = np.unravel_index(flat_idx, excess.shape)
    return {
        "max_excess": float(excess[time_idx, dim_idx]),
        "max_excess_time": int(time_idx),
        "max_excess_dim": int(dim_idx),
        "max_excess_dim_name": DIM_NAMES[int(dim_idx)],
        "max_excess_value": float(action[time_idx, dim_idx]),
        "orig_min": float(stats[task]["min"][dim_idx]),
        "orig_max": float(stats[task]["max"][dim_idx]),
    }


def select_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = records
    if args.rank is not None:
        wanted_ranks = {int(rank) for rank in args.rank}
        selected = [record for record in selected if int(record.get("rank", -1)) in wanted_ranks]
    if args.task:
        wanted = set(args.task)
        selected = [record for record in selected if record.get("task") in wanted]
    if args.sample_path:
        wanted_paths = {str(Path(path).resolve()) for path in args.sample_path}
        selected = [record for record in selected if str(Path(record["path"]).resolve()) in wanted_paths]
    if args.episode_id is not None:
        selected = [record for record in selected if int(record.get("episode_id", -1)) == int(args.episode_id)]
    if args.max_start_ts is not None:
        selected = [record for record in selected if int(record.get("start_ts", 10**9)) <= int(args.max_start_ts)]

    if args.select == "top_outliers":
        stats = load_original_stats(Path(args.original_parquet))
        for record in selected:
            record.update(compute_hard_excess(record, stats))
        selected = sorted(selected, key=lambda record: float(record.get("max_excess", 0.0)), reverse=True)
    elif args.select == "first":
        selected = sorted(selected, key=lambda record: (int(record.get("rank", 0)), int(record.get("global_step", 0))))
    elif args.select == "random":
        rng = np.random.default_rng(int(args.seed))
        selected = list(selected)
        rng.shuffle(selected)
    else:
        raise ValueError(f"Unknown --select: {args.select}")

    if int(args.num_samples) <= 0:
        return selected
    return selected[: int(args.num_samples)]


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("ffmpeg writer is not open")
        frame = np.asarray(frame)
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
        self.proc.stdin.write(frame.tobytes())

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.wait()
        self.proc = None


def _wrap_overlay_line(text: str, max_chars: int) -> list[str]:
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for token in text.split(" "):
        if not current:
            current = token
        elif len(current) + 1 + len(token) <= max_chars:
            current = f"{current} {token}"
        else:
            chunks.append(current)
            current = token
    if current:
        chunks.append(current)
    wrapped: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            wrapped.append(chunk)
        else:
            wrapped.extend(chunk[i : i + max_chars] for i in range(0, len(chunk), max_chars))
    return wrapped


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = np.ascontiguousarray(frame.copy())
    overlay = out.copy()
    font_scale = 0.32
    thickness = 1
    line_step = 14
    margin_x = 6
    margin_y = 5
    max_chars = max(42, int((out.shape[1] - 2 * margin_x) / 5.4))
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(_wrap_overlay_line(str(line), max_chars=max_chars))
    max_overlay_height = int(out.shape[0] * 0.45)
    max_lines = max(1, min(len(wrapped_lines), max(1, max_overlay_height // line_step)))
    wrapped_lines = wrapped_lines[:max_lines]
    height = line_step * len(wrapped_lines) + 2 * margin_y
    cv2.rectangle(overlay, (0, 0), (out.shape[1], height), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    y = margin_y + 10
    for line in wrapped_lines:
        cv2.putText(
            out,
            line,
            (margin_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_step
    return out


def current_qpos_from_obs(obs: dict[str, Any]) -> np.ndarray:
    joint = obs["joint_action"]
    if "vector" in joint:
        return np.asarray(joint["vector"], dtype=np.float32)
    return np.asarray(
        list(joint["left_arm"]) + [joint["left_gripper"]] + list(joint["right_arm"]) + [joint["right_gripper"]],
        dtype=np.float32,
    )


def capture_obs_and_frame(task_env) -> tuple[dict[str, Any], np.ndarray]:
    obs = task_env.get_obs()
    frame = obs["observation"]["head_camera"]["rgb"]
    return obs, np.ascontiguousarray(frame.astype(np.uint8))


@dataclass
class ReplayResult:
    video_path: str
    metrics_path: str
    max_tracking_linf: float
    mean_tracking_l2: float
    start_linf_to_raw: float


def raw_data_paths(task: str, task_config: str, episode_id: int) -> tuple[Path, Path]:
    task_dir = REPO_ROOT / "data" / task / task_config
    seed_path = task_dir / "seed.txt"
    hdf5_path = task_dir / "data" / f"episode{episode_id}.hdf5"
    return seed_path, hdf5_path


def replay_output_paths(record: dict[str, Any], output_dir: Path, include_perturb_state: bool) -> tuple[Path, Path]:
    task = str(record["task"])
    episode_id = int(record["episode_id"])
    start_ts = int(record["start_ts"])
    sample_tag = (
        f"{task}_rank{int(record.get('rank', -1)):02d}_step{int(record.get('global_step', -1)):06d}"
        f"_ep{episode_id:02d}_ts{start_ts:04d}"
    )
    sample_dir = output_dir / sample_tag
    if include_perturb_state:
        return sample_dir / "replay_with_perturb.mp4", sample_dir / "metrics_with_perturb.json"
    return sample_dir / "replay.mp4", sample_dir / "metrics.json"


def load_seed(seed_path: Path, episode_id: int) -> int:
    seeds = [int(token) for token in seed_path.read_text().split()]
    if not (0 <= int(episode_id) < len(seeds)):
        raise IndexError(f"episode_id={episode_id} outside seed list: {seed_path}")
    return int(seeds[int(episode_id)])


def load_raw_action_vector(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as file:
        if "joint_action/vector" in file:
            return file["joint_action/vector"][()].astype(np.float32)
        return np.concatenate(
            [
                file["joint_action/left_arm"][()].astype(np.float32),
                file["joint_action/left_gripper"][()].astype(np.float32)[:, None],
                file["joint_action/right_arm"][()].astype(np.float32),
                file["joint_action/right_gripper"][()].astype(np.float32)[:, None],
            ],
            axis=1,
        )


def replay_record(
    record: dict[str, Any],
    *,
    output_dir: Path,
    task_config: str,
    fps: int,
    record_prefix: bool,
    prefix_frame_stride: int,
    hold_frames: int,
    include_perturb_state: bool,
    perturb_bridge_steps: int,
    ray_tracing_denoiser: str | None = None,
) -> ReplayResult:
    task = str(record["task"])
    episode_id = int(record["episode_id"])
    start_ts = int(record["start_ts"])
    seed_path, hdf5_path = raw_data_paths(task, task_config, episode_id)
    scene_seed = load_seed(seed_path, episode_id)
    raw_actions = load_raw_action_vector(hdf5_path)

    with np.load(record["path"]) as sample:
        corr_actions = sample["corr_action_chunk_raw"].astype(np.float32)
        sim_compare_actions = (
            sample["sim_compare_action_chunk_raw"].astype(np.float32)
            if "sim_compare_action_chunk_raw" in sample
            else None
        )
        corr_qpos = sample["corr_qpos_raw"].astype(np.float32) if "corr_qpos_raw" in sample else None
        perturb_actions = (
            sample["perturb_action_prefix_raw"].astype(np.float32)
            if "perturb_action_prefix_raw" in sample
            else None
        )
        perturb_start_qpos = (
            sample["perturb_start_qpos_raw"].astype(np.float32)
            if "perturb_start_qpos_raw" in sample
            else None
        )
        perturb_final_qpos = (
            sample["perturb_final_qpos_raw"].astype(np.float32)
            if "perturb_final_qpos_raw" in sample
            else None
        )
        error_action_prefix = (
            sample["error_action_prefix_raw"].astype(np.float32)
            if "error_action_prefix_raw" in sample
            else None
        )
    if sim_compare_actions is not None:
        # World-model compare predicts the result of the perturb/action prefix
        # from the clean start image.  Replay that exact prefix once in the
        # simulator while keeping the original correction chunk intact.
        perturb_actions = sim_compare_actions

    sample_tag = (
        f"{task}_rank{int(record.get('rank', -1)):02d}_step{int(record.get('global_step', -1)):06d}"
        f"_ep{episode_id:02d}_ts{start_ts:04d}"
    )
    sample_dir = output_dir / sample_tag
    sample_dir.mkdir(parents=True, exist_ok=True)
    video_path, metrics_path = replay_output_paths(record, output_dir, include_perturb_state)

    env_save_path = sample_dir / "env_cache"
    task_args = load_task_args(
        task,
        task_config,
        save_path=env_save_path,
        ray_tracing_denoiser=ray_tracing_denoiser,
    )
    task_env = class_decorator(task)
    task_env.setup_demo(now_ep_num=episode_id, seed=scene_seed, is_test=True, **task_args)
    # We only want to replay actions, not terminate early or call task-specific success state.
    task_env.check_success = lambda *args, **kwargs: False

    width, height = 640, 480
    _, frame = capture_obs_and_frame(task_env)
    height, width = frame.shape[:2]

    qpos_trace: list[list[float]] = []
    cmd_trace: list[list[float]] = []
    tracking_linf: list[float] = []
    tracking_l2: list[float] = []
    perturb_tracking_linf: list[float] = []
    perturb_tracking_l2: list[float] = []

    prefix_actions = raw_actions[1 : min(start_ts + 1, len(raw_actions))]
    error_prefix_len = 0 if error_action_prefix is None else int(error_action_prefix.shape[0])
    error_prefix_matches_corr_prefix_linf = None
    if error_action_prefix is not None and len(corr_actions) > 0:
        n_cmp = min(int(error_action_prefix.shape[0]), int(corr_actions.shape[0]))
        error_prefix_matches_corr_prefix_linf = float(
            np.max(np.abs(error_action_prefix[:n_cmp] - corr_actions[:n_cmp]))
        )

    with FfmpegWriter(video_path, width=width, height=height, fps=fps) as writer:
        if record_prefix:
            for idx, action in enumerate(prefix_actions):
                task_env.take_action(action.tolist())
                task_env.eval_success = False
                if idx % max(1, int(prefix_frame_stride)) == 0:
                    _, frame = capture_obs_and_frame(task_env)
                    writer.write(
                        annotate_frame(
                            frame,
                            [
                                f"ORIGINAL PREFIX {task} ep={episode_id} seed={scene_seed}",
                                f"raw step {idx + 1}/{len(prefix_actions)} -> attach/start_ts={start_ts}",
                            ],
                        )
                    )
        else:
            for action in prefix_actions:
                task_env.take_action(action.tolist())
                task_env.eval_success = False

        start_obs, start_frame = capture_obs_and_frame(task_env)
        start_qpos = current_qpos_from_obs(start_obs)
        raw_start_qpos = raw_actions[min(start_ts, len(raw_actions) - 1)]
        start_linf_to_raw = float(np.max(np.abs(start_qpos - raw_start_qpos)))

        for _ in range(max(0, int(hold_frames))):
            writer.write(
                annotate_frame(
                    start_frame,
                    [
                        f"CLEAN START {task} ep={episode_id} seed={scene_seed}",
                        f"attach/start_ts={start_ts} start_linf_to_raw={start_linf_to_raw:.4f}",
                    ],
                )
            )

        perturb_linf_from_raw_start = None
        perturb_linf_to_corr_qpos = None
        correction_start_linf_to_corr_qpos = None
        perturb_replay_mode = "none"
        if include_perturb_state and perturb_actions is not None and int(perturb_actions.shape[0]) > 0:
            perturb_replay_mode = "simulator_actions"
            if perturb_start_qpos is not None:
                perturb_start_linf_to_clean = float(np.max(np.abs(start_qpos - perturb_start_qpos)))
            else:
                perturb_start_linf_to_clean = None
            for idx, action in enumerate(perturb_actions):
                task_env.take_action(action.tolist())
                task_env.eval_success = False
                obs, frame = capture_obs_and_frame(task_env)
                qpos = current_qpos_from_obs(obs)
                err = qpos - action
                perturb_tracking_linf.append(float(np.max(np.abs(err))))
                perturb_tracking_l2.append(float(np.linalg.norm(err)))
                writer.write(
                    annotate_frame(
                        frame,
                        [
                            f"PERTURB / ERROR ROLLOUT {task} ep={episode_id} seed={scene_seed}",
                            f"step {idx + 1}/{len(perturb_actions)} mode={record.get('sampled_error_mode')}",
                            f"track_linf={perturb_tracking_linf[-1]:.4f} "
                            f"start_linf={perturb_start_linf_to_clean}",
                        ],
                    )
                )

            pert_obs, pert_frame = capture_obs_and_frame(task_env)
            pert_qpos = current_qpos_from_obs(pert_obs)
            perturb_linf_from_raw_start = float(np.max(np.abs(pert_qpos - raw_start_qpos)))
            if corr_qpos is not None:
                perturb_linf_to_corr_qpos = float(np.max(np.abs(pert_qpos - corr_qpos)))
            elif perturb_final_qpos is not None:
                perturb_linf_to_corr_qpos = float(np.max(np.abs(pert_qpos - perturb_final_qpos)))
            for _ in range(max(0, int(hold_frames))):
                writer.write(
                    annotate_frame(
                        pert_frame,
                        [
                            f"ERROR END / CORRECTION START {task} ep={episode_id}",
                            f"linf_from_clean={perturb_linf_from_raw_start:.4f} "
                            f"linf_to_corr_qpos={perturb_linf_to_corr_qpos:.4f}",
                            f"recovery_prefix_len={error_prefix_len}",
                        ],
                    )
                )
        elif include_perturb_state and corr_qpos is not None:
            perturb_replay_mode = "visual_bridge_fallback"
            # Backward compatibility for old npz files. The original perturb
            # rollout actions were not saved, so bridge to saved corr_qpos_raw.
            bridge_steps = int(max(1, int(perturb_bridge_steps)))
            bridge_actions = np.linspace(start_qpos, corr_qpos, bridge_steps + 1, dtype=np.float32)[1:]
            for idx, action in enumerate(bridge_actions):
                task_env.take_action(action.tolist())
                task_env.eval_success = False
                obs, frame = capture_obs_and_frame(task_env)
                qpos = current_qpos_from_obs(obs)
                err = qpos - action
                perturb_tracking_linf.append(float(np.max(np.abs(err))))
                perturb_tracking_l2.append(float(np.linalg.norm(err)))
                writer.write(
                    annotate_frame(
                        frame,
                        [
                            f"PERTURB STATE BRIDGE {task} ep={episode_id} seed={scene_seed}",
                            f"visual step {idx + 1}/{len(bridge_actions)} -> saved corr_qpos_raw",
                            "note: true perturb rollout action was not saved",
                        ],
                    )
                )
            pert_obs, pert_frame = capture_obs_and_frame(task_env)
            pert_qpos = current_qpos_from_obs(pert_obs)
            perturb_linf_from_raw_start = float(np.max(np.abs(pert_qpos - raw_start_qpos)))
            perturb_linf_to_corr_qpos = float(np.max(np.abs(pert_qpos - corr_qpos)))
            for _ in range(max(0, int(hold_frames))):
                writer.write(
                    annotate_frame(
                        pert_frame,
                        [
                            f"ERROR END / CORRECTION START {task} ep={episode_id}",
                            f"linf_from_clean={perturb_linf_from_raw_start:.4f} "
                            f"linf_to_corr_qpos={perturb_linf_to_corr_qpos:.4f}",
                            f"recovery_prefix_len={error_prefix_len}",
                        ],
                    )
                )

        correction_start_obs, _ = capture_obs_and_frame(task_env)
        correction_start_qpos = current_qpos_from_obs(correction_start_obs)
        if corr_qpos is not None:
            correction_start_linf_to_corr_qpos = float(np.max(np.abs(correction_start_qpos - corr_qpos)))

        for idx, action in enumerate(corr_actions):
            task_env.take_action(action.tolist())
            task_env.eval_success = False
            obs, frame = capture_obs_and_frame(task_env)
            qpos = current_qpos_from_obs(obs)
            err = qpos - action
            qpos_trace.append(qpos.astype(float).tolist())
            cmd_trace.append(action.astype(float).tolist())
            tracking_linf.append(float(np.max(np.abs(err))))
            tracking_l2.append(float(np.linalg.norm(err)))
            if idx < error_prefix_len:
                stage_label = "CORRECTION PREFIX"
            else:
                stage_label = "ORIGINAL GT TAIL"
            writer.write(
                annotate_frame(
                    frame,
                    [
                        f"{stage_label} {task} ep={episode_id} seed={scene_seed}",
                        f"step {idx + 1}/{len(corr_actions)} mode={record.get('sampled_error_mode')} "
                        f"phase={record.get('sampled_phase_key')} bin={record.get('sampled_phase_bin_id')}",
                        f"track_linf={tracking_linf[-1]:.4f} track_l2={tracking_l2[-1]:.4f}",
                    ],
                )
            )

    task_env.close_env(clear_cache=False)
    if getattr(task_env, "render_freq", 0) and getattr(task_env, "viewer", None) is not None:
        task_env.viewer.close()

    metrics = {
        "record": record,
        "task": task,
        "task_config": task_config,
        "episode_id": episode_id,
        "scene_seed": scene_seed,
        "start_ts": start_ts,
        "sample_path": str(record["path"]),
        "video_path": str(video_path),
        "raw_hdf5_path": str(hdf5_path),
        "prefix_actions_executed": int(len(prefix_actions)),
        "include_perturb_state": bool(include_perturb_state),
        "perturb_replay_mode": perturb_replay_mode,
        "perturb_state_note": (
            "For new correction npz files, perturb_action_prefix_raw is replayed in the simulator. "
            "For old files without perturb_action_prefix_raw, this script falls back to a visual bridge "
            "from clean start qpos to saved corr_qpos_raw."
        ),
        "perturb_bridge_steps": int(max(0, int(perturb_bridge_steps))) if include_perturb_state else 0,
        "perturb_tracking_linf": perturb_tracking_linf,
        "perturb_tracking_l2": perturb_tracking_l2,
        "perturb_linf_from_raw_start": perturb_linf_from_raw_start,
        "perturb_linf_to_corr_qpos": perturb_linf_to_corr_qpos,
        "correction_start_linf_to_corr_qpos": correction_start_linf_to_corr_qpos,
        "correction_actions_executed": int(len(corr_actions)),
        "error_action_prefix_len": int(error_prefix_len),
        "error_prefix_matches_corr_prefix_linf": error_prefix_matches_corr_prefix_linf,
        "start_linf_to_raw": start_linf_to_raw,
        "max_tracking_linf": float(np.max(tracking_linf)) if tracking_linf else 0.0,
        "mean_tracking_l2": float(np.mean(tracking_l2)) if tracking_l2 else 0.0,
        "tracking_linf": tracking_linf,
        "tracking_l2": tracking_l2,
        "commanded_qpos": cmd_trace,
        "executed_qpos": qpos_trace,
    }
    if corr_qpos is not None:
        metrics["corr_qpos_raw"] = corr_qpos.astype(float).tolist()
    if error_action_prefix is not None:
        metrics["error_action_prefix_raw"] = error_action_prefix.astype(float).tolist()
    if perturb_actions is not None:
        metrics["perturb_action_prefix_raw"] = perturb_actions.astype(float).tolist()
    if perturb_start_qpos is not None:
        metrics["perturb_start_qpos_raw"] = perturb_start_qpos.astype(float).tolist()
    if perturb_final_qpos is not None:
        metrics["perturb_final_qpos_raw"] = perturb_final_qpos.astype(float).tolist()
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    return ReplayResult(
        video_path=str(video_path),
        metrics_path=str(metrics_path),
        max_tracking_linf=metrics["max_tracking_linf"],
        mean_tracking_l2=metrics["mean_tracking_l2"],
        start_linf_to_raw=start_linf_to_raw,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved SmolVLA correction samples in RoboTwin simulator.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--original-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--task", action="append", choices=DEFAULT_TASKS)
    parser.add_argument("--rank", action="append", type=int, default=None)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--sample-path", action="append", default=[])
    parser.add_argument("--select", choices=["top_outliers", "first", "random"], default="top_outliers")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-start-ts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument(
        "--ray-tracing-denoiser",
        type=str,
        default="none",
        help="SAPIEN ray tracing denoiser to use for replay; use 'none' to disable it.",
    )
    parser.add_argument("--record-prefix", action="store_true")
    parser.add_argument("--prefix-frame-stride", type=int, default=10)
    parser.add_argument(
        "--no-perturb-state",
        action="store_true",
        help="Do not visualize the saved perturbed input state before replaying correction actions.",
    )
    parser.add_argument(
        "--perturb-bridge-steps",
        type=int,
        default=16,
        help=(
            "Number of visual bridge actions from clean start qpos to saved corr_qpos_raw. "
            "The true perturb rollout actions were not saved in the correction npz."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if args.output_dir is None:
        output_dir = run_dir / "correction_replay_videos"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(run_dir)
    if not records:
        raise FileNotFoundError(f"No correction_data_manifest_rank*.jsonl found in {run_dir}")
    selected = select_records(args, records)
    if not selected:
        raise RuntimeError("No correction samples matched the requested filters.")

    print(f"Selected {len(selected)} sample(s). Output: {output_dir}")
    summary = []
    for idx, record in enumerate(selected):
        expected_video, expected_metrics = replay_output_paths(
            record,
            output_dir,
            include_perturb_state=not bool(args.no_perturb_state),
        )
        if bool(args.skip_existing) and expected_video.is_file() and expected_metrics.is_file():
            payload = {
                "record": {
                    "task": record.get("task"),
                    "rank": record.get("rank"),
                    "global_step": record.get("global_step"),
                    "episode_id": record.get("episode_id"),
                    "start_ts": record.get("start_ts"),
                    "sampled_error_mode": record.get("sampled_error_mode"),
                    "max_excess": record.get("max_excess"),
                    "max_excess_dim_name": record.get("max_excess_dim_name"),
                },
                "video_path": str(expected_video),
                "metrics_path": str(expected_metrics),
                "skipped_existing": True,
            }
            summary.append(payload)
            print(f"[{idx + 1}/{len(selected)}] skip existing {expected_video}")
            continue
        print(
            f"[{idx + 1}/{len(selected)}] replay "
            f"task={record.get('task')} rank={record.get('rank')} step={record.get('global_step')} "
            f"ep={record.get('episode_id')} start_ts={record.get('start_ts')} "
            f"max_excess={float(record.get('max_excess', 0.0)):.4f}"
        )
        result = replay_record(
            record,
            output_dir=output_dir,
            task_config=args.task_config,
            fps=int(args.fps),
            record_prefix=bool(args.record_prefix),
            prefix_frame_stride=int(args.prefix_frame_stride),
            hold_frames=int(args.hold_frames),
            include_perturb_state=not bool(args.no_perturb_state),
            perturb_bridge_steps=int(args.perturb_bridge_steps),
            ray_tracing_denoiser=str(args.ray_tracing_denoiser),
        )
        payload = {
            "record": {
                "task": record.get("task"),
                "rank": record.get("rank"),
                "global_step": record.get("global_step"),
                "episode_id": record.get("episode_id"),
                "start_ts": record.get("start_ts"),
                "sampled_error_mode": record.get("sampled_error_mode"),
                "max_excess": record.get("max_excess"),
                "max_excess_dim_name": record.get("max_excess_dim_name"),
            },
            "video_path": result.video_path,
            "metrics_path": result.metrics_path,
            "start_linf_to_raw": result.start_linf_to_raw,
            "max_tracking_linf": result.max_tracking_linf,
            "mean_tracking_l2": result.mean_tracking_l2,
        }
        summary.append(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
