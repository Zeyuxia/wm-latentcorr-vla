from __future__ import annotations

import argparse
import json
import os
import signal
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from policy.SmolVLA.replay_correction_samples import (
    FfmpegWriter,
    annotate_frame,
    capture_obs_and_frame,
    class_decorator,
    current_qpos_from_obs,
    load_raw_action_vector,
    load_seed,
    load_task_args,
)


def _as_float32_array(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 14), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def _infer_task_context(raw_data: dict[str, Any]) -> tuple[str, str, Path, int, int, Path]:
    episode_path = Path(str(raw_data.get("episode_path", "") or "")).resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"Missing raw episode file for sim backend: {episode_path}")

    source_task_dir = episode_path.parent
    if source_task_dir.name == "data":
        source_task_dir = source_task_dir.parent
    if not source_task_dir.parent:
        raise FileNotFoundError(f"Unable to infer task directory from episode path: {episode_path}")

    match = re.search(r"episode(\d+)\.hdf5$", episode_path.name)
    if not match:
        raise ValueError(f"Unable to infer episode id from episode path: {episode_path}")
    episode_id = int(match.group(1))

    task_config = source_task_dir.name
    task_name = source_task_dir.parent.name
    seed_path = source_task_dir / "seed.txt"
    if not seed_path.is_file():
        raise FileNotFoundError(f"Missing seed file for sim backend replay: {seed_path}")
    scene_seed = int(load_seed(seed_path, episode_id))
    return task_name, task_config, source_task_dir, episode_id, scene_seed, episode_path


def _save_rgb_frame(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), frame_bgr)


class SimInferenceTimeoutError(RuntimeError):
    pass


class SimInferenceSubprocessError(RuntimeError):
    pass


def _sim_inference_direct(
    curr_image: torch.Tensor,
    raw_data: dict[str, Any],
    device,
    *,
    start_ts: int,
    prefix_action_raw: np.ndarray | None,
    action_chunk_raw: np.ndarray,
    save_dir: str | None = None,
    fps: int = 30,
    hold_frames: int = 1,
) -> torch.Tensor:
    """Replay the current state in RoboTwin simulator and return the final frame."""

    task_name, task_config, source_task_dir, episode_id, scene_seed, episode_path = _infer_task_context(raw_data)
    raw_actions = load_raw_action_vector(episode_path)

    start_ts = int(max(0, start_ts))
    prefix_actions = raw_actions[1 : min(start_ts + 1, len(raw_actions))]
    prefix_action_raw = _as_float32_array(prefix_action_raw)
    action_chunk_raw = _as_float32_array(action_chunk_raw)
    if action_chunk_raw.ndim != 2 or action_chunk_raw.shape[1] != 14:
        raise ValueError(f"Expected sim action_chunk_raw [T,14], got {tuple(action_chunk_raw.shape)}")

    save_path = None if not save_dir else Path(save_dir)
    env_save_path = None if save_path is None else save_path / "env_cache"
    task_args = load_task_args(
        task_name,
        task_config,
        save_path=env_save_path,
        ray_tracing_denoiser="none",
    )

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))

    task_env = class_decorator(task_name)
    video_writer = None
    try:
        task_env.setup_demo(now_ep_num=episode_id, seed=scene_seed, is_test=True, **task_args)
        task_env.check_success = lambda *args, **kwargs: False

        for action in prefix_actions:
            task_env.take_action(np.asarray(action, dtype=np.float32).tolist())
            task_env.eval_success = False
        for action in prefix_action_raw:
            task_env.take_action(np.asarray(action, dtype=np.float32).tolist())
            task_env.eval_success = False

        obs, start_frame = capture_obs_and_frame(task_env)
        start_frame_rgb = np.ascontiguousarray(start_frame.astype(np.uint8))
        final_frame_rgb = start_frame_rgb

        outputs_path = None if save_path is None else save_path / "outputs.mp4"
        runtime_meta_path = None if save_path is None else save_path / "sim_runtime_meta.json"
        if outputs_path is not None:
            outputs_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = FfmpegWriter(
                outputs_path,
                width=int(start_frame_rgb.shape[1]),
                height=int(start_frame_rgb.shape[0]),
                fps=int(fps),
            )
            video_writer.__enter__()

        if save_path is not None:
            _save_rgb_frame(save_path / "frame_000.jpg", start_frame_rgb)
        if video_writer is not None:
            for _ in range(max(0, int(hold_frames))):
                video_writer.write(
                    annotate_frame(
                        start_frame_rgb,
                        [
                            f"SIM BACKEND {task_name} ep={episode_id} seed={scene_seed}",
                            f"current_state start_ts={start_ts} prefix_len={int(prefix_action_raw.shape[0])}",
                            f"action_chunk_len={int(action_chunk_raw.shape[0])}",
                        ],
                    )
                )

        for idx, action in enumerate(action_chunk_raw):
            task_env.take_action(np.asarray(action, dtype=np.float32).tolist())
            task_env.eval_success = False
            obs, frame = capture_obs_and_frame(task_env)
            final_frame_rgb = np.ascontiguousarray(frame.astype(np.uint8))
            if save_path is not None:
                _save_rgb_frame(save_path / f"frame_{idx + 1:03d}.jpg", final_frame_rgb)
            if video_writer is not None:
                video_writer.write(
                    annotate_frame(
                        final_frame_rgb,
                        [
                            f"SIM BACKEND {task_name} ep={episode_id} seed={scene_seed}",
                            f"step {idx + 1}/{int(action_chunk_raw.shape[0])}",
                            f"start_ts={start_ts} prefix_len={int(prefix_action_raw.shape[0])}",
                        ],
                    )
                )

        if video_writer is not None:
            video_writer.__exit__(None, None, None)
            video_writer = None

        if runtime_meta_path is not None:
            runtime_meta = {
                "backend": "sim",
                "task_name": task_name,
                "task_config": task_config,
                "episode_id": int(episode_id),
                "source_episode_path": str(episode_path),
                "source_task_dir": str(source_task_dir),
                "start_ts": int(start_ts),
                "scene_seed": int(scene_seed),
                "prefix_action_len": int(prefix_action_raw.shape[0]),
                "action_chunk_len": int(action_chunk_raw.shape[0]),
                "outputs_mp4": str(outputs_path) if outputs_path is not None else None,
                "final_qpos_raw": current_qpos_from_obs(obs).astype(np.float32).tolist(),
                "final_frame_hw": [int(final_frame_rgb.shape[0]), int(final_frame_rgb.shape[1])],
            }
            with runtime_meta_path.open("w", encoding="utf-8") as f:
                json.dump(runtime_meta, f, indent=2, ensure_ascii=False)

        pred = torch.from_numpy(np.ascontiguousarray(final_frame_rgb)).permute(2, 0, 1).float() / 255.0
        return pred.to(device)
    finally:
        try:
            if video_writer is not None:
                video_writer.__exit__(None, None, None)
        except Exception:
            pass
        try:
            task_env.close_env(clear_cache=False)
        except Exception:
            pass
        try:
            if getattr(task_env, "render_freq", 0) and getattr(task_env, "viewer", None) is not None:
                task_env.viewer.close()
        except Exception:
            pass


def _clean_distributed_env(env: dict[str, str]) -> dict[str, str]:
    cleaned = dict(env)
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_NUM_PROCESSES",
        "ACCELERATE_LOCAL_PROCESS_INDEX",
    ):
        cleaned.pop(key, None)
    return cleaned


def _rank_local_cuda_visible_devices(device) -> str | None:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    try:
        dev = torch.device(device)
    except Exception:
        return None
    if dev.type != "cuda":
        return None
    idx = 0 if dev.index is None else int(dev.index)
    if visible and 0 <= idx < len(visible):
        return visible[idx]
    return str(idx)


def _write_timeout_meta(save_dir: str | None, *, timeout_s: float, work_dir: Path, stdout_path: Path, stderr_path: Path) -> None:
    if not save_dir:
        return
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "backend": "sim",
        "error": "timeout",
        "timeout_s": float(timeout_s),
        "work_dir": str(work_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    with (save_path / "sim_subprocess_error.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _run_sim_subprocess(
    curr_image: torch.Tensor,
    raw_data: dict[str, Any],
    device,
    *,
    start_ts: int,
    prefix_action_raw: np.ndarray | None,
    action_chunk_raw: np.ndarray,
    save_dir: str | None,
    fps: int,
    hold_frames: int,
    timeout_s: float,
) -> torch.Tensor:
    parent_dir = Path(save_dir) if save_dir else Path(tempfile.gettempdir()) / "smolvla_sim_subprocess"
    work_dir = parent_dir / f"_sim_subprocess_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.pkl"
    output_path = work_dir / "output.pt"
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"

    payload = {
        "raw_data": raw_data,
        "start_ts": int(start_ts),
        "prefix_action_raw": _as_float32_array(prefix_action_raw),
        "action_chunk_raw": _as_float32_array(action_chunk_raw),
        "save_dir": save_dir,
        "fps": int(fps),
        "hold_frames": int(hold_frames),
    }
    with input_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    env = _clean_distributed_env(os.environ)
    local_visible = _rank_local_cuda_visible_devices(device)
    if local_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = local_visible
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        "-m",
        "policy.SmolVLA.latentcorr.correction_sim_rollout",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
                cmd,
                cwd="/data/zhenyangfan/RoboTwin",
                env=env,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
        try:
            returncode = proc.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(int(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=5.0)
            except Exception:
                pass
            _write_timeout_meta(save_dir, timeout_s=float(timeout_s), work_dir=work_dir, stdout_path=stdout_path, stderr_path=stderr_path)
            raise SimInferenceTimeoutError(
                f"sim subprocess timed out after {float(timeout_s):.1f}s; work_dir={work_dir}"
            ) from exc

    if int(returncode) != 0:
        err_tail = ""
        try:
            err_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except Exception:
            pass
        raise SimInferenceSubprocessError(
            f"sim subprocess failed with returncode={int(returncode)}; "
            f"work_dir={work_dir}; stderr_tail={err_tail}"
        )
    if not output_path.is_file():
        raise SimInferenceSubprocessError(f"sim subprocess did not write output tensor: {output_path}")

    payload_out = torch.load(output_path, map_location="cpu", weights_only=False)
    pred = payload_out["pred"] if isinstance(payload_out, dict) else payload_out
    pred = torch.as_tensor(pred, dtype=torch.float32)
    if bool(save_dir):
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass
    return pred.to(device)


def sim_inference(
    curr_image: torch.Tensor,
    raw_data: dict[str, Any],
    device,
    *,
    start_ts: int,
    prefix_action_raw: np.ndarray | None,
    action_chunk_raw: np.ndarray,
    save_dir: str | None = None,
    fps: int = 30,
    hold_frames: int = 1,
    use_subprocess: bool = True,
    timeout_s: float = 180.0,
) -> torch.Tensor:
    """Replay with a subprocess guard so native simulator hangs cannot stall explore."""
    if bool(use_subprocess):
        return _run_sim_subprocess(
            curr_image,
            raw_data,
            device,
            start_ts=int(start_ts),
            prefix_action_raw=prefix_action_raw,
            action_chunk_raw=action_chunk_raw,
            save_dir=save_dir,
            fps=int(fps),
            hold_frames=int(hold_frames),
            timeout_s=float(timeout_s),
        )
    return _sim_inference_direct(
        curr_image,
        raw_data,
        device,
        start_ts=int(start_ts),
        prefix_action_raw=prefix_action_raw,
        action_chunk_raw=action_chunk_raw,
        save_dir=save_dir,
        fps=int(fps),
        hold_frames=int(hold_frames),
    )


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        with open(args.input, "rb") as f:
            payload = pickle.load(f)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pred = _sim_inference_direct(
            torch.empty(0),
            payload["raw_data"],
            device,
            start_ts=int(payload["start_ts"]),
            prefix_action_raw=payload.get("prefix_action_raw"),
            action_chunk_raw=payload["action_chunk_raw"],
            save_dir=payload.get("save_dir"),
            fps=int(payload.get("fps", 30)),
            hold_frames=int(payload.get("hold_frames", 1)),
        )
        torch.save({"pred": pred.detach().cpu()}, args.output)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    _cli()
