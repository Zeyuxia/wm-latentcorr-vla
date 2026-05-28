from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch


RAW_CAMERA_NAMES = ("front_camera", "head_camera", "left_camera", "right_camera")


def _tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_chw_to_rgb_u8(image) -> np.ndarray:
    arr = _tensor_to_numpy(image).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW/HWC image, got {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if float(np.nanmax(arr)) <= 2.0:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(np.round(arr), 0, 255).astype(np.uint8))


def _decode_jpeg_rgb(encoded) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode raw RoboTwin rgb frame")
    # RoboTwin raw JPEGs are legacy-encoded through OpenCV from simulator RGB
    # arrays.  A direct cv2 decode gives the numeric RGB values expected by the
    # SmolVLA/LeRobot processing path; do not color-convert here.
    return frame


def _encode_rgb_jpeg(image: np.ndarray) -> np.bytes_:
    image = np.ascontiguousarray(image.astype(np.uint8))
    # Match the legacy RoboTwin raw encoding convention described above.
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode correction frame")
    return np.bytes_(buf.tobytes())


def _safe_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        return _safe_jsonable(arr)
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    return str(value)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_safe_jsonable(payload), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(_safe_jsonable(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_raw_data(raw_data_dir, episode_id):
    path = os.path.join(raw_data_dir, f"episode{episode_id}.hdf5")
    with h5py.File(path, "r") as f:
        frame0 = bytes(f["observation/head_camera/rgb"][0])
        img0 = cv2.imdecode(np.frombuffer(frame0, np.uint8), cv2.IMREAD_COLOR)
        h_native, w_native = img0.shape[:2]
        return {
            "episode_path": path,
            "left_endpose": f["endpose/left_endpose"][()].astype(np.float32),
            "right_endpose": f["endpose/right_endpose"][()].astype(np.float32),
            "left_gripper": f["endpose/left_gripper"][()].astype(np.float32),
            "right_gripper": f["endpose/right_gripper"][()].astype(np.float32),
            "gt_left_arm": (
                f["joint_action/left_arm"][()].astype(np.float32)
                if ("joint_action" in f and "left_arm" in f["joint_action"])
                else None
            ),
            "gt_right_arm": (
                f["joint_action/right_arm"][()].astype(np.float32)
                if ("joint_action" in f and "right_arm" in f["joint_action"])
                else None
            ),
            "intrinsic_cv": f["observation/head_camera/intrinsic_cv"][0].astype(np.float32),
            "extrinsic_cv": f["observation/head_camera/extrinsic_cv"][0].astype(np.float32),
            "native_resolution": (h_native, w_native),
        }


def init_export_episode_id(export_dir):
    if not os.path.isdir(export_dir):
        return 0
    max_id = -1
    for fn in os.listdir(export_dir):
        match = re.match(r"episode_?(\d+)\.hdf5$", fn) or re.match(r"sample_?(\d+)\.npz$", fn)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def export_correction_sample_as_episode(
    export_dir,
    episode_id,
    cam_names,
    corr_image,
    corr_qpos_norm,
    corr_action_norm,
    corr_is_pad,
    norm_stats,
):
    """Legacy tiny correction exporter kept for stage1 debug compatibility."""
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"episode_{episode_id}.hdf5")

    img = corr_image.detach().cpu().numpy()
    qn = corr_qpos_norm.detach().cpu().numpy()
    an = corr_action_norm.detach().cpu().numpy()
    pad = corr_is_pad.detach().cpu().numpy().astype(bool)

    valid_len = int(np.sum(~pad))
    valid_len = max(1, valid_len)

    qpos_raw = (qn * norm_stats["qpos_std"] + norm_stats["qpos_mean"]).astype(np.float32)
    action_raw = (an * norm_stats["action_std"] + norm_stats["action_mean"]).astype(np.float32)
    action_raw = action_raw[:valid_len]
    qpos_seq = np.repeat(qpos_raw[None, :], valid_len, axis=0).astype(np.float32)

    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    img_hwc = np.transpose(img_u8, (0, 2, 3, 1))

    with h5py.File(path, "w") as f:
        f.create_dataset("/action", data=action_raw, compression="gzip", compression_opts=1)
        f.create_dataset("/observations/qpos", data=qpos_seq, compression="gzip", compression_opts=1)
        image_group = f.create_group("/observations/images")
        for idx, cam_name in enumerate(cam_names):
            frames = np.repeat(img_hwc[idx][None, ...], valid_len, axis=0)
            image_group.create_dataset(cam_name, data=frames, compression="gzip", compression_opts=1)
    return path


def export_correction_state_action_sample(
    *,
    export_root: str | Path,
    task_name: str,
    task_config: str,
    rank: int,
    sample_id: int,
    source_raw_data_dir: str | Path,
    source_episode_id: int,
    start_ts: int,
    corr_action_raw: np.ndarray,
    corr_qpos_raw: np.ndarray,
    corr_image,
    corr_meta: dict[str, Any],
    corr_action_norm: np.ndarray | None = None,
    corr_qpos_norm: np.ndarray | None = None,
) -> dict[str, Any]:
    """Export the minimal correction target: perturbed state plus correction actions."""
    task_name = str(task_name)
    task_config = str(task_config)
    rank = int(rank)
    sample_id = int(sample_id)
    source_episode_id = int(source_episode_id)
    start_ts = int(start_ts)
    export_root = Path(export_root).resolve()
    shard_dir = export_root / task_name / f"{task_config}_state_action_shards" / f"rank{rank:02d}"
    data_dir = shard_dir / "correction_data"
    instr_dir = shard_dir / "instructions"
    meta_dir = shard_dir / "metadata"
    for path in (data_dir, instr_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_dir / "correction_manifest.jsonl"
    source_raw_data_dir = Path(source_raw_data_dir).resolve()
    source_task_dir = _resolve_raw_task_dir(source_raw_data_dir)
    instruction_payload = _load_instruction_payload(source_task_dir, source_episode_id)
    instruction_path = instr_dir / f"sample{sample_id}.json"
    _write_json(instruction_path, instruction_payload)
    dataset_info_path = shard_dir / "_dataset_info.json"
    _write_json(
        dataset_info_path,
        {
            "version": 1,
            "format": "state_action",
            "task": task_name,
            "task_config": task_config,
            "rank": int(rank),
            "description": "Minimal correction dataset: perturbed observation state plus correction action chunk.",
            "directories": {
                "correction_data": "sampleN.npz files",
                "instructions": "sampleN.json copied from source RoboTwin episode instructions",
                "metadata": "sampleN.json source/correction metadata",
            },
            "npz_schema": {
                "corr_image_chw_uint8": "uint8 [3,H,W], perturbed head-camera observation in SmolVLA/RoboTwin channel semantics",
                "corr_qpos_raw": "float32 [14], perturbed qpos/state",
                "corr_action_chunk_raw": "float32 [T,14], correction action labels only; perturb prefix is not a training label",
                "corr_action_chunk_norm": "optional float32 [T,14], normalized correction action labels",
                "corr_qpos_norm": "optional float32 [14], normalized perturbed qpos/state",
                "error_action_prefix_raw": "optional float32 [P,14], debug-only perturb/error actions",
                "perturb_action_prefix_raw": "optional float32 [P,14], debug-only perturb actions",
                "perturb_start_qpos_raw": "optional float32 [14], debug-only pre-perturb state",
                "perturb_final_qpos_raw": "optional float32 [14], debug-only final perturbed state",
            },
            "manifest": str(manifest_path),
        },
    )

    corr_meta = corr_meta if isinstance(corr_meta, dict) else {}
    recover_eval_last = corr_meta.get("recover_eval_last") if isinstance(corr_meta.get("recover_eval_last"), dict) else {}
    prefix_len = int(corr_meta.get("correction_prefix_len", 0) or 0)
    corr_action_raw = np.asarray(corr_action_raw, dtype=np.float32)
    if corr_action_raw.ndim != 2 or corr_action_raw.shape[1] != 14:
        raise ValueError(f"Expected corr_action_raw [T,14], got {corr_action_raw.shape}")
    prefix_len = int(np.clip(prefix_len, 1, corr_action_raw.shape[0]))
    corr_action_raw = corr_action_raw[:prefix_len].astype(np.float32)
    corr_qpos_raw = np.asarray(corr_qpos_raw, dtype=np.float32).reshape(14,)
    corr_image_chw_uint8 = _image_chw_to_rgb_u8(corr_image)
    if corr_image_chw_uint8.ndim == 3:
        corr_image_chw_uint8 = np.transpose(corr_image_chw_uint8, (2, 0, 1))
    arrays: dict[str, np.ndarray] = {
        "corr_image_chw_uint8": np.ascontiguousarray(corr_image_chw_uint8.astype(np.uint8)),
        "corr_qpos_raw": corr_qpos_raw,
        "corr_action_chunk_raw": corr_action_raw,
    }
    if corr_action_norm is not None:
        corr_action_norm = np.asarray(corr_action_norm, dtype=np.float32)
        arrays["corr_action_chunk_norm"] = corr_action_norm[:prefix_len]
    if corr_qpos_norm is not None:
        arrays["corr_qpos_norm"] = np.asarray(corr_qpos_norm, dtype=np.float32).reshape(14,)
    for key in (
        "error_action_prefix_raw",
        "perturb_action_prefix_raw",
        "perturb_start_qpos_raw",
        "perturb_final_qpos_raw",
    ):
        value = corr_meta.get(key)
        if value is not None:
            arrays[key] = np.asarray(value, dtype=np.float32)

    sample_path = data_dir / f"sample{sample_id}.npz"
    np.savez_compressed(sample_path, **arrays)
    metadata_path = meta_dir / f"sample{sample_id}.json"
    record = {
        "format": "state_action",
        "path": str(sample_path),
        "metadata_path": str(metadata_path),
        "task": task_name,
        "task_config": task_config,
        "rank": int(rank),
        "sample_id": int(sample_id),
        "source_episode_id": int(source_episode_id),
        "source_raw_data_dir": str(source_raw_data_dir),
        "source_task_config": source_task_dir.name,
        "source_hdf5": str(source_raw_data_dir / f"episode{source_episode_id}.hdf5"),
        "instruction_path": str(instruction_path),
        "instruction": _first_seen_instruction(instruction_payload),
        "start_ts": int(start_ts),
        "correction_prefix_len": int(prefix_len),
        "sampled_phase_key": corr_meta.get("sampled_phase_key"),
        "sampled_phase_bin_id": corr_meta.get("sampled_phase_bin_id"),
        "sampled_phase_instance_idx": corr_meta.get("sampled_phase_instance_idx"),
        "sampled_error_mode": corr_meta.get("sampled_error_mode"),
        "forced_error_mode": corr_meta.get("forced_error_mode"),
        "sampled_active_arm_pattern": corr_meta.get("sampled_active_arm_pattern"),
        "forced_dir_bin_id": corr_meta.get("forced_dir_bin_id"),
        "forced_mag_bin_id": corr_meta.get("forced_mag_bin_id"),
        "perturb_action_prefix_len": corr_meta.get("perturb_action_prefix_len"),
        "perturb_delta_linf": corr_meta.get("perturb_delta_linf"),
        "perturb_delta_l2": corr_meta.get("perturb_delta_l2"),
        "recover_eval_recoverable": recover_eval_last.get("recoverable"),
        "recover_eval_failed_thresholds": recover_eval_last.get("failed_thresholds"),
        "recover_eval_metrics": recover_eval_last.get("metrics"),
    }
    sample_metadata = {
        **record,
        "instruction_payload": instruction_payload,
        "npz_keys": sorted(arrays.keys()),
        "npz_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "corr_meta": corr_meta,
        "source_start_ts": int(start_ts),
        "source_attach_idx": int(corr_meta.get("attach_idx", start_ts) or start_ts),
        "deviation_idx": int(corr_meta.get("deviation_idx", start_ts) or start_ts),
        "perturb_prefix_len": int(corr_meta.get("perturb_action_prefix_len", 0) or 0),
        "correction_prefix_len": int(prefix_len),
        "note": "Only corr_action_chunk_* should be used as action labels. Perturb/error prefixes are debug provenance.",
    }
    _write_json(metadata_path, sample_metadata)
    _append_jsonl(manifest_path, record)
    return record


def _resolve_raw_task_dir(raw_data_dir: str | Path) -> Path:
    raw_data_dir = Path(raw_data_dir).resolve()
    if raw_data_dir.name == "data":
        return raw_data_dir.parent
    return raw_data_dir


def _copy_instruction(source_task_dir: Path, source_episode_id: int, dst_path: Path) -> None:
    src = source_task_dir / "instructions" / f"episode{int(source_episode_id)}.json"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst_path)
        return
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump({"seen": ["Do the task."], "unseen": []}, f, indent=2, ensure_ascii=False)


def _load_instruction_payload(source_task_dir: Path, source_episode_id: int) -> dict[str, Any]:
    src = source_task_dir / "instructions" / f"episode{int(source_episode_id)}.json"
    if src.is_file():
        try:
            with src.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {"seen": ["Do the task."], "unseen": []}


def _first_seen_instruction(payload: dict[str, Any]) -> str:
    seen = payload.get("seen", []) if isinstance(payload, dict) else []
    if isinstance(seen, list) and seen:
        return str(seen[0])
    return "Do the task."


def _write_video_from_encoded_head_frames(encoded_frames: list[np.bytes_], out_path: Path, fps: int = 30) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not encoded_frames:
        return
    first = _decode_jpeg_rgb(encoded_frames[0])
    height, width = first.shape[:2]
    frames = [first]
    for item in encoded_frames[1:]:
        frame = _decode_jpeg_rgb(item)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    if _write_h264_video_from_frames(frames, out_path, fps=fps):
        return
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.ascontiguousarray(frame.astype(np.uint8)), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_h264_video_from_frames(frames: list[np.ndarray], out_path: Path, fps: int = 30) -> bool:
    if not frames:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    height, width = frames[0].shape[:2]
    if width % 2 != 0 or height % 2 != 0:
        width = int(width - (width % 2))
        height = int(height - (height % 2))
        if width <= 0 or height <= 0:
            return False
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_bgr = cv2.cvtColor(np.ascontiguousarray(frame.astype(np.uint8)), cv2.COLOR_RGB2BGR)
            proc.stdin.write(frame_bgr.tobytes())
        proc.stdin.close()
        return int(proc.wait()) == 0 and out_path.is_file() and out_path.stat().st_size > 0
    except Exception:
        try:
            if "proc" in locals() and proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        return False


def _resolve_debug_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = Path.cwd() / path
        if candidate.is_file():
            return candidate
    return None


def _read_video_rgb_frames(video_path: Path | None, target_hw: tuple[int, int] | None = None) -> list[np.ndarray]:
    if video_path is None or not video_path.is_file():
        return []
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_hw is not None:
                h, w = int(target_hw[0]), int(target_hw[1])
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(frame.astype(np.uint8)))
    finally:
        cap.release()
    return frames


def _read_clean_rollout_rgb_frames(video_record: dict[str, Any] | None, target_hw: tuple[int, int] | None = None) -> list[np.ndarray]:
    """Read clean world-model frames without the composite action/projection panel.

    EVAC debug videos write both per-frame JPEGs and an outputs.mp4.  The mp4 is
    a visualization that includes gripper/action projection panels, so it must
    never be used as training observation.  The JPEG frames are the clean rollout
    observations.
    """
    if not isinstance(video_record, dict):
        return []
    path_value = video_record.get("path")
    video_path = _resolve_debug_path(path_value if isinstance(path_value, str) else None)
    if video_path is None:
        return []
    frame_dir = video_path.parent
    frame_paths = sorted(frame_dir.glob("frame_*.jpg")) + sorted(frame_dir.glob("frame_*.png"))
    frames: list[np.ndarray] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_hw is not None:
            h, w = int(target_hw[0]), int(target_hw[1])
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(np.ascontiguousarray(frame.astype(np.uint8)))
    return frames


def _read_rollout_debug_frames(video_record: dict[str, Any] | None, target_hw: tuple[int, int] | None = None) -> list[np.ndarray]:
    clean_frames = _read_clean_rollout_rgb_frames(video_record, target_hw=target_hw)
    if clean_frames:
        return clean_frames
    if not isinstance(video_record, dict):
        return []
    path_value = video_record.get("path")
    video_path = _resolve_debug_path(path_value if isinstance(path_value, str) else None)
    return _read_video_rgb_frames(video_path, target_hw=target_hw)


def _label_video_segment(frames: list[np.ndarray], label: str) -> list[np.ndarray]:
    if not frames:
        return []
    out: list[np.ndarray] = []
    label = str(label)
    for frame in frames:
        canvas = np.ascontiguousarray(frame.copy())
        h, w = canvas.shape[:2]
        bar_h = max(20, int(round(float(h) * 0.09)))
        cv2.rectangle(canvas, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
        scale = max(0.35, min(0.55, float(w) / 640.0))
        cv2.putText(
            canvas,
            label,
            (6, max(14, bar_h - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out.append(canvas)
    return out


def _drop_duplicate_boundary_frame(
    prev_frames: list[np.ndarray],
    next_frames: list[np.ndarray],
    *,
    mae_threshold: float = 3.0,
) -> list[np.ndarray]:
    if not prev_frames or not next_frames:
        return next_frames
    prev = np.asarray(prev_frames[-1], dtype=np.float32)
    nxt = np.asarray(next_frames[0], dtype=np.float32)
    if prev.shape != nxt.shape:
        return next_frames
    if float(np.mean(np.abs(prev - nxt))) <= float(mae_threshold):
        return next_frames[1:]
    return next_frames


def _video_frames_include_start_frame(
    start_frame: np.ndarray,
    video_frames: list[np.ndarray],
    *,
    mae_threshold: float = 3.0,
) -> bool:
    if not video_frames:
        return False
    start = np.asarray(start_frame, dtype=np.float32)
    first = np.asarray(video_frames[0], dtype=np.float32)
    if start.shape != first.shape:
        return False
    return float(np.mean(np.abs(start - first))) <= float(mae_threshold)


def _record_action_count(video_record: dict[str, Any] | None) -> int | None:
    if not isinstance(video_record, dict):
        return None
    for key in ("num_actions", "action_chunk_len", "n_valid"):
        value = video_record.get(key)
        if value is not None:
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count >= 0:
                return count

    meta_value = video_record.get("runtime_meta_path")
    meta_path = _resolve_debug_path(meta_value if isinstance(meta_value, str) else None)
    if meta_path is None or not meta_path.is_file():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    for key in ("num_actions", "action_chunk_len", "n_valid"):
        value = meta.get(key)
        if value is not None:
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count >= 0:
                return count
    summary = meta.get("action_summary")
    if isinstance(summary, dict):
        shape = summary.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            try:
                count = int(shape[0])
            except (TypeError, ValueError):
                count = -1
            if count >= 0:
                return count
    return None


def _rollout_frames_include_start_frame(
    video_record: dict[str, Any] | None,
    start_frame: np.ndarray,
    video_frames: list[np.ndarray],
    *,
    mae_threshold: float = 3.0,
) -> bool:
    action_count = _record_action_count(video_record)
    if action_count is not None:
        if len(video_frames) == action_count + 1:
            return True
        if len(video_frames) == action_count:
            return False
    return _video_frames_include_start_frame(
        start_frame,
        video_frames,
        mae_threshold=mae_threshold,
    )


def _encoded_to_frames(encoded_frames: list[np.bytes_], target_hw: tuple[int, int] | None = None) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for item in encoded_frames:
        frame = _decode_jpeg_rgb(item)
        if target_hw is not None and frame.shape[:2] != (int(target_hw[0]), int(target_hw[1])):
            frame = cv2.resize(frame, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return frames


def _write_video_from_frames(frames: list[np.ndarray], out_path: Path, fps: int = 30) -> None:
    if not frames:
        return
    encoded = [_encode_rgb_jpeg(frame) for frame in frames]
    _write_video_from_encoded_head_frames(encoded, out_path, fps=fps)


def _unique_debug_backends(corr_meta: dict[str, Any]) -> list[str]:
    backends: list[str] = []
    main_backend = str(corr_meta.get("world_model_backend", "") or "").strip().lower()
    if main_backend in {"evac", "cosmos", "sim"}:
        backends.append(main_backend)
    for key in (
        "world_model_rollout_videos",
        "world_model_compare",
        "world_model_compare_correction_videos",
    ):
        for record in corr_meta.get(key, []) or []:
            if not isinstance(record, dict):
                continue
            backend = str(record.get("backend", "") or "").strip().lower()
            if backend in {"evac", "cosmos", "sim"} and backend not in backends:
                backends.append(backend)
    return backends


def _backend_rollout_records(corr_meta: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    backend = str(backend).strip().lower()
    if backend == str(corr_meta.get("world_model_backend", "") or "").strip().lower():
        main_records = [
            record
            for record in (corr_meta.get("world_model_rollout_videos", []) or [])
            if isinstance(record, dict)
        ]
        if main_records:
            return sorted(main_records, key=lambda item: int(item.get("step", 0)))
    compare_records = [
        record
        for record in (corr_meta.get("world_model_compare", []) or [])
        if isinstance(record, dict) and str(record.get("backend", "")).strip().lower() == backend
    ]
    if compare_records:
        return sorted(compare_records, key=lambda item: int(item.get("step", 0)))
    return []


def _backend_correction_record(corr_meta: dict[str, Any], backend: str) -> dict[str, Any] | None:
    backend = str(backend).strip().lower()
    if backend == str(corr_meta.get("world_model_backend", "") or "").strip().lower():
        record = corr_meta.get("world_model_correction_video") or corr_meta.get("evac_correction_video")
        if isinstance(record, dict):
            return record
    for record in corr_meta.get("world_model_compare_correction_videos", []) or []:
        if isinstance(record, dict) and str(record.get("backend", "")).strip().lower() == backend:
            return record
    return None


def _build_debug_correction_prefix_frames(
    *,
    corr_image_rgb: np.ndarray,
    num_corr_states: int,
    corr_video_record: dict[str, Any] | None,
    target_hw: tuple[int, int] | None,
) -> tuple[list[np.ndarray], bool]:
    if num_corr_states <= 0:
        return [], False
    base = np.ascontiguousarray(corr_image_rgb.astype(np.uint8))
    if target_hw is not None and base.shape[:2] != (int(target_hw[0]), int(target_hw[1])):
        base = cv2.resize(base, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
    video_frames = _read_clean_rollout_rgb_frames(corr_video_record, target_hw=target_hw)
    used_rollout = bool(video_frames)
    if used_rollout:
        if _rollout_frames_include_start_frame(corr_video_record, base, video_frames):
            frames = list(video_frames[:num_corr_states])
        else:
            frames = [base] + list(video_frames[: max(0, num_corr_states - 1)])
        if len(frames) < num_corr_states:
            frames.extend([frames[-1].copy()] * (num_corr_states - len(frames)))
        return frames[:num_corr_states], True

    frames = [base]
    if len(frames) < num_corr_states:
        frames.extend([frames[-1].copy()] * (num_corr_states - len(frames)))
    return frames[:num_corr_states], False


def _fk_endpose_from_action_sequence(actions: np.ndarray, fk) -> tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for row in np.asarray(actions, dtype=np.float32):
        fr = fk.forward(row[0:6], row[7:13])
        left.append(np.concatenate([fr["left"][0], fr["left"][1]], axis=0).astype(np.float32))
        right.append(np.concatenate([fr["right"][0], fr["right"][1]], axis=0).astype(np.float32))
    return np.stack(left, axis=0), np.stack(right, axis=0)


def _make_corr_camera_frames(
    *,
    source_file: h5py.File,
    camera_name: str,
    source_start_idx: int,
    num_corr_states: int,
    corr_image_rgb: np.ndarray,
    correction_video_frames: list[np.ndarray] | None = None,
    correction_video_record: dict[str, Any] | None = None,
) -> list[np.bytes_]:
    source_rgb = source_file[f"observation/{camera_name}/rgb"]
    start_frame = source_rgb[int(np.clip(source_start_idx, 0, len(source_rgb) - 1))]
    if num_corr_states <= 0:
        return []
    if camera_name == "head_camera":
        source_start = _decode_jpeg_rgb(start_frame)
        target_hw = source_start.shape[:2]
        if corr_image_rgb.shape[:2] != target_hw:
            corr_image_rgb = cv2.resize(
                corr_image_rgb,
                (int(target_hw[1]), int(target_hw[0])),
                interpolation=cv2.INTER_AREA,
            )
        frames = [_encode_rgb_jpeg(corr_image_rgb)]
        video_frames = list(correction_video_frames or [])
        if _rollout_frames_include_start_frame(correction_video_record, corr_image_rgb, video_frames):
            # Some backends, e.g. simulator replay, include the perturbed start
            # frame.  Keep corr_image_rgb as the authoritative start frame.
            video_frames = video_frames[1:]
        for frame in video_frames[: max(0, num_corr_states - 1)]:
            if frame.shape[:2] != target_hw:
                frame = cv2.resize(
                    frame,
                    (int(target_hw[1]), int(target_hw[0])),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(_encode_rgb_jpeg(frame))
        if len(frames) < num_corr_states:
            frames.extend([frames[-1]] * (num_corr_states - len(frames)))
        return frames[:num_corr_states]
    return [np.bytes_(bytes(start_frame))] * num_corr_states


def export_correction_raw_episode(
    *,
    export_root: str | Path,
    task_name: str,
    task_config: str,
    rank: int,
    episode_id: int,
    source_raw_data_dir: str | Path,
    source_episode_id: int,
    start_ts: int,
    attach_idx: int,
    corr_action_raw: np.ndarray,
    corr_qpos_raw: np.ndarray,
    corr_image,
    corr_meta: dict[str, Any],
    fk=None,
    fps: int = 30,
    write_debug_video: bool = True,
    debug_video_dir: str | Path | None = None,
) -> dict[str, Any]:
    task_name = str(task_name)
    task_config = str(task_config)
    rank = int(rank)
    episode_id = int(episode_id)
    source_episode_id = int(source_episode_id)
    start_ts = int(start_ts)

    export_root = Path(export_root).resolve()
    shard_dir = export_root / task_name / f"{task_config}_shards" / f"rank{rank:02d}"
    data_dir = shard_dir / "data"
    video_dir = shard_dir / "video"
    instr_dir = shard_dir / "instructions"
    traj_dir = shard_dir / "_traj_data"
    manifest_path = shard_dir / "correction_manifest.jsonl"
    for path in (data_dir, video_dir, instr_dir, traj_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_raw_data_dir = Path(source_raw_data_dir).resolve()
    source_task_dir = _resolve_raw_task_dir(source_raw_data_dir)
    source_hdf5 = source_raw_data_dir / f"episode{source_episode_id}.hdf5"
    if not source_hdf5.is_file():
        raise FileNotFoundError(f"Missing source raw episode: {source_hdf5}")

    corr_meta = corr_meta if isinstance(corr_meta, dict) else {}
    recover_eval_last = corr_meta.get("recover_eval_last") if isinstance(corr_meta.get("recover_eval_last"), dict) else {}
    prefix_len = int(corr_meta.get("correction_prefix_len", 0) or 0)
    corr_action_raw = np.asarray(corr_action_raw, dtype=np.float32)
    if corr_action_raw.ndim != 2 or corr_action_raw.shape[1] != 14:
        raise ValueError(f"Expected corr_action_raw [T,14], got {corr_action_raw.shape}")
    prefix_len = int(np.clip(prefix_len, 1, corr_action_raw.shape[0]))
    corr_actions = corr_action_raw[:prefix_len].astype(np.float32)
    corr_qpos_raw = np.asarray(corr_qpos_raw, dtype=np.float32).reshape(14,)
    corr_image_rgb = _image_chw_to_rgb_u8(corr_image)
    corr_video_record = corr_meta.get("world_model_correction_video") or corr_meta.get("evac_correction_video")
    correction_video_frames: list[np.ndarray] = []

    with h5py.File(source_hdf5, "r") as src:
        num_source_steps = int(src["joint_action/vector"].shape[0])
        attach_idx = int(np.clip(int(attach_idx), 0, max(0, num_source_steps - 1)))
        if attach_idx < start_ts:
            attach_idx = int(start_ts)

        tail_states = src["joint_action/vector"][attach_idx:][()].astype(np.float32)
        state_vector = np.concatenate([corr_qpos_raw[None, :], corr_actions, tail_states], axis=0).astype(np.float32)

        if fk is not None:
            corr_state_left_ep, corr_state_right_ep = _fk_endpose_from_action_sequence(state_vector[: prefix_len + 1], fk)
        else:
            corr_state_left_ep = np.repeat(
                src["endpose/left_endpose"][start_ts:start_ts + 1][()].astype(np.float32),
                prefix_len + 1,
                axis=0,
            )
            corr_state_right_ep = np.repeat(
                src["endpose/right_endpose"][start_ts:start_ts + 1][()].astype(np.float32),
                prefix_len + 1,
                axis=0,
            )

        left_endpose = np.concatenate(
            [corr_state_left_ep, src["endpose/left_endpose"][attach_idx:][()].astype(np.float32)],
            axis=0,
        )
        right_endpose = np.concatenate(
            [corr_state_right_ep, src["endpose/right_endpose"][attach_idx:][()].astype(np.float32)],
            axis=0,
        )
        left_gripper = np.concatenate(
            [state_vector[: prefix_len + 1, 6], src["endpose/left_gripper"][attach_idx:][()].astype(np.float32)],
            axis=0,
        )
        right_gripper = np.concatenate(
            [state_vector[: prefix_len + 1, 13], src["endpose/right_gripper"][attach_idx:][()].astype(np.float32)],
            axis=0,
        )

        out_hdf5 = data_dir / f"episode{episode_id}.hdf5"
        with h5py.File(out_hdf5, "w") as dst:
            deviation_idx = int(corr_meta.get("deviation_idx", start_ts) or start_ts)
            dst.attrs["source_episode_id"] = int(source_episode_id)
            dst.attrs["source_start_ts"] = int(start_ts)
            dst.attrs["start_idx"] = int(start_ts)
            dst.attrs["source_attach_idx"] = int(attach_idx)
            dst.attrs["attach_idx"] = int(attach_idx)
            dst.attrs["deviation_idx"] = int(deviation_idx)
            dst.attrs["correction_prefix_len"] = int(prefix_len)

            endpose_grp = dst.create_group("endpose")
            endpose_grp.create_dataset("left_endpose", data=left_endpose.astype(np.float64))
            endpose_grp.create_dataset("right_endpose", data=right_endpose.astype(np.float64))
            endpose_grp.create_dataset("left_gripper", data=left_gripper.astype(np.float64))
            endpose_grp.create_dataset("right_gripper", data=right_gripper.astype(np.float64))

            joint_grp = dst.create_group("joint_action")
            joint_grp.create_dataset("left_arm", data=state_vector[:, 0:6].astype(np.float64))
            joint_grp.create_dataset("left_gripper", data=state_vector[:, 6].astype(np.float64))
            joint_grp.create_dataset("right_arm", data=state_vector[:, 7:13].astype(np.float64))
            joint_grp.create_dataset("right_gripper", data=state_vector[:, 13].astype(np.float64))
            joint_grp.create_dataset("vector", data=state_vector.astype(np.float64))

            obs_grp = dst.create_group("observation")
            head_encoded_for_video: list[np.bytes_] = []
            for camera_name in RAW_CAMERA_NAMES:
                src_cam = src[f"observation/{camera_name}"]
                cam_grp = obs_grp.create_group(camera_name)
                if camera_name == "head_camera" and not correction_video_frames:
                    source_rgb = src_cam["rgb"]
                    source_start_frame = source_rgb[int(np.clip(start_ts, 0, len(source_rgb) - 1))]
                    source_start_hw = _decode_jpeg_rgb(source_start_frame).shape[:2]
                    correction_video_frames = _read_clean_rollout_rgb_frames(
                        corr_video_record if isinstance(corr_video_record, dict) else None,
                        target_hw=source_start_hw,
                    )
                corr_rgb = _make_corr_camera_frames(
                    source_file=src,
                    camera_name=camera_name,
                    source_start_idx=start_ts,
                    num_corr_states=prefix_len + 1,
                    corr_image_rgb=corr_image_rgb,
                    correction_video_frames=correction_video_frames,
                    correction_video_record=corr_video_record if isinstance(corr_video_record, dict) else None,
                )
                tail_rgb = [np.bytes_(bytes(item)) for item in src_cam["rgb"][attach_idx:]]
                rgb_frames = corr_rgb + tail_rgb
                max_len = max(len(bytes(item)) for item in rgb_frames) if rgb_frames else 1
                rgb_arr = np.asarray(rgb_frames, dtype=f"S{max_len}")
                cam_grp.create_dataset("rgb", data=rgb_arr)
                for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                    src_data = src_cam[key]
                    prefix = np.repeat(
                        src_data[
                            int(np.clip(start_ts, 0, len(src_data) - 1)):
                            int(np.clip(start_ts, 0, len(src_data) - 1)) + 1
                        ][()],
                        prefix_len + 1,
                        axis=0,
                    )
                    tail = src_data[attach_idx:][()]
                    cam_grp.create_dataset(key, data=np.concatenate([prefix, tail], axis=0).astype(src_data.dtype))
                if camera_name == "head_camera":
                    head_encoded_for_video = rgb_frames

            dst.create_dataset("pointcloud", data=np.zeros((state_vector.shape[0], 0), dtype=np.float64))

    out_video = video_dir / f"episode{episode_id}.mp4"
    _write_video_from_encoded_head_frames(head_encoded_for_video, out_video, fps=fps)
    _copy_instruction(source_task_dir, source_episode_id, instr_dir / f"episode{episode_id}.json")

    with (traj_dir / f"episode{episode_id}.pkl").open("wb") as f:
        pickle.dump(
            {
                "left_joint_path": [{"status": "CorrectionExport", "position": state_vector[:, 0:6].astype(np.float32)}],
                "right_joint_path": [{"status": "CorrectionExport", "position": state_vector[:, 7:13].astype(np.float32)}],
            },
            f,
        )

    debug_video_path = None
    backend_debug_video_paths: dict[str, str] = {}
    if write_debug_video and debug_video_dir is not None:
        debug_video_dir = Path(debug_video_dir)
        debug_video_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(source_hdf5, "r") as src:
            prefix = [np.bytes_(bytes(item)) for item in src["observation/head_camera/rgb"][:start_ts]]
            target_hw = None
            if prefix:
                first_prefix = _decode_jpeg_rgb(prefix[0])
                target_hw = first_prefix.shape[:2]
            elif head_encoded_for_video:
                first_head = _decode_jpeg_rgb(head_encoded_for_video[0])
                target_hw = first_head.shape[:2]
            tail = [np.bytes_(bytes(item)) for item in src["observation/head_camera/rgb"][attach_idx:]]
        prefix_frames = _encoded_to_frames(prefix, target_hw=target_hw)
        tail_frames = _encoded_to_frames(tail, target_hw=target_hw)
        default_correction_prefix_frames = _encoded_to_frames(
            head_encoded_for_video[: prefix_len + 1],
            target_hw=target_hw,
        )
        main_backend = str(corr_meta.get("world_model_backend", "") or "").strip().lower()
        debug_backends = _unique_debug_backends(corr_meta)
        if not debug_backends:
            debug_backends = [main_backend] if main_backend in {"evac", "cosmos", "sim"} else []
        for backend in debug_backends:
            perturb_frames = []
            for record in _backend_rollout_records(corr_meta, backend):
                perturb_frames.extend(_read_rollout_debug_frames(record, target_hw=target_hw))
            corr_record = _backend_correction_record(corr_meta, backend)
            correction_prefix_frames, used_backend_correction = _build_debug_correction_prefix_frames(
                corr_image_rgb=corr_image_rgb,
                num_corr_states=prefix_len + 1,
                corr_video_record=corr_record,
                target_hw=target_hw,
            )
            if not used_backend_correction and backend == main_backend and default_correction_prefix_frames:
                correction_prefix_frames = list(default_correction_prefix_frames)
            # Keep only per-backend stitched videos. The generic
            # episodeN_debug.mp4 duplicates the main backend and wastes time.
            correction_prefix_frames = _drop_duplicate_boundary_frame(
                perturb_frames,
                correction_prefix_frames,
            )
            debug_frames = []
            debug_frames.extend(_label_video_segment(prefix_frames, "original prefix"))
            debug_frames.extend(_label_video_segment(perturb_frames, f"{backend} perturb rollout"))
            correction_label = f"{backend} correction rollout"
            if not used_backend_correction and backend != main_backend:
                correction_label = f"{backend} correction rollout (missing)"
            debug_frames.extend(_label_video_segment(correction_prefix_frames, correction_label))
            debug_frames.extend(_label_video_segment(tail_frames, "original tail"))
            backend_debug_path = debug_video_dir / f"episode{episode_id}_{backend}_debug.mp4"
            _write_video_from_frames(debug_frames, backend_debug_path, fps=fps)
            backend_debug_video_paths[backend] = str(backend_debug_path)
            if debug_video_path is None or backend == main_backend:
                debug_video_path = backend_debug_path

    record = {
        "export_path": str(out_hdf5),
        "video_path": str(out_video),
        "instruction_path": str(instr_dir / f"episode{episode_id}.json"),
        "traj_path": str(traj_dir / f"episode{episode_id}.pkl"),
        "debug_video_path": None if debug_video_path is None else str(debug_video_path),
        "backend_debug_video_paths": backend_debug_video_paths,
        "task": task_name,
        "task_config": task_config,
        "rank": int(rank),
        "episode_id": int(episode_id),
        "source_episode_id": int(source_episode_id),
        "source_hdf5": str(source_hdf5),
        "start_ts": int(start_ts),
        "start_idx": int(start_ts),
        "source_start_ts": int(start_ts),
        "attach_idx": int(attach_idx),
        "source_attach_idx": int(attach_idx),
        "deviation_idx": int(corr_meta.get("deviation_idx", start_ts) or start_ts),
        "correction_prefix_len": int(prefix_len),
        "num_steps": int(state_vector.shape[0]),
        "recover_eval_recoverable": recover_eval_last.get("recoverable"),
        "sampled_phase_key": corr_meta.get("sampled_phase_key"),
        "sampled_phase_bin_id": corr_meta.get("sampled_phase_bin_id"),
        "sampled_phase_instance_idx": corr_meta.get("sampled_phase_instance_idx"),
        "sampled_error_mode": corr_meta.get("sampled_error_mode"),
        "forced_error_mode": corr_meta.get("forced_error_mode"),
        "sampled_active_arm_pattern": corr_meta.get("sampled_active_arm_pattern"),
        "forced_dir_bin_id": corr_meta.get("forced_dir_bin_id"),
        "forced_mag_bin_id": corr_meta.get("forced_mag_bin_id"),
        "perturb_action_prefix_len": corr_meta.get("perturb_action_prefix_len"),
        "perturb_delta_linf": corr_meta.get("perturb_delta_linf"),
        "perturb_delta_l2": corr_meta.get("perturb_delta_l2"),
        "recover_eval_last": recover_eval_last,
    }
    _append_jsonl(manifest_path, record)
    return record
