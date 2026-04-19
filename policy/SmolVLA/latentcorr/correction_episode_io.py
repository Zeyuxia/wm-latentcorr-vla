from __future__ import annotations

import os
import re

import numpy as np


def load_raw_data(raw_data_dir, episode_id):
    import cv2 as _cv2
    import h5py

    path = os.path.join(raw_data_dir, f"episode{episode_id}.hdf5")
    with h5py.File(path, "r") as f:
        frame0 = bytes(f["observation/head_camera/rgb"][0])
        img0 = _cv2.imdecode(np.frombuffer(frame0, np.uint8), _cv2.IMREAD_COLOR)
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
        match = re.match(r"episode_(\d+)\.hdf5$", fn)
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
    import h5py

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
