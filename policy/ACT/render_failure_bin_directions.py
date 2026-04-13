#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import cv2
import h5py
import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
EVAC_PKG_ROOT = os.path.realpath(os.path.join(THIS_DIR, "evac"))
EVAC_INNER_ROOT = os.path.join(EVAC_PKG_ROOT, "evac")
for _p in [EVAC_INNER_ROOT, EVAC_PKG_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from imitate_episodes_pkg.correction import (  # noqa: E402
    _shared_extract_base_uv,
    _shared_project_base_uv_from_pose_np,
)
from imitate_episodes_pkg.perturbation import _front_hemisphere_dir_from_bin  # noqa: E402


def _load_episode(raw_data_dir, episode_id):
    ep_path = os.path.join(raw_data_dir, f"episode{episode_id}.hdf5")
    with h5py.File(ep_path, "r") as f:
        enc0 = bytes(f["observation/head_camera/rgb"][0])
        img0 = cv2.imdecode(np.frombuffer(enc0, np.uint8), cv2.IMREAD_COLOR)
        if img0 is None or img0.size == 0:
            raise RuntimeError(f"failed to decode frame 0 from {ep_path}")
        return {
            "episode_path": ep_path,
            "left_endpose": f["endpose/left_endpose"][()].astype(np.float32),
            "right_endpose": f["endpose/right_endpose"][()].astype(np.float32),
            "left_gripper": f["endpose/left_gripper"][()].astype(np.float32),
            "right_gripper": f["endpose/right_gripper"][()].astype(np.float32),
            "intrinsic_cv": f["observation/head_camera/intrinsic_cv"][0].astype(np.float32),
            "extrinsic_cv": f["observation/head_camera/extrinsic_cv"][0].astype(np.float32),
            "native_resolution": img0.shape[:2],
        }


def _load_frame_bgr(ep_path, idx):
    with h5py.File(ep_path, "r") as f:
        enc = bytes(f["observation/head_camera/rgb"][idx])
    img = cv2.imdecode(np.frombuffer(enc, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError(f"failed to decode frame {idx} from {ep_path}")
    return img


def _quat_wxyz_to_xyzw(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4,)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    if q_xyzw[3] < 0:
        q_xyzw = -q_xyzw
    return q_xyzw


def _make_pose_np(lp, lq_wxyz, lg, rp, rq_wxyz, rg):
    return np.concatenate(
        [
            np.asarray(lp, dtype=np.float32).reshape(3,),
            _quat_wxyz_to_xyzw(lq_wxyz),
            [float(np.clip(lg, 0.0, 1.0)) * 120.0],
            np.asarray(rp, dtype=np.float32).reshape(3,),
            _quat_wxyz_to_xyzw(rq_wxyz),
            [float(np.clip(rg, 0.0, 1.0)) * 120.0],
        ],
        axis=0,
    ).astype(np.float32)


def _scaled_camera(raw_data, img_shape):
    K = raw_data["intrinsic_cv"].astype(np.float32).copy()
    E = np.eye(4, dtype=np.float32)
    E[:3, :] = raw_data["extrinsic_cv"].astype(np.float32)
    h_native, w_native = raw_data["native_resolution"]
    h_img, w_img = img_shape[:2]
    if w_native > 0 and h_native > 0 and (w_native != w_img or h_native != h_img):
        sx = float(w_img) / float(w_native)
        sy = float(h_img) / float(h_native)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
    return K, E


def _project_base_uv_list(pose_arr, K, E):
    uvs_l, uvs_r, pts_l, pts_r = _shared_project_base_uv_from_pose_np(pose_arr, K, E)
    seq_l = _shared_extract_base_uv(uvs_l, pts_l.reshape(1, pts_l.shape[1], 4, 4))
    seq_r = _shared_extract_base_uv(uvs_r, pts_r.reshape(1, pts_r.shape[1], 4, 4))
    return seq_l, seq_r


def _fmt_vec(v):
    a = np.asarray(v, dtype=np.float32).reshape(3,)
    return f"[{a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}]"


def _put_text(img, text, org, scale=0.52, color=(20, 20, 20), bg=(245, 245, 245)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _panel_width(lines, scale=0.52):
    max_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(str(line), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        max_w = max(max_w, int(w))
    return max(360, max_w + 24)


def _bin_color(i, n):
    hue = int(round(179.0 * float(i) / float(max(1, n))))
    hsv = np.uint8([[[hue, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_direction_bins(
    raw_data_dir,
    episode_id,
    start_ts,
    mode,
    out_path,
    n_bins=6,
    theta_deg=60.0,
    vis_len_m=0.08,
):
    raw_data = _load_episode(raw_data_dir, episode_id)
    frame = _load_frame_bgr(raw_data["episode_path"], start_ts)
    K, E = _scaled_camera(raw_data, frame.shape)

    idx = int(np.clip(start_ts, 0, raw_data["left_endpose"].shape[0] - 1))
    lp = raw_data["left_endpose"][idx, :3].astype(np.float32)
    lq_wxyz = raw_data["left_endpose"][idx, 3:7].astype(np.float32)
    rp = raw_data["right_endpose"][idx, :3].astype(np.float32)
    rq_wxyz = raw_data["right_endpose"][idx, 3:7].astype(np.float32)
    lg = float(raw_data["left_gripper"][idx])
    rg = float(raw_data["right_gripper"][idx])

    base_pose = _make_pose_np(lp, lq_wxyz, lg, rp, rq_wxyz, rg)[None, :]
    base_l_seq, base_r_seq = _project_base_uv_list(base_pose, K, E)
    base_l = base_l_seq[0]
    base_r = base_r_seq[0]

    if base_l is None and base_r is None:
        raise RuntimeError("both gripper base projections are invalid at the selected frame")

    lines = [
        f"episode={episode_id}  start_ts={start_ts}",
        f"mode={mode}",
        f"n_bins={n_bins}  theta={theta_deg:.0f}deg  vis_len={vis_len_m:.2f}m",
        "anchors use shared gripper-base projection",
        "bins:",
    ]

    draw_items = []
    for bin_id in range(int(n_bins)):
        dir_l, _ = _front_hemisphere_dir_from_bin(lq_wxyz, bin_id, n_bins, theta_deg=theta_deg)
        dir_r, _ = _front_hemisphere_dir_from_bin(rq_wxyz, bin_id, n_bins, theta_deg=theta_deg)
        pose_tip = _make_pose_np(
            lp + dir_l * float(vis_len_m),
            lq_wxyz,
            lg,
            rp + dir_r * float(vis_len_m),
            rq_wxyz,
            rg,
        )[None, :]
        tip_l_seq, tip_r_seq = _project_base_uv_list(pose_tip, K, E)
        tip_l = tip_l_seq[0]
        tip_r = tip_r_seq[0]
        draw_items.append(
            {
                "bin_id": int(bin_id),
                "dir_l": dir_l,
                "dir_r": dir_r,
                "tip_l": tip_l,
                "tip_r": tip_r,
                "color": _bin_color(bin_id, int(n_bins)),
            }
        )
        prefix = "T" if str(mode).lower() == "translation" else "R"
        lines.append(f"L {prefix}{bin_id}: {_fmt_vec(dir_l)}")
        lines.append(f"R {prefix}{bin_id}: {_fmt_vec(dir_r)}")

    panel_w = _panel_width(lines, scale=0.52)
    canvas_h = max(frame.shape[0], 34 + 18 * len(lines))
    canvas = np.full((canvas_h, frame.shape[1] + panel_w, 3), 245, dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1], :] = frame
    cv2.line(canvas, (frame.shape[1], 0), (frame.shape[1], canvas_h - 1), (160, 160, 160), 1, cv2.LINE_AA)

    for item in draw_items:
        bin_id = item["bin_id"]
        color = item["color"]
        if base_l is not None:
            cv2.circle(canvas, (int(base_l[0]), int(base_l[1])), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if base_r is not None:
            cv2.circle(canvas, (int(base_r[0]), int(base_r[1])), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if base_l is not None and item["tip_l"] is not None:
            p0 = (int(base_l[0]), int(base_l[1]))
            p1 = (int(item["tip_l"][0]), int(item["tip_l"][1]))
            cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.18)
            _put_text(canvas, f"L{bin_id}", (p1[0] + 4, p1[1] - 4), scale=0.46, color=(0, 0, 180))
        if base_r is not None and item["tip_r"] is not None:
            p0 = (int(base_r[0]), int(base_r[1]))
            p1 = (int(item["tip_r"][0]), int(item["tip_r"][1]))
            cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.18)
            _put_text(canvas, f"R{bin_id}", (p1[0] + 4, p1[1] - 4), scale=0.46, color=(0, 0, 180))

    _put_text(canvas, "L/R anchor", (10, 20), scale=0.45, color=(0, 0, 180))
    x0 = frame.shape[1] + 10
    y0 = 24
    for i, line in enumerate(lines):
        _put_text(canvas, line, (x0, y0 + 18 * i), scale=0.52, color=(10, 10, 10))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--start_ts", type=int, required=True)
    parser.add_argument("--mode", choices=["translation", "rotation"], required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--n_bins", type=int, default=6)
    parser.add_argument("--theta_deg", type=float, default=60.0)
    parser.add_argument("--vis_len_m", type=float, default=0.08)
    args = parser.parse_args()
    render_direction_bins(
        raw_data_dir=args.raw_data_dir,
        episode_id=args.episode_id,
        start_ts=args.start_ts,
        mode=args.mode,
        out_path=args.out_path,
        n_bins=args.n_bins,
        theta_deg=args.theta_deg,
        vis_len_m=args.vis_len_m,
    )


if __name__ == "__main__":
    main()
