from __future__ import annotations

import os
import importlib
import sys

import numpy as np
import torch

from policy.SmolVLA.latentcorr.correction_perturbation import _infer_phase_key_from_gt_window
from policy.SmolVLA.latentcorr.failure_utils import get_failure_param_bins


def _debug_relpath(path):
    if path is None:
        return None
    try:
        return f"./{os.path.relpath(path, start=os.getcwd())}"
    except Exception:
        return path


def _import_ddpm3d_module():
    evac_repo_root = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "evac")
    )
    evac_pkg_root = os.path.join(evac_repo_root, "evac")
    for path in (evac_repo_root, evac_pkg_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    candidate_modules = [
        "evac.lvdm.models.ddpm3d",
        "policy.SmolVLA.evac.evac.lvdm.models.ddpm3d",
    ]
    last_error = None
    for module_name in candidate_modules:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError("Failed to import ddpm3d module")


def _phase_color(phase_key):
    key = str(phase_key).strip().lower()
    color_map = {
        "approach": (255, 120, 0),
        "pregrasp": (0, 165, 255),
        "transport": (0, 255, 255),
        "place": (255, 0, 255),
    }
    return color_map.get(key, (180, 180, 180))


def _draw_phase_polyline(img, seq, phase_seq, width=1):
    import cv2

    prev = None
    for idx, point in enumerate(seq):
        if point is None:
            prev = None
            continue
        if prev is not None:
            color = _phase_color(phase_seq[idx] if idx < len(phase_seq) else "unknown")
            cv2.line(img, (int(prev[0]), int(prev[1])), (int(point[0]), int(point[1])), color, width, cv2.LINE_AA)
        prev = point


def _draw_phase_legend(img):
    import cv2

    rows = [
        ("approach", _phase_color("approach")),
        ("pregrasp", _phase_color("pregrasp")),
        ("transport", _phase_color("transport")),
        ("place", _phase_color("place")),
    ]
    x0, y0 = 8, 8
    row_h = 14
    width = 126
    height = 6 + row_h * len(rows) + 6
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + height), (220, 220, 220), 1, cv2.LINE_AA)
    for idx, (name, color) in enumerate(rows):
        y = y0 + 14 + idx * row_h
        cv2.circle(img, (x0 + 9, y - 3), 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, name, (x0 + 17, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)


def draw_phase_polyline(img, seq, phase_seq, width=1):
    _draw_phase_polyline(img, seq, phase_seq, width=width)


def draw_phase_legend(img):
    _draw_phase_legend(img)


def _build_pose_np_from_raw_indices(raw_data, raw_idx_list):
    pose_list = []
    for raw_idx in raw_idx_list:
        lp_gt = raw_data["left_endpose"][raw_idx, :3].astype(np.float32)
        lq_gt_wxyz = raw_data["left_endpose"][raw_idx, 3:7].astype(np.float32)
        rp_gt = raw_data["right_endpose"][raw_idx, :3].astype(np.float32)
        rq_gt_wxyz = raw_data["right_endpose"][raw_idx, 3:7].astype(np.float32)
        lq_gt_xyzw = np.array([lq_gt_wxyz[1], lq_gt_wxyz[2], lq_gt_wxyz[3], lq_gt_wxyz[0]], dtype=np.float32)
        rq_gt_xyzw = np.array([rq_gt_wxyz[1], rq_gt_wxyz[2], rq_gt_wxyz[3], rq_gt_wxyz[0]], dtype=np.float32)
        if lq_gt_xyzw[3] < 0:
            lq_gt_xyzw = -lq_gt_xyzw
        if rq_gt_xyzw[3] < 0:
            rq_gt_xyzw = -rq_gt_xyzw
        lg_gt = float(np.clip(raw_data["left_gripper"][raw_idx], 0.0, 1.0)) * 120.0
        rg_gt = float(np.clip(raw_data["right_gripper"][raw_idx], 0.0, 1.0)) * 120.0
        pose_list.append(
            np.concatenate([lp_gt, lq_gt_xyzw, [lg_gt], rp_gt, rq_gt_xyzw, [rg_gt]], axis=0).astype(np.float32)
        )
    if len(pose_list) == 0:
        return None
    return np.stack(pose_list, axis=0)


def _endpose_wxyz_to_pose16(left_endpose_wxyz, right_endpose_wxyz, left_grip=120.0, right_grip=120.0):
    left_pose = np.asarray(left_endpose_wxyz, dtype=np.float32).reshape(7,)
    right_pose = np.asarray(right_endpose_wxyz, dtype=np.float32).reshape(7,)
    lq_wxyz = left_pose[3:7]
    rq_wxyz = right_pose[3:7]
    lq_xyzw = np.array([lq_wxyz[1], lq_wxyz[2], lq_wxyz[3], lq_wxyz[0]], dtype=np.float32)
    rq_xyzw = np.array([rq_wxyz[1], rq_wxyz[2], rq_wxyz[3], rq_wxyz[0]], dtype=np.float32)
    if lq_xyzw[3] < 0:
        lq_xyzw = -lq_xyzw
    if rq_xyzw[3] < 0:
        rq_xyzw = -rq_xyzw
    return np.concatenate(
        [
            left_pose[:3],
            lq_xyzw,
            np.array([float(left_grip)], dtype=np.float32),
            right_pose[:3],
            rq_xyzw,
            np.array([float(right_grip)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _project_base_uv_from_pose_np(pose_arr, intrinsic, extrinsic):
    ddpm3d_mod = _import_ddpm3d_module()

    w2c_t = torch.from_numpy(extrinsic).float().unsqueeze(0).unsqueeze(0)
    intrinsic_t = torch.from_numpy(intrinsic).float().unsqueeze(0).unsqueeze(0)
    cvt_matrix = torch.tensor(ddpm3d_mod.Gripper2EEFCvt, dtype=torch.float32).view(1, 1, 4, 4)
    ee_key_pts = torch.tensor(ddpm3d_mod.EndEffectorPts, dtype=torch.float32).view(1, 1, 4, 4).permute(0, 1, 3, 2)

    pose_t = torch.from_numpy(pose_arr).float()
    pose_l_mat = ddpm3d_mod.get_transformation_matrix_from_quat(pose_t[:, 0:7]).unsqueeze(0)
    pose_r_mat = ddpm3d_mod.get_transformation_matrix_from_quat(pose_t[:, 8:15]).unsqueeze(0)
    ee2cam_l = torch.matmul(torch.matmul(w2c_t, pose_l_mat), cvt_matrix)
    ee2cam_r = torch.matmul(torch.matmul(w2c_t, pose_r_mat), cvt_matrix)
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)
    uvs_l = torch.matmul(intrinsic_t, pts_l[:, :, :3, :])
    uvs_r = torch.matmul(intrinsic_t, pts_r[:, :, :3, :])
    uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)[0].cpu().numpy()
    uvs_r = (uvs_r / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)[0].cpu().numpy()
    return uvs_l, uvs_r, pts_l, pts_r


def _extract_base_uv(uvs, pts):
    out = []
    for idx in range(int(uvs.shape[0])):
        try:
            z = float(pts[0, idx, 2, 0].item())
            u = int(uvs[idx, 0, 0])
            v = int(uvs[idx, 0, 1])
        except Exception:
            out.append(None)
            continue
        if z > 1e-6 and np.isfinite(uvs[idx, 0, :]).all():
            out.append((u, v))
        else:
            out.append(None)
    return out


def _compute_phase_projection(raw_data, phase_window_len, intrinsic, extrinsic):
    n_total = int(raw_data["left_endpose"].shape[0])
    full_left_grip = np.asarray(raw_data["left_gripper"], dtype=np.float32).reshape(-1)
    full_right_grip = np.asarray(raw_data["right_gripper"], dtype=np.float32).reshape(-1)
    phase_raw_idx = []
    phase_seq = []
    if n_total <= 1:
        return None, None, None
    stride = int(max(1, n_total // 240))
    for raw_idx in range(0, n_total, stride):
        phase_key = _infer_phase_key_from_gt_window(
            full_left_grip[raw_idx:],
            full_right_grip[raw_idx:],
            phase_window_len,
        )
        phase_raw_idx.append(raw_idx)
        phase_seq.append(phase_key)
    if len(phase_raw_idx) < 2:
        return None, None, None
    phase_pose_np = _build_pose_np_from_raw_indices(raw_data, phase_raw_idx)
    if phase_pose_np is None:
        return None, None, None
    p_uvs_l, p_uvs_r, p_pts_l, p_pts_r = _project_base_uv_from_pose_np(phase_pose_np, intrinsic, extrinsic)
    p_l_seq = _extract_base_uv(p_uvs_l, p_pts_l.reshape(1, p_pts_l.shape[1], 4, 4))
    p_r_seq = _extract_base_uv(p_uvs_r, p_pts_r.reshape(1, p_pts_r.shape[1], 4, 4))
    return p_l_seq, p_r_seq, phase_seq


def compute_phase_projection(raw_data, phase_window_len, intrinsic, extrinsic):
    return _compute_phase_projection(raw_data, phase_window_len, intrinsic, extrinsic)


def _draw_phase_bin_starts(img, l_seq, r_seq, phase_seq, phase_bins):
    import cv2

    phase_bins = int(max(1, phase_bins))
    seq_len = len(phase_seq)
    if seq_len <= 0:
        return

    start = 0
    while start < seq_len:
        key = str(phase_seq[start])
        end = start + 1
        while end < seq_len and str(phase_seq[end]) == key:
            end += 1
        seg_len = end - start
        if seg_len > 0:
            for bin_id in range(phase_bins):
                rel = int(np.floor(float(bin_id) * float(seg_len) / float(phase_bins)))
                idx = start + min(seg_len - 1, max(0, rel))

                uv_l = None
                uv_r = None
                for cand in range(idx, end):
                    if uv_l is None and cand < len(l_seq) and l_seq[cand] is not None:
                        uv_l = l_seq[cand]
                    if uv_r is None and cand < len(r_seq) and r_seq[cand] is not None:
                        uv_r = r_seq[cand]
                    if uv_l is not None and uv_r is not None:
                        break
                if uv_l is None and uv_r is None:
                    for cand in range(idx - 1, start - 1, -1):
                        if uv_l is None and cand < len(l_seq) and l_seq[cand] is not None:
                            uv_l = l_seq[cand]
                        if uv_r is None and cand < len(r_seq) and r_seq[cand] is not None:
                            uv_r = r_seq[cand]
                        if uv_l is not None and uv_r is not None:
                            break

                label_anchor = None
                if uv_l is not None:
                    ul, vl = int(uv_l[0]), int(uv_l[1])
                    cv2.circle(img, (ul, vl), 2, (0, 255, 0), -1, cv2.LINE_AA)
                    label_anchor = (ul, vl) if label_anchor is None else label_anchor
                if uv_r is not None:
                    ur, vr = int(uv_r[0]), int(uv_r[1])
                    cv2.circle(img, (ur, vr), 2, (0, 0, 255), -1, cv2.LINE_AA)
                    label_anchor = (ur, vr) if label_anchor is None else label_anchor

                if label_anchor is not None:
                    tag_prefix = {
                        "approach": "A",
                        "pregrasp": "G",
                        "transport": "T",
                        "place": "L",
                    }.get(key, key[:1].upper() if len(key) > 0 else "?")
                    tag = f"{tag_prefix}{bin_id}"
                    tx, ty = (int(label_anchor[0]) + 4, int(label_anchor[1]) - 4)
                    cv2.putText(img, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1, cv2.LINE_AA)
        start = end


def draw_phase_bin_starts(img, l_seq, r_seq, phase_seq, phase_bins):
    _draw_phase_bin_starts(img, l_seq, r_seq, phase_seq, phase_bins)


def save_gt_projection_on_original(
    debug_corr_dir,
    image_data_s,
    raw_data,
    start_ts,
    phase_window_len,
    phase_bins,
):
    import cv2

    try:
        overlay_src = (image_data_s[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_ref_idx = int(np.clip(start_ts, 0, raw_data["left_endpose"].shape[0] - 1))
        try:
            episode_path = raw_data.get("episode_path")
            if episode_path is not None and os.path.isfile(episode_path):
                import h5py

                with h5py.File(episode_path, "r") as gt_file:
                    encoded = bytes(gt_file["observation/head_camera/rgb"][gt_ref_idx])
                gt_img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if gt_img is not None and gt_img.size > 0:
                    overlay_src = gt_img
        except Exception:
            pass

        intrinsic = raw_data["intrinsic_cv"].astype(np.float32).copy()
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :] = raw_data["extrinsic_cv"].astype(np.float32)
        h_native, w_native = raw_data.get("native_resolution", (overlay_src.shape[0], overlay_src.shape[1]))
        h_img, w_img = overlay_src.shape[:2]
        if w_native > 0 and h_native > 0 and (w_native != w_img or h_native != h_img):
            sx = float(w_img) / float(w_native)
            sy = float(h_img) / float(h_native)
            intrinsic[0, 0] *= sx
            intrinsic[0, 2] *= sx
            intrinsic[1, 1] *= sy
            intrinsic[1, 2] *= sy

        overlay = overlay_src.copy()
        p_l_seq, p_r_seq, phase_seq = _compute_phase_projection(raw_data, phase_window_len, intrinsic, extrinsic)
        if p_l_seq is not None and p_r_seq is not None and phase_seq is not None:
            _draw_phase_polyline(overlay, p_l_seq, phase_seq, width=2)
            _draw_phase_polyline(overlay, p_r_seq, phase_seq, width=2)
            _draw_phase_bin_starts(overlay, p_l_seq, p_r_seq, phase_seq, phase_bins)
            _draw_phase_legend(overlay)

        out_path = os.path.join(debug_corr_dir, "gt_projection_on_original.png")
        cv2.imwrite(out_path, overlay)
        return {"path": _debug_relpath(out_path), "exists": bool(os.path.exists(out_path)), "gt_ref_idx": int(gt_ref_idx)}
    except Exception:
        try:
            import traceback

            with open(os.path.join(debug_corr_dir, "gt_projection_on_original_error.txt"), "w") as file:
                file.write(traceback.format_exc())
        except Exception:
            pass
        return None


def save_recover_eval_compare_image(
    debug_corr_dir,
    raw_data,
    gt_ref_idx,
    recover_pred_img,
    step_idx,
    mode=None,
    recoverable=None,
    metric_name=None,
    metric=None,
    threshold=None,
    metrics=None,
    thresholds=None,
    passes=None,
    failed_thresholds=None,
    nearest_dist=None,
):
    import cv2

    def _put_text_hc(img, text, org, scale=0.5):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 1, cv2.LINE_AA)

    try:
        gt_idx = int(np.clip(int(gt_ref_idx), 0, raw_data["left_endpose"].shape[0] - 1))
        gt_img = None
        episode_path = raw_data.get("episode_path")
        if episode_path is not None and os.path.isfile(episode_path):
            import h5py

            with h5py.File(episode_path, "r") as gt_file:
                encoded = bytes(gt_file["observation/head_camera/rgb"][gt_idx])
            gt_img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if gt_img is None or gt_img.size == 0:
            return None

        pred_img = (recover_pred_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if pred_img.shape[:2] != gt_img.shape[:2]:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        compare = np.concatenate([gt_img, pred_img], axis=1)

        info_lines = [
            f"LEFT=GT_REF(idx={gt_idx})   RIGHT=RECOVER_ROLLOUT_LAST",
            f"mode={mode}",
            ("recoverable=unknown" if recoverable is None else f"recoverable={bool(recoverable)}"),
        ]
        if metric_name is not None and metric is not None:
            info_lines.append(f"{metric_name}={float(metric):.6f}")
        elif metric_name is not None:
            info_lines.append(f"criterion={metric_name}")
        if threshold is not None:
            info_lines.append(f"threshold={float(threshold):.6f}")
        if isinstance(metrics, dict):
            threshold_map = thresholds if isinstance(thresholds, dict) else {}
            pass_map = passes if isinstance(passes, dict) else {}
            for name in ("pos_err_m", "rot_err_deg", "gripper_err"):
                value = metrics.get(name)
                thresh = threshold_map.get(name)
                passed = pass_map.get(name)
                if value is None:
                    continue
                value_s = f"{float(value):.6f}"
                thresh_s = "None" if thresh is None else f"{float(thresh):.6f}"
                pass_s = "unknown" if passed is None else ("pass" if bool(passed) else "fail")
                info_lines.append(f"{name}={value_s} thresh={thresh_s} {pass_s}")
        if failed_thresholds is not None:
            failed_list = [str(item) for item in list(failed_thresholds)]
            info_lines.append(
                "failed_thresholds="
                + ("none" if len(failed_list) == 0 else ",".join(failed_list))
            )
        if nearest_dist is not None:
            info_lines.append(f"nearest_dist={float(nearest_dist):.6f}")

        panel_h = 28 + 18 * len(info_lines)
        canvas = np.zeros((compare.shape[0] + panel_h, compare.shape[1], 3), dtype=np.uint8)
        canvas[: compare.shape[0], :, :] = compare
        canvas[compare.shape[0] :, :, :] = 245
        cv2.line(canvas, (0, compare.shape[0]), (compare.shape[1] - 1, compare.shape[0]), (120, 120, 120), 1, cv2.LINE_AA)
        x0 = 10
        y0 = compare.shape[0] + 20
        for idx, text in enumerate(info_lines):
            _put_text_hc(canvas, text, (x0, y0 + idx * 16), scale=0.46)

        out_path = os.path.join(debug_corr_dir, f"recover_eval_gtref_vs_rollout_last_step_{int(step_idx):03d}.png")
        cv2.imwrite(out_path, canvas)
        return {
            "path": _debug_relpath(out_path),
            "exists": bool(os.path.exists(out_path)),
            "gt_ref_idx": int(gt_idx),
            "step": int(step_idx),
            "mode": mode,
            "recoverable": (None if recoverable is None else bool(recoverable)),
            "metric_name": metric_name,
            "metric": (None if metric is None else float(metric)),
            "threshold": (None if threshold is None else float(threshold)),
            "metrics": metrics if isinstance(metrics, dict) else None,
            "thresholds": thresholds if isinstance(thresholds, dict) else None,
            "passes": passes if isinstance(passes, dict) else None,
            "failed_thresholds": (
                None if failed_thresholds is None else [str(item) for item in list(failed_thresholds)]
            ),
            "nearest_dist": (None if nearest_dist is None else float(nearest_dist)),
        }
    except Exception:
        try:
            import traceback

            with open(os.path.join(debug_corr_dir, "recover_eval_compare_error.txt"), "w") as file:
                file.write(traceback.format_exc())
        except Exception:
            pass
        return None


def save_perturb_compare_image(
    debug_corr_dir,
    first_img,
    last_img,
    sampled_unit,
    rollout_last_record=None,
):
    import cv2

    def _put_text_hc(img, text, org, scale=0.5):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 1, cv2.LINE_AA)

    try:
        first = (first_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        last = (last_img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if last.shape[:2] != first.shape[:2]:
            last = cv2.resize(last, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_LINEAR)
        compare = np.concatenate([first, last], axis=1)

        sample_unit = sampled_unit if isinstance(sampled_unit, dict) else {}
        rollout_record = rollout_last_record if isinstance(rollout_last_record, dict) else {}
        tr_gain = rollout_record.get("perturb_translation_gain_m")
        rot_deg = rollout_record.get("perturb_rotation_deg")
        tr_gain_s = "None" if tr_gain is None else f"{float(tr_gain):.3f}"
        rot_deg_s = "None" if rot_deg is None else f"{float(rot_deg):.3f}"
        lines = [
            "LEFT=INPUT_FIRST(start)   RIGHT=PERTURBED_LAST(after_rollout)",
            f"phase={sample_unit.get('phase_key')} bin={sample_unit.get('phase_bin_id')} inst={sample_unit.get('phase_instance_idx')}",
            f"error_mode={sample_unit.get('error_mode')} arm={sample_unit.get('active_arm_pattern')}",
            f"dir_bin={sample_unit.get('dir_bin_id')} mag_bin={sample_unit.get('mag_bin_id')}",
            f"sampled_error_mode={rollout_record.get('sampled_error_mode')}",
            f"translation_gain_m={tr_gain_s} rotation_deg={rot_deg_s}",
        ]
        blur = rollout_record.get("evac_blur_filter")
        if isinstance(blur, dict):
            passed = blur.get("passed")
            ratio = blur.get("sharpness_ratio")
            min_ratio = blur.get("min_ratio")
            pred_s = blur.get("pred_sharpness")
            ref_s = blur.get("ref_sharpness")
            region = blur.get("region")
            bbox = blur.get("bbox_xyxy")
            ratio_s = "None" if ratio is None else f"{float(ratio):.3f}"
            min_ratio_s = "None" if min_ratio is None else f"{float(min_ratio):.3f}"
            pred_s_s = "None" if pred_s is None else f"{float(pred_s):.5f}"
            ref_s_s = "None" if ref_s is None else f"{float(ref_s):.5f}"
            lines.extend(
                [
                    f"evac_blur_filter passed={passed} region={region} ratio={ratio_s} min_ratio={min_ratio_s}",
                    f"blur_bbox_xyxy={bbox}",
                    f"sharpness pred={pred_s_s} ref={ref_s_s}",
                ]
            )

        panel_h = 28 + 18 * len(lines)
        canvas = np.zeros((compare.shape[0] + panel_h, compare.shape[1], 3), dtype=np.uint8)
        canvas[: compare.shape[0], :, :] = compare
        canvas[compare.shape[0] :, :, :] = 245
        cv2.line(canvas, (0, compare.shape[0]), (compare.shape[1] - 1, compare.shape[0]), (120, 120, 120), 1, cv2.LINE_AA)
        y0 = compare.shape[0] + 20
        for idx, text in enumerate(lines):
            _put_text_hc(canvas, str(text), (10, y0 + 16 * idx), scale=0.44)

        out_path = os.path.join(debug_corr_dir, "perturb_input_vs_last.png")
        cv2.imwrite(out_path, canvas)
        return {"path": _debug_relpath(out_path), "exists": bool(os.path.exists(out_path)), "sampled_unit": sample_unit}
    except Exception:
        try:
            import traceback

            with open(os.path.join(debug_corr_dir, "perturb_compare_error.txt"), "w") as file:
                file.write(traceback.format_exc())
        except Exception:
            pass
        return None


def save_loss_batch_projection(save_dir, image_cam, action_norm, is_pad, raw_data, fk, norm_stats, meta=None):
    import cv2
    import json

    os.makedirs(save_dir, exist_ok=True)

    act = np.asarray(action_norm.detach().cpu().numpy(), dtype=np.float32)
    pad = np.asarray(is_pad.detach().cpu().numpy(), dtype=bool).reshape(-1)
    if act.ndim != 2 or act.shape[1] < 14:
        return
    valid = np.where(~pad)[0]
    if valid.size == 0:
        return
    act = act[valid]
    act_raw = np.asarray(act * norm_stats["action_std"] + norm_stats["action_mean"], dtype=np.float32)
    if act_raw.shape[0] <= 0:
        return

    img_u8 = np.clip(image_cam.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
    overlay = img_u8.copy()

    intrinsic = raw_data["intrinsic_cv"].astype(np.float32).copy()
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :] = raw_data["extrinsic_cv"].astype(np.float32)
    h_native, w_native = raw_data.get("native_resolution", (overlay.shape[0], overlay.shape[1]))
    h_img, w_img = overlay.shape[:2]
    if w_native > 0 and h_native > 0 and (w_native != w_img or h_native != h_img):
        sx = float(w_img) / float(w_native)
        sy = float(h_img) / float(h_native)
        intrinsic[0, 0] *= sx
        intrinsic[0, 2] *= sx
        intrinsic[1, 1] *= sy
        intrinsic[1, 2] *= sy

    pose_list = []
    for idx in range(act_raw.shape[0]):
        fr = fk.forward(act_raw[idx, 0:6], act_raw[idx, 7:13])
        lp, lq_wxyz = fr["left"]
        rp, rq_wxyz = fr["right"]
        lq_xyzw = np.array([lq_wxyz[1], lq_wxyz[2], lq_wxyz[3], lq_wxyz[0]], dtype=np.float32)
        rq_xyzw = np.array([rq_wxyz[1], rq_wxyz[2], rq_wxyz[3], rq_wxyz[0]], dtype=np.float32)
        if lq_xyzw[3] < 0:
            lq_xyzw = -lq_xyzw
        if rq_xyzw[3] < 0:
            rq_xyzw = -rq_xyzw
        lg = float(np.clip(act_raw[idx, 6], 0.0, 1.0)) * 120.0
        rg = float(np.clip(act_raw[idx, 13], 0.0, 1.0)) * 120.0
        pose_list.append(np.concatenate([lp, lq_xyzw, [lg], rp, rq_xyzw, [rg]], axis=0).astype(np.float32))
    pose_np = np.stack(pose_list, axis=0)

    try:
        uvs_l, uvs_r, pts_l, pts_r = _project_base_uv_from_pose_np(pose_np, intrinsic, extrinsic)

        def _extract_visible_uv(uvs, pts):
            seq = []
            for idx in range(uvs.shape[0]):
                z = float(pts[0, idx, 2, 0].item())
                u = int(uvs[idx, 0, 0])
                v = int(uvs[idx, 0, 1])
                if z > 1e-6 and (0 <= u < w_img) and (0 <= v < h_img):
                    seq.append((u, v))
                else:
                    seq.append(None)
            return seq

        def _draw_polyline(img, seq, color):
            prev = None
            for point in seq:
                if point is None:
                    prev = None
                    continue
                if prev is not None:
                    cv2.line(img, (int(prev[0]), int(prev[1])), (int(point[0]), int(point[1])), color, 2, cv2.LINE_AA)
                prev = point

        def _annotate_start_end(img, seq, prefix, color):
            if len(seq) == 0:
                return
            start = seq[0]
            end = seq[-1]
            if start is not None:
                cv2.circle(img, (int(start[0]), int(start[1])), 4, color, -1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"{prefix}-S",
                    (int(start[0]) + 6, int(start[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            if end is not None:
                cv2.circle(img, (int(end[0]), int(end[1])), 4, color, -1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"{prefix}-E",
                    (int(end[0]) + 6, int(end[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        left_uv = _extract_visible_uv(uvs_l, pts_l.reshape(1, pts_l.shape[1], 4, 4))
        right_uv = _extract_visible_uv(uvs_r, pts_r.reshape(1, pts_r.shape[1], 4, 4))
        _draw_polyline(overlay, left_uv, (0, 255, 0))
        _draw_polyline(overlay, right_uv, (0, 0, 255))
        _annotate_start_end(overlay, left_uv, "L", (0, 255, 0))
        _annotate_start_end(overlay, right_uv, "R", (0, 0, 255))
        cv2.imwrite(os.path.join(save_dir, "loss_projection_on_input.png"), overlay)
    except Exception:
        import traceback

        with open(os.path.join(save_dir, "loss_projection_error.txt"), "w") as file:
            file.write(traceback.format_exc())
        cv2.imwrite(os.path.join(save_dir, "loss_projection_on_input.png"), overlay)

    meta_out = {"n_valid_actions": int(act_raw.shape[0]), "image_hw": [int(h_img), int(w_img)]}
    if isinstance(meta, dict):
        meta_out.update(meta)
    with open(os.path.join(save_dir, "loss_projection_meta.json"), "w") as file:
        json.dump(meta_out, file, indent=2)
