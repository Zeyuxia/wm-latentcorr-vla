"""
测试 get_traj 投影功能的单元测试脚本
"""
import os
import sys
import json
import h5py
import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix


# ========== 从原代码复制的常量和函数 ==========

ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

EndEffectorPts = [
    [0, 0, 0, 1],
    [0.1, 0, 0, 1],
    [0, 0.1, 0, 1],
    [0, 0, 0.1, 1]
]

Gripper2EEFCvt = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.23],
    [0, 0, 0, 1]
]


def get_transformation_matrix_from_quat(xyz_quat):
    """从 xyz + quaternion 构造 4x4 变换矩阵"""
    # xyz_quat: tensor, (b, 7)
    rot_quat = xyz_quat[:, 3:]
    # in pytorch3d, quaternion_to_matrix takes wxyz-quat as input
    rot_quat = rot_quat[:, [3, 0, 1, 2]]
    rot = quaternion_to_matrix(rot_quat)
    trans = xyz_quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(xyz_quat.shape[0], 1, 1)
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output


def get_traj_v2(sample_size, pose, c2w_input, intrinsic, radius=50, flip_x=True, flip_v=False, eef_offset=0.23):
    """
    可配置是否翻转X轴和V坐标的 get_traj 版本
    flip_x=True: 翻转X轴后求逆 (方案A)
    flip_x=False: 直接求逆 (方案B)
    flip_v=True: 翻转V坐标 (v = h - v)
    eef_offset: EEF Z轴偏移量 (AgiBotWorld=0.23, LIBERO/Panda可能需要更小或为0)
    """        
    h, w = sample_size

    if isinstance(c2w_input, np.ndarray):
        c2w_input = torch.tensor(c2w_input, dtype=torch.float32)

    if flip_x:
        # 方案 A: 翻转X轴后求逆
        c2w = c2w_input.clone()
        c2w[..., :3, 0] = -c2w[..., :3, 0]
        w2c = torch.inverse(c2w)
    else:
        # 方案 B: 直接求逆
        w2c = torch.inverse(c2w_input)

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)
    
    ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32, device=pose.device).view(1, 1, 4, 4).permute(0, 1, 3, 2)

    pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(0)
    pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(0)
    
    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    # 使用可配置的偏移量
    Gripper2EEFCvt_custom = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, eef_offset],
        [0, 0, 0, 1]
    ]
    cvt_matrix = torch.tensor(Gripper2EEFCvt_custom, dtype=torch.float32, device=pose.device).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, cvt_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, cvt_matrix)
    
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)
    
    intrinsic = intrinsic.unsqueeze(1)

    uvs_l = torch.matmul(intrinsic, pts_l[:, :, :3, :])
    uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2)
    
    uvs_r = torch.matmul(intrinsic, pts_r[:, :, :3, :])
    uvs_r = (uvs_r / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2)
    
    # 如果需要翻转 V 坐标
    if flip_v:
        uvs_l[:, :, :, 1] = h - uvs_l[:, :, :, 1]
        uvs_r[:, :, :, 1] = h - uvs_r[:, :, :, 1]
    
    uvs_l = uvs_l.to(dtype=torch.int64)
    uvs_r = uvs_r.to(dtype=torch.int64)

    all_img_list = []
    for iv in range(w2c.shape[0]):
        
        img_list = []
        for i in range(pose.shape[0]):
            
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            # Gripper Range in AgiBotWorld < 120
            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = ColorMapLeft(normalized_value_l)[:3]
            color_r = ColorMapRight(normalized_value_r)[:3]
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            for points, color, colors, lr_tag in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0])
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                point = np.array(points[0][:2])
                cv2.circle(img, tuple(point), radius, color, -1)

            for points, color, colors, lr_tag in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0])
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                for j, point in enumerate(points):
                    point = np.array(point[:2])
                    if j == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[j-1], 8)

            img_list.append(img / 255.)
        img_list = np.stack(img_list, axis=0)
        all_img_list.append(img_list)

    all_img_list = np.stack(all_img_list, axis=0)
    return all_img_list


def get_traj(sample_size, pose, w2c, c2w, intrinsic, radius=50):
    """
    原始的 get_traj 函数（从 ddpm3d.py 复制）
    this function takes camera info. and eef. poses as inputs, and outputs the trajectory maps.
    output traj map shape: (c, v, t, h, w)
    """        
    h, w = sample_size

    # 注意：传入的 w2c 参数实际上是 c2w（命名有误）
    if isinstance(w2c, np.ndarray):
        w2c = torch.tensor(w2c, dtype=torch.float32)

    # 传入的实际是 c2w，需要翻转 X 轴后求逆得到真正的 w2c
    c2w = w2c.clone()
    c2w[..., :3, 0] = -c2w[..., :3, 0]  # 翻转 X 轴（第一列）
    w2c = torch.inverse(c2w)  # 求逆得到 w2c

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)
    
    ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32, device=pose.device).view(1, 1, 4, 4).permute(0, 1, 3, 2)

    pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(0)
    pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(0)
    
    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32, device=pose.device).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, cvt_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, cvt_matrix)
    
    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)
    
    intrinsic = intrinsic.unsqueeze(1)

    uvs_l = torch.matmul(intrinsic, pts_l[:, :, :3, :])
    uvs_l = (uvs_l / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)

    uvs_r = torch.matmul(intrinsic, pts_r[:, :, :3, :])
    uvs_r = (uvs_r / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(dtype=torch.int64)

    all_img_list = []
    for iv in range(w2c.shape[0]):
        
        img_list = []
        for i in range(pose.shape[0]):
            
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            # Gripper Range in AgiBotWorld < 120
            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = ColorMapLeft(normalized_value_l)[:3]
            color_r = ColorMapRight(normalized_value_r)[:3]
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            i_coord_list = []
            for points, color, colors, lr_tag in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0])
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                point = np.array(points[0][:2])
                cv2.circle(img, tuple(point), radius, color, -1)

            for points, color, colors, lr_tag in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0])
                if base[0] < 0 or base[0] >= w or base[1] < 0 or base[1] >= h:
                    continue
                for j, point in enumerate(points):
                    point = np.array(point[:2])
                    if j == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[j-1], 8)

            img_list.append(img / 255.)
        img_list = np.stack(img_list, axis=0)
        all_img_list.append(img_list)

    all_img_list = np.stack(all_img_list, axis=0)
    # all_img_list = rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()

    return all_img_list


# ========== 数据加载函数 ==========

def load_intrinsic(intrinsic_path):
    """加载内参"""
    with open(intrinsic_path, "r") as f:
        info = json.load(f)["intrinsic"]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = info["fx"]
    intrinsic[1, 1] = info["fy"]
    intrinsic[0, 2] = info["ppx"]
    intrinsic[1, 2] = info["ppy"]
    return intrinsic


def load_extrinsic(extrinsic_path, num_frames=None):
    """加载外参（c2w）"""
    with open(extrinsic_path, "r") as f:
        info = json.load(f)
    
    if num_frames is None:
        num_frames = len(info)
    
    c2ws = []
    for i in range(min(num_frames, len(info))):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.array(info[i]["extrinsic"]["rotation_matrix"])
        c2w[:3, 3] = np.array(info[i]["extrinsic"]["translation_vector"])
        c2ws.append(c2w)
    
    c2ws = np.stack(c2ws, axis=0)
    return c2ws


def load_action(h5_path, num_frames=None):
    """
    加载动作数据并转换为 (T, 16) 格式
    格式: [left_pos(3), left_quat(4), left_gripper(1), right_pos(3), right_quat(4), right_gripper(1)]
    """
    with h5py.File(h5_path, "r") as f:
        positions = f['state/end/position'][:]  # (T, 2, 3)
        orientations = f['state/end/orientation'][:]  # (T, 2, 4)
        grippers = f['state/effector/position'][:]  # (T, 2)
    
    if num_frames is not None:
        positions = positions[:num_frames]
        orientations = orientations[:num_frames]
        grippers = grippers[:num_frames]
    
    T = positions.shape[0]
    action = np.zeros((T, 16), dtype=np.float32)
    
    # Left arm: pos(0:3), quat(3:7), gripper(7)
    action[:, 0:3] = positions[:, 0, :]
    action[:, 3:7] = orientations[:, 0, :]
    action[:, 7] = grippers[:, 0]
    
    # Right arm: pos(8:11), quat(11:15), gripper(15)
    action[:, 8:11] = positions[:, 1, :]
    action[:, 11:15] = orientations[:, 1, :]
    action[:, 15] = grippers[:, 1]
    
    return action


def load_video_frames(video_path, num_frames=None):
    """加载视频帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames is None:
        num_frames = total_frames
    else:
        num_frames = min(num_frames, total_frames)
    
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    return frames


# ========== 主测试函数 ==========

def debug_projection(pose, c2w_input, intrinsic, sample_size):
    """
    详细调试投影过程
    """
    print("\n========== DETAILED PROJECTION DEBUG ==========")
    
    h, w = sample_size
    
    # 只看第一帧
    pose_first = pose[0:1]  # (1, 16)
    c2w_first = c2w_input[:, 0:1, :, :]  # (1, 1, 4, 4)
    
    print(f"\n1. 原始 c2w (传入参数):")
    print(c2w_first[0, 0])
    
    # robosuite 的 get_camera_extrinsic_matrix 返回的是 c2w
    # 并且已经应用了 camera_axis_correction:
    # [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    # 这意味着相机坐标系是 OpenCV 约定: X右, Y下, Z前
    
    # 直接求逆得到 w2c，不需要额外翻转
    w2c_direct = torch.inverse(c2w_first)
    
    print(f"\n2. 直接求逆的 w2c:")
    print(w2c_direct[0, 0])
    
    # 获取 EEF 位置（世界坐标系）
    eef_pos = pose_first[0, 0:3]  # (3,)
    eef_pos_h = torch.cat([eef_pos, torch.tensor([1.0])])  # (4,)
    
    print(f"\n3. EEF 世界坐标: {eef_pos.tolist()}")
    
    # 变换到相机坐标系
    cam_pt_h = torch.matmul(w2c_direct[0, 0], eef_pos_h)
    cam_pt = cam_pt_h[:3]
    
    print(f"\n4. EEF 相机坐标: {cam_pt.tolist()}")
    print(f"   Z值 (深度): {cam_pt[2].item():.4f}")
    
    # 投影到图像
    uv_h = torch.matmul(intrinsic[0], cam_pt)
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    
    print(f"\n5. 直接投影的 UV: u={u.item():.2f}, v={v.item():.2f}")
    
    # 由于 robosuite 的外参已经校正过，相机坐标系的 Y 轴是向下的
    # 所以直接投影的 V 应该就是正确的图像 V 坐标
    # 但如果图像在渲染时没有翻转，可能需要 v_img = h - v
    
    v_flipped = h - v
    print(f"   翻转V后: u={u.item():.2f}, v_flipped={v_flipped.item():.2f}")
    
    # ========== 测试 EVAC 的方式 ==========
    # EVAC 中翻转 X 轴后求逆
    c2w_flipX = c2w_first.clone()
    c2w_flipX[..., :3, 0] = -c2w_flipX[..., :3, 0]
    w2c_flipX = torch.inverse(c2w_flipX)
    
    cam_pt_h_flipX = torch.matmul(w2c_flipX[0, 0], eef_pos_h)
    cam_pt_flipX = cam_pt_h_flipX[:3]
    
    uv_h_flipX = torch.matmul(intrinsic[0], cam_pt_flipX)
    u_flipX = uv_h_flipX[0] / uv_h_flipX[2]
    v_flipX = uv_h_flipX[1] / uv_h_flipX[2]
    
    print(f"\n6. EVAC方式 (翻转X后求逆):")
    print(f"   相机坐标: {cam_pt_flipX.tolist()}")
    print(f"   UV: u={u_flipX.item():.2f}, v={v_flipX.item():.2f}")
    print(f"   翻转V后: u={u_flipX.item():.2f}, v_flipped={h - v_flipX.item():.2f}")
    
    # ========== 测试不同方案 ==========
    print(f"\n========== 方案对比 ==========")
    print(f"图像尺寸: {w}x{h}")
    print(f"图像中心: ({w/2}, {h/2})")
    print(f"")
    print(f"方案 | U | V | V翻转 | 备注")
    print(f"直接求逆 | {u.item():.1f} | {v.item():.1f} | {v_flipped.item():.1f} |")
    print(f"翻转X后求逆 | {u_flipX.item():.1f} | {v_flipX.item():.1f} | {(h-v_flipX).item():.1f} |")


def test_get_traj(data_dir, output_dir, num_frames=20, sample_size=(256, 256)):
    """
    测试 get_traj 投影功能
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    print("Loading data...")
    intrinsic_path = os.path.join(data_dir, "head_intrinsic_params.json")
    extrinsic_path = os.path.join(data_dir, "head_extrinsic_params_aligned.json")
    action_path = os.path.join(data_dir, "proprio_stats.h5")
    video_path = os.path.join(data_dir, "head_color.mp4")
    
    intrinsic = load_intrinsic(intrinsic_path)
    c2ws = load_extrinsic(extrinsic_path, num_frames)
    action = load_action(action_path, num_frames)
    video_frames = load_video_frames(video_path, num_frames)
    
    print(f"  Intrinsic shape: {intrinsic.shape}")
    print(f"  Extrinsic (c2w) shape: {c2ws.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Video frames shape: {video_frames.shape}")
    
    # 2. 缩放内参以适应目标分辨率
    h_ori, w_ori = video_frames.shape[1], video_frames.shape[2]
    h_new, w_new = sample_size
    
    h_scale = h_new / h_ori
    w_scale = w_new / w_ori
    
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[0, 0] *= w_scale
    intrinsic_scaled[0, 2] *= w_scale
    intrinsic_scaled[1, 1] *= h_scale
    intrinsic_scaled[1, 2] *= h_scale
    
    print(f"\nScaled intrinsic:")
    print(intrinsic_scaled)
    
    # 3. 准备输入
    # c2ws: (T, 4, 4) -> 需要 (V, T, 4, 4)，这里 V=1
    c2ws_tensor = torch.tensor(c2ws, dtype=torch.float32).unsqueeze(0)  # (1, T, 4, 4)
    
    # 取第一帧的外参用于投影
    c2w_single = c2ws_tensor[:, 0:1, :, :].expand(-1, num_frames, -1, -1)  # (1, T, 4, 4)
    
    action_tensor = torch.tensor(action, dtype=torch.float32)
    intrinsic_tensor = torch.tensor(intrinsic_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
    
    print(f"\nInput shapes for get_traj:")
    print(f"  sample_size: {sample_size}")
    print(f"  pose: {action_tensor.shape}")
    print(f"  w2c (actually c2w): {c2w_single.shape}")
    print(f"  intrinsic: {intrinsic_tensor.shape}")
    
    # 4. 测试 offset=0, 不翻转V
    print("\nCalling get_traj (flipX=True, flipV=False, offset=0.0)...")
    traj_map = get_traj_v2(
        sample_size=sample_size,
        pose=action_tensor,
        c2w_input=c2w_single,
        intrinsic=intrinsic_tensor,
        radius=10,
        flip_x=True,
        flip_v=False,
        eef_offset=0.0
    )
    traj_maps_list = [traj_map]
    offsets = [0.0]
    
    print(f"  Output traj_maps shape: {traj_maps_list[0].shape}")  # (V, T, H, W, 3)
    
    # 5. 缩放视频帧
    video_resized = []
    for frame in video_frames:
        resized = cv2.resize(frame, (w_new, h_new))
        video_resized.append(resized)
    video_resized = np.stack(video_resized, axis=0)
    
    # 6. 叠加可视化并保存
    offsets_str = [f"off={o}" for o in offsets]
    print(f"\nSaving results to {output_dir}...")
    print(f"图片布局: [原始帧 | {' | '.join(offsets_str)}]")
    
    for t in range(num_frames):
        # 原始视频帧
        frame = video_resized[t].copy()
        
        # 叠加不同偏移的轨迹
        def overlay(base, traj):
            mask = (traj.sum(axis=-1) > 150).astype(np.uint8)
            mask = np.stack([mask] * 3, axis=-1)
            result = base.copy()
            result[mask > 0] = traj[mask > 0]
            return result
        
        overlays = [frame]
        for traj_map in traj_maps_list:
            traj = (traj_map[0, t] * 255).astype(np.uint8)
            overlays.append(overlay(frame, traj))
        
        # 保存
        combined = np.hstack(overlays)
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{t:04d}.jpg"),
            combined[:, :, ::-1]  # RGB -> BGR for OpenCV
        )
    
    # 7. 生成视频
    print("Generating video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    num_cols = 1 + len(offsets)
    out = cv2.VideoWriter(
        os.path.join(output_dir, "traj_test.mp4"),
        fourcc, 10.0, (w_new * num_cols, h_new)
    )
    
    for t in range(num_frames):
        frame_path = os.path.join(output_dir, f"frame_{t:04d}.jpg")
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"Done! Results saved to {output_dir}")
    
    # 8. 打印一些调试信息
    print("\n========== DEBUG INFO ==========")
    print(f"First frame action (pose):")
    print(f"  Left pos:  {action[0, 0:3]}")
    print(f"  Left quat: {action[0, 3:7]}")
    print(f"  Left grip: {action[0, 7]}")
    print(f"  Right pos:  {action[0, 8:11]}")
    print(f"  Right quat: {action[0, 11:15]}")
    print(f"  Right grip: {action[0, 15]}")
    
    print(f"\nFirst c2w matrix:")
    print(c2ws[0])
    
    print(f"\nScaled intrinsic:")
    print(intrinsic_scaled)
    
    # 9. 详细调试投影过程
    debug_projection(action_tensor, c2w_single, intrinsic_tensor, sample_size)


if __name__ == "__main__":
    data_dir = "/data/datasets/libero_agibotworld_format/train/task00-ep018-step001"
    output_dir = "/data/zhenyangfan/EVAC/test_traj_output"
    
    test_get_traj(
        data_dir=data_dir,
        output_dir=output_dir,
        num_frames=30,
        sample_size=(256, 256)
    )
