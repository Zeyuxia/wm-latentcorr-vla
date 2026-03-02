#!/usr/bin/env python3
"""验证FK误差 (sapien版) - 使用RoboTwin相同的引擎，支持可视化投影"""
# python script/verify_fk_sapien.py --vis

import argparse
import numpy as np
import h5py
import sapien
import cv2
import imageio
import base64
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore")


def quat_dist(q1, q2):
    """四元数角度距离(弧度), wxyz格式"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1/np.linalg.norm(q1), q2/np.linalg.norm(q2))), 0, 1))


def project_point(point_3d, intrinsic, extrinsic):
    """将3D点投影到图像平面"""
    point_homo = np.append(point_3d, 1.0)
    point_cam = extrinsic @ point_homo
    point_cam = point_cam[:3]
    if point_cam[2] <= 0:
        return None
    point_img = intrinsic @ point_cam
    u, v = point_img[0] / point_img[2], point_img[1] / point_img[2]
    return int(u), int(v)


def draw_marker(img, pos, color, size=5):
    """在图像上绘制标记"""
    if pos is None:
        return
    u, v = pos
    h, w = img.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(img, (u, v), size, color, -1)
        cv2.circle(img, (u, v), size+1, (255, 255, 255), 1)


def decode_image(encoded_bytes):
    """解码JPEG图像字节"""
    img = Image.open(BytesIO(encoded_bytes))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/open_laptop/demo_clean/data/episode0.hdf5')
    parser.add_argument('--urdf', default='assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')
    parser.add_argument('-n', type=int, default=-1, help='采样数,-1全部')
    parser.add_argument('--vis', action='store_true', help='生成可视化视频')
    parser.add_argument('--output', default='eval_result/fk_verify.mp4', help='输出视频路径')
    args = parser.parse_args()

    # 加载数据
    with h5py.File(args.data, 'r') as f:
        left_eef, right_eef = f['endpose/left_endpose'][:], f['endpose/right_endpose'][:]
        left_q, right_q = f['joint_action/left_arm'][:], f['joint_action/right_arm'][:]
        
        # 加载相机参数和图像（用于可视化）
        if args.vis:
            intrinsics = f['observation/head_camera/intrinsic_cv'][:]  # (T, 3, 3)
            extrinsics = f['observation/head_camera/extrinsic_cv'][:]  # (T, 3, 4)
            rgb_encoded = f['observation/head_camera/rgb'][:]  # encoded strings
    
    n = len(left_eef) if args.n < 0 else min(args.n, len(left_eef))
    
    # 加载机器人
    scene = sapien.Scene()
    robot = scene.create_urdf_loader().load(args.urdf)
    robot.set_root_pose(sapien.Pose([0, -0.65, 0], [0.707, 0, 0, 0.707]))
    
    # 获取关节索引和link
    jnames = [j.get_name() for j in robot.get_active_joints()]
    fl_idx = [jnames.index(f'fl_joint{i}') for i in range(1,7)]
    fr_idx = [jnames.index(f'fr_joint{i}') for i in range(1,7)]
    links = {l.get_name(): l for l in robot.get_links()}
    
    # 计算误差
    errs = {'left_pos': [], 'left_rot': [], 'right_pos': [], 'right_rot': []}
    frames = []
    
    for i in range(n):

        # 1
        qpos = np.zeros(len(jnames))
        qpos[fl_idx], qpos[fr_idx] = left_q[i], right_q[i]
        robot.set_qpos(qpos)
        
        fk_positions = {}
        for side, idx, eef in [('left', fl_idx, left_eef), ('right', fr_idx, right_eef)]:
            link = links[f'f{"l" if side=="left" else "r"}_link6']
            pose = link.entity_pose
            fk_pos, fk_q = pose.p, pose.q  # sapien: wxyz
            gt_pos, gt_q = eef[i, :3], eef[i, 3:7]
            errs[f'{side}_pos'].append(np.linalg.norm(fk_pos - gt_pos))
            errs[f'{side}_rot'].append(quat_dist(fk_q, gt_q))
            fk_positions[side] = fk_pos
        
        # 

        # 可视化
        if args.vis:
            # 解码图像
            frame = decode_image(rgb_encoded[i])
            intrinsic = intrinsics[i]
            extrinsic_3x4 = extrinsics[i]
            extrinsic = np.vstack([extrinsic_3x4, [0, 0, 0, 1]])
            
            # 投影 FK 位置（绿色）和 GT 位置（红色）
            for side in ['left', 'right']:
                fk_pos = fk_positions[side]
                gt_pos = left_eef[i, :3] if side == 'left' else right_eef[i, :3]
                
                fk_proj = project_point(fk_pos, intrinsic, extrinsic)
                gt_proj = project_point(gt_pos, intrinsic, extrinsic)
                
                draw_marker(frame, fk_proj, (0, 255, 0), 4)  # 绿色 - FK
                draw_marker(frame, gt_proj, (255, 0, 0), 4)   # 红色 - GT
                
                # 连线
                if fk_proj and gt_proj:
                    cv2.line(frame, fk_proj, gt_proj, (255, 255, 0), 1)
            
            # 添加文字
            pos_err = (errs['left_pos'][-1] + errs['right_pos'][-1]) / 2 * 1000
            cv2.putText(frame, f"Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(frame, f"Frame {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Pos Err: {pos_err:.2f}mm", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(frame, f"Pos Err: {pos_err:.2f}mm", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, "Green=FK Red=GT", (10, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
            cv2.putText(frame, "Green=FK Red=GT", (10, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            
            frames.append(frame)
    
    # 保存视频
    if args.vis and frames:
        import os
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        imageio.mimsave(args.output, frames, fps=30)
        print(f"\n✓ 可视化视频已保存: {args.output}")
    
    # 输出结果
    print(f"\n{'='*50}\nFK误差统计 (sapien) - {n}帧\n{'='*50}")
    for side in ['left', 'right']:
        pos, rot = np.array(errs[f'{side}_pos']), np.degrees(errs[f'{side}_rot'])
        print(f"{side}: 位置={np.mean(pos)*1000:.3f}mm(max {np.max(pos)*1000:.2f}), "
              f"旋转={np.mean(rot):.4f}°(max {np.max(rot):.2f}°)")
    
    all_pos = errs['left_pos'] + errs['right_pos']
    all_rot = errs['left_rot'] + errs['right_rot']
    print(f"\n总体: {np.mean(all_pos)*1000:.3f}mm, {np.degrees(np.mean(all_rot)):.4f}°")
    print("✓ 误差可接受" if np.mean(all_pos)<0.01 and np.degrees(np.mean(all_rot))<5 else "✗ 误差较大")


if __name__ == "__main__":
    main()