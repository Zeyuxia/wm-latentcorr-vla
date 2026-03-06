import numpy as np
import sapien


class SapienFK:
    """基于 sapien URDF 的双臂正运动学"""
    def __init__(self, urdf_path, root_pos=(0, -0.65, 0), root_quat=(0.707, 0, 0, 0.707)):
        # FK only needs kinematics/physics, not rendering.
        # Prefer a headless scene to avoid GPU render-device dependency..
        self.scene = sapien.Scene()
        
        self.robot = self.scene.create_urdf_loader().load(urdf_path)
        self.robot.set_root_pose(sapien.Pose(list(root_pos), list(root_quat)))
        jnames = [j.get_name() for j in self.robot.get_active_joints()]
        self.jnames = jnames
        self.fl_idx = [jnames.index(f'fl_joint{i}') for i in range(1, 7)]
        self.fr_idx = [jnames.index(f'fr_joint{i}') for i in range(1, 7)]
        self.links = {l.get_name(): l for l in self.robot.get_links()}

    def forward(self, left_q, right_q):
        qpos = np.zeros(len(self.jnames))
        qpos[self.fl_idx] = left_q
        qpos[self.fr_idx] = right_q
        self.robot.set_qpos(qpos)
        result = {}
        for side, prefix in [('left', 'fl'), ('right', 'fr')]:
            pose = self.links[f'{prefix}_link6'].entity_pose
            result[side] = (pose.p.copy(), pose.q.copy())  # pos, quat(wxyz)
        return result


if __name__ == "__main__":
    import argparse, h5py, cv2, imageio, os
    from io import BytesIO
    from PIL import Image

    def quat_dist(q1, q2):
        return 2 * np.arccos(np.clip(np.abs(np.dot(q1/np.linalg.norm(q1), q2/np.linalg.norm(q2))), 0, 1))

    def project_point(pt, K, E):
        p = E @ np.append(pt, 1.0)
        if p[2] <= 0: return None
        uv = K @ p[:3]; return int(uv[0]/uv[2]), int(uv[1]/uv[2])

    def draw_marker(img, pos, color, sz=4):
        if pos and 0 <= pos[0] < img.shape[1] and 0 <= pos[1] < img.shape[0]:
            cv2.circle(img, pos, sz, color, -1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/data/zhenyangfan/RoboTwin/data/open_laptop/demo_clean/data/episode0.hdf5')
    parser.add_argument('--urdf', default='/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--output', default='./fk_verify.mp4')
    args = parser.parse_args()

    with h5py.File(args.data, 'r') as f:
        left_eef, right_eef = f['endpose/left_endpose'][:], f['endpose/right_endpose'][:]
        left_q, right_q = f['joint_action/left_arm'][:], f['joint_action/right_arm'][:]
        if args.vis:
            intrinsics, extrinsics = f['observation/head_camera/intrinsic_cv'][:], f['observation/head_camera/extrinsic_cv'][:]
            rgb_encoded = f['observation/head_camera/rgb'][:]

    n = len(left_eef) if args.n < 0 else min(args.n, len(left_eef))
    fk = SapienFK(args.urdf)

    errs = {'left_pos': [], 'left_rot': [], 'right_pos': [], 'right_rot': []}
    frames = []

    for i in range(n):
        result = fk.forward(left_q[i], right_q[i])
        for side, eef in [('left', left_eef), ('right', right_eef)]:
            fk_pos, fk_q = result[side]
            errs[f'{side}_pos'].append(np.linalg.norm(fk_pos - eef[i, :3]))
            errs[f'{side}_rot'].append(quat_dist(fk_q, eef[i, 3:7]))

        if args.vis:
            frame = np.array(Image.open(BytesIO(rgb_encoded[i])))
            E = np.vstack([extrinsics[i], [0, 0, 0, 1]])
            for side, eef in [('left', left_eef), ('right', right_eef)]:
                draw_marker(frame, project_point(result[side][0], intrinsics[i], E), (0, 255, 0))
                draw_marker(frame, project_point(eef[i, :3], intrinsics[i], E), (255, 0, 0))
            pos_err = (errs['left_pos'][-1] + errs['right_pos'][-1]) / 2 * 1000
            cv2.putText(frame, f"F{i} Err:{pos_err:.1f}mm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            frames.append(frame)

    if args.vis and frames:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        imageio.mimsave(args.output, frames, fps=30)
        print(f"✓ 视频已保存: {args.output}")

    print(f"\nFK误差 ({n}帧):")
    for side in ['left', 'right']:
        p, r = np.array(errs[f'{side}_pos']), np.degrees(errs[f'{side}_rot'])
        print(f"  {side}: {np.mean(p)*1000:.3f}mm, {np.mean(r):.4f}°")
    all_p = errs['left_pos'] + errs['right_pos']
    print(f"  总体: {np.mean(all_p)*1000:.3f}mm")