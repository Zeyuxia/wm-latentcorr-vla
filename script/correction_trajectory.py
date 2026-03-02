"""
生成满足动力学约束的纠错轨迹（可选避障）
使用 TOPP (Time-Optimal Path Parameterization) 算法

用法示例:
    from correction_trajectory import CorrectionTrajectoryGenerator
    
    generator = CorrectionTrajectoryGenerator(task_env)
    trajectory = generator.generate(
        start_qpos=current_state,
        end_qpos=target_state,
        num_steps=50,
        collision_free=True  # 启用避障
    )
"""

import sys
sys.path.append("./")

import numpy as np
import toppra as ta
from scipy.interpolate import interp1d
import sapien.core as sapien

# ruckig 

class CorrectionTrajectoryGenerator:
    """
    生成满足动力学约束的纠错轨迹（可选避障）
    
    支持三种模式:
    1. toppra: 只满足动力学约束，不避障
    2. mplib_topp: 使用 mplib TOPP，只满足动力学约束
    3. collision_free: 使用 CuRobo/mplib 规划器生成避障路径 + TOPP
    """
    
    def __init__(self, task_env=None, planner=None):
        """
        Args:
            task_env: RoboTwin 任务环境 (用于避障规划)
            planner: mplib planner (可选，如果不提供 task_env)
        """
        ta.setup_logging("CRITICAL")  # 隐藏日志
        
        self.task_env = task_env
        self.robot = None
        
        if task_env is not None:
            self.robot = task_env.robot
            self.left_planner = task_env.robot.left_mplib_planner
            self.right_planner = task_env.robot.right_mplib_planner
            # CuRobo 规划器（用于避障）
            self.left_curobo = getattr(task_env.robot, 'left_planner', None)
            self.right_curobo = getattr(task_env.robot, 'right_planner', None)
        elif planner is not None:
            self.left_planner = planner
            self.right_planner = planner
            self.left_curobo = None
            self.right_curobo = None
        else:
            self.left_planner = None
            self.right_planner = None
            self.left_curobo = None
            self.right_curobo = None
    
    def qpos_to_pose(self, qpos, arm_tag="left"):
        """
        通过正向运动学将关节角转换为末端位姿
        需要 task_env 支持
        """
        if self.robot is None:
            raise ValueError("Need task_env to compute forward kinematics")
        
        # 保存当前状态
        if arm_tag == "left":
            entity = self.robot.left_entity
        else:
            entity = self.robot.right_entity
        
        original_qpos = entity.get_qpos()
        
        # 构建完整的qpos（包括夹爪关节）
        full_qpos = original_qpos.copy()
        full_qpos[:len(qpos)] = qpos  # 只更新手臂关节
        
        # 设置新的关节角并获取末端位姿
        entity.set_qpos(full_qpos)
        self.task_env.scene.step()  # 更新
        
        if arm_tag == "left":
            pose = self.robot.get_left_tcp_pose()  # [x,y,z,w,qx,qy,qz]
        else:
            pose = self.robot.get_right_tcp_pose()
        
        # 恢复原状态
        entity.set_qpos(original_qpos)
        
        # pose 是 [x, y, z, w, qx, qy, qz] 列表格式
        # left_plan_path 期望接收列表或数组，不是 sapien.Pose
        return pose  # 直接返回列表
    
    def generate_collision_free(self, start_qpos, end_qpos, arm_tag="left", num_steps=50):
        """
        使用 CuRobo 生成避障轨迹
        
        工作流程:
        1. 将 end_qpos 转换为末端位姿
        2. 使用 CuRobo 从 start_qpos 规划到目标位姿（考虑碰撞）
        3. 返回无碰撞轨迹
        
        Args:
            start_qpos: 起始关节角 (6,)
            end_qpos: 目标关节角 (6,)
            arm_tag: "left" 或 "right"
            num_steps: 输出步数
            
        Returns:
            dict: 包含 positions, velocities, success 等
        """
        if self.robot is None:
            return {'success': False, 'error': 'Need task_env for collision-free planning'}
        
        # 获取目标位姿
        target_pose = self.qpos_to_pose(end_qpos, arm_tag)
        
        # 构建完整的起始qpos（包括夹爪关节）
        if arm_tag == "left":
            entity = self.robot.left_entity
        else:
            entity = self.robot.right_entity
        
        current_full_qpos = entity.get_qpos()
        start_full_qpos = current_full_qpos.copy()
        start_full_qpos[:len(start_qpos)] = start_qpos
        
        # 使用 CuRobo 规划，传入 last_qpos 以使用指定的起始位置
        if arm_tag == "left":
            result = self.robot.left_plan_path(target_pose, last_qpos=start_full_qpos)
        else:
            result = self.robot.right_plan_path(target_pose, last_qpos=start_full_qpos)
        
        if result.get("status") != "Success":
            return {'success': False, 'error': f'CuRobo planning failed: {result.get("status")}'}
        
        positions = result["position"]
        velocities = result["velocity"]
        
        # CuRobo 返回的是 active joints，需要取前6个(手臂关节)
        if positions.shape[1] > 6:
            positions = positions[:, :6]
            velocities = velocities[:, :6]
        
        # 重采样到指定步数（如果指定了 num_steps）
        n_steps = positions.shape[0]
        if num_steps is not None and n_steps != num_steps:
            indices = np.linspace(0, n_steps - 1, num_steps).astype(int)
            positions = positions[indices]
            # 重新计算速度
            dt = 1 / 250
            velocities = np.gradient(positions, dt, axis=0)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'success': True,
            'collision_free': True
        }
    
    def generate_with_topp(self, start_qpos, end_qpos, arm_tag="left", dt=1/250):
        """
        使用 mplib TOPP 生成满足动力学约束的轨迹
        
        Args:
            start_qpos: 起始关节角 (numpy array)
            end_qpos: 目标关节角 (numpy array)
            arm_tag: "left" 或 "right"
            dt: 时间步长
            
        Returns:
            dict: {
                'times': 时间戳数组,
                'positions': 关节位置序列 (n_steps, n_joints),
                'velocities': 关节速度序列 (n_steps, n_joints),
                'accelerations': 关节加速度序列,
                'duration': 总时长,
                'success': 是否成功
            }
        """
        planner = self.left_planner if arm_tag == "left" else self.right_planner
        
        if planner is None:
            raise ValueError("Planner not initialized. Provide task_env or planner.")
        
        # 构建路径 (至少2个点)
        path = np.vstack([start_qpos, end_qpos])
        
        try:
            times, positions, velocities, accelerations, duration = planner.TOPP(
                path, 
                dt,
                verbose=True
            )
            return {
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'duration': duration,
                'success': True
            }
        except Exception as e:
            print(f"TOPP failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_with_toppra(self, start_qpos, end_qpos, 
                              vel_limits=None, acc_limits=None,
                              num_waypoints=10, dt=1/250):
        """
        使用 toppra 库直接生成满足动力学约束的轨迹
        不需要 task_env，可以独立使用
        
        Args:
            start_qpos: 起始关节角 (numpy array)
            end_qpos: 目标关节角 (numpy array)
            vel_limits: 关节速度限制 (n_joints,) 或 (n_joints, 2) for [min, max]
            acc_limits: 关节加速度限制
            num_waypoints: 路径点数量
            dt: 输出轨迹的时间步长
            
        Returns:
            dict: 包含 positions, velocities, times 等
        """
        n_joints = len(start_qpos)
        
        # 默认限制
        if vel_limits is None:
            vel_limits = np.ones(n_joints) * 2.0  # 默认 2 rad/s
        if acc_limits is None:
            acc_limits = np.ones(n_joints) * 5.0  # 默认 5 rad/s^2
            
        # 构建路径点 (使用多个中间点可以更好地控制)
        ss = np.linspace(0, 1, num_waypoints)
        waypoints = np.array([start_qpos + s * (end_qpos - start_qpos) for s in ss])
        
        # 创建样条路径
        path = ta.SplineInterpolator(ss, waypoints)
        
        # 设置约束
        if vel_limits.ndim == 1:
            vel_constraint = ta.constraint.JointVelocityConstraint(
                np.column_stack([-vel_limits, vel_limits])
            )
        else:
            vel_constraint = ta.constraint.JointVelocityConstraint(vel_limits)
            
        if acc_limits.ndim == 1:
            acc_constraint = ta.constraint.JointAccelerationConstraint(
                np.column_stack([-acc_limits, acc_limits])
            )
        else:
            acc_constraint = ta.constraint.JointAccelerationConstraint(acc_limits)
        
        # TOPPRA 算法
        instance = ta.algorithm.TOPPRA(
            [vel_constraint, acc_constraint],
            path
        )
        
        try:
            jnt_traj = instance.compute_trajectory()
            
            if jnt_traj is None:
                return {'success': False, 'error': 'TOPPRA failed to find solution'}
            
            # 采样轨迹
            duration = jnt_traj.duration
            times = np.arange(0, duration, dt)
            if times[-1] < duration:
                times = np.append(times, duration)
            
            positions = jnt_traj(times)
            velocities = jnt_traj(times, 1)  # 一阶导数
            accelerations = jnt_traj(times, 2)  # 二阶导数
            
            return {
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'duration': duration,
                'success': True
            }
        except Exception as e:
            print(f"TOPPRA failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def resample_trajectory(self, trajectory, num_steps):
        """
        将轨迹重采样到指定步数
        
        Args:
            trajectory: generate_* 返回的轨迹字典
            num_steps: 目标步数
            
        Returns:
            dict: 重采样后的轨迹
        """
        if not trajectory['success']:
            return trajectory
        
        times = trajectory['times']
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        
        # 创建插值函数
        pos_interp = interp1d(times, positions, axis=0, kind='cubic')
        vel_interp = interp1d(times, velocities, axis=0, kind='cubic')
        
        # 新的时间点
        new_times = np.linspace(times[0], times[-1], num_steps)
        
        return {
            'times': new_times,
            'positions': pos_interp(new_times),
            'velocities': vel_interp(new_times),
            'duration': trajectory['duration'],
            'success': True
        }
    
    def generate(self, start_qpos, end_qpos, num_steps=None, 
                 method='toppra', arm_tag='left',
                 vel_limits=None, acc_limits=None,
                 collision_free=False):
        """
        生成纠错轨迹的主接口
        
        Args:
            start_qpos: 起始关节角
            end_qpos: 目标关节角
            num_steps: 输出步数，如果为 None 则使用动力学最优步数（不重采样）
            method: 'toppra' 或 'mplib_topp' (当 collision_free=False 时使用)
            arm_tag: 'left' 或 'right'
            vel_limits: 速度限制 (toppra 使用)
            acc_limits: 加速度限制 (toppra 使用)
            collision_free: 是否生成避障轨迹 (需要 task_env)
            
        Returns:
            numpy array: (n_steps, n_joints) 的轨迹，n_steps 取决于 num_steps 参数
        """
        start_qpos = np.array(start_qpos)
        end_qpos = np.array(end_qpos)
        
        # 避障模式：使用 CuRobo
        if collision_free:
            # 传递 num_steps，如果为 None 则使用 CuRobo 返回的最优步数
            traj = self.generate_collision_free(start_qpos, end_qpos, arm_tag, num_steps)
            if traj['success']:
                return traj['positions']
            else:
                print(f"Warning: Collision-free planning failed ({traj.get('error')}), falling back to toppra")
                # 降级到 toppra
        
        # 非避障模式：使用 toppra 或 mplib_topp
        if method == 'toppra':
            traj = self.generate_with_toppra(
                start_qpos, end_qpos,
                vel_limits=vel_limits,
                acc_limits=acc_limits
            )
        elif method == 'mplib_topp':
            traj = self.generate_with_topp(start_qpos, end_qpos, arm_tag=arm_tag)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not traj['success']:
            print(f"Warning: Trajectory generation failed, falling back to linear interpolation")
            # 降级为线性插值
            fallback_steps = num_steps if num_steps is not None else 50
            return np.linspace(start_qpos, end_qpos, fallback_steps)
        
        # 如果指定了步数，重采样；否则返回原始最优轨迹
        if num_steps is not None:
            resampled = self.resample_trajectory(traj, num_steps)
            return resampled['positions']
        else:
            # 返回动力学最优的原始轨迹
            return traj['positions']
    
    def generate_bimanual(self, left_start, left_end, right_start, right_end, 
                          num_steps=50, vel_limits=None, acc_limits=None,
                          collision_free=False):
        """
        生成双臂纠错轨迹
        
        Args:
            left_start, left_end: 左臂起始和目标关节角
            right_start, right_end: 右臂起始和目标关节角
            num_steps: 输出步数
            vel_limits: 速度限制
            acc_limits: 加速度限制
            collision_free: 是否避障
        
        Returns:
            dict: {
                'left': (num_steps, n_joints) 左臂轨迹,
                'right': (num_steps, n_joints) 右臂轨迹,
                'combined': (num_steps, 2*n_joints) 合并轨迹
            }
        """
        left_traj = self.generate(
            left_start, left_end, num_steps,
            arm_tag='left',
            vel_limits=vel_limits, acc_limits=acc_limits,
            collision_free=collision_free
        )
        right_traj = self.generate(
            right_start, right_end, num_steps,
            arm_tag='right',
            vel_limits=vel_limits, acc_limits=acc_limits,
            collision_free=collision_free
        )
        
        return {
            'left': left_traj,
            'right': right_traj,
            'combined': np.hstack([left_traj, right_traj])
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例 1: 独立使用 toppra (不需要环境，不避障)
    print("=" * 50)
    print("示例 1: 使用 toppra 生成纠错轨迹（仅动力学约束）")
    print("=" * 50)
    
    generator = CorrectionTrajectoryGenerator()
    
    # 模拟 7 自由度机械臂
    start_state = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])
    target_state = np.array([0.5, -0.3, 0.2, -1.2, 0.1, 0.8, 0.2])
    
    # 设置动力学约束
    vel_limits = np.array([2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # rad/s
    acc_limits = np.array([5.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0])  # rad/s^2
    
    trajectory = generator.generate(
        start_state, 
        target_state, 
        num_steps=50,
        vel_limits=vel_limits,
        acc_limits=acc_limits
    )
    
    print(f"生成轨迹形状: {trajectory.shape}")
    print(f"起始状态: {trajectory[0]}")
    print(f"目标状态: {trajectory[-1]}")
    
    # 验证速度约束
    dt = 1/50  # 假设 50Hz
    velocities = np.diff(trajectory, axis=0) / dt
    max_vel = np.max(np.abs(velocities), axis=0)
    print(f"最大速度: {max_vel}")
    print(f"速度限制: {vel_limits}")
    print(f"满足速度约束: {np.all(max_vel <= vel_limits * 1.1)}")  # 允许 10% 误差
    
    # 示例 2: 避障模式使用说明
    print("\n" + "=" * 50)
    print("示例 2: 使用 CuRobo 生成避障纠错轨迹")
    print("=" * 50)
    print("""
    # 需要 RoboTwin 环境（CuRobo 会考虑桌面等障碍物）:
    
    from script.correction_trajectory import CorrectionTrajectoryGenerator
    
    # 初始化（使用任务环境）
    generator = CorrectionTrajectoryGenerator(task_env=TASK_ENV)
    
    # 获取当前状态
    current_state = TASK_ENV.robot.get_left_arm_jointState()[:-1]
    target_state = desired_state
    
    # 生成 50 步的避障纠错轨迹
    trajectory = generator.generate(
        current_state, 
        target_state, 
        num_steps=50,
        arm_tag='left',
        collision_free=True  # ← 启用避障！
    )
    
    # 执行轨迹
    for state in trajectory:
        TASK_ENV.robot.set_arm_joints(state, arm_tag='left')
        TASK_ENV.scene.step()
    """)
    
    print("\n" + "=" * 50)
    print("三种模式对比")
    print("=" * 50)
    print("""
    ┌─────────────────┬────────────┬────────────┬─────────────┐
    │ 模式            │ 动力学约束 │ 避障       │ 需要环境    │
    ├─────────────────┼────────────┼────────────┼─────────────┤
    │ toppra          │ ✅         │ ❌         │ ❌          │
    │ mplib_topp      │ ✅         │ ❌         │ ✅          │
    │ collision_free  │ ✅         │ ✅ (CuRobo)│ ✅          │
    └─────────────────┴────────────┴────────────┴─────────────┘
    
    推荐用法:
    - 数据增强/离线生成: toppra（快，不需要环境）
    - 在线纠错（无障碍物）: toppra 或 mplib_topp
    - 在线纠错（有障碍物）: collision_free=True
    """)
