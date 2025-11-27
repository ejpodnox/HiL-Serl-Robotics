#!/usr/bin/env python3
"""
独立的 VisionPro 遥操作数据采集程序

特点：
1. 不依赖 KinovaEnv，直接使用 RobotCommander
2. 快速启动，用于测试和数据采集
3. 保存原始遥操作数据（可选择保存为 HIL-SERL 格式）

使用方法:
    python record_teleop_demos.py --save_dir ./teleop_demos --num_demos 5
"""

import argparse
import rclpy
import numpy as np
import pickle
import time
from pathlib import Path
import yaml

from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.coordinate_mapper import CoordinateMapper
from kinova_rl_env.kinova_env.kinova_interface import KinovaInterface
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


class TeleopDataRecorder:
    """遥操作数据记录器（解耦版本）"""

    def __init__(self, config_file: str):
        """
        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # 初始化 ROS2
        rclpy.init()

        # 初始化组件
        self.vp_bridge = VisionProBridge(
            avp_ip=self.config['visionpro']['ip'],
            use_right_hand=self.config['visionpro']['use_right_hand']
        )

        # 使用 KinovaInterface（已验证可用）
        self.interface = KinovaInterface(node_name='teleop_recorder')
        self.interface.connect()

        # 等待发布者被 ROS2 发现
        print("  等待 ROS2 连接...")
        time.sleep(0.5)

        # 等待关节状态数据
        print("  等待关节状态...")
        end_time = time.time() + 3.0
        while time.time() < end_time:
            rclpy.spin_once(self.interface.node, timeout_sec=0.1)
            if self.interface.get_joint_state() is not None:
                print("  ✓ 关节状态已就绪")
                break

        # 加载标定文件
        calibration_file = Path(__file__).parent / self.config['calibration']['file']
        if not calibration_file.exists():
            print("\n" + "=" * 60)
            print("✗ 标定文件不存在！")
            print("=" * 60)
            print(f"缺失文件: {calibration_file}")
            print("\n需要先运行标定流程生成标定文件。请执行:")
            print("  python vision_pro_control/nodes/teleop_node.py")
            print("\n或者运行快速标定:")
            print("  python tools/calibrate_visionpro.py")
            print("=" * 60)
            raise FileNotFoundError(f"标定文件不存在: {calibration_file}")

        self.mapper = CoordinateMapper(calibration_file=calibration_file)

        # 设置参数
        mapper_cfg = self.config['mapper']
        self.mapper.set_gains(
            position_gain=mapper_cfg['position_gain'],
            rotation_gain=mapper_cfg['rotation_gain']
        )
        self.mapper.set_velocity_limits(
            max_linear=mapper_cfg['max_linear_velocity'],
            max_angular=mapper_cfg['max_angular_velocity']
        )

        # 控制频率
        self.control_frequency = 50  # Hz
        self.dt = 1.0 / self.control_frequency

        print("✓ 遥操作记录器初始化完成")

    def record_trajectory(self):
        """
        记录一条遥操作轨迹

        Returns:
            trajectory: list of dicts，包含每个时间步的数据
        """
        trajectory = []

        print("\n>>> 开始记录，按 'q' 停止 <<<")

        start_time = time.time()
        step = 0

        with KeyboardMonitor() as kb:
            while True:
                loop_start = time.time()

                # 检查按键
                key = kb.get_key(timeout=0.001)
                if key == 'q':
                    print("\n>>> 记录停止 <<<")
                    break

                try:
                    # 1. Spin 接收关节状态
                    rclpy.spin_once(self.interface.node, timeout_sec=0.001)

                    # 2. 获取 VisionPro 数据
                    position, rotation = self.vp_bridge.get_hand_relative_to_head()
                    pinch_distance = self.vp_bridge.get_pinch_distance()

                    # 3. 映射到 Twist（字典格式）
                    twist = self.mapper.map_to_twist(position, rotation)

                    # 4. 转换为关节速度并发送
                    twist_array = np.array([
                        twist['linear']['x'], twist['linear']['y'], twist['linear']['z'],
                        twist['angular']['x'], twist['angular']['y'], twist['angular']['z']
                    ])
                    joint_velocities = self._twist_to_joint_velocity(twist_array)
                    self.interface.send_joint_velocities(joint_velocities.tolist())

                    # 4. 控制夹爪
                    gripper_position = self._pinch_to_gripper(pinch_distance)
                    self.interface.send_gripper_command(gripper_position)

                    # 6. 记录数据
                    data_point = {
                        'timestamp': time.time() - start_time,
                        'visionpro': {
                            'hand_position': position.copy(),
                            'hand_rotation': rotation.copy(),
                            'pinch_distance': pinch_distance
                        },
                        'robot': {
                            'twist': twist.copy(),
                            'gripper_position': gripper_position
                        }
                    }
                    trajectory.append(data_point)

                    # 5. 打印状态（每秒一次）
                    if step % 50 == 0:
                        joint_state = self.interface.get_joint_state()
                        q = joint_state[0] if joint_state else None  # 元组的第一个元素是 positions
                        vx = twist['linear']['x']
                        vy = twist['linear']['y']
                        vz = twist['linear']['z']

                        print(f"  [{step:4d}] {time.time() - start_time:.1f}s | "
                              f"手[{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}] | "
                              f"速度[{vx:.3f},{vy:.3f},{vz:.3f}]")

                        if q is not None:
                            print(f"        关节: " + " ".join([f"{j:5.2f}" for j in q]))

                    step += 1

                except Exception as e:
                    print(f"记录错误: {e}")
                    break

                # 维持控制频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)

        return trajectory

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        精确雅可比矩阵（基于 Kinova Gen3 7-DOF URDF）

        使用几何雅可比：J = [J_v; J_ω]
        - J_v[i] = z_i × (p_ee - p_i)  (线速度)
        - J_ω[i] = z_i                 (角速度)

        Args:
            q: 关节位置 (7,)

        Returns:
            J: 雅可比矩阵 (6, 7)
        """
        # URDF 精确参数（gen3/7dof/urdf/gen3_macro.xacro）
        d1 = 0.15643
        d2 = 0.12838
        d3 = 0.21038
        d4 = 0.21038
        d5 = 0.20843
        d6 = 0.10593
        d7 = 0.10593
        d_ee = 0.061525

        # 齐次变换矩阵工具函数
        def rot_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        def rot_x(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

        def trans(x, y, z):
            return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

        # 构建变换链
        T0 = np.eye(4)
        T1 = T0 @ trans(0, 0, d1) @ rot_x(np.pi) @ rot_z(q[0])
        T2 = T1 @ trans(0, 0.005375, -d2) @ rot_x(np.pi/2) @ rot_z(q[1])
        T3 = T2 @ trans(0, -d3, -0.006375) @ rot_x(-np.pi/2) @ rot_z(q[2])
        T4 = T3 @ trans(0, 0.006375, -d4) @ rot_x(np.pi/2) @ rot_z(q[3])
        T5 = T4 @ trans(0, -d5, -0.006375) @ rot_x(-np.pi/2) @ rot_z(q[4])
        T6 = T5 @ trans(0, 0.00017505, -d6) @ rot_x(np.pi/2) @ rot_z(q[5])
        T7 = T6 @ trans(0, -d7, -0.00017505) @ rot_x(-np.pi/2) @ rot_z(q[6])
        T_ee = T7 @ trans(0, 0, -d_ee) @ rot_x(np.pi)

        # 提取位置和旋转轴
        p = [T0[:3, 3], T1[:3, 3], T2[:3, 3], T3[:3, 3],
             T4[:3, 3], T5[:3, 3], T6[:3, 3], T7[:3, 3]]
        p_ee = T_ee[:3, 3]

        z = [T0[:3, 2], T1[:3, 2], T2[:3, 2], T3[:3, 2],
             T4[:3, 2], T5[:3, 2], T6[:3, 2], T7[:3, 2]]

        # 计算几何雅可比
        J = np.zeros((6, 7))
        for i in range(7):
            J[:3, i] = np.cross(z[i], p_ee - p[i])  # 线速度
            J[3:, i] = z[i]                          # 角速度

        return J

    def _twist_to_joint_velocity(self, twist: np.ndarray) -> np.ndarray:
        """
        Twist → 关节速度（精确雅可比逆运动学）

        Args:
            twist: [vx, vy, vz, wx, wy, wz]
        Returns:
            joint_velocities: (7,)
        """
        # 获取当前关节状态
        joint_state = self.interface.get_joint_state()
        if joint_state is None:
            return np.zeros(7)

        q = joint_state[0]  # positions

        # 计算精确雅可比
        J = self._compute_jacobian(q)

        # 阻尼最小二乘伪逆（DLS）避免奇异性
        lambda_damping = 0.01
        JJT = J @ J.T + lambda_damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)

        # 计算关节速度
        joint_vel = J_pinv @ twist

        # 安全限制
        max_vel = 0.2  # rad/s
        joint_vel = np.clip(joint_vel, -max_vel, max_vel)

        return joint_vel

    def _pinch_to_gripper(self, pinch_distance: float) -> float:
        """
        将捏合距离映射到夹爪位置

        Args:
            pinch_distance: 捏合距离 (m)
        Returns:
            gripper_position: 夹爪位置 [0, 1]
        """
        gripper_cfg = self.config['gripper']

        if gripper_cfg['control_mode'] == 'continuous':
            # 连续映射
            pinch_open = gripper_cfg['pinch_distance_open']
            pinch_close = gripper_cfg['pinch_distance_close']
            gripper_open = gripper_cfg['gripper_open_position']
            gripper_close = gripper_cfg['gripper_close_position']

            pinch_distance = np.clip(pinch_distance, pinch_close, pinch_open)
            normalized = (pinch_distance - pinch_close) / (pinch_open - pinch_close)
            gripper_position = gripper_close + (gripper_open - gripper_close) * normalized

            return float(gripper_position)
        else:
            # 二值映射
            threshold = gripper_cfg['pinch_threshold']
            if pinch_distance < threshold:
                return gripper_cfg['close_position']
            else:
                return gripper_cfg['open_position']

    def start(self):
        """启动 VisionPro 数据流"""
        self.vp_bridge.start()
        time.sleep(1.0)
        print("✓ VisionPro 数据流已启动")

    def stop(self):
        """停止所有组件"""
        self.vp_bridge.stop()
        self.interface.send_joint_velocities([0.0] * 7)
        self.interface.disconnect()
        rclpy.shutdown()
        print("✓ 已停止所有组件")


def save_trajectory(trajectory, save_path, metadata=None):
    """
    保存轨迹数据

    Args:
        trajectory: list of dicts
        save_path: 保存路径
        metadata: 额外的元数据（可选）
    """
    data = {
        'trajectory': trajectory,
        'metadata': metadata or {},
        'num_steps': len(trajectory),
        'duration': trajectory[-1]['timestamp'] if trajectory else 0.0
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='VisionPro 遥操作数据采集')
    parser.add_argument('--save_dir', type=str, default='./teleop_demos',
                        help='保存目录')
    parser.add_argument('--num_demos', type=int, default=5,
                        help='需要收集的demo数量')
    parser.add_argument('--config', type=str,
                        default='vision_pro_control/config/teleop_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--task_name', type=str, default='teleop',
                        help='任务名称（用于文件命名）')

    args = parser.parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir) / args.task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VisionPro 遥操作数据采集（独立版本）")
    print("=" * 60)
    print(f"保存目录: {save_dir}")
    print(f"目标数量: {args.num_demos}")
    print("=" * 60)

    try:
        # 初始化记录器
        recorder = TeleopDataRecorder(config_file=args.config)
        recorder.start()

        print("\n按键说明:")
        print("  Enter - 开始记录新的演示")
        print("  q     - 停止当前记录")
        print("  Ctrl+C - 退出程序")
        print("=" * 60)

        demo_count = 0

        while demo_count < args.num_demos:
            print(f"\n【Demo #{demo_count + 1}/{args.num_demos}】")
            print("按 Enter 开始...")
            input()

            # 记录轨迹
            trajectory = recorder.record_trajectory()

            if len(trajectory) == 0:
                print("✗ 轨迹为空，跳过")
                continue

            # 询问是否保存
            print(f"\n轨迹长度: {len(trajectory)} 步")
            print(f"时长: {trajectory[-1]['timestamp']:.2f} 秒")
            save_choice = input("保存此演示? (y/n): ").strip().lower()

            if save_choice == 'y':
                # 保存
                demo_path = save_dir / f"demo_{demo_count:03d}.pkl"
                metadata = {
                    'demo_id': demo_count,
                    'task_name': args.task_name,
                    'timestamp': time.time()
                }
                save_trajectory(trajectory, demo_path, metadata)
                demo_count += 1
            else:
                print("✗ 已丢弃")

        print("\n" + "=" * 60)
        print(f"✓ 完成！共收集 {demo_count} 条演示")
        print(f"保存位置: {save_dir}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.stop()


if __name__ == '__main__':
    main()
