#!/usr/bin/env python3
"""
调试版遥操作程序 - 详细显示错误信息

特点：
1. 捕获并显示所有异常和错误堆栈
2. 实时打印关键数据（位置、速度、关节状态）
3. 安全检查和限制验证
4. 不需要 kortex_api，仅使用 ros2_kortex
5. 颜色标注警告和错误
"""

import argparse
import rclpy
import numpy as np
import time
import sys
import traceback
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.coordinate_mapper import CoordinateMapper
from vision_pro_control.core.calibrator import WorkspaceCalibrator
from kinova_rl_env.kinova_env.kinova_interface import KinovaInterface
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor
import yaml


# ANSI颜色代码
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_info(msg):
    print(f"{Colors.CYAN}ℹ {msg}{Colors.RESET}")


def print_section(title):
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}{Colors.RESET}")


class DebugTeleopRecorder:
    """调试版遥操作记录器 - 详细错误输出"""

    def __init__(self, config_file: str):
        print_section("初始化调试遥操作记录器")

        # 加载配置
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        print_success(f"加载配置文件: {config_file}")

        # 初始化 ROS2
        try:
            rclpy.init()
            print_success("ROS2 初始化成功")
        except Exception as e:
            print_error(f"ROS2 初始化失败: {e}")
            raise

        # 初始化组件
        try:
            self.vp_bridge = VisionProBridge(
                avp_ip=self.config['visionpro']['ip'],
                use_right_hand=self.config['visionpro']['use_right_hand']
            )
            print_success(f"VisionPro Bridge 创建成功 (IP: {self.config['visionpro']['ip']})")
        except Exception as e:
            print_error(f"VisionPro Bridge 创建失败: {e}")
            traceback.print_exc()
            raise

        # 初始化 KinovaInterface
        try:
            self.interface = KinovaInterface(node_name='debug_teleop')
            self.interface.connect()
            print_success("KinovaInterface 连接成功")

            # 等待关节状态
            print_info("等待关节状态...")
            time.sleep(1.0)
            rclpy.spin_once(self.interface.node, timeout_sec=0.5)

            joint_state = self.interface.get_joint_state()
            if joint_state is None:
                print_error("无法获取关节状态！机器人驱动可能未运行")
                print_warning("请先启动: ros2 launch kortex_bringup gen3.launch.py")
                raise RuntimeError("关节状态不可用")
            else:
                print_success(f"关节状态已就绪: {len(joint_state[0])} 个关节")
                print_info(f"当前位置: {[f'{p:.2f}' for p in joint_state[0]]}")

        except Exception as e:
            print_error(f"KinovaInterface 初始化失败: {e}")
            traceback.print_exc()
            raise

        # 标定
        self.calibration_file = Path(__file__).parent.parent / 'vision_pro_control' / self.config['calibration']['file']
        self._run_calibration()

        # 加载 Mapper
        try:
            self.mapper = CoordinateMapper(calibration_file=self.calibration_file)

            mapper_cfg = self.config['mapper']
            self.mapper.set_gains(
                position_gain=mapper_cfg['position_gain'],
                rotation_gain=mapper_cfg['rotation_gain']
            )
            self.mapper.set_velocity_limits(
                max_linear=mapper_cfg['max_linear_velocity'],
                max_angular=mapper_cfg['max_angular_velocity']
            )

            print_success("CoordinateMapper 初始化成功")
            print_info(f"  位置增益: {mapper_cfg['position_gain']}")
            print_info(f"  最大线速度: {mapper_cfg['max_linear_velocity']} m/s")

        except Exception as e:
            print_error(f"CoordinateMapper 初始化失败: {e}")
            traceback.print_exc()
            raise

        # 控制参数
        control_cfg = self.config['control']
        self.control_frequency = control_cfg['frequency']
        self.dt = 1.0 / self.control_frequency
        self.max_joint_velocity = control_cfg['max_joint_velocity']
        self.jacobian_damping = control_cfg['jacobian_damping']

        print_success("初始化完成")
        print_info(f"控制频率: {self.control_frequency} Hz (dt={self.dt:.3f}s)")
        print_info(f"最大关节速度: {self.max_joint_velocity} rad/s")

        # 统计数据
        self.stats = {
            'iterations': 0,
            'errors': 0,
            'warnings': 0,
            'max_joint_vel': 0.0,
            'max_linear_vel': 0.0,
        }

    def _run_calibration(self):
        """运行标定流程"""
        print_section("自动标定流程")

        from vision_pro_control.core.calibrator import WorkspaceCalibrator

        print_info("按键说明:")
        print("  's'     - 采样当前手部位置")
        print("  'c'     - 保存中心点并完成标定")
        print("  'p'     - 打印当前位置信息")
        print("  'q'     - 退出程序")

        calibrator = WorkspaceCalibrator(
            control_radius=0.25,
            deadzone_radius=0.10
        )

        sample_count = 0

        try:
            with KeyboardMonitor() as kb:
                while True:
                    key = kb.get_key(timeout=0.05)

                    if not key:
                        continue

                    if key == 'q':
                        print_error("用户退出标定")
                        raise KeyboardInterrupt()

                    elif key == 's':
                        try:
                            position, rotation = self.vp_bridge.get_hand_relative_to_head()
                            calibrator.add_sample(position, rotation)
                            sample_count += 1
                            print_success(f"采样 #{sample_count}: [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}]")
                        except Exception as e:
                            print_error(f"采样失败: {e}")

                    elif key == 'c':
                        if calibrator.save_center():
                            self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
                            calibrator.save_to_file(self.calibration_file, overwrite=True)
                            print_success("标定完成！")
                            print_info(f"标定文件: {self.calibration_file}")
                            return
                        else:
                            print_error("需要至少 1 个采样点")

                    elif key == 'p':
                        try:
                            position, rotation = self.vp_bridge.get_hand_relative_to_head()
                            print_info(f"当前位置: [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}]")
                            if calibrator.center_position is not None:
                                distance = np.linalg.norm(position - calibrator.center_position)
                                print_info(f"距离中心: {distance:.3f} m ({distance*100:.1f} cm)")
                        except Exception as e:
                            print_error(f"获取位置失败: {e}")

        except KeyboardInterrupt:
            print_error("标定被中断")
            raise

    def run_debug_teleop(self):
        """运行调试遥操作"""
        print_section("开始调试遥操作")

        print_info("按 'q' 停止")
        print("")

        start_time = time.time()
        step = 0

        # 安全限制
        velocity_limit = self.config['robot']['joint_limits']['velocity_max']

        try:
            with KeyboardMonitor() as kb:
                while True:
                    loop_start = time.time()

                    # 检查按键
                    key = kb.get_key(timeout=0.001)
                    if key == 'q':
                        print_warning("用户停止遥操作")
                        break

                    try:
                        # ===== 1. Spin 接收关节状态 =====
                        rclpy.spin_once(self.interface.node, timeout_sec=0.001)

                        # ===== 2. 获取关节状态 =====
                        joint_state = self.interface.get_joint_state()
                        if joint_state is None:
                            print_error(f"[{step:4d}] 无法获取关节状态")
                            self.stats['errors'] += 1
                            time.sleep(self.dt)
                            continue

                        q, q_dot = joint_state
                        current_max_vel = np.max(np.abs(q_dot))

                        # ===== 3. 获取 VisionPro 数据 =====
                        try:
                            position, rotation = self.vp_bridge.get_hand_relative_to_head()
                            pinch_distance = self.vp_bridge.get_pinch_distance()
                        except Exception as e:
                            print_error(f"[{step:4d}] VisionPro 数据获取失败: {e}")
                            self.stats['errors'] += 1
                            time.sleep(self.dt)
                            continue

                        # ===== 4. 映射到 Twist =====
                        try:
                            twist = self.mapper.map_to_twist(position, rotation)
                            twist_array = np.array([
                                twist['linear']['x'], twist['linear']['y'], twist['linear']['z'],
                                twist['angular']['x'], twist['angular']['y'], twist['angular']['z']
                            ])

                            linear_speed = np.linalg.norm(twist_array[:3])
                            self.stats['max_linear_vel'] = max(self.stats['max_linear_vel'], linear_speed)

                        except Exception as e:
                            print_error(f"[{step:4d}] Twist 映射失败: {e}")
                            traceback.print_exc()
                            self.stats['errors'] += 1
                            time.sleep(self.dt)
                            continue

                        # ===== 5. 转换为关节速度 =====
                        try:
                            joint_velocities = self._twist_to_joint_velocity(twist_array, q)
                            commanded_max_vel = np.max(np.abs(joint_velocities))
                            self.stats['max_joint_vel'] = max(self.stats['max_joint_vel'], commanded_max_vel)

                        except Exception as e:
                            print_error(f"[{step:4d}] 关节速度计算失败: {e}")
                            traceback.print_exc()
                            self.stats['errors'] += 1
                            time.sleep(self.dt)
                            continue

                        # ===== 6. 安全检查 =====
                        safety_ok = True
                        for i, (vel, limit) in enumerate(zip(joint_velocities, velocity_limit)):
                            if abs(vel) > limit:
                                print_error(f"[{step:4d}] 关节{i+1} 速度超限: {vel:.3f} > {limit:.3f} rad/s")
                                safety_ok = False
                                self.stats['warnings'] += 1

                        if not safety_ok:
                            print_warning(f"[{step:4d}] 安全检查失败，跳过此步")
                            time.sleep(self.dt)
                            continue

                        # ===== 7. 发送命令 =====
                        try:
                            self.interface.send_joint_velocities(joint_velocities.tolist(), dt=self.dt)
                        except Exception as e:
                            print_error(f"[{step:4d}] 发送命令失败: {e}")
                            traceback.print_exc()
                            self.stats['errors'] += 1
                            time.sleep(self.dt)
                            continue

                        # ===== 8. 夹爪控制 =====
                        try:
                            gripper_position = self._pinch_to_gripper(pinch_distance)
                            self.interface.send_gripper_command(gripper_position)
                        except Exception as e:
                            # 夹爪错误不致命
                            if step % 100 == 0:
                                print_warning(f"[{step:4d}] 夹爪命令失败: {e}")

                        # ===== 9. 打印状态 =====
                        if step % 20 == 0:  # 每1秒打印一次
                            elapsed = time.time() - start_time

                            print(f"\n{Colors.BOLD}[{step:4d}] t={elapsed:.1f}s{Colors.RESET}")
                            print(f"  手部位置: [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}] m")
                            print(f"  线速度  : [{twist_array[0]:6.3f}, {twist_array[1]:6.3f}, {twist_array[2]:6.3f}] m/s (速率={linear_speed:.4f})")
                            print(f"  角速度  : [{twist_array[3]:6.3f}, {twist_array[4]:6.3f}, {twist_array[5]:6.3f}] rad/s")
                            print(f"  关节速度: [" + ", ".join([f"{v:5.2f}" for v in joint_velocities]) + "] rad/s")
                            print(f"  当前最大: {current_max_vel:.3f} rad/s, 命令最大: {commanded_max_vel:.3f} rad/s")

                            # 警告检查
                            if linear_speed > 0.02:
                                print_warning(f"  线速度较高: {linear_speed:.4f} m/s")
                            if commanded_max_vel > 0.15:
                                print_warning(f"  关节速度较高: {commanded_max_vel:.3f} rad/s")

                        step += 1
                        self.stats['iterations'] = step

                    except Exception as e:
                        print_error(f"[{step:4d}] 循环异常: {e}")
                        traceback.print_exc()
                        self.stats['errors'] += 1

                    # 维持控制频率
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, self.dt - elapsed)
                    if sleep_time == 0 and step % 100 == 0:
                        print_warning(f"[{step:4d}] 控制周期超时: {elapsed*1000:.1f}ms > {self.dt*1000:.1f}ms")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print_warning("\n用户中断")
        except Exception as e:
            print_error(f"\n遥操作异常: {e}")
            traceback.print_exc()
        finally:
            self._print_statistics()

    def _twist_to_joint_velocity(self, twist: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Twist → 关节速度（使用雅可比）"""
        # 计算雅可比矩阵
        J = self._compute_jacobian(q)

        # DLS 伪逆
        JJT = J @ J.T + self.jacobian_damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)

        # 计算关节速度
        joint_vel = J_pinv @ twist

        # 限制
        joint_vel = np.clip(joint_vel, -self.max_joint_velocity, self.max_joint_velocity)

        return joint_vel

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵（Kinova Gen3 7-DOF）"""
        # URDF 参数
        d1 = 0.15643
        d2 = 0.12838
        d3 = 0.21038
        d4 = 0.21038
        d5 = 0.20843
        d6 = 0.10593
        d7 = 0.10593
        d_ee = 0.061525

        def rot_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        def rot_x(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

        def trans(x, y, z):
            return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

        T0 = np.eye(4)
        T1 = T0 @ trans(0, 0, d1) @ rot_x(np.pi) @ rot_z(q[0])
        T2 = T1 @ trans(0, 0.005375, -d2) @ rot_x(np.pi/2) @ rot_z(q[1])
        T3 = T2 @ trans(0, -d3, -0.006375) @ rot_x(-np.pi/2) @ rot_z(q[2])
        T4 = T3 @ trans(0, 0.006375, -d4) @ rot_x(np.pi/2) @ rot_z(q[3])
        T5 = T4 @ trans(0, -d5, -0.006375) @ rot_x(-np.pi/2) @ rot_z(q[4])
        T6 = T5 @ trans(0, 0.00017505, -d6) @ rot_x(np.pi/2) @ rot_z(q[5])
        T7 = T6 @ trans(0, -d7, -0.00017505) @ rot_x(-np.pi/2) @ rot_z(q[6])
        T_ee = T7 @ trans(0, 0, -d_ee) @ rot_x(np.pi)

        p = [T0[:3, 3], T1[:3, 3], T2[:3, 3], T3[:3, 3],
             T4[:3, 3], T5[:3, 3], T6[:3, 3], T7[:3, 3]]
        p_ee = T_ee[:3, 3]

        z = [T0[:3, 2], T1[:3, 2], T2[:3, 2], T3[:3, 2],
             T4[:3, 2], T5[:3, 2], T6[:3, 2], T7[:3, 2]]

        J = np.zeros((6, 7))
        for i in range(7):
            J[:3, i] = np.cross(z[i], p_ee - p[i])
            J[3:, i] = z[i]

        return J

    def _pinch_to_gripper(self, pinch_distance: float) -> float:
        """捏合距离 → 夹爪位置"""
        gripper_cfg = self.config['gripper']

        if gripper_cfg['control_mode'] == 'continuous':
            pinch_open = gripper_cfg['pinch_distance_open']
            pinch_close = gripper_cfg['pinch_distance_close']
            gripper_open = gripper_cfg['gripper_open_position']
            gripper_close = gripper_cfg['gripper_close_position']

            pinch_distance = np.clip(pinch_distance, pinch_close, pinch_open)
            normalized = (pinch_distance - pinch_close) / (pinch_open - pinch_close)
            gripper_position = gripper_close + (gripper_open - gripper_close) * normalized

            return float(gripper_position)
        else:
            threshold = gripper_cfg['pinch_threshold']
            if pinch_distance < threshold:
                return gripper_cfg['close_position']
            else:
                return gripper_cfg['open_position']

    def _print_statistics(self):
        """打印统计信息"""
        print_section("运行统计")

        print(f"  总步数: {self.stats['iterations']}")
        print(f"  错误数: {self.stats['errors']}")
        print(f"  警告数: {self.stats['warnings']}")
        print(f"  最大线速度: {self.stats['max_linear_vel']:.4f} m/s")
        print(f"  最大关节速度: {self.stats['max_joint_vel']:.3f} rad/s")

        if self.stats['iterations'] > 0:
            error_rate = self.stats['errors'] / self.stats['iterations'] * 100
            print(f"  错误率: {error_rate:.1f}%")

    def start(self):
        """启动 VisionPro"""
        try:
            self.vp_bridge.start()
            time.sleep(1.0)
            print_success("VisionPro 数据流已启动")
        except Exception as e:
            print_error(f"VisionPro 启动失败: {e}")
            raise

    def stop(self):
        """停止所有组件"""
        try:
            self.vp_bridge.stop()
            self.interface.send_joint_velocities([0.0] * 7)
            self.interface.disconnect()
            rclpy.shutdown()
            print_success("已停止所有组件")
        except Exception as e:
            print_error(f"停止组件时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='调试版 VisionPro 遥操作')
    parser.add_argument('--config', type=str,
                        default='vision_pro_control/config/teleop_config_safe.yaml',
                        help='配置文件路径（默认使用安全配置）')

    args = parser.parse_args()

    print_section("调试版遥操作程序")
    print_info(f"配置文件: {args.config}")
    print("")

    try:
        recorder = DebugTeleopRecorder(config_file=args.config)
        recorder.start()

        print_info("\n按任意键开始遥操作...")
        input()

        recorder.run_debug_teleop()

    except KeyboardInterrupt:
        print_warning("\n\n用户中断")
    except Exception as e:
        print_error(f"\n程序异常: {e}")
        traceback.print_exc()
    finally:
        try:
            recorder.stop()
        except:
            pass


if __name__ == '__main__':
    main()
