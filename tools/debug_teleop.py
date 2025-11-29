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

        # 【修复1：加载完整的机器人配置，获取真实关节限制】
        print_info("加载机器人配置...")
        try:
            from kinova_rl_env.kinova_env.config_loader import KinovaConfig
            kinova_config_path = Path(__file__).parent.parent / 'kinova_rl_env/config/kinova_config.yaml'
            self.kinova_config = KinovaConfig.from_yaml(str(kinova_config_path))

            # 获取真实的关节限制
            self.joint_velocity_limits = np.array(self.kinova_config.robot.joint_limits.velocity_max)
            self.joint_position_min = np.array(self.kinova_config.robot.joint_limits.position_min)
            self.joint_position_max = np.array(self.kinova_config.robot.joint_limits.position_max)

            print_success("机器人配置加载成功")
            print_info(f"  关节速度限制: {self.joint_velocity_limits}")
            print_info(f"  位置范围: [{self.joint_position_min[0]:.2f}, {self.joint_position_max[0]:.2f}] rad")

        except Exception as e:
            print_warning(f"无法加载机器人配置: {e}")
            print_warning("使用默认限制")
            # 使用Kinova Gen3的默认限制
            self.joint_velocity_limits = np.array([1.3, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2])
            self.joint_position_min = np.array([-3.14, -2.41, -3.14, -2.66, -3.14, -2.23, -3.14])
            self.joint_position_max = np.array([3.14, 2.41, 3.14, 2.66, 3.14, 2.23, 3.14])

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

        # 标定文件路径
        self.calibration_file = Path(__file__).parent.parent / 'vision_pro_control' / self.config['calibration']['file']

        # 注意：标定需要在 start() 之后进行，这里先不标定
        self.mapper = None

        # 控制参数
        control_cfg = self.config['control']
        self.control_frequency = control_cfg['frequency']
        self.dt = 1.0 / self.control_frequency
        self.max_joint_velocity = control_cfg['max_joint_velocity']
        self.jacobian_damping = control_cfg['jacobian_damping']

        print_success("基础初始化完成")
        print_info(f"控制频率: {self.control_frequency} Hz (dt={self.dt:.3f}s)")
        print_info(f"最大关节速度: {self.max_joint_velocity} rad/s")
        print_warning("注意：需要调用 start() 启动VisionPro并执行标定")

        # 【修复4：速度平滑 - 初始化状态】
        self.last_joint_velocities = np.zeros(7)
        self.max_acceleration = 1.0  # rad/s² - 最大加速度

        # 【修复3：启动保护】
        self.startup_steps = 100  # 前100步（5秒）使用启动保护
        self.startup_scale = 0.2  # 启动期间速度缩放到20%

        # 【修复5：关节位置安全裕度】
        self.position_safety_margin = 0.5  # rad - 安全裕度阈值
        self.position_danger_margin = 0.3  # rad - 危险裕度阈值

        # 【修复6：工作空间边界保护】
        self.workspace_center = np.array([0.0, 0.0, 0.3])  # 机器人基座坐标系
        self.workspace_radius_safe = 0.8  # m - 安全半径
        self.workspace_radius_max = 0.9   # m - 最大半径
        self.workspace_height_min = -0.1  # m - 最低高度
        self.workspace_height_max = 1.0   # m - 最高高度

        # 【修复7：紧急停止和异常检测】
        self.emergency_stop = False
        self.consecutive_errors = 0
        self.consecutive_warnings = 0
        self.max_consecutive_errors = 5
        self.max_consecutive_warnings = 10

        # 统计数据
        self.stats = {
            'iterations': 0,
            'errors': 0,
            'warnings': 0,
            'max_joint_vel': 0.0,
            'max_linear_vel': 0.0,
            'max_cond_number': 0.0,  # 最大条件数
            'singularity_warnings': 0,  # 奇异性警告次数
            'position_limit_activations': 0,  # 位置限制激活次数
            'workspace_limit_activations': 0,  # 工作空间限制激活次数
            'emergency_stops': 0,  # 紧急停止次数
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
        print_section("开始调试遥操作 - 【只读模式】")

        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'*' * 80}")
        print(f"                    ⚠️  只读模式 - 不会发送任何运动命令  ⚠️")
        print(f"                    仅用于诊断和数据收集")
        print(f"{'*' * 80}{Colors.RESET}\n")

        print_info("按键说明:")
        print("  'q' - 停止程序")
        print("")

        start_time = time.time()
        step = 0

        # 使用真实的关节速度限制（机器人硬件限制）
        velocity_limit = self.joint_velocity_limits

        try:
            with KeyboardMonitor() as kb:
                while True:
                    loop_start = time.time()

                    # 【修复7：检查按键 - 添加紧急停止】
                    key = kb.get_key(timeout=0.001)
                    if key == 'q':
                        print_warning("用户停止遥操作")
                        break
                    elif key == 'e':
                        print_error("⚠️  紧急停止激活！")
                        self.emergency_stop = True
                        self.stats['emergency_stops'] += 1
                        # 立即发送零速度
                        self.interface.send_joint_velocities([0.0] * 7, dt=self.dt)
                        print_warning("机器人已停止，按 'r' 恢复运行，按 'q' 退出")
                        continue
                    elif key == 'r' and self.emergency_stop:
                        print_success("恢复运行")
                        self.emergency_stop = False
                        self.consecutive_errors = 0
                        self.consecutive_warnings = 0
                        continue

                    # 【修复7：如果处于紧急停止状态，跳过控制循环】
                    if self.emergency_stop:
                        time.sleep(self.dt)
                        continue

                    try:
                        # ===== 1. Spin 接收关节状态 =====
                        rclpy.spin_once(self.interface.node, timeout_sec=0.001)

                        # ===== 2. 获取关节状态 =====
                        joint_state = self.interface.get_joint_state()
                        if joint_state is None:
                            print_error(f"[{step:4d}] 无法获取关节状态")
                            self.stats['errors'] += 1
                            self.consecutive_errors += 1

                            # 【修复7：检查连续错误】
                            if self.consecutive_errors >= self.max_consecutive_errors:
                                print_error(f"连续{self.consecutive_errors}次错误，触发紧急停止！")
                                self.emergency_stop = True
                                self.stats['emergency_stops'] += 1

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

                            # 【修复4：速度平滑 - 限制加速度】
                            max_delta = self.max_acceleration * self.dt  # 单步最大速度变化
                            delta = joint_velocities - self.last_joint_velocities

                            # 检查是否有过大的加速度
                            if np.max(np.abs(delta)) > max_delta:
                                print_warning(f"[{step:4d}] 加速度过大，平滑处理: max_delta={np.max(np.abs(delta)):.3f}")

                            # 限制加速度变化
                            delta = np.clip(delta, -max_delta, max_delta)
                            joint_velocities = self.last_joint_velocities + delta

                            # 【修复3：启动保护 - 前N步降低速度】
                            if step < self.startup_steps:
                                scale = self.startup_scale + (1.0 - self.startup_scale) * (step / self.startup_steps)
                                joint_velocities *= scale
                                if step % 20 == 0:
                                    print_info(f"[{step:4d}] 启动保护中，速度缩放: {scale:.2f}")

                            # 【修复5：应用关节位置安全裕度检查】 - 临时注释测试
                            # joint_velocities = self._apply_joint_position_safety(joint_velocities, q)

                            # 【修复6：应用工作空间边界保护】 - 临时注释测试
                            # joint_velocities = self._apply_workspace_safety(joint_velocities, q)

                            # 更新上次速度
                            self.last_joint_velocities = joint_velocities.copy()

                            commanded_max_vel = np.max(np.abs(joint_velocities))
                            self.stats['max_joint_vel'] = max(self.stats['max_joint_vel'], commanded_max_vel)

                        except Exception as e:
                            print_error(f"[{step:4d}] 关节速度计算失败: {e}")
                            traceback.print_exc()
                            self.stats['errors'] += 1
                            self.consecutive_errors += 1

                            # 【修复7：检查连续错误】
                            if self.consecutive_errors >= self.max_consecutive_errors:
                                print_error(f"连续{self.consecutive_errors}次错误，触发紧急停止！")
                                self.emergency_stop = True
                                self.stats['emergency_stops'] += 1

                            time.sleep(self.dt)
                            continue

                        # ===== 6. 安全检查（使用真实硬件限制）=====
                        safety_ok = True
                        for i, (vel, limit) in enumerate(zip(joint_velocities, velocity_limit)):
                            if abs(vel) > limit * 0.9:  # 接近限制的90%就警告
                                print_warning(f"[{step:4d}] 关节{i+1} 速度接近限制: {vel:.3f} / {limit:.3f} rad/s")
                                self.stats['warnings'] += 1

                            if abs(vel) > limit:  # 超过限制
                                print_error(f"[{step:4d}] 关节{i+1} 速度超限: {vel:.3f} > {limit:.3f} rad/s")
                                safety_ok = False
                                self.stats['warnings'] += 1

                        if not safety_ok:
                            print_warning(f"[{step:4d}] 安全检查失败，跳过此步")
                            self.consecutive_warnings += 1

                            # 【修复7：检查连续警告】
                            if self.consecutive_warnings >= self.max_consecutive_warnings:
                                print_error(f"连续{self.consecutive_warnings}次警告，触发紧急停止！")
                                self.emergency_stop = True
                                self.stats['emergency_stops'] += 1

                            time.sleep(self.dt)
                            continue
                        else:
                            # 【修复7：成功执行，重置连续警告计数】
                            self.consecutive_warnings = 0

                        # ===== 7. 【只读模式】不发送命令，仅输出 =====
                        print(f"{Colors.MAGENTA}[DRY RUN] 计算完成，不发送命令{Colors.RESET}")
                        # 注释掉实际发送
                        # self.interface.send_joint_velocities(joint_velocities.tolist(), dt=self.dt)
                        self.consecutive_errors = 0

                        # ===== 8. 【只读模式】夹爪也不发送 =====
                        gripper_position = self._pinch_to_gripper(pinch_distance)
                        # 注释掉实际发送
                        # self.interface.send_gripper_command(gripper_position)

                        # ===== 9. 打印详细状态 =====
                        if step % 10 == 0:  # 每0.5秒打印一次
                            elapsed = time.time() - start_time

                            print(f"\n{Colors.BOLD}{'=' * 80}")
                            print(f"[{step:4d}] t={elapsed:.1f}s - 【只读模式】不发送命令")
                            print(f"{'=' * 80}{Colors.RESET}")

                            # VisionPro数据
                            print(f"\n{Colors.CYAN}📱 VisionPro 数据:{Colors.RESET}")
                            print(f"  手部位置: [{position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f}] m")
                            print(f"  线速度  : [{twist_array[0]:7.3f}, {twist_array[1]:7.3f}, {twist_array[2]:7.3f}] m/s (速率={linear_speed:.4f})")
                            print(f"  角速度  : [{twist_array[3]:7.3f}, {twist_array[4]:7.3f}, {twist_array[5]:7.3f}] rad/s")

                            # 当前关节状态
                            print(f"\n{Colors.GREEN}🤖 当前关节状态（从机器人读取）:{Colors.RESET}")
                            print(f"  位置: [" + ", ".join([f"{p:6.3f}" for p in q]) + "] rad")
                            print(f"  位置: [" + ", ".join([f"{np.rad2deg(p):6.1f}°" for p in q]) + "]")
                            print(f"  速度: [" + ", ".join([f"{v:6.3f}" for v in q_dot]) + "] rad/s")

                            # 计算出的目标速度
                            print(f"\n{Colors.YELLOW}🎯 计算的目标关节速度（准备发送）:{Colors.RESET}")
                            print(f"  速度: [" + ", ".join([f"{v:6.3f}" for v in joint_velocities]) + "] rad/s")
                            print(f"  最大: {commanded_max_vel:.3f} rad/s")

                            # 每个关节的对比
                            print(f"\n  关节对比:")
                            for i in range(7):
                                margin_min = q[i] - self.joint_position_min[i]
                                margin_max = self.joint_position_max[i] - q[i]
                                status = "✓" if margin_min > 0.3 and margin_max > 0.3 else "⚠"
                                print(f"    J{i+1}: pos={q[i]:6.3f}, vel_cmd={joint_velocities[i]:6.3f}, "
                                      f"margin=({margin_min:5.2f}, {margin_max:5.2f}) {status}")

                            # 警告检查
                            if linear_speed > 0.02:
                                print_warning(f"\n  线速度较高: {linear_speed:.4f} m/s")
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

    def _apply_joint_position_safety(self, joint_velocities: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        【修复5：关节位置安全裕度主动避让】

        检查关节位置,当接近极限时主动降低朝向极限方向的速度

        Args:
            joint_velocities: 原始关节速度 [7]
            q: 当前关节位置 [7]

        Returns:
            安全缩放后的关节速度 [7]
        """
        safe_velocities = joint_velocities.copy()
        safety_activated = False

        for i in range(7):
            margin_min = q[i] - self.joint_position_min[i]
            margin_max = self.joint_position_max[i] - q[i]

            # 检查下限
            if margin_min < self.position_safety_margin:
                if joint_velocities[i] < 0:  # 朝向下限移动
                    if margin_min < self.position_danger_margin:
                        # 危险区域: 完全阻止
                        safe_velocities[i] = 0.0
                        print_error(f"关节{i+1}危险接近下限！余量={margin_min:.3f} rad, 阻止负向运动")
                        safety_activated = True
                    else:
                        # 警告区域: 按比例缩放
                        scale = (margin_min - self.position_danger_margin) / \
                                (self.position_safety_margin - self.position_danger_margin)
                        safe_velocities[i] *= scale
                        print_warning(f"关节{i+1}接近下限，速度缩放={scale:.2f}, 余量={margin_min:.3f} rad")
                        safety_activated = True

            # 检查上限
            if margin_max < self.position_safety_margin:
                if joint_velocities[i] > 0:  # 朝向上限移动
                    if margin_max < self.position_danger_margin:
                        # 危险区域: 完全阻止
                        safe_velocities[i] = 0.0
                        print_error(f"关节{i+1}危险接近上限！余量={margin_max:.3f} rad, 阻止正向运动")
                        safety_activated = True
                    else:
                        # 警告区域: 按比例缩放
                        scale = (margin_max - self.position_danger_margin) / \
                                (self.position_safety_margin - self.position_danger_margin)
                        safe_velocities[i] *= scale
                        print_warning(f"关节{i+1}接近上限，速度缩放={scale:.2f}, 余量={margin_max:.3f} rad")
                        safety_activated = True

        if safety_activated:
            self.stats['position_limit_activations'] += 1

        return safe_velocities

    def _apply_workspace_safety(self, joint_velocities: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        【修复6：工作空间边界保护】

        检查末端执行器位置,防止超出安全工作空间

        Args:
            joint_velocities: 关节速度 [7]
            q: 当前关节位置 [7]

        Returns:
            安全缩放后的关节速度 [7]
        """
        # 计算末端执行器位置 (使用正运动学)
        ee_pos = self._compute_end_effector_position(q)

        # 相对于工作空间中心的位置
        relative_pos = ee_pos - self.workspace_center
        distance_horizontal = np.linalg.norm(relative_pos[:2])  # xy平面距离
        height = ee_pos[2]

        scale = 1.0
        safety_activated = False

        # 检查水平距离
        if distance_horizontal > self.workspace_radius_safe:
            if distance_horizontal > self.workspace_radius_max:
                print_error(f"末端超出最大工作空间！距离={distance_horizontal:.3f}m > {self.workspace_radius_max}m")
                # 计算是否在向外移动
                # 简化处理: 大幅降低速度
                scale = 0.1
                safety_activated = True
            else:
                # 警告区域: 按比例缩放
                margin = self.workspace_radius_max - distance_horizontal
                scale_factor = margin / (self.workspace_radius_max - self.workspace_radius_safe)
                scale = min(scale, scale_factor)
                print_warning(f"末端接近工作空间边界，距离={distance_horizontal:.3f}m, 速度缩放={scale:.2f}")
                safety_activated = True

        # 检查高度
        if height < self.workspace_height_min + 0.1:
            if height < self.workspace_height_min:
                print_error(f"末端低于最小高度！h={height:.3f}m < {self.workspace_height_min}m")
                scale = min(scale, 0.1)
                safety_activated = True
            else:
                margin = height - self.workspace_height_min
                scale_factor = margin / 0.1
                scale = min(scale, scale_factor)
                print_warning(f"末端接近最小高度，h={height:.3f}m, 速度缩放={scale:.2f}")
                safety_activated = True

        if height > self.workspace_height_max - 0.1:
            if height > self.workspace_height_max:
                print_error(f"末端高于最大高度！h={height:.3f}m > {self.workspace_height_max}m")
                scale = min(scale, 0.1)
                safety_activated = True
            else:
                margin = self.workspace_height_max - height
                scale_factor = margin / 0.1
                scale = min(scale, scale_factor)
                print_warning(f"末端接近最大高度，h={height:.3f}m, 速度缩放={scale:.2f}")
                safety_activated = True

        if safety_activated:
            self.stats['workspace_limit_activations'] += 1

        return joint_velocities * scale

    def _compute_end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """
        计算末端执行器位置 (正运动学)

        Args:
            q: 关节位置 [7]

        Returns:
            末端执行器位置 [x, y, z]
        """
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

        return T_ee[:3, 3]

    def _twist_to_joint_velocity(self, twist: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Twist → 关节速度（使用雅可比）

        包含奇异性检查和安全限制
        """
        # 计算雅可比矩阵
        J = self._compute_jacobian(q)

        # 【修复2：奇异性检查】
        try:
            cond_number = np.linalg.cond(J)
            self.stats['max_cond_number'] = max(self.stats['max_cond_number'], cond_number)

            # 【详细诊断】雅可比矩阵信息
            J_rank = np.linalg.matrix_rank(J)
            J_det_JJT = np.linalg.det(J @ J.T)
            singular_values = np.linalg.svd(J, compute_uv=False)

            # 条件数过高 = 接近奇异点
            if cond_number > 100:
                print_warning(f"雅可比条件数过高: {cond_number:.1f} - 接近奇异点！")
                self.stats['singularity_warnings'] += 1

                # 详细诊断信息
                print(f"{Colors.RED}  【奇异性详细诊断】{Colors.RESET}")
                print(f"    条件数: {cond_number:.1f} (正常应 < 100)")
                print(f"    秩: {J_rank}/6 (满秩应为6)")
                print(f"    det(J*J^T): {J_det_JJT:.6e}")
                print(f"    奇异值: " + ", ".join([f"{sv:.3f}" for sv in singular_values]))
                print(f"    最大/最小奇异值比: {singular_values[0]/singular_values[-1]:.1f}")

                # 动态增加阻尼，避免速度爆炸
                adaptive_damping = self.jacobian_damping * (cond_number / 100)
                adaptive_damping = min(adaptive_damping, 0.5)  # 最大0.5
                print_warning(f"  自适应阻尼: {adaptive_damping:.3f}")
            else:
                adaptive_damping = self.jacobian_damping

        except Exception as e:
            print_error(f"条件数计算失败: {e}")
            adaptive_damping = self.jacobian_damping

        # DLS 伪逆（使用自适应阻尼）
        JJT = J @ J.T + adaptive_damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)

        # 计算关节速度
        joint_vel = J_pinv @ twist

        # 限制到配置的最大速度
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
        print(f"  奇异性警告: {self.stats['singularity_warnings']}")
        print(f"  位置限制激活: {self.stats['position_limit_activations']} 次")
        print(f"  工作空间限制激活: {self.stats['workspace_limit_activations']} 次")
        print(f"  紧急停止次数: {self.stats['emergency_stops']} 次")
        print(f"  最大条件数: {self.stats['max_cond_number']:.1f}")
        print(f"  最大线速度: {self.stats['max_linear_vel']:.4f} m/s")
        print(f"  最大关节速度: {self.stats['max_joint_vel']:.3f} rad/s")

        if self.stats['iterations'] > 0:
            error_rate = self.stats['errors'] / self.stats['iterations'] * 100
            print(f"  错误率: {error_rate:.1f}%")

    def start(self):
        """启动 VisionPro 并执行标定"""
        try:
            # 1. 启动VisionPro数据流
            self.vp_bridge.start()
            print_success("VisionPro 数据流已启动")

            # 2. 等待数据稳定
            print_info("等待VisionPro数据稳定...")
            time.sleep(3.0)

            # 3. 验证数据
            test_pos, _ = self.vp_bridge.get_hand_relative_to_head()
            print_info(f"VisionPro数据测试: {test_pos}")

            if np.allclose(test_pos, 0):
                print_error("VisionPro数据全是0！")
                print_warning("可能原因：")
                print("  1. VisionPro应用未运行")
                print("  2. 网络连接问题")
                print("  3. IP地址错误")
                raise RuntimeError("VisionPro数据不可用")
            else:
                print_success("VisionPro数据正常")

            # 4. 执行标定
            self._run_calibration()

            # 5. 加载Mapper
            mapper_cfg = self.config['mapper']
            self.mapper = CoordinateMapper(calibration_file=self.calibration_file)
            self.mapper.set_gains(
                position_gain=mapper_cfg['position_gain'],
                rotation_gain=mapper_cfg['rotation_gain']
            )
            self.mapper.set_velocity_limits(
                max_linear=mapper_cfg['max_linear_velocity'],
                max_angular=mapper_cfg['max_angular_velocity']
            )
            print_success("CoordinateMapper 初始化成功")

        except Exception as e:
            print_error(f"启动失败: {e}")
            raise

    def stop(self):
        """停止所有组件"""
        try:
            self.vp_bridge.stop()
            self.interface.send_joint_velocities([0.0] * 7)
            self.interface.disconnect()
            # disconnect() 已经调用了 rclpy.shutdown()，不要重复调用
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
