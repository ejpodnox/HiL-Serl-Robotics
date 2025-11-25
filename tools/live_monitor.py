#!/usr/bin/env python3
"""
实时监控工具

监控机械臂状态、VisionPro 数据流、相机画面等，用于调试和演示。
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path


class LiveMonitor:
    """实时监控所有系统组件"""

    def __init__(self, vp_ip=None, robot_ip=None, camera_id=None):
        self.vp_ip = vp_ip
        self.robot_ip = robot_ip
        self.camera_id = camera_id

        self.vp_bridge = None
        self.robot_commander = None
        self.camera = None

        self.running = False

    def start_visionpro(self):
        """启动 VisionPro 监控"""
        if self.vp_ip is None:
            return False

        try:
            from vision_pro_control.core import VisionProBridge

            print(f"启动 VisionPro 监控 ({self.vp_ip})...")
            self.vp_bridge = VisionProBridge(avp_ip=self.vp_ip)
            self.vp_bridge.start()
            print("✓ VisionPro 监控已启动")
            return True
        except Exception as e:
            print(f"✗ VisionPro 启动失败: {e}")
            return False

    def start_robot(self):
        """启动机械臂监控"""
        if self.robot_ip is None:
            return False

        try:
            import rclpy
            from vision_pro_control.core import RobotCommander

            print(f"启动机械臂监控 ({self.robot_ip})...")

            if not rclpy.ok():
                rclpy.init()

            self.robot_commander = RobotCommander(robot_ip=self.robot_ip)
            print("✓ 机械臂监控已启动")
            return True
        except Exception as e:
            print(f"✗ 机械臂启动失败: {e}")
            return False

    def start_camera(self):
        """启动相机监控"""
        if self.camera_id is None:
            return False

        try:
            from kinova_rl_env import WebCamera

            print(f"启动相机监控 (ID={self.camera_id})...")
            self.camera = WebCamera(camera_id=self.camera_id, target_size=(128, 128))
            self.camera.start()
            print("✓ 相机监控已启动")
            return True
        except Exception as e:
            print(f"✗ 相机启动失败: {e}")
            return False

    def display_visionpro_status(self):
        """显示 VisionPro 状态"""
        if self.vp_bridge is None:
            return

        try:
            data = self.vp_bridge.get_latest_data()

            if data['timestamp'] > 0:
                wrist_pos = data['wrist_pose'][:3, 3]
                pinch = data['pinch_distance']

                print(f"  VisionPro: 手腕=[{wrist_pos[0]:6.3f}, {wrist_pos[1]:6.3f}, {wrist_pos[2]:6.3f}] "
                      f"捏合={pinch:5.3f}")
            else:
                print(f"  VisionPro: ⚠️  等待数据...")

        except Exception as e:
            print(f"  VisionPro: ✗ 错误 ({e})")

    def display_robot_status(self):
        """显示机械臂状态"""
        if self.robot_commander is None:
            return

        try:
            tcp_pose = self.robot_commander.get_tcp_pose()

            if tcp_pose is not None:
                print(f"  Kinova:    TCP=[{tcp_pose[0]:6.3f}, {tcp_pose[1]:6.3f}, {tcp_pose[2]:6.3f}] "
                      f"姿态=[{tcp_pose[3]:5.2f}, {tcp_pose[4]:5.2f}, {tcp_pose[5]:5.2f}, {tcp_pose[6]:5.2f}]")
            else:
                print(f"  Kinova:    ⚠️  等待数据...")

        except Exception as e:
            print(f"  Kinova:    ✗ 错误 ({e})")

    def display_camera_status(self):
        """显示相机状态"""
        if self.camera is None:
            return

        try:
            image = self.camera.get_image()

            if image is not None:
                mean_brightness = np.mean(image)
                print(f"  相机:      图像={image.shape} 亮度={mean_brightness:6.1f}")
            else:
                print(f"  相机:      ⚠️  等待图像...")

        except Exception as e:
            print(f"  相机:      ✗ 错误 ({e})")

    def run(self, duration=None, frequency=10):
        """
        运行监控

        Args:
            duration: 运行时长（秒），None 表示无限运行
            frequency: 更新频率（Hz）
        """
        print("\n" + "=" * 70)
        print("实时监控")
        print("=" * 70)
        print("按 Ctrl+C 停止\n")

        dt = 1.0 / frequency
        start_time = time.time()
        self.running = True

        try:
            while self.running:
                # 清屏（可选）
                # print("\033[2J\033[H")  # ANSI escape code to clear screen

                elapsed = time.time() - start_time
                print(f"\r运行时间: {elapsed:6.1f}s", end="")

                # 显示所有状态
                print()
                self.display_visionpro_status()
                self.display_robot_status()
                self.display_camera_status()
                print()

                # 检查是否超时
                if duration is not None and elapsed >= duration:
                    print(f"\n已运行 {duration}s，监控结束")
                    break

                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n\n用户中断，停止监控")

        finally:
            self.stop()

    def stop(self):
        """停止所有监控"""
        print("\n关闭监控...")

        if self.vp_bridge is not None:
            try:
                self.vp_bridge.stop()
                print("✓ VisionPro 已关闭")
            except:
                pass

        if self.camera is not None:
            try:
                self.camera.stop()
                print("✓ 相机已关闭")
            except:
                pass

        self.running = False


def main():
    # 尝试从配置文件读取默认值
    default_config_path = 'kinova_rl_env/config/kinova_config.yaml'
    default_vp_ip = '192.168.1.125'
    default_robot_ip = '192.168.8.10'
    default_camera_id = 0

    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml(default_config_path)
        default_robot_ip = config.robot.ip
        # 读取 VisionPro IP
        if hasattr(config, 'visionpro') and hasattr(config.visionpro, 'ip'):
            default_vp_ip = config.visionpro.ip
        # 读取第一个 webcam 相机的 device_id
        if config.camera.backend == 'webcam' and config.camera.webcam_cameras:
            first_camera = list(config.camera.webcam_cameras.values())[0]
            default_camera_id = first_camera['device_id']
    except Exception:
        # 配置读取失败，使用硬编码默认值
        pass

    parser = argparse.ArgumentParser(description='实时监控系统组件')

    parser.add_argument('--vp-ip', type=str, default=None,
                        help='VisionPro IP 地址（不填则跳过）')
    parser.add_argument('--robot-ip', type=str, default=None,
                        help='Kinova 机械臂 IP（不填则跳过）')
    parser.add_argument('--camera-id', type=int, default=None,
                        help='USB 相机 ID（不填则跳过）')
    parser.add_argument('--duration', type=float, default=None,
                        help='运行时长（秒），不填则无限运行')
    parser.add_argument('--frequency', type=float, default=10,
                        help='更新频率（Hz，默认 10）')

    # 预设配置
    parser.add_argument('--all', action='store_true',
                        help='监控所有组件（使用默认 IP）')

    args = parser.parse_args()

    # 如果使用 --all，设置默认值（从配置文件读取）
    if args.all:
        args.vp_ip = args.vp_ip or default_vp_ip
        args.robot_ip = args.robot_ip or default_robot_ip
        args.camera_id = args.camera_id if args.camera_id is not None else default_camera_id

    # 检查至少有一个组件
    if args.vp_ip is None and args.robot_ip is None and args.camera_id is None:
        print("错误: 至少需要监控一个组件")
        print("使用 --vp-ip, --robot-ip, 或 --camera-id 指定组件")
        print("或使用 --all 监控所有组件")
        sys.exit(1)

    # 创建监控器
    monitor = LiveMonitor(
        vp_ip=args.vp_ip,
        robot_ip=args.robot_ip,
        camera_id=args.camera_id
    )

    # 启动各组件
    components_started = 0

    if args.vp_ip:
        if monitor.start_visionpro():
            components_started += 1

    if args.robot_ip:
        if monitor.start_robot():
            components_started += 1

    if args.camera_id is not None:
        if monitor.start_camera():
            components_started += 1

    if components_started == 0:
        print("\n✗ 没有组件成功启动，退出")
        sys.exit(1)

    print(f"\n✓ {components_started} 个组件已启动\n")

    # 运行监控
    monitor.run(duration=args.duration, frequency=args.frequency)

    print("\n监控已停止")
    return 0


if __name__ == '__main__':
    sys.exit(main())
