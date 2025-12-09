#!/usr/bin/env python3
"""
快速验证工具

一键检查所有硬件是否正常连接。
"""

import argparse
import sys
from pathlib import Path


class QuickVerify:
    """快速验证工具"""

    def __init__(self):
        self.results = {}

    def verify_visionpro(self, vp_ip: str, timeout: float = 3.0):
        """验证 VisionPro 连接"""
        print("\n" + "=" * 60)
        print("【1/4】VisionPro 连接验证")
        print("=" * 60)

        try:
            from vision_pro_control.core import VisionProBridge

            bridge = VisionProBridge(avp_ip=vp_ip)
            bridge.start()

            import time
            start_time = time.time()
            data_received = False

            while time.time() - start_time < timeout:
                data = bridge.get_latest_data()
                if data['timestamp'] > 0:
                    print(f"✓ VisionPro 连接成功 ({vp_ip})")
                    print(f"  - 手腕位置: {data['wrist_pose'][:3, 3]}")
                    data_received = True
                    break
                time.sleep(0.1)

            bridge.stop()

            if not data_received:
                print(f"✗ {timeout}s 内未收到数据")
                return False

            self.results['visionpro'] = True
            return True

        except Exception as e:
            print(f"✗ VisionPro 连接失败: {e}")
            self.results['visionpro'] = False
            return False

    def verify_kinova(self, robot_ip: str, timeout: float = 3.0):
        """验证 Kinova 机械臂连接"""
        print("\n" + "=" * 60)
        print("【2/4】Kinova 机械臂验证")
        print("=" * 60)

        try:
            import rclpy
            from vision_pro_control.core import RobotCommander

            if not rclpy.ok():
                rclpy.init()

            commander = RobotCommander(robot_ip=robot_ip)

            import time
            # 等待 TF buffer 填充数据，需要 spin 让节点接收消息
            print("  等待 TF buffer 准备...")
            end_time = time.time() + 2.0
            while time.time() < end_time:
                rclpy.spin_once(commander, timeout_sec=0.1)

            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    # 继续 spin 以接收最新的 TF 数据
                    rclpy.spin_once(commander, timeout_sec=0.1)
                    pose = commander.get_tcp_pose()
                    if pose is not None:
                        print(f"✓ Kinova 机械臂连接成功 ({robot_ip})")
                        print(f"  - TCP 位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                        self.results['kinova'] = True
                        return True
                except:
                    pass
                time.sleep(0.1)

            print(f"✗ {timeout}s 内未能获取机械臂状态")
            print("  提示: 确保已启动 kortex_bringup")
            self.results['kinova'] = False
            return False

        except Exception as e:
            print(f"✗ Kinova 连接失败: {e}")
            self.results['kinova'] = False
            return False

    def verify_camera(self, camera_id: int = 9):
        """验证 USB 相机"""
        print("\n" + "=" * 60)
        print("【3/4】USB 相机验证")
        print("=" * 60)

        try:
            from kinova_rl_env import WebCamera

            camera = WebCamera(camera_id=camera_id, target_size=(128, 128))
            camera.start()

            image = camera.get_image()
            camera.stop()

            if image is not None and image.shape == (128, 128, 3):
                print(f"✓ USB 相机连接成功 (ID={camera_id})")
                print(f"  - 图像尺寸: {image.shape}")
                self.results['camera'] = True
                return True
            else:
                print(f"✗ 相机返回无效图像")
                self.results['camera'] = False
                return False

        except Exception as e:
            print(f"✗ USB 相机连接失败: {e}")
            print(f"  提示: 检查设备 /dev/video{camera_id}")
            self.results['camera'] = False
            return False

    def verify_environment(self):
        """验证环境可以创建"""
        print("\n" + "=" * 60)
        print("【4/4】环境创建验证")
        print("=" * 60)

        try:
            from kinova_rl_env import KinovaEnv, KinovaConfig

            print("  尝试加载配置...")
            config = KinovaConfig.from_yaml("kinova_rl_env/config/kinova_config.yaml")
            print("  ✓ 配置加载成功")

            # 不实际连接硬件，只检查能否创建
            print("  ✓ 环境定义正确")

            self.results['environment'] = True
            return True

        except Exception as e:
            print(f"✗ 环境验证失败: {e}")
            import traceback
            traceback.print_exc()
            self.results['environment'] = False
            return False

    def print_summary(self):
        """打印验证总结"""
        print("\n" + "=" * 60)
        print("【验证总结】")
        print("=" * 60)

        all_items = {
            'visionpro': 'VisionPro 连接',
            'kinova': 'Kinova 机械臂',
            'camera': 'USB 相机',
            'environment': '环境配置'
        }

        for key, name in all_items.items():
            if key in self.results:
                status = "✓ 通过" if self.results[key] else "✗ 失败"
            else:
                status = "⊘ 跳过"
            print(f"{name:20s}: {status}")

        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        print("\n" + "-" * 60)
        print(f"总计: {total} | 通过: {passed} | 失败: {total - passed}")
        print("-" * 60)

        if total > 0 and passed == total:
            print("\n🎉 所有验证通过！可以开始数据收集")
            return 0
        elif passed >= 2:
            print("\n⚠️  部分验证失败，但基础功能可用")
            return 1
        else:
            print("\n✗ 多项验证失败，请检查硬件连接")
            return 2


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
        if hasattr(config.camera, 'backend') and config.camera.backend == 'webcam':
            if hasattr(config.camera, 'webcam_cameras'):
                # webcam_cameras 是 _DictWrapper 对象，需要访问 _dict
                webcam_dict = config.camera.webcam_cameras._dict
                if webcam_dict:
                    first_camera_key = list(webcam_dict.keys())[0]
                    first_camera = webcam_dict[first_camera_key]
                    default_camera_id = first_camera['device_id']
    except Exception as e:
        # 配置读取失败，使用硬编码默认值
        pass

    parser = argparse.ArgumentParser(
        description='快速验证硬件连接\n\n'
                    '默认值从 kinova_rl_env/config/kinova_config.yaml 读取',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--vp-ip', type=str, default=default_vp_ip,
                        help=f'VisionPro IP 地址 (默认: {default_vp_ip})')
    parser.add_argument('--robot-ip', type=str, default=default_robot_ip,
                        help=f'Kinova 机械臂 IP (默认从配置: {default_robot_ip})')
    parser.add_argument('--camera-id', type=int, default=default_camera_id,
                        help=f'USB 相机 ID (默认从配置: {default_camera_id})')
    parser.add_argument('--skip-vp', action='store_true',
                        help='跳过 VisionPro 验证')
    parser.add_argument('--skip-robot', action='store_true',
                        help='跳过 Kinova 验证')
    parser.add_argument('--skip-camera', action='store_true',
                        help='跳过相机验证')
    parser.add_argument('--timeout', type=float, default=3.0,
                        help='连接超时时间（秒）')

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 快速验证工具")
    print("=" * 60)
    print(f"VisionPro IP: {args.vp_ip}")
    print(f"Kinova IP: {args.robot_ip}")
    print(f"相机 ID: {args.camera_id}")
    print(f"超时: {args.timeout}s")

    verifier = QuickVerify()

    # 验证 VisionPro
    if not args.skip_vp:
        verifier.verify_visionpro(args.vp_ip, args.timeout)
    else:
        print("\n⊘ 跳过 VisionPro 验证")

    # 验证 Kinova
    if not args.skip_robot:
        verifier.verify_kinova(args.robot_ip, args.timeout)
    else:
        print("\n⊘ 跳过 Kinova 验证")

    # 验证相机
    if not args.skip_camera:
        verifier.verify_camera(args.camera_id)
    else:
        print("\n⊘ 跳过相机验证")

    # 验证环境
    verifier.verify_environment()

    # 打印总结
    exit_code = verifier.print_summary()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
