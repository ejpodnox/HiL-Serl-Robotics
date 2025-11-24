#!/usr/bin/env python3
"""
测试 Kinova 机械臂连接

独立测试，不依赖其他组件。
"""

import argparse
import time
import numpy as np


def test_kinova_imports():
    """测试导入"""
    print("\n" + "=" * 60)
    print("【测试 1】Kinova 模块导入")
    print("=" * 60)

    try:
        from kinova_rl_env.kinova_env import KinovaInterface
        print("✓ KinovaInterface 导入成功")

        from vision_pro_control.core import RobotCommander
        print("✓ RobotCommander 导入成功")

        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ros2_connection():
    """测试 ROS2 是否运行"""
    print("\n" + "=" * 60)
    print("【测试 2】ROS2 环境检查")
    print("=" * 60)

    try:
        import rclpy

        # 初始化 ROS2（如果未初始化）
        if not rclpy.ok():
            rclpy.init()
            print("✓ ROS2 初始化成功")
        else:
            print("✓ ROS2 已运行")

        return True

    except Exception as e:
        print(f"✗ ROS2 环境检查失败: {e}")
        print("提示: 请确保已安装 ROS2 并 source 了环境")
        return False


def test_robot_commander(robot_ip: str, timeout: float = 5.0):
    """测试 RobotCommander 连接"""
    print("\n" + "=" * 60)
    print("【测试 3】Kinova RobotCommander 连接")
    print("=" * 60)
    print(f"机械臂 IP: {robot_ip}")

    try:
        import rclpy
        from vision_pro_control.core import RobotCommander

        if not rclpy.ok():
            rclpy.init()

        commander = RobotCommander(robot_ip=robot_ip)
        print("✓ RobotCommander 创建成功")

        # 尝试获取机械臂状态
        print(f"尝试获取机械臂状态（超时 {timeout}s）...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pose = commander.get_tcp_pose()
                if pose is not None:
                    print("✓ 成功获取末端位姿:")
                    print(f"  - 位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                    print(f"  - 姿态: [{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}]")
                    return True
            except Exception as e:
                pass

            time.sleep(0.5)

        print(f"✗ {timeout}s 内未能获取机械臂状态")
        print("提示: 请检查机械臂是否已启动 kortex_bringup")
        return False

    except Exception as e:
        print(f"✗ RobotCommander 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kinova_dummy():
    """虚拟机械臂模式（用于无硬件开发）"""
    print("\n" + "=" * 60)
    print("【测试 4】Kinova 虚拟模式")
    print("=" * 60)

    try:
        # 模拟机械臂状态
        dummy_pose = np.array([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])  # [x,y,z, qx,qy,qz,qw]
        dummy_joints = np.zeros(7)

        print("✓ 虚拟机械臂数据生成成功:")
        print(f"  - 末端位置: [{dummy_pose[0]:.3f}, {dummy_pose[1]:.3f}, {dummy_pose[2]:.3f}]")
        print(f"  - 关节角度: {dummy_joints}")

        return True

    except Exception as e:
        print(f"✗ 虚拟模式测试失败: {e}")
        return False


def main():
    # 尝试从配置文件读取默认 robot IP
    default_robot_ip = '192.168.8.10'
    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml('kinova_rl_env/config/kinova_config.yaml')
        default_robot_ip = config.robot.ip
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='Kinova 机械臂连接测试')
    parser.add_argument('--robot_ip', type=str, default=default_robot_ip,
                        help=f'Kinova 机械臂 IP 地址 (默认从配置: {default_robot_ip})')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='连接超时时间（秒）')
    parser.add_argument('--skip-connection', action='store_true',
                        help='跳过真实连接测试')

    args = parser.parse_args()

    results = {}

    # 测试 1: 导入
    results['imports'] = test_kinova_imports()

    # 测试 2: ROS2 环境
    if results['imports']:
        results['ros2'] = test_ros2_connection()
    else:
        results['ros2'] = None

    # 测试 3: 真实连接（可跳过）
    if not args.skip_connection and results['ros2']:
        results['connection'] = test_robot_commander(args.robot_ip, args.timeout)
    else:
        if args.skip_connection:
            print("\n⚠️  跳过真实连接测试")
        results['connection'] = None

    # 测试 4: 虚拟模式
    results['dummy'] = test_kinova_dummy()

    # 总结
    print("\n" + "=" * 60)
    print("【测试总结】")
    print("=" * 60)

    for test_name, result in results.items():
        if result is None:
            status = "⊘ 跳过"
        elif result:
            status = "✓ 通过"
        else:
            status = "✗ 失败"
        print(f"{test_name:20s}: {status}")

    # 基础功能可用
    essential_passed = results['imports'] and results['dummy']

    if essential_passed:
        print("\n✓ 基础功能测试通过，Kinova 模块可用")
        if results['connection'] is False:
            print("⚠️  真实设备连接失败，但可以使用虚拟模式进行开发")
        return 0
    else:
        print("\n✗ 基础测试失败，请检查环境配置")
        return 1


if __name__ == '__main__':
    exit(main())
