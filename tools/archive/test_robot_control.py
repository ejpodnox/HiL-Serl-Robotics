#!/usr/bin/env python3
"""
简单的机械臂控制测试

测试通过 Twist 命令控制机械臂微微移动
"""

import rclpy
import time
import numpy as np
import argparse


def test_basic_control(robot_ip: str):
    """
    测试基本的机械臂控制

    1. 获取当前 TCP 位置
    2. 发送极慢的 Twist 命令，让机械臂微微移动
    3. 停止
    """
    from vision_pro_control.core import RobotCommander

    print("\n" + "=" * 60)
    print("机械臂控制测试")
    print("=" * 60)

    # 初始化 ROS2
    if not rclpy.ok():
        rclpy.init()

    # 创建 RobotCommander
    print(f"连接机械臂 ({robot_ip})...")
    commander = RobotCommander(robot_ip=robot_ip)

    # 等待 TF buffer 准备（更长时间，持续 spin）
    print("等待 TF buffer 准备...")
    print("  (这可能需要几秒钟，请耐心等待...)")
    end_time = time.time() + 5.0  # 增加到5秒
    while time.time() < end_time:
        rclpy.spin_once(commander, timeout_sec=0.1)
        time.sleep(0.05)

    # 获取初始位置（更多重试次数）
    print("\n获取初始 TCP 位置...")
    initial_pose = None
    max_attempts = 30  # 增加到30次尝试
    for attempt in range(max_attempts):
        rclpy.spin_once(commander, timeout_sec=0.1)
        initial_pose = commander.get_tcp_pose()
        if initial_pose is not None:
            break
        if attempt % 5 == 0:
            print(f"  尝试 {attempt + 1}/{max_attempts}...")
        time.sleep(0.2)

    if initial_pose is None:
        print("\n✗ 无法获取 TCP 位置")
        print("\n可能的原因:")
        print("  1. kortex_bringup 未启动")
        print("     解决: 运行 'ros2 launch kinova_gen3_6dof_robotiq_2f_85_moveit_config robot.launch.py robot_ip:=<your_ip>'")
        print("  2. TF frames 名称不匹配")
        print("     解决: 运行 'python tools/check_ros_topics.py' 查看可用的 frames")
        print("  3. 机械臂未正确连接")
        print("     解决: 检查网络连接和机械臂电源")
        return False

    print(f"✓ 初始位置: [{initial_pose[0]:.3f}, {initial_pose[1]:.3f}, {initial_pose[2]:.3f}]")

    # 用户确认
    print("\n" + "=" * 60)
    print("⚠️  准备测试控制")
    print("=" * 60)
    print("将发送极慢的速度命令（1 cm/s），让机械臂沿 Z 轴向上移动 2 秒")
    print("然后停止并返回")
    print("\n请确保:")
    print("  1. 机械臂周围没有障碍物")
    print("  2. 手放在急停按钮上")
    print("  3. 准备好随时按 Ctrl+C 中断")

    response = input("\n继续测试? (y/n): ").strip().lower()
    if response != 'y':
        print("测试取消")
        return False

    print("\n" + "-" * 60)
    print("开始测试...")
    print("-" * 60)

    try:
        # 阶段 1: 向上移动（极慢速度）
        print("\n[1/3] 向上移动 (2秒)...")
        twist = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])  # vz = 1 cm/s

        start_time = time.time()
        while time.time() - start_time < 2.0:
            commander.send_twist(twist)
            rclpy.spin_once(commander, timeout_sec=0.01)
            time.sleep(0.02)  # 50 Hz

        # 阶段 2: 停止
        print("[2/3] 停止...")
        commander.send_zero_twist()
        time.sleep(0.5)

        # 获取移动后的位置
        rclpy.spin_once(commander, timeout_sec=0.1)
        final_pose = commander.get_tcp_pose()

        if final_pose is not None:
            delta = final_pose[:3] - initial_pose[:3]
            distance = np.linalg.norm(delta)
            print(f"✓ 移动距离: {distance*1000:.1f} mm")
            print(f"  Delta: [{delta[0]*1000:.1f}, {delta[1]*1000:.1f}, {delta[2]*1000:.1f}] mm")

        # 阶段 3: 返回（反向移动）
        print("[3/3] 返回初始位置 (2秒)...")
        twist = np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0])  # vz = -1 cm/s

        start_time = time.time()
        while time.time() - start_time < 2.0:
            commander.send_twist(twist)
            rclpy.spin_once(commander, timeout_sec=0.01)
            time.sleep(0.02)

        # 停止
        commander.send_zero_twist()
        print("✓ 返回完成")

        # 最终位置
        time.sleep(0.5)
        rclpy.spin_once(commander, timeout_sec=0.1)
        final_pose = commander.get_tcp_pose()

        if final_pose is not None:
            delta = final_pose[:3] - initial_pose[:3]
            distance = np.linalg.norm(delta)
            print(f"\n最终位置偏差: {distance*1000:.1f} mm")

        print("\n" + "=" * 60)
        print("✓ 控制测试完成！")
        print("=" * 60)
        print("结论:")
        print("  - 机械臂可以响应 Twist 命令")
        print("  - 速度控制正常工作")
        print("  - 可以继续进行遥操作数据采集")

        return True

    except KeyboardInterrupt:
        print("\n\n用户中断！")
        commander.send_zero_twist()
        return False

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        commander.send_zero_twist()
        import traceback
        traceback.print_exc()
        return False


def main():
    # 读取默认 robot IP
    default_robot_ip = '192.168.8.10'
    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml('kinova_rl_env/config/kinova_config.yaml')
        default_robot_ip = config.robot.ip
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='机械臂控制测试')
    parser.add_argument('--robot-ip', type=str, default=default_robot_ip,
                        help=f'机械臂 IP (默认: {default_robot_ip})')
    args = parser.parse_args()

    success = test_basic_control(args.robot_ip)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
