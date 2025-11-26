#!/usr/bin/env python3
"""统一的机械臂控制测试（自动选择最佳控制器）"""

import rclpy
import time
import numpy as np
import argparse
from vision_pro_control.core import robot_commander


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-ip', type=str, default='192.168.8.10')
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'joint', 'ros2', 'kortex'])
    args = parser.parse_args()

    rclpy.init()
    commander = robot_commander(robot_ip=args.robot_ip, backend=args.backend)

    print("等待关节状态...")
    for _ in range(20):
        rclpy.spin_once(commander, timeout_sec=0.1)
        if hasattr(commander, 'current_joint_positions') and commander.current_joint_positions is not None:
            break
        time.sleep(0.1)

    if hasattr(commander, 'current_joint_positions'):
        if commander.current_joint_positions is None:
            print("✗ 无法获取关节状态")
            return

    print("✓ 就绪")
    input("\n按 Enter 开始测试...")

    twist = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    for _ in range(100):
        commander.send_twist(twist)
        rclpy.spin_once(commander, timeout_sec=0.01)
        time.sleep(0.02)

    commander.send_zero_twist()
    print("✓ 完成")


if __name__ == '__main__':
    main()
