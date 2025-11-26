#!/usr/bin/env python3
"""测试 Joint Velocity Commander"""

import rclpy
import time
import numpy as np
from vision_pro_control.core import JointVelocityCommander


def main():
    rclpy.init()

    commander = JointVelocityCommander(robot_ip='192.168.8.10')

    print("等待关节状态...")
    for _ in range(20):
        rclpy.spin_once(commander, timeout_sec=0.1)
        if commander.current_joint_positions is not None:
            break
        time.sleep(0.1)

    if commander.current_joint_positions is None:
        print("✗ 无法获取关节状态")
        return

    print("✓ 关节状态已获取")

    input("\n按 Enter 开始测试（向上移动2秒）...")

    # 向上移动
    twist = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    for _ in range(100):
        commander.send_twist(twist)
        rclpy.spin_once(commander, timeout_sec=0.01)
        time.sleep(0.02)

    commander.send_zero_twist()
    print("✓ 测试完成")


if __name__ == '__main__':
    main()
