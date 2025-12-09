#!/usr/bin/env python3
"""验证雅可比计算精度"""

import numpy as np
import sys
sys.path.append('/home/user/kinova-hil-serl')

from vision_pro_control.core.joint_velocity_commander import JointVelocityCommander


def numerical_jacobian(commander, q, delta=1e-6):
    """数值雅可比（用于验证）"""
    J_num = np.zeros((6, 7))

    # 暂时设置关节位置
    commander.current_joint_positions = q

    # 计算当前 TCP 位姿
    pose_0 = commander.get_tcp_pose()
    if pose_0 is None:
        return None

    pos_0 = pose_0[:3]

    # 对每个关节扰动
    for i in range(7):
        q_pert = q.copy()
        q_pert[i] += delta

        commander.current_joint_positions = q_pert
        pose_pert = commander.get_tcp_pose()

        # 位置变化 / delta
        J_num[:3, i] = (pose_pert[:3] - pos_0) / delta

        # 简化：只验证位置雅可比（姿态雅可比需要更复杂的四元数微分）

    return J_num


def main():
    # 创建一个临时 commander（不初始化 ROS2）
    class DummyCommander:
        def __init__(self):
            self.current_joint_positions = None

        def get_tcp_pose(self):
            commander = JointVelocityCommander.__new__(JointVelocityCommander)
            commander.current_joint_positions = self.current_joint_positions
            return JointVelocityCommander.get_tcp_pose(commander)

        def compute_jacobian_analytical(self, q):
            commander = JointVelocityCommander.__new__(JointVelocityCommander)
            return JointVelocityCommander.compute_jacobian_analytical(commander, q)

    commander = DummyCommander()

    # 测试几个配置
    configs = [
        np.zeros(7),  # 零位
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # 中间位置
        np.random.randn(7) * 0.5,  # 随机位置
    ]

    print("验证雅可比计算精度\n")

    for idx, q in enumerate(configs):
        print(f"配置 {idx + 1}: q = {q}")

        # 解析雅可比
        J_analytical = commander.compute_jacobian_analytical(q)

        # 数值雅可比
        J_numerical = numerical_jacobian(commander, q)

        if J_numerical is None:
            print("  无法计算数值雅可比\n")
            continue

        # 比较位置雅可比（前3行）
        diff = np.linalg.norm(J_analytical[:3, :] - J_numerical[:3, :])
        print(f"  位置雅可比误差: {diff:.6e}")

        if diff < 1e-4:
            print("  ✓ 验证通过\n")
        else:
            print("  ✗ 误差较大")
            print(f"  解析雅可比（位置部分）:\n{J_analytical[:3, :]}")
            print(f"  数值雅可比（位置部分）:\n{J_numerical[:3, :]}\n")


if __name__ == '__main__':
    main()
