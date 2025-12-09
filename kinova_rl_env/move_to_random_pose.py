#!/usr/bin/env python3
"""
Move robot to a random safe joint configuration to escape singularity.
"""
import numpy as np
import time
from kinova_rl_env.kinova_env.kinova_interface import KinovaInterface
from kinova_rl_env.kinova_env.config_loader import KinovaConfig

def main():
    print("\n" + "=" * 60)
    print("【移动到随机姿态】")
    print("=" * 60)

    # Load config
    config = KinovaConfig.from_yaml("kinova_rl_env/config/kinova_config.yaml")

    # Initialize interface
    print("\n初始化机器人接口...")
    interface = KinovaInterface(
        node_name="random_pose_mover",
        joint_state_topic=config.ros2.get('joint_state_topic', '/joint_states'),
        trajectory_topic=config.ros2.get('trajectory_command_topic', '/joint_trajectory_controller/joint_trajectory'),
        twist_topic=config.ros2.get('twist_command_topic', '/twist_controller/commands'),
    )
    interface.connect()
    print("✓ 已连接")

    # IMPORTANT: Wait for publishers to be ready
    print("\n等待发布者初始化...")
    time.sleep(1.0)  # Give ROS2 time to advertise publishers

    # Get current position
    pos, vel = interface.get_joint_state()
    print(f"\n当前关节位置: {np.round(pos, 3)}")

    # Generate random safe pose
    # Use safe ranges to avoid joint limits and singularities
    safe_ranges = [
        (-2.5, 2.5),   # joint_1
        (-1.5, 1.5),   # joint_2
        (-2.5, 2.5),   # joint_3
        (-2.0, 2.0),   # joint_4
        (-2.5, 2.5),   # joint_5
        (-1.5, 1.5),   # joint_6
        (-2.5, 2.5),   # joint_7
    ]

    random_pose = np.array([
        np.random.uniform(low, high) for low, high in safe_ranges
    ])

    print(f"目标随机位置: {np.round(random_pose, 3)}")
    print("\n发送单次轨迹到目标位置...")
    interface.send_joint_positions(random_pose, duration=4.0)
    # 等待执行完成
    time.sleep(4.5)

    final_pos, _ = interface.get_joint_state()
    print(f"最终位置: {np.round(final_pos, 3)}")

    # Cleanup
    interface.disconnect()
    print("\n✓ 完成")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
