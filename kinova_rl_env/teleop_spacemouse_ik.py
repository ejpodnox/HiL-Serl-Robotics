#!/usr/bin/env python3
"""
SpaceMouse teleoperation using Cartesian IK -> joint trajectory.

Workflow:
  - Read SpaceMouse (6-DoF twist)
  - Integrate a small delta to a target EE pose
  - Solve IK (KDL) to get joint targets
  - Send joint positions via joint_trajectory_controller
"""

import sys
import select
import termios
import tty
import time
import numpy as np
from pathlib import Path

from scipy.spatial.transform import Rotation

from kinova_rl_env.kinova_env.kinova_interface import KinovaInterface
from kinova_rl_env.kinova_env.config_loader import KinovaConfig
from kinova_rl_env.kinova_env.motion_planner import PinMotionPlanner

# SpaceMouse driver
try:
    from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    sm_path = repo_root / "hil-serl" / "serl_robot_infra"
    if sm_path.exists():
        sys.path.insert(0, str(sm_path.parent))
    from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert


class KeyboardMonitor:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout=0.0):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SpaceMouse IK teleoperation (Cartesian->joint trajectory)")
    parser.add_argument("--config", type=str, default="kinova_rl_env/config/kinova_config.yaml", help="Kinova config")
    parser.add_argument("--urdf", type=str,
                        default="/home/cli/AAAA/kinova-hil-serl/ros2_kortex_ws/src/ros2_kortex/kortex_description/robots/gen3_2f85.urdf",
                        help="URDF path for IK")
    parser.add_argument("--ee_frame", type=str, default=None, help="End-effector frame name (override)")
    parser.add_argument("--base_frame", type=str, default=None, help="Base frame name (informational)")
    parser.add_argument("--translation_gain", type=float, default=0.3, help="Translation gain (m/s)")
    parser.add_argument("--rotation_gain", type=float, default=0.8, help="Rotation gain (rad/s)")
    parser.add_argument("--deadband", type=float, default=0.02, help="Deadband threshold")
    parser.add_argument("--control_freq", type=float, default=20.0, help="Control frequency (Hz)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("【SpaceMouse IK Cartesian 遥操作】")
    print("=" * 60)

    # Load config
    cfg = KinovaConfig.from_yaml(args.config)

    # Planner (KDL IK/FK)
    base_link = args.base_frame or cfg.ros2.get('base_frame') or 'gen3_base_link'
    tool_frame_cfg = args.ee_frame or cfg.ros2.get('tool_frame')
    ee_candidates = []
    if tool_frame_cfg:
        ee_candidates.append(tool_frame_cfg)
    ee_candidates += [
        'gen3_end_effector_link',
        'end_effector_link',
        'tool_frame',
        'tcp',
        'tool_link',
    ]

    print("初始化 Pinocchio IK...")
    planner = PinMotionPlanner(
        urdf_path=args.urdf,
        base_link=base_link,
        end_effector_link=ee_candidates[0],
        max_iterations=20,
        position_tolerance=1e-3,
        ee_candidates=ee_candidates,
        active_joints=getattr(cfg.robot, 'joint_names', None) or [f"joint_{i}" for i in range(1, 8)],
    )
    planner_ready = getattr(planner, "model", None) is not None
    if not planner_ready:
        print(f"❌ 无法初始化 Pinocchio IK，请检查URDF路径与末端frame: {ee_candidates}")
        return

    # Interface
    print("\n初始化机器人接口...")
    interface = KinovaInterface(
        node_name="spacemouse_ik_teleop",
        joint_state_topic=cfg.ros2.get('joint_state_topic', '/joint_states'),
        trajectory_topic=cfg.ros2.get('trajectory_command_topic', '/joint_trajectory_controller/joint_trajectory'),
        gripper_command_topic=cfg.ros2.get('gripper_command_topic', '/robotiq_gripper_controller/gripper_cmd'),
        base_frame=base_link,
        tool_frame=planner.end_effector_link,
        joint_names=getattr(cfg.robot, 'joint_names', None) or [f"joint_{i}" for i in range(1, 8)],
    )
    interface.connect()
    # Wait briefly for joint_states to arrive
    has_state = False
    for _ in range(20):
        interface._executor.spin_once(timeout_sec=0.05)
        if interface._latest_joint_state is not None:
            has_state = True
            break
        time.sleep(0.05)
    if not has_state:
        print("⚠️ 未获取 joint_states，命令可能被忽略（检查控制器是否激活）")
    print("✓ 机器人接口已连接")

    # SpaceMouse
    print("\n初始化 SpaceMouse...")
    spacemouse = SpaceMouseExpert()
    print("✓ SpaceMouse 已连接")

    dt = 1.0 / args.control_freq

    print("\n" + "=" * 60)
    print("开始遥操作 (IK → JointTrajectory)")
    print("=" * 60)
    print("按 'q' 退出")
    print("=" * 60 + "\n")

    try:
        with KeyboardMonitor() as kb:
            while True:
                loop_start = time.time()

                key = kb.get_key(timeout=0.001)
                if key == 'q':
                    print("\n退出遥操作...")
                    break

                # Read state
                q, _ = interface.get_joint_state()
                if q is None:
                    continue

                fk = planner.forward_kinematics(q)
                if fk is None:
                    continue
                ee_pos, ee_rot = fk

                # SpaceMouse input
                raw, buttons = spacemouse.get_action()
                action = np.array(raw, dtype=np.float32)
                action[np.abs(action) < args.deadband] = 0.0

                # Twist -> target pose
                twist = np.zeros(6, dtype=np.float32)
                twist[:3] = action[:3] * args.translation_gain
                twist[3:] = action[3:] * args.rotation_gain

                target_pos, target_rot = planner.apply_twist(ee_pos, ee_rot, twist, dt)

                # IK
                q_target = planner.solve_ik(target_pos, target_rot, seed_joints=q)
                if q_target is None:
                    continue

                # Send joint trajectory point
                interface.send_joint_positions(q_target, duration=dt)

                # Gripper buttons
                if buttons:
                    if buttons[0]:
                        interface.send_gripper_command(1.0)
                    elif len(buttons) > 1 and buttons[1]:
                        interface.send_gripper_command(0.0)

                # Timing
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n键盘中断...")
    finally:
        try:
            spacemouse.close()
        except Exception:
            pass
        try:
            interface.send_joint_velocities(np.zeros(interface.num_joints))
        except Exception:
            pass
        interface.disconnect()
        print("✓ 清理完成\n")


if __name__ == "__main__":
    main()
