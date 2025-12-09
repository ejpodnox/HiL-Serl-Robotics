#!/usr/bin/env python3
"""
Use a 3Dconnexion SpaceMouse to collect Kinova demonstrations (HIL-SERL format).

Controls:
  Space  - start recording a demo (reset env)
  s      - mark current demo success and save
  f      - mark current demo as failure and discard
  r      - reset env (no save)
  p      - pause/resume recording
  q      - quit

Buttons on the SpaceMouse:
  left  button  -> close gripper (value=1.0 while held)
  right button  -> open gripper  (value=0.0 while held)
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import sys
import select
import termios
import tty

from kinova_rl_env.kinova_env.kinova_env import KinovaEnv
from kinova_rl_env.kinova_env.config_loader import KinovaConfig

# Simple keyboard monitor (non-blocking) without external deps
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


# SpaceMouse driver from serl_robot_infra (vendored under hil-serl/serl_robot_infra)
try:
    from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    sm_path = repo_root / "hil-serl" / "serl_robot_infra"
    if sm_path.exists():
        # Insert the parent so Python can find the `serl_robot_infra` package dir
        sys.path.insert(0, str(sm_path.parent))
    try:
        from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
    except ImportError as exc:
        raise ImportError(
            "未找到 serl_robot_infra（SpaceMouse 驱动）。请运行 `pip install -e hil-serl/serl_robot_infra`，"
            "或确保路径正确后重试。"
        ) from exc


def spacemouse_to_action(
    raw_action: np.ndarray,
    buttons,
    translation_gain: float,
    rotation_gain: float,
    deadband: float,
    last_gripper: float,
    use_gripper: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Map SpaceMouse readings to a Kinova action vector.

    Returns:
        action: np.ndarray shape (7,) -> [dx, dy, dz, drx, dry, drz, gripper]
        gripper: latest gripper value (carry over if buttons not pressed)
    """
    action_vec = np.array(raw_action, dtype=np.float32)
    action_vec[np.abs(action_vec) < deadband] = 0.0

    linear = action_vec[:3] * translation_gain
    angular = action_vec[3:6] * rotation_gain

    # Clip to the normalized range expected by the env
    delta = np.concatenate([linear, angular])
    delta = np.clip(delta, -1.0, 1.0)

    gripper = last_gripper
    if use_gripper and buttons:
        # buttons[0]=left, buttons[1]=right for single device
        if buttons[0]:
            gripper = 1.0
        elif len(buttons) > 1 and buttons[1]:
            gripper = 0.0

    if use_gripper:
        delta = np.concatenate([delta, np.array([gripper], dtype=np.float32)])

    return delta.astype(np.float32), gripper


def save_demo(trajectory, save_dir: Path, demo_id: int, success: bool = True):
    """Persist a trajectory using the HIL-SERL format."""
    demo_path = save_dir / f"demo_{demo_id:03d}.pkl"

    demo_data = {
        "observations": [t["observation"] for t in trajectory],
        "actions": [t["action"] for t in trajectory],
        "rewards": [t["reward"] for t in trajectory],
        "terminals": [t["terminated"] for t in trajectory],
        "truncations": [t["truncated"] for t in trajectory],
        "success": success,
    }

    with open(demo_path, "wb") as f:
        pickle.dump(demo_data, f)

    print(f"    ✓ 已保存: {demo_path.name}")


def main():
    # Defaults from kinova_config.yaml
    default_robot_ip = "192.168.8.10"
    try:
        default_cfg = KinovaConfig.from_yaml("kinova_rl_env/config/kinova_config.yaml")
        if hasattr(default_cfg, "robot") and hasattr(default_cfg.robot, "ip"):
            default_robot_ip = default_cfg.robot.ip
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="SpaceMouse demo collection for Kinova")
    parser.add_argument("--save_dir", type=str, default="./demos", help="保存demo的目录")
    parser.add_argument("--num_demos", type=int, default=10, help="需要收集的demo数量")
    parser.add_argument("--task", type=str, default="reaching", help="任务名称")
    parser.add_argument(
        "--config", type=str, default="kinova_rl_env/config/kinova_config.yaml", help="Kinova配置文件路径"
    )
    parser.add_argument("--translation_gain", type=float, default=0.6, help="平移轴增益")
    parser.add_argument("--rotation_gain", type=float, default=0.6, help="旋转轴增益")
    parser.add_argument("--deadband", type=float, default=0.02, help="轴向死区，过滤噪声")
    parser.add_argument("--no_gripper", action="store_true", help="禁用gripper映射")
    parser.add_argument("--robot_ip", type=str, default=default_robot_ip, help="Kinova IP (覆盖配置)")

    args = parser.parse_args()

    # ============================================================
    # 1. 初始化 Kinova 环境
    # ============================================================
    print("\n" + "=" * 60)
    print("【初始化Kinova RL环境】")
    print("=" * 60)

    config = KinovaConfig.from_yaml(args.config)
    # 可选覆盖 IP
    if args.robot_ip:
        config.robot.ip = args.robot_ip

    env = KinovaEnv(config=config)

    print(f"✓ 环境已初始化")
    print(f"  任务: {args.task}")
    print(f"  控制频率: {config.control.frequency} Hz")
    print(f"  时间步长: {config.control.dt} s")

    # ============================================================
    # 2. 初始化 SpaceMouse
    # ============================================================
    print("\n" + "=" * 60)
    print("【初始化SpaceMouse】")
    print("=" * 60)
    spacemouse = SpaceMouseExpert()
    print("✓ SpaceMouse 已连接（移动设备查看反馈）")

    # ============================================================
    # 3. 数据保存目录
    # ============================================================
    save_dir = Path(args.save_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 数据保存目录: {save_dir}")

    # ============================================================
    # 4. 数据收集循环
    # ============================================================
    print("\n" + "=" * 60)
    print(f"【准备收集 {args.num_demos} 条演示】")
    print("=" * 60)
    print("按键说明:")
    print("  Space  - 开始记录当前演示（重置环境）")
    print("  's'    - 标记当前演示为成功并保存")
    print("  'f'    - 标记当前演示为失败并丢弃")
    print("  'r'    - 重置环境（不记录）")
    print("  'p'    - 暂停/恢复记录")
    print("  'q'    - 退出程序")
    print("=" * 60 + "\n")

    demo_count = 0
    recording = False
    paused = False
    trajectory = []
    gripper_state = 0.0

    with KeyboardMonitor() as kb:
        while demo_count < args.num_demos:
            key = kb.get_key(timeout=0.01)

            if key == "q":
                print("\n✗ 用户退出")
                break

            elif key == " ":
                if not recording:
                    print(f"\n>>> 【Demo #{demo_count + 1}】开始记录 <<<")
                    recording = True
                    paused = False
                    trajectory = []
                    gripper_state = 0.0

                    obs, info = env.reset()
                    print(f"  环境已重置")
                    print(f"  目标位置: {config.task.target_pose[:3]}")

            elif key == "s" and recording:
                success_count = sum([t["reward"] > 0 for t in trajectory])
                print(f"\n✓ 【Demo #{demo_count + 1}】标记为成功")
                print(f"  轨迹长度: {len(trajectory)} 步")
                print(f"  成功步数: {success_count}")

                save_demo(trajectory, save_dir, demo_count, success=True)
                demo_count += 1
                recording = False
                trajectory = []

            elif key == "f" and recording:
                print(f"\n✗ 【Demo #{demo_count + 1}】标记为失败，已丢弃")
                recording = False
                trajectory = []

            elif key == "r":
                print("\n⟳ 重置环境...")
                obs, info = env.reset()
                recording = False
                trajectory = []

            elif key == "p" and recording:
                paused = not paused
                if paused:
                    print("\n⏸  记录已暂停")
                else:
                    print("\n▶  记录已恢复")

            # ====================================================
            # 记录循环
            # ====================================================
            if recording and not paused:
                try:
                    raw_action, buttons = spacemouse.get_action()
                    action, gripper_state = spacemouse_to_action(
                        raw_action,
                        buttons,
                        translation_gain=args.translation_gain,
                        rotation_gain=args.rotation_gain,
                        deadband=args.deadband,
                        last_gripper=gripper_state,
                        use_gripper=not args.no_gripper,
                    )

                    obs, reward, terminated, truncated, info = env.step(action)

                    trajectory.append(
                        {
                            "observation": obs,
                            "action": action,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                        }
                    )

                    if len(trajectory) % 10 == 0:
                        distance = 0.0
                        try:
                            distance = float(env._compute_distance_to_target(obs))
                        except Exception:
                            pass
                        cumulative_reward = sum([t["reward"] for t in trajectory])
                        print(
                            f"  步数: {len(trajectory):3d} | "
                            f"距离目标: {distance:.3f}m | "
                            f"累积奖励: {cumulative_reward:.2f}"
                        )

                    if terminated or truncated:
                        if reward > 0:
                            print(f"\n✓ 任务成功完成！自动保存...")
                            save_demo(trajectory, save_dir, demo_count, success=True)
                            demo_count += 1
                        else:
                            print(f"\n✗ Episode结束（超时），请按's'保存或'f'丢弃")
                        recording = False
                        trajectory = []

                except Exception as e:
                    print(f"\n✗ 错误: {e}")
                    import traceback

                    traceback.print_exc()
                    recording = False
                    trajectory = []

            time.sleep(config.control.dt)

    # ============================================================
    # 清理
    # ============================================================
    print("\n" + "=" * 60)
    print("【清理资源】")
    print("=" * 60)
    try:
        spacemouse.close()
    except Exception:
        pass
    env.close()

    print(f"\n✓ 完成！共收集 {demo_count} 条演示")
    print(f"✓ 保存位置: {save_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
