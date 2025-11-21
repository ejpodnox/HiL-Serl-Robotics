#!/usr/bin/env python3
"""
使用VisionPro遥操作收集Kinova演示数据

功能：
1. 初始化KinovaEnv (RL环境)
2. 初始化VisionPro遥操作 (手部追踪)
3. 用户通过VisionPro控制机械臂完成任务
4. 记录轨迹数据 (observations, actions, rewards)
5. 保存为HIL-SERL格式的.pkl文件

使用方法:
    python record_kinova_demos.py --save_dir ./demos --num_demos 10 --task reaching
"""

import numpy as np
import pickle
import time
from pathlib import Path
import sys

# Kinova RL环境
sys.path.insert(0, str(Path(__file__).parent))
from kinova_env.kinova_env import KinovaEnv
from kinova_env.config_loader import KinovaConfig

# VisionPro遥操作
sys.path.insert(0, str(Path(__file__).parent.parent))
from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.coordinate_mapper import CoordinateMapper
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


def twist_to_action(twist, dt, gripper_position):
    """
    将Twist速度转换为RL action (delta pose)

    Args:
        twist: dict, {'linear': {x, y, z}, 'angular': {x, y, z}}
        dt: float, 时间步长 (秒)
        gripper_position: float, 0-1

    Returns:
        action: np.array (7,) [dx, dy, dz, drx, dry, drz, gripper]
    """
    # Twist是速度，乘以dt得到位移增量
    delta_pos = np.array([
        twist['linear']['x'] * dt,
        twist['linear']['y'] * dt,
        twist['linear']['z'] * dt
    ])

    delta_rot = np.array([
        twist['angular']['x'] * dt,
        twist['angular']['y'] * dt,
        twist['angular']['z'] * dt
    ])

    # 拼接 [delta_position(3), delta_rotation(3), gripper(1)]
    action = np.concatenate([delta_pos, delta_rot, [gripper_position]])

    return action


def main():
    import argparse

    parser = argparse.ArgumentParser(description='收集Kinova演示数据（VisionPro遥操作）')
    parser.add_argument('--save_dir', type=str, default='./demos', help='保存demo的目录')
    parser.add_argument('--num_demos', type=int, default=10, help='需要收集的demo数量')
    parser.add_argument('--task', type=str, default='reaching', help='任务名称')
    parser.add_argument('--config', type=str, default='kinova_env/config/kinova_config.yaml',
                        help='Kinova配置文件路径')
    parser.add_argument('--vp_ip', type=str, default='192.168.1.125', help='VisionPro IP地址')
    parser.add_argument('--use_right_hand', action='store_true', default=True, help='使用右手')

    args = parser.parse_args()

    # ============================================================
    # 1. 初始化Kinova环境
    # ============================================================
    print("\n" + "=" * 60)
    print("【初始化Kinova RL环境】")
    print("=" * 60)

    config_path = Path(__file__).parent / args.config
    config = KinovaConfig.from_yaml(config_path)
    env = KinovaEnv(config=config)

    print(f"✓ 环境已初始化")
    print(f"  任务: {args.task}")
    print(f"  控制频率: {config.control.frequency} Hz")
    print(f"  时间步长: {config.control.dt} s")

    # ============================================================
    # 2. 初始化VisionPro遥操作
    # ============================================================
    print("\n" + "=" * 60)
    print("【初始化VisionPro遥操作】")
    print("=" * 60)

    # VisionPro连接
    vp_bridge = VisionProBridge(
        avp_ip=args.vp_ip,
        use_right_hand=args.use_right_hand
    )
    print(f"✓ 连接VisionPro: {args.vp_ip}")
    vp_bridge.start()
    time.sleep(1.0)
    print(f"✓ VisionPro数据流已启动")

    # 坐标映射器
    calibration_file = Path(__file__).parent.parent / "vision_pro_control/config/calibration.yaml"
    mapper = CoordinateMapper(calibration_file=calibration_file)

    # 设置低增益（因为我们要记录离散动作，不是连续速度控制）
    mapper.set_gains(position_gain=0.5, rotation_gain=0.5)
    mapper.set_velocity_limits(max_linear=0.05, max_angular=0.2)
    print(f"✓ 坐标映射器已初始化")

    # ============================================================
    # 3. 准备数据保存
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

    with KeyboardMonitor() as kb:
        while demo_count < args.num_demos:
            # 检查按键
            key = kb.get_key(timeout=0.01)

            if key == 'q':
                print("\n✗ 用户退出")
                break

            elif key == ' ':
                if not recording:
                    # 开始新的记录
                    print(f"\n>>> 【Demo #{demo_count + 1}】开始记录 <<<")
                    recording = True
                    paused = False
                    trajectory = []

                    # 重置环境
                    obs, info = env.reset()
                    print(f"  环境已重置")
                    print(f"  目标位置: {config.task.target_pose[:3]}")

            elif key == 's' and recording:
                # 标记成功并保存
                success_count = sum([t['reward'] > 0 for t in trajectory])
                print(f"\n✓ 【Demo #{demo_count + 1}】标记为成功")
                print(f"  轨迹长度: {len(trajectory)} 步")
                print(f"  成功步数: {success_count}")

                save_demo(trajectory, save_dir, demo_count, success=True)
                demo_count += 1
                recording = False
                trajectory = []

            elif key == 'f' and recording:
                # 标记失败并丢弃
                print(f"\n✗ 【Demo #{demo_count + 1}】标记为失败，已丢弃")
                recording = False
                trajectory = []

            elif key == 'r':
                # 重置环境（不记录）
                print("\n⟳ 重置环境...")
                obs, info = env.reset()
                recording = False
                trajectory = []

            elif key == 'p' and recording:
                # 暂停/恢复
                paused = not paused
                if paused:
                    print("\n⏸  记录已暂停")
                else:
                    print("\n▶  记录已恢复")

            # ============================================================
            # 记录循环
            # ============================================================
            if recording and not paused:
                try:
                    # 1. 获取VisionPro手部位姿
                    position, rotation = vp_bridge.get_hand_relative_to_head()

                    # 2. 映射到Twist
                    twist = mapper.map_to_twist(position, rotation)

                    # 3. 获取gripper状态
                    pinch_distance = vp_bridge.get_pinch_distance()
                    # 映射pinch distance到gripper position
                    # pinch_distance: 0.01(闭合) ~ 0.08(张开)
                    # gripper_position: 0.0(张开) ~ 1.0(闭合)
                    gripper_position = np.clip(1.0 - (pinch_distance - 0.01) / 0.07, 0.0, 1.0)

                    # 4. 转换Twist为Action
                    dt = config.control.dt
                    action = twist_to_action(twist, dt, gripper_position)

                    # 5. 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)

                    # 6. 记录到轨迹
                    trajectory.append({
                        'observation': obs,
                        'action': action,
                        'reward': reward,
                        'terminated': terminated,
                        'truncated': truncated
                    })

                    # 7. 打印状态（每10步）
                    if len(trajectory) % 10 == 0:
                        distance = env._compute_distance_to_target(obs)
                        cumulative_reward = sum([t['reward'] for t in trajectory])
                        print(f"  步数: {len(trajectory):3d} | "
                              f"距离目标: {distance:.3f}m | "
                              f"累积奖励: {cumulative_reward:.2f}")

                    # 8. 自动停止（如果到达目标或超时）
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

            # 控制循环频率
            time.sleep(config.control.dt)

    # ============================================================
    # 清理
    # ============================================================
    print("\n" + "=" * 60)
    print("【清理资源】")
    print("=" * 60)
    vp_bridge.stop()
    env.close()

    print(f"\n✓ 完成！共收集 {demo_count} 条演示")
    print(f"✓ 保存位置: {save_dir}")
    print("=" * 60 + "\n")


def save_demo(trajectory, save_dir, demo_id, success=True):
    """
    保存一条演示轨迹（HIL-SERL格式）

    Args:
        trajectory: list of dicts
        save_dir: Path
        demo_id: int
        success: bool
    """
    demo_path = save_dir / f"demo_{demo_id:03d}.pkl"

    # 转换为HIL-SERL格式
    demo_data = {
        'observations': [t['observation'] for t in trajectory],
        'actions': [t['action'] for t in trajectory],
        'rewards': [t['reward'] for t in trajectory],
        'terminals': [t['terminated'] for t in trajectory],
        'truncations': [t['truncated'] for t in trajectory],
        'success': success
    }

    with open(demo_path, 'wb') as f:
        pickle.dump(demo_data, f)

    print(f"    ✓ 已保存: {demo_path.name}")


if __name__ == '__main__':
    main()
