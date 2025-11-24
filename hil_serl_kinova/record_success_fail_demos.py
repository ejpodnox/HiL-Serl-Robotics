#!/usr/bin/env python3
"""
收集成功/失败标签数据（用于训练 Reward Classifier）

与普通数据收集的区别：
- 明确标记每条演示为成功/失败
- 可以手动标记或自动判定
- 用于训练 Reward Classifier

使用方法:
    python record_success_fail_demos.py \
        --save_dir ./demos/labeled \
        --num_success 20 \
        --num_fail 20
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import rclpy
import time

from kinova_rl_env.kinova_env.kinova_env import KinovaEnv
from kinova_rl_env.kinova_env.config_loader import KinovaConfig
from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.coordinate_mapper import CoordinateMapper
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


def twist_to_action(twist, dt, gripper_position):
    """将 Twist 转换为 Action"""
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

    action = np.concatenate([delta_pos, delta_rot, [gripper_position]])
    return action


def main():
    # 尝试从配置文件读取默认 VisionPro IP
    default_vp_ip = '192.168.1.125'
    default_config_path = 'kinova_rl_env/config/kinova_config.yaml'

    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml(default_config_path)
        # VisionPro IP 可以从配置中读取（如果配置了 visionpro 部分）
        # 这里暂时保持硬编码，因为 kinova_config.yaml 中可能没有 visionpro IP
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='收集成功/失败标签数据')
    parser.add_argument('--save_dir', type=str, default='./demos/labeled',
                        help='保存目录')
    parser.add_argument('--num_success', type=int, default=10,
                        help='需要收集的成功演示数量')
    parser.add_argument('--num_fail', type=int, default=10,
                        help='需要收集的失败演示数量')
    parser.add_argument('--config', type=str,
                        default=default_config_path,
                        help='Kinova配置文件路径')
    parser.add_argument('--vp_ip', type=str, default=default_vp_ip,
                        help=f'VisionPro IP地址 (默认: {default_vp_ip})')
    parser.add_argument('--auto_label', action='store_true',
                        help='自动标记（基于 reward）')

    args = parser.parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("收集成功/失败标签数据")
    print("=" * 60)
    print(f"保存目录: {save_dir}")
    print(f"目标: {args.num_success} 成功, {args.num_fail} 失败")
    print(f"自动标记: {'是' if args.auto_label else '否'}")
    print("=" * 60)

    # 初始化环境
    config = KinovaConfig.from_yaml(args.config)
    env = KinovaEnv(config=config)

    # 初始化 VisionPro
    vp_bridge = VisionProBridge(avp_ip=args.vp_ip, use_right_hand=True)
    vp_bridge.start()
    time.sleep(1.0)

    # 坐标映射器
    calibration_file = Path(__file__).parent.parent / "vision_pro_control/config/calibration.yaml"
    mapper = CoordinateMapper(calibration_file=calibration_file)
    mapper.set_gains(position_gain=0.5, rotation_gain=0.5)

    print("\n按键说明:")
    print("  Space - 开始记录新的演示")
    print("  's'   - 标记为成功并保存")
    print("  'f'   - 标记为失败并保存")
    print("  'r'   - 重置环境（不记录）")
    print("  'q'   - 退出程序")
    print("=" * 60 + "\n")

    success_count = 0
    fail_count = 0
    recording = False
    trajectory = []

    with KeyboardMonitor() as kb:
        while success_count < args.num_success or fail_count < args.num_fail:
            # 检查按键
            key = kb.get_key(timeout=0.01)

            if key == 'q':
                print("\n✗ 用户退出")
                break

            elif key == ' ':
                if not recording:
                    # 开始新的记录
                    total_collected = success_count + fail_count
                    print(f"\n>>> 【Demo #{total_collected + 1}】开始记录 <<<")
                    print(f"  已收集: 成功 {success_count}/{args.num_success}, "
                          f"失败 {fail_count}/{args.num_fail}")
                    recording = True
                    trajectory = []

                    # 重置环境
                    obs, info = env.reset()
                    print(f"  环境已重置")

            elif key == 's' and recording:
                # 标记成功并保存
                if success_count < args.num_success:
                    demo_id = success_count + fail_count
                    save_demo(trajectory, save_dir, demo_id, success=True)
                    success_count += 1
                    print(f"\n✓ 标记为成功并保存")
                    print(f"  进度: 成功 {success_count}/{args.num_success}, "
                          f"失败 {fail_count}/{args.num_fail}")
                else:
                    print(f"\n⚠️  已达到成功演示数量上限")

                recording = False
                trajectory = []

            elif key == 'f' and recording:
                # 标记失败并保存
                if fail_count < args.num_fail:
                    demo_id = success_count + fail_count
                    save_demo(trajectory, save_dir, demo_id, success=False)
                    fail_count += 1
                    print(f"\n✗ 标记为失败并保存")
                    print(f"  进度: 成功 {success_count}/{args.num_success}, "
                          f"失败 {fail_count}/{args.num_fail}")
                else:
                    print(f"\n⚠️  已达到失败演示数量上限")

                recording = False
                trajectory = []

            elif key == 'r':
                # 重置环境
                print("\n⟳ 重置环境...")
                obs, info = env.reset()
                recording = False
                trajectory = []

            # 记录循环
            if recording:
                try:
                    # 获取 VisionPro 数据
                    position, rotation = vp_bridge.get_hand_relative_to_head()
                    twist = mapper.map_to_twist(position, rotation)

                    # Gripper
                    pinch_distance = vp_bridge.get_pinch_distance()
                    gripper_position = np.clip(1.0 - (pinch_distance - 0.01) / 0.07, 0.0, 1.0)

                    # 转换为 Action
                    dt = config.control.dt
                    action = twist_to_action(twist, dt, gripper_position)

                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)

                    # 记录
                    trajectory.append({
                        'observation': obs,
                        'action': action,
                        'reward': reward,
                        'terminated': terminated,
                        'truncated': truncated
                    })

                    # 打印状态
                    if len(trajectory) % 10 == 0:
                        distance = info.get('distance_to_target', -1)
                        print(f"  步数: {len(trajectory):3d} | 距离: {distance:.3f}m")

                    # 自动标记（如果启用）
                    if args.auto_label and (terminated or truncated):
                        is_success = reward > 0  # 根据 reward 判断

                        if is_success and success_count < args.num_success:
                            demo_id = success_count + fail_count
                            save_demo(trajectory, save_dir, demo_id, success=True)
                            success_count += 1
                            print(f"\n✓ 自动标记为成功")
                        elif not is_success and fail_count < args.num_fail:
                            demo_id = success_count + fail_count
                            save_demo(trajectory, save_dir, demo_id, success=False)
                            fail_count += 1
                            print(f"\n✗ 自动标记为失败")

                        recording = False
                        trajectory = []

                except Exception as e:
                    print(f"\n✗ 错误: {e}")
                    recording = False
                    trajectory = []

            # 控制循环频率
            time.sleep(config.control.dt)

    # 清理
    vp_bridge.stop()
    env.close()

    print("\n" + "=" * 60)
    print(f"✓ 完成！共收集:")
    print(f"  成功演示: {success_count}")
    print(f"  失败演示: {fail_count}")
    print(f"  总计: {success_count + fail_count}")
    print(f"✓ 保存位置: {save_dir}")
    print("=" * 60 + "\n")


def save_demo(trajectory, save_dir, demo_id, success=True):
    """保存演示数据"""
    demo_path = save_dir / f"demo_{demo_id:03d}.pkl"

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

    label_str = "成功" if success else "失败"
    print(f"    ✓ 已保存 ({label_str}): {demo_path.name}")


if __name__ == '__main__':
    main()
