#!/usr/bin/env python3
"""
使用VisionPro遥操作采集演示数据

用法:
    python record_teleop_demos.py --config task_config.yaml --num-demos 20
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

from kinova_rl_env import KinovaEnv
from vision_pro_control.core import VisionProBridge


class TeleopDataCollector:
    """遥操作数据采集器"""

    def __init__(
        self,
        env,
        vp_bridge,
        output_dir,
        tcp_position_scale=1.0,
        gripper_threshold=0.5
    ):
        self.env = env
        self.vp_bridge = vp_bridge
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tcp_position_scale = tcp_position_scale
        self.gripper_threshold = gripper_threshold

        # 演示计数
        self.demo_count = 0

    def collect_demo(self, demo_id=None):
        """采集一条演示"""

        if demo_id is None:
            demo_id = self.demo_count

        print(f"\n{'=' * 60}")
        print(f"开始采集演示 #{demo_id}")
        print(f"{'=' * 60}")

        # 重置环境
        obs, info = self.env.reset()
        print(f"环境已重置")
        print(f"  任务: {info.get('phase', 'N/A')}")
        print(f"  目标: {info.get('target_pose', info.get('socket_position', 'N/A'))}")

        # 数据缓冲
        demo_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "terminals": [],
            "infos": [],
        }

        print("\n开始遥操作（按 Ctrl+C 停止）...")
        input("按 Enter 开始...")

        episode_reward = 0.0
        step = 0

        try:
            while True:
                # 获取VisionPro数据
                vp_data = self.vp_bridge.get_latest_data()

                if vp_data['timestamp'] == 0:
                    print("等待VisionPro数据...")
                    continue

                # 转换为机器人动作
                action = self._vp_to_action(vp_data)

                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # 保存数据
                demo_data["observations"].append(obs)
                demo_data["actions"].append(action)
                demo_data["rewards"].append(reward)
                demo_data["next_observations"].append(next_obs)
                demo_data["terminals"].append(terminated or truncated)
                demo_data["infos"].append(info)

                episode_reward += reward
                step += 1

                # 打印进度
                if step % 10 == 0:
                    print(f"  Step {step}: reward={reward:.3f}, total={episode_reward:.3f}, "
                          f"phase={info.get('phase', 'N/A')}")

                # 检查终止条件
                if terminated or truncated:
                    success = info.get("success", False)
                    print(f"\n{'✓' if success else '✗'} Episode 结束: "
                          f"{'成功' if success else '未完成'}")
                    break

                obs = next_obs

        except KeyboardInterrupt:
            print("\n用户中断")

        # 保存演示
        if len(demo_data["observations"]) > 0:
            save_path = self.output_dir / f"demo_{demo_id}.pkl"

            # 转换为numpy数组
            for key in ["observations", "actions", "rewards", "terminals"]:
                if key in ["observations", "next_observations"]:
                    # 观测是字典，需要特殊处理
                    continue
                demo_data[key] = np.array(demo_data[key])

            with open(save_path, 'wb') as f:
                pickle.dump(demo_data, f)

            print(f"\n✓ 演示已保存: {save_path}")
            print(f"  步数: {len(demo_data['observations'])}")
            print(f"  总奖励: {episode_reward:.3f}")
            print(f"  成功: {demo_data['infos'][-1].get('success', False)}")

            self.demo_count += 1
            return True
        else:
            print("\n✗ 演示为空，未保存")
            return False

    def _vp_to_action(self, vp_data):
        """
        将VisionPro数据转换为机器人动作

        Args:
            vp_data: VisionPro数据字典

        Returns:
            action: (7,) 关节速度 或 (8,) 关节速度+夹爪
        """
        # 获取手腕位置增量（相对于上一帧）
        wrist_pose = vp_data['wrist_pose']
        wrist_pos = wrist_pose[:3, 3]

        # 简化：直接使用位置作为TCP目标增量
        # 在实际应用中，你可能需要更复杂的映射
        tcp_delta = wrist_pos * self.tcp_position_scale

        # 夹爪控制（基于捏合距离）
        pinch = vp_data['pinch_distance']
        gripper_action = 1.0 if pinch < self.gripper_threshold else 0.0

        # TODO: 将TCP增量转换为关节速度
        # 这里需要逆运动学(IK)或雅可比矩阵
        # 简化版本：假设前3个维度是TCP位置控制
        action = np.zeros(7)
        action[:3] = tcp_delta[:3]  # x, y, z 增量

        # 如果环境支持夹爪
        if self.env.action_space.shape[0] == 8:
            action = np.append(action, gripper_action)

        return action


def main():
    # 从配置文件读取默认 VisionPro IP
    default_vp_ip = '192.168.1.125'
    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml('kinova_rl_env/config/kinova_config.yaml')
        if hasattr(config, 'visionpro') and hasattr(config.visionpro, 'ip'):
            default_vp_ip = config.visionpro.ip
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='采集VisionPro遥操作演示数据')

    parser.add_argument('--config', type=str, required=True,
                        help='任务配置文件路径')
    parser.add_argument('--num-demos', type=int, default=10,
                        help='采集演示数量')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认从配置读取）')

    parser.add_argument('--vp-ip', type=str, default=default_vp_ip,
                        help=f'VisionPro IP地址 (默认: {default_vp_ip})')
    parser.add_argument('--tcp-scale', type=float, default=1.0,
                        help='TCP位置缩放因子')
    parser.add_argument('--gripper-threshold', type=float, default=0.5,
                        help='夹爪触发阈值')

    args = parser.parse_args()

    print("=" * 70)
    print("VisionPro 遥操作演示采集")
    print("=" * 70)

    # 创建环境
    print(f"\n加载配置: {args.config}")
    env = KinovaEnv(config_path=args.config)

    # 确定输出目录
    if args.output_dir is None:
        # 从配置读取
        output_dir = env.config.get("data_collection", {}).get("demos_dir", "./demos")
    else:
        output_dir = args.output_dir

    print(f"输出目录: {output_dir}")

    # 连接VisionPro
    print(f"\n连接VisionPro ({args.vp_ip})...")
    vp_bridge = VisionProBridge(avp_ip=args.vp_ip)
    vp_bridge.start()
    print("✓ VisionPro已连接")

    # 创建采集器
    collector = TeleopDataCollector(
        env=env,
        vp_bridge=vp_bridge,
        output_dir=output_dir,
        tcp_position_scale=args.tcp_scale,
        gripper_threshold=args.gripper_threshold
    )

    # 采集演示
    print(f"\n准备采集 {args.num_demos} 条演示")
    print("提示：")
    print("  - 每条演示都会重置环境")
    print("  - 按 Enter 开始采集")
    print("  - 按 Ctrl+C 停止当前演示")
    print("  - 完成任务后环境会自动终止\n")

    successful_demos = 0

    for i in range(args.num_demos):
        success = collector.collect_demo(demo_id=i)

        if success:
            successful_demos += 1

        if i < args.num_demos - 1:
            print(f"\n进度: {i+1}/{args.num_demos} 条采集完成")
            cont = input("继续采集下一条？[Y/n] ").strip().lower()
            if cont == 'n':
                break

    # 清理
    vp_bridge.stop()
    env.close()

    print("\n" + "=" * 70)
    print(f"采集完成！")
    print(f"  成功: {successful_demos}/{args.num_demos}")
    print(f"  保存到: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
