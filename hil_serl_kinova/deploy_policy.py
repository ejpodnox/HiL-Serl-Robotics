#!/usr/bin/env python3
"""
策略部署脚本 for Kinova（ROS2 + SpaceMouse 或纯策略）

支持多种部署模式：
1. 纯策略控制（policy_only）
2. 混合控制（hybrid: SpaceMouse + Policy）
3. 纯遥操作（teleop_only: SpaceMouse）
4. 在线评估（evaluation）

使用方法:
    # 纯策略控制
    python deploy_policy.py --checkpoint checkpoints/bc_kinova/best_model.pt \
                            --mode policy_only

    # 混合控制（SpaceMouse）
    python deploy_policy.py --checkpoint checkpoints/bc_kinova/best_model.pt \
                            --mode hybrid --alpha 0.5

    # 纯遥操作（SpaceMouse）
    python deploy_policy.py --checkpoint checkpoints/bc_kinova/best_model.pt \
                            --mode teleop_only

    # 在线评估
    python deploy_policy.py --checkpoint checkpoints/bc_kinova/best_model.pt \
                            --mode evaluation --num_episodes 10
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
import rclpy

from kinova_rl_env.kinova_env.kinova_env import KinovaEnv
from kinova_rl_env.kinova_env.config_loader import KinovaConfig

# 导入训练脚本中的策略网络
from hil_serl_kinova.train_bc_kinova import BCPolicy


class PolicyDeployer:
    """策略部署器"""

    def __init__(
        self,
        checkpoint_path: str,
        env_config_path: str,
        mode: str = 'policy_only',
        device: str = 'cuda',
        spacemouse_translation_gain: float = 0.6,
        spacemouse_rotation_gain: float = 0.6,
        spacemouse_deadband: float = 0.02,
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            env_config_path: 环境配置路径
            mode: 部署模式
            device: 设备
        """
        self.mode = mode
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"部署模式: {mode}")
        print(f"使用设备: {self.device}")
        print("遥操作来源: spacemouse")

        # 加载策略
        self.policy = self._load_policy(checkpoint_path)

        # 初始化环境
        config = KinovaConfig.from_yaml(env_config_path)
        self.env = KinovaEnv(config=config)

        # 遥操作设备（仅 SpaceMouse）
        self.spacemouse = None
        self.sm_gripper = 0.0
        self.sm_translation_gain = spacemouse_translation_gain
        self.sm_rotation_gain = spacemouse_rotation_gain
        self.sm_deadband = spacemouse_deadband

        if mode in ('hybrid', 'teleop_only'):
            self._init_spacemouse()

        print("✓ 策略部署器初始化完成")

    def _load_policy(self, checkpoint_path):
        """加载策略网络"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 从配置中获取网络参数
        config_dict = checkpoint['config']

        policy = BCPolicy(
            state_dim=config_dict['obs_config']['state_dim'],
            action_dim=config_dict['action_config']['dim'],
            image_size=tuple(config_dict['obs_config']['image_size']),
            hidden_dims=config_dict['bc_config']['policy_hidden_dims'],
            activation=config_dict['bc_config']['activation'],
            dropout=config_dict['bc_config']['dropout']
        )

        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.to(self.device)
        policy.eval()

        print(f"✓ 策略已加载: {checkpoint_path}")
        print(f"  训练轮数: {checkpoint['epoch']}")

        return policy

    def _init_spacemouse(self):
        """初始化 SpaceMouse 遥操作"""
        try:
            from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
        except ImportError as exc:
            raise ImportError("SpaceMouse 依赖未安装，请运行: pip install pyspacemouse hidapi") from exc

        self.spacemouse = SpaceMouseExpert()
        self.sm_gripper = 0.0
        print("✓ SpaceMouse 已初始化")

    def _obs_to_tensor(self, obs):
        """
        将环境观测转换为策略输入

        Args:
            obs: 环境观测字典
        Returns:
            state: (1, state_dim) tensor
            image: (1, C, H, W) tensor
        """
        # 提取 state
        state_dict = obs['state']
        state_parts = []

        for key in ['tcp_pose', 'tcp_vel', 'gripper_pose']:
            if key in state_dict:
                state_parts.append(np.array(state_dict[key], dtype=np.float32))

        state_np = np.concatenate(state_parts)

        # 提取 image
        if 'images' in obs and 'wrist_1' in obs['images']:
            image_np = obs['images']['wrist_1']  # (H, W, 3)
            # (H, W, 3) -> (C, H, W)，归一化
            image_np = np.transpose(image_np, (2, 0, 1)).astype(np.float32) / 255.0
        else:
            image_np = np.zeros((3, 128, 128), dtype=np.float32)

        # 转换为 tensor，添加 batch 维度
        state = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        image = torch.from_numpy(image_np).unsqueeze(0).to(self.device)

        return state, image

    def _get_policy_action(self, obs):
        """
        获取策略动作

        Args:
            obs: 环境观测
        Returns:
            action: numpy array
        """
        state, image = self._obs_to_tensor(obs)

        with torch.no_grad():
            action_tensor = self.policy(state, image)

        action = action_tensor.cpu().numpy()[0]  # 去除 batch 维度

        return action

    def _get_spacemouse_action(self):
        """
        获取 SpaceMouse 动作

        Returns:
            action: numpy array (7,)
        """
        if self.spacemouse is None:
            raise RuntimeError("SpaceMouse 未初始化")

        raw_action, buttons = self.spacemouse.get_action()
        action_vec = np.array(raw_action, dtype=np.float32)
        action_vec[np.abs(action_vec) < self.sm_deadband] = 0.0

        linear = action_vec[:3] * self.sm_translation_gain
        angular = action_vec[3:6] * self.sm_rotation_gain
        delta = np.concatenate([linear, angular])
        delta = np.clip(delta, -1.0, 1.0)

        # 按住左键闭合，右键打开；否则保持上一次值
        if buttons:
            if buttons[0]:
                self.sm_gripper = 1.0
            elif len(buttons) > 1 and buttons[1]:
                self.sm_gripper = 0.0

        action = np.concatenate([delta, np.array([self.sm_gripper], dtype=np.float32)])
        return action

    def run_episode(self, max_steps=200, alpha=0.5):
        """
        运行一个 episode

        Args:
            max_steps: 最大步数
            alpha: 混合系数（alpha=1: 纯人工输入，alpha=0: 纯策略）
        Returns:
            episode_info: dict
        """
        obs, info = self.env.reset()

        episode_reward = 0.0
        episode_steps = 0
        success = False

        print("\n>>> Episode 开始 <<<")

        for step in range(max_steps):
            # 获取动作
            if self.mode == 'policy_only':
                action = self._get_policy_action(obs)
            elif self.mode == 'hybrid':
                policy_action = self._get_policy_action(obs)
                teleop_action = self._get_spacemouse_action()
                action = alpha * teleop_action + (1 - alpha) * policy_action
            elif self.mode == 'teleop_only':
                action = self._get_spacemouse_action()
            else:
                raise ValueError(f"未知模式: {self.mode}")

            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1

            # 打印状态
            if step % 10 == 0:
                distance = info.get('distance_to_target', -1)
                print(f"  步数: {step:3d} | 奖励: {episode_reward:.2f} | 距离: {distance:.3f}m")

            # 检查终止
            if terminated or truncated:
                success = info.get('success', False)
                break

        print(f">>> Episode 结束 | 步数: {episode_steps} | 奖励: {episode_reward:.2f} | "
              f"成功: {'✓' if success else '✗'} <<<\n")

        return {
            'steps': episode_steps,
            'reward': episode_reward,
            'success': success
        }

    def evaluate(self, num_episodes=10, max_steps=200):
        """
        评估策略

        Args:
            num_episodes: 评估回合数
            max_steps: 每回合最大步数
        Returns:
            results: dict
        """
        print("=" * 60)
        print(f"开始评估 ({num_episodes} 回合)")
        print("=" * 60)

        results = []

        for episode_idx in range(num_episodes):
            print(f"\n【Episode {episode_idx + 1}/{num_episodes}】")
            episode_info = self.run_episode(max_steps=max_steps)
            results.append(episode_info)

        # 统计
        success_rate = sum(r['success'] for r in results) / num_episodes
        avg_reward = np.mean([r['reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])

        print("=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"成功率: {success_rate * 100:.1f}%")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print("=" * 60)

        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'episodes': results
        }

    def interactive_run(self):
        """
        交互式运行

        按键控制：
        - Enter: 开始新回合
        - q: 退出
        """
        print("\n" + "=" * 60)
        print("交互式运行模式")
        print("=" * 60)
        print("按键说明:")
        print("  Enter - 开始新回合")
        print("  q     - 退出")
        print("=" * 60)

        while True:
            print("\n按 Enter 开始，或按 'q' 退出...")
            key = input().strip()

            if key == 'q':
                print("退出")
                break

            self.run_episode()

    def cleanup(self):
        """清理资源"""
        if self.vp_bridge is not None:
            self.vp_bridge.stop()

        if self.spacemouse is not None:
            try:
                self.spacemouse.close()
            except Exception:
                pass

        self.env.close()

        print("✓ 资源已清理")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='Kinova 策略部署')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--env_config', type=str,
                        default='kinova_rl_env/config/kinova_config.yaml',
                        help='环境配置路径')
    parser.add_argument('--mode', type=str, default='policy_only',
                        choices=['policy_only', 'hybrid', 'evaluation', 'teleop_only'],
                        help='部署模式')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='混合系数（hybrid 模式）')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='评估回合数（evaluation 模式）')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='每回合最大步数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（cuda/cpu）')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式运行')
    parser.add_argument('--sm_translation_gain', type=float, default=0.6,
                        help='SpaceMouse 平移增益')
    parser.add_argument('--sm_rotation_gain', type=float, default=0.6,
                        help='SpaceMouse 旋转增益')
    parser.add_argument('--sm_deadband', type=float, default=0.02,
                        help='SpaceMouse 死区')

    args = parser.parse_args()

    # 创建部署器
    deployer = PolicyDeployer(
        checkpoint_path=args.checkpoint,
        env_config_path=args.env_config,
        mode=args.mode,
        device=args.device,
        spacemouse_translation_gain=args.sm_translation_gain,
        spacemouse_rotation_gain=args.sm_rotation_gain,
        spacemouse_deadband=args.sm_deadband,
    )

    try:
        if args.interactive:
            # 交互式运行
            deployer.interactive_run()
        elif args.mode == 'evaluation':
            # 评估模式
            deployer.evaluate(
                num_episodes=args.num_episodes,
                max_steps=args.max_steps
            )
        else:
            # 单次运行
            deployer.run_episode(
                max_steps=args.max_steps,
                alpha=args.alpha
            )

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        deployer.cleanup()


if __name__ == '__main__':
    main()
