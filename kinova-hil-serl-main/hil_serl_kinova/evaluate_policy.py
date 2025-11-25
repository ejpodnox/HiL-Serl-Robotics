#!/usr/bin/env python3
"""
评估训练好的策略

用法:
    # 离线评估（模拟）
    python evaluate_policy.py --config task_config.yaml --checkpoint model.pt --mode offline

    # 在线评估（真实机器人）
    python evaluate_policy.py --config task_config.yaml --checkpoint model.pt --mode online
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time

from kinova_rl_env import KinovaEnv


class PolicyEvaluator:
    """策略评估器"""

    def __init__(self, env, policy, device='cuda'):
        self.env = env
        self.policy = policy
        self.device = device

        # 将策略移到设备
        self.policy.to(device)
        self.policy.eval()

    def evaluate_episode(self, visualize=False):
        """评估单个episode"""

        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        success = False

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
        }

        while True:
            # 提取策略输入
            policy_input = self._obs_to_policy_input(obs)

            # 获取动作
            with torch.no_grad():
                # 根据策略类型调用
                if hasattr(self.policy, 'deterministic_action'):
                    # SAC Actor
                    action = self.policy.deterministic_action(
                        policy_input['state'],
                        policy_input.get('images')
                    )
                else:
                    # BC Policy
                    action = self.policy(
                        policy_input['state'],
                        policy_input.get('images')
                    )

                action = action.cpu().numpy().flatten()

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # 记录
            trajectory["observations"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)

            episode_reward += reward
            episode_steps += 1

            # 可视化
            if visualize:
                self._visualize(obs, action, reward, info)

            # 检查终止
            if terminated or truncated:
                success = info.get("success", False)
                break

            obs = next_obs

        return {
            "reward": episode_reward,
            "steps": episode_steps,
            "success": success,
            "trajectory": trajectory,
        }

    def evaluate(self, num_episodes=10, visualize=False):
        """评估多个episodes"""

        print(f"\n开始评估 {num_episodes} 个episodes...")
        print("=" * 70)

        results = []

        for i in range(num_episodes):
            print(f"\nEpisode {i+1}/{num_episodes}")
            result = self.evaluate_episode(visualize=visualize)

            print(f"  Reward: {result['reward']:.2f}")
            print(f"  Steps: {result['steps']}")
            print(f"  Success: {'✓' if result['success'] else '✗'}")

            results.append(result)

        # 统计
        rewards = [r["reward"] for r in results]
        steps = [r["steps"] for r in results]
        successes = [r["success"] for r in results]

        print("\n" + "=" * 70)
        print("【评估结果】")
        print("=" * 70)

        print(f"\n成功率: {sum(successes)}/{num_episodes} ({100*sum(successes)/num_episodes:.1f}%)")

        print(f"\n奖励统计:")
        print(f"  平均: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  最小: {np.min(rewards):.2f}")
        print(f"  最大: {np.max(rewards):.2f}")

        print(f"\n步数统计:")
        print(f"  平均: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"  最小: {np.min(steps)}")
        print(f"  最大: {np.max(steps)}")

        print("=" * 70)

        return results

    def _obs_to_policy_input(self, obs):
        """将观测转换为策略输入"""

        # 假设策略输入是state + images
        # 根据你的实际策略架构调整

        state = obs["state"]
        images = obs["images"]

        # 拼接state
        state_list = []
        for key, val in state.items():
            if isinstance(val, np.ndarray):
                state_list.append(val.flatten())

        state_vec = np.concatenate(state_list)

        # 处理图像
        image_list = []
        for key, img in images.items():
            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            image_list.append(img)

        image_array = np.stack(image_list)

        # 转换为tensor
        state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).to(self.device)

        return {"state": state_tensor, "images": image_tensor}

    def _visualize(self, obs, action, reward, info):
        """可视化（可选）"""

        # TODO: 使用OpenCV显示图像和状态
        # cv2.imshow("Camera", obs["images"]["wrist_1"])
        # cv2.waitKey(1)
        pass


def load_policy(checkpoint_path, state_dim=None, action_dim=None, device='cuda'):
    """加载训练好的策略"""

    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
        state_dim = config.get('state_dim', state_dim)
        action_dim = config.get('action_dim', action_dim)

    # 确定模型类型（BC 或 RLPD）
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # 根据state_dict键名推断
        if 'policy_state_dict' in checkpoint:
            model_type = 'bc'
        elif 'actor_state_dict' in checkpoint:
            model_type = 'sac'
        else:
            model_type = 'bc'  # 默认

    # 创建策略网络
    from .networks import BCPolicy, SACActor

    if model_type == 'bc' or model_type == 'BC':
        policy = BCPolicy(
            state_dim=state_dim or 14,
            action_dim=action_dim or 7,
            image_size=(128, 128),
            use_image=True
        )

        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            policy.load_state_dict(checkpoint)

    elif model_type == 'sac' or model_type == 'SAC':
        policy = SACActor(
            state_dim=state_dim or 14,
            action_dim=action_dim or 7,
            image_size=(128, 128),
            use_image=True
        )

        if 'actor_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['actor_state_dict'])
        else:
            policy.load_state_dict(checkpoint)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    policy.to(device)
    policy.eval()

    print(f"✓ 加载 {model_type.upper()} 模型成功")
    return policy


def main():
    parser = argparse.ArgumentParser(description='评估训练好的策略')

    parser.add_argument('--config', type=str, required=True,
                        help='任务配置文件')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')

    parser.add_argument('--mode', type=str, default='offline',
                        choices=['offline', 'online'],
                        help='评估模式: offline(模拟) 或 online(真实机器人)')

    parser.add_argument('--num-episodes', type=int, default=10,
                        help='评估episode数量')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化评估过程')

    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')

    args = parser.parse_args()

    print("=" * 70)
    print("策略评估")
    print("=" * 70)

    # 创建环境
    print(f"\n加载环境配置: {args.config}")
    env = KinovaEnv(config_path=args.config)

    # 加载策略
    policy = load_policy(args.checkpoint, device=args.device)

    if policy is None:
        print("\n✗ 策略加载失败")
        print("提示: 需要在 evaluate_policy.py 中实现 load_policy() 函数")
        return

    # 创建评估器
    evaluator = PolicyEvaluator(env, policy, device=args.device)

    # 评估
    if args.mode == 'online':
        print("\n⚠️  在线评估模式")
        print("警告: 将在真实机器人上运行策略")
        confirm = input("确认继续? [y/N] ").strip().lower()
        if confirm != 'y':
            print("取消评估")
            return

    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        visualize=args.visualize
    )

    # 清理
    env.close()

    print("\n评估完成")


if __name__ == '__main__':
    main()
