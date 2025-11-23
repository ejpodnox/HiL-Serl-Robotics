#!/usr/bin/env python3
"""
RLPD (Reinforcement Learning with Prior Data) 训练脚本

HIL-SERL 核心训练循环：
1. 离线预训练（使用演示数据）
2. 在线学习（与环境交互）
3. 人类介入（可选）

使用方法:
    # 从 BC 策略开始
    python train_rlpd_kinova.py \
        --config hil_serl_kinova/experiments/kinova_reaching/config.py \
        --bc_checkpoint checkpoints/bc_kinova/best_model.pt \
        --demos_dir ./demos/reaching

    # 从头开始
    python train_rlpd_kinova.py \
        --config hil_serl_kinova/experiments/kinova_reaching/config.py \
        --demos_dir ./demos/reaching
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from collections import deque
import pickle
import time

from kinova_rl_env.kinova_env.kinova_env import KinovaEnv
from kinova_rl_env.kinova_env.config_loader import KinovaConfig

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ============ SAC 网络 ============

class Critic(nn.Module):
    """Q 网络（状态-动作价值函数）"""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 256]):
        super().__init__()

        # Q1
        layers1 = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers1.append(nn.Linear(input_dim, hidden_dim))
            layers1.append(nn.ReLU())
            input_dim = hidden_dim
        layers1.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers1)

        # Q2 (twin Q)
        layers2 = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers2.append(nn.Linear(input_dim, hidden_dim))
            layers2.append(nn.ReLU())
            input_dim = hidden_dim
        layers2.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        """
        Args:
            state: (B, state_dim)
            action: (B, action_dim)
        Returns:
            q1, q2: (B, 1)
        """
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


class Actor(nn.Module):
    """策略网络（随机策略）"""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 256], log_std_min=-20, log_std_max=2):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 主干网络
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 均值和标准差
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        """
        Args:
            state: (B, state_dim)
        Returns:
            mean: (B, action_dim)
            log_std: (B, action_dim)
        """
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        # 计算 log_prob（需要修正 tanh 变换）
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state):
        """确定性动作（用于评估）"""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


# ============ Replay Buffer ============

class ReplayBuffer:
    """经验回放缓冲区（混合离线和在线数据）"""

    def __init__(self, capacity=100000, offline_ratio=0.5):
        """
        Args:
            capacity: 最大容量
            offline_ratio: 离线数据采样比例
        """
        self.capacity = capacity
        self.offline_ratio = offline_ratio

        self.offline_buffer = []  # 离线数据（演示）
        self.online_buffer = deque(maxlen=capacity)  # 在线数据

    def add_offline_data(self, demos_dir):
        """从演示数据加载离线数据"""
        demo_files = sorted(Path(demos_dir).glob("demo_*.pkl"))

        print(f"加载 {len(demo_files)} 条演示数据...")

        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                demo = pickle.load(f)

            for i in range(len(demo['actions'])):
                transition = {
                    'state': self._extract_state(demo['observations'][i]),
                    'action': demo['actions'][i],
                    'reward': demo['rewards'][i],
                    'next_state': self._extract_state(demo['observations'][min(i+1, len(demo['observations'])-1)]),
                    'done': demo['terminals'][i]
                }
                self.offline_buffer.append(transition)

        print(f"✓ 离线数据: {len(self.offline_buffer)} 个 transitions")

    def add_online_transition(self, state, action, reward, next_state, done):
        """添加在线数据"""
        self.online_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

    def sample(self, batch_size):
        """采样一个 batch"""
        # 分别从离线和在线缓冲区采样
        offline_size = int(batch_size * self.offline_ratio)
        online_size = batch_size - offline_size

        batch = []

        # 离线数据
        if len(self.offline_buffer) > 0:
            offline_indices = np.random.choice(len(self.offline_buffer), offline_size, replace=True)
            batch.extend([self.offline_buffer[i] for i in offline_indices])

        # 在线数据
        if len(self.online_buffer) > 0:
            online_indices = np.random.choice(len(self.online_buffer), online_size, replace=True)
            batch.extend([self.online_buffer[i] for i in online_indices])

        # 如果数据不足，从现有数据中补齐
        while len(batch) < batch_size:
            if len(self.offline_buffer) > 0:
                batch.append(self.offline_buffer[np.random.randint(len(self.offline_buffer))])
            elif len(self.online_buffer) > 0:
                batch.append(self.online_buffer[np.random.randint(len(self.online_buffer))])

        # 转换为 tensors
        states = torch.FloatTensor([t['state'] for t in batch])
        actions = torch.FloatTensor([t['action'] for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([t['next_state'] for t in batch])
        dones = torch.FloatTensor([t['done'] for t in batch]).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def _extract_state(self, obs):
        """从观测中提取状态向量"""
        state_parts = []
        for key in ['tcp_pose', 'tcp_vel', 'gripper_pose']:
            if key in obs['state']:
                state_parts.append(np.array(obs['state'][key], dtype=np.float32))
        return np.concatenate(state_parts) if state_parts else np.array([])

    def __len__(self):
        return len(self.offline_buffer) + len(self.online_buffer)


# ============ SAC Agent ============

class SACAgent:
    """SAC (Soft Actor-Critic) Agent"""

    def __init__(self, state_dim, action_dim, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim

        # 网络
        self.actor = Actor(state_dim, action_dim, config.rlpd_config.actor_hidden_dims).to(self.device)
        self.critic = Critic(state_dim, action_dim, config.rlpd_config.critic_hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, config.rlpd_config.critic_hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.rlpd_config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.rlpd_config.critic_lr)

        # 温度参数（自动调整）
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.rlpd_config.temp_lr)
        self.target_entropy = -action_dim if config.rlpd_config.target_entropy is None else config.rlpd_config.target_entropy

        # 超参数
        self.gamma = config.rlpd_config.gamma
        self.tau = config.rlpd_config.tau

        print(f"✓ SAC Agent 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  目标熵: {self.target_entropy}")

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)

            return action.cpu().numpy()[0]

    def update(self, batch):
        """更新网络"""
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ===== 更新 Critic =====
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== 更新 Actor =====
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== 更新温度参数 =====
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ===== 软更新目标网络 =====
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'q_value': q1.mean().item()
        }

    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


# ============ RLPD 训练器 ============

class RLPDTrainer:
    """RLPD 训练器（离线 + 在线）"""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device

        # 创建环境
        env_config = KinovaConfig.from_yaml(config.env_config_path)
        self.env = KinovaEnv(config=env_config)

        # 创建 Agent
        state_dim = config.obs_config.state_dim
        action_dim = config.action_config.dim
        self.agent = SACAgent(state_dim, action_dim, config, device)

        # 创建 Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=100000, offline_ratio=0.5)

        # 加载离线数据
        self.replay_buffer.add_offline_data(config.data_config.demos_dir)

        # 日志
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = Path(config.logging.log_dir) / 'rlpd'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        print("✓ RLPD 训练器初始化完成")

    def train_offline(self, steps):
        """离线预训练"""
        print("\n" + "=" * 60)
        print(f"离线预训练 ({steps} 步)")
        print("=" * 60)

        for step in range(steps):
            batch = self.replay_buffer.sample(self.config.rlpd_config.batch_size)
            metrics = self.agent.update(batch)

            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{steps} | "
                      f"Critic: {metrics['critic_loss']:.4f}, "
                      f"Actor: {metrics['actor_loss']:.4f}, "
                      f"Alpha: {metrics['alpha']:.3f}, "
                      f"Q: {metrics['q_value']:.3f}")

                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'offline/{k}', v, step)

        print("✓ 离线预训练完成")

    def train_online(self, steps):
        """在线学习"""
        print("\n" + "=" * 60)
        print(f"在线学习 ({steps} 步)")
        print("=" * 60)

        obs, _ = self.env.reset()
        state = self._obs_to_state(obs)
        episode_reward = 0
        episode_steps = 0

        for step in range(steps):
            # 选择动作
            action = self.agent.select_action(state, deterministic=False)

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_state = self._obs_to_state(next_obs)
            done = terminated or truncated

            # 添加到 buffer
            self.replay_buffer.add_online_transition(state, action, reward, next_state, done)

            episode_reward += reward
            episode_steps += 1

            # 更新网络（UTD 比例）
            for _ in range(self.config.rlpd_config.utd_ratio):
                batch = self.replay_buffer.sample(self.config.rlpd_config.batch_size)
                metrics = self.agent.update(batch)

            # Episode 结束
            if done:
                success = info.get('success', False)
                print(f"  Episode | Steps: {episode_steps}, Reward: {episode_reward:.2f}, Success: {success}")

                if self.writer:
                    self.writer.add_scalar('online/episode_reward', episode_reward, step)
                    self.writer.add_scalar('online/episode_steps', episode_steps, step)

                obs, _ = self.env.reset()
                state = self._obs_to_state(obs)
                episode_reward = 0
                episode_steps = 0
            else:
                state = next_state

            # 定期打印和保存
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{steps} | Buffer: {len(self.replay_buffer)}")

                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'online/{k}', v, step)

            if (step + 1) % 1000 == 0:
                self.save_checkpoint(step)

        print("✓ 在线学习完成")

    def _obs_to_state(self, obs):
        """观测转状态"""
        state_parts = []
        for key in ['tcp_pose', 'tcp_vel', 'gripper_pose']:
            if key in obs['state']:
                state_parts.append(np.array(obs['state'][key], dtype=np.float32))
        return np.concatenate(state_parts) if state_parts else np.array([])

    def save_checkpoint(self, step):
        """保存检查点"""
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = checkpoint_dir / f'rlpd_step_{step}.pt'
        self.agent.save(path)
        print(f"  ✓ 检查点已保存: {path}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='RLPD 训练')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--demos_dir', type=str, required=True,
                        help='演示数据目录')
    parser.add_argument('--bc_checkpoint', type=str, default=None,
                        help='BC 预训练检查点（可选）')
    parser.add_argument('--offline_steps', type=int, default=None,
                        help='离线训练步数')
    parser.add_argument('--online_steps', type=int, default=None,
                        help='在线训练步数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.get_rlpd_config()

    # 覆盖参数
    config.data_config.demos_dir = args.demos_dir

    if args.offline_steps:
        config.rlpd_config.offline_steps = args.offline_steps
    if args.online_steps:
        config.rlpd_config.online_steps = args.online_steps

    # 创建训练器
    trainer = RLPDTrainer(config, device=args.device)

    # 训练
    trainer.train_offline(config.rlpd_config.offline_steps)
    trainer.train_online(config.rlpd_config.online_steps)

    print("\n" + "=" * 60)
    print("✓ RLPD 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
