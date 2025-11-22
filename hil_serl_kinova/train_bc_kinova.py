#!/usr/bin/env python3
"""
Behavior Cloning (BC) 训练脚本 for Kinova

模块化设计，支持配置驱动：
- 数据加载器（可插拔）
- 策略网络（可自定义）
- 训练循环（标准化）

使用方法:
    python train_bc_kinova.py --config experiments/kinova_reaching/config.py \
                              --demos_dir ./demos/reaching \
                              --checkpoint_dir ./checkpoints/bc_kinova
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Tensorboard（可选）
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  Tensorboard 不可用，将跳过日志记录")


# ============ 数据加载器 ============

class KinovaDemoDataset(Dataset):
    """
    Kinova 演示数据集

    支持加载 HIL-SERL 格式的 .pkl 文件
    """

    def __init__(self, demos_dir: Path, num_demos: int = None, image_key='wrist_1'):
        """
        Args:
            demos_dir: 演示数据目录
            num_demos: 使用的演示数量（None=全部）
            image_key: 图像键名
        """
        self.demos_dir = Path(demos_dir)
        self.image_key = image_key

        # 加载演示文件
        demo_files = sorted(self.demos_dir.glob("demo_*.pkl"))

        if num_demos is not None:
            demo_files = demo_files[:num_demos]

        if len(demo_files) == 0:
            raise ValueError(f"在 {demos_dir} 中未找到演示文件")

        print(f"加载 {len(demo_files)} 条演示...")

        # 解析数据
        self.observations = []
        self.actions = []

        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                demo = pickle.load(f)

            # 提取 observations 和 actions
            for i in range(len(demo['actions'])):
                obs = demo['observations'][i]
                action = demo['actions'][i]

                self.observations.append(obs)
                self.actions.append(action)

        print(f"✓ 总计 {len(self.actions)} 个样本")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            state: (state_dim,) 状态向量
            image: (C, H, W) 图像
            action: (action_dim,) 动作向量
        """
        obs = self.observations[idx]
        action = self.actions[idx]

        # 提取 state
        state_dict = obs['state']
        state_parts = []

        # 拼接所有 state 特征
        for key in ['tcp_pose', 'tcp_vel', 'gripper_pose']:
            if key in state_dict:
                state_parts.append(np.array(state_dict[key], dtype=np.float32))

        state = np.concatenate(state_parts) if state_parts else np.array([])

        # 提取 image
        if 'images' in obs and self.image_key in obs['images']:
            image = obs['images'][self.image_key]  # (H, W, 3)
            # 转换为 PyTorch 格式: (C, H, W)，归一化到 [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # 没有图像，创建零图像
            image = torch.zeros(3, 128, 128, dtype=torch.float32)

        # 转换为 Tensor
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(np.array(action, dtype=np.float32)).float()

        return state, image, action


# ============ 策略网络 ============

class BCPolicy(nn.Module):
    """
    BC 策略网络（状态 + 图像 → 动作）

    架构:
    - 图像编码器: CNN
    - 状态编码器: MLP
    - 融合层: 拼接 + MLP
    - 输出层: 动作预测
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_size: Tuple[int, int] = (128, 128),
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            image_size: 图像尺寸 (H, W)
            hidden_dims: 隐藏层维度
            activation: 激活函数
            dropout: Dropout 概率
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 图像编码器（简单 CNN）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算图像特征维度
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, *image_size)
            image_feat_dim = self.image_encoder(dummy_image).shape[1]

        # 状态编码器（MLP）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层（拼接 image_feat + state_feat）
        fusion_input_dim = image_feat_dim + 128

        # 动作预测头
        layers = []
        input_dim = fusion_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.action_head = nn.Sequential(*layers)

    def forward(self, state, image):
        """
        前向传播

        Args:
            state: (B, state_dim)
            image: (B, C, H, W)
        Returns:
            action: (B, action_dim)
        """
        # 编码图像
        image_feat = self.image_encoder(image)  # (B, image_feat_dim)

        # 编码状态
        state_feat = self.state_encoder(state)  # (B, 128)

        # 拼接特征
        fused_feat = torch.cat([image_feat, state_feat], dim=1)

        # 预测动作
        action = self.action_head(fused_feat)

        return action


# ============ 训练器 ============

class BCTrainer:
    """BC 训练器"""

    def __init__(self, config, device='cuda'):
        """
        Args:
            config: 配置字典
            device: 设备
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")

        # 加载数据
        self.train_loader, self.val_loader = self._load_data()

        # 创建策略网络
        self.policy = self._create_policy()
        self.policy.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.bc_config.learning_rate,
            weight_decay=config.bc_config.weight_decay
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 日志
        self.writer = None
        if TENSORBOARD_AVAILABLE and config.logging.get('use_tensorboard', False):
            log_dir = Path(config.logging.log_dir) / 'bc'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        print("✓ BC 训练器初始化完成")

    def _load_data(self):
        """加载训练和验证数据"""
        # 创建数据集
        full_dataset = KinovaDemoDataset(
            demos_dir=self.config.data_config.demos_dir,
            num_demos=self.config.data_config.demos_num
        )

        # 划分训练集和验证集
        train_ratio = self.config.data_config.train_ratio
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.bc_config.batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.bc_config.batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        print(f"✓ 训练集: {train_size} 样本，验证集: {val_size} 样本")

        return train_loader, val_loader

    def _create_policy(self):
        """创建策略网络"""
        policy = BCPolicy(
            state_dim=self.config.obs_config.state_dim,
            action_dim=self.config.action_config.dim,
            image_size=tuple(self.config.obs_config.image_size),
            hidden_dims=self.config.bc_config.policy_hidden_dims,
            activation=self.config.bc_config.activation,
            dropout=self.config.bc_config.dropout
        )

        total_params = sum(p.numel() for p in policy.parameters())
        print(f"✓ 策略网络参数量: {total_params:,}")

        return policy

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.policy.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (state, image, action) in enumerate(self.train_loader):
            # 移动到设备
            state = state.to(self.device)
            image = image.to(self.device)
            action = action.to(self.device)

            # 前向传播
            pred_action = self.policy(state, image)

            # 计算损失
            loss = self.criterion(pred_action, action)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.config.bc_config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.bc_config.clip_grad_norm
                )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            num_batches += 1

            # 记录
            if self.writer and batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """验证"""
        self.policy.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for state, image, action in self.val_loader:
                state = state.to(self.device)
                image = image.to(self.device)
                action = action.to(self.device)

                pred_action = self.policy(state, image)
                loss = self.criterion(pred_action, action)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, epochs: int):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练 BC 策略")
        print("=" * 60)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate()

            # 打印
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")

            # 记录
            if self.writer:
                self.writer.add_scalar('val/loss', val_loss, epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("=" * 60)
        print(f"✓ 训练完成！最佳验证损失: {best_val_loss:.6f}")
        print("=" * 60)

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict()
        }

        if is_best:
            path = checkpoint_dir / 'best_model.pt'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)
        print(f"  ✓ 检查点已保存: {path}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='Kinova BC 训练')
    parser.add_argument('--config', type=str,
                        default='hil_serl_kinova/experiments/kinova_reaching/config.py',
                        help='配置文件路径')
    parser.add_argument('--demos_dir', type=str, default='./demos/reaching',
                        help='演示数据目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/bc_kinova',
                        help='检查点保存目录')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（cuda/cpu）')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 动态导入配置
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.get_bc_config()  # 使用 BC 配置

    # 覆盖参数
    config.data_config.demos_dir = args.demos_dir
    config.logging.checkpoint_dir = args.checkpoint_dir

    if args.epochs is not None:
        config.bc_config.epochs = args.epochs

    # 验证配置
    config_module.validate_config(config)

    # 创建训练器
    trainer = BCTrainer(config, device=args.device)

    # 开始训练
    trainer.train(epochs=config.bc_config.epochs)


if __name__ == '__main__':
    main()
