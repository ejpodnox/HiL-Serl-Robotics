#!/usr/bin/env python3
"""
Reward Classifier 训练脚本

用途：
- 从演示数据中学习成功/失败的判别器
- 替代手工设计的 reward 函数
- 支持基于图像的成功判定

使用方法:
    # 1. 收集带标签的数据
    python record_success_fail_demos.py --save_dir ./demos/labeled

    # 2. 训练分类器
    python train_reward_classifier.py \
        --demos_dir ./demos/labeled \
        --checkpoint_dir ./checkpoints/classifier

    # 3. 评估分类器
    python train_reward_classifier.py \
        --demos_dir ./demos/labeled \
        --checkpoint ./checkpoints/classifier/best_model.pt \
        --evaluate
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ============ 数据加载器 ============

class LabeledDemoDataset(Dataset):
    """
    带标签的演示数据集

    每个样本包含观测和成功/失败标签
    """

    def __init__(self, demos_dir: Path, num_demos: int = None, image_key='wrist_1'):
        """
        Args:
            demos_dir: 演示数据目录
            num_demos: 使用的演示数量
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
        self.samples = []
        success_count = 0
        fail_count = 0

        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                demo = pickle.load(f)

            # 提取成功标签
            success = demo.get('success', False)

            if success:
                success_count += 1
            else:
                fail_count += 1

            # 提取每个时间步的观测
            for obs in demo['observations']:
                self.samples.append({
                    'observation': obs,
                    'label': 1.0 if success else 0.0  # 1=成功, 0=失败
                })

        print(f"✓ 总计 {len(self.samples)} 个样本")
        print(f"  成功演示: {success_count}, 失败演示: {fail_count}")
        print(f"  正样本: {sum(s['label'] == 1.0 for s in self.samples)}")
        print(f"  负样本: {sum(s['label'] == 0.0 for s in self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本

        Returns:
            state: (state_dim,) 状态向量
            image: (C, H, W) 图像
            label: float, 0 或 1
        """
        sample = self.samples[idx]
        obs = sample['observation']
        label = sample['label']

        # 提取 state
        state_dict = obs['state']
        state_parts = []

        for key in ['tcp_pose', 'tcp_vel', 'gripper_pose']:
            if key in state_dict:
                state_parts.append(np.array(state_dict[key], dtype=np.float32))

        state = np.concatenate(state_parts) if state_parts else np.array([])

        # 提取 image
        if 'images' in obs and self.image_key in obs['images']:
            image = obs['images'][self.image_key]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image = torch.zeros(3, 128, 128, dtype=torch.float32)

        # 转换为 Tensor
        state = torch.from_numpy(state).float()
        label = torch.tensor(label, dtype=torch.float32)

        return state, image, label


# ============ Reward Classifier 网络 ============

class RewardClassifier(nn.Module):
    """
    Reward 分类器（状态 + 图像 → 成功概率）

    架构与 BCPolicy 类似，但输出是二分类概率
    """

    def __init__(
        self,
        state_dim: int,
        image_size: Tuple[int, int] = (128, 128),
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim

        # 图像编码器
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

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头
        fusion_input_dim = image_feat_dim + 128

        layers = []
        input_dim = fusion_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # 输出层：二分类
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 [0, 1] 概率

        self.classifier = nn.Sequential(*layers)

    def forward(self, state, image):
        """
        前向传播

        Args:
            state: (B, state_dim)
            image: (B, C, H, W)
        Returns:
            prob: (B, 1) 成功概率
        """
        # 编码
        image_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)

        # 融合
        fused_feat = torch.cat([image_feat, state_feat], dim=1)

        # 分类
        prob = self.classifier(fused_feat)

        return prob


# ============ 训练器 ============

class ClassifierTrainer:
    """Reward Classifier 训练器"""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")

        # 加载数据
        self.train_loader, self.val_loader = self._load_data()

        # 创建分类器
        self.classifier = self._create_classifier()
        self.classifier.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=config.classifier_config.learning_rate
        )

        # 损失函数（二分类交叉熵）
        self.criterion = nn.BCELoss()

        # 日志
        self.writer = None
        if TENSORBOARD_AVAILABLE and config.logging.get('use_tensorboard', False):
            log_dir = Path(config.logging.log_dir) / 'classifier'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        print("✓ Classifier 训练器初始化完成")

    def _load_data(self):
        """加载数据"""
        full_dataset = LabeledDemoDataset(
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.classifier_config.batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.classifier_config.batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        print(f"✓ 训练集: {train_size} 样本，验证集: {val_size} 样本")

        return train_loader, val_loader

    def _create_classifier(self):
        """创建分类器"""
        classifier = RewardClassifier(
            state_dim=self.config.obs_config.state_dim,
            image_size=tuple(self.config.obs_config.image_size),
            hidden_dims=self.config.classifier_config.hidden_dims,
            activation='relu',
            dropout=0.1
        )

        total_params = sum(p.numel() for p in classifier.parameters())
        print(f"✓ 分类器参数量: {total_params:,}")

        return classifier

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.classifier.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (state, image, label) in enumerate(self.train_loader):
            state = state.to(self.device)
            image = image.to(self.device)
            label = label.to(self.device).unsqueeze(1)  # (B, 1)

            # 前向传播
            pred_prob = self.classifier(state, image)

            # 计算损失
            loss = self.criterion(pred_prob, label)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            all_preds.extend((pred_prob > 0.5).cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

            # 记录
            if self.writer and batch_idx % 10 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self):
        """验证"""
        self.classifier.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for state, image, label in self.val_loader:
                state = state.to(self.device)
                image = image.to(self.device)
                label = label.to(self.device).unsqueeze(1)

                pred_prob = self.classifier(state, image)
                loss = self.criterion(pred_prob, label)

                total_loss += loss.item()
                all_probs.extend(pred_prob.cpu().numpy().flatten())
                all_preds.extend((pred_prob > 0.5).cpu().numpy().flatten())
                all_labels.extend(label.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

    def train(self, epochs: int):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练 Reward Classifier")
        print("=" * 60)

        best_f1 = 0.0

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_metrics = self.validate()

            # 打印
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.3f}, "
                  f"F1: {val_metrics['f1']:.3f}")

            # 记录
            if self.writer:
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)

            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ 新的最佳 F1: {best_f1:.3f}")

        print("=" * 60)
        print(f"✓ 训练完成！最佳 F1: {best_f1:.3f}")
        print("=" * 60)

        # 最终验证
        final_metrics = self.validate()
        print("\n最终验证结果:")
        print(f"  准确率: {final_metrics['accuracy']:.3f}")
        print(f"  精确率: {final_metrics['precision']:.3f}")
        print(f"  召回率: {final_metrics['recall']:.3f}")
        print(f"  F1 分数: {final_metrics['f1']:.3f}")
        print(f"\n混淆矩阵:")
        print(final_metrics['confusion_matrix'])

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict()
        }

        if is_best:
            path = checkpoint_dir / 'best_classifier.pt'
        else:
            path = checkpoint_dir / f'classifier_epoch_{epoch}.pt'

        torch.save(checkpoint, path)


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description='Reward Classifier 训练')
    parser.add_argument('--config', type=str,
                        default='hil_serl_kinova/experiments/kinova_reaching/config.py',
                        help='配置文件路径')
    parser.add_argument('--demos_dir', type=str, default='./demos/labeled',
                        help='带标签的演示数据目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/classifier',
                        help='检查点保存目录')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--evaluate', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='要评估的检查点')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config = config_module.get_config()

    # 覆盖参数
    config.data_config.demos_dir = args.demos_dir
    config.logging.checkpoint_dir = args.checkpoint_dir

    if args.epochs is not None:
        config.classifier_config.epochs = args.epochs

    # 创建训练器
    trainer = ClassifierTrainer(config, device=args.device)

    if args.evaluate:
        # 评估模式
        if args.checkpoint is None:
            print("评估模式需要指定 --checkpoint")
            return

        # 加载检查点
        checkpoint = torch.load(args.checkpoint)
        trainer.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        print("\n评估模式...")
        metrics = trainer.validate()

        print("\n评估结果:")
        print(f"  准确率: {metrics['accuracy']:.3f}")
        print(f"  精确率: {metrics['precision']:.3f}")
        print(f"  召回率: {metrics['recall']:.3f}")
        print(f"  F1 分数: {metrics['f1']:.3f}")
        print(f"\n混淆矩阵:")
        print(metrics['confusion_matrix'])
    else:
        # 训练模式
        trainer.train(epochs=config.classifier_config.epochs)


if __name__ == '__main__':
    main()
