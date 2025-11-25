"""
神经网络模型定义

提供完整的视觉-运动融合网络，用于 BC 和 RLPD 训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


# ============ 共享组件 ============

class ImageEncoder(nn.Module):
    """
    图像编码器 (CNN)

    将图像 (C, H, W) 编码为特征向量
    """

    def __init__(
        self,
        input_channels: int = 3,
        image_size: Tuple[int, int] = (128, 128),
        output_dim: int = 256,
        architecture: str = 'simple'
    ):
        super().__init__()

        if architecture == 'simple':
            # 简单 CNN（快速训练）
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
        elif architecture == 'resnet':
            # ResNet18 backbone（更强大但慢）
            import torchvision.models as models
            resnet = models.resnet18(pretrained=False)
            self.conv = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # 计算卷积输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *image_size)
            conv_out_dim = self.conv(dummy).shape[1]

        # 投影到目标维度
        self.projection = nn.Linear(conv_out_dim, output_dim)

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) 或 (B, N, C, H, W) 多相机
        Returns:
            features: (B, output_dim)
        """
        if image.dim() == 5:
            # 多相机：(B, N, C, H, W)
            B, N, C, H, W = image.shape
            image = image.view(B * N, C, H, W)
            features = self.conv(image)
            features = features.view(B, N, -1)
            features = features.mean(dim=1)  # 平均池化
        else:
            # 单相机：(B, C, H, W)
            features = self.conv(image)

        features = self.projection(features)
        return features


class StateEncoder(nn.Module):
    """
    状态编码器 (MLP)

    将机器人状态编码为特征向量
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int = 256,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, state):
        """
        Args:
            state: (B, state_dim)
        Returns:
            features: (B, output_dim)
        """
        return self.mlp(state)


# ============ BC 网络 ============

class BCPolicy(nn.Module):
    """
    Behavior Cloning 策略网络

    输入: 状态 + 图像
    输出: 动作
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_size: Tuple[int, int] = (128, 128),
        image_encoder_dim: int = 256,
        state_encoder_dim: int = 256,
        hidden_dims: List[int] = [512, 512, 256],
        dropout: float = 0.1,
        use_image: bool = True
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_image = use_image

        # 图像编码器
        if use_image:
            self.image_encoder = ImageEncoder(
                input_channels=3,
                image_size=image_size,
                output_dim=image_encoder_dim
            )

        # 状态编码器
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            output_dim=state_encoder_dim,
            dropout=dropout
        )

        # 融合层
        fusion_dim = image_encoder_dim + state_encoder_dim if use_image else state_encoder_dim

        # 动作预测头
        layers = []
        input_dim = fusion_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # 动作限制在 [-1, 1]

        self.action_head = nn.Sequential(*layers)

    def forward(self, state, image=None):
        """
        Args:
            state: (B, state_dim)
            image: (B, C, H, W) or None
        Returns:
            action: (B, action_dim)
        """
        # 编码状态
        state_feat = self.state_encoder(state)

        # 编码图像
        if self.use_image and image is not None:
            image_feat = self.image_encoder(image)
            features = torch.cat([state_feat, image_feat], dim=1)
        else:
            features = state_feat

        # 预测动作
        action = self.action_head(features)

        return action


# ============ RLPD 网络 ============

class SACCritic(nn.Module):
    """
    SAC Critic (双 Q 网络，支持图像)

    输入: 状态 + 图像 + 动作
    输出: Q值
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_size: Tuple[int, int] = (128, 128),
        image_encoder_dim: int = 256,
        state_encoder_dim: int = 256,
        hidden_dims: List[int] = [512, 512, 256],
        use_image: bool = True
    ):
        super().__init__()

        self.use_image = use_image

        # 共享编码器
        if use_image:
            self.image_encoder = ImageEncoder(
                input_channels=3,
                image_size=image_size,
                output_dim=image_encoder_dim
            )

        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            output_dim=state_encoder_dim
        )

        # 融合维度
        fusion_dim = image_encoder_dim + state_encoder_dim if use_image else state_encoder_dim
        input_dim = fusion_dim + action_dim

        # Q1 网络
        q1_layers = []
        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(input_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            input_dim = hidden_dim
        q1_layers.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2 网络
        input_dim = fusion_dim + action_dim
        q2_layers = []
        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(input_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            input_dim = hidden_dim
        q2_layers.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(self, state, action, image=None):
        """
        Args:
            state: (B, state_dim)
            action: (B, action_dim)
            image: (B, C, H, W) or None
        Returns:
            q1: (B, 1)
            q2: (B, 1)
        """
        # 编码状态
        state_feat = self.state_encoder(state)

        # 编码图像
        if self.use_image and image is not None:
            image_feat = self.image_encoder(image)
            features = torch.cat([state_feat, image_feat], dim=1)
        else:
            features = state_feat

        # 拼接动作
        x = torch.cat([features, action], dim=1)

        # 计算 Q 值
        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


class SACActor(nn.Module):
    """
    SAC Actor (随机策略，支持图像)

    输入: 状态 + 图像
    输出: 动作分布 (mean, log_std)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_size: Tuple[int, int] = (128, 128),
        image_encoder_dim: int = 256,
        state_encoder_dim: int = 256,
        hidden_dims: List[int] = [512, 512, 256],
        log_std_min: float = -20,
        log_std_max: float = 2,
        use_image: bool = True
    ):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_image = use_image

        # 编码器
        if use_image:
            self.image_encoder = ImageEncoder(
                input_channels=3,
                image_size=image_size,
                output_dim=image_encoder_dim
            )

        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            output_dim=state_encoder_dim
        )

        # 融合维度
        fusion_dim = image_encoder_dim + state_encoder_dim if use_image else state_encoder_dim

        # 主干网络
        layers = []
        input_dim = fusion_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 均值和标准差
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

    def forward(self, state, image=None):
        """
        Args:
            state: (B, state_dim)
            image: (B, C, H, W) or None
        Returns:
            mean: (B, action_dim)
            log_std: (B, action_dim)
        """
        # 编码状态
        state_feat = self.state_encoder(state)

        # 编码图像
        if self.use_image and image is not None:
            image_feat = self.image_encoder(image)
            features = torch.cat([state_feat, image_feat], dim=1)
        else:
            features = state_feat

        # 提取特征
        x = self.backbone(features)

        # 计算 mean 和 log_std
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, image=None):
        """
        采样动作（重参数化技巧）
        """
        mean, log_std = self.forward(state, image)
        std = log_std.exp()

        # 重参数化采样
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()  # rsample 支持梯度回传
        action = torch.tanh(x)

        # 计算 log_prob（修正 tanh 变换）
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state, image=None):
        """
        确定性动作（评估时使用）
        """
        mean, _ = self.forward(state, image)
        return torch.tanh(mean)


# ============ 工具函数 ============

def create_bc_policy(state_dim, action_dim, use_image=True, **kwargs):
    """创建 BC 策略"""
    return BCPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        use_image=use_image,
        **kwargs
    )


def create_sac_networks(state_dim, action_dim, use_image=True, **kwargs):
    """创建 SAC 网络（Actor + Critic）"""
    actor = SACActor(
        state_dim=state_dim,
        action_dim=action_dim,
        use_image=use_image,
        **kwargs
    )

    critic = SACCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        use_image=use_image,
        **kwargs
    )

    return actor, critic


__all__ = [
    'ImageEncoder',
    'StateEncoder',
    'BCPolicy',
    'SACActor',
    'SACCritic',
    'create_bc_policy',
    'create_sac_networks',
]
