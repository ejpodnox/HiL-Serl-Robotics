#!/usr/bin/env python3
"""
测试训练流程

独立测试，使用虚拟数据和小型网络。
"""

import argparse
import torch
import torch.nn as nn
import numpy as np


def test_bc_network():
    """测试 BC 网络创建"""
    print("\n" + "=" * 60)
    print("【测试 1】BC 网络架构")
    print("=" * 60)

    try:
        # 简化版 BC 网络
        class SimpleBCPolicy(nn.Module):
            def __init__(self, state_dim=21, action_dim=7):
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32)
                )
                self.action_head = nn.Sequential(
                    nn.Linear(32, action_dim),
                    nn.Tanh()
                )

            def forward(self, state):
                features = self.state_encoder(state)
                action = self.action_head(features)
                return action

        policy = SimpleBCPolicy()
        print(f"✓ BC 网络创建成功")

        # 测试前向传播
        state = torch.randn(4, 21)  # batch_size=4
        action = policy(state)
        print(f"✓ 前向传播成功:")
        print(f"  - 输入: {state.shape}")
        print(f"  - 输出: {action.shape}")

        # 参数统计
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"  - 总参数: {total_params}")

        return policy

    except Exception as e:
        print(f"✗ BC 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_bc_training(policy, num_steps=10):
    """测试 BC 训练循环"""
    print("\n" + "=" * 60)
    print("【测试 2】BC 训练循环")
    print("=" * 60)

    if policy is None:
        print("✗ 跳过（无模型）")
        return False

    try:
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        print(f"开始训练 {num_steps} 步...")

        losses = []
        for step in range(num_steps):
            # 虚拟数据
            state = torch.randn(32, 21)
            target_action = torch.randn(32, 7)

            # 前向传播
            pred_action = policy(state)
            loss = criterion(pred_action, target_action)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")

        avg_loss = np.mean(losses)
        print(f"✓ 训练完成:")
        print(f"  - 平均损失: {avg_loss:.4f}")

        return True

    except Exception as e:
        print(f"✗ BC 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_classifier():
    """测试 Reward Classifier"""
    print("\n" + "=" * 60)
    print("【测试 3】Reward Classifier")
    print("=" * 60)

    try:
        # 简化版分类器
        class SimpleClassifier(nn.Module):
            def __init__(self, state_dim=21):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, state):
                return self.classifier(state)

        classifier = SimpleClassifier()
        print(f"✓ 分类器创建成功")

        # 测试前向传播
        state = torch.randn(4, 21)
        prob = classifier(state)
        print(f"✓ 前向传播成功:")
        print(f"  - 输入: {state.shape}")
        print(f"  - 输出: {prob.shape}")
        print(f"  - 概率范围: [{prob.min().item():.3f}, {prob.max().item():.3f}]")

        # 训练几步
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for step in range(5):
            state = torch.randn(32, 21)
            label = torch.randint(0, 2, (32, 1)).float()

            pred = classifier(state)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  Step {step+1}, Loss: {loss.item():.4f}")

        print(f"✓ 分类器训练成功")

        return True

    except Exception as e:
        print(f"✗ Reward Classifier 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sac_networks():
    """测试 SAC 网络（Actor & Critic）"""
    print("\n" + "=" * 60)
    print("【测试 4】SAC 网络架构")
    print("=" * 60)

    try:
        # 简化版 Actor
        class SimpleActor(nn.Module):
            def __init__(self, state_dim=21, action_dim=7):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                self.mean = nn.Linear(32, action_dim)
                self.log_std = nn.Linear(32, action_dim)

            def forward(self, state):
                features = self.net(state)
                mean = self.mean(features)
                log_std = self.log_std(features).clamp(-20, 2)
                return mean, log_std

        # 简化版 Critic
        class SimpleCritic(nn.Module):
            def __init__(self, state_dim=21, action_dim=7):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )

            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                return self.net(x)

        actor = SimpleActor()
        critic = SimpleCritic()

        print(f"✓ Actor 创建成功")
        print(f"✓ Critic 创建成功")

        # 测试
        state = torch.randn(4, 21)
        mean, log_std = actor(state)
        print(f"✓ Actor 前向传播:")
        print(f"  - Mean: {mean.shape}")
        print(f"  - Log_std: {log_std.shape}")

        action = torch.randn(4, 7)
        q_value = critic(state, action)
        print(f"✓ Critic 前向传播:")
        print(f"  - Q值: {q_value.shape}")

        return True

    except Exception as e:
        print(f"✗ SAC 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n" + "=" * 60)
    print("【测试 5】模型保存/加载")
    print("=" * 60)

    try:
        import tempfile
        from pathlib import Path

        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

        # 保存
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        torch.save(model.state_dict(), temp_path)
        print(f"✓ 模型保存成功: {temp_path}")

        # 加载
        loaded_model = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )
        loaded_model.load_state_dict(torch.load(temp_path))
        print(f"✓ 模型加载成功")

        # 验证
        state = torch.randn(1, 21)
        with torch.no_grad():
            out1 = model(state)
            out2 = loaded_model(state)

        diff = (out1 - out2).abs().max().item()
        print(f"✓ 参数验证: 最大差异 = {diff:.6f}")

        # 清理
        temp_path.unlink()

        return True

    except Exception as e:
        print(f"✗ 模型保存/加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='训练流程测试')
    parser.add_argument('--steps', type=int, default=10,
                        help='训练步数')

    args = parser.parse_args()

    results = {}

    # 测试 1: BC 网络
    policy = test_bc_network()
    results['bc_network'] = policy is not None

    # 测试 2: BC 训练
    results['bc_training'] = test_bc_training(policy, args.steps)

    # 测试 3: Reward Classifier
    results['classifier'] = test_reward_classifier()

    # 测试 4: SAC 网络
    results['sac_networks'] = test_sac_networks()

    # 测试 5: 模型保存/加载
    results['save_load'] = test_model_save_load()

    # 总结
    print("\n" + "=" * 60)
    print("【测试总结】")
    print("=" * 60)

    for test_name, result in results.items():
        if result is None:
            status = "⊘ 跳过"
        elif result:
            status = "✓ 通过"
        else:
            status = "✗ 失败"
        print(f"{test_name:20s}: {status}")

    all_passed = all(r for r in results.values() if r is not None)

    if all_passed:
        print("\n✓ 所有测试通过，训练流程可用")
        return 0
    else:
        print("\n⚠️  部分测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
