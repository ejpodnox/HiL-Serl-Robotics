#!/usr/bin/env python3
"""
测试数据加载和处理流程

独立测试，使用虚拟数据。
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
import tempfile


def test_data_format():
    """测试 HIL-SERL 数据格式"""
    print("\n" + "=" * 60)
    print("【测试 1】HIL-SERL 数据格式")
    print("=" * 60)

    try:
        # 创建虚拟演示数据
        demo = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }

        # 生成 50 步数据
        for i in range(50):
            # 观测
            obs = {
                'state': {
                    'tcp_pose': np.random.randn(7).astype(np.float32),
                    'tcp_vel': np.random.randn(6).astype(np.float32),
                    'joint_positions': np.random.randn(7).astype(np.float32),
                    'gripper_position': np.random.rand(1).astype(np.float32),
                },
                'images': {
                    'wrist_1': (np.random.rand(128, 128, 3) * 255).astype(np.uint8),
                }
            }
            demo['observations'].append(obs)

            # 动作
            action = np.random.randn(7).astype(np.float32) * 0.01
            demo['actions'].append(action)

            # 奖励
            reward = 0.0 if i < 49 else 1.0
            demo['rewards'].append(reward)

            # 完成标志
            done = (i == 49)
            demo['dones'].append(done)

            # 额外信息
            info = {'step': i}
            demo['infos'].append(info)

        print(f"✓ 创建虚拟演示数据:")
        print(f"  - 步数: {len(demo['observations'])}")
        print(f"  - 状态维度: {len(demo['observations'][0]['state'])}")
        print(f"  - 图像数量: {len(demo['observations'][0]['images'])}")
        print(f"  - 动作维度: {demo['actions'][0].shape}")

        return demo

    except Exception as e:
        print(f"✗ 数据格式创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_saving(demo):
    """测试数据保存"""
    print("\n" + "=" * 60)
    print("【测试 2】数据保存")
    print("=" * 60)

    if demo is None:
        print("✗ 跳过（无数据）")
        return None, False

    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)

        # 保存
        with open(temp_path, 'wb') as f:
            pickle.dump(demo, f)

        file_size = temp_path.stat().st_size / 1024  # KB
        print(f"✓ 数据保存成功:")
        print(f"  - 路径: {temp_path}")
        print(f"  - 大小: {file_size:.1f} KB")

        return temp_path, True

    except Exception as e:
        print(f"✗ 数据保存失败: {e}")
        return None, False


def test_data_loading(demo_path):
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("【测试 3】数据加载")
    print("=" * 60)

    if demo_path is None:
        print("✗ 跳过（无文件）")
        return False

    try:
        with open(demo_path, 'rb') as f:
            loaded_demo = pickle.load(f)

        print(f"✓ 数据加载成功:")
        print(f"  - 步数: {len(loaded_demo['observations'])}")
        print(f"  - 总奖励: {sum(loaded_demo['rewards'])}")

        # 验证数据完整性
        assert len(loaded_demo['observations']) == len(loaded_demo['actions'])
        assert len(loaded_demo['observations']) == len(loaded_demo['rewards'])
        print("✓ 数据完整性验证通过")

        # 清理临时文件
        demo_path.unlink()
        print(f"✓ 清理临时文件")

        return True

    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """测试 PyTorch Dataset 创建"""
    print("\n" + "=" * 60)
    print("【测试 4】PyTorch Dataset")
    print("=" * 60)

    try:
        import torch
        from torch.utils.data import Dataset

        # 简单的 Dataset 类
        class DummyDataset(Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                state = torch.randn(21)
                image = torch.randn(3, 128, 128)
                action = torch.randn(7)
                return state, image, action

        dataset = DummyDataset(num_samples=100)
        print(f"✓ Dataset 创建成功:")
        print(f"  - 样本数: {len(dataset)}")

        # 测试获取样本
        state, image, action = dataset[0]
        print(f"✓ 样本获取成功:")
        print(f"  - 状态: {state.shape}")
        print(f"  - 图像: {image.shape}")
        print(f"  - 动作: {action.shape}")

        return True

    except Exception as e:
        print(f"✗ Dataset 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试 DataLoader"""
    print("\n" + "=" * 60)
    print("【测试 5】DataLoader")
    print("=" * 60)

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader

        class DummyDataset(Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                state = torch.randn(21)
                image = torch.randn(3, 128, 128)
                action = torch.randn(7)
                return state, image, action

        dataset = DummyDataset(num_samples=100)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        print(f"✓ DataLoader 创建成功:")
        print(f"  - Batch size: {dataloader.batch_size}")
        print(f"  - Batches: {len(dataloader)}")

        # 测试迭代
        for batch_idx, (states, images, actions) in enumerate(dataloader):
            print(f"✓ Batch {batch_idx}:")
            print(f"  - States: {states.shape}")
            print(f"  - Images: {images.shape}")
            print(f"  - Actions: {actions.shape}")

            if batch_idx >= 2:  # 只测试前3个batch
                break

        return True

    except Exception as e:
        print(f"✗ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='数据流程测试')
    args = parser.parse_args()

    results = {}

    # 测试 1: 数据格式
    demo = test_data_format()
    results['format'] = demo is not None

    # 测试 2: 数据保存
    demo_path, save_success = test_data_saving(demo)
    results['saving'] = save_success

    # 测试 3: 数据加载
    results['loading'] = test_data_loading(demo_path)

    # 测试 4: Dataset 创建
    results['dataset'] = test_dataset_creation()

    # 测试 5: DataLoader
    results['dataloader'] = test_dataloader()

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
        print("\n✓ 所有测试通过，数据流程可用")
        return 0
    else:
        print("\n⚠️  部分测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
