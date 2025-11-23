#!/usr/bin/env python3
"""
数据工具集

功能：
- 查看演示数据
- 验证数据格式
- 数据统计分析
- 数据转换（pkl ↔ hdf5）

使用方法:
    # 查看演示
    python data_utils.py --view demos/reaching/demo_000.pkl

    # 统计分析
    python data_utils.py --stats demos/reaching

    # 验证格式
    python data_utils.py --validate demos/reaching

    # 转换格式
    python data_utils.py --convert demos/reaching --format hdf5
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import h5py
from tabulate import tabulate


def view_demo(demo_path):
    """查看单个演示的详细信息"""
    with open(demo_path, 'rb') as f:
        demo = pickle.load(f)

    print("=" * 60)
    print(f"演示数据: {demo_path}")
    print("=" * 60)

    # 基本信息
    print(f"\n【基本信息】")
    print(f"轨迹长度: {len(demo['actions'])} 步")
    print(f"成功标记: {demo.get('success', 'N/A')}")

    # 观测信息
    print(f"\n【观测空间】")
    obs = demo['observations'][0]

    if 'state' in obs:
        print("状态:")
        for key, value in obs['state'].items():
            arr = np.array(value)
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")

    if 'images' in obs:
        print("图像:")
        for key, value in obs['images'].items():
            arr = np.array(value)
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")

    # 动作信息
    print(f"\n【动作空间】")
    actions = np.array(demo['actions'])
    print(f"形状: {actions.shape}")
    print(f"范围: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"均值: {actions.mean(axis=0)}")
    print(f"标准差: {actions.std(axis=0)}")

    # 奖励信息
    print(f"\n【奖励】")
    rewards = np.array(demo['rewards'])
    print(f"总奖励: {rewards.sum():.3f}")
    print(f"平均奖励: {rewards.mean():.3f}")
    print(f"最大奖励: {rewards.max():.3f}")

    # 终止信息
    print(f"\n【终止标志】")
    terminals = np.array(demo['terminals'])
    truncations = np.array(demo.get('truncations', [False] * len(terminals)))
    print(f"Terminated: {terminals.sum()} / {len(terminals)}")
    print(f"Truncated: {truncations.sum()} / {len(truncations)}")


def stats_demos(demos_dir):
    """统计分析整个数据集"""
    demos_dir = Path(demos_dir)
    demo_files = sorted(demos_dir.glob("demo_*.pkl"))

    if len(demo_files) == 0:
        print(f"✗ 在 {demos_dir} 中未找到演示文件")
        return

    print("=" * 60)
    print(f"数据集统计: {demos_dir}")
    print("=" * 60)

    # 收集统计信息
    lengths = []
    total_rewards = []
    success_flags = []
    action_dims = []

    for demo_file in demo_files:
        with open(demo_file, 'rb') as f:
            demo = pickle.load(f)

        lengths.append(len(demo['actions']))
        total_rewards.append(sum(demo['rewards']))
        success_flags.append(demo.get('success', False))
        action_dims.append(np.array(demo['actions'][0]).shape[0])

    # 打印统计
    print(f"\n【数据集概览】")
    print(f"演示数量: {len(demo_files)}")
    print(f"成功演示: {sum(success_flags)} ({sum(success_flags)/len(success_flags)*100:.1f}%)")
    print(f"失败演示: {len(success_flags) - sum(success_flags)}")

    print(f"\n【轨迹长度】")
    print(f"最短: {min(lengths)} 步")
    print(f"最长: {max(lengths)} 步")
    print(f"平均: {np.mean(lengths):.1f} 步")
    print(f"中位数: {np.median(lengths):.1f} 步")

    print(f"\n【总奖励】")
    print(f"最小: {min(total_rewards):.3f}")
    print(f"最大: {max(total_rewards):.3f}")
    print(f"平均: {np.mean(total_rewards):.3f}")

    print(f"\n【动作空间】")
    print(f"动作维度: {action_dims[0]}")

    # 分类统计
    success_lengths = [l for l, s in zip(lengths, success_flags) if s]
    fail_lengths = [l for l, s in zip(lengths, success_flags) if not s]

    if success_lengths and fail_lengths:
        print(f"\n【成功 vs 失败】")
        table_data = [
            ["", "成功", "失败"],
            ["数量", len(success_lengths), len(fail_lengths)],
            ["平均长度", f"{np.mean(success_lengths):.1f}", f"{np.mean(fail_lengths):.1f}"],
            ["平均奖励", f"{np.mean([r for r, s in zip(total_rewards, success_flags) if s]):.3f}",
                         f"{np.mean([r for r, s in zip(total_rewards, success_flags) if not s]):.3f}"]
        ]
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))


def validate_demos(demos_dir):
    """验证演示数据格式"""
    demos_dir = Path(demos_dir)
    demo_files = sorted(demos_dir.glob("demo_*.pkl"))

    print("=" * 60)
    print(f"验证数据格式: {demos_dir}")
    print("=" * 60)

    errors = []

    for demo_file in demo_files:
        try:
            with open(demo_file, 'rb') as f:
                demo = pickle.load(f)

            # 检查必需字段
            required_keys = ['observations', 'actions', 'rewards', 'terminals']
            for key in required_keys:
                if key not in demo:
                    errors.append(f"{demo_file.name}: 缺少字段 '{key}'")

            # 检查长度一致性
            lengths = {
                'observations': len(demo.get('observations', [])),
                'actions': len(demo.get('actions', [])),
                'rewards': len(demo.get('rewards', [])),
                'terminals': len(demo.get('terminals', []))
            }

            if len(set(lengths.values())) > 1:
                errors.append(f"{demo_file.name}: 长度不一致 {lengths}")

            # 检查观测格式
            if 'observations' in demo and len(demo['observations']) > 0:
                obs = demo['observations'][0]
                if 'state' not in obs:
                    errors.append(f"{demo_file.name}: observation 缺少 'state'")
                if 'images' not in obs:
                    errors.append(f"{demo_file.name}: observation 缺少 'images'")

        except Exception as e:
            errors.append(f"{demo_file.name}: 读取错误 - {e}")

    # 打印结果
    if errors:
        print(f"\n✗ 发现 {len(errors)} 个错误:\n")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n✓ 所有 {len(demo_files)} 个演示文件格式正确！")


def convert_to_hdf5(demos_dir):
    """将 pkl 格式转换为 hdf5"""
    demos_dir = Path(demos_dir)
    demo_files = sorted(demos_dir.glob("demo_*.pkl"))

    print(f"转换 {len(demo_files)} 个文件到 HDF5...")

    for demo_file in demo_files:
        with open(demo_file, 'rb') as f:
            demo = pickle.load(f)

        h5_path = demo_file.with_suffix('.h5')

        with h5py.File(h5_path, 'w') as f:
            # 基本数据
            actions = np.array(demo['actions'])
            rewards = np.array(demo['rewards'])
            terminals = np.array(demo['terminals'])

            f.create_dataset('actions', data=actions, compression='gzip')
            f.create_dataset('rewards', data=rewards, compression='gzip')
            f.create_dataset('terminals', data=terminals, compression='gzip')

            # 观测数据
            obs_group = f.create_group('observations')
            state_group = obs_group.create_group('state')
            images_group = obs_group.create_group('images')

            # 提取并保存
            observations = demo['observations']

            for state_key in observations[0]['state'].keys():
                data = np.array([obs['state'][state_key] for obs in observations])
                state_group.create_dataset(state_key, data=data, compression='gzip')

            for image_key in observations[0]['images'].keys():
                data = np.array([obs['images'][image_key] for obs in observations])
                images_group.create_dataset(image_key, data=data, compression='gzip')

            # 元数据
            f.attrs['success'] = demo.get('success', False)

        print(f"  ✓ {demo_file.name} → {h5_path.name}")

    print(f"✓ 转换完成！")


def main():
    parser = argparse.ArgumentParser(description='数据工具')
    parser.add_argument('--view', type=str, help='查看单个演示')
    parser.add_argument('--stats', type=str, help='统计分析数据集')
    parser.add_argument('--validate', type=str, help='验证数据格式')
    parser.add_argument('--convert', type=str, help='转换数据格式')
    parser.add_argument('--format', type=str, default='hdf5', choices=['hdf5', 'pkl'],
                        help='目标格式')

    args = parser.parse_args()

    if args.view:
        view_demo(args.view)
    elif args.stats:
        stats_demos(args.stats)
    elif args.validate:
        validate_demos(args.validate)
    elif args.convert:
        if args.format == 'hdf5':
            convert_to_hdf5(args.convert)
        else:
            print("pkl 格式转换暂未实现")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
