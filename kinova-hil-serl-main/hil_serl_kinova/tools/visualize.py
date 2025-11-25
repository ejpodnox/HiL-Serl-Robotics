#!/usr/bin/env python3
"""
可视化工具

功能：
- 绘制轨迹曲线
- 绘制动作分布
- 绘制奖励曲线
- 绘制训练曲线

使用方法:
    # 绘制单个演示的轨迹
    python visualize.py --trajectory demos/reaching/demo_000.pkl

    # 绘制数据集的统计
    python visualize.py --dataset demos/reaching --output plots/

    # 绘制训练曲线
    python visualize.py --training logs/bc --output plots/training.png
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_trajectory(demo_path, output_path=None):
    """绘制单个轨迹的详细信息"""
    with open(demo_path, 'rb') as f:
        demo = pickle.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'演示轨迹: {Path(demo_path).name}', fontsize=16)

    # 1. TCP 位置轨迹
    ax = axes[0, 0]
    tcp_poses = np.array([obs['state']['tcp_pose'] for obs in demo['observations']])
    positions = tcp_poses[:, :3]  # x, y, z

    ax.plot(positions[:, 0], label='X', linewidth=2)
    ax.plot(positions[:, 1], label='Y', linewidth=2)
    ax.plot(positions[:, 2], label='Z', linewidth=2)
    ax.set_title('TCP 位置轨迹')
    ax.set_xlabel('时间步')
    ax.set_ylabel('位置 (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 动作分布
    ax = axes[0, 1]
    actions = np.array(demo['actions'])

    for i in range(min(3, actions.shape[1])):  # 只显示前3维
        ax.plot(actions[:, i], label=f'Action {i}', alpha=0.7)

    ax.set_title('动作序列')
    ax.set_xlabel('时间步')
    ax.set_ylabel('动作值')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 奖励曲线
    ax = axes[1, 0]
    rewards = np.array(demo['rewards'])
    cumulative_rewards = np.cumsum(rewards)

    ax.plot(rewards, label='即时奖励', alpha=0.7)
    ax.plot(cumulative_rewards, label='累积奖励', linewidth=2)
    ax.set_title('奖励曲线')
    ax.set_xlabel('时间步')
    ax.set_ylabel('奖励')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 速度曲线
    ax = axes[1, 1]
    if 'tcp_vel' in demo['observations'][0]['state']:
        tcp_vels = np.array([obs['state']['tcp_vel'] for obs in demo['observations']])
        linear_vels = tcp_vels[:, :3]  # vx, vy, vz

        vel_magnitude = np.linalg.norm(linear_vels, axis=1)
        ax.plot(vel_magnitude, label='速度大小', linewidth=2)
        ax.set_title('TCP 速度')
        ax.set_xlabel('时间步')
        ax.set_ylabel('速度 (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_dataset(demos_dir, output_dir=None):
    """绘制整个数据集的统计图"""
    demos_dir = Path(demos_dir)
    demo_files = sorted(demos_dir.glob("demo_*.pkl"))

    # 收集数据
    lengths = []
    total_rewards = []
    success_flags = []
    all_actions = []

    for demo_file in demo_files:
        with open(demo_file, 'rb') as f:
            demo = pickle.load(f)

        lengths.append(len(demo['actions']))
        total_rewards.append(sum(demo['rewards']))
        success_flags.append(demo.get('success', False))
        all_actions.extend(demo['actions'])

    all_actions = np.array(all_actions)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'数据集统计: {demos_dir.name}', fontsize=16)

    # 1. 轨迹长度分布
    ax = axes[0, 0]
    ax.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'均值: {np.mean(lengths):.1f}')
    ax.set_title('轨迹长度分布')
    ax.set_xlabel('长度')
    ax.set_ylabel('数量')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 总奖励分布
    ax = axes[0, 1]
    success_rewards = [r for r, s in zip(total_rewards, success_flags) if s]
    fail_rewards = [r for r, s in zip(total_rewards, success_flags) if not s]

    if success_rewards and fail_rewards:
        ax.hist([success_rewards, fail_rewards], bins=15, label=['成功', '失败'],
                alpha=0.7, color=['green', 'red'], edgecolor='black')
        ax.set_title('总奖励分布')
        ax.set_xlabel('总奖励')
        ax.set_ylabel('数量')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. 成功率饼图
    ax = axes[0, 2]
    success_count = sum(success_flags)
    fail_count = len(success_flags) - success_count

    ax.pie([success_count, fail_count], labels=['成功', '失败'],
           autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
    ax.set_title(f'成功率: {success_count}/{len(success_flags)}')

    # 4-6. 动作维度分布
    for i in range(min(3, all_actions.shape[1])):
        ax = axes[1, i]
        ax.hist(all_actions[:, i], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(all_actions[:, i].mean(), color='red', linestyle='--',
                   label=f'均值: {all_actions[:, i].mean():.3f}')
        ax.set_title(f'动作维度 {i}')
        ax.set_xlabel('值')
        ax.set_ylabel('频数')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'dataset_stats.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_training(log_dir, output_path=None):
    """绘制训练曲线（从 Tensorboard 日志）"""
    from tensorboard.backend.event_processing import event_accumulator

    log_dir = Path(log_dir)

    # 查找 event 文件
    event_files = list(log_dir.glob('events.out.tfevents.*'))

    if not event_files:
        print(f"✗ 在 {log_dir} 中未找到 Tensorboard 事件文件")
        return

    # 加载事件
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()

    # 获取所有标量标签
    tags = ea.Tags()['scalars']

    print(f"找到 {len(tags)} 个标量:")
    for tag in tags:
        print(f"  - {tag}")

    # 创建图表
    n_plots = len(tags)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, tag in enumerate(tags):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        axes[i].plot(steps, values, linewidth=2)
        axes[i].set_title(tag)
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(tags), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_trajectories(demos_dir, output_path=None, max_demos=10):
    """在同一张图上绘制多条轨迹"""
    demos_dir = Path(demos_dir)
    demo_files = sorted(demos_dir.glob("demo_*.pkl"))[:max_demos]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'多轨迹对比 ({len(demo_files)} 条)', fontsize=16)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(demo_files)))

    for demo_file, color in zip(demo_files, colors):
        with open(demo_file, 'rb') as f:
            demo = pickle.load(f)

        success = demo.get('success', False)
        label = f"{demo_file.stem} ({'✓' if success else '✗'})"

        # TCP 位置
        tcp_poses = np.array([obs['state']['tcp_pose'] for obs in demo['observations']])
        positions = tcp_poses[:, :3]

        # 3D 轨迹
        ax = axes[0, 0]
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                label=label, color=color, alpha=0.7)
        ax.set_title('3D 轨迹')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # 奖励曲线
        ax = axes[0, 1]
        rewards = np.cumsum(demo['rewards'])
        ax.plot(rewards, label=label, color=color, alpha=0.7)
        ax.set_title('累积奖励')
        ax.set_xlabel('时间步')
        ax.set_ylabel('累积奖励')

        # XY 平面轨迹
        ax = axes[1, 0]
        ax.plot(positions[:, 0], positions[:, 1], label=label, color=color, alpha=0.7)
        ax.scatter(positions[0, 0], positions[0, 1], marker='o', s=100, color=color, edgecolors='black')
        ax.scatter(positions[-1, 0], positions[-1, 1], marker='s', s=100, color=color, edgecolors='black')
        ax.set_title('XY 平面轨迹')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        # 轨迹长度
        ax = axes[1, 1]
        ax.bar(demo_file.stem, len(demo['actions']), color=color, alpha=0.7)

    axes[0, 1].legend(fontsize=8, loc='best')
    axes[1, 0].legend(fontsize=8, loc='best')
    axes[1, 1].set_title('轨迹长度')
    axes[1, 1].set_xlabel('演示')
    axes[1, 1].set_ylabel('长度')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化工具')
    parser.add_argument('--trajectory', type=str, help='绘制单个轨迹')
    parser.add_argument('--dataset', type=str, help='绘制数据集统计')
    parser.add_argument('--training', type=str, help='绘制训练曲线')
    parser.add_argument('--multi', type=str, help='绘制多条轨迹对比')
    parser.add_argument('--output', type=str, help='输出路径')
    parser.add_argument('--max_demos', type=int, default=10, help='最多显示的轨迹数')

    args = parser.parse_args()

    if args.trajectory:
        plot_trajectory(args.trajectory, args.output)
    elif args.dataset:
        plot_dataset(args.dataset, args.output)
    elif args.training:
        plot_training(args.training, args.output)
    elif args.multi:
        plot_multiple_trajectories(args.multi, args.output, args.max_demos)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
