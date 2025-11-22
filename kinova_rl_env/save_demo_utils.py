#!/usr/bin/env python3
"""
Demo数据保存工具：支持pkl和hdf5两种格式

pkl格式：
- 优点：简单、兼容HIL-SERL、便于调试
- 适用：小到中等规模数据集（<100条demo）

hdf5格式：
- 优点：高效、压缩、可增量读写
- 适用：大规模数据集（>100条demo）
"""

import pickle
import numpy as np
from pathlib import Path
import h5py


def save_demo_pkl(trajectory, save_path, success=True):
    """
    保存为pkl格式（HIL-SERL标准）

    Args:
        trajectory: list of dicts
        save_path: Path or str
        success: bool
    """
    demo_data = {
        'observations': [t['observation'] for t in trajectory],
        'actions': [t['action'] for t in trajectory],
        'rewards': [t['reward'] for t in trajectory],
        'terminals': [t['terminated'] for t in trajectory],
        'truncations': [t['truncated'] for t in trajectory],
        'success': success
    }

    with open(save_path, 'wb') as f:
        pickle.dump(demo_data, f)

    return demo_data


def save_demo_hdf5(trajectory, save_path, success=True):
    """
    保存为hdf5格式（高效压缩）

    Args:
        trajectory: list of dicts
        save_path: Path or str
        success: bool

    HDF5结构：
    demo.h5
    ├── observations/
    │   ├── state/
    │   │   ├── tcp_pose (T, 7)
    │   │   ├── tcp_vel (T, 6)
    │   │   └── gripper_pose (T, 1)
    │   └── images/
    │       └── wrist_1 (T, 128, 128, 3)
    ├── actions (T, 7)
    ├── rewards (T,)
    ├── terminals (T,)
    ├── truncations (T,)
    └── success (scalar)
    """
    with h5py.File(save_path, 'w') as f:
        # 提取数据
        observations = [t['observation'] for t in trajectory]
        actions = np.array([t['action'] for t in trajectory])
        rewards = np.array([t['reward'] for t in trajectory])
        terminals = np.array([t['terminated'] for t in trajectory])
        truncations = np.array([t['truncated'] for t in trajectory])

        # 保存actions, rewards, terminals
        f.create_dataset('actions', data=actions, compression='gzip')
        f.create_dataset('rewards', data=rewards, compression='gzip')
        f.create_dataset('terminals', data=terminals, compression='gzip')
        f.create_dataset('truncations', data=truncations, compression='gzip')
        f.create_dataset('success', data=success)

        # 保存observations（嵌套结构）
        obs_group = f.create_group('observations')
        state_group = obs_group.create_group('state')
        images_group = obs_group.create_group('images')

        # 提取state数据
        tcp_poses = np.array([obs['state']['tcp_pose'] for obs in observations])
        tcp_vels = np.array([obs['state']['tcp_vel'] for obs in observations])
        gripper_poses = np.array([obs['state']['gripper_pose'] for obs in observations])

        state_group.create_dataset('tcp_pose', data=tcp_poses, compression='gzip')
        state_group.create_dataset('tcp_vel', data=tcp_vels, compression='gzip')
        state_group.create_dataset('gripper_pose', data=gripper_poses, compression='gzip')

        # 提取images数据
        wrist_1_images = np.array([obs['images']['wrist_1'] for obs in observations])
        images_group.create_dataset('wrist_1', data=wrist_1_images, compression='gzip')


def load_demo_pkl(demo_path):
    """加载pkl格式的demo"""
    with open(demo_path, 'rb') as f:
        return pickle.load(f)


def load_demo_hdf5(demo_path):
    """
    加载hdf5格式的demo，转换为pkl兼容格式

    Returns:
        dict: 与pkl格式相同的结构
    """
    with h5py.File(demo_path, 'r') as f:
        # 读取基本数据
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        truncations = f['truncations'][:]
        success = f['success'][()]

        # 读取observations
        tcp_poses = f['observations/state/tcp_pose'][:]
        tcp_vels = f['observations/state/tcp_vel'][:]
        gripper_poses = f['observations/state/gripper_pose'][:]
        wrist_1_images = f['observations/images/wrist_1'][:]

        # 重构为嵌套字典列表
        observations = []
        for i in range(len(actions)):
            obs = {
                'state': {
                    'tcp_pose': tcp_poses[i],
                    'tcp_vel': tcp_vels[i],
                    'gripper_pose': gripper_poses[i]
                },
                'images': {
                    'wrist_1': wrist_1_images[i]
                }
            }
            observations.append(obs)

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'truncations': truncations,
            'success': success
        }


def convert_pkl_to_hdf5(pkl_path, hdf5_path=None):
    """
    将pkl格式转换为hdf5格式

    Args:
        pkl_path: str or Path
        hdf5_path: str or Path (如果为None，自动替换扩展名)
    """
    # 加载pkl
    demo = load_demo_pkl(pkl_path)

    # 生成hdf5路径
    if hdf5_path is None:
        hdf5_path = Path(pkl_path).with_suffix('.h5')

    # 重构为trajectory格式
    trajectory = []
    for i in range(len(demo['actions'])):
        trajectory.append({
            'observation': demo['observations'][i],
            'action': demo['actions'][i],
            'reward': demo['rewards'][i],
            'terminated': demo['terminals'][i],
            'truncated': demo['truncations'][i]
        })

    # 保存为hdf5
    save_demo_hdf5(trajectory, hdf5_path, success=demo['success'])

    print(f"✓ 已转换: {pkl_path} → {hdf5_path}")


def batch_convert_pkl_to_hdf5(demo_dir, pattern="demo_*.pkl"):
    """批量转换pkl到hdf5"""
    demo_dir = Path(demo_dir)
    pkl_files = list(demo_dir.glob(pattern))

    print(f"找到 {len(pkl_files)} 个pkl文件")

    for pkl_file in pkl_files:
        convert_pkl_to_hdf5(pkl_file)

    print(f"✓ 批量转换完成！")


# 使用示例
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', type=str, help='转换单个pkl到hdf5')
    parser.add_argument('--batch_convert', type=str, help='批量转换目录下的pkl')
    parser.add_argument('--test_hdf5', type=str, help='测试读取hdf5文件')

    args = parser.parse_args()

    if args.convert:
        convert_pkl_to_hdf5(args.convert)

    elif args.batch_convert:
        batch_convert_pkl_to_hdf5(args.batch_convert)

    elif args.test_hdf5:
        demo = load_demo_hdf5(args.test_hdf5)
        print(f"✓ 成功加载hdf5文件")
        print(f"  轨迹长度: {len(demo['actions'])}")
        print(f"  成功: {demo['success']}")
