#!/usr/bin/env python3
"""
测试收集的demo数据格式是否正确

使用方法:
    python test_demo_format.py --demo_path demos/reaching/demo_000.pkl
"""

import pickle
import numpy as np
from pathlib import Path


def test_demo_format(demo_path):
    """
    验证demo文件格式是否符合HIL-SERL要求

    HIL-SERL期望格式:
    {
        'observations': list of dicts,
        'actions': list of arrays (7,),
        'rewards': list of floats,
        'terminals': list of bools,
        'truncations': list of bools (可选),
        'success': bool
    }
    """
    print(f"\n{'=' * 60}")
    print(f"测试Demo文件: {demo_path}")
    print(f"{'=' * 60}\n")

    # 加载文件
    with open(demo_path, 'rb') as f:
        demo = pickle.load(f)

    print("【1. 检查顶层键】")
    required_keys = ['observations', 'actions', 'rewards', 'terminals']
    optional_keys = ['truncations', 'success']

    for key in required_keys:
        if key in demo:
            print(f"  ✓ '{key}' 存在")
        else:
            print(f"  ✗ '{key}' 缺失！")
            return False

    for key in optional_keys:
        if key in demo:
            print(f"  ✓ '{key}' 存在 (可选)")

    # 获取轨迹长度
    traj_length = len(demo['observations'])
    print(f"\n【2. 轨迹长度】")
    print(f"  轨迹长度: {traj_length} 步")

    # 检查长度一致性
    print(f"\n【3. 检查数组长度一致性】")
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        if key in demo:
            length = len(demo[key])
            match = "✓" if length == traj_length else "✗"
            print(f"  {match} {key}: {length}")

    # 检查observation格式
    print(f"\n【4. 检查observation格式】")
    first_obs = demo['observations'][0]

    if 'state' in first_obs and 'images' in first_obs:
        print(f"  ✓ observation包含'state'和'images'键")

        # 检查state子键
        state = first_obs['state']
        if isinstance(state, dict):
            print(f"  ✓ state是字典")
            for key in ['tcp_pose', 'tcp_vel', 'gripper_pose']:
                if key in state:
                    shape = state[key].shape if hasattr(state[key], 'shape') else len(state[key])
                    print(f"    ✓ state['{key}']: shape={shape}")
                else:
                    print(f"    ✗ state['{key}'] 缺失")
        else:
            print(f"  ✗ state不是字典")

        # 检查images子键
        images = first_obs['images']
        if isinstance(images, dict):
            print(f"  ✓ images是字典")
            for cam_name, img in images.items():
                print(f"    ✓ images['{cam_name}']: shape={img.shape}, dtype={img.dtype}")
        else:
            print(f"  ✗ images不是字典")
    else:
        print(f"  ✗ observation格式不正确")
        print(f"  实际键: {first_obs.keys()}")

    # 检查action格式
    print(f"\n【5. 检查action格式】")
    first_action = demo['actions'][0]
    print(f"  Action shape: {first_action.shape}")
    print(f"  Action dtype: {first_action.dtype}")
    print(f"  Action范围: [{first_action.min():.3f}, {first_action.max():.3f}]")

    if first_action.shape == (7,):
        print(f"  ✓ Action维度正确 (7,)")
        print(f"    [delta_pos(3), delta_rot(3), gripper(1)]")
    else:
        print(f"  ✗ Action维度不正确，期望(7,)，实际{first_action.shape}")

    # 检查reward格式
    print(f"\n【6. 检查reward格式】")
    rewards = demo['rewards']
    unique_rewards = np.unique(rewards)
    total_reward = sum(rewards)
    success_steps = sum([1 for r in rewards if r > 0])

    print(f"  Reward类型: {type(rewards[0])}")
    print(f"  唯一值: {unique_rewards}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  成功步数: {success_steps} / {traj_length}")

    # 检查success标志
    print(f"\n【7. 检查success标志】")
    if 'success' in demo:
        success = demo['success']
        print(f"  Success: {success}")
        print(f"  Success类型: {type(success)}")
    else:
        print(f"  ⚠ 'success'键缺失")

    print(f"\n{'=' * 60}")
    print(f"✓ Demo格式验证完成！")
    print(f"{'=' * 60}\n")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='测试demo数据格式')
    parser.add_argument('--demo_path', type=str, required=True, help='Demo文件路径')
    args = parser.parse_args()

    demo_path = Path(args.demo_path)

    if not demo_path.exists():
        print(f"✗ 文件不存在: {demo_path}")
        return

    test_demo_format(demo_path)


if __name__ == '__main__':
    main()
