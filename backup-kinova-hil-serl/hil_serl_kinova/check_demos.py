#!/usr/bin/env python3
"""
检查演示数据质量

用法:
    python check_demos.py --demos-dir ./demos/socket_insertion
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import glob


def load_demo(demo_path):
    """加载单条演示"""
    with open(demo_path, 'rb') as f:
        demo = pickle.load(f)
    return demo


def check_demo_validity(demo, demo_path):
    """检查单条演示的有效性"""

    errors = []
    warnings = []

    # 检查必需字段
    required_keys = ["observations", "actions", "rewards", "terminals"]
    for key in required_keys:
        if key not in demo:
            errors.append(f"缺少字段: {key}")

    if errors:
        return False, errors, warnings

    # 检查长度一致性
    obs_len = len(demo["observations"])
    act_len = len(demo["actions"])
    rew_len = len(demo["rewards"])

    if not (obs_len == act_len == rew_len):
        errors.append(f"长度不一致: obs={obs_len}, act={act_len}, rew={rew_len}")

    # 检查是否太短
    if obs_len < 10:
        warnings.append(f"演示太短: {obs_len} 步")

    # 检查是否太长
    if obs_len > 1000:
        warnings.append(f"演示很长: {obs_len} 步（可能未正常终止）")

    # 检查数据类型
    if len(demo["actions"]) > 0:
        action = demo["actions"][0]
        if not isinstance(action, np.ndarray):
            errors.append(f"动作类型错误: {type(action)}")

    # 检查奖励范围
    if len(demo["rewards"]) > 0:
        rewards = np.array(demo["rewards"])
        if np.any(np.isnan(rewards)):
            errors.append("奖励包含NaN")
        if np.any(np.isinf(rewards)):
            errors.append("奖励包含Inf")

    return len(errors) == 0, errors, warnings


def analyze_demos(demo_dir):
    """分析所有演示数据"""

    demo_dir = Path(demo_dir)
    demo_files = sorted(glob.glob(str(demo_dir / "*.pkl")))

    if len(demo_files) == 0:
        print(f"✗ 未找到演示文件: {demo_dir}")
        return

    print(f"找到 {len(demo_files)} 条演示\n")

    # 统计信息
    valid_demos = []
    invalid_demos = []
    total_steps = []
    total_rewards = []
    success_count = 0

    # 逐个检查
    for i, demo_file in enumerate(demo_files):
        print(f"[{i+1}/{len(demo_files)}] 检查: {Path(demo_file).name}")

        try:
            demo = load_demo(demo_file)

            # 检查有效性
            valid, errors, warnings = check_demo_validity(demo, demo_file)

            if valid:
                valid_demos.append(demo_file)

                # 统计信息
                steps = len(demo["observations"])
                total_steps.append(steps)

                reward = np.sum(demo["rewards"])
                total_rewards.append(reward)

                # 检查成功
                if len(demo["infos"]) > 0:
                    success = demo["infos"][-1].get("success", False)
                    if success:
                        success_count += 1

                # 打印状态
                status = "✓" if success else "○"
                print(f"  {status} 有效: {steps} 步, 奖励={reward:.2f}")

                # 打印警告
                for warning in warnings:
                    print(f"    ⚠️  {warning}")

            else:
                invalid_demos.append(demo_file)
                print(f"  ✗ 无效")
                for error in errors:
                    print(f"    ✗ {error}")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            invalid_demos.append(demo_file)

    # 打印总结
    print("\n" + "=" * 70)
    print("【数据质量报告】")
    print("=" * 70)

    print(f"\n总演示数: {len(demo_files)}")
    print(f"  ✓ 有效: {len(valid_demos)}")
    print(f"  ✗ 无效: {len(invalid_demos)}")

    if len(valid_demos) > 0:
        print(f"\n步数统计:")
        print(f"  平均: {np.mean(total_steps):.1f} 步")
        print(f"  最小: {np.min(total_steps)} 步")
        print(f"  最大: {np.max(total_steps)} 步")
        print(f"  标准差: {np.std(total_steps):.1f} 步")

        print(f"\n奖励统计:")
        print(f"  平均: {np.mean(total_rewards):.2f}")
        print(f"  最小: {np.min(total_rewards):.2f}")
        print(f"  最大: {np.max(total_rewards):.2f}")
        print(f"  标准差: {np.std(total_rewards):.2f}")

        print(f"\n成功率: {success_count}/{len(valid_demos)} ({100*success_count/len(valid_demos):.1f}%)")

        # 检查观测和动作空间
        demo = load_demo(valid_demos[0])
        obs = demo["observations"][0]
        action = demo["actions"][0]

        print(f"\n观测空间:")
        if isinstance(obs, dict):
            for key, val in obs.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k2, v2 in val.items():
                        if isinstance(v2, np.ndarray):
                            print(f"    {k2}: {v2.shape} {v2.dtype}")
                        else:
                            print(f"    {k2}: {type(v2)}")
                elif isinstance(val, np.ndarray):
                    print(f"  {key}: {val.shape} {val.dtype}")
        else:
            print(f"  {type(obs)}")

        print(f"\n动作空间:")
        if isinstance(action, np.ndarray):
            print(f"  shape: {action.shape}")
            print(f"  dtype: {action.dtype}")
            print(f"  范围: [{np.min(action):.3f}, {np.max(action):.3f}]")

    # 建议
    print("\n" + "=" * 70)
    print("【建议】")
    print("=" * 70)

    if len(valid_demos) < 10:
        print("⚠️  演示数量较少（< 10），建议至少采集 10-20 条")
    elif len(valid_demos) < 20:
        print("○ 演示数量一般（10-20），建议增加到 20-30 条以提高性能")
    else:
        print("✓ 演示数量充足")

    if len(valid_demos) > 0:
        success_rate = success_count / len(valid_demos)
        if success_rate < 0.5:
            print("⚠️  成功率较低（< 50%），可能影响BC训练效果")
        elif success_rate < 0.8:
            print("○ 成功率一般（50-80%），建议提高演示质量")
        else:
            print("✓ 成功率良好")

        avg_steps = np.mean(total_steps)
        if avg_steps > 500:
            print("⚠️  平均步数较多（> 500），可能演示不够高效")

        std_steps = np.std(total_steps)
        if std_steps > 100:
            print("○ 步数标准差较大，演示长度差异明显")

    if len(invalid_demos) > 0:
        print(f"\n⚠️  发现 {len(invalid_demos)} 条无效演示，建议删除:")
        for demo_file in invalid_demos:
            print(f"  - {demo_file}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='检查演示数据质量')
    parser.add_argument('--demos-dir', type=str, required=True,
                        help='演示数据目录')
    args = parser.parse_args()

    analyze_demos(args.demos_dir)


if __name__ == '__main__':
    main()
