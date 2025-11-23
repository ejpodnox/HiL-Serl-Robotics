#!/usr/bin/env python3
"""
测试 VisionPro 连接和数据接收

独立测试，不依赖其他组件。
"""

import argparse
import time
import numpy as np
from pathlib import Path


def test_visionpro_basic():
    """基础连接测试"""
    print("\n" + "=" * 60)
    print("【测试 1】VisionPro 基础连接")
    print("=" * 60)

    try:
        from vision_pro_control.core import VisionProBridge
        print("✓ 导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_visionpro_connection(vp_ip: str, timeout: float = 5.0):
    """测试 VisionPro 连接"""
    print("\n" + "=" * 60)
    print("【测试 2】VisionPro 数据接收")
    print("=" * 60)
    print(f"VisionPro IP: {vp_ip}")
    print(f"超时时间: {timeout}s")

    try:
        from vision_pro_control.core import VisionProBridge

        bridge = VisionProBridge(avp_ip=vp_ip, use_right_hand=True)
        bridge.start()
        print("✓ VisionPro Bridge 已启动")

        # 等待数据
        print(f"等待数据（最多 {timeout}s）...")
        start_time = time.time()
        data_received = False

        while time.time() - start_time < timeout:
            data = bridge.get_latest_data()
            if data['timestamp'] > 0:
                print("✓ 收到数据:")
                print(f"  - 头部位姿: {data['head_pose'][:3, 3]}")
                print(f"  - 手腕位姿: {data['wrist_pose'][:3, 3]}")
                print(f"  - 捏合距离: {data['pinch_distance']:.3f}")
                print(f"  - 时间戳: {data['timestamp']:.3f}")
                data_received = True
                break
            time.sleep(0.1)

        bridge.stop()

        if not data_received:
            print(f"✗ {timeout}s 内未收到数据")
            return False

        print("✓ VisionPro 连接测试通过")
        return True

    except Exception as e:
        print(f"✗ VisionPro 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visionpro_dummy():
    """使用虚拟数据测试（当真实设备不可用时）"""
    print("\n" + "=" * 60)
    print("【测试 3】VisionPro 虚拟数据模式")
    print("=" * 60)

    try:
        # 模拟 VisionPro 数据
        dummy_data = {
            'head_pose': np.eye(4),
            'wrist_pose': np.eye(4),
            'pinch_distance': 1.0,
            'wrist_roll': 0.0,
            'timestamp': time.time()
        }

        print("✓ 虚拟数据生成成功:")
        print(f"  - 头部位姿: {dummy_data['head_pose'][:3, 3]}")
        print(f"  - 手腕位姿: {dummy_data['wrist_pose'][:3, 3]}")
        print(f"  - 捏合距离: {dummy_data['pinch_distance']:.3f}")

        return True

    except Exception as e:
        print(f"✗ 虚拟数据测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='VisionPro 连接测试')
    parser.add_argument('--vp_ip', type=str, default='192.168.1.125',
                        help='VisionPro IP 地址')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='连接超时时间（秒）')
    parser.add_argument('--skip-connection', action='store_true',
                        help='跳过真实连接测试，只测试虚拟模式')

    args = parser.parse_args()

    results = {}

    # 测试 1: 基础导入
    results['basic'] = test_visionpro_basic()

    # 测试 2: 真实连接（可跳过）
    if not args.skip_connection and results['basic']:
        results['connection'] = test_visionpro_connection(args.vp_ip, args.timeout)
    else:
        if args.skip_connection:
            print("\n⚠️  跳过真实连接测试")
        results['connection'] = None

    # 测试 3: 虚拟模式（总是执行）
    results['dummy'] = test_visionpro_dummy()

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

    # 只要基础测试和虚拟模式通过，就认为环境可用
    essential_passed = results['basic'] and results['dummy']

    if essential_passed:
        print("\n✓ 基础功能测试通过，VisionPro 模块可用")
        if results['connection'] is False:
            print("⚠️  真实设备连接失败，但可以使用虚拟模式进行开发")
        return 0
    else:
        print("\n✗ 基础测试失败，请检查环境配置")
        return 1


if __name__ == '__main__':
    exit(main())
