#!/usr/bin/env python3
"""
VisionPro 连接测试和诊断工具

用于诊断 VisionPro 数据问题
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from avp_stream import VisionProStreamer
import numpy as np


def test_visionpro_connection(ip: str):
    """测试 VisionPro 连接和数据"""

    print("=" * 60)
    print(f"VisionPro 连接测试 (IP: {ip})")
    print("=" * 60)

    try:
        # 创建 streamer
        print(f"\n1. 创建 VisionProStreamer...")
        streamer = VisionProStreamer(ip=ip, record=False)
        print("   ✓ Streamer 创建成功")

        # 等待数据
        print(f"\n2. 等待接收数据 (10秒)...")

        for i in range(10):
            time.sleep(1)

            # 获取最新数据
            data = streamer.latest

            print(f"\n[{i+1}s] 原始数据检查:")
            print(f"  - 数据类型: {type(data)}")
            print(f"  - 数据键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

            if isinstance(data, dict):
                # 检查 head
                if 'head' in data:
                    head = data['head']
                    print(f"  - head 类型: {type(head)}, 形状: {head.shape if hasattr(head, 'shape') else 'N/A'}")
                    if hasattr(head, 'shape') and len(head) > 0:
                        print(f"    head[0] 形状: {head[0].shape}")
                        print(f"    head[0] 数据:\n{head[0]}")
                else:
                    print(f"  ✗ 缺少 'head' 键")

                # 检查 right_wrist
                if 'right_wrist' in data:
                    wrist = data['right_wrist']
                    print(f"  - right_wrist 类型: {type(wrist)}, 形状: {wrist.shape if hasattr(wrist, 'shape') else 'N/A'}")
                    if hasattr(wrist, 'shape') and len(wrist) > 0:
                        print(f"    right_wrist[0] 形状: {wrist[0].shape}")
                        print(f"    right_wrist[0] 数据:\n{wrist[0]}")
                else:
                    print(f"  ✗ 缺少 'right_wrist' 键")

                # 检查 left_wrist
                if 'left_wrist' in data:
                    wrist = data['left_wrist']
                    print(f"  - left_wrist 类型: {type(wrist)}, 形状: {wrist.shape if hasattr(wrist, 'shape') else 'N/A'}")
                    if hasattr(wrist, 'shape') and len(wrist) > 0:
                        print(f"    left_wrist[0] 形状: {wrist[0].shape}")
                        print(f"    left_wrist[0] 数据:\n{wrist[0]}")
                else:
                    print(f"  ✗ 缺少 'left_wrist' 键")

                # 检查 pinch
                if 'right_pinch' in data:
                    pinch = data['right_pinch']
                    print(f"  - right_pinch: {pinch}")

                if 'left_pinch' in data:
                    pinch = data['left_pinch']
                    print(f"  - left_pinch: {pinch}")

        print("\n" + "=" * 60)
        print("✓ 测试完成")
        print("=" * 60)

        streamer.stop()

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='VisionPro 连接测试')
    parser.add_argument('--ip', type=str, default='192.168.1.125',
                        help='VisionPro IP 地址')

    args = parser.parse_args()

    test_visionpro_connection(args.ip)
