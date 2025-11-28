#!/usr/bin/env python3
"""
夹爪通信测试程序

功能：
1. 测试夹爪通信是否正常
2. 测试夹爪开合控制
3. 测试夹爪状态反馈

使用方法:
    python tools/test_gripper.py
"""

import argparse
import rclpy
import time
import numpy as np
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinova_rl_env.kinova_env.kinova_interface import KinovaInterface


class GripperTester:
    """夹爪测试器"""

    def __init__(self):
        """初始化夹爪测试器"""
        # 初始化 ROS2
        if not rclpy.ok():
            rclpy.init()

        # 初始化 KinovaInterface
        self.interface = KinovaInterface(node_name='gripper_tester')
        self.interface.connect()

        print("✓ 夹爪测试器初始化完成")
        print(f"  夹爪命令话题: {self.interface.gripper_command_topic}")

    def test_communication(self):
        """测试夹爪通信"""
        print("\n" + "=" * 60)
        print("【测试 1: 夹爪通信测试】")
        print("=" * 60)

        try:
            # 发送测试命令
            print("发送测试命令: 位置=0.5 (半开状态)")
            self.interface.send_gripper_command(0.5)
            time.sleep(0.5)

            # 检查是否有错误
            rclpy.spin_once(self.interface.node, timeout_sec=0.1)

            print("✓ 夹爪命令发送成功，通信正常")
            return True

        except Exception as e:
            print(f"✗ 夹爪通信测试失败: {e}")
            return False

    def test_open_close(self, num_cycles=3, delay=2.0):
        """
        测试夹爪开合

        Args:
            num_cycles: 开合循环次数
            delay: 每次开合之间的延迟（秒）
        """
        print("\n" + "=" * 60)
        print(f"【测试 2: 夹爪开合测试】(共 {num_cycles} 次循环)")
        print("=" * 60)

        positions = [
            (0.0, "完全打开"),
            (1.0, "完全闭合"),
        ]

        try:
            for cycle in range(num_cycles):
                print(f"\n>>> 循环 {cycle + 1}/{num_cycles}")

                for position, description in positions:
                    print(f"  命令: 位置={position:.1f} ({description})")
                    self.interface.send_gripper_command(position)

                    # 等待夹爪运动
                    time.sleep(delay)

                    # Spin 接收可能的反馈
                    rclpy.spin_once(self.interface.node, timeout_sec=0.1)

                    # 获取状态（如果有反馈）
                    current_state = self.interface.get_gripper_state()
                    print(f"  当前状态: {current_state:.2f}")

            print("\n✓ 夹爪开合测试完成")
            return True

        except Exception as e:
            print(f"\n✗ 夹爪开合测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_gradual_movement(self, num_steps=10, step_delay=0.5):
        """
        测试夹爪渐进运动

        Args:
            num_steps: 步数
            step_delay: 每步之间的延迟（秒）
        """
        print("\n" + "=" * 60)
        print(f"【测试 3: 夹爪渐进运动测试】({num_steps} 步)")
        print("=" * 60)

        try:
            # 从 0.0 到 1.0 渐进
            print("\n>>> 阶段 1: 从打开到闭合")
            for i in range(num_steps + 1):
                position = i / num_steps
                print(f"  [{i:2d}/{num_steps}] 位置={position:.2f}")
                self.interface.send_gripper_command(position)
                time.sleep(step_delay)
                rclpy.spin_once(self.interface.node, timeout_sec=0.01)

            # 从 1.0 到 0.0 渐进
            print("\n>>> 阶段 2: 从闭合到打开")
            for i in range(num_steps + 1):
                position = 1.0 - (i / num_steps)
                print(f"  [{i:2d}/{num_steps}] 位置={position:.2f}")
                self.interface.send_gripper_command(position)
                time.sleep(step_delay)
                rclpy.spin_once(self.interface.node, timeout_sec=0.01)

            print("\n✓ 夹爪渐进运动测试完成")
            return True

        except Exception as e:
            print(f"\n✗ 夹爪渐进运动测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_interactive(self):
        """交互式夹爪测试"""
        print("\n" + "=" * 60)
        print("【测试 4: 交互式夹爪测试】")
        print("=" * 60)
        print("按键说明:")
        print("  'o' - 打开夹爪 (位置=0.0)")
        print("  'c' - 闭合夹爪 (位置=1.0)")
        print("  'h' - 半开夹爪 (位置=0.5)")
        print("  '0-9' - 设置位置 (0=完全打开, 9=完全闭合)")
        print("  'q' - 退出")
        print("=" * 60)

        from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor

        try:
            with KeyboardMonitor() as kb:
                while True:
                    key = kb.get_key(timeout=0.1)

                    if not key:
                        continue

                    if key == 'q':
                        print("\n退出交互式测试")
                        break

                    elif key == 'o':
                        position = 0.0
                        print(f"命令: 打开夹爪 (位置={position})")
                        self.interface.send_gripper_command(position)

                    elif key == 'c':
                        position = 1.0
                        print(f"命令: 闭合夹爪 (位置={position})")
                        self.interface.send_gripper_command(position)

                    elif key == 'h':
                        position = 0.5
                        print(f"命令: 半开夹爪 (位置={position})")
                        self.interface.send_gripper_command(position)

                    elif key.isdigit():
                        position = int(key) / 9.0  # 0-9 映射到 0.0-1.0
                        print(f"命令: 设置位置={position:.2f}")
                        self.interface.send_gripper_command(position)

                    # Spin 接收反馈
                    rclpy.spin_once(self.interface.node, timeout_sec=0.01)

                    # 显示当前状态
                    current_state = self.interface.get_gripper_state()
                    print(f"  当前状态: {current_state:.2f}")

            return True

        except KeyboardInterrupt:
            print("\n用户中断")
            return False

        except Exception as e:
            print(f"\n✗ 交互式测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """清理资源"""
        # 将夹爪设置为安全状态（半开）
        print("\n设置夹爪为安全状态 (位置=0.0)")
        self.interface.send_gripper_command(0.0)
        time.sleep(0.5)

        # 断开连接
        self.interface.disconnect()
        rclpy.shutdown()
        print("✓ 已清理资源")


def main():
    parser = argparse.ArgumentParser(description='夹爪通信测试程序')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'communication', 'open_close', 'gradual', 'interactive'],
                        help='测试模式: all(所有测试), communication(通信), open_close(开合), '
                             'gradual(渐进), interactive(交互式)')
    parser.add_argument('--cycles', type=int, default=3,
                        help='开合测试的循环次数')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='开合测试的延迟时间（秒）')
    parser.add_argument('--steps', type=int, default=10,
                        help='渐进测试的步数')

    args = parser.parse_args()

    print("=" * 60)
    print("夹爪通信测试程序")
    print("=" * 60)
    print(f"测试模式: {args.mode}")
    print("=" * 60)

    tester = GripperTester()

    try:
        if args.mode == 'all' or args.mode == 'communication':
            tester.test_communication()

        if args.mode == 'all' or args.mode == 'open_close':
            tester.test_open_close(num_cycles=args.cycles, delay=args.delay)

        if args.mode == 'all' or args.mode == 'gradual':
            tester.test_gradual_movement(num_steps=args.steps)

        if args.mode == 'interactive':
            tester.test_interactive()

        print("\n" + "=" * 60)
        print("✓ 所有测试完成")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n用户中断测试")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        tester.cleanup()


if __name__ == '__main__':
    main()
