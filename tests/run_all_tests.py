#!/usr/bin/env python3
"""
运行所有测试

解耦设计：每个测试独立运行，失败不影响后续测试。
"""

import argparse
import subprocess
import sys
from pathlib import Path


class TestRunner:
    """测试运行器"""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {}
        self.test_dir = Path(__file__).parent

    def run_test(self, test_name, test_script, args=None):
        """运行单个测试"""
        print("\n" + "=" * 80)
        print(f"运行测试: {test_name}")
        print("=" * 80)

        cmd = [sys.executable, str(self.test_dir / test_script)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True,
                timeout=60
            )

            success = (result.returncode == 0)
            self.results[test_name] = success

            if self.verbose or not success:
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)

            if success:
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败 (返回码: {result.returncode})")

            return success

        except subprocess.TimeoutExpired:
            print(f"✗ {test_name} 超时")
            self.results[test_name] = False
            return False
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
            self.results[test_name] = False
            return False

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("【测试总结】")
        print("=" * 80)

        passed = sum(1 for r in self.results.values() if r)
        failed = sum(1 for r in self.results.values() if not r)
        total = len(self.results)

        for test_name, result in self.results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name:30s}: {status}")

        print("\n" + "-" * 80)
        print(f"总计: {total} | 通过: {passed} | 失败: {failed}")
        print("-" * 80)

        if failed == 0:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print(f"\n⚠️  {failed} 个测试失败")
            return 1


def main():
    parser = argparse.ArgumentParser(description='运行所有测试')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    parser.add_argument('--test', type=str,
                        choices=['visionpro', 'kinova', 'camera', 'env',
                                'data', 'training', 'all'],
                        default='all',
                        help='选择要运行的测试')
    parser.add_argument('--skip-hardware', action='store_true',
                        help='跳过所有硬件连接测试')

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    # 定义测试列表
    tests = []

    if args.test in ['visionpro', 'all']:
        test_args = ['--skip-connection'] if args.skip_hardware else []
        tests.append(('VisionPro 连接', 'test_visionpro_connection.py', test_args))

    if args.test in ['kinova', 'all']:
        test_args = ['--skip-connection'] if args.skip_hardware else []
        tests.append(('Kinova 连接', 'test_kinova_connection.py', test_args))

    if args.test in ['camera', 'all']:
        tests.append(('相机模块', 'test_camera.py', ['--backend', 'dummy']))

    if args.test in ['env', 'all']:
        tests.append(('Gym 环境', 'test_environment.py', []))

    if args.test in ['data', 'all']:
        tests.append(('数据流程', 'test_data_pipeline.py', []))

    if args.test in ['training', 'all']:
        tests.append(('训练流程', 'test_training.py', ['--steps', '10']))

    # 运行测试
    print("开始运行测试...")
    print(f"跳过硬件测试: {args.skip_hardware}")
    print(f"测试数量: {len(tests)}")

    for test_name, test_script, test_args in tests:
        runner.run_test(test_name, test_script, test_args)

    # 打印总结
    return runner.print_summary()


if __name__ == '__main__':
    exit(main())
