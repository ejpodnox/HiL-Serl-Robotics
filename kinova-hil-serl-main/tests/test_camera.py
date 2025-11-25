#!/usr/bin/env python3
"""
测试相机模块

独立测试，支持多种相机后端。
"""

import argparse
import time
import numpy as np
import cv2


def test_camera_imports():
    """测试相机模块导入"""
    print("\n" + "=" * 60)
    print("【测试 1】相机模块导入")
    print("=" * 60)

    try:
        from kinova_rl_env import (
            CameraInterface,
            RealSenseCamera,
            WebCamera,
            DummyCamera
        )
        print("✓ CameraInterface 导入成功")
        print("✓ RealSenseCamera 导入成功")
        print("✓ WebCamera 导入成功")
        print("✓ DummyCamera 导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dummy_camera():
    """测试虚拟相机"""
    print("\n" + "=" * 60)
    print("【测试 2】虚拟相机（DummyCamera）")
    print("=" * 60)

    try:
        from kinova_rl_env import DummyCamera

        camera = DummyCamera(image_size=(128, 128))
        camera.start()
        print("✓ DummyCamera 启动成功")

        # 获取图像
        for i in range(3):
            image = camera.get_image()
            print(f"✓ 获取图像 #{i+1}: shape={image.shape}, dtype={image.dtype}")
            assert image.shape == (128, 128, 3), "图像尺寸错误"

        camera.stop()
        print("✓ DummyCamera 测试通过")
        return True

    except Exception as e:
        print(f"✗ DummyCamera 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webcam(device_id: int = 0, test_duration: float = 2.0):
    """测试 WebCam"""
    print("\n" + "=" * 60)
    print("【测试 3】WebCam")
    print("=" * 60)
    print(f"设备 ID: {device_id}")

    try:
        from kinova_rl_env import WebCamera

        camera = WebCamera(device_id=device_id, image_size=(128, 128))
        camera.start()
        print("✓ WebCamera 启动成功")

        # 测试获取图像
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < test_duration:
            image = camera.get_image()
            frame_count += 1
            time.sleep(0.1)

        camera.stop()

        fps = frame_count / test_duration
        print(f"✓ WebCamera 测试通过")
        print(f"  - 帧数: {frame_count}")
        print(f"  - 帧率: {fps:.1f} FPS")

        return True

    except Exception as e:
        print(f"✗ WebCamera 测试失败: {e}")
        print("提示: 请确保有可用的摄像头")
        import traceback
        traceback.print_exc()
        return False


def test_realsense(serial_number: str = None, test_duration: float = 2.0):
    """测试 RealSense 相机"""
    print("\n" + "=" * 60)
    print("【测试 4】RealSense 相机")
    print("=" * 60)

    try:
        from kinova_rl_env import RealSenseCamera

        camera = RealSenseCamera(
            serial_number=serial_number,
            image_size=(128, 128)
        )
        camera.start()
        print("✓ RealSenseCamera 启动成功")

        # 测试获取图像
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < test_duration:
            image = camera.get_image()
            frame_count += 1
            time.sleep(0.1)

        camera.stop()

        fps = frame_count / test_duration
        print(f"✓ RealSenseCamera 测试通过")
        print(f"  - 帧数: {frame_count}")
        print(f"  - 帧率: {fps:.1f} FPS")

        return True

    except Exception as e:
        print(f"✗ RealSenseCamera 测试失败: {e}")
        print("提示: 请确保 RealSense 相机已连接且驱动已安装")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='相机模块测试')
    parser.add_argument('--backend', type=str,
                        choices=['dummy', 'webcam', 'realsense', 'all'],
                        default='dummy',
                        help='测试的相机后端')
    parser.add_argument('--webcam-id', type=int, default=0,
                        help='WebCam 设备 ID')
    parser.add_argument('--realsense-serial', type=str, default=None,
                        help='RealSense 序列号')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='视频测试时长（秒）')

    args = parser.parse_args()

    results = {}

    # 测试 1: 导入
    results['imports'] = test_camera_imports()

    if not results['imports']:
        print("\n✗ 导入失败，无法继续测试")
        return 1

    # 测试 2: DummyCamera（总是执行）
    results['dummy'] = test_dummy_camera()

    # 测试 3: WebCam（可选）
    if args.backend in ['webcam', 'all']:
        results['webcam'] = test_webcam(args.webcam_id, args.duration)
    else:
        results['webcam'] = None

    # 测试 4: RealSense（可选）
    if args.backend in ['realsense', 'all']:
        results['realsense'] = test_realsense(args.realsense_serial, args.duration)
    else:
        results['realsense'] = None

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

    # 只要 dummy 相机可用，就认为环境可用
    essential_passed = results['dummy']

    if essential_passed:
        print("\n✓ 基础功能测试通过，相机模块可用")
        print("提示: 可以使用 DummyCamera 进行开发，无需真实相机")
        return 0
    else:
        print("\n✗ 基础测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
