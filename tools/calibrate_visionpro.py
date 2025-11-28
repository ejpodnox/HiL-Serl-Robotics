#!/usr/bin/env python3
"""
VisionPro 快速标定工具

独立的标定工具，用于生成标定文件
"""

import sys
import time
import numpy as np
from pathlib import Path
import argparse


def run_calibration(vp_ip: str, output_file: str):
    """
    运行标定流程

    Args:
        vp_ip: VisionPro IP 地址
        output_file: 输出标定文件路径
    """
    from vision_pro_control.core.visionpro_bridge import VisionProBridge
    from vision_pro_control.core.calibrator import WorkspaceCalibrator
    from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor

    print("\n" + "=" * 60)
    print("VisionPro 快速标定工具")
    print("=" * 60)
    print(f"VisionPro IP: {vp_ip}")
    print(f"输出文件: {output_file}")

    # 初始化 VisionPro
    print("\n连接 VisionPro...")
    bridge = VisionProBridge(avp_ip=vp_ip, use_right_hand=True)
    bridge.start()
    time.sleep(1.0)
    print("✓ VisionPro 已连接")

    # 初始化标定器
    calibrator = WorkspaceCalibrator(
        control_radius=0.25,      # 控制半径 25cm
        deadzone_radius=0.10      # 死区半径 10cm（避免微小抖动）
    )

    print("\n" + "=" * 60)
    print("【标定流程】")
    print("=" * 60)
    print("目标：确定一个舒适的操作中心位置")
    print()
    print("步骤:")
    print("  1. 将手移动到你认为舒适的操作中心位置")
    print("  2. 按 's' 键采样该位置（建议采样 5-10 次）")
    print("  3. 按 'c' 键保存中心点")
    print("  4. 按 Enter 确认完成")
    print()
    print("按键说明:")
    print("  's'     - 采样当前手部位置")
    print("  'c'     - 保存中心点（采样的平均值）")
    print("  'p'     - 打印当前位置信息")
    print("  'Enter' - 完成标定")
    print("  'q'     - 退出")
    print("=" * 60)

    sample_count = 0

    try:
        with KeyboardMonitor() as kb:
            while True:
                key = kb.get_key(timeout=0.05)

                if not key:
                    continue

                if key == 'q':
                    print("\n退出标定")
                    bridge.stop()
                    return False

                elif key == 's':
                    # 采样
                    try:
                        position, rotation = bridge.get_hand_relative_to_head()
                        calibrator.add_sample(position, rotation)
                        sample_count += 1
                        print(f"✓ 采样 #{sample_count}: "
                              f"位置=[{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}]")
                    except Exception as e:
                        print(f"✗ 采样失败: {e}")

                elif key == 'c':
                    # 保存中心点
                    if calibrator.save_center():
                        sample_count = 0
                        print("✓ 中心点已保存")
                        print("  按 Enter 确认完成标定")
                    else:
                        print("✗ 需要至少 1 个采样点")

                elif key == 'p':
                    # 打印当前位置
                    try:
                        position, rotation = bridge.get_hand_relative_to_head()
                        print(f"\n当前位置: [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}]")

                        if calibrator.center_position is not None:
                            distance = np.linalg.norm(position - calibrator.center_position)
                            print(f"距离中心: {distance:.3f} m ({distance*100:.1f} cm)")
                    except Exception as e:
                        print(f"获取位置失败: {e}")

                elif key == '\n' or key == '\r':
                    # Enter 确认
                    if calibrator.is_complete():
                        # 保存到文件
                        output_path = Path(output_file)
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        calibrator.save_to_file(output_path, overwrite=True)

                        print("\n" + "=" * 60)
                        print("✓ 标定完成！")
                        print("=" * 60)
                        print(f"标定文件已保存: {output_path}")
                        print()
                        print("标定信息:")
                        print(f"  中心位置: {calibrator.center_position}")
                        print(f"  控制半径: {calibrator.control_radius} m")
                        print(f"  死区半径: {calibrator.deadzone_radius} m")
                        print("=" * 60)

                        bridge.stop()
                        return True
                    else:
                        print("\n✗ 请先完成标定:")
                        print("  1. 按 's' 采样手部位置")
                        print("  2. 按 'c' 保存中心点")

    except KeyboardInterrupt:
        print("\n\n用户中断")
        bridge.stop()
        return False

    except Exception as e:
        print(f"\n✗ 标定失败: {e}")
        import traceback
        traceback.print_exc()
        bridge.stop()
        return False


def main():
    # 读取默认 VisionPro IP
    default_vp_ip = '192.168.1.125'
    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml('kinova_rl_env/config/kinova_config.yaml')
        if hasattr(config, 'visionpro') and hasattr(config.visionpro, 'ip'):
            default_vp_ip = config.visionpro.ip
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='VisionPro 快速标定')
    parser.add_argument('--vp-ip', type=str, default=default_vp_ip,
                        help=f'VisionPro IP (默认: {default_vp_ip})')
    parser.add_argument('--output', type=str,
                        default='vision_pro_control/config/calibration.yaml',
                        help='输出标定文件路径')
    args = parser.parse_args()

    success = run_calibration(args.vp_ip, args.output)

    if success:
        print("\n✓ 可以开始数据采集了！")
        print("  运行: python vision_pro_control/record_teleop_demos.py")
        return 0
    else:
        print("\n✗ 标定未完成")
        return 1


if __name__ == '__main__':
    sys.exit(main())
