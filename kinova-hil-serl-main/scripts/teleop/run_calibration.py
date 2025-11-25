#!/usr/bin/env python3
"""
VisionPro 工作空间标定脚本
简化版：只标定一个中心点，使用球形工作空间模型
"""
import sys
import time
import argparse
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.calibrator import WorkspaceCalibrator
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


def print_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("VisionPro 工作空间标定 (单点标定)")
    print("="*60)
    print("\n标定步骤:")
    print("  1. 将手移动到舒适的操作中心位置")
    print("  2. 按 's' 多次采样（建议 5-10 次）")
    print("  3. 按 'c' 保存中心点")
    print("  4. 按 'w' 保存标定文件")
    print("\n按键说明:")
    print("  's' - 添加采样")
    print("  'c' - 保存中心点（计算采样平均值）")
    print("  'x' - 清空当前采样")
    print("  'w' - 写入标定文件")
    print("  'p' - 打印当前手部位姿")
    print("  't' - 打印标定状态")
    print("  '1' - 设置控制半径")
    print("  '2' - 设置死区半径")
    print("  'q' - 退出")
    print("\n工作空间说明:")
    print("  - 死区内 (r < deadzone): 机械臂静止")
    print("  - 工作区 (deadzone < r < control): 速度线性映射")
    print("  - 超出范围 (r > control): 速度饱和")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='VisionPro 空间标定')
    parser.add_argument('--ip', type=str, default='10.31.181.201',
                        help='VisionPro IP 地址')
    parser.add_argument('--left_hand', action='store_true',
                        help='使用左手')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--control_radius', type=float, default=0.25,
                        help='控制半径 (m)')
    parser.add_argument('--deadzone_radius', type=float, default=0.03,
                        help='死区半径 (m)')

    args = parser.parse_args()

    # 配置
    AVP_IP = args.ip
    USE_RIGHT_HAND = not args.left_hand
    
    # 输出路径
    if args.output:
        SAVE_PATH = Path(args.output)
    else:
        SAVE_PATH = Path(__file__).parent.parent / "config" / "calibration.yaml"
    
    # 检查是否覆盖
    overwrite = False
    if SAVE_PATH.exists():
        print(f"\n警告: 标定文件已存在: {SAVE_PATH}")
        response = input("是否覆盖? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
        overwrite = True
        print()
    
    # 初始化
    print("初始化 VisionPro 连接...")
    bridge = VisionProBridge(AVP_IP, use_right_hand=USE_RIGHT_HAND)
    calibrator = WorkspaceCalibrator(
        control_radius=args.control_radius,
        deadzone_radius=args.deadzone_radius
    )
    
    try:
        bridge.start()
        time.sleep(1.0)  # 等待连接稳定
        
        print_instructions()
        
        print(f"当前参数:")
        print(f"  控制半径: {calibrator.control_radius} m")
        print(f"  死区半径: {calibrator.deadzone_radius} m")
        print(f"  输出文件: {SAVE_PATH}")
        print()
        
        # 键盘监听
        with KeyboardMonitor() as kb:
            sample_count = 0
            
            print("准备就绪! 开始标定...\n")
            
            while True:
                # 获取按键
                key = kb.get_key(timeout=0.05)
                
                if not key:
                    continue
                
                # 处理按键
                if key == 'q':
                    print("\n退出标定")
                    break
                    
                elif key == 's':
                    # 添加采样
                    try:
                        position, rotation = bridge.get_hand_relative_to_head()
                        calibrator.add_sample(position, rotation)
                        sample_count += 1
                        print(f"  采样 #{sample_count}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                    except Exception as e:
                        print(f"  采样失败: {e}")
                    
                elif key == 'c':
                    # 保存中心点
                    if calibrator.save_center():
                        sample_count = 0
                    
                elif key == 'x':
                    # 清空采样
                    calibrator.clear_samples()
                    sample_count = 0
                    
                elif key == 'w':
                    # 写入文件
                    if calibrator.is_complete():
                        success = calibrator.save_to_file(SAVE_PATH, overwrite=True)
                        if success:
                            print("\n✓ 标定完成! 文件已保存")
                            print(f"  路径: {SAVE_PATH}")
                    else:
                        print("\n✗ 标定未完成，请先保存中心点 (按 'c')")
                        
                elif key == 'p':
                    # 打印当前位姿
                    try:
                        position, rotation = bridge.get_hand_relative_to_head()
                        print(f"\n当前手部位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
                        
                        # 计算与中心的距离
                        if calibrator.center_position is not None:
                            import numpy as np
                            distance = np.linalg.norm(position - calibrator.center_position)
                            print(f"距离中心: {distance:.4f} m")
                            
                            if distance < calibrator.deadzone_radius:
                                print("状态: 死区内 (机械臂静止)")
                            elif distance < calibrator.control_radius:
                                ratio = (distance - calibrator.deadzone_radius) / (calibrator.control_radius - calibrator.deadzone_radius)
                                print(f"状态: 工作区 (速度 {ratio*100:.1f}%)")
                            else:
                                print("状态: 超出范围 (速度饱和)")
                        print()
                    except Exception as e:
                        print(f"获取位姿失败: {e}")
                        
                elif key == 't':
                    # 打印状态
                    calibrator.print_status()
                    
                elif key == '1':
                    # 设置控制半径
                    print("\n", end='')
                    try:
                        radius = float(input("输入控制半径 (m): "))
                        calibrator.set_workspace_params(control_radius=radius)
                    except ValueError:
                        print("无效输入")
                    except EOFError:
                        pass
                        
                elif key == '2':
                    # 设置死区半径
                    print("\n", end='')
                    try:
                        radius = float(input("输入死区半径 (m): "))
                        calibrator.set_workspace_params(deadzone_radius=radius)
                    except ValueError:
                        print("无效输入")
                    except EOFError:
                        pass
                
    except KeyboardInterrupt:
        print("\n\n用户中断")
        
    finally:
        bridge.stop()
        print("连接已关闭")


if __name__ == "__main__":
    main()