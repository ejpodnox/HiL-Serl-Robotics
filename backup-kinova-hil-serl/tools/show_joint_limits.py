#!/usr/bin/env python3
"""
显示关节限位信息 - 可视化当前位置和配置限位
"""

import rclpy
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinova_rl_env.kinova_env.kinova_interface import KinovaInterface
from kinova_rl_env.kinova_env.config_loader import KinovaConfig


def print_bar(value, min_val, max_val, width=50):
    """打印进度条显示当前位置"""
    # 归一化到 [0, 1]
    range_val = max_val - min_val
    normalized = (value - min_val) / range_val

    # 计算位置
    pos = int(normalized * width)
    pos = max(0, min(width, pos))

    # 构建进度条
    bar = ['─'] * width
    bar[0] = '├'
    bar[-1] = '┤'

    # 标记当前位置
    if 0 <= pos < width:
        bar[pos] = '●'

    # 标记安全区域（中间80%是绿色）
    safe_start = int(width * 0.1)
    safe_end = int(width * 0.9)

    result = ""
    for i, char in enumerate(bar):
        if char == '●':
            # 当前位置
            if safe_start <= i <= safe_end:
                result += f"\033[92m{char}\033[0m"  # 绿色
            else:
                result += f"\033[91m{char}\033[0m"  # 红色
        elif char in ['├', '┤']:
            result += f"\033[90m{char}\033[0m"  # 灰色边界
        else:
            if safe_start <= i <= safe_end:
                result += f"\033[90m{char}\033[0m"  # 灰色
            else:
                result += f"\033[93m{char}\033[0m"  # 黄色警告区

    return result


def main():
    print("\n" + "=" * 80)
    print("关节限位检查工具".center(80))
    print("=" * 80 + "\n")

    # 1. 加载配置
    print("📋 加载机器人配置...")
    kinova_config_path = Path(__file__).parent.parent / 'kinova_rl_env/config/kinova_config.yaml'
    config = KinovaConfig.from_yaml(str(kinova_config_path))

    position_min = np.array(config.robot.joint_limits.position_min)
    position_max = np.array(config.robot.joint_limits.position_max)
    velocity_max = np.array(config.robot.joint_limits.velocity_max)

    print(f"✓ 配置文件: {kinova_config_path}")
    print()

    # 2. 连接机器人
    print("🤖 连接机器人...")
    rclpy.init()
    interface = KinovaInterface(node_name='joint_limit_checker')
    interface.connect()

    import time
    time.sleep(1.0)
    rclpy.spin_once(interface.node, timeout_sec=0.5)

    # 3. 获取当前关节状态
    joint_state = interface.get_joint_state()
    if joint_state is None:
        print("✗ 无法获取关节状态！请确保机器人驱动已启动")
        interface.disconnect()
        return

    q, q_dot = joint_state
    print("✓ 关节状态获取成功\n")

    # 4. 显示每个关节的详细信息
    print("=" * 80)
    print("关节位置限位检查".center(80))
    print("=" * 80 + "\n")

    joint_names = [
        "Joint 1 (Base rotation)",
        "Joint 2 (Shoulder)",
        "Joint 3 (Elbow rotation)",
        "Joint 4 (Forearm)",
        "Joint 5 (Wrist rotation)",
        "Joint 6 (Wrist tilt)",
        "Joint 7 (End effector)"
    ]

    has_violation = False

    for i in range(7):
        print(f"\n{'─' * 80}")
        print(f"关节 {i+1}: {joint_names[i]}")
        print(f"{'─' * 80}")

        current = q[i]
        min_pos = position_min[i]
        max_pos = position_max[i]

        margin_min = current - min_pos
        margin_max = max_pos - current

        # 显示数值
        print(f"  配置限位: [{min_pos:7.3f}, {max_pos:7.3f}] rad")
        print(f"             [{np.rad2deg(min_pos):7.1f}°, {np.rad2deg(max_pos):7.1f}°]")
        print(f"  当前位置:  {current:7.3f} rad ({np.rad2deg(current):7.1f}°)")
        print(f"  速度限制:  {velocity_max[i]:7.3f} rad/s")
        print()

        # 显示余量
        print(f"  下限余量:  {margin_min:7.3f} rad ({np.rad2deg(margin_min):7.1f}°)", end="")
        if margin_min < 0:
            print(" ⚠️  超出下限！", end="")
            has_violation = True
        elif margin_min < 0.3:
            print(" ⚠️  接近下限", end="")
        print()

        print(f"  上限余量:  {margin_max:7.3f} rad ({np.rad2deg(margin_max):7.1f}°)", end="")
        if margin_max < 0:
            print(" ⚠️  超出上限！", end="")
            has_violation = True
        elif margin_max < 0.3:
            print(" ⚠️  接近上限", end="")
        print()

        # 显示进度条
        print(f"\n  位置可视化:")
        print(f"  {print_bar(current, min_pos, max_pos, width=60)}")
        print(f"  ↑Min                          Center                          Max↑")

        # 百分比
        percentage = (current - min_pos) / (max_pos - min_pos) * 100
        print(f"  在范围内的位置: {percentage:.1f}%")

    # 5. 总结
    print("\n" + "=" * 80)
    print("总结".center(80))
    print("=" * 80 + "\n")

    if has_violation:
        print("⚠️  警告：检测到关节位置超出配置限位！")
        print()
        print("可能原因：")
        print("  1. kinova_config.yaml 中的限位设置过于保守")
        print("  2. 机器人实际运动范围 > 配置限位")
        print("  3. 配置文件需要更新以匹配硬件规格")
        print()
        print("建议：")
        print("  1. 检查 Kinova Gen3 官方文档中的关节范围")
        print("  2. 使用 Web 界面移动到极限位置，记录实际值")
        print("  3. 更新 kinova_config.yaml 中的 position_min/max")
        print("  4. 或在配置限位基础上增加 0.1-0.2 rad 的余量")
    else:
        print("✓ 所有关节位置在安全范围内")

        # 检查是否接近极限
        close_to_limit = []
        for i in range(7):
            margin_min = q[i] - position_min[i]
            margin_max = position_max[i] - q[i]
            if margin_min < 0.3 or margin_max < 0.3:
                close_to_limit.append(i+1)

        if close_to_limit:
            print(f"\n⚠️  注意：关节 {close_to_limit} 接近极限 (余量 < 0.3 rad)")
            print("   建议移动到工作空间中心位置再开始遥操作")

    print()
    print("=" * 80)
    print()

    # 清理
    interface.disconnect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
