#!/usr/bin/env python3
"""
激活和诊断 twist_controller

由于 Kortex API 不能通过 pip 安装，这个脚本帮助你：
1. 检查 twist_controller 状态
2. 尝试激活控制器
3. 提供其他解决方案
"""

import subprocess
import sys
import time


def run_command(cmd, timeout=5):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "超时", -1
    except Exception as e:
        return "", str(e), -1


def check_controller_state():
    """检查控制器状态"""
    print("=" * 60)
    print("检查控制器状态")
    print("=" * 60)

    stdout, stderr, code = run_command("ros2 control list_controllers")

    if code != 0:
        print(f"✗ 无法列出控制器: {stderr}")
        print("\n可能的原因:")
        print("  1. ROS2 环境未 source")
        print("  2. kortex_bringup 未启动")
        print("\n请先运行:")
        print("  source /opt/ros/humble/setup.bash")
        print("  ros2 launch <your_launch_file>")
        return False

    print(stdout)

    # 检查 twist_controller
    if "twist_controller" in stdout:
        lines = stdout.split('\n')
        for line in lines:
            if 'twist_controller' in line:
                print(f"\n找到 twist_controller: {line}")

                if 'active' in line:
                    print("✓ twist_controller 已激活")
                    return True
                elif 'inactive' in line:
                    print("⚠️  twist_controller 未激活")
                    return False
                else:
                    print("⚠️  twist_controller 状态未知")
                    return False
    else:
        print("\n✗ 未找到 twist_controller")
        print("\n可用的控制器:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"  - {line.strip()}")
        return False


def activate_controller():
    """尝试激活 twist_controller"""
    print("\n" + "=" * 60)
    print("尝试激活 twist_controller")
    print("=" * 60)

    stdout, stderr, code = run_command(
        "ros2 control set_controller_state twist_controller active",
        timeout=10
    )

    if code == 0:
        print("✓ twist_controller 已激活")
        return True
    else:
        print(f"✗ 激活失败: {stderr}")
        return False


def test_twist_publishing():
    """测试发布 twist 命令"""
    print("\n" + "=" * 60)
    print("测试发布 Twist 命令")
    print("=" * 60)

    print("\n正在发布测试命令到 /twist_controller/commands...")
    print("(3秒，vz=0.01 m/s)")

    # 发布一个小的向上速度
    cmd = """
ros2 topic pub --once /twist_controller/commands geometry_msgs/msg/Twist "
linear:
  x: 0.0
  y: 0.0
  z: 0.01
angular:
  x: 0.0
  y: 0.0
  z: 0.0
"
"""

    stdout, stderr, code = run_command(cmd, timeout=5)

    if code == 0:
        print("✓ 命令已发布")
        print("\n请观察机械臂是否有微小移动")
        print("如果没有移动，可能需要:")
        print("  1. 检查控制器配置")
        print("  2. 使用其他控制方式")
    else:
        print(f"✗ 发布失败: {stderr}")


def provide_alternatives():
    """提供替代方案"""
    print("\n" + "=" * 60)
    print("替代方案")
    print("=" * 60)

    print("""
如果 twist_controller 仍然不工作，你有以下选择：

方案1: 使用 joint_trajectory_controller（推荐）
  - 更可靠，Kinova 标准控制方式
  - 需要将笛卡尔速度转换为关节速度
  - 我可以帮你实现这个转换层

方案2: 手动安装 Kortex API
  - 从 GitHub 下载: https://github.com/Kinovarobotics/kortex
  - 按照官方文档安装 Python API
  - 然后可以使用我之前实现的 KortexCommander

方案3: 检查 kortex_bringup 配置
  - twist_controller 可能需要特定的启动参数
  - 检查你的 launch 文件

方案4: 使用 ros2_kortex 的服务
  - kortex_driver 可能提供了直接的速度控制服务
  - 需要检查可用的服务列表

你想尝试哪个方案？
""")


def main():
    print("\n" + "=" * 60)
    print("Twist Controller 诊断和修复工具")
    print("=" * 60)

    # 步骤1: 检查状态
    is_active = check_controller_state()

    if not is_active:
        # 步骤2: 尝试激活
        print("\n按 Enter 尝试激活控制器，或 Ctrl+C 取消...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n已取消")
            provide_alternatives()
            return

        activated = activate_controller()

        if activated:
            # 步骤3: 测试
            print("\n按 Enter 测试控制器，或 Ctrl+C 跳过...")
            try:
                input()
                test_twist_publishing()
            except KeyboardInterrupt:
                print("\n已跳过测试")
        else:
            print("\n控制器激活失败")
    else:
        # 已激活，直接测试
        print("\n控制器已激活，按 Enter 测试，或 Ctrl+C 跳过...")
        try:
            input()
            test_twist_publishing()
        except KeyboardInterrupt:
            print("\n已跳过测试")

    provide_alternatives()


if __name__ == '__main__':
    main()
