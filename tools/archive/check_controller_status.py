#!/usr/bin/env python3
"""
检查 ROS2 控制器状态

诊断为什么机械臂收到命令但不移动
"""

import subprocess
import sys


def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3.0
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "命令超时", -1
    except Exception as e:
        return "", f"错误: {e}", -1


def main():
    print("=" * 60)
    print("ROS2 控制器状态检查")
    print("=" * 60)

    # 1. 列出所有控制器
    print("\n【步骤1: 列出所有控制器】")
    print("运行: ros2 control list_controllers")
    stdout, stderr, code = run_command("ros2 control list_controllers")

    if code == 0 and stdout:
        print(stdout)

        # 检查关键控制器的状态
        if "twist_controller" in stdout:
            if "active" in stdout and "twist_controller" in stdout.split("active")[0].split("\n")[-1]:
                print("✓ twist_controller 已激活")
            elif "inactive" in stdout:
                print("⚠️  twist_controller 存在但未激活")
                print("   需要激活控制器！")
            else:
                print("⚠️  twist_controller 状态未知")
        else:
            print("✗ 未找到 twist_controller")
    else:
        print(f"✗ 命令失败: {stderr}")

    # 2. 检查 twist_controller 的详细信息
    print("\n【步骤2: 检查 twist_controller 配置】")
    print("运行: ros2 control list_hardware_interfaces")
    stdout, stderr, code = run_command("ros2 control list_hardware_interfaces")

    if code == 0 and stdout:
        print(stdout[:500])  # 只显示前500字符
    else:
        print(f"命令失败: {stderr}")

    # 3. 监听 twist_controller 话题
    print("\n【步骤3: 检查 /twist_controller/commands 话题】")
    print("运行: ros2 topic info /twist_controller/commands")
    stdout, stderr, code = run_command("ros2 topic info /twist_controller/commands")

    if code == 0:
        print(stdout)
    else:
        print(f"命令失败: {stderr}")

    # 4. 检查话题发布和订阅
    print("\n【步骤4: 检查话题订阅者】")
    print("运行: ros2 topic info /twist_controller/commands -v")
    stdout, stderr, code = run_command("ros2 topic info /twist_controller/commands -v")

    if code == 0:
        print(stdout)
        if "Subscription count: 0" in stdout:
            print("\n⚠️  警告: 没有节点订阅此话题！")
            print("   这意味着发送的命令没有被任何控制器接收")
    else:
        print(f"命令失败: {stderr}")

    # 5. 检查 controller_manager 状态
    print("\n【步骤5: 检查 controller_manager】")
    print("运行: ros2 node info /controller_manager")
    stdout, stderr, code = run_command("ros2 node info /controller_manager")

    if code == 0:
        print(stdout[:800])
    else:
        print(f"命令失败: {stderr}")

    # 6. 提供解决方案
    print("\n" + "=" * 60)
    print("【常见问题和解决方案】")
    print("=" * 60)
    print("""
1. 如果 twist_controller 未激活:
   解决: ros2 control set_controller_state twist_controller activate

2. 如果没有订阅者:
   - 检查控制器配置文件
   - 可能需要使用不同的话题名称
   - 尝试: ros2 topic list | grep -i twist
   - 尝试: ros2 topic list | grep -i velocity

3. 如果控制器不存在:
   - 检查 kortex_bringup 的启动参数
   - 可能需要使用 joint_trajectory_controller 而不是 twist_controller
   - 查看可用的控制器: ros2 control list_controllers

4. 替代方案 - 使用 joint_trajectory_controller:
   - Kinova 机械臂通常使用关节轨迹控制
   - 话题: /joint_trajectory_controller/joint_trajectory
   - 需要将笛卡尔速度转换为关节速度

5. 替代方案 - 使用 Kortex API:
   - 直接使用 Kortex API 发送笛卡尔速度命令
   - 绕过 ROS2 控制器层
""")

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
