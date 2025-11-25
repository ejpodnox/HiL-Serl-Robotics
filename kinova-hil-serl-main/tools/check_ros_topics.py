#!/usr/bin/env python3
"""
ROS2 话题和服务检查工具

用于诊断 Kinova 机械臂相关的 ROS2 通信
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
        return result.stdout
    except subprocess.TimeoutExpired:
        return "命令超时"
    except Exception as e:
        return f"错误: {e}"


def check_topics():
    """检查关键话题"""
    print("\n" + "=" * 60)
    print("【ROS2 话题检查】")
    print("=" * 60)

    topics_to_check = [
        '/joint_states',
        '/twist_controller/commands',
        '/velocity_controller/commands',
        '/robotiq_gripper_controller/gripper_cmd',
        '/tf',
        '/tf_static'
    ]

    print("\n运行: ros2 topic list")
    all_topics = run_command("ros2 topic list")
    print(all_topics)

    print("\n关键话题检查:")
    for topic in topics_to_check:
        if topic in all_topics:
            print(f"  ✓ {topic}")
        else:
            print(f"  ✗ {topic} (不存在)")

    return all_topics


def check_nodes():
    """检查 ROS2 节点"""
    print("\n" + "=" * 60)
    print("【ROS2 节点检查】")
    print("=" * 60)

    print("\n运行: ros2 node list")
    nodes = run_command("ros2 node list")
    print(nodes)

    # 检查关键节点
    key_nodes = [
        'kortex_driver',
        'twist_controller',
        'velocity_controller',
        'robotiq',
        'gripper'
    ]

    print("\n关键节点检查:")
    for node in key_nodes:
        if node in nodes.lower():
            print(f"  ✓ 包含 '{node}' 的节点")
        else:
            print(f"  ⚠️  未找到包含 '{node}' 的节点")

    return nodes


def check_actions():
    """检查 Action 服务器"""
    print("\n" + "=" * 60)
    print("【Action 服务器检查】")
    print("=" * 60)

    print("\n运行: ros2 action list")
    actions = run_command("ros2 action list")
    print(actions)

    if 'gripper' in actions.lower():
        print("\n✓ 找到夹爪相关的 Action 服务器")
    else:
        print("\n✗ 未找到夹爪 Action 服务器")
        print("  提示: 夹爪控制器可能未启动")

    return actions


def check_tf_frames():
    """检查 TF 坐标系"""
    print("\n" + "=" * 60)
    print("【TF 坐标系检查】")
    print("=" * 60)

    # 先列出所有可用的 frames
    print("\n运行: ros2 run tf2_ros tf2_echo --print-tf-tree")
    print("(3秒超时)")
    tree_output = run_command("timeout 3 ros2 run tf2_tools view_frames.py 2>&1 | head -20")

    # 使用更简单的方法：监听 /tf 话题
    print("\n检查 /tf 话题是否有数据...")
    tf_output = run_command("timeout 2 ros2 topic echo /tf --once 2>&1")

    if "transforms" in tf_output or "frame_id" in tf_output:
        print("✓ /tf 话题有数据")
        # 尝试提取 frame 名称
        lines = tf_output.split('\n')
        frames = []
        for line in lines:
            if 'frame_id:' in line:
                frame = line.split('frame_id:')[-1].strip().strip("'\"")
                if frame and frame not in frames:
                    frames.append(frame)
        if frames:
            print(f"  发现的 frames: {', '.join(frames[:5])}")
    else:
        print("✗ /tf 话题无数据或超时")
        print("  提示: kortex_bringup 可能未正确启动")

    print("\n运行: ros2 run tf2_ros tf2_echo base_link tool_frame")
    print("(3秒超时)")

    output = run_command("timeout 3 ros2 run tf2_ros tf2_echo base_link tool_frame")

    if "Translation" in output:
        print("✓ TF 变换正常")
        print(output[:200] + "...")  # 只显示前 200 字符
    else:
        print("✗ TF 变换获取失败")
        print(output)
        print("\n可能的原因:")
        print("  1. kortex_bringup 未启动或启动失败")
        print("  2. 机械臂未正确连接")
        print("  3. frame 名称可能不是 'base_link' 或 'tool_frame'")
        print("  4. TF 发布器需要更多时间初始化")


def check_gripper_topics():
    """详细检查夹爪相关话题"""
    print("\n" + "=" * 60)
    print("【夹爪话题详细检查】")
    print("=" * 60)

    # 查找所有包含 gripper 的话题
    print("\n查找所有包含 'gripper' 的话题:")
    output = run_command("ros2 topic list | grep -i gripper")
    if output.strip():
        print(output)
    else:
        print("  ✗ 未找到包含 'gripper' 的话题")

    # 查找所有包含 robotiq 的话题
    print("\n查找所有包含 'robotiq' 的话题:")
    output = run_command("ros2 topic list | grep -i robotiq")
    if output.strip():
        print(output)
    else:
        print("  ✗ 未找到包含 'robotiq' 的话题")

    # 检查常见的夹爪话题名称
    possible_gripper_topics = [
        '/gripper_controller/gripper_cmd',
        '/robotiq_gripper_controller/gripper_cmd',
        '/gripper/command',
        '/gripper_cmd',
    ]

    print("\n检查可能的夹爪话题:")
    all_topics = run_command("ros2 topic list")
    for topic in possible_gripper_topics:
        if topic in all_topics:
            print(f"  ✓ {topic}")
            # 检查话题类型
            topic_type = run_command(f"ros2 topic type {topic}")
            print(f"    类型: {topic_type.strip()}")
        else:
            print(f"  ✗ {topic}")


def provide_solutions():
    """提供解决方案"""
    print("\n" + "=" * 60)
    print("【常见问题解决方案】")
    print("=" * 60)

    print("""
1. 如果夹爪 Action 服务器不可用:
   - 检查是否启动了夹爪控制器
   - 可能需要在 kortex_bringup 启动文件中启用夹爪
   - 临时解决：修改代码，让夹爪失败时不影响其他功能

2. 如果 TF 坐标系不存在:
   - 确保 kortex_bringup 正在运行
   - 检查机械臂连接
   - 尝试重启 kortex_bringup

3. 如果关键话题缺失:
   - 检查 kortex_bringup 启动参数
   - 确认使用正确的启动文件（gen3.launch.py）
   - 检查 robot_ip 是否正确
""")


def main():
    print("=" * 60)
    print("ROS2 话题和服务检查工具")
    print("=" * 60)
    print("\n此工具将检查 Kinova 机械臂相关的 ROS2 通信")
    print("请确保已启动 kortex_bringup")
    print()

    input("按 Enter 继续...")

    # 执行检查
    check_nodes()
    check_topics()
    check_actions()
    check_gripper_topics()
    check_tf_frames()
    provide_solutions()

    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
