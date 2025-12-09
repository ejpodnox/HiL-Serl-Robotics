#!/usr/bin/env python3
"""
ROS2 系统诊断工具

用于诊断Kinova机器人的ROS2连接问题：
- 检查节点和话题
- 检查/joint_states是否有数据
- 检查夹爪话题
- 检查控制器状态
"""

import sys
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import subprocess


class ROS2Diagnostics(Node):
    """ROS2诊断节点"""

    def __init__(self):
        super().__init__('ros2_diagnostics')

        # 标志
        self.joint_state_received = False
        self.joint_state_count = 0
        self.last_joint_state = None

        # 订阅 /joint_states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        print("✓ ROS2诊断节点已启动")

    def joint_state_callback(self, msg):
        """接收joint_states消息"""
        self.joint_state_received = True
        self.joint_state_count += 1
        self.last_joint_state = msg


def run_command(cmd):
    """运行shell命令并返回输出"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        return f"错误: {e}"


def check_ros2_topics():
    """检查ROS2话题"""
    print("\n" + "=" * 60)
    print("【检查 ROS2 话题】")
    print("=" * 60)

    # 获取话题列表
    output = run_command("ros2 topic list")
    if output:
        topics = output.split('\n')
        print(f"✓ 找到 {len(topics)} 个话题:")
        for topic in topics:
            print(f"  - {topic}")

        # 检查关键话题
        print("\n关键话题检查:")
        key_topics = [
            '/joint_states',
            '/joint_trajectory_controller/joint_trajectory',
            '/gripper_controller/gripper_cmd',
            '/velocity_controller/commands'
        ]

        for topic in key_topics:
            if topic in topics:
                print(f"  ✓ {topic}")
                # 获取话题信息
                info = run_command(f"ros2 topic info {topic}")
                if info:
                    print(f"    {info.replace(chr(10), chr(10) + '    ')}")
            else:
                print(f"  ✗ {topic} (缺失)")
    else:
        print("✗ 无法获取话题列表")


def check_ros2_nodes():
    """检查ROS2节点"""
    print("\n" + "=" * 60)
    print("【检查 ROS2 节点】")
    print("=" * 60)

    output = run_command("ros2 node list")
    if output:
        nodes = output.split('\n')
        print(f"✓ 找到 {len(nodes)} 个节点:")
        for node in nodes:
            print(f"  - {node}")
    else:
        print("✗ 无法获取节点列表")


def check_controllers():
    """检查ROS2控制器"""
    print("\n" + "=" * 60)
    print("【检查控制器状态】")
    print("=" * 60)

    # 尝试列出控制器
    output = run_command("ros2 control list_controllers")
    if output and "error" not in output.lower():
        print("✓ 控制器列表:")
        print(output)
    else:
        print("⚠️  无法获取控制器列表（可能未安装ros2-control）")
        print("  尝试手动检查...")


def check_joint_states_data(diagnostics_node, timeout=5.0):
    """检查/joint_states是否有数据"""
    print("\n" + "=" * 60)
    print("【检查 /joint_states 数据流】")
    print("=" * 60)

    print(f"等待 {timeout} 秒，监听 /joint_states 话题...")

    start_time = time.time()
    while (time.time() - start_time) < timeout:
        rclpy.spin_once(diagnostics_node, timeout_sec=0.1)

        if diagnostics_node.joint_state_received:
            print(f"\n✓ 接收到 {diagnostics_node.joint_state_count} 条消息")

            # 显示最新消息
            msg = diagnostics_node.last_joint_state
            print("\n最新消息内容:")
            print(f"  时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
            print(f"  关节数量: {len(msg.name)}")
            print(f"  关节名称: {msg.name}")

            if len(msg.position) > 0:
                print(f"  关节位置: {[f'{p:.3f}' for p in msg.position[:7]]}")
            if len(msg.velocity) > 0:
                print(f"  关节速度: {[f'{v:.3f}' for v in msg.velocity[:7]]}")

            return True

    print("\n✗ 超时！未接收到任何 /joint_states 消息")
    print("\n可能的原因:")
    print("  1. Kinova机器人驱动未启动")
    print("  2. ROS2控制器未运行")
    print("  3. 机器人未连接或未上电")
    print("\n建议操作:")
    print("  1. 检查机器人是否上电")
    print("  2. 启动Kinova驱动: ros2 launch kortex_bringup gen3.launch.py")
    print("  3. 检查网络连接: ping 192.168.8.10")

    return False


def check_gripper_topics():
    """检查夹爪相关话题"""
    print("\n" + "=" * 60)
    print("【检查夹爪话题】")
    print("=" * 60)

    # 可能的夹爪话题
    possible_topics = [
        '/gripper_controller/gripper_cmd',
        '/gripper_controller/gripper_cmd/goal',
        '/robotiq_gripper_controller/gripper_cmd',
        '/gripper_command',
        '/gripper/command'
    ]

    all_topics = run_command("ros2 topic list").split('\n')

    print("搜索夹爪相关话题:")
    found_topics = []
    for topic in all_topics:
        if 'gripper' in topic.lower():
            found_topics.append(topic)
            print(f"  ✓ {topic}")

            # 获取话题类型
            info = run_command(f"ros2 topic info {topic}")
            if info:
                print(f"    类型: {info}")

    if not found_topics:
        print("  ✗ 未找到夹爪相关话题")
        print("\n建议:")
        print("  1. 检查夹爪控制器是否已启动")
        print("  2. 查看 ros2 control list_controllers")

    return found_topics


def main():
    print("=" * 60)
    print("ROS2 系统诊断工具 - Kinova 机器人")
    print("=" * 60)

    # 初始化 ROS2
    try:
        rclpy.init()
    except Exception as e:
        print(f"✗ ROS2 初始化失败: {e}")
        return 1

    try:
        # 创建诊断节点
        diagnostics = ROS2Diagnostics()

        # 1. 检查节点
        check_ros2_nodes()

        # 2. 检查话题
        check_ros2_topics()

        # 3. 检查控制器
        check_controllers()

        # 4. 检查 /joint_states 数据
        has_joint_states = check_joint_states_data(diagnostics, timeout=5.0)

        # 5. 检查夹爪话题
        gripper_topics = check_gripper_topics()

        # 总结
        print("\n" + "=" * 60)
        print("【诊断总结】")
        print("=" * 60)

        if has_joint_states:
            print("✓ /joint_states 数据正常")
        else:
            print("✗ /joint_states 无数据 - 这是主要问题！")
            print("  遥操作无法工作，因为无法获取机器人状态")

        if gripper_topics:
            print(f"✓ 找到 {len(gripper_topics)} 个夹爪话题")
            print(f"  建议使用: {gripper_topics[0]}")
        else:
            print("⚠️  未找到夹爪话题")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 诊断失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        diagnostics.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main() or 0)
