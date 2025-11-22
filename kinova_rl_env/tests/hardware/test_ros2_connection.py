#!/usr/bin/env python3
"""
Test Step 1: 读取Kinova关节状态
功能: 订阅/joint_states并打印关节位置和速度

使用方法:
终端1: ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10
终端2: python tests/test_step1_read_states.py

预期结果: 应该看到7个关节的位置和速度不断打印
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateReader(Node):
    def __init__(self):
        super().__init__('joint_state_reader')
        
        # 订阅joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',  # 如果topic名不对，回实验室用 ros2 topic list 查看
            self.state_callback,
            10
        )
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("开始监听 /joint_states")
        self.get_logger().info("按 Ctrl+C 退出")
        self.get_logger().info("=" * 50)
        
        self.count = 0
    
    def state_callback(self, msg):
        """每次收到消息时被调用"""
        self.count += 1
        
        # 只详细打印前3次
        if self.count <= 3:
            self.get_logger().info(f"\n【第 {self.count} 次接收】")
            self.get_logger().info(f"时间戳: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
            self.get_logger().info(f"关节数量: {len(msg.name)}")
            self.get_logger().info(f"关节名称: {msg.name}")
            
            if len(msg.position) > 0:
                pos_str = ", ".join([f"{p:.3f}" for p in msg.position])
                self.get_logger().info(f"关节位置(rad): [{pos_str}]")
            
            if len(msg.velocity) > 0:
                vel_str = ", ".join([f"{v:.3f}" for v in msg.velocity])
                self.get_logger().info(f"关节速度(rad/s): [{vel_str}]")
            
        elif self.count == 4:
            self.get_logger().info("\n后续数据不再打印，但仍在接收...")
            self.get_logger().info(f"当前接收频率约: {self.count / 1.0:.1f} Hz")


def main():
    print("\n启动 Kinova Joint State 读取测试...\n")
    
    rclpy.init()
    node = JointStateReader()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n用户中断，正在退出...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("测试结束\n")


if __name__ == '__main__':
    main()