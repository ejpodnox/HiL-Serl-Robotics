#!/usr/bin/env python3
"""
Test Step 2: 发送关节速度命令
测试能否控制机器人运动

使用: python tests/test_step2_send_velocity.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time


class VelocitySender(Node):
    def __init__(self):
        super().__init__('velocity_sender')
        
        # 发布velocity命令
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/velocity_controller/commands',
            10
        )
        
        self.get_logger().info("Velocity sender ready")
    
    def send_velocity(self, velocities):
        """发送7维关节速度"""
        msg = Float64MultiArray()
        msg.data = velocities
        self.vel_pub.publish(msg)
        self.get_logger().info(f"Sent: {velocities}")


def main():
    rclpy.init()
    node = VelocitySender()
    
    time.sleep(1.0)  # 等待publisher准备好
    
    try:
        # 测试序列
        node.get_logger().info("=== 测试1: 动第7关节 ===")
        node.send_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
        time.sleep(2.0)
        
        node.get_logger().info("=== 停止 ===")
        node.send_velocity([0.0] * 7)
        time.sleep(1.0)
        
        node.get_logger().info("=== 测试2: 动第2关节 ===")
        node.send_velocity([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(2.0)
        
        node.get_logger().info("=== 停止 ===")
        node.send_velocity([0.0] * 7)
        
        node.get_logger().info("测试完成")
        
    except KeyboardInterrupt:
        node.send_velocity([0.0] * 7)  # 紧急停止
        print("\n紧急停止")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()