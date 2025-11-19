#!/usr/bin/env python3
"""
KinovaInterface: ROS2通信层
功能: 封装与Kinova机器人的所有ROS2交互
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
import cv2

# 打开摄像头 (0是默认摄像头，可能是1或2)
cap = cv2.VideoCapture(0)


class KinovaInterface:
    """
    Kinova机器人的ROS2接口封装
    """
    
    def __init__(self, node_name='kinova_interface'):
        self.node_name = node_name
        self.node = None
        
        # 状态缓存
        self._latest_joint_state = None
        
        # 配置
        self.joint_state_topic = '/joint_states'
        self.velocity_command_topic = '/velocity_controller/commands'
        self.num_joints = 7

        self.gripper_command_topic = '/gripper_controller/gripper_cmd'
        self._gripper_state = 0.0  # 缓存gripper位置
        
        # __init__中
        self.camera_id = 0  # USB摄像头ID
        self._cap = None
    
    def _joint_state_callback(self, msg):
        self._latest_joint_state = msg
    
    def connect(self):
        if rclpy.ok() is not True:
            rclpy.init()

        self.node = Node(self.node_name)
        self._state_sub = self.node.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            10 
        )
        
        self._gripper_pub = self.node.create_publisher(
            Float64,
            self.gripper_command_topic,
            10
        )
        self._vel_pub = self.node.create_publisher(
            Float64MultiArray,
            self.velocity_command_topic,
            10 
        )
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            self.node.get_logger().warn("无法打开摄像头")
    
    def disconnect(self):
        self.node.destroy_node()
        rclpy.shutdown()
        self._latest_joint_state = None
    
        if self._cap is not None:
            self._cap.release()

    def get_joint_state(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        if self._latest_joint_state is None:
            print("当前缓存为空！")
            return
        positions = self._latest_joint_state.position[:7]
        velocites = self._latest_joint_state.velocity[:7]
        return np.array(positions),np.array(velocites)
    
    def send_joint_velocities(self, velocities):
        if len(velocities) != 7:
            raise ValueError(f"需要7个速度，收到{len(velocities)}个")
        v_list = list(velocities)  # list()兼容list和array
        msg = Float64MultiArray()
        msg.data = v_list

        self._vel_pub.publish(msg)

    def get_image(self):
        """获取摄像头图像"""
        if self._cap is None or not self._cap.isOpened():
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        # BGR转RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image


# ============ 使用示例 ============
if __name__ == '__main__':
    # 这个示例明天可以用来测试
    interface = KinovaInterface()
    
    try:
        interface.connect()
        print("连接成功")
        
        # 读状态
        pos, vel = interface.get_joint_state()
        print(f"Position: {pos}")
        print(f"Velocity: {vel}")
        
        # 发命令（小心！会真的动）
        # interface.send_joint_velocities([0, 0, 0, 0, 0, 0, 0.05])
        
    finally:
        interface.disconnect()
        print("断开连接")