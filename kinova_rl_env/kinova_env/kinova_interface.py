#!/usr/bin/env python3
"""
KinovaInterface: ROS2通信层
功能: 封装与Kinova机器人的所有ROS2交互
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
import cv2

# TF2用于获取TCP位姿
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R

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
        self._latest_tcp_pose = None  # 缓存TCP位姿
        self._latest_tcp_velocity = None  # 缓存TCP速度

        # 配置
        self.joint_state_topic = '/joint_states'
        self.trajectory_topic = '/joint_trajectory_controller/joint_trajectory'
        self.num_joints = 7
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]

        self.gripper_command_topic = '/gripper_controller/gripper_cmd'
        self._gripper_state = 0.0  # 缓存gripper位置

        # TF2配置
        self.base_frame = 'base_link'  # 基座坐标系
        self.tool_frame = 'tool_frame'  # 末端坐标系（可能是 'end_effector_link' 或 'tool_frame'）
        self._tf_buffer = None
        self._tf_listener = None

        # 相机配置
        self.camera_id = 0  # USB摄像头ID
        self._cap = None
    
    def _joint_state_callback(self, msg):
        self._latest_joint_state = msg
    
    def connect(self):
        """初始化ROS2连接和所有订阅/发布者"""
        if rclpy.ok() is not True:
            rclpy.init()

        self.node = Node(self.node_name)

        # 订阅关节状态
        self._state_sub = self.node.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            10
        )

        # 发布关节轨迹（使用 joint_trajectory_controller）
        self._trajectory_pub = self.node.create_publisher(
            JointTrajectory,
            self.trajectory_topic,
            10
        )

        # 发布gripper命令
        self._gripper_pub = self.node.create_publisher(
            Float64,
            self.gripper_command_topic,
            10
        )

        # 初始化TF2监听器
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self.node)

        # 等待 TF buffer 填充数据，需要 spin 让节点接收消息
        self.node.get_logger().info("等待 TF buffer 准备...")
        import time
        end_time = time.time() + 2.0
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # 初始化摄像头
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            self.node.get_logger().warn("无法打开摄像头")

        self.node.get_logger().info(f"KinovaInterface已连接，监听TF: {self.base_frame} → {self.tool_frame}")
    
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
        """
        发送关节速度命令（使用 joint_trajectory_controller）

        Args:
            velocities: 7个关节的速度 (rad/s)
        """
        if len(velocities) != 7:
            raise ValueError(f"需要7个速度，收到{len(velocities)}个")

        # 获取当前关节位置
        rclpy.spin_once(self.node, timeout_sec=0.01)
        if self._latest_joint_state is None:
            return

        current_positions = np.array(self._latest_joint_state.position[:7])
        joint_velocities = np.array(velocities)

        # 创建轨迹消息
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        dt = 0.1  # 100ms
        target_positions = current_positions + joint_velocities * dt
        point.positions = [float(x) for x in target_positions]
        point.velocities = [float(x) for x in joint_velocities]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(dt * 1e9)

        trajectory.points = [point]
        self._trajectory_pub.publish(trajectory)

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

    def get_tcp_pose(self):
        """
        获取TCP位姿（笛卡尔空间）

        Returns:
            tcp_pose: np.array shape (7,) [x, y, z, qx, qy, qz, qw]
                     位置 + 四元数姿态（在base_link坐标系下）
                     如果获取失败，返回None

        原理：
        - 从TF tree查询 base_link → tool_frame 的变换
        - 这个变换就是TCP在基座坐标系下的位姿
        """
        # 先spin一次，更新TF buffer
        rclpy.spin_once(self.node, timeout_sec=0.01)

        if self._tf_buffer is None:
            self.node.get_logger().warn("TF buffer未初始化")
            return None

        try:
            # 查询最新的变换（timeout=0表示获取最新）
            transform = self._tf_buffer.lookup_transform(
                self.base_frame,  # 目标坐标系
                self.tool_frame,  # 源坐标系
                rclpy.time.Time(),  # 最新时间
                timeout=Duration(seconds=0.1)
            )

            # 提取位置
            pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            # 提取四元数 [x, y, z, w]
            quat = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

            # 拼接为 [x, y, z, qx, qy, qz, qw]
            tcp_pose = np.concatenate([pos, quat])

            # 缓存
            self._latest_tcp_pose = tcp_pose

            return tcp_pose

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # TF查询失败（可能是TF树还没准备好，或者坐标系名称错误）
            self.node.get_logger().warn(f"无法获取TCP位姿: {e}")

            # 返回缓存值（如果有）
            if self._latest_tcp_pose is not None:
                return self._latest_tcp_pose

            return None

    def get_tcp_velocity(self):
        """
        获取TCP速度（笛卡尔空间）

        Returns:
            tcp_vel: np.array shape (6,) [vx, vy, vz, wx, wy, wz]
                    线速度 + 角速度（在base_link坐标系下）

        原理：
        - 从关节速度 + 雅可比矩阵计算
        - v_tcp = J * q_dot
        - 这里简化实现：通过数值微分TCP位姿获得

        注意：Kinova的ROS2驱动可能发布专门的速度topic，
             如果有，建议直接订阅而不是数值微分
        """
        # 简化实现：从关节速度计算（需要雅可比矩阵）
        # 这里先返回零向量，后续可以改进

        # TODO: 改进方案
        # 1. 订阅 /twist_controller/状态 话题（如果有）
        # 2. 或者用数值微分：记录上一时刻的pose，计算差分
        # 3. 或者订阅 /cartesian_velocity 话题

        tcp_vel = np.zeros(6)

        # 缓存
        self._latest_tcp_velocity = tcp_vel

        return tcp_vel

    def get_gripper_state(self):
        """
        获取gripper当前位置

        Returns:
            float: 0.0(全开) 到 1.0(全闭)

        TODO: 需要订阅gripper状态topic
        当前返回缓存值
        """
        return self._gripper_state

    def send_gripper_command(self, position):
        """
        发送gripper命令

        Args:
            position: float, 0.0(全开) 到 1.0(全闭)
        """
        position = np.clip(position, 0.0, 1.0)

        msg = Float64()
        msg.data = float(position)
        self._gripper_pub.publish(msg)

        # 更新缓存
        self._gripper_state = position


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