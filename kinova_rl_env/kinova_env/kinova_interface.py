#!/usr/bin/env python3
"""
KinovaInterface: ROS2通信层
功能: 封装与Kinova机器人的所有ROS2交互
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from control_msgs.action import GripperCommand

# TF2用于获取TCP位姿
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R


class KinovaInterface:
    """
    Kinova机器人的ROS2接口封装
    """

    def __init__(
        self,
        node_name: str = 'kinova_interface',
        joint_state_topic: str = '/joint_states',
        trajectory_topic: str = '/joint_trajectory_controller/joint_trajectory',
        twist_topic: str = '/twist_controller/commands',
        gripper_command_topic: str = '/robotiq_gripper_controller/gripper_cmd',
        base_frame: str = 'base_link',
        tool_frame: str = 'tool_frame',
        joint_names=None,
    ):
        self.node_name = node_name
        self.node = None

        # 状态缓存
        self._latest_joint_state = None
        self._latest_tcp_pose = None  # 缓存TCP位姿
        self._latest_tcp_velocity = None  # 缓存TCP速度

        # 配置
        self.joint_state_topic = joint_state_topic
        self.trajectory_topic = trajectory_topic
        self.twist_topic = twist_topic
        self.joint_names = joint_names or [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]
        self.num_joints = len(self.joint_names)

        self.gripper_command_topic = gripper_command_topic
        self._gripper_state = 0.0  # 缓存gripper位置
        self._gripper_action_client = None
        self.gripper_available = False

        # TF2配置
        self.base_frame = base_frame  # 基座坐标系
        self.tool_frame = tool_frame  # 末端坐标系
        self._tf_buffer = None
        self._tf_listener = None
        self._executor = None
        self._missing_joint_warned = False
    
    def _joint_state_callback(self, msg):
        self._latest_joint_state = msg

    def _get_mapped_joint_state(self):
        """
        Return joint positions/velocities ordered by self.joint_names.
        """
        if self._latest_joint_state is None:
            return None
        name_to_idx = {n: i for i, n in enumerate(self._latest_joint_state.name)}
        positions = []
        velocities = []
        for jn in self.joint_names:
            if jn not in name_to_idx:
                if not self._missing_joint_warned:
                    print(f"未在 joint_states 中找到关节 {jn}，请检查控制器 joint_names")
                    self._missing_joint_warned = True
                return None
            idx = name_to_idx[jn]
            positions.append(self._latest_joint_state.position[idx])
            if idx < len(self._latest_joint_state.velocity):
                velocities.append(self._latest_joint_state.velocity[idx])
            else:
                velocities.append(0.0)
        return np.array(positions), np.array(velocities)
    
    def connect(self):
        """初始化ROS2连接和所有订阅/发布者"""
        if rclpy.ok() is not True:
            rclpy.init()

        self.node = Node(self.node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        # 订阅关节状态
        self._state_sub = self.node.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            10
        )

        # 发布关节指令（Trajectory 或 Velocity）
        self._trajectory_pub = None
        self._velocity_pub = None
        if "trajectory" in self.trajectory_topic:
            self._trajectory_pub = self.node.create_publisher(
                JointTrajectory,
                self.trajectory_topic,
                10
            )
        else:
            self._velocity_pub = self.node.create_publisher(
                Float64MultiArray,
                self.trajectory_topic,
                10
            )

        # 发布Cartesian twist指令
        self._twist_pub = self.node.create_publisher(
            Twist,
            self.twist_topic,
            10
        )

        # 创建夹爪Action客户端
        self._gripper_action_client = ActionClient(
            self.node,
            GripperCommand,
            self.gripper_command_topic
        )

        # 检查夹爪服务器是否可用
        if self._gripper_action_client.wait_for_server(timeout_sec=2.0):
            self.gripper_available = True
            self.node.get_logger().info(f"✓ 夹爪Action服务器已连接: {self.gripper_command_topic}")
        else:
            self.gripper_available = False
            self.node.get_logger().warn(f"⚠️  夹爪Action服务器未响应: {self.gripper_command_topic}")

        # 初始化TF2监听器
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self.node)

        # 等待 TF buffer 填充数据，需要 spin 让节点接收消息
        self.node.get_logger().info("等待 TF buffer 准备...")
        import time
        end_time = time.time() + 2.0
        while time.time() < end_time:
            self._executor.spin_once(timeout_sec=0.1)

        self.node.get_logger().info(f"KinovaInterface已连接，监听TF: {self.base_frame} → {self.tool_frame}")
    
    def disconnect(self):
        if self._executor:
            try:
                self._executor.remove_node(self.node)
                self._executor.shutdown()
            except Exception:
                pass
            self._executor = None
        self.node.destroy_node()
        rclpy.shutdown()
        self._latest_joint_state = None

    def get_joint_state(self):
        self._executor.spin_once(timeout_sec=0.01)
        mapped = self._get_mapped_joint_state()
        if mapped is None:
            return None, None
        return mapped
    
    def send_joint_velocities(self, velocities, dt=0.05):
        """
        发送关节速度命令

        如果绑定到 velocity_controller 则发布 Float64MultiArray；
        如果绑定到 trajectory_controller 则发布 JointTrajectory 增量。

        Args:
            velocities: 7个关节的速度 (rad/s)
            dt: 轨迹执行时间（秒），应匹配控制周期。默认 0.05s (20Hz)
        """
        if len(velocities) != self.num_joints:
            raise ValueError(f"需要{self.num_joints}个速度，收到{len(velocities)}个")

        # 获取当前关节位置
        self._executor.spin_once(timeout_sec=0.01)
        mapped = self._get_mapped_joint_state()
        if mapped is None:
            return
        current_positions, _ = mapped
        joint_velocities = np.array(velocities)

        # 如果有velocity发布者，直接发布速度
        if self._velocity_pub is not None:
            msg = Float64MultiArray()
            msg.data = [float(x) for x in joint_velocities]
            self._velocity_pub.publish(msg)
            return

        # 否则作为轨迹增量发送
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        target_positions = current_positions + joint_velocities * dt
        point.positions = [float(x) for x in target_positions]
        point.velocities = [float(x) for x in joint_velocities]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(dt * 1e9)

        trajectory.points = [point]
        self._trajectory_pub.publish(trajectory)

    def send_joint_positions(self, positions, duration=3.0):
        """
        发送关节位置目标（使用 joint_trajectory_controller）

        Args:
            positions: 目标关节位置列表/ndarray（长度=DOF）
            duration: 到达目标所用时间（秒）
        """
        if len(positions) != self.num_joints:
            raise ValueError(f"需要 {self.num_joints} 个关节位置，收到 {len(positions)} 个")

        if self._trajectory_pub is None:
            raise RuntimeError("当前配置未启用 trajectory 控制器，无法发送位置命令")

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.velocities = [0.0] * self.num_joints
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        traj.points = [point]
        self._trajectory_pub.publish(traj)

    def send_cartesian_twist(self, twist_cmd):
        """
        发送Cartesian twist命令（Cartesian空间速度控制）

        Args:
            twist_cmd: np.ndarray shape (6,) -> [vx, vy, vz, wx, wy, wz]
                      线速度 (m/s) + 角速度 (rad/s)
        """
        if len(twist_cmd) != 6:
            raise ValueError(f"需要6个twist值，收到{len(twist_cmd)}个")

        twist_msg = Twist()
        twist_msg.linear.x = float(twist_cmd[0])
        twist_msg.linear.y = float(twist_cmd[1])
        twist_msg.linear.z = float(twist_cmd[2])
        twist_msg.angular.x = float(twist_cmd[3])
        twist_msg.angular.y = float(twist_cmd[4])
        twist_msg.angular.z = float(twist_cmd[5])

        self._twist_pub.publish(twist_msg)

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
        self._executor.spin_once(timeout_sec=0.01)

        if self._tf_buffer is None:
            self.node.get_logger().warn("TF buffer未初始化")
            return None

        try:
            # 查询最新的变换（timeout=0表示获取最新）
            candidate_frames = [self.tool_frame, "end_effector_link", "tool_link"]
            transform = None
            last_error = None
            for frame in candidate_frames:
                try:
                    transform = self._tf_buffer.lookup_transform(
                        self.base_frame,
                        frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.1),
                    )
                    # 更新当前tool_frame为有效的frame，减少后续警告
                    self.tool_frame = frame
                    break
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    last_error = e
                    continue

            if transform is None:
                raise last_error or LookupException("TF lookup failed")

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

    def send_gripper_command(self, position, max_effort=100.0):
        """
        发送gripper命令（使用Action接口）

        Args:
            position: float, 0.0(全开) 到 0.8(全闭) - Robotiq夹爪的标准范围
            max_effort: float, 最大力度 (0-100)
        """
        # Robotiq夹爪：0.0=完全打开，0.8=完全闭合
        # 输入范围 [0, 1] 映射到 [0, 0.8]
        position = np.clip(position * 0.8, 0.0, 0.8)

        if not self.gripper_available:
            self.node.get_logger().warn("夹爪不可用，命令未发送")
            return

        # 创建夹爪命令
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)

        # 异步发送命令（不等待完成）
        self._gripper_action_client.send_goal_async(goal)

        # 更新缓存
        self._gripper_state = position / 0.8  # 转换回 [0, 1] 范围


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
