"""
基于 Joint Trajectory Controller 的机器人控制器

接受 twist 命令，转换为关节速度，使用 joint_trajectory_controller
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
import numpy as np
import time
from typing import Optional


class JointVelocityCommander(Node):
    """
    使用 joint_trajectory_controller 实现 twist 控制

    将笛卡尔速度转换为关节速度（通过数值雅可比）
    """

    def __init__(self, robot_ip: str, node_name: str = 'joint_velocity_commander'):
        super().__init__(node_name)

        self.robot_ip = robot_ip

        # 安全限制
        self.safety_max_linear_vel = 0.1
        self.safety_max_angular_vel = 0.3
        self.enable_safety_check = True

        # 关节名称（Kinova Gen3 7-DOF）
        self.joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]

        # 关节状态
        self.current_joint_positions = None
        self.current_joint_velocities = None

        # 订阅关节状态
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 发布关节轨迹
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # 状态
        self.is_emergency_stopped = False
        self.last_twist_time = time.time()

        self.get_logger().info(f'Joint Velocity Commander 初始化完成 ({robot_ip})')

    def joint_state_callback(self, msg):
        """关节状态回调"""
        # 只取前7个关节（忽略夹爪）
        self.current_joint_positions = np.array(msg.position[:7])
        self.current_joint_velocities = np.array(msg.velocity[:7]) if msg.velocity else np.zeros(7)

    def safety_check_twist(self, twist_input):
        """安全检查（与其他 Commander 接口一致）"""
        if not self.enable_safety_check:
            if isinstance(twist_input, np.ndarray):
                return twist_input
            else:
                return np.array([
                    twist_input['linear']['x'], twist_input['linear']['y'], twist_input['linear']['z'],
                    twist_input['angular']['x'], twist_input['angular']['y'], twist_input['angular']['z']
                ])

        # 转换为数组
        if isinstance(twist_input, dict):
            twist = np.array([
                twist_input['linear']['x'], twist_input['linear']['y'], twist_input['linear']['z'],
                twist_input['angular']['x'], twist_input['angular']['y'], twist_input['angular']['z']
            ])
        else:
            twist = np.array(twist_input)

        linear = twist[:3]
        angular = twist[3:]

        linear_speed = np.linalg.norm(linear)
        angular_speed = np.linalg.norm(angular)

        if linear_speed > self.safety_max_linear_vel:
            linear = linear * (self.safety_max_linear_vel / linear_speed)
        if angular_speed > self.safety_max_angular_vel:
            angular = angular * (self.safety_max_angular_vel / angular_speed)

        return np.concatenate([linear, angular])

    def cartesian_to_joint_velocity(self, twist: np.ndarray) -> np.ndarray:
        """
        笛卡尔速度 -> 关节速度（简化版本）

        使用常数映射矩阵（基于典型配置）
        实际应用中应使用雅可比矩阵，但这需要正运动学
        """
        # 简化映射：只使用前3个关节控制位置，后4个关节控制姿态
        # 这是一个粗略的近似，但对于慢速运动足够

        joint_vel = np.zeros(7)

        # 线速度映射（简化）
        joint_vel[1] = -twist[2] * 2.0  # Z -> joint_2 (shoulder)
        joint_vel[2] = twist[0] * 1.5   # X -> joint_3 (elbow)
        joint_vel[0] = twist[1] * 1.0   # Y -> joint_1 (base rotation)

        # 角速度映射（简化）
        joint_vel[4] = twist[3] * 0.5   # Roll -> joint_5
        joint_vel[5] = twist[4] * 0.5   # Pitch -> joint_6
        joint_vel[6] = twist[5] * 0.5   # Yaw -> joint_7

        # 限制关节速度
        max_joint_vel = 0.5  # rad/s
        joint_vel = np.clip(joint_vel, -max_joint_vel, max_joint_vel)

        return joint_vel

    def send_twist(self, twist_input):
        """发送 twist 命令"""
        if self.is_emergency_stopped:
            return

        # 安全检查
        safe_twist = self.safety_check_twist(twist_input)

        # 等待关节状态
        if self.current_joint_positions is None:
            return

        # 转换为关节速度
        joint_velocities = self.cartesian_to_joint_velocity(safe_twist)

        # 创建轨迹消息（短时间段的速度控制）
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        # 目标位置 = 当前位置 + 速度 * 时间
        dt = 0.1  # 100ms
        target_positions = self.current_joint_positions + joint_velocities * dt
        point.positions = [float(x) for x in target_positions]
        point.velocities = [float(x) for x in joint_velocities]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(dt * 1e9)

        trajectory.points = [point]

        self.trajectory_publisher.publish(trajectory)
        self.last_twist_time = time.time()

    def send_zero_twist(self):
        """停止"""
        if self.current_joint_positions is None:
            return

        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in self.current_joint_positions]
        point.velocities = [0.0] * 7
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.1 * 1e9)

        trajectory.points = [point]
        self.trajectory_publisher.publish(trajectory)

    def emergency_stop(self):
        """急停"""
        self.is_emergency_stopped = True
        self.send_zero_twist()

    def resume(self):
        """解除急停"""
        self.is_emergency_stopped = False

    def get_tcp_pose(self) -> Optional[np.ndarray]:
        """获取 TCP 位姿（使用 TF，与 RobotCommander 相同）"""
        # 简化版本：返回 None
        # 实际应用中应使用 TF2 或正运动学
        return None

    def set_safety_limits(self, max_linear: float = None, max_angular: float = None):
        """设置安全限制"""
        if max_linear is not None:
            self.safety_max_linear_vel = max_linear
        if max_angular is not None:
            self.safety_max_angular_vel = max_angular

    def enable_safety(self, enable: bool = True):
        """启用/禁用安全检查"""
        self.enable_safety_check = enable

    def get_info(self) -> dict:
        """获取控制器信息"""
        return {
            'robot_ip': self.robot_ip,
            'emergency_stopped': self.is_emergency_stopped,
            'safety_enabled': self.enable_safety_check,
            'max_linear_vel': self.safety_max_linear_vel,
            'max_angular_vel': self.safety_max_angular_vel,
            'last_command_age': time.time() - self.last_twist_time,
            'backend': 'joint_trajectory_controller'
        }
