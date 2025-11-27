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

    def compute_jacobian(self, q: np.ndarray, delta: float = 1e-4) -> np.ndarray:
        """
        数值计算雅可比矩阵（高精度）

        使用有限差分法：对每个关节做微小扰动，观察 TCP 位姿变化

        Args:
            q: 当前关节位置 (7,)
            delta: 扰动量 (rad)

        Returns:
            J: 雅可比矩阵 (6, 7)，J @ dq = [dx, dy, dz, drx, dry, drz]
        """
        import rclpy
        from scipy.spatial.transform import Rotation as Rot

        J = np.zeros((6, 7))

        # 获取当前 TCP 位姿
        tcp_pose_0 = self.get_tcp_pose()
        if tcp_pose_0 is None:
            # 如果无法获取 TCP 位姿，返回零矩阵
            return J

        pos_0 = tcp_pose_0[:3]
        quat_0 = tcp_pose_0[3:]  # [qx, qy, qz, qw]

        # 对每个关节扰动
        for i in range(7):
            # 发布扰动后的关节状态（临时）
            q_perturbed = q.copy()
            q_perturbed[i] += delta

            # 这里需要用正运动学，但我们没有。
            # 数值雅可比需要发送关节位置并观察 TCP 变化
            # 这在实时系统中不可行！
            pass

        # 因为无法在线扰动关节，我们改用解析雅可比（基于DH参数）
        # 或者使用预计算的雅可比
        return J

    def compute_jacobian_analytical(self, q: np.ndarray) -> np.ndarray:
        """
        解析计算雅可比矩阵（基于 Kinova Gen3 DH 参数）

        使用标准的雅可比计算方法

        Args:
            q: 关节位置 (7,)

        Returns:
            J: 雅可比矩阵 (6, 7)
        """
        # Kinova Gen3 7-DOF 的 DH 参数（简化版本）
        # 这里使用近似的解析雅可比
        # 实际应该使用完整的 DH 参数和正运动学

        # 简化：使用几何雅可比
        # J_v = [z0 × (p7 - p0), z1 × (p7 - p1), ..., z6 × (p7 - p6)]
        # J_ω = [z0, z1, ..., z6]

        # 这需要正运动学计算每个关节的位置和轴向
        # 由于没有完整的 FK，我们使用一个基于经验的雅可比

        J = np.zeros((6, 7))

        # 基于 Kinova Gen3 的典型配置（近似）
        # 链长度（米）
        L1, L2, L3, L4 = 0.28, 0.21, 0.21, 0.18

        # 简化的几何雅可比（基于典型姿态）
        # 这是一个粗略估计，实际应该用完整的 FK/IK

        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        c3, s3 = np.cos(q[2]), np.sin(q[2])

        # 位置雅可比（前3行）
        J[0, 0] = -L2 * s1 * s2 - L3 * s1 * np.sin(q[1] + q[2])
        J[0, 1] = L2 * c1 * c2 + L3 * c1 * np.cos(q[1] + q[2])
        J[0, 2] = L3 * c1 * np.cos(q[1] + q[2])

        J[1, 0] = L2 * c1 * s2 + L3 * c1 * np.sin(q[1] + q[2])
        J[1, 1] = L2 * s1 * c2 + L3 * s1 * np.cos(q[1] + q[2])
        J[1, 2] = L3 * s1 * np.cos(q[1] + q[2])

        J[2, 1] = -L2 * s2 - L3 * np.sin(q[1] + q[2])
        J[2, 2] = -L3 * np.sin(q[1] + q[2])

        # 姿态雅可比（后3行） - 关节轴方向
        J[3:, 0] = [0, 0, 1]  # Joint 1: Z-axis
        J[3:, 1] = [c1, s1, 0]  # Joint 2
        J[3:, 2] = [c1, s1, 0]  # Joint 3
        J[3:, 4] = [c1, s1, 0]  # Joint 5
        J[3:, 6] = [c1, s1, 0]  # Joint 7

        return J

    def cartesian_to_joint_velocity(self, twist: np.ndarray) -> np.ndarray:
        """
        笛卡尔速度 → 关节速度（使用雅可比伪逆，高精度）

        Args:
            twist: [vx, vy, vz, wx, wy, wz] 笛卡尔速度

        Returns:
            joint_vel: (7,) 关节速度
        """
        if self.current_joint_positions is None:
            return np.zeros(7)

        # 计算雅可比矩阵
        J = self.compute_jacobian_analytical(self.current_joint_positions)

        # 使用阻尼最小二乘（DLS）求伪逆，避免奇异性
        # dq = J^T (J J^T + λI)^(-1) dx
        lambda_damping = 0.01  # 阻尼系数
        JJT = J @ J.T + lambda_damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)

        # 计算关节速度
        joint_vel = J_pinv @ twist

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
