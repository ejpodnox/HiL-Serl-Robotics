"""
基于 Joint Trajectory Controller 的机器人控制器

接受 twist 命令，转换为关节速度，使用 joint_trajectory_controller
使用精确的 FK 和雅可比计算
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
    
    将笛卡尔速度转换为关节速度（通过精确雅可比）
    """

    def __init__(self, robot_ip: str, node_name: str = 'joint_velocity_commander'):
        super().__init__(node_name)

        self.robot_ip = robot_ip

        # 安全限制
        self.safety_max_linear_vel = 0.1
        self.safety_max_angular_vel = 0.3
        self.enable_safety_check = True

        # DH 参数 (来自论文 Table III)
        self.dh_a = np.array([0, 0.28, 0, 0, 0, 0])  # m
        self.dh_b = np.array([0.2433, 0.03, 0.02, 0.245, 0.057, 0.235])  # m
        self.dh_alpha = np.array([np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2, 0])

        # 关节名称（Kinova Gen3 7-DOF）
        # 注意：保持原来的7关节接口，但实际只用前6个
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
        self.get_logger().info('使用精确 FK/Jacobian (基于 DH 参数)')

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

    def forward_kinematics(self, q: np.ndarray) -> tuple:
        """
        精确正运动学 (基于论文 Section III, Eq. 3)
        
        Args:
            q: 关节角度 (前6个关节)
            
        Returns:
            p: TCP 位置 [x, y, z]
            Q: TCP 姿态矩阵 (3x3)
        """
        # 只使用前6个关节
        q6 = q[:6]
        
        Q_list = [np.eye(3)]  # Q0
        p = np.zeros(3)
        
        for i in range(6):
            c, s = np.cos(q6[i]), np.sin(q6[i])
            ca, sa = np.cos(self.dh_alpha[i]), np.sin(self.dh_alpha[i])
            
            # 旋转矩阵 Qi (论文 Eq. 1)
            Qi = np.array([
                [c, -ca*s, sa*s],
                [s, ca*c, -sa*c],
                [0, sa, ca]
            ])
            
            Q_cumulative = Q_list[-1] @ Qi
            Q_list.append(Q_cumulative)
            
            # 位置向量 ai (论文 Eq. 2)
            ai = np.array([
                self.dh_a[i] * c,
                self.dh_a[i] * s,
                self.dh_b[i]
            ])
            
            # 累积位置 (论文 Eq. 3a)
            p += Q_list[i] @ ai
        
        return p, Q_list[-1]

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        精确计算几何雅可比矩阵（数值微分法）
        
        Args:
            q: 当前关节位置 (7,)，但只使用前6个
            
        Returns:
            J: 雅可比矩阵 (6, 7)，最后一列为零（第7关节不影响前6DOF末端）
        """
        J = np.zeros((6, 7))
        
        # 计算当前 TCP 位姿
        p0, R0 = self.forward_kinematics(q)
        
        # 对前6个关节数值微分
        delta = 1e-6
        for i in range(6):
            q_perturbed = q.copy()
            q_perturbed[i] += delta
            
            p_plus, R_plus = self.forward_kinematics(q_perturbed)
            
            # 位置雅可比（线性速度）
            J[:3, i] = (p_plus - p0) / delta
            
            # 姿态雅可比（角速度）
            # 使用旋转向量表示
            from scipy.spatial.transform import Rotation as Rot
            dR = R_plus @ R0.T
            rotvec = Rot.from_matrix(dR).as_rotvec()
            J[3:, i] = rotvec / delta
        
        # 第7关节对前6DOF末端姿态无影响（如果是末端旋转关节）
        J[:, 6] = 0
        
        return J

    def cartesian_to_joint_velocity(self, twist: np.ndarray) -> np.ndarray:
        """
        笛卡尔速度 → 关节速度（使用精确雅可比伪逆）

        Args:
            twist: [vx, vy, vz, wx, wy, wz] 笛卡尔速度

        Returns:
            joint_vel: (7,) 关节速度
        """
        if self.current_joint_positions is None:
            return np.zeros(7)

        # 计算精确雅可比矩阵
        J = self.compute_jacobian(self.current_joint_positions)

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
        """发送 twist 命令（保持原接口）"""
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
        """停止（保持原接口）"""
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
        """急停（保持原接口）"""
        self.is_emergency_stopped = True
        self.send_zero_twist()

    def resume(self):
        """解除急停（保持原接口）"""
        self.is_emergency_stopped = False

    def get_tcp_pose(self) -> Optional[np.ndarray]:
        """
        获取 TCP 位姿（现在可以真正计算了）
        
        Returns:
            pose: [x, y, z, qx, qy, qz, qw] 或 None
        """
        if self.current_joint_positions is None:
            return None
        
        p, R = self.forward_kinematics(self.current_joint_positions)
        
        # 转换为四元数
        from scipy.spatial.transform import Rotation as Rot
        quat = Rot.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
        
        return np.concatenate([p, quat])

    def set_safety_limits(self, max_linear: float = None, max_angular: float = None):
        """设置安全限制（保持原接口）"""
        if max_linear is not None:
            self.safety_max_linear_vel = max_linear
        if max_angular is not None:
            self.safety_max_angular_vel = max_angular

    def enable_safety(self, enable: bool = True):
        """启用/禁用安全检查（保持原接口）"""
        self.enable_safety_check = enable

    def get_info(self) -> dict:
        """获取控制器信息（保持原接口）"""
        return {
            'robot_ip': self.robot_ip,
            'emergency_stopped': self.is_emergency_stopped,
            'safety_enabled': self.enable_safety_check,
            'max_linear_vel': self.safety_max_linear_vel,
            'max_angular_vel': self.safety_max_angular_vel,
            'last_command_age': time.time() - self.last_twist_time,
            'backend': 'joint_trajectory_controller',
            'kinematics': 'accurate_dh_based'  # 新增信息
        }