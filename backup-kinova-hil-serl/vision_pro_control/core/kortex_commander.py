"""
直接使用 Kortex API 控制机械臂（绕过 ROS2 控制器）

这个实现使用 Kinova 的 Kortex Python API 直接发送笛卡尔速度命令，
而不依赖 ROS2 的 twist_controller。
"""

import numpy as np
import time
import threading
from typing import Optional

try:
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TransportClientTcp import TransportClientTcp
    KORTEX_API_AVAILABLE = True
except ImportError:
    KORTEX_API_AVAILABLE = False
    print("警告: Kortex API 未安装")
    print("安装方法: pip install kortex_api")


class KortexCommander:
    """
    使用 Kortex API 直接控制机械臂

    提供与 RobotCommander 相同的接口，但绕过 ROS2 控制器
    """

    def __init__(self, robot_ip: str, username: str = "admin", password: str = "admin"):
        """
        Args:
            robot_ip: 机械臂 IP 地址
            username: Kortex API 用户名（默认: admin）
            password: Kortex API 密码（默认: admin）
        """
        if not KORTEX_API_AVAILABLE:
            raise ImportError(
                "Kortex API 未安装。请运行: pip install kortex_api\n"
                "或从 https://github.com/Kinovarobotics/kortex 下载"
            )

        self.robot_ip = robot_ip
        self.username = username
        self.password = password

        # 安全限制参数
        self.safety_max_linear_vel = 0.1   # 最大线速度 0.1 m/s
        self.safety_max_angular_vel = 0.3  # 最大角速度 0.3 rad/s
        self.enable_safety_check = True

        # 状态变量
        self.is_emergency_stopped = False
        self.last_twist_time = time.time()
        self._stop_command_thread = False

        # Kortex API 连接
        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        self.base_cyclic = None

        # 初始化连接
        self._connect()

        # 启动持续发送命令的线程（Kortex API 需要持续发送）
        self._command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._current_twist = np.zeros(6)
        self._twist_lock = threading.Lock()
        self._command_thread.start()

        print(f"✓ Kortex Commander 初始化完成 ({robot_ip})")

    def _connect(self):
        """建立 Kortex API 连接"""
        try:
            # 创建连接
            self.transport = TransportClientTcp()
            self.transport.connect(self.robot_ip, 10000)

            # 创建路由器
            self.router = RouterClient(self.transport, RouterClient.basicMessageCallback)

            # 创建会话
            self.session_manager = SessionManager(self.router)
            self.session_manager.CreateSession(self.username, self.password)

            # 创建服务客户端
            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router)

            print(f"✓ 已连接到 Kortex API ({self.robot_ip})")

        except Exception as e:
            raise ConnectionError(f"无法连接到机械臂: {e}")

    def _disconnect(self):
        """断开 Kortex API 连接"""
        if self.session_manager:
            try:
                self.session_manager.CloseSession()
            except:
                pass

        if self.transport:
            try:
                self.transport.disconnect()
            except:
                pass

    def safety_check_twist(self, twist_input):
        """
        安全检查：限制速度到安全范围

        Args:
            twist_input: numpy数组 [vx, vy, vz, wx, wy, wz] 或字典
        Returns:
            安全限制后的 numpy 数组
        """
        if not self.enable_safety_check:
            if isinstance(twist_input, np.ndarray):
                return twist_input
            else:
                # 字典转数组
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

        if len(twist) != 6:
            raise ValueError(f"Twist数组长度必须为6，当前为{len(twist)}")

        linear = twist[:3]
        angular = twist[3:]

        linear_speed = np.linalg.norm(linear)
        angular_speed = np.linalg.norm(angular)

        # 限制线速度
        if linear_speed > self.safety_max_linear_vel:
            scale = self.safety_max_linear_vel / linear_speed
            linear = linear * scale
            print(f'警告: 线速度超限，已缩放: {linear_speed:.3f} -> {self.safety_max_linear_vel:.3f} m/s')

        # 限制角速度
        if angular_speed > self.safety_max_angular_vel:
            scale = self.safety_max_angular_vel / angular_speed
            angular = angular * scale
            print(f'警告: 角速度超限，已缩放: {angular_speed:.3f} -> {self.safety_max_angular_vel:.3f} rad/s')

        return np.concatenate([linear, angular])

    def send_twist(self, twist_input):
        """
        发送 Twist 指令到机械臂

        Args:
            twist_input: numpy数组 [vx, vy, vz, wx, wy, wz] 或字典
        """
        if self.is_emergency_stopped:
            print('警告: 急停状态，忽略 Twist 指令')
            return

        # 安全检查
        safe_twist = self.safety_check_twist(twist_input)

        # 更新当前速度命令
        with self._twist_lock:
            self._current_twist = safe_twist
            self.last_twist_time = time.time()

    def _command_loop(self):
        """
        持续发送速度命令的线程

        Kortex API 需要以一定频率持续发送命令，否则机械臂会停止
        """
        rate = 100  # Hz
        period = 1.0 / rate

        while not self._stop_command_thread:
            try:
                with self._twist_lock:
                    twist = self._current_twist.copy()

                # 检查命令是否过期（超过0.5秒没有新命令，发送零速度）
                if time.time() - self.last_twist_time > 0.5:
                    twist = np.zeros(6)

                # 发送笛卡尔速度命令
                self._send_cartesian_velocity(twist)

            except Exception as e:
                print(f"命令发送错误: {e}")

            time.sleep(period)

    def _send_cartesian_velocity(self, twist: np.ndarray):
        """
        发送笛卡尔速度命令到 Kortex API

        Args:
            twist: [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        """
        if self.base is None:
            return

        try:
            # 创建 TwistCommand
            command = Base_pb2.TwistCommand()

            # 参考坐标系（base frame）
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

            # 设置线速度 (m/s)
            command.twist.linear_x = float(twist[0])
            command.twist.linear_y = float(twist[1])
            command.twist.linear_z = float(twist[2])

            # 设置角速度 (deg/s) - Kortex API 使用角度
            command.twist.angular_x = float(np.rad2deg(twist[3]))
            command.twist.angular_y = float(np.rad2deg(twist[4]))
            command.twist.angular_z = float(np.rad2deg(twist[5]))

            # 持续时间（毫秒）- 设为0表示持续发送
            command.duration = 0

            # 发送命令
            self.base.SendTwistCommand(command)

        except Exception as e:
            # 静默失败，避免刷屏
            pass

    def send_zero_twist(self):
        """发送零速度（停止）"""
        with self._twist_lock:
            self._current_twist = np.zeros(6)

        # 立即发送停止命令
        try:
            self._send_cartesian_velocity(np.zeros(6))
        except:
            pass

    def emergency_stop(self):
        """急停：立即停止所有运动"""
        print('!!! 急停触发 !!!')
        self.is_emergency_stopped = True

        # 发送零速度
        self.send_zero_twist()

        # 使用 Kortex API 的停止功能
        if self.base:
            try:
                self.base.Stop()
            except:
                pass

    def resume(self):
        """解除急停"""
        print('解除急停')
        self.is_emergency_stopped = False

    def get_tcp_pose(self) -> Optional[np.ndarray]:
        """
        获取当前 TCP 位姿

        Returns:
            np.array: [x, y, z, qx, qy, qz, qw] 或 None
        """
        if self.base is None:
            return None

        try:
            # 获取笛卡尔位姿
            pose = self.base.GetMeasuredCartesianPose()

            # 提取位置 (m)
            pos = np.array([pose.x, pose.y, pose.z])

            # 提取四元数（Kortex 使用 theta_x, theta_y, theta_z 表示旋转）
            # 这里需要转换为四元数
            # 简化版本：使用欧拉角转四元数
            from scipy.spatial.transform import Rotation

            theta_x = np.deg2rad(pose.theta_x)
            theta_y = np.deg2rad(pose.theta_y)
            theta_z = np.deg2rad(pose.theta_z)

            rot = Rotation.from_euler('xyz', [theta_x, theta_y, theta_z])
            quat = rot.as_quat()  # [x, y, z, w]

            # 拼接为 [x, y, z, qx, qy, qz, qw]
            return np.concatenate([pos, quat])

        except Exception as e:
            print(f"获取 TCP 位姿失败: {e}")
            return None

    def set_safety_limits(self, max_linear: float = None, max_angular: float = None):
        """设置安全限制"""
        if max_linear is not None:
            self.safety_max_linear_vel = max_linear
            print(f'安全限制 - 最大线速度: {self.safety_max_linear_vel} m/s')

        if max_angular is not None:
            self.safety_max_angular_vel = max_angular
            print(f'安全限制 - 最大角速度: {self.safety_max_angular_vel} rad/s')

    def enable_safety(self, enable: bool = True):
        """启用/禁用安全检查"""
        self.enable_safety_check = enable
        status = "启用" if enable else "禁用"
        print(f'安全检查已{status}')

    def get_info(self) -> dict:
        """获取控制器信息"""
        return {
            'robot_ip': self.robot_ip,
            'emergency_stopped': self.is_emergency_stopped,
            'safety_enabled': self.enable_safety_check,
            'max_linear_vel': self.safety_max_linear_vel,
            'max_angular_vel': self.safety_max_angular_vel,
            'last_command_age': time.time() - self.last_twist_time,
            'backend': 'Kortex_API'
        }

    def __del__(self):
        """析构函数：清理资源"""
        self._stop_command_thread = True
        if hasattr(self, '_command_thread'):
            self._command_thread.join(timeout=1.0)
        self._disconnect()
