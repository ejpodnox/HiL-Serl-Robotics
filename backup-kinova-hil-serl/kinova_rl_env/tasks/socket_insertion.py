"""
Socket Insertion 任务

插2孔插座任务（类似论文中的插USB任务）。
"""

import numpy as np
from typing import Dict, Tuple
from .base_task import BaseTask


class SocketInsertionTask(BaseTask):
    """
    Socket Insertion 任务

    任务描述：
    1. 抓取插头（已经在夹爪中）
    2. 移动到插座附近
    3. 对准插座孔
    4. 插入插座

    这是一个多阶段任务，需要精确的位置和姿态控制。
    """

    def __init__(self, config=None):
        super().__init__(config)

        # 插座位置（世界坐标系）
        self.socket_position = np.array(
            self.config.get("socket_position", [0.45, 0.15, 0.25])
        )

        # 插座方向（四元数: qx, qy, qz, qw）
        # 默认：垂直于桌面，孔朝向机器人
        self.socket_orientation = np.array(
            self.config.get("socket_orientation", [0.0, 0.707, 0.0, 0.707])
        )

        # 插头初始位置（相对于机械臂base）
        self.plug_initial_position = np.array(
            self.config.get("plug_initial_position", [0.3, -0.2, 0.15])
        )

        # 阈值参数
        thresholds = self.config.get("success_threshold", {})
        self.approach_distance = thresholds.get("approach", 0.05)  # 5cm - 接近阶段
        self.alignment_distance = thresholds.get("alignment", 0.02)  # 2cm - 对齐阶段
        self.insertion_distance = thresholds.get("insertion", 0.01)  # 1cm - 插入阶段
        self.orientation_threshold = thresholds.get("orientation", 0.1)  # 姿态误差阈值

        # 奖励权重
        reward_cfg = self.config.get("reward", {})
        self.approach_reward_weight = reward_cfg.get("approach_weight", 1.0)
        self.alignment_reward_weight = reward_cfg.get("alignment_weight", 2.0)
        self.insertion_reward_weight = reward_cfg.get("insertion_weight", 5.0)
        self.success_bonus = reward_cfg.get("success_bonus", 10.0)

        # 任务阶段
        self.phases = ["reach_plug", "approach_socket", "align", "insert"]
        self.current_phase = self.phases[0]

    def reset(self, env_state: Dict) -> Dict:
        """重置任务"""
        super().reset(env_state)

        # 重置阶段
        self.current_phase = self.phases[0]

        return {
            "socket_position": self.socket_position,
            "socket_orientation": self.socket_orientation,
            "plug_initial_position": self.plug_initial_position,
            "phase": self.current_phase,
            "success": False,
            "phase_rewards": {phase: 0.0 for phase in self.phases},
        }

    def compute_reward(
        self,
        prev_state: Dict,
        action: np.ndarray,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, bool, Dict]:
        """
        计算奖励

        使用多阶段奖励：
        1. reach_plug: 到达插头位置（假设插头已经在夹爪中，跳过）
        2. approach_socket: 接近插座
        3. align: 对齐插座孔
        4. insert: 插入插座
        """

        tcp_pose = next_state["tcp_pose"]
        tcp_pos = tcp_pose[:3]
        tcp_quat = tcp_pose[3:]  # [qx, qy, qz, qw]

        reward = 0.0
        done = False
        success = False

        # 计算到插座的距离
        distance_to_socket = np.linalg.norm(tcp_pos - self.socket_position)

        # 计算姿态误差（四元数距离）
        orientation_error = self._quaternion_distance(tcp_quat, self.socket_orientation)

        # 阶段状态机
        if self.current_phase == "reach_plug":
            # 假设插头已经在夹爪中，直接进入下一阶段
            self.current_phase = "approach_socket"
            reward += 1.0

        if self.current_phase == "approach_socket":
            # 接近插座：奖励减少与插座的距离
            reward = -self.approach_reward_weight * distance_to_socket

            # 如果足够接近，进入对齐阶段
            if distance_to_socket < self.approach_distance:
                self.current_phase = "align"
                reward += 3.0  # 阶段完成bonus
                info["phase_rewards"]["approach_socket"] = 3.0

        if self.current_phase == "align":
            # 对齐阶段：同时考虑位置和姿态
            position_reward = -self.alignment_reward_weight * distance_to_socket
            orientation_reward = -self.alignment_reward_weight * orientation_error
            reward = position_reward + orientation_reward

            # 如果位置和姿态都对齐，进入插入阶段
            if (distance_to_socket < self.alignment_distance and
                    orientation_error < self.orientation_threshold):
                self.current_phase = "insert"
                reward += 5.0  # 阶段完成bonus
                info["phase_rewards"]["align"] = 5.0

        if self.current_phase == "insert":
            # 插入阶段：需要非常精确的位置
            reward = -self.insertion_reward_weight * distance_to_socket

            # 判断是否成功插入
            if (distance_to_socket < self.insertion_distance and
                    orientation_error < self.orientation_threshold):
                success = True
                done = True
                reward += self.success_bonus
                info["phase_rewards"]["insert"] = self.success_bonus

        # 更新 info
        info["phase"] = self.current_phase
        info["success"] = success
        info["distance_to_socket"] = distance_to_socket
        info["orientation_error"] = orientation_error

        return reward, done, info

    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        计算两个四元数之间的距离

        Args:
            q1: 四元数1 [qx, qy, qz, qw]
            q2: 四元数2 [qx, qy, qz, qw]

        Returns:
            距离（0-2之间）
        """
        # 归一化
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)

        # 计算点积
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)

        # 距离 = 1 - |q1 · q2|
        # 完全相同：distance = 0
        # 完全相反：distance ≈ 1
        return 1.0 - dot
