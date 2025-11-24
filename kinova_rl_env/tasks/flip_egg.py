"""
Flip Egg 任务

反转鸡蛋任务（需要精细控制和力反馈）。
"""

import numpy as np
from typing import Dict, Tuple
from .base_task import BaseTask


class FlipEggTask(BaseTask):
    """
    Flip Egg 任务

    任务描述：
    1. 检测鸡蛋位置
    2. 轻柔地抓取鸡蛋（不能太用力，否则会破）
    3. 抬起鸡蛋
    4. 翻转180度
    5. 轻放回桌面
    6. 松开

    这是一个需要精细力控制的复杂任务。
    """

    def __init__(self, config=None):
        super().__init__(config)

        # 鸡蛋初始位置（桌面上）
        self.egg_initial_position = np.array(
            self.config.get("egg_initial_position", [0.4, 0.0, 0.05])  # 桌面高度5cm
        )

        # 鸡蛋初始方向（直立，尖端朝上）
        # 四元数 [qx, qy, qz, qw]
        self.egg_initial_orientation = np.array(
            self.config.get("egg_initial_orientation", [0.0, 0.0, 0.0, 1.0])
        )

        # 目标方向（翻转180度，尖端朝下）
        self.egg_target_orientation = np.array(
            self.config.get("egg_target_orientation", [1.0, 0.0, 0.0, 0.0])
        )

        # 抓取高度（鸡蛋中心位置）
        self.grasp_height = self.config.get("grasp_height", 0.08)  # 8cm

        # 翻转高度（抬起后翻转的高度）
        self.flip_height = self.config.get("flip_height", 0.20)  # 20cm

        # 放置高度（放回桌面）
        self.place_height = self.config.get("place_height", 0.05)  # 5cm

        # 阈值参数
        thresholds = self.config.get("success_threshold", {})
        self.approach_threshold = thresholds.get("approach", 0.03)  # 3cm
        self.grasp_threshold = thresholds.get("grasp", 0.01)  # 1cm
        self.lift_threshold = thresholds.get("lift", 0.02)  # 2cm
        self.flip_orientation_threshold = thresholds.get("flip_orientation", 0.2)  # 姿态阈值
        self.place_threshold = thresholds.get("place", 0.02)  # 2cm

        # 夹爪力度阈值（模拟，实际需要力传感器）
        self.safe_gripper_force_min = self.config.get("safe_gripper_force_min", 0.3)
        self.safe_gripper_force_max = self.config.get("safe_gripper_force_max", 0.7)

        # 奖励权重
        reward_cfg = self.config.get("reward", {})
        self.approach_weight = reward_cfg.get("approach_weight", 1.0)
        self.grasp_weight = reward_cfg.get("grasp_weight", 2.0)
        self.lift_weight = reward_cfg.get("lift_weight", 1.5)
        self.flip_weight = reward_cfg.get("flip_weight", 3.0)
        self.place_weight = reward_cfg.get("place_weight", 2.0)
        self.success_bonus = reward_cfg.get("success_bonus", 20.0)

        # 失败惩罚
        self.drop_penalty = reward_cfg.get("drop_penalty", -10.0)
        self.break_penalty = reward_cfg.get("break_penalty", -20.0)

        # 任务阶段
        self.phases = ["approach", "grasp", "lift", "flip", "place", "release"]
        self.current_phase = self.phases[0]

        # 鸡蛋状态跟踪
        self.egg_grasped = False
        self.egg_broken = False
        self.egg_dropped = False

    def reset(self, env_state: Dict) -> Dict:
        """重置任务"""
        super().reset(env_state)

        # 重置阶段和状态
        self.current_phase = self.phases[0]
        self.egg_grasped = False
        self.egg_broken = False
        self.egg_dropped = False

        return {
            "egg_position": self.egg_initial_position.copy(),
            "egg_orientation": self.egg_initial_orientation.copy(),
            "target_orientation": self.egg_target_orientation.copy(),
            "phase": self.current_phase,
            "success": False,
            "egg_grasped": False,
            "egg_broken": False,
            "egg_dropped": False,
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

        多阶段奖励：
        1. approach: 接近鸡蛋
        2. grasp: 抓取鸡蛋（需要合适的力度）
        3. lift: 抬起鸡蛋到翻转高度
        4. flip: 翻转鸡蛋180度
        5. place: 放回桌面
        6. release: 松开夹爪
        """

        tcp_pose = next_state["tcp_pose"]
        tcp_pos = tcp_pose[:3]
        tcp_quat = tcp_pose[3:]

        # 从 info 获取鸡蛋当前状态
        egg_pos = info.get("egg_position", self.egg_initial_position)
        egg_quat = info.get("egg_orientation", self.egg_initial_orientation)

        # 夹爪位置（简化：假设从观测中能获取，实际可能需要FK）
        gripper_pos = next_state.get("gripper_position", 0.5)  # 0=关闭, 1=打开

        reward = 0.0
        done = False
        success = False

        # 检查鸡蛋是否破碎（力度过大）
        if not self.egg_broken and self.egg_grasped:
            # 简化的破碎检测：如果夹爪闭合过紧
            if gripper_pos < 0.2:  # 闭合太紧
                self.egg_broken = True
                reward += self.break_penalty
                done = True
                info["egg_broken"] = True
                info["phase"] = "failed_broken"
                return reward, done, info

        # 检查鸡蛋是否掉落（位置突然下降）
        if self.egg_grasped and not self.egg_dropped:
            # 简化的掉落检测：鸡蛋高度突然降到桌面以下
            if egg_pos[2] < 0.02:  # 低于2cm
                self.egg_dropped = True
                reward += self.drop_penalty
                done = True
                info["egg_dropped"] = True
                info["phase"] = "failed_dropped"
                return reward, done, info

        # === 阶段状态机 ===

        if self.current_phase == "approach":
            # 接近鸡蛋
            distance_to_egg = np.linalg.norm(tcp_pos - egg_pos)
            reward = -self.approach_weight * distance_to_egg

            # 如果足够接近，进入抓取阶段
            if distance_to_egg < self.approach_threshold:
                self.current_phase = "grasp"
                reward += 2.0

        elif self.current_phase == "grasp":
            # 抓取鸡蛋
            distance_to_egg = np.linalg.norm(tcp_pos - egg_pos)

            # 奖励：接近鸡蛋 + 合适的夹爪力度
            position_reward = -self.grasp_weight * distance_to_egg

            # 夹爪力度奖励（鼓励合适的力度）
            if self.safe_gripper_force_min < gripper_pos < self.safe_gripper_force_max:
                force_reward = 1.0
            else:
                force_reward = -0.5

            reward = position_reward + force_reward

            # 如果位置正确且夹爪力度合适，认为抓取成功
            if (distance_to_egg < self.grasp_threshold and
                    self.safe_gripper_force_min < gripper_pos < self.safe_gripper_force_max):
                self.egg_grasped = True
                self.current_phase = "lift"
                reward += 5.0
                info["egg_grasped"] = True

                # 鸡蛋位置跟随TCP（简化）
                info["egg_position"] = tcp_pos.copy()

        elif self.current_phase == "lift":
            # 抬起鸡蛋到翻转高度
            target_lift_pos = np.array([egg_pos[0], egg_pos[1], self.flip_height])
            distance_to_lift = np.linalg.norm(tcp_pos - target_lift_pos)
            reward = -self.lift_weight * distance_to_lift

            # 鸡蛋跟随TCP
            if self.egg_grasped:
                info["egg_position"] = tcp_pos.copy()
                egg_pos = tcp_pos

            # 如果达到翻转高度，进入翻转阶段
            if tcp_pos[2] > (self.flip_height - self.lift_threshold):
                self.current_phase = "flip"
                reward += 3.0

        elif self.current_phase == "flip":
            # 翻转鸡蛋180度
            # 姿态误差
            orientation_error = self._quaternion_distance(tcp_quat, self.egg_target_orientation)
            reward = -self.flip_weight * orientation_error

            # 鸡蛋跟随TCP
            if self.egg_grasped:
                info["egg_position"] = tcp_pos.copy()
                info["egg_orientation"] = tcp_quat.copy()
                egg_pos = tcp_pos
                egg_quat = tcp_quat

            # 如果姿态接近目标，进入放置阶段
            if orientation_error < self.flip_orientation_threshold:
                self.current_phase = "place"
                reward += 5.0

        elif self.current_phase == "place":
            # 放回桌面
            target_place_pos = np.array([egg_pos[0], egg_pos[1], self.place_height])
            distance_to_place = np.linalg.norm(tcp_pos - target_place_pos)
            reward = -self.place_weight * distance_to_place

            # 鸡蛋跟随TCP
            if self.egg_grasped:
                info["egg_position"] = tcp_pos.copy()
                egg_pos = tcp_pos

            # 如果接近桌面，进入松开阶段
            if tcp_pos[2] < (self.place_height + self.place_threshold):
                self.current_phase = "release"
                reward += 3.0

        elif self.current_phase == "release":
            # 松开夹爪
            # 鼓励打开夹爪
            if gripper_pos > 0.8:  # 夹爪打开
                success = True
                done = True
                reward += self.success_bonus
                info["success"] = True
            else:
                reward = -1.0 + gripper_pos  # 鼓励打开

        # 更新 info
        info["phase"] = self.current_phase

        return reward, done, info

    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        计算两个四元数之间的距离

        Args:
            q1: 四元数1 [qx, qy, qz, qw]
            q2: 四元数2 [qx, qy, qz, qw]

        Returns:
            距离（0-1之间）
        """
        # 归一化
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)

        # 计算点积
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)

        # 距离 = 1 - |q1 · q2|
        return 1.0 - dot
