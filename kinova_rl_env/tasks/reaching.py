"""
Reaching 任务

简单的到达目标位置任务。
"""

import numpy as np
from typing import Dict, Tuple
from .base_task import BaseTask


class ReachingTask(BaseTask):
    """
    Reaching 任务

    目标：让机械臂末端到达指定的目标位置。
    """

    def __init__(self, config=None):
        super().__init__(config)

        # 从配置读取参数
        self.target_pose = np.array(
            self.config.get("target_pose", [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])
        )
        self.success_threshold = self.config.get("success_threshold", {})
        self.position_threshold = self.success_threshold.get("position", 0.02)  # 2cm

        # 奖励参数
        self.reward_type = self.config.get("reward", {}).get("type", "sparse")

    def reset(self, env_state: Dict) -> Dict:
        """重置任务"""
        super().reset(env_state)

        return {
            "target_pose": self.target_pose,
            "success": False,
        }

    def compute_reward(
        self,
        prev_state: Dict,
        action: np.ndarray,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, bool, Dict]:
        """计算奖励"""

        # 获取当前 TCP 位置
        tcp_pose = next_state["tcp_pose"]  # (7,) [x, y, z, qx, qy, qz, qw]
        tcp_pos = tcp_pose[:3]
        target_pos = self.target_pose[:3]

        # 计算距离
        distance = np.linalg.norm(tcp_pos - target_pos)

        # 判断是否成功
        success = distance < self.position_threshold
        info["success"] = success
        info["distance"] = distance

        # 计算奖励
        if self.reward_type == "sparse":
            reward = 1.0 if success else 0.0
        elif self.reward_type == "dense":
            # 距离越小，奖励越大
            reward = -distance
            if success:
                reward += 1.0  # 成功bonus
        else:
            reward = 0.0

        # 终止条件
        done = success

        return reward, done, info
