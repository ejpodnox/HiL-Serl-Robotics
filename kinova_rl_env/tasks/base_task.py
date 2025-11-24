"""
基础任务类

所有任务都应该继承这个基类。
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any


class BaseTask(ABC):
    """
    任务基类

    定义了所有任务必须实现的接口。
    """

    def __init__(self, config=None):
        """
        初始化任务

        Args:
            config: 任务配置（通常来自 YAML 文件的 task 部分）
        """
        self.config = config or {}
        self.episode_step = 0
        self.episode_data = {}  # 用于存储 episode 级别的数据

    @abstractmethod
    def reset(self, env_state: Dict) -> Dict:
        """
        重置任务

        Args:
            env_state: 当前环境状态，包含：
                - tcp_pose: (7,) [x, y, z, qx, qy, qz, qw]
                - joint_positions: (7,)
                - joint_velocities: (7,)
                - gripper_position: float
                - images: Dict[str, np.ndarray]

        Returns:
            任务信息字典，包含：
                - target_pose: 目标位姿（如果有）
                - phase: 当前阶段（多阶段任务）
                - 其他任务相关信息
        """
        self.episode_step = 0
        self.episode_data = {}
        return {}

    @abstractmethod
    def compute_reward(
        self,
        prev_state: Dict,
        action: np.ndarray,
        next_state: Dict,
        info: Dict
    ) -> Tuple[float, bool, Dict]:
        """
        计算奖励

        Args:
            prev_state: 上一步的状态
            action: 执行的动作
            next_state: 执行动作后的状态
            info: 任务信息（来自 reset 或上一步）

        Returns:
            reward: 奖励值
            done: 是否完成（成功或失败）
            info: 更新后的任务信息
        """
        raise NotImplementedError

    def step(self, prev_state: Dict, action: np.ndarray, next_state: Dict, info: Dict):
        """
        任务步进

        Args:
            prev_state: 上一步状态
            action: 执行的动作
            next_state: 下一步状态
            info: 任务信息

        Returns:
            reward, done, info
        """
        self.episode_step += 1
        reward, done, info = self.compute_reward(prev_state, action, next_state, info)
        return reward, done, info

    def is_success(self, state: Dict, info: Dict) -> bool:
        """
        判断是否成功完成任务

        Args:
            state: 当前状态
            info: 任务信息

        Returns:
            是否成功
        """
        return info.get("success", False)

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取任务度量指标

        Returns:
            度量指标字典
        """
        return {
            "episode_steps": self.episode_step,
        }
