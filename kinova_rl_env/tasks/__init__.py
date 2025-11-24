"""
任务模块

提供各种机械臂操作任务的定义和奖励函数。
"""

from .base_task import BaseTask
from .reaching import ReachingTask
from .socket_insertion import SocketInsertionTask
from .flip_egg import FlipEggTask

# 任务注册表
TASK_REGISTRY = {
    "reaching": ReachingTask,
    "socket_insertion": SocketInsertionTask,
    "flip_egg": FlipEggTask,
}


def get_task(task_name: str, config=None):
    """
    根据任务名称获取任务实例

    Args:
        task_name: 任务名称
        config: 任务配置

    Returns:
        BaseTask 实例
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"未知任务: {task_name}. 可用任务: {list(TASK_REGISTRY.keys())}")

    task_class = TASK_REGISTRY[task_name]
    return task_class(config)


__all__ = [
    "BaseTask",
    "ReachingTask",
    "SocketInsertionTask",
    "FlipEggTask",
    "TASK_REGISTRY",
    "get_task",
]
