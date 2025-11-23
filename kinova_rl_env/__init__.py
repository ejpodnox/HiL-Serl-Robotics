"""
Kinova RL Environment Package

提供 Kinova Gen3 机械臂的 Gym 环境接口，支持多种相机后端。
"""

from .kinova_env.kinova_env import KinovaEnv
from .kinova_env.kinova_interface import KinovaInterface
from .kinova_env.camera_interface import (
    CameraInterface,
    RealSenseCamera,
    WebCamera,
    DummyCamera,
)
from .kinova_env.config_loader import KinovaConfig

__version__ = "1.0.0"
__all__ = [
    "KinovaEnv",
    "KinovaInterface",
    "CameraInterface",
    "RealSenseCamera",
    "WebCamera",
    "DummyCamera",
    "KinovaConfig",
]
