"""
VisionPro Control Package

提供 Apple Vision Pro 遥操作的完整功能，包括数据接收、坐标映射、机械臂控制。
"""

from .core.visionpro_bridge import VisionProBridge
from .core.coordinate_mapper import CoordinateMapper
from .core.robot_commander import RobotCommander
from .core.calibrator import WorkspaceCalibrator

__version__ = "1.0.0"
__all__ = [
    "VisionProBridge",
    "CoordinateMapper",
    "RobotCommander",
    "WorkspaceCalibrator",
]
