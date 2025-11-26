"""
机器人控制器工厂函数

提供统一接口来创建 RobotCommander 或 KortexCommander
"""

from typing import Optional


def create_commander(
    robot_ip: str,
    backend: str = 'auto',
    node_name: str = 'robot_commander',
    username: str = 'admin',
    password: str = 'admin'
):
    """
    创建机器人控制器（自动选择或指定后端）

    Args:
        robot_ip: 机械臂 IP 地址
        backend: 控制后端
            - 'auto': 自动选择（优先 joint，最可靠）
            - 'joint': 使用 joint_trajectory_controller（推荐）
            - 'ros2': 使用 ROS2 twist_controller
            - 'kortex': 使用 Kortex API（需手动安装）
        node_name: ROS2 节点名称
        username: Kortex API 用户名（仅用于 kortex 后端）
        password: Kortex API 密码（仅用于 kortex 后端）

    Returns:
        Commander 实例

    Examples:
        >>> # 自动选择（推荐）
        >>> commander = create_commander('192.168.8.10')

        >>> # 使用 joint 控制
        >>> commander = create_commander('192.168.8.10', backend='joint')
    """

    if backend == 'auto':
        # 优先使用 joint（最可靠）
        from .joint_velocity_commander import JointVelocityCommander
        return JointVelocityCommander(robot_ip=robot_ip, node_name=node_name)

    elif backend == 'joint':
        from .joint_velocity_commander import JointVelocityCommander
        return JointVelocityCommander(robot_ip=robot_ip, node_name=node_name)

    elif backend == 'ros2':
        from .robot_commander import RobotCommander
        return RobotCommander(robot_ip=robot_ip, node_name=node_name)

    elif backend == 'kortex':
        from .kortex_commander import KortexCommander
        return KortexCommander(
            robot_ip=robot_ip,
            username=username,
            password=password
        )

    else:
        raise ValueError(
            f"未知的 backend: {backend}。"
            f"支持的选项: 'auto', 'joint', 'ros2', 'kortex'"
        )


def create_commander_from_config(config: dict):
    """
    从配置字典创建控制器

    Args:
        config: 配置字典，格式:
            {
                'robot': {
                    'ip': '192.168.8.10',
                    'control_backend': 'auto'  # 可选
                }
            }

    Returns:
        Commander 实例
    """
    robot_config = config.get('robot', {})
    robot_ip = robot_config.get('ip', '192.168.8.10')
    backend = robot_config.get('control_backend', 'auto')

    return create_commander(robot_ip=robot_ip, backend=backend)
