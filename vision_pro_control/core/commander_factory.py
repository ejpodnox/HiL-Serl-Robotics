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
            - 'auto': 自动选择（优先 kortex_api，失败则用 ros2）
            - 'kortex': 使用 Kortex API（推荐）
            - 'ros2': 使用 ROS2 控制器
        node_name: ROS2 节点名称（仅用于 ros2 后端）
        username: Kortex API 用户名（仅用于 kortex 后端）
        password: Kortex API 密码（仅用于 kortex 后端）

    Returns:
        RobotCommander 或 KortexCommander 实例

    Examples:
        >>> # 自动选择（推荐）
        >>> commander = create_commander('192.168.8.10')

        >>> # 强制使用 Kortex API
        >>> commander = create_commander('192.168.8.10', backend='kortex')

        >>> # 强制使用 ROS2
        >>> commander = create_commander('192.168.8.10', backend='ros2')
    """

    if backend == 'auto':
        # 优先尝试 Kortex API
        try:
            from .kortex_commander import KortexCommander, KORTEX_API_AVAILABLE

            if KORTEX_API_AVAILABLE:
                print(f"✓ 使用 Kortex API 控制器")
                return KortexCommander(
                    robot_ip=robot_ip,
                    username=username,
                    password=password
                )
        except Exception as e:
            print(f"⚠️  Kortex API 不可用: {e}")
            print("  回退到 ROS2 控制器")

        # 回退到 ROS2
        from .robot_commander import RobotCommander
        print(f"✓ 使用 ROS2 控制器")
        return RobotCommander(robot_ip=robot_ip, node_name=node_name)

    elif backend == 'kortex':
        # 强制使用 Kortex API
        from .kortex_commander import KortexCommander
        print(f"✓ 使用 Kortex API 控制器")
        return KortexCommander(
            robot_ip=robot_ip,
            username=username,
            password=password
        )

    elif backend == 'ros2':
        # 强制使用 ROS2
        from .robot_commander import RobotCommander
        print(f"✓ 使用 ROS2 控制器")
        return RobotCommander(robot_ip=robot_ip, node_name=node_name)

    else:
        raise ValueError(
            f"未知的 backend: {backend}。"
            f"支持的选项: 'auto', 'kortex', 'ros2'"
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
