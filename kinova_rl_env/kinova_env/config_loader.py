#!/usr/bin/env python3
"""
Config loader for KinovaEnv
支持从YAML文件加载配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class KinovaConfig:
    """
    Kinova环境配置类
    
    使用示例:
        # 从文件加载
        config = KinovaConfig.from_yaml("config/kinova_config.yaml")
        
        # 访问配置
        freq = config.control.frequency
        topics = config.ros2.joint_state_topic
        
        # 覆盖配置
        config = KinovaConfig.from_yaml("config.yaml", 
                                        overrides={"control.frequency": 50})
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """从字典初始化配置"""
        self._config = config_dict
        
        # 创建嵌套访问属性
        self.robot = self._DictWrapper(config_dict.get('robot', {}))
        self.ros2 = self._DictWrapper(config_dict.get('ros2', {}))
        self.control = self._DictWrapper(config_dict.get('control', {}))
        self.observation = self._DictWrapper(config_dict.get('observation', {}))
        self.action = self._DictWrapper(config_dict.get('action', {}))
        self.reward = self._DictWrapper(config_dict.get('reward', {}))
        self.camera = self._DictWrapper(config_dict.get('camera', {}))
        self.logging = self._DictWrapper(config_dict.get('logging', {}))
        self.debug = self._DictWrapper(config_dict.get('debug', {}))
        
        # 自动计算dt (如果没有手动设置)
        if 'dt' not in self._config.get('control', {}):
            self.control.dt = 1.0 / self.control.frequency
    
    class _DictWrapper:
        """字典包装类，支持点访问"""
        def __init__(self, d: Dict):
            self._dict = d
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, KinovaConfig._DictWrapper(value))
                else:
                    setattr(self, key, value)
        
        def __getitem__(self, key):
            return self._dict[key]
        
        def get(self, key, default=None):
            return self._dict.get(key, default)
    
    @classmethod
    def from_yaml(
        cls, 
        yaml_path: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> "KinovaConfig":
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            overrides: 覆盖配置，格式如 {"control.frequency": 50}
        
        Returns:
            KinovaConfig实例
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        
        # 加载YAML
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 应用覆盖
        if overrides:
            config_dict = cls._apply_overrides(config_dict, overrides)
        
        return cls(config_dict)
    
    @staticmethod
    def _apply_overrides(
        config: Dict[str, Any], 
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        应用配置覆盖
        
        示例:
            overrides = {"control.frequency": 50, "robot.ip": "192.168.1.20"}
            会修改 config['control']['frequency'] = 50
        """
        for key, value in overrides.items():
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        return config
    
    @classmethod
    def get_default(cls) -> "KinovaConfig":
        """
        返回默认配置
        
        当没有config文件时使用
        """
        default_config = {
            'robot': {
                'ip': '192.168.1.10',
                'dof': 7,
                'joint_limits': {
                    'velocity_max': [1.3, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2]
                },
                'home_position': [0.0, 0.26, 0.0, -1.57, 0.0, 1.31, 0.0]
            },
            'ros2': {
                'joint_state_topic': '/joint_states',
                'velocity_command_topic': '/velocity_controller/commands',
                'node_name': 'kinova_rl_env'
            },
            'control': {
                'frequency': 20,
                'action_scale': 0.5,
                'max_episode_steps': 500,
                'reset_timeout': 5.0
            },
            'observation': {
                'include_joint_positions': True,
                'include_joint_velocities': True,
                'state_dim': 14
            },
            'action': {
                'type': 'joint_velocity',
                'dim': 7,
                'low': -1.0,
                'high': 1.0
            },
            'reward': {
                'type': 'sparse'
            },
            'camera': {
                'enabled': False
            }
        }
        return cls(default_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config
    
    def save(self, path: str):
        """保存配置到YAML文件"""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def __repr__(self):
        return f"KinovaConfig({yaml.dump(self._config, default_flow_style=False)})"


# ============ 测试 ============
if __name__ == '__main__':
    # 测试1: 加载默认配置
    print("=== 测试1: 默认配置 ===")
    config = KinovaConfig.get_default()
    print(f"Control频率: {config.control.frequency}")
    print(f"机器人IP: {config.robot.ip}")
    print(f"Action维度: {config.action.dim}")
    
    # 测试2: 从YAML加载 (如果文件存在)
    yaml_path = "config/kinova_config.yaml"
    try:
        print(f"\n=== 测试2: 从{yaml_path}加载 ===")
        config = KinovaConfig.from_yaml(yaml_path)
        print(f"Joint state topic: {config.ros2.joint_state_topic}")
        print(f"Max episode steps: {config.control.max_episode_steps}")
    except FileNotFoundError:
        print(f"文件不存在: {yaml_path}")
    
    # 测试3: 覆盖配置
    print("\n=== 测试3: 覆盖配置 ===")
    config = KinovaConfig.get_default()
    config_override = KinovaConfig.from_yaml(
        yaml_path,
        overrides={
            "control.frequency": 50,
            "robot.ip": "192.168.1.20"
        }
    ) if Path(yaml_path).exists() else None
    
    if config_override:
        print(f"覆盖后频率: {config_override.control.frequency}")
        print(f"覆盖后IP: {config_override.robot.ip}")