import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional
import time

class WorkspaceCalibrator:
    """简化的工作空间标定器 - 球形工作空间模型"""
    
    def __init__(self, control_radius: float = 0.25, deadzone_radius: float = 0.03):
        """
        Args:
            control_radius: 最大控制半径 (m)
            deadzone_radius: 死区半径 (m)
        """
        self.center_position = None
        self.center_rotation = None
        self.control_radius = control_radius
        self.deadzone_radius = deadzone_radius
        
        self.samples = []  # 存储采样点
        
    def add_sample(self, position: np.ndarray, rotation: np.ndarray):
        """
        添加标定采样
        Args:
            position: 位置 (3,)
            rotation: 旋转矩阵 (3, 3)
        """
        sample = {
            'position': position.copy(),
            'rotation': rotation.copy()
        }
        self.samples.append(sample)
        print(f"✓ 已添加采样 #{len(self.samples)}")
        
    def save_center(self):
        """保存中心点（取所有采样的平均）"""
        if len(self.samples) == 0:
            print("❌ 错误：无采样数据")
            return False
            
        # 计算位置平均
        positions = [s['position'] for s in self.samples]
        self.center_position = np.mean(positions, axis=0)
        
        # 计算旋转平均（简单平均，实际应用可考虑四元数平均）
        rotations = [s['rotation'] for s in self.samples]
        self.center_rotation = np.mean(rotations, axis=0)
        
        print(f"✓ 已保存中心点")
        print(f"  位置: {self.center_position}")
        print(f"  基于 {len(self.samples)} 个采样")
        
        self.samples = []  # 清空采样
        return True
        
    def clear_samples(self):
        """清空当前采样"""
        self.samples = []
        print("✓ 已清空采样")
        
    def is_complete(self) -> bool:
        """检查标定是否完成"""
        return (self.center_position is not None and 
                self.center_rotation is not None)
    
    def set_workspace_params(self, control_radius: float = None, deadzone_radius: float = None):
        """设置工作空间参数"""
        if control_radius is not None:
            self.control_radius = control_radius
            print(f"✓ 控制半径: {self.control_radius} m")
            
        if deadzone_radius is not None:
            self.deadzone_radius = deadzone_radius
            print(f"✓ 死区半径: {self.deadzone_radius} m")
    
    def save_to_file(self, filepath: Path, overwrite: bool = False):
        """保存标定数据到文件"""
        filepath = Path(filepath)
        
        if filepath.exists() and not overwrite:
            print(f"❌ 文件已存在: {filepath}")
            print("   请设置 overwrite=True")
            return False
        
        if not self.is_complete():
            print("❌ 标定未完成")
            return False
        
        # 准备数据
        data = {
            'calibration_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'workspace_center': {
                'position': self.center_position.tolist(),
                'rotation': self.center_rotation.tolist()
            },
            'workspace_params': {
                'control_radius': float(self.control_radius),
                'deadzone_radius': float(self.deadzone_radius)
            }
        }
        
        # 创建目录
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"✓ 标定数据已保存到: {filepath}")
        return True
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> Optional['WorkspaceCalibrator']:
        """从文件加载标定数据"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ 文件不存在: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # 创建标定器
        calibrator = cls()
        
        # 加载中心点
        center = data['workspace_center']
        calibrator.center_position = np.array(center['position'])
        calibrator.center_rotation = np.array(center['rotation'])
        
        # 加载参数
        params = data['workspace_params']
        calibrator.control_radius = params['control_radius']
        calibrator.deadzone_radius = params['deadzone_radius']
        
        print(f"✓ 已加载标定数据: {filepath}")
        print(f"  标定时间: {data['calibration_time']}")
        
        return calibrator
    
    def print_status(self):
        """打印标定状态"""
        print("\n" + "="*60)
        print("标定状态:")
        print("-"*60)
        
        if self.center_position is not None:
            print(f"✓ 中心位置: {self.center_position}")
        else:
            print("✗ 中心位置: 未标定")
            
        if self.center_rotation is not None:
            print(f"✓ 中心姿态: 已标定")
        else:
            print("✗ 中心姿态: 未标定")
        
        print(f"\n工作空间参数:")
        print(f"  控制半径: {self.control_radius} m")
        print(f"  死区半径: {self.deadzone_radius} m")
        
        if len(self.samples) > 0:
            print(f"\n当前采样: {len(self.samples)} 个")
        
        print("="*60 + "\n")