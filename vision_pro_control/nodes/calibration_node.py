"""
VisionPro 工作空间标定节点
简化版：只标定一个中心点，使用球形工作空间模型
"""
import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import time
from pathlib import Path

# 导入自定义模块
from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.calibrator import WorkspaceCalibrator
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


class CalibrationNode(Node):
    """VisionPro 工作空间标定节点"""
    
    def __init__(self, config_file: str = None):
        super().__init__('visionpro_calibration_node')
        
        self.get_logger().info('初始化标定节点...')
        
        # 加载配置
        if config_file:
            self.config = self.load_config(config_file)
        else:
            self.config = self.default_config()
        
        # 初始化 VisionPro 连接
        self.visionpro_bridge = VisionProBridge(
            avp_ip=self.config['visionpro']['ip'],
            use_right_hand=self.config['visionpro']['use_right_hand']
        )
        
        # 初始化标定器
        self.calibrator = WorkspaceCalibrator(
            control_radius=self.config.get('workspace', {}).get('control_radius', 0.25),
            deadzone_radius=self.config.get('workspace', {}).get('deadzone_radius', 0.03)
        )
        
        # 输出文件路径
        self.output_file = Path(self.config.get('calibration', {}).get(
            'file', 'config/calibration.yaml'
        ))
        
        # 采样状态
        self.is_sampling = False
        self.sample_count = 0
        
        self.get_logger().info('标定节点初始化完成')
        
    def default_config(self) -> dict:
        """默认配置"""
        return {
            'visionpro': {
                'ip': '10.31.181.201',
                'use_right_hand': True
            },
            'workspace': {
                'control_radius': 0.25,
                'deadzone_radius': 0.03
            },
            'calibration': {
                'file': 'config/calibration.yaml'
            }
        }
        
    def load_config(self, config_file: str) -> dict:
        """加载配置文件"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.get_logger().warn(f'配置文件不存在: {config_file}，使用默认配置')
            return self.default_config()
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.get_logger().info(f'已加载配置文件: {config_file}')
        return config
        
    def start(self):
        """启动标定"""
        self.get_logger().info('启动 VisionPro 连接...')
        self.visionpro_bridge.start()
        time.sleep(1.0)  # 等待连接稳定
        self.get_logger().info('VisionPro 连接成功')
        
    def stop(self):
        """停止标定"""
        self.visionpro_bridge.stop()
        self.get_logger().info('VisionPro 连接已断开')
        
    def add_sample(self):
        """添加当前手部位姿作为采样"""
        try:
            position, rotation = self.visionpro_bridge.get_hand_relative_to_head()
            self.calibrator.add_sample(position, rotation)
            self.sample_count += 1
            
            self.get_logger().info(
                f'采样 #{self.sample_count}: 位置 = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]'
            )
            
        except Exception as e:
            self.get_logger().error(f'采样失败: {e}')
            
    def save_center(self):
        """保存中心点"""
        if self.calibrator.save_center():
            self.get_logger().info('✓ 中心点已保存')
            self.sample_count = 0
        else:
            self.get_logger().error('✗ 保存中心点失败')
            
    def clear_samples(self):
        """清空采样"""
        self.calibrator.clear_samples()
        self.sample_count = 0
        self.get_logger().info('✓ 采样已清空')
        
    def save_calibration(self, overwrite: bool = True):
        """保存标定数据到文件"""
        if not self.calibrator.is_complete():
            self.get_logger().error('✗ 标定未完成，无法保存')
            return False
            
        # 确保路径正确
        output_path = Path(__file__).parent.parent.parent / self.output_file
        
        if self.calibrator.save_to_file(output_path, overwrite=overwrite):
            self.get_logger().info(f'✓ 标定数据已保存到: {output_path}')
            return True
        else:
            self.get_logger().error('✗ 保存失败')
            return False
            
    def set_workspace_params(self, control_radius: float = None, deadzone_radius: float = None):
        """设置工作空间参数"""
        self.calibrator.set_workspace_params(control_radius, deadzone_radius)
        
    def print_current_pose(self):
        """打印当前手部位姿"""
        try:
            position, rotation = self.visionpro_bridge.get_hand_relative_to_head()
            print(f"\n当前手部位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            
            # 计算与中心的距离（如果已标定）
            if self.calibrator.center_position is not None:
                distance = np.linalg.norm(position - self.calibrator.center_position)
                print(f"距离中心: {distance:.4f} m")
                
                if distance < self.calibrator.deadzone_radius:
                    print("状态: 死区内 (机械臂静止)")
                elif distance < self.calibrator.control_radius:
                    print("状态: 工作区 (正常控制)")
                else:
                    print("状态: 超出范围 (速度饱和)")
                    
        except Exception as e:
            print(f"获取位姿失败: {e}")
            
    def print_status(self):
        """打印标定状态"""
        self.calibrator.print_status()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VisionPro 工作空间标定')
    parser.add_argument(
        '--config',
        type=str,
        default='config/teleop_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（覆盖配置文件中的设置）'
    )
    args = parser.parse_args()
    
    # 初始化 ROS2
    rclpy.init()
    
    try:
        # 创建节点
        node = CalibrationNode(config_file=args.config)
        
        # 覆盖输出路径
        if args.output:
            node.output_file = Path(args.output)
        
        # 启动
        node.start()
        
        # 打印帮助
        print("\n" + "="*60)
        print("VisionPro 工作空间标定 (单点标定)")
        print("="*60)
        print("\n按键说明:")
        print("  's' - 添加采样（建议采样 5-10 次）")
        print("  'c' - 保存中心点（计算所有采样的平均值）")
        print("  'x' - 清空当前采样")
        print("  'w' - 写入文件（保存标定结果）")
        print("  'p' - 打印当前手部位姿")
        print("  't' - 打印标定状态")
        print("  '1' - 设置控制半径")
        print("  '2' - 设置死区半径")
        print("  'q' - 退出")
        print("\n标定流程:")
        print("  1. 将手部移动到舒适的操作中心位置")
        print("  2. 按 's' 多次采样（5-10次）")
        print("  3. 按 'c' 保存中心点")
        print("  4. 按 'w' 写入文件")
        print("="*60 + "\n")
        
        # 键盘监听
        with KeyboardMonitor() as kb:
            while rclpy.ok():
                # 处理 ROS2 回调
                rclpy.spin_once(node, timeout_sec=0.01)
                
                # 检查按键
                key = kb.get_key(timeout=0.01)
                
                if key == 'q':
                    print("\n退出标定...")
                    break
                    
                elif key == 's':
                    node.add_sample()
                    
                elif key == 'c':
                    node.save_center()
                    
                elif key == 'x':
                    node.clear_samples()
                    
                elif key == 'w':
                    node.save_calibration(overwrite=True)
                    
                elif key == 'p':
                    node.print_current_pose()
                    
                elif key == 't':
                    node.print_status()
                    
                elif key == '1':
                    try:
                        radius = float(input("\n输入控制半径 (m): "))
                        node.set_workspace_params(control_radius=radius)
                    except ValueError:
                        print("无效输入")
                        
                elif key == '2':
                    try:
                        radius = float(input("\n输入死区半径 (m): "))
                        node.set_workspace_params(deadzone_radius=radius)
                    except ValueError:
                        print("无效输入")
                        
    except KeyboardInterrupt:
        print("\n用户中断")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        if 'node' in locals():
            node.stop()
            node.destroy_node()
        rclpy.shutdown()
        
    print("标定程序已退出")


if __name__ == '__main__':
    main()