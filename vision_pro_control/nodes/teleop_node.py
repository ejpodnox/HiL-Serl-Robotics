"""
VisionPro 遥操作主节点（安全版）
- 启动时自动标定
- 按键控制开始/暂停
- 超低速安全参数
"""
import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import time
import threading
from pathlib import Path

# 导入自定义模块
from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.coordinate_mapper import CoordinateMapper
from vision_pro_control.core import robot_commander
from vision_pro_control.core.calibrator import WorkspaceCalibrator
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


class TeleopNode(Node):
    """VisionPro 遥操作节点（安全版）"""
    
    def __init__(self, config_file: str):
        super().__init__('visionpro_teleop_node')

        self.get_logger().info('初始化 VisionPro 遥操作节点（安全版）...')

        # 保存配置文件路径
        self.config_file = config_file

        # 加载配置
        self.config = self.load_config(config_file)
        
        # 初始化组件
        self.visionpro_bridge = None
        self.coordinate_mapper = None
        self.robot_commander = None
        self.calibrator = None
        
        # 状态变量
        self.is_running = False
        self.is_control_enabled = False  # 默认暂停
        self.is_calibrated = False
        self.last_gripper_position = 0.0
        self.control_thread = None
        
        # 统计信息
        self.loop_count = 0
        self.start_time = None
        
        self.init_components()
        
        self.get_logger().info('遥操作节点初始化完成')
        
    def load_config(self, config_file: str) -> dict:
        """加载配置文件"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.get_logger().error(f'配置文件不存在: {config_file}')
            raise FileNotFoundError(f'配置文件不存在: {config_file}')
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.get_logger().info(f'已加载配置文件: {config_file}')
        return config
        
    def init_components(self):
        """初始化所有组件"""
        # VisionPro 连接
        self.visionpro_bridge = VisionProBridge(
            avp_ip=self.config['visionpro']['ip'],
            use_right_hand=self.config['visionpro']['use_right_hand']
        )
        self.get_logger().info(f"VisionPro IP: {self.config['visionpro']['ip']}")

        # 标定器
        self.calibrator = WorkspaceCalibrator(
            control_radius=0.25,
            deadzone_radius=0.03
        )

        # 机械臂控制器（ROS2 节点需要单独处理）
        # 这里我们先不初始化，等标定完成后再初始化
        
        self.get_logger().info('组件初始化完成')
        
    def init_robot_commander(self):
        """初始化机械臂控制器"""
        self.robot_commander = robot_commander(
            robot_ip=self.config['robot']['ip'],
            config_file=self.config_file
        )

        # 等待 TF buffer 填充数据，需要 spin 让节点接收消息
        self.get_logger().info("等待 TF buffer 准备...")
        end_time = time.time() + 2.0
        while time.time() < end_time:
            rclpy.spin_once(self.robot_commander, timeout_sec=0.1)

        # 设置超低速安全参数
        safety_config = self.config['safety']
        self.robot_commander.safety_max_linear_vel = safety_config['max_linear_velocity']
        self.robot_commander.safety_max_angular_vel = safety_config['max_angular_velocity']
        self.robot_commander.enable_safety_check = safety_config['enable_safety_check']

        self.get_logger().info(f"机械臂已连接: {self.config['robot']['ip']}")
        self.get_logger().info(f"安全限速: {safety_config['max_linear_velocity']} m/s")
        
    def init_coordinate_mapper(self, calibration_file: Path):
        """初始化坐标映射器"""
        self.coordinate_mapper = CoordinateMapper(calibration_file=calibration_file)
        
        mapper_config = self.config['mapper']
        self.coordinate_mapper.set_gains(
            position_gain=mapper_config['position_gain'],
            rotation_gain=mapper_config['rotation_gain']
        )
        self.coordinate_mapper.set_velocity_limits(
            max_linear=mapper_config['max_linear_velocity'],
            max_angular=mapper_config['max_angular_velocity']
        )
        self.coordinate_mapper.set_filter_alpha(mapper_config['filter_alpha'])
        
        self.get_logger().info("坐标映射器已初始化")
        
    def run_calibration(self):
        """运行标定流程"""
        print("\n" + "="*60)
        print("【标定模式】")
        print("="*60)
        print("按键说明:")
        print("  's' - 添加采样（建议 5-10 次）")
        print("  'c' - 保存中心点")
        print("  'p' - 打印当前位姿")
        print("  'Enter' - 确认标定完成，进入遥操作模式")
        print("  'q' - 退出程序")
        print("="*60)
        print("\n请将手移动到舒适的操作中心位置，然后按 's' 采样\n")
        
        sample_count = 0
        
        with KeyboardMonitor() as kb:
            while rclpy.ok():
                key = kb.get_key(timeout=0.05)
                
                if not key:
                    continue
                    
                if key == 'q':
                    print("\n退出程序")
                    return False
                    
                elif key == 's':
                    # 添加采样
                    try:
                        position, rotation = self.visionpro_bridge.get_hand_relative_to_head()
                        self.calibrator.add_sample(position, rotation)
                        sample_count += 1
                        print(f"✓ 采样 #{sample_count}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
                    except Exception as e:
                        print(f"✗ 采样失败: {e}")
                        
                elif key == 'c':
                    # 保存中心点
                    if self.calibrator.save_center():
                        sample_count = 0
                        print("✓ 中心点已保存，按 Enter 确认完成标定")
                        
                elif key == 'p':
                    # 打印当前位姿
                    try:
                        position, rotation = self.visionpro_bridge.get_hand_relative_to_head()
                        print(f"\n当前位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
                        
                        if self.calibrator.center_position is not None:
                            distance = np.linalg.norm(position - self.calibrator.center_position)
                            print(f"距离中心: {distance:.4f} m")
                    except Exception as e:
                        print(f"获取位姿失败: {e}")
                        
                elif key == '\n' or key == '\r':
                    # Enter 键确认
                    if self.calibrator.is_complete():
                        # 保存标定文件
                        calibration_file = Path(__file__).parent.parent / "config" / "calibration.yaml"
                        self.calibrator.save_to_file(calibration_file, overwrite=True)
                        
                        print("\n✓ 标定完成！")
                        self.is_calibrated = True
                        return True
                    else:
                        print("\n✗ 请先完成标定（按 's' 采样，按 'c' 保存中心点）")
                        
        return False
        
    def start(self):
        """启动遥操作"""
        if self.is_running:
            self.get_logger().warn('遥操作已在运行')
            return
            
        self.get_logger().info('启动遥操作...')
        
        # 启动控制循环
        self.is_running = True
        self.start_time = time.time()
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
        self.get_logger().info('遥操作已启动（暂停状态）')
        
    def stop(self):
        """停止遥操作"""
        if not self.is_running:
            return
            
        self.get_logger().info('停止遥操作...')
        
        self.is_running = False
        self.is_control_enabled = False
        
        # 等待控制线程结束
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
            
        # 发送零速度
        if self.robot_commander:
            self.robot_commander.send_zero_twist()
        
        # 停止 VisionPro 数据流
        self.visionpro_bridge.stop()
        
        self.print_statistics()
        
        self.get_logger().info('遥操作已停止')
        
    def control_loop(self):
        """主控制循环"""
        self.get_logger().info('控制循环已启动')
        
        loop_rate = 50  # Hz
        dt = 1.0 / loop_rate
        
        while self.is_running and rclpy.ok():
            loop_start = time.time()
            
            try:
                # 如果控制未启用，发送零速度
                if not self.is_control_enabled:
                    self.robot_commander.send_zero_twist()
                    time.sleep(dt)
                    continue

                # 获取手部位姿
                position, rotation = self.visionpro_bridge.get_hand_relative_to_head()

                # 映射到 Twist
                twist = self.coordinate_mapper.map_to_twist(position, rotation)

                # 发送 Twist 到机械臂
                self.robot_commander.send_twist(twist)

                # 处理夹爪控制
                gripper_config = self.config['gripper']
                control_mode = gripper_config.get('control_mode', 'binary')

                if control_mode == 'continuous':
                    # 连续控制模式
                    pinch_distance = self.visionpro_bridge.get_pinch_distance()
                    
                    pinch_open = gripper_config['pinch_distance_open']
                    pinch_close = gripper_config['pinch_distance_close']
                    gripper_open = gripper_config['gripper_open_position']
                    gripper_close = gripper_config['gripper_close_position']
                    
                    pinch_distance = np.clip(pinch_distance, pinch_close, pinch_open)
                    normalized = (pinch_distance - pinch_close) / (pinch_open - pinch_close)
                    gripper_position = gripper_close + (gripper_open - gripper_close) * normalized
                    
                    update_threshold = gripper_config.get('update_threshold', 0.005)
                    
                    if abs(gripper_position - self.last_gripper_position) > update_threshold:
                        self.robot_commander.control_gripper(
                            position=float(gripper_position),
                            max_effort=gripper_config['max_effort']
                        )
                        self.last_gripper_position = gripper_position

                elif control_mode == 'binary':
                    pinch_threshold = gripper_config['pinch_threshold']
                    current_pinch = self.visionpro_bridge.get_pinch_state(threshold=pinch_threshold)

                    if not hasattr(self, 'last_gripper_state'):
                        self.last_gripper_state = False
                        
                    if current_pinch != self.last_gripper_state:
                        if current_pinch:
                            self.robot_commander.control_gripper(
                                position=gripper_config['close_position'],
                                max_effort=gripper_config['max_effort']
                            )
                        else:
                            self.robot_commander.control_gripper(
                                position=gripper_config['open_position'],
                                max_effort=gripper_config['max_effort']
                            )
                        self.last_gripper_state = current_pinch

                self.loop_count += 1
                
                # 每200次循环打印一次状态
                if self.loop_count % 200 == 0:
                    self.print_status()
                    
            except Exception as e:
                self.get_logger().error(f'控制循环错误: {e}')
                
            # 保持循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            
        self.get_logger().info('控制循环已退出')
        
    def toggle_control(self):
        """切换控制启用/暂停"""
        self.is_control_enabled = not self.is_control_enabled
        
        if self.is_control_enabled:
            # 重置滤波器
            if self.coordinate_mapper:
                self.coordinate_mapper.reset_filter()
            print("\n>>> 控制已启用 - 机械臂开始跟随手部 <<<")
        else:
            # 发送零速度
            if self.robot_commander:
                self.robot_commander.send_zero_twist()
            print("\n>>> 控制已暂停 - 机械臂停止 <<<")
            
    def print_status(self):
        """打印状态信息"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            freq = self.loop_count / elapsed if elapsed > 0 else 0
            
            status = "运行中" if self.is_control_enabled else "暂停"
            self.get_logger().info(
                f'[{status}] 时间: {elapsed:.1f}s | 循环: {self.loop_count} | 频率: {freq:.1f} Hz'
            )
            
    def print_statistics(self):
        """打印统计信息"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_freq = self.loop_count / elapsed if elapsed > 0 else 0
            
            self.get_logger().info('='*50)
            self.get_logger().info('遥操作统计:')
            self.get_logger().info(f'  总运行时间: {elapsed:.2f} s')
            self.get_logger().info(f'  总循环次数: {self.loop_count}')
            self.get_logger().info(f'  平均频率: {avg_freq:.2f} Hz')
            self.get_logger().info('='*50)
            
    def emergency_stop(self):
        """急停"""
        self.get_logger().error('!!! 触发急停 !!!')
        self.is_control_enabled = False
        if self.robot_commander:
            self.robot_commander.emergency_stop()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VisionPro 遥操作节点（安全版）')
    parser.add_argument(
        '--config',
        type=str,
        default='vision_pro_control/config/teleop_config.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    # 初始化 ROS2
    rclpy.init()
    
    try:
        # 创建节点
        node = TeleopNode(config_file=args.config)
        
        # 启动 VisionPro 连接
        print("\n连接 VisionPro...")
        node.visionpro_bridge.start()
        time.sleep(1.0)
        print("✓ VisionPro 连接成功\n")
        
        # 运行标定
        if not node.run_calibration():
            print("标定取消，退出程序")
            node.visionpro_bridge.stop()
            rclpy.shutdown()
            return
            
        # 初始化机械臂和坐标映射器
        print("\n初始化机械臂控制器...")
        node.init_robot_commander()
        
        calibration_file = Path(__file__).parent.parent / "config" / "calibration.yaml"
        node.init_coordinate_mapper(calibration_file)
        
        # 启动遥操作
        node.start()
        
        # 打印控制说明
        print("\n" + "="*60)
        print("【遥操作模式】- 当前状态: 暂停")
        print("="*60)
        print("按键说明:")
        print("  'Space' - 开始/暂停遥操作")
        print("  'e'     - 急停")
        print("  's'     - 打印状态")
        print("  'q'     - 退出")
        print("="*60)
        print("\n⚠️  安全提示: 手放在急停按钮附近！")
        print("    当前速度限制: {} cm/s\n".format(
            int(node.config['safety']['max_linear_velocity'] * 100)
        ))
        print("准备好后，按 Space 开始遥操作...\n")
        
        # 键盘监听
        with KeyboardMonitor() as kb:
            while rclpy.ok() and node.is_running:
                # 处理 ROS2 回调
                rclpy.spin_once(node, timeout_sec=0.01)
                
                # 检查按键
                key = kb.get_key(timeout=0.01)
                
                if key == 'q':
                    print("\n退出遥操作...")
                    break
                elif key == ' ':
                    node.toggle_control()
                elif key == 's':
                    node.print_status()
                elif key == 'e':
                    node.emergency_stop()
                    print("\n!!! 急停已触发 !!!")
                    print("按 Space 可恢复控制")
                    
    except KeyboardInterrupt:
        print("\n\n用户中断")
        
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
        
    print("程序已退出")


if __name__ == '__main__':
    main()