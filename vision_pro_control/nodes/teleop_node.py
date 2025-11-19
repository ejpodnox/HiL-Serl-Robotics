"""
VisionPro 遥操作主节点
整合 VisionPro 数据接收、坐标映射、机械臂控制
"""
import rclpy
from rclpy.node import Node
import numpy as np
import yaml
import time
import threading
from pathlib import Path

# 导入自定义模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vision_pro_control.core.visionpro_bridge import VisionProBridge
from vision_pro_control.core.coordinate_mapper import CoordinateMapper
from vision_pro_control.core.robot_commander import RobotCommander
from vision_pro_control.utils.keyboard_monitor import KeyboardMonitor


class TeleopNode(Node):
    """VisionPro 遥操作节点"""
    
    def __init__(self, config_file: str):
        super().__init__('visionpro_teleop_node')
        
        self.get_logger().info('初始化 VisionPro 遥操作节点...')
        
        # 加载配置
        self.config = self.load_config(config_file)
        
        # 初始化组件
        self.visionpro_bridge = None
        self.coordinate_mapper = None
        self.robot_commander = None
        
        self.init_components()
        
        # 状态变量
        self.is_running = False
        self.last_gripper_state = False  # 上次夹爪状态
        self.control_thread = None
        
        # 统计信息
        self.loop_count = 0
        self.start_time = None
        
        self.get_logger().info('VisionPro 遥操作节点初始化完成')
        
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
        self.visionpro_bridge = VisionProBridge(
            avp_ip=self.config['visionpro']['ip'],
            use_right_hand=self.config['visionpro']['use_right_hand']
        )
        self.get_logger().info(f"VisionPro 连接: {self.config['visionpro']['ip']}")

        calibration_file = Path(__file__).parent.parent.parent / self.config['calibration']['file']
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

        self.get_logger().info('所有组件初始化完成')
        
    def start(self):
        """启动遥操作"""
        if self.is_running:
            self.get_logger().warn('遥操作已在运行')
            return
            
        self.get_logger().info('启动遥操作...')
        
        # 启动 VisionPro 数据流
        self.visionpro_bridge.start()
        time.sleep(1.0)  # 等待连接稳定
        
        # 启动控制循环
        self.is_running = True
        self.start_time = time.time()
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
        self.get_logger().info('遥操作已启动')
        
    def stop(self):
        """停止遥操作"""
        if not self.is_running:
            return
            
        self.get_logger().info('停止遥操作...')
        
        self.is_running = False
        
        # 等待控制线程结束
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
            
        # 发送零速度
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

                # 获取手部位姿
                position, rotation = self.visionpro_bridge.get_hand_relative_to_head()

                # 映射到 Twist
                twist = self.coordinate_mapper.map_to_twist(position, rotation)

                # 发送 Twist 到机械臂
                self.robot_commander.send_twist(twist)


                # ============================================
                # 夹爪控制代码片段
                # 用于替换 teleop_node.py 中 control_loop() 的夹爪控制部分
                # 找到原来的 "# 处理夹爪控制" 部分并替换
                # ============================================

                # 处理夹爪控制
                gripper_config = self.config['gripper']
                control_mode = gripper_config.get('control_mode', 'binary')

                if control_mode == 'continuous':
                    # 连续控制模式
                    pinch_distance = self.visionpro_bridge.get_pinch_distance()
                    
                    # 线性映射：pinch_distance [open, close] -> gripper_position [0, 0.8]
                    pinch_open = gripper_config['pinch_distance_open']
                    pinch_close = gripper_config['pinch_distance_close']
                    gripper_open = gripper_config['gripper_open_position']
                    gripper_close = gripper_config['gripper_close_position']
                    
                    # 限制范围
                    pinch_distance = np.clip(pinch_distance, pinch_close, pinch_open)
                    
                    # 映射到夹爪位置（注意：pinch大 -> gripper小）
                    normalized = (pinch_distance - pinch_close) / (pinch_open - pinch_close)
                    gripper_position = gripper_close + (gripper_open - gripper_close) * normalized
                    
                    # 防抖：只有变化超过阈值才发送
                    update_threshold = gripper_config.get('update_threshold', 0.005)
                    if not hasattr(self, 'last_gripper_position'):
                        self.last_gripper_position = gripper_position
                    
                    if abs(gripper_position - self.last_gripper_position) > update_threshold:
                        self.robot_commander.control_gripper(
                            position=float(gripper_position),
                            max_effort=gripper_config['max_effort']
                        )
                        self.last_gripper_position = gripper_position
                        
                        # 每50次循环打印一次夹爪状态
                        if self.loop_count % 50 == 0:
                            self.get_logger().info(
                                f'夹爪位置: {gripper_position:.3f} (捏合距离: {pinch_distance:.3f}m)'
                            )

                else:
                    # 二值控制模式（原有逻辑）
                    pinch_threshold = gripper_config['pinch_threshold']
                    current_pinch = self.visionpro_bridge.get_pinch_state(threshold=pinch_threshold)

                    if current_pinch != self.last_gripper_state:
                        if current_pinch:
                            self.robot_commander.control_gripper(
                                position=gripper_config['close_position'],
                                max_effort=gripper_config['max_effort']
                            )
                            self.get_logger().info('夹爪闭合')
                        else:
                            self.robot_commander.control_gripper(
                                position=gripper_config['open_position'],
                                max_effort=gripper_config['max_effort']
                            )
                            self.get_logger().info('夹爪打开')
                        
                        self.last_gripper_state = current_pinch

                # # 处理夹爪控制
                # gripper_config = self.config['gripper']
                # control_mode = gripper_config.get('control_mode', 'binary')


                # if control_mode == 'continuous':
                #     # 连续控制模式
                #     pinch_distance = self.visionpro_bridge.get_pinch_distance()
                    
                #     # 线性映射：pinch_distance [open, close] -> gripper_position [0, 0.8]
                #     pinch_open = gripper_config['pinch_distance_open']
                #     pinch_close = gripper_config['pinch_distance_close']
                #     gripper_open = gripper_config['gripper_open_position']
                #     gripper_close = gripper_config['gripper_close_position']
                    
                #     # 限制范围
                #     pinch_distance = np.clip(pinch_distance, pinch_close, pinch_open)
                    
                #     # 映射到夹爪位置（注意：pinch大 -> gripper小）
                #     normalized = (pinch_distance - pinch_close) / (pinch_open - pinch_close)
                #     gripper_position = gripper_close + (gripper_open - gripper_close) * normalized
                    
                #     # 防抖：只有变化超过阈值才发送
                #     update_threshold = gripper_config.get('update_threshold', 0.005)
                #     if not hasattr(self, 'last_gripper_position'):
                #         self.last_gripper_position = gripper_position
                    
                #     if abs(gripper_position - self.last_gripper_position) > update_threshold:
                #         self.robot_commander.control_gripper(
                #             position=float(gripper_position),
                #             max_effort=gripper_config['max_effort']
                #         )
                #         self.last_gripper_position = gripper_position
                        
                #         # 每50次循环打印一次夹爪状态
                #         if self.loop_count % 50 == 0:
                #             self.get_logger().info(
                #                 f'夹爪位置: {gripper_position:.3f} (捏合距离: {pinch_distance:.3f}m)'
                #             )

                # else:
                #     # 二值控制模式（原有逻辑）
                #     pinch_threshold = gripper_config['pinch_threshold']
                #     current_pinch = self.visionpro_bridge.get_pinch_state(threshold=pinch_threshold)

                #     if current_pinch != self.last_gripper_state:
                #         if current_pinch:
                #             self.robot_commander.control_gripper(
                #                 position=gripper_config['close_position'],
                #                 max_effort=gripper_config['max_effort']
                #             )
                #             self.get_logger().info('夹爪闭合')
                #         else:
                #             self.robot_commander.control_gripper(
                #                 position=gripper_config['open_position'],
                #                 max_effort=gripper_config['max_effort']
                #             )
                #             self.get_logger().info('夹爪打开')
                        
                #         self.last_gripper_state = current_pinch     

                self.loop_count += 1
                
                # 每100次循环打印一次状态
                if self.loop_count % 100 == 0:
                    self.print_status()
                    
            except Exception as e:
                self.get_logger().error(f'控制循环错误: {e}')
                
            # 保持循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            
        self.get_logger().info('控制循环已退出')
        
    def print_status(self):
        """打印状态信息"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            freq = self.loop_count / elapsed if elapsed > 0 else 0
            
            self.get_logger().info(
                f'运行时间: {elapsed:.1f}s | '
                f'循环次数: {self.loop_count} | '
                f'频率: {freq:.1f} Hz'
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
        self.robot_commander.emergency_stop()
        self.stop()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VisionPro 遥操作节点')
    parser.add_argument(
        '--config',
        type=str,
        default='config/teleop_config.yaml',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    # 初始化 ROS2
    rclpy.init()
    
    try:
        # 创建节点
        node = TeleopNode(config_file=args.config)
        
        # 启动遥操作
        node.start()
        
        # 键盘监听
        print("\n" + "="*60)
        print("VisionPro 遥操作运行中")
        print("="*60)
        print("按键说明:")
        print("  'q' - 退出")
        print("  's' - 打印状态")
        print("  'e' - 急停")
        print("  'r' - 解除急停")
        print("="*60 + "\n")
        
        with KeyboardMonitor() as kb:
            while rclpy.ok():
                # 处理 ROS2 回调
                rclpy.spin_once(node, timeout_sec=0.01)
                
                # 检查按键
                key = kb.get_key(timeout=0.01)
                
                if key == 'q':
                    print("\n退出遥操作...")
                    break
                elif key == 's':
                    node.print_status()
                elif key == 'e':
                    node.emergency_stop()
                elif key == 'r':
                    node.robot_commander.resume()
                    print("已解除急停")
                    
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
        
    print("程序已退出")


if __name__ == '__main__':
    main()