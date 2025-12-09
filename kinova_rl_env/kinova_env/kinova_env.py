#!/usr/bin/env python3
"""
KinovaEnv: Gym-style环境包装
基于KinovaInterface，提供标准Gym接口
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import cv2

from .kinova_interface import KinovaInterface
from .config_loader import KinovaConfig
from .camera_interface import WebCamera, RealSenseCamera, DummyCamera
from std_msgs.msg import Float64

class KinovaEnv(gym.Env):
    def __init__(self, config_path=None, config=None):
        super().__init__()

        # 加载配置
        if config:
            self.config = config
        elif config_path:
            self.config = KinovaConfig.from_yaml(config_path)
        else:
            self.config = KinovaConfig.get_default()

        # 从config读取参数
        self.control_frequency = self.config.control.frequency
        self.control_dt = self.config.control.dt
        self.action_scale = self.config.control.action_scale
        self.max_episode_steps = self.config.control.max_episode_steps

        # 机器人接口
        joint_names = getattr(self.config.robot, "joint_names", None)
        # 选择控制话题：优先 trajectory_command_topic，其次 velocity_command_topic
        traj_topic = self.config.ros2.get(
            'trajectory_command_topic',
            self.config.ros2.get('velocity_command_topic', '/joint_trajectory_controller/joint_trajectory'),
        )

        self.interface = KinovaInterface(
            node_name=self.config.ros2.node_name,
            joint_state_topic=self.config.ros2.get('joint_state_topic', '/joint_states'),
            trajectory_topic=traj_topic,
            twist_topic=self.config.ros2.get('twist_command_topic', '/twist_controller/commands'),
            gripper_command_topic=self.config.ros2.get('gripper_command_topic', '/robotiq_gripper_controller/gripper_cmd'),
            base_frame=self.config.ros2.get('base_frame', 'base_link'),
            tool_frame=self.config.ros2.get('tool_frame', 'tool_frame'),
            joint_names=joint_names,
        )

        # 相机接口
        self.cameras = {}
        self._setup_cameras()
        
        # 修改为HIL-SERL标准格式：嵌套字典
        self.observation_space = spaces.Dict({
            "state": spaces.Dict({
                "tcp_pose": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),  # [x, y, z, qx, qy, qz, qw]
                    dtype=np.float32
                ),
                "tcp_vel": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),  # [vx, vy, vz, wx, wy, wz]
                    dtype=np.float32
                ),
                "gripper_pose": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),  # [gripper_position]
                    dtype=np.float32
                ),
            }),
            "images": spaces.Dict({
                # 注意：从"image"改为"images"（复数），支持多相机
                "wrist_1": spaces.Box(
                    low=0,
                    high=255,
                    shape=(128, 128, 3),  # HIL-SERL标准：128x128
                    dtype=np.uint8
                )
            })
        })

        self.action_space = spaces.Box(
            low=self.config.action.low,
            high=self.config.action.high,
            shape=(self.config.action.dim,),
            dtype=np.float32
        )
        # 状态变量
        self.current_step = 0
        self.episode_return = 0.0

    def _setup_cameras(self):
        """设置相机（根据配置选择后端）"""
        if not self.config.camera.enabled:
            print("⚠️  相机未启用，使用虚拟图像")
            return

        backend = self.config.camera.backend
        print(f"📷 初始化相机（后端: {backend}）")

        try:
            if backend == "webcam":
                # USB 相机
                for cam_name, cam_cfg in self.config.camera.webcam_cameras.items():
                    camera = WebCamera(
                        camera_id=cam_cfg['device_id'],
                        target_size=tuple(cam_cfg['image_size'])
                    )
                    camera.start()
                    self.cameras[cam_name] = camera
                    print(f"  ✓ {cam_name}: USB 相机 (ID={cam_cfg['device_id']})")

            elif backend == "realsense":
                # RealSense 相机（本地 pyrealsense2）
                for cam_name, cam_cfg in self.config.camera.realsense_cameras.items():
                    camera = RealSenseCamera(
                        camera_name=cam_name,
                        serial_number=cam_cfg.get('serial_number'),
                        target_size=tuple(cam_cfg['image_size']),
                        fps=cam_cfg.get('fps', 30),
                        exposure=cam_cfg.get('exposure', 40000),
                        enable_depth=cam_cfg.get('depth', False),
                    )
                    camera.start()
                    self.cameras[cam_name] = camera
                    print(f"  ✓ {cam_name}: RealSense 相机")

            elif backend == "dummy":
                # 虚拟相机（测试用）
                for cam_name, cam_cfg in self.config.camera.dummy_cameras.items():
                    camera = DummyCamera(
                        target_size=tuple(cam_cfg['image_size']),
                        mode=cam_cfg.get('mode', 'noise'),
                    )
                    camera.start()
                    self.cameras[cam_name] = camera
                    print(f"  ✓ {cam_name}: 虚拟相机")

            else:
                print(f"  ✗ 未知的相机后端: {backend}")

        except Exception as e:
            print(f"  ✗ 相机初始化失败: {e}")
            print(f"  ⚠️  将使用虚拟图像")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.interface.node is None:
            self.interface.connect()
            time.sleep(1.0)
        
        obs = self._get_obs()

        self.current_step = 0
        self.episode_return = 0

        info = {}
        return obs, info
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        velocity_max = np.array(self.config.robot.joint_limits.velocity_max)
        scaled_velocity = action * self.action_scale * velocity_max

        self.interface.send_joint_velocities(scaled_velocity)

        time.sleep(self.control_dt)

        obs = self._get_obs()

        reward = self._compute_reward(obs,action)

        terminated = False
        truncated = (self.current_step >= self.max_episode_steps)

        self.current_step += 1
        self.episode_return += reward

        info = {"episode_return": self.episode_return}
        return obs, reward, terminated, truncated, info
    
    def _check_safety(self, positions):
        lower = np.array(self.config.robot.joint_limits.position_min)
        upper = np.array(self.config.robot.joint_limits.position_max)
        in_range = np.all((positions >= lower) & (positions <= upper))
        
        if in_range == False:
            safe_positions = np.clip(positions, lower, upper)
            return safe_positions
        else:
            return positions

    def go_home(self, timeout=None):
        thehold = 0.05
        Kp = 5

        if timeout is None:
            timeout = self.config.control.reset_timeout
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                print("go_home超时")
                break

            home_pos = self.config.robot.home_position
            pos, _ = self.interface.get_joint_state()

            if np.allclose(home_pos, pos, atol = thehold):
                print("到达目标位置")
                break


            velocity = Kp * (home_pos - pos)
            velocity_max = np.array(self.config.robot.joint_limits.velocity_max)
            velocity = np.clip(velocity, -velocity_max, velocity_max)  # 限制速度
            self.interface.send_joint_velocities(velocity)
                    
        self.interface.send_joint_velocities([0.0]*7)

        time.sleep(self.control_dt)
    
    def set_gripper(self, position, wait=True):
        """
        设置gripper位置

        Args:
            position: float, 0.0(全开) 到 1.0(全闭)
            wait: bool, 是否等待gripper运动完成

        原理：
        - 调用interface.send_gripper_command()发送ROS命令
        - 如果wait=True，等待一小段时间让gripper运动
        """
        # 限制范围
        position = np.clip(position, 0.0, 1.0)

        # 发送命令到ROS
        self.interface.send_gripper_command(position)

        # 如果需要等待，sleep一小段时间
        if wait:
            # Gripper运动时间，取决于位置变化量
            # 简化：固定等待时间
            wait_time = 0.3  # 300ms，足够gripper完成运动
            time.sleep(wait_time)

    def get_gripper_state(self):
        """
        获取gripper当前位置

        Returns:
            float: 0.0(全开) 到 1.0(全闭)
        """
        return self.interface.get_gripper_state()

    def _get_obs(self):
        """
        获取observation，格式符合HIL-SERL标准

        Returns:
            obs: dict {
                "state": {
                    "tcp_pose": (7,),  # [x, y, z, qx, qy, qz, qw]
                    "tcp_vel": (6,),   # [vx, vy, vz, wx, wy, wz]
                    "gripper_pose": (1,)  # [position]
                },
                "images": {
                    "wrist_1": (128, 128, 3)  # RGB图像
                }
            }
        """
        # 获取TCP位姿
        tcp_pose = self.interface.get_tcp_pose()
        if tcp_pose is None:
            # 如果获取失败，返回零向量
            tcp_pose = np.zeros(7, dtype=np.float32)
        else:
            tcp_pose = tcp_pose.astype(np.float32)

        # 获取TCP速度
        tcp_vel = self.interface.get_tcp_velocity()
        tcp_vel = tcp_vel.astype(np.float32)

        # 获取gripper状态
        gripper_position = self.interface.get_gripper_state()
        gripper_pose = np.array([gripper_position], dtype=np.float32)

        # 构建state字典
        state_dict = {
            "tcp_pose": tcp_pose,
            "tcp_vel": tcp_vel,
            "gripper_pose": gripper_pose
        }

        # 获取图像（从相机或使用虚拟图像）
        images_dict = {}
        if self.cameras:
            # 从实际相机获取图像
            for cam_name, camera in self.cameras.items():
                try:
                    image = camera.get_image()
                    images_dict[cam_name] = image
                except Exception as e:
                    # 如果获取失败，使用黑图像
                    print(f"⚠️  相机 {cam_name} 获取图像失败: {e}")
                    images_dict[cam_name] = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            # 使用虚拟图像（相机未启用）
            images_dict["wrist_1"] = np.zeros((128, 128, 3), dtype=np.uint8)

        # 返回嵌套字典
        return {
            "state": state_dict,
            "images": images_dict
        }
    
    def _compute_reward(self, obs, action):
        """
        计算奖励函数

        支持两种模式：
        1. sparse: 成功=1.0，否则=0.0
        2. dense: -distance_to_target（负距离作为奖励）

        Args:
            obs: observation字典
            action: action数组

        Returns:
            reward: float
        """
        reward_type = self.config.reward.type

        if reward_type == "sparse":
            # 稀疏奖励：只有成功时给1.0
            success = self._check_success(obs)
            if success:
                reward = self.config.reward.sparse_reward.success
            else:
                reward = self.config.reward.sparse_reward.step
            return reward

        elif reward_type == "dense":
            # 稠密奖励：负距离
            distance = self._compute_distance_to_target(obs)
            reward = self.config.reward.dense_reward.distance_scale * distance

            # 如果成功，加上bonus
            if self._check_success(obs):
                reward += self.config.reward.dense_reward.success_bonus

            return reward

        else:
            # 默认返回0
            return 0.0

    def _check_success(self, obs):
        """
        检查是否完成任务

        Args:
            obs: observation字典

        Returns:
            success: bool
        """
        # 获取当前TCP位置
        tcp_pose = obs["state"]["tcp_pose"]
        tcp_position = tcp_pose[:3]  # [x, y, z]

        # 获取目标位置
        target_pose = np.array(self.config.task.target_pose)
        target_position = target_pose[:3]

        # 计算距离
        distance = np.linalg.norm(tcp_position - target_position)

        # 判断是否在阈值内
        threshold = self.config.task.success_threshold.position
        success = (distance < threshold)

        return success

    def _compute_distance_to_target(self, obs):
        """
        计算TCP到目标的距离

        Args:
            obs: observation字典

        Returns:
            distance: float (米)
        """
        tcp_pose = obs["state"]["tcp_pose"]
        tcp_position = tcp_pose[:3]

        target_pose = np.array(self.config.task.target_pose)
        target_position = target_pose[:3]

        distance = np.linalg.norm(tcp_position - target_position)

        return distance
    
    def close(self):
        """关闭环境，释放资源"""
        # 停止机械臂
        self.interface.send_joint_velocities([0.0] * 7)
        self.interface.disconnect()

        # 关闭所有相机
        for cam_name, camera in self.cameras.items():
            try:
                camera.stop()
                print(f"✓ 相机 {cam_name} 已关闭")
            except Exception as e:
                print(f"⚠️  关闭相机 {cam_name} 时出错: {e}")

        self.cameras.clear()


# ============ 测试代码 ============
if __name__ == '__main__':
    """
    明天可以用这个测试环境
    """
    print("初始化 KinovaEnv...")
    env = KinovaEnv()
    
    try:
        # 重置
        print("重置环境...")
        obs, info = env.reset()
        print(f"初始observation: {obs}")
        
        # 运行几步
        print("\n运行10步...")
        for i in range(10):
            # 小随机动作
            action = np.random.randn(7) * 0.1
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i}: reward={reward}, done={done}")
            
            if done or truncated:
                print("Episode结束")
                break
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        env.close()
        print("环境已关闭")
