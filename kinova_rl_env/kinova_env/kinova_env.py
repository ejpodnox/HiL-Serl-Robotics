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

from kinova_interface import KinovaInterface
from config_loader import KinovaConfig
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
        self.interface = KinovaInterface(node_name=self.config.ros2.node_name)
        
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape = (self.config.observation.state_dim,),
                dtype = np.float32
            ),
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(self.config.camera.image_size[0], 
                    self.config.camera.image_size[1], 3),  # (H, W, 3)
                dtype=np.uint8
            )
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
    
    def set_gripper(self, position):
        """
        设置gripper位置
        
        TODO:
        1. 检查gripper是否启用
        2. 发送gripper命令到对应topic
        3. 等待执行
        
        Args:
            position: 0.0(全开) 到 1.0(全闭)
        """
        # 不知道怎么设置gripper

    def send_gripper_command(self, position):
        """
        发送gripper命令
        
        Args:
            position: 0.0(全开) 到 1.0(全闭)
        """

        
        # 限制范围
        position = np.clip(position, 0.0, 1.0)
        
        msg = Float64()
        msg.data = float(position)
        self._gripper_pub.publish(msg)

    def get_gripper_state(self):
        """
        获取gripper当前位置
        
        Returns:
            float: 0.0(全开) 到 1.0(全闭)
        """
        # TODO: 需要订阅gripper状态topic
        # 现在先返回缓存值
        return self._gripper_state

    def _get_obs(self):
        
        # 获取关节状态
        pos, vel = self.interface.get_joint_state()
        if pos is None:
            pos = np.zeros(7)
            vel = np.zeros(7)
        
        # 获取gripper状态
        gripper = self.interface.get_gripper_state()
        
        # 拼接state
        state = np.concatenate([pos, vel, [gripper]]).astype(np.float32)
        
        # 获取图像
        image = self.interface.get_image()
        if image is None:
            image = np.zeros(
                (self.config.camera.image_size[0],
                self.config.camera.image_size[1], 3),
                dtype=np.uint8
            )
        else:
            # resize到指定尺寸
            image = cv2.resize(
                image,
                (self.config.camera.image_size[1], 
                self.config.camera.image_size[0])
            )
        
        return {
            "state": state,
            "image": image
        }
    
    def _compute_reward(self, obs, action):

        return 0.0
    
    def close(self):
        self.interface.send_joint_velocities([0.0] * 7)
        self.interface.disconnect()


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