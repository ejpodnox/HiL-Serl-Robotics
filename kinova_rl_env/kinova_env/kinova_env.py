#!/usr/bin/env python3
"""
KinovaEnv: Gym-style环境包装
基于KinovaInterface，提供标准Gym接口
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

# 假设kinova_interface在同一目录
from kinova_interface import KinovaInterface
from config_loader import KinovaConfig

class KinovaEnv(gym.Env):
    """
    Kinova机器人的Gym环境
    
    使用示例:
        env = KinovaEnv()
        obs, info = env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
    """
    
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
        ...
        """
        初始化环境
        
        TODO思路:
        1. 保存配置参数
        2. 创建KinovaInterface实例
        3. 定义observation_space (Box类型)
        4. 定义action_space (Box类型)
        5. 初始化步数计数器
        """
        
        
        # 机器人接口
        self.interface = KinovaInterface(node_name=self.config.ros2.node_name)
        
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape = (self.config.observation.state_dim,),
                dtype = np.float32
            )
        })

        self.action_space = spaces.Dict({
            "state": spaces.Box(
                low = self.config.action.low,
                high = self.config.action.high,
                shape = (self.config.action.dim,),
                dtype = np.float32
            )
        })
        # 状态变量
        self.current_step = 0
        self.episode_return = 0.0
    
    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        TODO思路:
        1. 调用super().reset()处理seed
        2. 连接机器人 (如果还没连接)
        3. 发送零速度命令停止机器人
        4. 等待机器人稳定
        5. 获取初始observation
        6. 重置计数器
        7. 返回 (observation, info)
        
        Returns:
            observation: 初始观察
            info: 额外信息字典
        """
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
        """
        执行一步动作
        
        TODO思路:
        1. 检查action维度
        2. 裁剪action到[-1, 1]
        3. 缩放action: scaled = action * action_scale
        4. 发送scaled velocity到机器人
        5. 等待control_dt时间
        6. 获取新的observation
        7. 计算reward (暂时返回0)
        8. 判断done/truncated
        9. 更新计数器
        10. 返回 (obs, reward, done, truncated, info)
        
        Args:
            action: (7,) numpy array, 范围[-1, 1]
        
        Returns:
            observation: 新观察
            reward: 奖励值
            terminated: 是否因达成目标而结束
            truncated: 是否因超时而结束
            info: 额外信息
        """
        action = np.clip(-1.0,1.0)

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
    
    def _get_obs(self):
        pos, vel = self.interface.get_joint_state()
        state = np.concatenate(pos, vel)
        return {"state":state}

    
    def _compute_reward(self, obs, action):
        """
        todo:
        Phase 2: 直接返回 0.0 (稀疏奖励)
        Phase 4: 使用reward classifier
        """
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
    env = KinovaEnv(control_frequency=20)
    
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