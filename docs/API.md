# API 文档

本文档提供 Kinova HIL-SERL 包的详细 API 参考。

## 目录

- [VisionPro Control](#visionpro-control)
- [Kinova RL Environment](#kinova-rl-environment)
- [HIL-SERL Training](#hil-serl-training)
- [工具函数](#工具函数)

---

## VisionPro Control

### VisionProBridge

VisionPro 数据接收接口。

```python
from vision_pro_control.core import VisionProBridge

bridge = VisionProBridge(
    avp_ip="192.168.1.125",
    use_right_hand=True
)
```

#### 参数

- **avp_ip** (`str`): VisionPro 设备 IP 地址
- **use_right_hand** (`bool`, optional): 使用右手还是左手，默认 `True`

#### 方法

**`start()`**

启动数据接收线程。

```python
bridge.start()
```

**`stop()`**

停止数据接收。

```python
bridge.stop()
```

**`get_latest_data()`**

获取最新的 VisionPro 数据。

```python
data = bridge.get_latest_data()
# Returns:
# {
#     'head_pose': np.ndarray (4x4),      # 头部位姿矩阵
#     'wrist_pose': np.ndarray (4x4),     # 手腕位姿矩阵
#     'pinch_distance': float,            # 捏合距离
#     'wrist_roll': float,                # 手腕翻转角度
#     'timestamp': float                  # 时间戳
# }
```

---

### CoordinateMapper

VisionPro 坐标到机械臂坐标的映射。

```python
from vision_pro_control.core import CoordinateMapper

mapper = CoordinateMapper(calibration_file="config/calibration.pkl")
```

#### 参数

- **calibration_file** (`str`, optional): 标定文件路径

#### 方法

**`set_gains(position_gain, rotation_gain)`**

设置位置和旋转增益。

```python
mapper.set_gains(position_gain=1.0, rotation_gain=0.5)
```

**`set_velocity_limits(max_linear, max_angular)`**

设置速度限制。

```python
mapper.set_velocity_limits(max_linear=0.01, max_angular=0.05)
```

**`map_to_robot_twist(vp_data, current_robot_pose)`**

映射 VisionPro 数据到机械臂 Twist 命令。

```python
twist = mapper.map_to_robot_twist(vp_data, robot_pose)
# Returns:
# {
#     'linear': {'x': float, 'y': float, 'z': float},
#     'angular': {'x': float, 'y': float, 'z': float}
# }
```

---

### RobotCommander

Kinova 机械臂 ROS2 控制接口。

```python
from vision_pro_control.core import RobotCommander

commander = RobotCommander(robot_ip="192.168.8.10")
```

#### 方法

**`get_tcp_pose()`**

获取末端执行器位姿。

```python
pose = commander.get_tcp_pose()
# Returns: np.ndarray (7,) [x, y, z, qx, qy, qz, qw]
```

**`send_twist_command(twist)`**

发送 Twist 速度命令。

```python
twist = {
    'linear': {'x': 0.01, 'y': 0.0, 'z': 0.0},
    'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
}
commander.send_twist_command(twist)
```

**`set_gripper(position)`**

控制夹爪。

```python
commander.set_gripper(0.5)  # 0.0 = 全开, 1.0 = 全闭
```

**`emergency_stop()`**

紧急停止。

```python
commander.emergency_stop()
```

---

## Kinova RL Environment

### KinovaEnv

Kinova 机械臂的 Gymnasium 环境。

```python
from kinova_rl_env import KinovaEnv, KinovaConfig

config = KinovaConfig.from_yaml("config/kinova_config.yaml")
env = KinovaEnv(config=config)
```

#### 方法

**`reset()`**

重置环境到初始状态。

```python
obs, info = env.reset()
# obs: Dict[str, np.ndarray]
# {
#     'state': {
#         'tcp_pose': np.ndarray (7,),
#         'tcp_vel': np.ndarray (6,),
#         'joint_positions': np.ndarray (7,),
#         'gripper_position': np.ndarray (1,)
#     },
#     'images': {
#         'wrist_1': np.ndarray (128, 128, 3),
#         'wrist_2': np.ndarray (128, 128, 3)
#     }
# }
```

**`step(action)`**

执行一个动作。

```python
action = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])  # (7,)
obs, reward, terminated, truncated, info = env.step(action)
```

**`close()`**

关闭环境。

```python
env.close()
```

#### 属性

**`observation_space`**

观测空间定义。

```python
print(env.observation_space)
```

**`action_space`**

动作空间定义。

```python
print(env.action_space)
```

---

### CameraInterface

相机抽象接口（可插拔设计）。

```python
from kinova_rl_env import RealSenseCamera, WebCamera, DummyCamera

# RealSense
camera = RealSenseCamera(
    topic="/camera/color/image_raw",
    image_size=(128, 128)
)

# WebCam
camera = WebCamera(
    device_id=0,
    image_size=(128, 128)
)

# Dummy（测试用）
camera = DummyCamera(image_size=(128, 128))
```

#### 方法

**`start()`**

启动相机。

```python
camera.start()
```

**`get_image()`**

获取当前帧。

```python
image = camera.get_image()  # np.ndarray (H, W, 3)
```

**`stop()`**

停止相机。

```python
camera.stop()
```

---

## HIL-SERL Training

### BC 训练

```python
from hil_serl_kinova.train_bc_kinova import BCPolicy, BCTrainer

# 加载配置
from hil_serl_kinova.experiments.kinova_reaching.config import get_config
config = get_config()

# 创建策略
policy = BCPolicy(
    state_dim=config.bc_config.state_dim,
    action_dim=config.bc_config.action_dim,
    image_size=config.bc_config.image_size,
    hidden_dims=config.bc_config.hidden_dims
)

# 训练
trainer = BCTrainer(
    policy=policy,
    demos_dir="./demos/reaching",
    config=config.bc_config
)
trainer.train(epochs=50)
```

#### BCPolicy

**`forward(state, image)`**

前向传播。

```python
import torch

state = torch.randn(batch_size, 21)
image = torch.randn(batch_size, 3, 128, 128)

action = policy.forward(state, image)  # (batch_size, 7)
```

**`get_action(state, image, deterministic=True)`**

获取动作（推理用）。

```python
action = policy.get_action(state, image, deterministic=True)
# Returns: np.ndarray (7,)
```

---

### Reward Classifier

```python
from hil_serl_kinova.train_reward_classifier import RewardClassifier

classifier = RewardClassifier(
    state_dim=21,
    image_size=(128, 128),
    hidden_dims=[256, 128]
)

# 训练
trainer = RewardClassifierTrainer(
    classifier=classifier,
    demos_dir="./demos/labeled"
)
trainer.train(epochs=20)

# 推理
success_prob = classifier.predict(state, image)  # float [0, 1]
```

---

### RLPD 训练

```python
from hil_serl_kinova.train_rlpd_kinova import RLPDTrainer

trainer = RLPDTrainer(
    env=env,
    config=config.rlpd_config,
    demos_dir="./demos/reaching",
    bc_checkpoint="checkpoints/bc_kinova/best_model.pt"
)

# 离线预训练
trainer.offline_training(steps=10000)

# 在线学习
trainer.online_training(steps=50000)
```

---

### 策略部署

```python
from hil_serl_kinova.deploy_policy import PolicyDeployer

deployer = PolicyDeployer(
    checkpoint_path="checkpoints/bc_kinova/best_model.pt",
    mode="policy_only",  # 'policy_only' / 'hybrid' / 'evaluation'
    config_path="config/kinova_config.yaml"
)

# 运行一个回合
success, reward = deployer.run_episode()

# 交互式模式
deployer.interactive_mode()
```

---

## 工具函数

### 数据工具

```python
from hil_serl_kinova.tools.data_utils import (
    view_demo,
    stats_demos,
    validate_demos,
    convert_to_hdf5
)

# 查看演示
view_demo("demos/reaching/demo_000.pkl")

# 统计信息
stats_demos("demos/reaching")

# 验证格式
validate_demos("demos/reaching")

# 转换格式
convert_to_hdf5("demos/reaching")
```

### 可视化工具

```python
from hil_serl_kinova.tools.visualize import (
    plot_trajectory,
    plot_dataset,
    plot_training,
    plot_multiple_trajectories
)

# 绘制单条轨迹
plot_trajectory("demos/reaching/demo_000.pkl", output_path="plots/traj.png")

# 绘制数据集统计
plot_dataset("demos/reaching", output_dir="plots/")

# 绘制训练曲线
plot_training("logs/bc", output_path="plots/training.png")

# 绘制多条轨迹对比
plot_multiple_trajectories("demos/reaching", output_path="plots/multi.png", max_demos=5)
```

---

## 命令行工具

安装包后，以下命令行工具可用：

```bash
# 数据收集
kinova-record-teleop --save_dir ./teleop_demos --num_demos 5
kinova-record-demos --save_dir ./demos --num_demos 10
kinova-record-labels --save_dir ./demos/labeled --num_success 20 --num_fail 20

# 训练
kinova-train-bc --config experiments/kinova_reaching/config.py --demos_dir ./demos
kinova-train-classifier --demos_dir ./demos/labeled
kinova-train-rlpd --config experiments/kinova_reaching/config.py --demos_dir ./demos

# 部署
kinova-deploy --checkpoint checkpoints/bc_kinova/best_model.pt --mode evaluation

# 工具
kinova-data-utils --stats ./demos
kinova-visualize --dataset ./demos --output plots/
```

---

## 示例代码

### 完整训练流程

```python
import numpy as np
from kinova_rl_env import KinovaEnv, KinovaConfig
from hil_serl_kinova.train_bc_kinova import BCTrainer
from hil_serl_kinova.experiments.kinova_reaching.config import get_config

# 1. 加载配置
config = get_config()
env_config = KinovaConfig.from_yaml("config/kinova_config.yaml")

# 2. 创建环境
env = KinovaEnv(config=env_config)

# 3. BC 训练
bc_trainer = BCTrainer(
    demos_dir="./demos/reaching",
    config=config.bc_config
)
bc_trainer.train(epochs=50)

# 4. 评估
env.reset()
for _ in range(100):
    state, image = env.get_observation()
    action = bc_trainer.policy.get_action(state, image)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()
```

### 自定义相机

```python
from kinova_rl_env import CameraInterface
import numpy as np

class MyCustomCamera(CameraInterface):
    def __init__(self, image_size):
        self.image_size = image_size

    def start(self):
        print("启动自定义相机")

    def get_image(self) -> np.ndarray:
        # 实现你的图像获取逻辑
        return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def stop(self):
        print("停止自定义相机")

# 使用
env_config.camera.backend = "custom"
env = KinovaEnv(config=env_config, custom_camera=MyCustomCamera)
```

---

## 类型定义

### Observation

```python
Observation = Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]
# {
#     'state': {
#         'tcp_pose': np.ndarray (7,),
#         'tcp_vel': np.ndarray (6,),
#         ...
#     },
#     'images': {
#         'wrist_1': np.ndarray (H, W, 3),
#         ...
#     }
# }
```

### Action

```python
Action = np.ndarray  # (7,) for delta_pose mode
# [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
```

### Twist

```python
Twist = Dict[str, Dict[str, float]]
# {
#     'linear': {'x': float, 'y': float, 'z': float},
#     'angular': {'x': float, 'y': float, 'z': float}
# }
```

---

## 下一步

- [配置说明](CONFIGURATION.md) - 详细配置参数
- [快速开始](QUICKSTART.md) - 实践示例
- [实现总结](IMPLEMENTATION_SUMMARY.md) - 技术细节
