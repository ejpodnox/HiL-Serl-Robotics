# 配置说明

本文档详细说明各个配置文件的参数和用途。

## 配置文件概览

```
kinova-hil-serl/
├── vision_pro_control/config/
│   └── teleop_config.yaml          # VisionPro 遥操作配置
├── kinova_rl_env/config/
│   └── kinova_config.yaml          # Kinova 环境配置
└── hil_serl_kinova/experiments/
    └── kinova_reaching/
        └── config.py                # 任务和训练配置
```

## 1. VisionPro 遥操作配置

**文件**: `vision_pro_control/config/teleop_config.yaml`

```yaml
visionpro:
  ip: "192.168.1.125"              # VisionPro IP 地址
  use_right_hand: true             # 使用右手控制（false 为左手）
  port: 50051                      # gRPC 端口

robot:
  ip: "192.168.8.10"               # Kinova 机械臂 IP
  namespace: ""                    # ROS2 命名空间
  dof: 7                          # 自由度

calibration:
  file: "config/calibration.pkl"   # 标定文件路径
  auto_calibrate: true            # 启动时自动标定

mapper:
  position_gain: 1.0              # 位置增益
  rotation_gain: 0.5              # 旋转增益
  max_linear_velocity: 0.01       # 最大线速度 (m/s)
  max_angular_velocity: 0.05      # 最大角速度 (rad/s)
  deadzone: 0.02                  # 死区阈值

safety:
  workspace_radius: 0.6           # 工作空间半径 (m)
  max_acceleration: 0.1           # 最大加速度 (m/s²)
  emergency_stop_enabled: true    # 启用紧急停止
```

### 参数说明

#### VisionPro 设置
- **ip**: VisionPro 设备的网络地址
- **use_right_hand**: 控制用哪只手。true=右手，false=左手
- **port**: gRPC 通信端口，默认 50051

#### 机械臂设置
- **ip**: Kinova 机械臂的网络地址
- **namespace**: ROS2 话题命名空间
- **dof**: 机械臂自由度（Gen3 为 7）

#### 坐标映射
- **position_gain**: 手部移动到机械臂移动的缩放比例
- **rotation_gain**: 手部旋转到机械臂旋转的缩放比例
- **max_linear_velocity**: 末端执行器最大线速度
- **max_angular_velocity**: 末端执行器最大角速度
- **deadzone**: 小于此值的输入被忽略

#### 安全设置
- **workspace_radius**: 机械臂可到达的最大半径
- **max_acceleration**: 加速度限制
- **emergency_stop_enabled**: 是否启用紧急停止

---

## 2. Kinova 环境配置

**文件**: `kinova_rl_env/config/kinova_config.yaml`

```yaml
robot:
  ip: "192.168.8.10"
  namespace: ""
  dof: 7
  control_mode: "twist"           # 控制模式: twist / joint
  gripper_enabled: true

control:
  frequency: 50                   # 控制频率 (Hz)
  dt: 0.02                       # 时间步长 (s)

workspace:
  center: [0.5, 0.0, 0.3]       # 工作空间中心 (m)
  radius: 0.4                    # 工作空间半径 (m)
  z_min: 0.1                     # Z 轴最小高度 (m)
  z_max: 0.6                     # Z 轴最大高度 (m)

camera:
  enabled: true                  # 是否启用相机
  backend: "realsense"           # 相机后端: realsense / webcam / dummy
  cameras:
    wrist_1:
      type: "realsense"
      topic: "/camera/wrist_1/color/image_raw"
      serial_number: ""         # RealSense 序列号（可选）
    wrist_2:
      type: "realsense"
      topic: "/camera/wrist_2/color/image_raw"

  image_size: [128, 128]        # 图像大小
  color_mode: "RGB"             # 颜色模式: RGB / BGR
  frame_stack: 1                # 帧堆叠数量

observation:
  state_keys:
    - "tcp_pose"                # 末端位姿 (7,)
    - "tcp_vel"                 # 末端速度 (6,)
    - "joint_positions"         # 关节位置 (7,)
    - "joint_velocities"        # 关节速度 (7,)
    - "gripper_position"        # 夹爪位置 (1,)

  image_keys:
    - "wrist_1"
    - "wrist_2"

action:
  type: "delta_pose"            # 动作类型: delta_pose / absolute_pose / joint
  dimensions: 7                 # 动作维度
  limits:                       # 动作范围
    delta_pos: [-0.02, 0.02]    # 位置增量 (m)
    delta_rot: [-0.1, 0.1]      # 旋转增量 (rad)
    gripper: [0.0, 1.0]         # 夹爪开合度

reward:
  type: "sparse"                # 奖励类型: sparse / dense / classifier
  success_threshold: 0.02       # 成功判定阈值 (m)
  time_penalty: -0.001          # 时间惩罚

episode:
  max_steps: 200                # 最大步数
  timeout: 10.0                 # 超时时间 (s)
```

### 参数说明

#### 机械臂控制
- **control_mode**:
  - `twist`: 使用速度控制（推荐）
  - `joint`: 使用关节空间控制
- **frequency**: 控制循环频率，影响响应速度
- **dt**: 时间步长，应等于 1/frequency

#### 工作空间
- **center**: 任务的中心位置
- **radius**: 机械臂允许移动的范围
- **z_min/z_max**: 高度限制，防止碰撞

#### 相机配置
- **backend**: 选择相机类型
  - `realsense`: Intel RealSense 深度相机
  - `webcam`: 普通 USB 摄像头
  - `dummy`: 虚拟相机（用于测试）
- **cameras**: 多相机配置
- **image_size**: 输入神经网络的图像尺寸

#### 观测空间
- **state_keys**: 包含在状态向量中的信息
- **image_keys**: 使用的相机图像

#### 动作空间
- **type**:
  - `delta_pose`: 位姿增量（推荐用于RL）
  - `absolute_pose`: 绝对位姿
  - `joint`: 关节角度
- **limits**: 动作限制，确保安全

---

## 3. 任务和训练配置

**文件**: `hil_serl_kinova/experiments/kinova_reaching/config.py`

```python
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    # ============ 任务配置 ============
    config.task_name = "kinova_reaching"
    config.task_description = "Reach to a target pose"

    # 目标位姿 [x, y, z, qx, qy, qz, qw]
    config.target_pose = [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]

    # 成功判定
    config.success_threshold = 0.02  # meters

    # ============ BC 配置 ============
    config.bc_config = ConfigDict()
    config.bc_config.epochs = 50
    config.bc_config.batch_size = 256
    config.bc_config.learning_rate = 3e-4
    config.bc_config.weight_decay = 1e-4

    # 网络结构
    config.bc_config.state_dim = 21      # tcp_pose(7) + tcp_vel(6) + joints(7) + gripper(1)
    config.bc_config.action_dim = 7      # delta_pose(6) + gripper(1)
    config.bc_config.image_size = [128, 128]
    config.bc_config.hidden_dims = [256, 256]

    # 数据增强
    config.bc_config.augmentation = ConfigDict()
    config.bc_config.augmentation.enabled = True
    config.bc_config.augmentation.random_crop = True
    config.bc_config.augmentation.color_jitter = 0.1

    # ============ Reward Classifier 配置 ============
    config.classifier_config = ConfigDict()
    config.classifier_config.epochs = 20
    config.classifier_config.batch_size = 64
    config.classifier_config.learning_rate = 1e-4
    config.classifier_config.hidden_dims = [256, 128]

    # ============ RLPD 配置 ============
    config.rlpd_config = ConfigDict()

    # SAC 参数
    config.rlpd_config.gamma = 0.99               # 折扣因子
    config.rlpd_config.tau = 0.005                # 软更新系数
    config.rlpd_config.alpha = 0.2                # 熵权重
    config.rlpd_config.actor_lr = 3e-4
    config.rlpd_config.critic_lr = 3e-4

    # 训练步数
    config.rlpd_config.offline_steps = 10000      # 离线预训练步数
    config.rlpd_config.online_steps = 50000       # 在线学习步数
    config.rlpd_config.update_every = 1           # 每 N 步更新一次
    config.rlpd_config.num_updates = 1            # 每次更新的梯度步数

    # Replay Buffer
    config.rlpd_config.buffer_size = 100000
    config.rlpd_config.demo_ratio = 0.5           # 演示数据占比

    # 评估
    config.rlpd_config.eval_frequency = 1000      # 评估频率
    config.rlpd_config.eval_episodes = 10         # 评估回合数

    return config
```

### 配置说明

#### 任务配置
- **target_pose**: 目标位姿，四元数格式 [x, y, z, qx, qy, qz, qw]
- **success_threshold**: 到达目标的距离阈值

#### BC 训练
- **epochs**: 训练轮数
- **batch_size**: 批大小
- **learning_rate**: 学习率
- **hidden_dims**: 隐藏层维度

#### RLPD 训练
- **gamma**: 折扣因子，控制未来奖励的权重
- **tau**: 目标网络软更新系数
- **alpha**: SAC 熵正则化系数
- **demo_ratio**: Replay Buffer 中演示数据的比例

---

## 配置最佳实践

### 1. 快速测试

```yaml
# kinova_config.yaml
camera:
  enabled: false  # 禁用相机

episode:
  max_steps: 50   # 减少步数
```

```python
# config.py
config.bc_config.epochs = 5
config.bc_config.batch_size = 32
```

### 2. 生产环境

```yaml
camera:
  enabled: true
  backend: "realsense"

control:
  frequency: 100  # 提高控制频率
```

```python
config.bc_config.epochs = 100
config.bc_config.batch_size = 256
config.rlpd_config.online_steps = 100000
```

### 3. 安全优先

```yaml
mapper:
  max_linear_velocity: 0.005   # 降低速度
  max_angular_velocity: 0.02

safety:
  emergency_stop_enabled: true
  workspace_radius: 0.4         # 缩小工作空间
```

## 环境变量

可以通过环境变量覆盖部分配置：

```bash
export KINOVA_IP="192.168.8.10"
export VISIONPRO_IP="192.168.1.125"
export CAMERA_BACKEND="dummy"
```

## 下一步

- [快速开始](QUICKSTART.md) - 使用配置运行示例
- [API 文档](API.md) - 编程接口说明
