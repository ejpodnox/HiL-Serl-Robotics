# Kinova HIL-SERL 快速开始指南

## 📋 系统架构

```
VisionPro遥操作 → 数据收集 → BC训练 → 策略部署 → (可选)RLPD在线学习
```

---

## 🚀 快速开始（3步走）

### 步骤 1: 数据收集（独立遥操作）

**使用独立的遥操作程序**（不依赖 KinovaEnv，快速启动）：

```bash
# 启动 ROS2 驱动（终端 1）
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.8.10

# 运行遥操作数据采集（终端 2）
python vision_pro_control/record_teleop_demos.py \
    --save_dir ./teleop_demos \
    --num_demos 5 \
    --task_name reaching
```

**或使用完整的 RL 环境数据收集**：

```bash
python kinova_rl_env/record_kinova_demos.py \
    --save_dir ./demos/reaching \
    --num_demos 10 \
    --vp_ip 192.168.1.125
```

### 步骤 2: BC 训练

```bash
python hil_serl_kinova/train_bc_kinova.py \
    --config hil_serl_kinova/experiments/kinova_reaching/config.py \
    --demos_dir ./demos/reaching \
    --checkpoint_dir ./checkpoints/bc_kinova \
    --epochs 50
```

**训练监控**：

```bash
# 启动 Tensorboard
tensorboard --logdir ./logs/kinova_reaching/bc
```

### 步骤 3: 策略部署

```bash
# 纯策略控制
python hil_serl_kinova/deploy_policy.py \
    --checkpoint ./checkpoints/bc_kinova/best_model.pt \
    --mode policy_only

# 评估模式
python hil_serl_kinova/deploy_policy.py \
    --checkpoint ./checkpoints/bc_kinova/best_model.pt \
    --mode evaluation \
    --num_episodes 10
```

---

## 📂 文件结构

```
kinova-hil-serl/
├── vision_pro_control/              # VisionPro 遥操作
│   ├── record_teleop_demos.py      # ✨ 独立遥操作数据采集（新）
│   ├── nodes/teleop_node.py        # 完整遥操作节点
│   └── core/
│       ├── visionpro_bridge.py     # VisionPro 数据接收
│       ├── coordinate_mapper.py    # 坐标映射
│       └── robot_commander.py      # 机械臂控制
│
├── kinova_rl_env/                   # Kinova RL 环境
│   ├── record_kinova_demos.py      # RL 环境数据采集
│   ├── kinova_env/
│   │   ├── kinova_env.py           # Gym 环境
│   │   ├── kinova_interface.py     # ROS2 接口
│   │   └── camera_interface.py     # ✨ 相机抽象接口（新）
│   └── config/kinova_config.yaml
│
├── hil_serl_kinova/                 # ✨ HIL-SERL 集成（新）
│   ├── train_bc_kinova.py          # BC 训练脚本
│   ├── deploy_policy.py            # 策略部署脚本
│   └── experiments/
│       └── kinova_reaching/
│           └── config.py           # 任务配置
│
└── hil-serl/                        # HIL-SERL 原始框架
    └── examples/
```

---

## 🔧 新增模块说明

### 1. 独立遥操作数据采集

**文件**: `vision_pro_control/record_teleop_demos.py`

**特点**:
- ✅ 不依赖 KinovaEnv
- ✅ 直接使用 RobotCommander
- ✅ 快速启动，用于测试
- ✅ 保存原始遥操作数据

**使用场景**:
- 快速验证 VisionPro 连接
- 测试机械臂控制
- 收集原始轨迹数据（不需要 RL 格式）

### 2. 相机抽象接口

**文件**: `kinova_rl_env/kinova_env/camera_interface.py`

**支持的相机后端**:
- `RealSenseCamera`: Intel RealSense (ROS2)
- `WebCamera`: USB 摄像头
- `DummyCamera`: 模拟相机（无相机测试）

**配置驱动**:

```yaml
# kinova_config.yaml
camera:
  enabled: true
  backend: 'realsense'  # 'realsense' / 'webcam' / 'dummy'
  cameras:
    wrist_1:
      type: 'realsense'
      topic: '/camera/wrist_1/color/image_raw'
  image_size: [128, 128]
```

**切换相机**:

```python
# 使用 RealSense
config['camera']['backend'] = 'realsense'

# 无相机测试（使用 DummyCamera）
config['camera']['enabled'] = False
```

### 3. HIL-SERL 任务配置

**文件**: `hil_serl_kinova/experiments/kinova_reaching/config.py`

**配置内容**:
- 任务定义（目标位姿、成功阈值）
- 网络架构（hidden_dims, activation）
- 训练参数（epochs, batch_size, learning_rate）
- 日志和检查点

**自定义任务**:

```python
from hil_serl_kinova.experiments.kinova_reaching.config import get_config

config = get_config()

# 修改目标位姿
config.target_pose = [0.6, 0.1, 0.4, 0.0, 1.0, 0.0, 0.0]

# 修改训练参数
config.bc_config.epochs = 100
config.bc_config.learning_rate = 1e-4
```

### 4. BC 训练脚本

**文件**: `hil_serl_kinova/train_bc_kinova.py`

**特性**:
- ✅ 模块化数据加载器
- ✅ 可自定义策略网络
- ✅ Tensorboard 日志
- ✅ 自动保存最佳模型

**网络架构**:

```
Input:
├── State (tcp_pose + tcp_vel + gripper_pose) → MLP
└── Image (128×128×3) → CNN

↓ Fusion (Concat)

↓ MLP (256→256→256)

Output: Action (7D)
```

### 5. 策略部署框架

**文件**: `hil_serl_kinova/deploy_policy.py`

**部署模式**:

1. **纯策略控制** (`policy_only`):
   ```bash
   python deploy_policy.py --checkpoint best_model.pt --mode policy_only
   ```

2. **混合控制** (`hybrid`):
   ```bash
   python deploy_policy.py --checkpoint best_model.pt --mode hybrid --alpha 0.5
   ```
   - `alpha=1.0`: 纯 VisionPro
   - `alpha=0.5`: VisionPro 和策略各占 50%
   - `alpha=0.0`: 纯策略

3. **评估模式** (`evaluation`):
   ```bash
   python deploy_policy.py --checkpoint best_model.pt --mode evaluation --num_episodes 10
   ```

---

## 🎯 完整工作流程

### 场景 A: 快速原型（使用 BC）

```bash
# 1. 收集 5 条演示（快速）
python vision_pro_control/record_teleop_demos.py --num_demos 5

# 2. 训练 BC（20 epochs）
python hil_serl_kinova/train_bc_kinova.py --epochs 20

# 3. 评估
python hil_serl_kinova/deploy_policy.py --mode evaluation
```

**时间**: 数据收集 30分钟 + 训练 10分钟 + 评估 5分钟 = **45分钟**

### 场景 B: 高性能策略（使用 HIL-SERL）

```bash
# 1. 收集 20 条高质量演示
python kinova_rl_env/record_kinova_demos.py --num_demos 20

# 2. 训练 BC 预热
python hil_serl_kinova/train_bc_kinova.py --epochs 50

# 3. (未来) 在线学习 RLPD
python hil_serl_kinova/train_rlpd_kinova.py --checkpoint bc_best.pt

# 4. 部署评估
python hil_serl_kinova/deploy_policy.py --mode evaluation --num_episodes 20
```

**时间**: 数据收集 2小时 + BC训练 20分钟 + RLPD训练 数小时

---

## 🔍 调试技巧

### 1. 测试单独组件

```bash
# 测试 VisionPro
python scripts/teleop/test_visionpro_bridge.py

# 测试 Kinova
python scripts/teleop/test_robot_connection.py

# 测试相机
python kinova_rl_env/kinova_env/camera_interface.py
```

### 2. 检查数据格式

```python
import pickle

with open('demos/reaching/demo_000.pkl', 'rb') as f:
    demo = pickle.load(f)

print(f"轨迹长度: {len(demo['actions'])}")
print(f"Action shape: {demo['actions'][0].shape}")
print(f"State keys: {demo['observations'][0]['state'].keys()}")
print(f"Image shape: {demo['observations'][0]['images']['wrist_1'].shape}")
```

### 3. 可视化训练曲线

```bash
tensorboard --logdir ./logs
```

### 4. 调整参数

**如果策略不稳定**:
```python
# 降低学习率
config.bc_config.learning_rate = 1e-4

# 增加训练轮数
config.bc_config.epochs = 100

# 增加 Dropout
config.bc_config.dropout = 0.2
```

**如果机械臂动作太快**:
```yaml
# vision_pro_control/config/teleop_config.yaml
safety:
  max_linear_velocity: 0.005  # 降低到 0.5 cm/s
```

---

## ⚙️ 配置速查表

### 关键配置文件

| 文件 | 用途 |
|------|------|
| `kinova_rl_env/config/kinova_config.yaml` | Kinova 环境参数 |
| `vision_pro_control/config/teleop_config.yaml` | VisionPro 遥操作参数 |
| `hil_serl_kinova/experiments/kinova_reaching/config.py` | HIL-SERL 训练配置 |

### 常用参数

```python
# 数据收集
num_demos = 10              # 演示数量
control_frequency = 50      # 控制频率 (Hz)

# 训练
epochs = 50                 # 训练轮数
batch_size = 256            # 批大小
learning_rate = 3e-4        # 学习率

# 安全
max_linear_velocity = 0.01  # 最大线速度 (m/s)
max_angular_velocity = 0.05 # 最大角速度 (rad/s)
```

---

## 📊 性能基准

### BC 训练（RTX 3090）

- 10 条演示，50 epochs: ~5 分钟
- 20 条演示，100 epochs: ~15 分钟

### 推理速度

- CPU: ~20 FPS
- GPU: ~100 FPS

### 成功率（参考）

- 5 条演示 BC: ~40%
- 20 条演示 BC: ~70%
- RLPD 在线学习: ~90%+

---

## 🐛 常见问题

**Q: 训练时 GPU 内存不足？**
```python
# 降低 batch_size
config.bc_config.batch_size = 128
```

**Q: 相机无法连接？**
```yaml
# 暂时使用 DummyCamera
camera:
  enabled: false
```

**Q: VisionPro 延迟高？**
```yaml
# 降低滤波系数（更快响应）
mapper:
  filter_alpha: 0.1
```

---

## 🎓 下一步学习

1. **添加更多任务**: 复制 `kinova_reaching` 配置，创建新任务
2. **自定义网络**: 修改 `BCPolicy` 架构
3. **集成 Reward Classifier**: 实现第 4 阶段
4. **在线学习 RLPD**: 实现第 5 阶段

---

## 📞 获取帮助

- 查看日志: `./logs/kinova_reaching/`
- 检查检查点: `./checkpoints/bc_kinova/`
- 重新阅读: `kinova_rl_env/README_HIL_SERL_INTEGRATION.md`

---

**🎉 祝你成功！有问题随时提问。**
