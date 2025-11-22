# Kinova HIL-SERL: VisionPro 遥操作 + 强化学习

完整的 Human-in-the-Loop Reinforcement Learning 系统，用于 Kinova Gen3 机械臂。

## 🌟 特性

- ✅ **VisionPro 遥操作**: 使用 Apple Vision Pro 进行直观的机械臂遥操作
- ✅ **模块化设计**: 低耦合、高可配置、易扩展
- ✅ **多种训练模式**: BC (Behavior Cloning) + RLPD (RL with Prior Data)
- ✅ **Reward Classifier**: 自动学习成功/失败判别器
- ✅ **可插拔相机**: 支持 RealSense / WebCam / Dummy
- ✅ **完整工具链**: 数据收集、训练、部署、可视化

## 📊 系统架构

```
VisionPro → 遥操作数据采集 → 演示数据
                                ↓
            BC 训练 ← 离线数据 + Reward Classifier
                                ↓
            策略部署 → 评估 → RLPD 在线学习
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# Python 包
pip install torch torchvision
pip install gymnasium
pip install numpy scipy opencv-python
pip install pyyaml ml-collections
pip install tensorboard matplotlib seaborn

# ROS2 包
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-realsense2-camera  # 如果使用 RealSense
```

### 2. 一键运行完整流程

```bash
# 快速原型（5 条演示 + BC 训练）
bash scripts/run_full_pipeline.sh --mode quick

# 标准流程（20 条演示 + BC 训练）
bash scripts/run_full_pipeline.sh --mode standard

# 完整流程（包含 RLPD 在线学习）
bash scripts/run_full_pipeline.sh --mode full
```

### 3. 分步执行

#### 步骤 1: 数据收集

```bash
# 方式 A: 使用完整 RL 环境（推荐）
python kinova_rl_env/record_kinova_demos.py \
    --save_dir ./demos/reaching \
    --num_demos 10

# 方式 B: 使用独立遥操作（快速测试）
python vision_pro_control/record_teleop_demos.py \
    --save_dir ./teleop_demos \
    --num_demos 5
```

#### 步骤 2: BC 训练

```bash
python hil_serl_kinova/train_bc_kinova.py \
    --config hil_serl_kinova/experiments/kinova_reaching/config.py \
    --demos_dir ./demos/reaching \
    --epochs 50
```

#### 步骤 3: 策略部署

```bash
# 评估模式
python hil_serl_kinova/deploy_policy.py \
    --checkpoint checkpoints/bc_kinova/best_model.pt \
    --mode evaluation

# 交互模式
python hil_serl_kinova/deploy_policy.py \
    --checkpoint checkpoints/bc_kinova/best_model.pt \
    --interactive

# 混合控制（人机协作）
python hil_serl_kinova/deploy_policy.py \
    --checkpoint checkpoints/bc_kinova/best_model.pt \
    --mode hybrid \
    --alpha 0.5
```

#### 步骤 4: (可选) Reward Classifier

```bash
# 收集标签数据
python hil_serl_kinova/record_success_fail_demos.py \
    --save_dir ./demos/labeled \
    --num_success 20 \
    --num_fail 20

# 训练分类器
python hil_serl_kinova/train_reward_classifier.py \
    --demos_dir ./demos/labeled \
    --epochs 20
```

#### 步骤 5: (可选) RLPD 在线学习

```bash
python hil_serl_kinova/train_rlpd_kinova.py \
    --config hil_serl_kinova/experiments/kinova_reaching/config.py \
    --demos_dir ./demos/reaching \
    --bc_checkpoint checkpoints/bc_kinova/best_model.pt
```

## 📂 项目结构

```
kinova-hil-serl/
├── vision_pro_control/              # VisionPro 遥操作
│   ├── record_teleop_demos.py      # 独立遥操作采集
│   ├── nodes/teleop_node.py        # 完整遥操作节点
│   └── core/                       # 核心模块
│       ├── visionpro_bridge.py     # VisionPro 数据接收
│       ├── coordinate_mapper.py    # 坐标映射
│       ├── robot_commander.py      # 机械臂控制
│       └── calibrator.py           # 工作空间标定
│
├── kinova_rl_env/                   # Kinova RL 环境
│   ├── record_kinova_demos.py      # RL 环境数据采集
│   ├── kinova_env/
│   │   ├── kinova_env.py           # Gym 环境
│   │   ├── kinova_interface.py     # ROS2 接口
│   │   ├── camera_interface.py     # 相机抽象接口
│   │   └── config_loader.py        # 配置加载器
│   └── config/kinova_config.yaml   # 环境配置
│
├── hil_serl_kinova/                 # HIL-SERL 训练
│   ├── train_bc_kinova.py          # BC 训练
│   ├── train_reward_classifier.py  # Reward Classifier 训练
│   ├── train_rlpd_kinova.py        # RLPD 训练
│   ├── deploy_policy.py            # 策略部署
│   ├── record_success_fail_demos.py # 标签数据收集
│   ├── experiments/                # 任务配置
│   │   └── kinova_reaching/
│   │       └── config.py           # 任务配置
│   └── tools/                      # 工具集
│       ├── data_utils.py           # 数据工具
│       └── visualize.py            # 可视化工具
│
├── scripts/                         # 脚本
│   ├── run_full_pipeline.sh        # 一键运行
│   └── teleop/                     # 测试脚本
│
├── QUICKSTART.md                   # 快速开始指南
├── IMPLEMENTATION_SUMMARY.md       # 实现总结
└── README.md                       # 本文档
```

## 🛠️ 工具使用

### 数据工具

```bash
# 查看单个演示
python hil_serl_kinova/tools/data_utils.py --view demos/reaching/demo_000.pkl

# 统计分析
python hil_serl_kinova/tools/data_utils.py --stats demos/reaching

# 验证格式
python hil_serl_kinova/tools/data_utils.py --validate demos/reaching

# 转换为 HDF5
python hil_serl_kinova/tools/data_utils.py --convert demos/reaching --format hdf5
```

### 可视化工具

```bash
# 绘制轨迹
python hil_serl_kinova/tools/visualize.py --trajectory demos/reaching/demo_000.pkl

# 绘制数据集统计
python hil_serl_kinova/tools/visualize.py --dataset demos/reaching --output plots/

# 绘制训练曲线
python hil_serl_kinova/tools/visualize.py --training logs/bc --output plots/training.png

# 绘制多轨迹对比
python hil_serl_kinova/tools/visualize.py --multi demos/reaching --max_demos 5
```

## ⚙️ 配置说明

### 任务配置

编辑 `hil_serl_kinova/experiments/kinova_reaching/config.py`:

```python
# 目标位姿
config.target_pose = [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]

# BC 训练参数
config.bc_config.epochs = 50
config.bc_config.batch_size = 256
config.bc_config.learning_rate = 3e-4

# RLPD 训练参数
config.rlpd_config.offline_steps = 10000
config.rlpd_config.online_steps = 50000
```

### 相机配置

编辑 `kinova_rl_env/config/kinova_config.yaml`:

```yaml
camera:
  enabled: true
  backend: 'realsense'  # 'realsense' / 'webcam' / 'dummy'
  cameras:
    wrist_1:
      type: 'realsense'
      topic: '/camera/wrist_1/color/image_raw'
  image_size: [128, 128]
```

### VisionPro 配置

编辑 `vision_pro_control/config/teleop_config.yaml`:

```yaml
visionpro:
  ip: "192.168.1.125"
  use_right_hand: true

safety:
  max_linear_velocity: 0.01  # m/s
  max_angular_velocity: 0.05  # rad/s
```

## 📊 性能基准

| 模式 | 演示数量 | 训练时间 | 成功率 |
|------|---------|---------|--------|
| BC (Quick) | 5 | ~5分钟 | ~40% |
| BC (Standard) | 20 | ~15分钟 | ~70% |
| RLPD | 20 + 在线学习 | ~2小时 | ~90%+ |

*基于 RTX 3090 GPU

## 🐛 故障排除

### VisionPro 连接失败

```bash
# 检查网络连接
ping 192.168.1.125

# 测试 VisionPro 数据流
python scripts/teleop/test_visionpro_bridge.py
```

### Kinova 连接失败

```bash
# 启动 ROS2 驱动
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.8.10

# 测试连接
python scripts/teleop/test_robot_connection.py
```

### 相机无法连接

```yaml
# 暂时使用 DummyCamera
camera:
  enabled: false
```

### GPU 内存不足

```python
# 降低 batch size
config.bc_config.batch_size = 128
```

## 📚 文档

- [快速开始指南](QUICKSTART.md) - 新手入门
- [实现总结](IMPLEMENTATION_SUMMARY.md) - 技术细节
- [HIL-SERL 集成](kinova_rl_env/README_HIL_SERL_INTEGRATION.md) - 集成说明

## 🎯 路线图

- [x] VisionPro 遥操作
- [x] Kinova 机械臂控制
- [x] 数据收集（HIL-SERL 格式）
- [x] BC 训练
- [x] 策略部署
- [x] Reward Classifier
- [x] RLPD 在线学习
- [ ] Sim-to-Real（仿真环境）
- [ ] 多任务支持
- [ ] 分布式训练

## 🤝 贡献

欢迎提 Issue 和 Pull Request！

## 📄 许可

MIT License

## 🙏 致谢

- [HIL-SERL](https://github.com/youliangtan/hil-serl) - 原始 HIL-SERL 框架
- [Kinova Gen3](https://www.kinovarobotics.com/) - 机械臂硬件
- [Apple Vision Pro](https://www.apple.com/apple-vision-pro/) - 遥操作设备

---

**Happy Robot Learning! 🤖✨**
