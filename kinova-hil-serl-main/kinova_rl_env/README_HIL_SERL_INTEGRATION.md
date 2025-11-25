# Kinova HIL-SERL Integration Guide

## 📋 概述

本文档说明如何使用Kinova Gen3机械臂 + VisionPro遥操作收集HIL-SERL演示数据。

---

## ✅ 已完成的实现（阶段1和阶段2）

### 阶段1：完善KinovaEnv

#### 1.1 修正Observation Space格式
- ✅ 改为嵌套字典，符合HIL-SERL标准
- ✅ 支持TCP位姿、速度、gripper状态
- ✅ 图像resize到128x128

#### 1.2 TCP位姿获取
- ✅ 使用TF2从ROS获取TCP位姿
- ✅ `get_tcp_pose()` 返回 `[x, y, z, qx, qy, qz, qw]`
- ✅ `get_tcp_velocity()` 返回 `[vx, vy, vz, wx, wy, wz]`

#### 1.3 Reward函数
- ✅ 支持sparse和dense两种模式
- ✅ `_check_success()` 判断是否到达目标
- ✅ 可配置目标位姿和成功阈值

#### 1.4 Gripper控制
- ✅ `set_gripper(position)` 发送gripper命令
- ✅ `get_gripper_state()` 获取当前状态

### 阶段2：VisionPro数据收集

#### 2.1 数据收集脚本
- ✅ `record_kinova_demos.py` - 主脚本
- ✅ 集成KinovaEnv + VisionPro + CoordinateMapper
- ✅ 实时显示距离、累积奖励
- ✅ 保存为HIL-SERL格式

#### 2.2 Twist到Action转换
- ✅ `twist_to_action()` 函数
- ✅ 速度 × dt → 位移增量
- ✅ Gripper: pinch distance → position

#### 2.3 测试脚本
- ✅ `tests/unit/test_demo_format.py` - 验证数据格式
- ✅ `tests/run_all_tests.sh` - 统一测试运行器
- ✅ `tests/utils/save_demo_utils.py` - pkl/hdf5转换工具

---

## 🚀 使用流程

### 1. 准备工作

#### 检查依赖
```bash
# Python包
pip install numpy scipy opencv-python gymnasium pyyaml

# ROS2依赖
sudo apt install ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
```

#### 检查Kinova机械臂
```bash
# 启动Kinova驱动
ros2 launch kortex_bringup kortex_control.launch.py

# 检查topics
ros2 topic list
# 应该看到: /joint_states, /twist_controller/commands, /tf, etc.
```

#### 检查TF坐标系名称
```bash
# 查看TF树
ros2 run tf2_tools view_frames

# 确认坐标系名称，可能是：
# - base_link
# - tool_frame, end_effector_link, 或 tcp_link
```

**重要**：如果坐标系名称不同，修改 `kinova_interface.py` Line 48-49：
```python
self.base_frame = 'base_link'  # 你的基座坐标系名称
self.tool_frame = 'tool_frame'  # 你的末端坐标系名称
```

#### 配置任务目标
编辑 `config/kinova_config.yaml`：
```yaml
task:
  name: "reaching"
  # 设置目标位置（在base_link坐标系下）
  target_pose: [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]  # [x, y, z, qx, qy, qz, qw]
  success_threshold:
    position: 0.02  # 2cm
```

### 2. 标定VisionPro（如果还没标定）

```bash
cd vision_pro_control
python nodes/teleop_node.py

# 按照提示标定：
# 1. 's' - 采样5-10次
# 2. 'c' - 保存中心点
# 3. Enter - 确认完成
```

### 3. 收集演示数据

```bash
cd kinova_rl_env

python record_kinova_demos.py \
    --save_dir ./demos \
    --num_demos 10 \
    --task reaching \
    --vp_ip 192.168.1.125
```

**操作流程**：
1. 按 **Space** - 开始记录新的demo
2. 用VisionPro遥操作控制机械臂完成任务
3. 按 **'s'** - 标记成功并保存
4. 按 **'f'** - 标记失败并丢弃
5. 重复直到收集足够数据

**按键说明**：
- `Space` - 开始记录（重置环境）
- `s` - 标记成功并保存
- `f` - 标记失败并丢弃
- `r` - 重置环境（不记录）
- `p` - 暂停/恢复记录
- `q` - 退出

### 4. 验证数据格式

```bash
python tests/unit/test_demo_format.py --demo_path demos/reaching/demo_000.pkl
```

应该看到：
```
✓ 'observations' 存在
✓ 'actions' 存在
✓ 'rewards' 存在
✓ observation包含'state'和'images'键
✓ Action维度正确 (7,)
...
```

---

## 📊 数据格式说明

### Demo文件格式 (.pkl)

```python
{
    'observations': [
        {
            'state': {
                'tcp_pose': np.array([x, y, z, qx, qy, qz, qw]),  # (7,)
                'tcp_vel': np.array([vx, vy, vz, wx, wy, wz]),   # (6,)
                'gripper_pose': np.array([position])              # (1,)
            },
            'images': {
                'wrist_1': np.array([128, 128, 3], dtype=uint8)
            }
        },
        ...  # T个时间步
    ],
    'actions': [
        np.array([dx, dy, dz, drx, dry, drz, gripper]),  # (7,)
        ...  # T个时间步
    ],
    'rewards': [0.0, 0.0, ..., 1.0],  # T个浮点数
    'terminals': [False, False, ..., True],  # T个布尔值
    'truncations': [False, False, ..., False],  # T个布尔值（可选）
    'success': True  # 布尔值
}
```

### Action维度说明

Action是7维向量：`[dx, dy, dz, drx, dry, drz, gripper]`

- `dx, dy, dz`: TCP位置增量（米）
- `drx, dry, drz`: TCP姿态增量（弧度，轴角表示）
- `gripper`: Gripper位置，0.0（全开）~ 1.0（全闭）

**Twist到Action的转换**：
```python
# Twist是速度（m/s, rad/s）
# Action是位移增量
action = twist × dt
```

---

## 🐛 常见问题排查

### 问题1：无法获取TCP位姿

**症状**：`get_tcp_pose()` 返回None

**原因**：TF坐标系名称不匹配

**解决**：
```bash
# 查看可用的坐标系
ros2 run tf2_ros tf2_echo base_link <TAB>  # 按TAB补全

# 修改 kinova_interface.py 的坐标系名称
self.tool_frame = 'end_effector_link'  # 或其他名称
```

### 问题2：VisionPro连接失败

**症状**：`Connecting VisionPro...` 卡住

**解决**：
1. 检查VisionPro和电脑在同一WiFi
2. 检查IP地址：`ping 192.168.1.125`
3. 确保VisionPro上的Tracking Streamer应用正在运行

### 问题3：Gripper不动作

**症状**：发送gripper命令，但gripper不动

**原因**：
- Gripper话题名称不正确
- Gripper controller未启动

**解决**：
```bash
# 检查gripper话题
ros2 topic list | grep gripper

# 修改 kinova_interface.py 的话题名称
self.gripper_command_topic = '/robotiq_gripper_controller/gripper_cmd'
```

### 问题4：机械臂动作不平滑

**症状**：机械臂抖动或动作突变

**原因**：
- 控制频率太低
- VisionPro数据延迟
- 增益太高

**解决**：
1. 降低增益：编辑 `vision_pro_control/config/teleop_config.yaml`
```yaml
mapper:
  position_gain: 0.2  # 降低
  rotation_gain: 0.2
```

2. 增加滤波：
```yaml
mapper:
  filter_alpha: 0.1  # 更平滑（0越小越平滑）
```

---

## 🔧 下一步：训练RL策略

### 创建任务配置文件

参考 `hil-serl/examples/experiments/*/config.py`，创建Kinova任务配置：

```python
# kinova_experiments/reaching/config.py

from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.task_name = "kinova_reaching"
    config.server_url = None  # 不使用server，直接ROS2

    # 从kinova_config.yaml读取
    config.target_pose = [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]
    config.reset_pose = [0.3, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0]

    # 训练参数
    config.bc_epochs = 20
    config.batch_size = 256
    config.demos_num = 10
    config.utd_ratio = 4

    return config
```

### 修改训练脚本

修改 `hil-serl/examples/train_rlpd.py`：

```python
# 导入KinovaEnv
from kinova_rl_env.kinova_env import KinovaEnv

# 在actor函数中创建环境
env = KinovaEnv(config_path="path/to/kinova_config.yaml")

# 其他代码保持不变，HIL-SERL的训练逻辑可以直接复用
```

---

## 📝 文件结构

```
kinova_rl_env/
├── kinova_env/
│   ├── kinova_env.py          # ✅ Gym环境（已改进）
│   ├── kinova_interface.py    # ✅ ROS2接口（已添加TCP位姿）
│   └── config_loader.py       # 配置加载器
├── config/
│   └── kinova_config.yaml     # ✅ 配置文件（已添加任务参数）
├── tests/                      # ✅ 统一测试目录（已重组）
│   ├── hardware/               # 硬件测试
│   │   ├── test_ros2_connection.py
│   │   ├── test_velocity_control.py
│   │   └── test_robot_connection.py
│   ├── unit/                   # 单元测试
│   │   └── test_demo_format.py
│   ├── visionpro/              # VisionPro测试
│   │   ├── test_visionpro_bridge.py
│   │   ├── test_calibration.py
│   │   └── test_teleop.py
│   ├── integration/            # 集成测试
│   │   └── test_teleop_all.py
│   ├── utils/                  # 测试工具
│   │   └── save_demo_utils.py  # pkl/hdf5转换工具
│   ├── run_all_tests.sh        # 统一测试运行器
│   └── README.md               # 测试文档
├── record_kinova_demos.py      # ✅ 数据收集脚本
├── run_tests.sh                # 兼容性脚本（重定向到tests/）
└── README_HIL_SERL_INTEGRATION.md  # 本文档

vision_pro_control/             # VisionPro遥操作（已有）
├── core/
│   ├── visionpro_bridge.py
│   ├── coordinate_mapper.py
│   └── calibrator.py
└── config/
    └── calibration.yaml
```

---

## 📞 技术支持

遇到问题？检查：
1. ROS2话题是否正常发布：`ros2 topic echo /joint_states`
2. TF是否正常：`ros2 run tf2_ros tf2_echo base_link tool_frame`
3. VisionPro数据是否正常：运行 `vision_pro_control/nodes/teleop_node.py`

---

**🎉 恭喜！你现在可以开始收集数据并训练HIL-SERL策略了！**
