# Kinova HIL-SERL 测试文档

## 📋 目录结构

```
tests/
├── hardware/               # 硬件层测试
│   ├── test_ros2_connection.py      # ROS2连接和关节状态读取
│   ├── test_velocity_control.py     # 速度控制测试
│   └── test_robot_connection.py     # 机器人连接测试
│
├── unit/                   # 单元测试
│   └── test_demo_format.py          # Demo数据格式验证
│
├── visionpro/              # VisionPro测试
│   ├── test_visionpro_bridge.py     # VisionPro连接测试
│   ├── test_calibration.py          # 校准测试
│   └── test_teleop.py               # 遥操作测试
│
├── integration/            # 集成测试
│   └── test_teleop_all.py           # 完整遥操作集成测试
│
├── utils/                  # 测试工具
│   └── save_demo_utils.py           # Demo数据保存工具（pkl/hdf5）
│
├── run_all_tests.sh        # 统一测试运行器
└── README.md               # 本文档
```

---

## 🚀 快速开始

### 运行所有自动化测试

```bash
cd kinova_rl_env
./tests/run_all_tests.sh
```

或使用兼容性脚本：

```bash
cd kinova_rl_env
./run_tests.sh  # 自动重定向到 tests/run_all_tests.sh
```

### 运行特定类别的测试

```bash
# 仅运行硬件测试
./tests/run_all_tests.sh hardware

# 仅运行单元测试
./tests/run_all_tests.sh unit

# 仅运行VisionPro测试（需要硬件）
./tests/run_all_tests.sh visionpro

# 仅运行集成测试
./tests/run_all_tests.sh integration
```

---

## 📊 测试分类说明

### 1. 硬件测试 (Hardware Tests)

测试ROS2连接和机器人硬件功能。

**前置条件：**
- ROS2环境已source
- Kinova机械臂已连接并启动驱动

**运行硬件驱动：**
```bash
ros2 launch kortex_bringup kortex_control.launch.py robot_ip:=192.168.1.10
```

**测试内容：**
- ✅ ROS2环境检查
- ✅ `/joint_states`话题检查
- ✅ TF变换检查（`base_link` → `tool_frame`）
- ✅ 关节状态读取测试
- ✅ 速度控制测试
- ✅ 机器人连接测试

**单独运行：**
```bash
# ROS2关节状态读取
python tests/hardware/test_ros2_connection.py

# 速度控制测试
python tests/hardware/test_velocity_control.py

# 机器人连接测试
python tests/hardware/test_robot_connection.py
```

---

### 2. 单元测试 (Unit Tests)

测试各个模块的基础功能。

**测试内容：**
- ✅ `KinovaInterface` 基础功能
  - 连接/断开
  - 获取关节状态
  - 获取TCP位姿
- ✅ `KinovaEnv` 环境测试
  - Observation space格式
  - Reward计算
  - 环境重置
- ✅ Demo数据格式验证

**单独运行：**
```bash
# Demo格式验证
python tests/unit/test_demo_format.py --demo_path demos/reaching/demo_000.pkl
```

---

### 3. VisionPro测试 (VisionPro Tests)

测试VisionPro集成和遥操作功能。

**前置条件：**
- VisionPro已连接到同一WiFi
- Tracking Streamer应用正在运行
- 知道VisionPro的IP地址

**测试内容：**
- ✅ VisionPro数据接收
- ✅ 手部追踪
- ✅ Pinch检测
- ✅ 校准流程
- ✅ 遥操作控制

**单独运行：**
```bash
# VisionPro连接测试
python tests/visionpro/test_visionpro_bridge.py

# 校准测试
python tests/visionpro/test_calibration.py

# 遥操作测试
python tests/visionpro/test_teleop.py
```

---

### 4. 集成测试 (Integration Tests)

测试完整的数据收集和控制流程。

**前置条件：**
- 所有硬件测试通过
- 所有单元测试通过
- VisionPro已校准

**测试内容：**
- ✅ VisionPro + Kinova 完整遥操作
- ✅ 数据收集流程
- ✅ 端到端控制链路

**单独运行：**
```bash
# 完整遥操作测试
python tests/integration/test_teleop_all.py

# 数据收集流程测试（收集1条demo）
python record_kinova_demos.py --save_dir ./demos --num_demos 1 --task reaching
```

---

## 🔧 测试工具

### Demo数据工具 (`tests/utils/save_demo_utils.py`)

提供pkl和hdf5两种格式的demo数据保存和转换。

**使用方法：**

```bash
# 转换单个pkl到hdf5
python tests/utils/save_demo_utils.py --convert demos/reaching/demo_000.pkl

# 批量转换目录下的所有pkl
python tests/utils/save_demo_utils.py --batch_convert demos/reaching

# 测试读取hdf5文件
python tests/utils/save_demo_utils.py --test_hdf5 demos/reaching/demo_000.h5
```

**格式对比：**

| 格式 | 优点 | 适用场景 |
|------|------|----------|
| pkl  | 简单、兼容HIL-SERL、便于调试 | 小到中等规模数据集（<100条demo） |
| hdf5 | 高效、压缩、可增量读写 | 大规模数据集（>100条demo） |

---

## 🐛 常见问题排查

### 问题1：ROS2环境未启动

**症状：**
```
✗ ROS2环境未启动或未source
```

**解决：**
```bash
source /opt/ros/humble/setup.bash
```

### 问题2：/joint_states话题不存在

**症状：**
```
✗ /joint_states 不存在
```

**解决：**
```bash
# 启动Kinova驱动
ros2 launch kortex_bringup kortex_control.launch.py robot_ip:=192.168.1.10
```

### 问题3：TF查询超时

**症状：**
```
⚠ TF查询超时，可能需要检查坐标系名称
```

**解决：**
```bash
# 查看可用的TF坐标系
ros2 run tf2_tools view_frames

# 检查坐标系名称
ros2 run tf2_ros tf2_echo base_link <TAB>  # 按TAB补全

# 修改 kinova_env/kinova_interface.py 中的坐标系名称
# Line 48-49:
# self.base_frame = 'base_link'  # 你的基座坐标系名称
# self.tool_frame = 'tool_frame'  # 你的末端坐标系名称
```

### 问题4：VisionPro连接失败

**症状：**
```
Connecting VisionPro... (卡住)
```

**解决：**
1. 检查VisionPro和电脑在同一WiFi
2. 检查IP地址：`ping 192.168.1.125`
3. 确保VisionPro上的Tracking Streamer应用正在运行
4. 检查防火墙设置

### 问题5：未找到demo文件

**症状：**
```
⚠ 未找到demo文件（尚未收集数据）
```

**解决：**
```bash
# 收集演示数据
python record_kinova_demos.py --save_dir ./demos --num_demos 10 --task reaching
```

---

## 📝 测试流程总览

### 阶段1：环境检查（必需）

```bash
# 1. 检查ROS2
source /opt/ros/humble/setup.bash
ros2 topic list

# 2. 启动Kinova驱动
ros2 launch kortex_bringup kortex_control.launch.py robot_ip:=192.168.1.10

# 3. 运行硬件测试
./tests/run_all_tests.sh hardware
```

### 阶段2：单元测试（必需）

```bash
# 运行单元测试
./tests/run_all_tests.sh unit
```

### 阶段3：VisionPro测试（可选，需要硬件）

```bash
# 手动运行VisionPro测试
python tests/visionpro/test_visionpro_bridge.py
python tests/visionpro/test_calibration.py
```

### 阶段4：数据收集（生产环境）

```bash
# 收集演示数据
python record_kinova_demos.py \
    --save_dir ./demos \
    --num_demos 10 \
    --task reaching \
    --vp_ip 192.168.1.125

# 验证数据格式
python tests/unit/test_demo_format.py --demo_path demos/reaching/demo_000.pkl

# （可选）转换为hdf5
python tests/utils/save_demo_utils.py --batch_convert demos/reaching
```

### 阶段5：训练RL策略（未来）

```bash
# 参考 hil-serl 文档
# 修改训练脚本以使用KinovaEnv
```

---

## 📞 技术支持

遇到问题？检查：

1. **ROS2话题是否正常：** `ros2 topic echo /joint_states`
2. **TF是否正常：** `ros2 run tf2_ros tf2_echo base_link tool_frame`
3. **VisionPro数据是否正常：** 运行 `tests/visionpro/test_visionpro_bridge.py`
4. **配置文件是否正确：** 检查 `config/kinova_config.yaml`

---

## 🎯 下一步

完成所有测试后：

1. ✅ 收集足够的演示数据（建议10-50条）
2. ✅ 验证数据格式和质量
3. ✅ 准备训练环境（参考 `README_HIL_SERL_INTEGRATION.md`）
4. ✅ 开始HIL-SERL训练

**🎉 祝你训练顺利！**
