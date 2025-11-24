# 硬件测试指南

完成代码配置后，开始测试所有硬件连接。

## 快速开始

### 一键验证所有硬件

```bash
# 测试所有硬件（推荐首次使用）
python tools/quick_verify.py

# 自定义 IP 地址
python tools/quick_verify.py \
    --vp-ip 192.168.1.125 \
    --robot-ip 192.168.8.10 \
    --camera-id 0

# 跳过某些测试
python tools/quick_verify.py --skip-vp      # 跳过 VisionPro
python tools/quick_verify.py --skip-robot   # 跳过 Kinova
python tools/quick_verify.py --skip-camera  # 跳过相机
```

**预期输出：**
```
============================================================
🔍 快速验证工具
============================================================
VisionPro IP: 192.168.1.125
Kinova IP: 192.168.8.10
相机 ID: 0
超时: 3.0s

============================================================
【1/4】VisionPro 连接验证
============================================================
✓ VisionPro 连接成功 (192.168.1.125)
  - 手腕位置: [0.234, -0.156, 0.987]

============================================================
【2/4】Kinova 机械臂验证
============================================================
✓ Kinova 机械臂连接成功 (192.168.8.10)
  - TCP 位置: [0.450, 0.023, 0.312]

============================================================
【3/4】USB 相机验证
============================================================
✓ USB 相机连接成功 (ID=0)
  - 图像尺寸: (128, 128, 3)

============================================================
【4/4】环境创建验证
============================================================
  尝试加载配置...
  ✓ 配置加载成功
  ✓ 环境定义正确

============================================================
【验证总结】
============================================================
VisionPro 连接        : ✓ 通过
Kinova 机械臂         : ✓ 通过
USB 相机             : ✓ 通过
环境配置             : ✓ 通过

------------------------------------------------------------
总计: 4 | 通过: 4 | 失败: 0
------------------------------------------------------------

🎉 所有验证通过！可以开始数据收集
```

---

## 分步测试（如果一键验证失败）

### 1. 测试 VisionPro 连接

#### 前置条件
- VisionPro 已开机并连接到同一网络
- 已在 VisionPro 上启动 VisionProControl app
- 知道 VisionPro 的 IP 地址（设置 → Wi-Fi → 查看详情）

#### 运行测试
```bash
python tests/test_visionpro_connection.py --vp-ip 192.168.1.125
```

#### 成功标志
```
============================================================
VisionPro 连接测试
============================================================
VisionPro IP: 192.168.1.125

【测试 1/3】基本导入测试
✓ VisionProBridge 类导入成功

【测试 2/3】VisionPro 连接测试
  尝试连接到 192.168.1.125...
✓ VisionPro 连接成功
✓ 接收到数据
  - 时间戳: 1234567890.123
  - 手腕位置: [0.234, -0.156, 0.987]
  - 手腕四元数: [0.707, 0.000, 0.707, 0.000]

【测试总结】
基本导入: ✓
VisionPro连接: ✓

------------------------------------------------------------
测试通过: 2/2
```

#### 常见问题

**问题 1: 连接超时**
```
✗ 5.0s 内未收到数据
```

**解决方法：**
1. 检查 VisionPro 和电脑是否在同一网络
2. 确认 IP 地址正确
3. 检查 VisionProControl app 是否在前台运行
4. 尝试 ping VisionPro:
   ```bash
   ping 192.168.1.125
   ```

**问题 2: 连接被拒绝**
```
✗ VisionPro 连接失败: Connection refused
```

**解决方法：**
1. 重启 VisionProControl app
2. 检查防火墙设置
3. 确认 app 正在监听正确的端口

---

### 2. 测试 Kinova 机械臂连接

#### 前置条件
- Kinova 机械臂已开机并完成自检
- ROS2 Humble 环境已激活
- kortex_bringup 已启动：
  ```bash
  # 终端 1: 启动 kortex_bringup
  ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.8.10
  ```

#### 运行测试
```bash
# 终端 2: 运行测试
python tests/test_kinova_connection.py --robot-ip 192.168.8.10
```

#### 成功标志
```
============================================================
Kinova 机械臂连接测试
============================================================
机械臂 IP: 192.168.8.10

【测试 1/5】基本导入测试
✓ rclpy 模块导入成功
✓ RobotCommander 类导入成功

【测试 2/5】ROS2 初始化测试
✓ ROS2 初始化成功

【测试 3/5】机械臂连接测试
✓ RobotCommander 创建成功

【测试 4/5】机械臂状态读取测试
✓ 获取 TCP 位姿成功
  - 位置: [0.450, 0.023, 0.312]
  - 姿态: [0.000, 1.000, 0.000, 0.000]

【测试总结】
基本导入: ✓
ROS2初始化: ✓
机械臂连接: ✓
状态读取: ✓

------------------------------------------------------------
测试通过: 4/4
```

#### 常见问题

**问题 1: 超时未获取状态**
```
✗ 5.0s 内未能获取机械臂状态
  提示: 确保已启动 kortex_bringup
```

**解决方法：**
1. 检查 kortex_bringup 是否在运行：
   ```bash
   ros2 topic list | grep joint
   ```
   应该看到 `/joint_states`

2. 手动测试话题：
   ```bash
   ros2 topic echo /joint_states --once
   ```

3. 检查机械臂网络连接：
   ```bash
   ping 192.168.8.10
   ```

**问题 2: ROS2 初始化失败**
```
✗ ROS2 初始化失败: context already initialized
```

**解决方法：**
这通常不是错误，只是 ROS2 已经初始化过。测试应该继续。

---

### 3. 测试 USB 相机

#### 前置条件
- USB 相机已连接到电脑
- 相机权限正确（Linux 下可能需要 sudo）

#### 检测相机设备
```bash
# 查看可用相机
ls /dev/video*

# 输出示例：
# /dev/video0  /dev/video1
```

#### 运行测试
```bash
# 测试 USB 相机 (device_id=0)
python tests/test_camera.py --backend webcam --webcam-id 0

# 如果有多个相机，测试 device_id=1
python tests/test_camera.py --backend webcam --webcam-id 1
```

#### 成功标志
```
============================================================
相机测试
============================================================
相机后端: webcam

【测试 1/3】基本导入测试
✓ cv2 模块导入成功
✓ WebCamera 类导入成功

【测试 2/3】USB 相机连接测试 (ID=0)
✓ USB 相机创建成功
✓ 相机启动成功
✓ 图像获取成功
  - 图像形状: (128, 128, 3)
  - 数据类型: uint8
  - 取值范围: [0, 255]
✓ 相机关闭成功

【测试总结】
基本导入: ✓
USB相机(0): ✓

------------------------------------------------------------
测试通过: 2/2
```

#### 常见问题

**问题 1: 设备未找到**
```
✗ USB 相机连接失败: Cannot open camera 0
```

**解决方法：**
1. 检查设备是否存在：
   ```bash
   ls /dev/video*
   ```

2. 检查权限：
   ```bash
   # 查看设备权限
   ls -l /dev/video0

   # 如果需要，添加用户到 video 组
   sudo usermod -a -G video $USER

   # 注销并重新登录后生效
   ```

3. 测试相机是否被其他程序占用：
   ```bash
   # 使用 cheese 测试
   sudo apt install cheese
   cheese
   ```

**问题 2: 图像全黑**
```
✓ 图像获取成功
  - 图像形状: (128, 128, 3)
  - 取值范围: [0, 0]  # 全黑
```

**解决方法：**
1. 检查相机镜头盖是否打开
2. 增加曝光时间（可能需要等待几帧）
3. 尝试其他 device_id

---

### 4. 测试环境创建

#### 运行测试
```bash
python tests/test_environment.py
```

#### 成功标志
```
============================================================
Kinova 环境测试
============================================================

【测试 1/4】基本导入测试
✓ gymnasium 模块导入成功
✓ KinovaEnv 类导入成功

【测试 2/4】配置加载测试
✓ 从 YAML 加载配置成功

【测试 3/4】环境创建测试
✓ 环境创建成功
✓ 观测空间定义正确
✓ 动作空间定义正确

【测试 4/4】虚拟模式测试
✓ 虚拟相机正常工作

【测试总结】
基本导入: ✓
配置加载: ✓
环境创建: ✓
虚拟模式: ✓

------------------------------------------------------------
测试通过: 4/4
```

---

## 完整系统测试

所有单独测试通过后，进行集成测试：

### 测试完整工作流

```bash
# 1. 启动 kortex_bringup（终端 1）
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.8.10

# 2. 测试环境（终端 2）
python -c "
from kinova_rl_env import KinovaEnv
import time

print('创建环境...')
env = KinovaEnv(config_path='kinova_rl_env/config/kinova_config.yaml')

print('重置环境...')
obs, info = env.reset()
print(f'观测形状: state={obs[\"state\"][\"tcp_pose\"].shape}, image={obs[\"images\"][\"wrist_1\"].shape}')

print('运行 10 步...')
for i in range(10):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step {i}: reward={reward:.3f}')

print('测试完成，关闭环境...')
env.close()
print('✓ 完整工作流测试通过！')
"
```

### 测试 VisionPro 遥操作

```bash
# 确保 VisionPro 已启动 VisionProControl app

python -c "
from vision_pro_control.core import VisionProBridge
import time

print('连接 VisionPro...')
bridge = VisionProBridge(avp_ip='192.168.1.125')
bridge.start()

print('请移动你的右手...')
for i in range(50):
    data = bridge.get_latest_data()
    if data['timestamp'] > 0:
        pos = data['wrist_pose'][:3, 3]
        print(f'手腕位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')
    time.sleep(0.1)

bridge.stop()
print('✓ VisionPro 遥操作测试通过！')
"
```

---

## 测试检查清单

完成以下所有项目后，即可进入数据收集阶段：

- [ ] **VisionPro 测试**
  - [ ] 基本导入成功
  - [ ] 连接成功并接收数据
  - [ ] 手部位置实时更新

- [ ] **Kinova 机械臂测试**
  - [ ] ROS2 初始化成功
  - [ ] kortex_bringup 正常运行
  - [ ] 能够读取关节状态
  - [ ] 能够读取 TCP 位姿

- [ ] **USB 相机测试**
  - [ ] 设备被识别 (/dev/video0)
  - [ ] 能够获取图像
  - [ ] 图像尺寸正确 (128x128x3)
  - [ ] 图像不是全黑

- [ ] **环境测试**
  - [ ] 配置加载成功
  - [ ] 环境创建成功
  - [ ] 观测空间正确
  - [ ] 动作空间正确

- [ ] **集成测试**
  - [ ] 环境 reset() 成功
  - [ ] 环境 step() 成功
  - [ ] VisionPro 数据实时更新
  - [ ] 相机图像正常

---

## 下一步

所有测试通过后，你可以开始：

1. **收集演示数据**：
   ```bash
   python hil_serl_kinova/record_teleop_demos.py --num-demos 10
   ```

2. **训练 BC 模型**：
   ```bash
   python hil_serl_kinova/train_bc_kinova.py --demos-dir ./demos/reaching
   ```

3. **评估模型**：
   ```bash
   python hil_serl_kinova/deploy_policy.py --checkpoint ./checkpoints/bc_latest.pt
   ```

详细步骤请参考：`docs/NEXT_STEPS.md`

---

## 故障排除总结

### VisionPro 连接失败
1. 检查网络连接和 IP 地址
2. 重启 VisionProControl app
3. 检查防火墙设置

### Kinova 连接失败
1. 确认 kortex_bringup 已启动
2. 检查 ROS2 话题是否发布
3. 验证机械臂 IP 地址

### 相机无法打开
1. 检查 /dev/video* 设备
2. 检查用户权限（video 组）
3. 确认相机未被其他程序占用

### 环境创建失败
1. 检查配置文件路径
2. 验证所有依赖已安装
3. 查看详细错误信息

---

## 技术支持

如果遇到无法解决的问题：

1. 查看日志输出，记录错误信息
2. 检查 `docs/` 目录中的其他文档
3. 参考测试脚本源代码：`tests/test_*.py`
4. 使用虚拟模式测试代码逻辑（跳过硬件要求）
