# 故障排除指南

本文档提供常见问题的诊断和解决方案。

## 目录
1. [遥操作程序报错 "Joint fault"](#遥操作程序报错-joint-fault)
2. [夹爪不动作](#夹爪不动作)
3. [/joint_states 无数据](#joint_states-无数据)
4. [诊断工具使用](#诊断工具使用)

---

## 遥操作程序报错 "Joint fault"

### 症状
运行遥操作程序时立即报错：
```
Error safety raised Joint fault
```

### 可能原因
1. `/joint_states` 话题无数据（机器人控制器未运行）
2. VisionPro标定数据错误或过期
3. 机器人未上电或未连接

### 解决方案

#### 步骤1：检查ROS2系统状态
```bash
# 运行诊断工具
python tools/diagnose_ros2.py
```

该工具会检查：
- ROS2节点和话题
- `/joint_states` 是否有数据
- 夹爪话题状态
- 控制器状态

#### 步骤2：确保机器人控制器运行

如果诊断显示 `/joint_states` 无数据，需要启动Kinova驱动：

```bash
# 启动 Gen3 机器人驱动（根据您的机器人型号调整）
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.8.10 gripper:=robotiq_2f_85

# 或者使用您项目特定的启动文件
# ros2 launch your_package your_launch_file.launch.py
```

启动后，再次运行诊断工具确认 `/joint_states` 有数据。

#### 步骤3：使用自动重新标定

即使控制器正常运行，过期的VisionPro标定也可能导致错误。使用自动重新标定功能：

```bash
# 使用 --auto-calibrate 参数
python vision_pro_control/record_teleop_demos.py --auto-calibrate
```

标定流程：
1. 程序启动后进入标定模式
2. 将手移动到舒适的操作中心位置
3. 按 `s` 键采样（建议3-5次）
4. 按 `c` 键完成标定
5. 自动进入遥操作模式

---

## 夹爪不动作

### 症状
发送夹爪命令但夹爪不移动

### 可能原因
1. 夹爪Action服务器未运行
2. 夹爪话题配置错误
3. 夹爪未正确连接

### 解决方案

#### 步骤1：测试夹爪通信
```bash
# 运行夹爪测试程序
python tools/test_gripper.py --mode all

# 或者只测试通信
python tools/test_gripper.py --mode communication

# 交互式测试（手动控制）
python tools/test_gripper.py --mode interactive
```

#### 步骤2：检查夹爪话题

```bash
# 列出所有夹爪相关话题
ros2 topic list | grep gripper

# 查看夹爪Action服务器
ros2 action list | grep gripper
```

应该能看到：
```
/robotiq_gripper_controller/gripper_cmd
```

#### 步骤3：手动测试夹爪

```bash
# 使用ROS2命令行测试夹爪
# 打开夹爪 (position=0.0)
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command:{position: 0.0, max_effort: 100.0}}"

# 闭合夹爪 (position=0.8)
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command:{position: 0.8, max_effort: 100.0}}"
```

如果手动测试成功但程序不工作，可能是代码问题。

#### 步骤4：检查夹爪配置

检查 `kinova_rl_env/kinova_env/kinova_interface.py` 中的夹爪话题：

```python
self.gripper_command_topic = '/robotiq_gripper_controller/gripper_cmd'
```

确保与您的机器人配置匹配。

---

## /joint_states 无数据

### 症状
```bash
ros2 topic echo /joint_states
# 无任何输出
```

### 可能原因
1. Kinova驱动未启动
2. 机器人未上电
3. 网络连接问题
4. ROS2控制器配置错误

### 解决方案

#### 步骤1：检查机器人连接

```bash
# Ping 机器人IP（根据配置文件调整）
ping 192.168.8.10

# 如果无法ping通，检查网络配置
ifconfig  # 查看本机IP
```

#### 步骤2：检查Kinova驱动进程

```bash
# 查看ROS2节点
ros2 node list

# 应该能看到 Kinova 相关的节点，例如：
# /gen3_base_controller
# /joint_state_broadcaster
```

#### 步骤3：重启Kinova驱动

```bash
# 停止现有驱动（如果有）
# Ctrl+C 或者 pkill -9 -f kortex

# 重新启动
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.8.10 gripper:=robotiq_2f_85
```

#### 步骤4：验证数据流

```bash
# 等待几秒后检查
ros2 topic echo /joint_states --once

# 应该能看到类似输出：
# header:
#   stamp:
#     sec: ...
# name: [joint_1, joint_2, ...]
# position: [...]
# velocity: [...]
```

---

## 诊断工具使用

### 工具1：ROS2系统诊断

**用途**：全面检查ROS2系统状态

```bash
python tools/diagnose_ros2.py
```

**输出内容**：
- ROS2节点列表
- 话题列表和类型
- 控制器状态
- `/joint_states` 数据流检查
- 夹爪话题检测

**使用时机**：
- 遥操作程序无法启动时
- 机器人无响应时
- 作为常规健康检查

---

### 工具2：夹爪测试程序

**用途**：测试夹爪通信和控制

```bash
# 完整测试
python tools/test_gripper.py

# 单项测试
python tools/test_gripper.py --mode communication    # 仅测试通信
python tools/test_gripper.py --mode open_close       # 测试开合
python tools/test_gripper.py --mode gradual          # 测试渐进运动
python tools/test_gripper.py --mode interactive      # 交互式控制
```

**测试模式说明**：

1. **communication** - 通信测试
   - 检查能否发送命令
   - 验证Action接口可用性

2. **open_close** - 开合测试
   - 循环测试夹爪开合
   - 可调整循环次数和延迟
   ```bash
   python tools/test_gripper.py --mode open_close --cycles 5 --delay 3.0
   ```

3. **gradual** - 渐进运动测试
   - 测试夹爪平滑移动
   - 可调整步数
   ```bash
   python tools/test_gripper.py --mode gradual --steps 20
   ```

4. **interactive** - 交互式测试
   - 手动控制夹爪
   - 按键说明：
     - `o` - 打开夹爪
     - `c` - 闭合夹爪
     - `h` - 半开夹爪
     - `0-9` - 设置位置（0=开，9=闭）
     - `q` - 退出

---

## 常见问题快速参考

| 问题 | 快速检查 | 解决方案 |
|------|---------|---------|
| 遥操作报 Joint fault | `python tools/diagnose_ros2.py` | 启动机器人驱动 + 重新标定 |
| 夹爪不动 | `python tools/test_gripper.py` | 检查Action服务器 |
| /joint_states 无数据 | `ros2 topic list` | 启动 kortex_bringup |
| VisionPro连接失败 | Ping VisionPro IP | 检查网络配置 |
| 标定文件不存在 | 查看配置文件路径 | 使用 --auto-calibrate |

---

## 控制频率检查

**所有组件的控制频率已统一为 20 Hz (dt=0.05s)**

配置文件位置：
- `kinova_rl_env/config/kinova_config.yaml` - frequency: 20
- `vision_pro_control/config/teleop_config.yaml` - frequency: 20

代码实现：
- `KinovaInterface.send_joint_velocities()` - 默认 dt=0.05
- `KinovaEnv` - 从配置读取
- `TeleopDataRecorder` - 从配置计算

**无需调整**，所有组件已同步。

---

## 获取帮助

如果以上方法都无法解决问题：

1. **收集诊断信息**：
   ```bash
   python tools/diagnose_ros2.py > diagnostics.log 2>&1
   ```

2. **检查日志**：
   查看ROS2和程序的详细日志

3. **提供信息**：
   - 诊断日志
   - 错误堆栈
   - 系统配置（机器人型号、ROS2版本等）

---

## 更新日志

- 2025-01-XX: 添加自动重新标定功能
- 2025-01-XX: 修复夹爪控制接口（改用Action）
- 2025-01-XX: 添加ROS2诊断工具
- 2025-01-XX: 添加夹爪测试工具
