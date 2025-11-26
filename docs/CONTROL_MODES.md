# 机械臂控制模式说明

本项目支持两种控制机械臂的方式：

## 方式1: ROS2 控制器（RobotCommander）

使用 ROS2 的 twist_controller 来控制机械臂。

**优点：**
- 与ROS2生态集成
- 支持ROS2的工具和可视化
- 可以使用rviz等工具监控

**缺点：**
- 依赖ROS2控制器正确配置
- 可能存在控制器激活问题
- 需要kortex_bringup运行

**使用方法：**
```python
from vision_pro_control.core import RobotCommander

commander = RobotCommander(robot_ip='192.168.8.10')
commander.send_twist(np.array([vx, vy, vz, wx, wy, wz]))
```

**测试：**
```bash
python tools/test_robot_control.py
```

---

## 方式2: Kortex API（KortexCommander）

直接使用 Kinova 的 Kortex Python API，绕过 ROS2 控制器。

**⚠️ 注意：Kortex API 不在 PyPI 上，需要手动安装**

**优点：**
- ✅ 不依赖ROS2控制器
- ✅ 响应更快，延迟更低
- ✅ 更可靠（直接与机械臂通信）
- ✅ 仍然保持twist控制接口

**缺点：**
- ❌ **需要手动安装** kortex_api（不能 pip install）
- 不能使用ROS2的监控工具

**安装依赖：**
```bash
# Kortex API 需要从 GitHub 手动安装
# 参考: https://github.com/Kinovarobotics/kortex

# scipy 可以正常安装
pip install scipy
```

**由于安装复杂，建议先尝试激活 twist_controller（方式1）**

**使用方法：**
```python
from vision_pro_control.core import KortexCommander

commander = KortexCommander(robot_ip='192.168.8.10')
commander.send_twist(np.array([vx, vy, vz, wx, wy, wz]))
```

**测试：**
```bash
python tools/test_kortex_control.py
```

---

## 两种方式的接口一致性

两个 Commander 类提供**完全相同的接口**：

```python
# 发送速度命令（numpy数组或字典都支持）
commander.send_twist(twist)

# 停止
commander.send_zero_twist()

# 急停
commander.emergency_stop()

# 获取TCP位姿
pose = commander.get_tcp_pose()  # [x, y, z, qx, qy, qz, qw]

# 设置安全限制
commander.set_safety_limits(max_linear=0.1, max_angular=0.3)

# 获取状态
info = commander.get_info()
```

---

## 如何选择？

### 使用 KortexCommander（方式2），如果：
- ✅ ROS2控制器有问题或不工作
- ✅ 需要最低延迟
- ✅ 只需要控制机械臂，不需要ROS2生态

### 使用 RobotCommander（方式1），如果：
- 需要ROS2的可视化和监控工具
- 需要与其他ROS2节点集成
- ROS2控制器已经正确配置并工作

---

## HIL-SERL项目的推荐配置

对于 **Human-in-the-Loop SERL** 项目，推荐使用 **KortexCommander**：

1. **更可靠** - 不依赖ROS2控制器配置
2. **更快** - 降低遥操作延迟
3. **更简单** - 减少依赖和配置

只需修改初始化代码：

```python
# 旧方式
from vision_pro_control.core import RobotCommander
commander = RobotCommander(robot_ip='192.168.8.10')

# 新方式（推荐）
from vision_pro_control.core import KortexCommander
commander = KortexCommander(robot_ip='192.168.8.10')
```

其他代码**完全不需要修改**！

---

## 故障排查

### KortexCommander 连接失败

```bash
# 检查网络连通性
ping 192.168.8.10

# 检查Kortex API安装
python -c "import kortex_api; print('OK')"

# 检查机械臂Web界面
# 浏览器打开: http://192.168.8.10
```

### RobotCommander 不移动

```bash
# 检查控制器状态
python tools/check_controller_status.py

# 激活控制器
ros2 control set_controller_state twist_controller activate
```

---

## 性能对比

| 特性 | RobotCommander | KortexCommander |
|------|----------------|-----------------|
| 延迟 | ~10-50ms | ~5-10ms |
| 可靠性 | 中 | 高 |
| 配置复杂度 | 高 | 低 |
| ROS2集成 | ✓ | ✗ |
| 直接控制 | ✗ | ✓ |
| **推荐用于HIL-SERL** | | ✓ |
