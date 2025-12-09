# 遥操作安全指南 - 防止"爆红"错误

## 🔴 问题：遥操作启动就报错/机械臂突然运动

### 核心原因分析

1. **坐标单位**：VisionPro返回的是**米（m）**，不是厘米
   - 手部微小抖动可能产生较大速度指令
   - 需要设置**死区**来过滤小幅度运动

2. **速度过大**：映射增益太高导致超出安全限制
   - 默认配置可能对新手不够保守
   - 需要降低 `position_gain` 和 `max_joint_velocity`

3. **没有滤波**：VisionPro数据有噪声
   - 手部轻微抖动被直接传递给机械臂
   - 需要添加低通滤波平滑指令

4. **启动瞬间**：初始姿态可能距离中心较远
   - 第一帧数据可能产生大幅度运动
   - 需要启动保护机制

---

## ✅ 解决方案（按优先级）

### 方案1：使用安全配置文件（推荐）⭐

```bash
# 使用安全配置运行遥操作
python vision_pro_control/record_teleop_demos.py \
    --auto-calibrate \
    --config vision_pro_control/config/teleop_config_safe.yaml
```

**安全配置特点**：
- ✅ 死区扩大到 **10cm**（`deadzone_radius: 0.10`）
- ✅ 位置增益降低到 **0.15**（原来0.3）
- ✅ 最大速度降低到 **3mm/s**（非常保守）
- ✅ 最大关节速度 **0.1 rad/s**（原来0.2）
- ✅ 滤波系数 **0.1**（极度平滑）

---

### 方案2：手动调整参数

#### 2.1 修改死区（10cm）

编辑标定时的死区设置（已自动修改）：
```python
# vision_pro_control/record_teleop_demos.py:134
deadzone_radius=0.10  # 10cm死区
```

#### 2.2 降低速度增益

编辑 `vision_pro_control/config/teleop_config.yaml`:
```yaml
mapper:
  position_gain: 0.15          # 从0.3降到0.15
  rotation_gain: 0.15
  max_linear_velocity: 0.005   # 从0.01降到0.005

control:
  max_joint_velocity: 0.1      # 从0.2降到0.1

safety:
  max_linear_velocity: 0.003   # 从0.005降到0.003
```

#### 2.3 增加滤波强度

```yaml
mapper:
  filter_alpha: 0.1   # 从0.2降到0.1（更平滑）
```

---

### 方案3：渐进式测试流程

**不要一开始就运行遥操作！**按以下步骤测试：

#### 步骤1：测试VisionPro连接
```bash
# 测试VisionPro数据是否正常
python scripts/teleop/test_visionpro_bridge.py
```

观察手部坐标范围，例如：
```
Hand position: [0.35, 0.02, -0.05]
```

#### 步骤2：运行标定并观察
```bash
python vision_pro_control/record_teleop_demos.py --auto-calibrate
```

标定时观察：
- 按 `p` 查看当前位置
- 确保中心位置合理
- 多采样几次（5-10次）求平均

#### 步骤3：干运行测试（不发送命令）

修改代码添加调试模式：
```python
# record_teleop_demos.py 中的循环
if step % 10 == 0:  # 每0.5秒打印一次
    print(f"Twist: linear=[{vx:.4f},{vy:.4f},{vz:.4f}]")
    print(f"       angular=[{wx:.4f},{wy:.4f},{wz:.4f}]")
    print(f"Joint vel (max): {np.max(np.abs(joint_velocities)):.4f} rad/s")
```

**检查点**：
- 线速度应该 < 0.01 m/s
- 角速度应该 < 0.05 rad/s
- 关节速度应该 < 0.15 rad/s

#### 步骤4：实际运行

确认上述数值合理后，再启动机械臂驱动：
```bash
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.8.10 gripper:=robotiq_2f_85
```

---

## 🛡️ 安全检查清单

运行遥操作前确保：

- [ ] **机械臂在安全姿态**（远离奇异点和极限位置）
- [ ] **周围没有障碍物**
- [ ] **急停按钮触手可及**
- [ ] **使用安全配置文件**（`teleop_config_safe.yaml`）
- [ ] **已完成标定**（--auto-calibrate）
- [ ] **死区设置为10cm**
- [ ] **手放在中心附近再启动**（避免初始大幅度运动）
- [ ] **准备随时按 Ctrl+C**

---

## 📊 参数调优指南

### 参数对照表

| 参数 | 作用 | 保守值 | 标准值 | 激进值 | 单位 |
|------|------|--------|--------|--------|------|
| `deadzone_radius` | 死区大小 | **0.10** | 0.05 | 0.03 | m |
| `position_gain` | 位置增益 | **0.15** | 0.30 | 0.50 | - |
| `max_linear_velocity` | 最大线速度 | **0.003** | 0.01 | 0.05 | m/s |
| `max_joint_velocity` | 最大关节速度 | **0.1** | 0.2 | 0.5 | rad/s |
| `filter_alpha` | 滤波系数 | **0.1** | 0.2 | 0.4 | - |

**建议**：
- 初学者：使用**保守值**
- 熟练后：逐步调到标准值
- 专家：根据任务需求使用激进值

### 调参流程

1. **从保守开始**：用 `teleop_config_safe.yaml`
2. **观察响应**：手部运动→机械臂运动的延迟和幅度
3. **逐步调整**：
   - 太慢？增加 `position_gain` (+0.05)
   - 太抖？降低 `filter_alpha` (-0.05)
   - 死区太大？降低 `deadzone_radius` (-0.02)
4. **记录配置**：保存有效的参数组合

---

## 🔧 常见问题排查

### Q1: 手在中心附近，机械臂还在动

**原因**：死区太小或标定中心不准

**解决**：
```bash
# 重新标定，多采样几次
python vision_pro_control/record_teleop_demos.py --auto-calibrate

# 或增加死区
deadzone_radius: 0.15  # 15cm
```

### Q2: 手移动很大，机械臂几乎不动

**原因**：增益太小或速度限制太严格

**解决**：增加增益
```yaml
position_gain: 0.25  # 从0.15增加到0.25
max_linear_velocity: 0.008
```

### Q3: 机械臂抖动严重

**原因**：滤波不够或VisionPro数据质量差

**解决**：
```yaml
filter_alpha: 0.05  # 极度平滑（但响应会变慢）
```

**高级方案**：添加卡尔曼滤波（见下一节）

### Q4: 启动瞬间爆红

**原因**：初始姿态距离中心太远

**解决方案A**：标定后保持手部在中心附近
```bash
# 标定完成后，提示信息：
print("请将手放在中心位置，按Enter继续...")
input()
```

**解决方案B**：添加启动保护（已在safe配置中）
```yaml
safety:
  startup_duration: 5.0        # 前5秒使用低速
  startup_velocity_scale: 0.3  # 速度缩放到30%
```

---

## 🚀 需要卡尔曼滤波吗？

### 简单答案：**暂时不需要**

当前使用的**指数移动平均（EMA）滤波**已经足够：
- ✅ 简单高效
- ✅ 无需调参（只有一个alpha）
- ✅ 计算量小
- ✅ 延迟低

### 何时需要卡尔曼滤波？

如果满足以下情况，可以考虑：
1. EMA滤波后仍然抖动严重
2. VisionPro数据质量很差
3. 需要预测性滤波（补偿延迟）
4. 需要融合多个传感器数据

### 卡尔曼滤波实现（如果需要）

```python
# 在 coordinate_mapper.py 中添加

from filterpy.kalman import KalmanFilter

class CoordinateMapper:
    def __init__(self, ...):
        # ... 现有代码 ...

        # 初始化卡尔曼滤波器（6维：位置+速度）
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # 状态转移矩阵（假设恒速模型）
        dt = 0.05  # 20Hz
        self.kf.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        # 观测矩阵（只观测位置）
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # 过程噪声和观测噪声（需要根据实际调参）
        self.kf.Q *= 0.01  # 过程噪声
        self.kf.R *= 0.1   # 观测噪声

    def position_to_linear_velocity(self, position):
        # 预测
        self.kf.predict()

        # 更新
        self.kf.update(position)

        # 使用滤波后的位置
        filtered_position = self.kf.x[:3]
        filtered_velocity = self.kf.x[3:]  # 卡尔曼还能估计速度！

        # ... 后续处理 ...
```

**但建议**：先把 `filter_alpha` 调到 **0.05~0.1**，看看效果再决定。

---

## 📝 使用总结

### 推荐配置（新手友好）

```bash
# 1. 使用安全配置
python vision_pro_control/record_teleop_demos.py \
    --auto-calibrate \
    --config vision_pro_control/config/teleop_config_safe.yaml

# 2. 标定时多采样
#    - 按 's' 采样 5-10 次
#    - 按 'c' 保存

# 3. 标定后将手放在中心附近再开始
```

### 快速参数参考

**极度保守**（绝对安全，响应慢）：
```yaml
deadzone_radius: 0.12
position_gain: 0.10
max_linear_velocity: 0.002
filter_alpha: 0.05
```

**平衡**（推荐）：
```yaml
deadzone_radius: 0.10
position_gain: 0.15
max_linear_velocity: 0.005
filter_alpha: 0.1
```

**灵敏**（需要熟练操作）：
```yaml
deadzone_radius: 0.05
position_gain: 0.30
max_linear_velocity: 0.01
filter_alpha: 0.2
```

---

## ✅ 验证成功的标志

正常运行时应该看到：
```
✓ 遥操作记录器初始化完成
  控制频率: 20 Hz (dt=0.050s)
  最大关节速度: 0.1 rad/s

>>> 开始记录，按 'q' 停止 <<<
  [   0] 0.0s | 手[0.35,0.02,-0.05] | 速度[0.000,0.000,0.000]
  [  20] 1.0s | 手[0.36,0.03,-0.04] | 速度[0.001,0.001,0.001]
  [  40] 2.0s | 手[0.38,0.04,-0.03] | 速度[0.003,0.002,0.002]
```

**关键指标**：
- ✅ 速度 < 0.01 m/s
- ✅ 没有突然跳变
- ✅ 平滑增长
- ✅ 无报错信息

祝您使用顺利！🎉
