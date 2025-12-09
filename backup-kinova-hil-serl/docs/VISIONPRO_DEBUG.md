# VisionPro 数据全是0的问题诊断

## 🔍 快速诊断

**问题**：标定时手部位置采样一直是 `[0.000, 0.000, 0.000]`

### 步骤1：测试VisionPro原始数据

```bash
# 运行原始数据测试
python tools/test_visionpro_raw.py --ip 192.168.1.125
```

**期望看到**：
```
[1s] 原始数据检查:
  - head[0] 数据:
    [[1.0, 0.0, 0.0, x],
     [0.0, 1.0, 0.0, y],
     [0.0, 0.0, 1.0, z],
     [0.0, 0.0, 0.0, 1.0]]
  - right_wrist[0] 数据:
    [[...]]
  - right_pinch: 0.05
```

**如果全是0或空**：
- VisionPro 没有发送数据
- 网络连接问题
- VisionPro 应用未运行

---

## 🛠️ 可能的原因和解决方案

### 原因1：VisionPro应用未运行 ⭐ **最可能**

**检查**：
- VisionPro 上的 AVP Stream 应用是否已启动？
- 应用界面显示"Connected"还是"Waiting"？

**解决**：
1. 在 VisionPro 上打开 AVP Stream 应用
2. 确认显示"Streaming"状态
3. 再运行测试

---

### 原因2：网络连接问题

**检查**：
```bash
# Ping VisionPro
ping 192.168.1.125

# 检查是否能telnet到VisionPro端口
telnet 192.168.1.125 8080  # 或其他端口
```

**解决**：
- 确保VisionPro和电脑在同一网络
- 检查防火墙设置
- 确认VisionPro IP地址正确

---

### 原因3：avp_stream库版本问题

**检查**：
```bash
# 检查 avp_stream 是否正确安装
python -c "from avp_stream import VisionProStreamer; print('OK')"
```

**解决**：
```bash
# 重新安装 avp_stream
cd VisionProTeleop
pip install -e .
```

---

### 原因4：VisionProBridge没有正确启动

**检查代码**：

在 `debug_teleop.py` 或 `record_teleop_demos.py` 中，确保：
```python
# 1. 创建 bridge
self.vp_bridge = VisionProBridge(avp_ip=ip, use_right_hand=True)

# 2. 启动数据流 (重要！)
self.vp_bridge.start()

# 3. 等待数据填充
time.sleep(2.0)  # 至少等待1-2秒

# 4. 然后再读取数据
position, rotation = self.vp_bridge.get_hand_relative_to_head()
```

**修复**：在标定前确保调用了 `start()` 并等待足够时间。

---

### 原因5：数据格式变化

**检查**：VisionProStreamer返回的数据格式可能不同

运行测试查看实际数据结构：
```bash
python tools/test_visionpro_raw.py
```

根据输出调整 `VisionProBridge._update_loop()` 中的数据提取逻辑。

---

## ✅ 临时解决方案：模拟数据测试

如果暂时无法获取VisionPro数据，可以用模拟数据测试其他部分：

**创建模拟VisionPro**：
```python
class MockVisionProBridge:
    def __init__(self, *args, **kwargs):
        self.center = np.array([0.3, 0.0, -0.1])
        self.t = 0

    def start(self):
        pass

    def stop(self):
        pass

    def get_hand_relative_to_head(self):
        # 模拟手部运动：圆周运动
        self.t += 0.05
        x = self.center[0] + 0.05 * np.cos(self.t)
        y = self.center[1] + 0.05 * np.sin(self.t)
        z = self.center[2]

        position = np.array([x, y, z])
        rotation = np.eye(3)

        return position, rotation

    def get_pinch_distance(self):
        return 0.05  # 固定值
```

**使用方式**：
```python
# 在 debug_teleop.py 中临时替换
# from vision_pro_control.core.visionpro_bridge import VisionProBridge
# 改为：
from mock_visionpro import MockVisionProBridge as VisionProBridge
```

---

## 📋 完整诊断流程

1. **运行原始数据测试**
   ```bash
   python tools/test_visionpro_raw.py
   ```

2. **检查输出**
   - 如果看到实际数据（非零矩阵）→ VisionPro工作正常，问题在Bridge
   - 如果全是0或空 → VisionPro连接问题

3. **如果是连接问题**
   - 检查VisionPro应用是否运行
   - Ping VisionPro IP
   - 检查网络配置

4. **如果是Bridge问题**
   - 确认 `vp_bridge.start()` 已调用
   - 等待时间足够（2秒+）
   - 检查数据提取逻辑

5. **运行调试遥操作**
   ```bash
   python tools/debug_teleop.py
   ```
   观察标定时的详细输出

---

## 🎯 最可能的解决方案

**90%的情况是这个问题**：

```bash
# 在标定前没有等待足够时间
self.vp_bridge.start()
# ❌ 立即读取 → 数据还没来
position, rotation = self.vp_bridge.get_hand_relative_to_head()

# ✅ 正确做法
self.vp_bridge.start()
time.sleep(2.0)  # 等待数据流稳定
position, rotation = self.vp_bridge.get_hand_relative_to_head()
```

**检查这几个文件**：
1. `tools/debug_teleop.py` 的 `__init__` 方法
2. `vision_pro_control/record_teleop_demos.py` 的 `_run_calibration` 方法
3. 确保标定前调用了 `recorder.start()`

---

## 💡 快速修复

在标定函数开始处添加：

```python
def _run_calibration(self):
    """运行标定流程"""

    # 确保VisionPro已启动并等待数据
    if not hasattr(self, '_vp_started'):
        print("启动VisionPro数据流...")
        self.vp_bridge.start()
        self._vp_started = True

        # 等待数据稳定
        print("等待VisionPro数据... (2秒)")
        time.sleep(2.0)

        # 验证数据
        try:
            test_pos, _ = self.vp_bridge.get_hand_relative_to_head()
            print(f"VisionPro数据测试: {test_pos}")

            if np.allclose(test_pos, 0):
                print("⚠️  警告：VisionPro数据全是0，请检查连接！")
        except Exception as e:
            print(f"✗ VisionPro数据获取失败: {e}")

    # ... 原有的标定代码 ...
```

先运行 `python tools/test_visionpro_raw.py`，把输出发给我，我帮你定位问题！
