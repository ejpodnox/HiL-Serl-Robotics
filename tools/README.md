# 工具目录

实用工具集，用于测试、调试和监控系统。

**⚠️ 重要：所有命令都需要在项目根目录 `kinova-hil-serl/` 下运行！**

```bash
# 进入项目根目录
cd ~/Documents/kinova-hil-serl  # 根据你的实际路径调整
```

---

## 工具列表

### 1. quick_verify.py - 快速验证工具

**用途：** 一键检查所有硬件是否正常连接

**使用场景：**
- 首次设置系统后验证配置
- 排查硬件连接问题
- CI/CD 集成测试

**基本用法：**
```bash
# 测试所有硬件
python tools/quick_verify.py

# 自定义 IP 地址
python tools/quick_verify.py --vp-ip 192.168.1.125 --robot-ip 192.168.8.10

# 跳过某些测试
python tools/quick_verify.py --skip-vp --skip-robot
```

**参数说明：**
- `--vp-ip`: VisionPro IP 地址（默认 192.168.1.125）
- `--robot-ip`: Kinova 机械臂 IP（默认 192.168.8.10）
- `--camera-id`: USB 相机 ID（默认 0）
- `--timeout`: 连接超时时间（默认 3.0s）
- `--skip-vp`: 跳过 VisionPro 验证
- `--skip-robot`: 跳过机械臂验证
- `--skip-camera`: 跳过相机验证

**输出示例：**
```
============================================================
【验证总结】
============================================================
VisionPro 连接        : ✓ 通过
Kinova 机械臂         : ✓ 通过
USB 相机             : ✓ 通过
环境配置             : ✓ 通过

总计: 4 | 通过: 4 | 失败: 0

🎉 所有验证通过！可以开始数据收集
```

---

### 2. live_monitor.py - 实时监控工具

**用途：** 实时显示系统各组件状态

**使用场景：**
- 调试硬件连接问题
- 演示系统运行状态
- 监控数据采集过程

**基本用法：**
```bash
# 监控所有组件（使用默认 IP）
python tools/live_monitor.py --all

# 只监控 VisionPro
python tools/live_monitor.py --vp-ip 192.168.1.125

# 监控机械臂和相机
python tools/live_monitor.py --robot-ip 192.168.8.10 --camera-id 0

# 运行 30 秒后自动停止
python tools/live_monitor.py --all --duration 30

# 设置更新频率为 5Hz
python tools/live_monitor.py --all --frequency 5
```

**参数说明：**
- `--vp-ip`: VisionPro IP 地址
- `--robot-ip`: Kinova 机械臂 IP
- `--camera-id`: USB 相机 ID
- `--duration`: 运行时长（秒），不填则无限运行
- `--frequency`: 更新频率（Hz，默认 10）
- `--all`: 监控所有组件（使用默认 IP）

**输出示例：**
```
运行时间:   12.3s

  VisionPro: 手腕=[ 0.234, -0.156,  0.987] 捏合=0.023
  Kinova:    TCP=[ 0.450,  0.023,  0.312] 姿态=[ 0.00,  1.00,  0.00,  0.00]
  相机:      图像=(128, 128, 3) 亮度= 127.3
```

按 `Ctrl+C` 停止监控。

---

## 推荐工作流

### 初次设置

1. **快速验证所有硬件：**
   ```bash
   python tools/quick_verify.py
   ```

2. **如果验证失败，查看详细指南：**
   ```bash
   # 参考文档
   cat docs/HARDWARE_TESTING_GUIDE.md
   ```

3. **使用实时监控调试：**
   ```bash
   python tools/live_monitor.py --all
   ```

### 日常使用

1. **每次启动系统前快速检查：**
   ```bash
   python tools/quick_verify.py
   ```

2. **数据采集时监控状态：**
   ```bash
   # 终端 1: 监控
   python tools/live_monitor.py --all

   # 终端 2: 数据采集
   python hil_serl_kinova/record_teleop_demos.py
   ```

---

## 故障排除

### quick_verify.py 报错

**问题：某个组件验证失败**

**解决方法：**
1. 使用 `--skip-xxx` 跳过失败组件，测试其他组件
2. 参考 `docs/HARDWARE_TESTING_GUIDE.md` 详细排查
3. 使用 `tests/test_*.py` 单独测试失败组件

**示例：**
```bash
# VisionPro 暂时不可用，跳过
python tools/quick_verify.py --skip-vp

# 只测试相机
python tests/test_camera.py --backend webcam --webcam-id 0
```

### live_monitor.py 无输出

**问题：监控无数据显示**

**可能原因：**
1. 硬件未正确连接
2. IP 地址错误
3. 相关服务未启动（如 kortex_bringup）

**解决方法：**
```bash
# 先运行快速验证
python tools/quick_verify.py

# 检查具体组件
python tests/test_visionpro_connection.py --vp-ip 192.168.1.125
python tests/test_kinova_connection.py --robot-ip 192.168.8.10
```

---

## 添加新工具

如果需要添加新的工具脚本：

1. 在 `tools/` 目录下创建 Python 文件
2. 添加 shebang: `#!/usr/bin/env python3`
3. 添加命令行参数解析（使用 argparse）
4. 在本 README 中添加使用说明
5. 添加执行权限：
   ```bash
   chmod +x tools/your_new_tool.py
   ```

**工具脚本模板：**
```python
#!/usr/bin/env python3
"""
工具描述

简短说明工具的用途和功能。
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='工具描述')
    parser.add_argument('--param', type=str, help='参数说明')
    args = parser.parse_args()

    # 工具逻辑
    print("工具运行中...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
```

---

## 相关文档

- **测试文档：** `tests/README.md` - 详细的测试套件说明
- **硬件测试：** `docs/HARDWARE_TESTING_GUIDE.md` - 硬件连接测试指南
- **快速测试：** `TESTING.md` - 快速测试指南
- **下一步：** `docs/NEXT_STEPS.md` - 系统完整工作流

---

## 技术细节

### quick_verify.py 实现原理

- 独立测试每个组件，失败不影响其他测试
- 超时机制防止卡死
- 返回退出码用于 CI/CD：
  - `0`: 所有测试通过
  - `1`: 部分测试失败
  - `2`: 多项测试失败

### live_monitor.py 实现原理

- 多组件并发监控
- 可配置更新频率
- 优雅退出处理（Ctrl+C）
- 错误不中断监控

---

## 未来计划

可能添加的工具：

- [ ] `calibrate_camera.py` - 相机标定工具
- [ ] `record_trajectory.py` - 轨迹记录和可视化
- [ ] `replay_demo.py` - 回放演示数据
- [ ] `benchmark_performance.py` - 性能基准测试
- [ ] `export_model.py` - 模型导出工具（ONNX/TorchScript）
