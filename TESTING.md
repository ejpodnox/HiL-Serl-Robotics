# 测试指南

快速测试指南，帮助你验证环境配置和系统功能。

## 🚀 快速开始

### 无硬件环境（推荐用于开发）

```bash
# 运行所有测试（虚拟模式）
python tests/run_all_tests.py --skip-hardware

# 预期结果：所有测试应该通过 ✓
```

### 有硬件环境

```bash
# 1. 确保硬件已启动
# - VisionPro 应用运行
# - Kinova 机械臂启动：ros2 launch kortex_bringup gen3.launch.py
# - 相机已连接

# 2. 运行完整测试
python tests/run_all_tests.py

# 3. 查看详细输出
python tests/run_all_tests.py --verbose
```

## 📋 测试清单

### 基础环境测试

```bash
# 1. 测试 Python 导入
python -c "from kinova_rl_env import KinovaEnv; print('✓ 导入成功')"

# 2. 测试 PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# 3. 测试 ROS2
ros2 topic list
```

### 模块测试

#### VisionPro 模块
```bash
# 虚拟模式（无需硬件）
python tests/test_visionpro_connection.py --skip-connection

# 真实连接（需要 VisionPro）
python tests/test_visionpro_connection.py --vp_ip 192.168.1.125
```

#### Kinova 机械臂
```bash
# 虚拟模式
python tests/test_kinova_connection.py --skip-connection

# 真实连接（需要启动 kortex_bringup）
python tests/test_kinova_connection.py --robot_ip 192.168.8.10
```

#### 相机模块
```bash
# 虚拟相机（总是可用）
python tests/test_camera.py --backend dummy

# WebCam
python tests/test_camera.py --backend webcam

# RealSense
python tests/test_camera.py --backend realsense
```

#### Gym 环境
```bash
# 测试环境定义（无需硬件）
python tests/test_environment.py
```

#### 数据流程
```bash
# 测试数据格式和 DataLoader（无需硬件）
python tests/test_data_pipeline.py
```

#### 训练流程
```bash
# 测试网络和训练循环（无需硬件）
python tests/test_training.py --steps 10
```

## 🎯 测试场景

### 场景 1: 新环境配置验证

刚安装完成，想验证环境是否正确：

```bash
# Step 1: 基础测试（无需硬件）
python tests/run_all_tests.py --skip-hardware

# Step 2: 如果通过，环境配置正确 ✓
# Step 3: 可以开始开发了！
```

### 场景 2: 硬件连接调试

硬件已连接，但不确定是否正常工作：

```bash
# Step 1: 单独测试 VisionPro
python tests/test_visionpro_connection.py --vp_ip 192.168.1.125 --timeout 10

# Step 2: 单独测试 Kinova
python tests/test_kinova_connection.py --robot_ip 192.168.8.10 --timeout 10

# Step 3: 测试相机
python tests/test_camera.py --backend realsense
```

### 场景 3: 训练前验证

准备开始训练，想确保数据和网络都正常：

```bash
# Step 1: 验证数据流程
python tests/test_data_pipeline.py

# Step 2: 验证训练流程
python tests/test_training.py --steps 50

# Step 3: 如果通过，可以开始真实训练 ✓
```

### 场景 4: CI/CD 集成

在 CI 环境中自动测试：

```bash
# GitHub Actions / Jenkins / GitLab CI
python tests/run_all_tests.py --skip-hardware --verbose
```

## 📊 测试输出解读

### 状态标记

- **✓ 通过**: 测试成功
- **✗ 失败**: 测试失败，需要检查错误信息
- **⚠️ 警告**: 非关键问题，可能影响部分功能
- **⊘ 跳过**: 测试被跳过（通常是硬件测试）

### 示例输出

#### 成功情况
```
============================================================
【测试总结】
============================================================
VisionPro 连接          : ✓ 通过
Kinova 连接            : ⊘ 跳过
相机模块               : ✓ 通过
Gym 环境              : ✓ 通过
数据流程               : ✓ 通过
训练流程               : ✓ 通过

总计: 6 | 通过: 5 | 失败: 0

🎉 所有测试通过！
```

#### 失败情况
```
============================================================
【测试 1】VisionPro 基础连接
============================================================
✗ 导入失败: No module named 'vision_pro_control'

提示: 请运行 pip install -e . 安装包
```

## 🔧 故障排除

### 问题 1: 导入错误

```bash
# 错误: ModuleNotFoundError: No module named 'kinova_rl_env'

# 解决:
pip install -e .
```

### 问题 2: ROS2 环境未初始化

```bash
# 错误: ROS2 环境检查失败

# 解决:
source /opt/ros/humble/setup.bash
source install/setup.bash  # 如果使用了 colcon build
```

### 问题 3: PyTorch CUDA 不可用

```
# 警告: CUDA is not available

# 这是正常的，测试会自动使用 CPU
# 如果需要 GPU，安装 CUDA 版本:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题 4: 测试卡住

```bash
# 如果某个测试卡住，按 Ctrl+C 中断
# 其他测试会继续运行

# 或设置更短的超时时间:
python tests/test_kinova_connection.py --timeout 5
```

## 📈 测试覆盖率

| 模块 | 测试内容 | 虚拟模式 | 硬件模式 |
|------|---------|----------|----------|
| VisionPro | 数据接收、坐标映射 | ✓ | ✓ |
| Kinova | 机械臂控制、状态读取 | ✓ | ✓ |
| Camera | 图像获取、多后端 | ✓ | ✓ |
| Environment | 空间定义、配置加载 | ✓ | - |
| Data | 格式、保存、加载 | ✓ | - |
| Training | 网络、优化器、训练 | ✓ | - |

## 💡 最佳实践

### 1. 开发时使用虚拟模式

```bash
# 快速迭代，无需等待硬件
python tests/run_all_tests.py --skip-hardware
```

### 2. 部署前测试硬件

```bash
# 验证硬件集成
python tests/run_all_tests.py
```

### 3. 定期运行测试

```bash
# 在 git commit 前运行
git add .
python tests/run_all_tests.py --skip-hardware
git commit -m "Your message"
```

### 4. 使用 verbose 模式调试

```bash
# 查看详细输出定位问题
python tests/test_training.py --verbose
```

## 🔗 相关文档

- [测试 README](tests/README.md) - 详细的测试文档
- [快速开始](docs/QUICKSTART.md) - 实际使用指南
- [API 文档](docs/API.md) - 编程接口参考

## 📞 获取帮助

如果测试失败且无法解决：

1. 检查错误信息和堆栈跟踪
2. 查看 [故障排除](#故障排除) 部分
3. 在 GitHub Issues 提问
4. 提供完整的错误输出（使用 `--verbose`）

---

**记住**: 即使硬件测试失败，只要虚拟模式测试通过，你就可以开始开发！🚀
