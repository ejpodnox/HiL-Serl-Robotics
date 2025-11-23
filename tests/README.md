# 测试套件

完整的独立测试套件，每个测试都可以单独运行，失败不影响其他测试。

## 测试列表

### 1. VisionPro 连接测试
```bash
python tests/test_visionpro_connection.py --vp_ip 192.168.1.125

# 跳过真实连接，只测试虚拟模式
python tests/test_visionpro_connection.py --skip-connection
```

**测试内容**:
- 模块导入
- VisionPro 数据接收
- 虚拟数据模式

### 2. Kinova 机械臂连接测试
```bash
python tests/test_kinova_connection.py --robot_ip 192.168.8.10

# 跳过真实连接
python tests/test_kinova_connection.py --skip-connection
```

**测试内容**:
- 模块导入
- ROS2 环境检查
- RobotCommander 连接
- 虚拟机械臂模式

### 3. 相机模块测试
```bash
# 测试虚拟相机
python tests/test_camera.py --backend dummy

# 测试 WebCam
python tests/test_camera.py --backend webcam --webcam-id 0

# 测试 RealSense
python tests/test_camera.py --backend realsense

# 测试所有相机
python tests/test_camera.py --backend all
```

**测试内容**:
- 相机接口导入
- DummyCamera（虚拟相机）
- WebCamera（USB 摄像头）
- RealSenseCamera（深度相机）

### 4. Gym 环境测试
```bash
python tests/test_environment.py --config kinova_rl_env/config/kinova_config.yaml
```

**测试内容**:
- 环境模块导入
- 配置加载
- 动作空间定义
- 观测空间定义

### 5. 数据流程测试
```bash
python tests/test_data_pipeline.py
```

**测试内容**:
- HIL-SERL 数据格式
- 数据保存/加载
- PyTorch Dataset 创建
- DataLoader 功能

### 6. 训练流程测试
```bash
python tests/test_training.py --steps 10
```

**测试内容**:
- BC 网络架构
- BC 训练循环
- Reward Classifier
- SAC 网络（Actor & Critic）
- 模型保存/加载

## 运行所有测试

### 完整测试（包括硬件）
```bash
python tests/run_all_tests.py
```

### 跳过硬件测试（推荐无硬件环境）
```bash
python tests/run_all_tests.py --skip-hardware
```

### 只运行特定测试
```bash
# 只测试 VisionPro
python tests/run_all_tests.py --test visionpro

# 只测试训练流程
python tests/run_all_tests.py --test training

# 可选: visionpro, kinova, camera, env, data, training, all
```

### 显示详细输出
```bash
python tests/run_all_tests.py --verbose
```

## 测试设计原则

### 1. 解耦设计
每个测试都是独立的，不依赖其他测试的结果。

### 2. 优雅降级
- 硬件不可用时，自动切换到虚拟模式
- 可选步骤失败不影响必需步骤
- 明确区分失败和跳过

### 3. 清晰反馈
- 每个测试步骤都有明确的输出
- 使用颜色标记（✓ ✗ ⚠️  ⊘）
- 详细的错误信息和堆栈跟踪

### 4. 快速运行
- 使用虚拟数据代替真实硬件
- 小型网络和少量训练步数
- 临时文件自动清理

## 测试矩阵

| 测试 | 需要硬件 | 可跳过 | 虚拟模式 |
|------|----------|--------|----------|
| VisionPro 连接 | VisionPro | ✓ | ✓ |
| Kinova 连接 | Kinova + ROS2 | ✓ | ✓ |
| 相机模块 | 相机 | ✓ | ✓ |
| Gym 环境 | ✗ | ✗ | ✓ |
| 数据流程 | ✗ | ✗ | ✓ |
| 训练流程 | ✗ | ✗ | ✓ |

## 故障排除

### 测试失败：导入错误
```bash
# 确保已安装包
pip install -e .
```

### 测试失败：ROS2 环境
```bash
# Source ROS2
source /opt/ros/humble/setup.bash
```

### 测试失败：CUDA 不可用
这是正常的，测试会自动使用 CPU。

### 某个测试卡住
使用 `Ctrl+C` 中断，其他测试会继续运行。

## 示例输出

```
============================================================
运行测试: VisionPro 连接
============================================================

============================================================
【测试 1】VisionPro 基础连接
============================================================
✓ 导入成功

⚠️  跳过真实连接测试

============================================================
【测试 3】VisionPro 虚拟数据模式
============================================================
✓ 虚拟数据生成成功:
  - 头部位姿: [0. 0. 0.]
  - 手腕位姿: [0. 0. 0.]
  - 捏合距离: 1.000

============================================================
【测试总结】
============================================================
basic               : ✓ 通过
connection          : ⊘ 跳过
dummy               : ✓ 通过

✓ 基础功能测试通过，VisionPro 模块可用
⚠️  真实设备连接失败，但可以使用虚拟模式进行开发
```

## CI/CD 集成

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -e .
      - name: Run tests
        run: python tests/run_all_tests.py --skip-hardware
```

## 下一步

- [快速开始](../docs/QUICKSTART.md) - 实际使用指南
- [API 文档](../docs/API.md) - 详细 API 参考
