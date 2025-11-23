# 安装指南

## 系统要求

- Ubuntu 20.04/22.04 (推荐)
- Python 3.8+
- ROS2 Humble
- CUDA 11.8+ (推荐，用于GPU加速)

## 安装步骤

### 1. 安装依赖

#### 系统依赖

```bash
# 更新软件源
sudo apt update

# ROS2 Humble（如果未安装）
sudo apt install ros-humble-desktop

# ROS2 开发工具
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep

# ROS2 必需包
sudo apt install ros-humble-tf2-ros
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-image-transport
sudo apt install ros-humble-realsense2-camera  # 可选，使用 RealSense 相机时
```

#### Python 依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/kinova-hil-serl.git
cd kinova-hil-serl

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装包（开发模式）
pip install -e .

# 或安装包 + 开发工具
pip install -e .[dev]
```

### 2. 配置 Kinova 机械臂

#### 安装 Kinova ROS2 驱动

Kinova ROS2 驱动已包含在 `ros2_kortex/` 目录中。

```bash
# 构建 Kinova ROS2 包
cd kinova-hil-serl
colcon build --packages-select kortex_bringup kortex_description

# Source 工作空间
source install/setup.bash
```

#### 测试 Kinova 连接

```bash
# 启动 Kinova 驱动（替换为你的机械臂IP）
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.8.10

# 在另一个终端测试连接
python scripts/teleop/test_robot_connection.py
```

### 3. 配置 VisionPro

#### VisionPro 应用端设置

1. 在 VisionPro 上安装数据流应用（参考 `VisionProTeleop/` 目录）
2. 确保 VisionPro 和工作站在同一网络
3. 记录 VisionPro 的 IP 地址

#### 测试 VisionPro 连接

```bash
# 修改配置文件中的 VisionPro IP
nano vision_pro_control/config/teleop_config.yaml

# 测试连接
python kinova_rl_env/tests/visionpro/test_visionpro_bridge.py --vp_ip 192.168.1.125
```

### 4. 相机配置（可选）

#### RealSense

```bash
# 安装 RealSense SDK
sudo apt install ros-humble-realsense2-camera

# 测试相机
realsense-viewer
```

#### WebCam

无需额外配置，系统会自动检测。

#### DummyCamera

用于无相机测试：

```yaml
# kinova_rl_env/config/kinova_config.yaml
camera:
  enabled: false  # 或使用 dummy backend
```

## 验证安装

运行完整测试套件：

```bash
# 单元测试
pytest kinova_rl_env/tests/unit/

# 硬件测试（需要连接硬件）
pytest kinova_rl_env/tests/hardware/

# 集成测试
pytest kinova_rl_env/tests/integration/
```

## 常见问题

### Q: ROS2 找不到 Kinova 包

A: 确保已 source 工作空间：
```bash
source install/setup.bash
```

### Q: PyTorch CUDA 不可用

A: 安装 CUDA 版本的 PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q: VisionPro 连接超时

A: 检查网络配置和防火墙设置：
```bash
# 测试连通性
ping <visionpro_ip>

# 检查端口（默认 50051）
nc -zv <visionpro_ip> 50051
```

## 下一步

安装完成后，参考以下文档继续：

- [快速开始](QUICKSTART.md) - 快速上手指南
- [API 文档](API.md) - 详细 API 参考
- [配置说明](CONFIGURATION.md) - 配置文件详解
