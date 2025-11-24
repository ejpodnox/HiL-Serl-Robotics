# 安装指南

完整的安装步骤，从零开始。

---

## 📋 方法一：一键安装（推荐）

**最简单的方式！**

```bash
# 1. 进入项目目录
cd ~/Documents/kinova-hil-serl

# 2. 运行一键安装脚本
bash install.sh
```

脚本会自动：
- ✅ 检查 Python 版本
- ✅ 安装所有 Python 依赖
- ✅ 安装项目（可编辑模式）
- ✅ 检查 ROS2 环境
- ✅ 检查 VisionProTeleop

---

## 📋 方法二：手动安装

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda（推荐）
conda create -n hilserl python=3.10
conda activate hilserl

# 或使用 venv
python3 -m venv ~/envs/hilserl
source ~/envs/hilserl/bin/activate
```

### 2. 安装 Python 依赖

```bash
cd ~/Documents/kinova-hil-serl

# 从 requirements.txt 安装（推荐）
pip install -r requirements.txt

# 安装项目（可编辑模式）
pip install -e .
```

### 3. 安装 ROS2 Humble（如果使用真实机器人）

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install ros-humble-desktop

# 添加到 ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 安装 Kinova 驱动
sudo apt install ros-humble-kortex*
```

### 4. VisionProTeleop（可选）

**如果暂时没有 VisionPro 硬件，可以跳过！**

系统会自动跳过 VisionPro 相关功能。

---

## ✅ 验证安装

```bash
cd ~/Documents/kinova-hil-serl

# 测试软件环境（不需要硬件）
python tools/quick_verify.py --skip-vp --skip-robot --skip-camera
```

**预期输出：**
```
环境配置: ✓ 通过
```

---

## 🚨 常见问题

### Q: `No module named 'VisionProTeleop'`

**A:** 如果没有 VisionPro 硬件，跳过测试：
```bash
python tools/quick_verify.py --skip-vp --skip-robot
```

### Q: torch 安装太慢

**A:** 使用清华镜像：
```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 下一步

安装完成后：
1. **快速开始**：`cat QUICKSTART.md`
2. **硬件测试**：`cat docs/HARDWARE_TESTING_GUIDE.md`
