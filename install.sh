#!/bin/bash
# Kinova HIL-SERL 一键安装脚本
# 使用方法：bash install.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "Kinova HIL-SERL 一键安装"
echo "========================================"

# 检查 Python 版本
echo ""
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 检查是否在虚拟环境中
if [[ -z "${VIRTUAL_ENV}" && -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo ""
    echo "⚠️  警告：你不在虚拟环境中！"
    echo "建议创建虚拟环境："
    echo "  conda create -n hilserl python=3.10"
    echo "  conda activate hilserl"
    echo ""
    read -p "是否继续安装？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "安装已取消"
        exit 1
    fi
fi

# 升级 pip
echo ""
echo "升级 pip..."
pip install --upgrade pip

# 安装基础依赖
echo ""
echo "安装基础依赖（从 requirements.txt）..."
pip install -r requirements.txt

# 安装项目（可编辑模式）
echo ""
echo "安装 kinova-hil-serl 包（可编辑模式）..."
pip install -e .

# 检查 ROS2
echo ""
echo "检查 ROS2 环境..."
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠️  未检测到 ROS2 环境"
    echo "如果需要使用真实机器人，请先安装 ROS2 Humble："
    echo "  https://docs.ros.org/en/humble/Installation.html"
else
    echo "✓ ROS2 $ROS_DISTRO 已安装"
fi

# VisionProTeleop 检查
echo ""
echo "检查 VisionProTeleop..."
if python3 -c "import VisionProTeleop" 2>/dev/null; then
    echo "✓ VisionProTeleop 已安装"
else
    echo "⚠️  VisionProTeleop 未安装"
    echo ""
    echo "VisionProTeleop 用于 Apple Vision Pro 遥操作"
    echo "如果你没有 VisionPro 硬件，可以跳过"
    echo ""
    echo "如果需要安装，请查看:"
    echo "  docs/INSTALLATION.md - VisionPro 安装指南"
fi

# 完成
echo ""
echo "========================================"
echo "✓ 安装完成！"
echo "========================================"
echo ""
echo "快速测试："
echo "  python tools/quick_verify.py --skip-vp --skip-robot --skip-camera"
echo ""
echo "完整文档："
echo "  cat QUICKSTART.md"
echo ""
