# Kinova HIL-SERL 文档

欢迎来到 Kinova HIL-SERL 项目文档！

## 📚 文档目录

### 新手入门

- **[快速开始 (QUICKSTART.md)](QUICKSTART.md)** - 5分钟快速上手指南
- **[安装指南 (INSTALLATION.md)](INSTALLATION.md)** - 详细的安装步骤和依赖配置

### 使用指南

- **[配置说明 (CONFIGURATION.md)](CONFIGURATION.md)** - 所有配置文件的详细说明
- **[API 文档 (API.md)](API.md)** - 完整的 API 参考和示例代码
- **[硬件测试指南 (HARDWARE_TESTING_GUIDE.md)](HARDWARE_TESTING_GUIDE.md)** - 分步测试所有硬件连接
- **[USB 相机配置 (USB_CAMERA_SETUP.md)](USB_CAMERA_SETUP.md)** - USB 相机设置和使用
- **[任务设计指南 (TASKS.md)](TASKS.md)** - 如何设计和实现新任务
- **[下一步行动 (NEXT_STEPS.md)](NEXT_STEPS.md)** - 完成配置后的工作流程

### 技术文档

- **[实现总结 (IMPLEMENTATION_SUMMARY.md)](IMPLEMENTATION_SUMMARY.md)** - 系统架构和技术细节
- **[相机对比 (CAMERA_COMPARISON.md)](CAMERA_COMPARISON.md)** - RealSense vs USB 相机对比

### 其他资源

- **[主 README](../README.md)** - 项目概览和特性介绍

---

## 🚀 快速导航

### 我想...

#### 第一次使用这个项目
→ 阅读 [快速开始](QUICKSTART.md)

#### 安装和配置环境
→ 阅读 [安装指南](INSTALLATION.md)

#### 测试硬件连接
→ 阅读 [硬件测试指南](HARDWARE_TESTING_GUIDE.md)

#### 配置 USB 相机
→ 阅读 [USB 相机配置](USB_CAMERA_SETUP.md)

#### 修改配置参数
→ 阅读 [配置说明](CONFIGURATION.md)

#### 设计新任务
→ 阅读 [任务设计指南](TASKS.md)

#### 了解下一步工作
→ 阅读 [下一步行动](NEXT_STEPS.md)

#### 在代码中使用 API
→ 阅读 [API 文档](API.md)

#### 了解系统实现原理
→ 阅读 [实现总结](IMPLEMENTATION_SUMMARY.md)

#### 报告问题或贡献代码
→ 访问 [GitHub Issues](https://github.com/yourusername/kinova-hil-serl/issues)

---

## 📖 文档结构

```
docs/
├── README.md                    # 本文档（文档索引）
├── QUICKSTART.md               # 快速开始指南
├── INSTALLATION.md             # 安装指南
├── HARDWARE_TESTING_GUIDE.md   # 硬件测试指南
├── USB_CAMERA_SETUP.md         # USB 相机配置
├── CONFIGURATION.md            # 配置说明
├── TASKS.md                    # 任务设计指南
├── NEXT_STEPS.md               # 下一步行动
├── API.md                      # API 文档
├── IMPLEMENTATION_SUMMARY.md   # 实现总结
└── CAMERA_COMPARISON.md        # 相机对比
```

---

## 🎯 学习路径

### 路径 1: 快速实践（推荐新手）

1. [快速开始](QUICKSTART.md) - 运行第一个演示
2. [硬件测试指南](HARDWARE_TESTING_GUIDE.md) - 测试所有硬件连接
3. [下一步行动](NEXT_STEPS.md) - 开始数据收集和训练
4. [配置说明](CONFIGURATION.md) - 理解配置参数
5. [API 文档](API.md) - 学习编程接口

### 路径 2: 深入理解（推荐开发者）

1. [安装指南](INSTALLATION.md) - 理解系统依赖
2. [实现总结](IMPLEMENTATION_SUMMARY.md) - 了解架构设计
3. [任务设计指南](TASKS.md) - 学习如何设计新任务
4. [API 文档](API.md) - 掌握编程接口
5. [配置说明](CONFIGURATION.md) - 优化配置参数

### 路径 3: 硬件部署（推荐实际使用）

1. [安装指南](INSTALLATION.md) - 正确安装所有依赖
2. [USB 相机配置](USB_CAMERA_SETUP.md) - 配置 USB 相机
3. [硬件测试指南](HARDWARE_TESTING_GUIDE.md) - 验证所有硬件
4. [配置说明](CONFIGURATION.md) - 调整配置参数
5. [下一步行动](NEXT_STEPS.md) - 完整工作流程

---

## 💡 常见问题

### Q: 我应该从哪个文档开始？

**A**: 如果你是第一次使用，从 [快速开始](QUICKSTART.md) 开始。如果想深入了解，阅读 [实现总结](IMPLEMENTATION_SUMMARY.md)。

### Q: 如何找到某个功能的文档？

**A**:
- 配置相关 → [配置说明](CONFIGURATION.md)
- API 相关 → [API 文档](API.md)
- 安装问题 → [安装指南](INSTALLATION.md)

### Q: 文档有错误或过时怎么办？

**A**: 请在 [GitHub Issues](https://github.com/yourusername/kinova-hil-serl/issues) 提交问题。

---

## 🔧 示例代码

所有文档中的示例代码都可以直接运行。主要示例位于：

- **快速开始**: 完整流程示例
- **API 文档**: 各模块使用示例
- **配置说明**: 配置文件示例

---

## 📝 文档约定

### 代码块

```python
# Python 代码
from kinova_rl_env import KinovaEnv
env = KinovaEnv()
```

```bash
# Shell 命令
python train_bc_kinova.py --config config.py
```

```yaml
# YAML 配置
robot:
  ip: "192.168.8.10"
```

### 符号说明

- ✅ 已完成/推荐
- ⚠️  警告/注意
- 🚀 快速/高效
- 📊 数据/统计
- 🎯 目标/重点

---

## 🤝 贡献文档

欢迎帮助改进文档！

1. Fork 项目
2. 修改文档
3. 提交 Pull Request

文档规范：
- 使用 Markdown 格式
- 添加代码示例
- 保持结构清晰
- 中英文之间加空格

---

## 📞 获取帮助

- **GitHub Issues**: 报告问题和功能请求
- **Discussions**: 技术讨论和问答
- **Email**: youremail@example.com

---

**Happy Learning! 🎉**
