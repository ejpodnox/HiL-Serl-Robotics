# 快速开始 - 5分钟上手

**⚠️ 所有命令都在项目根目录 `kinova-hil-serl/` 下运行！**

---

## 第一步：进入项目目录

```bash
# 进入项目根目录
cd ~/Documents/kinova-hil-serl  # 根据你的实际路径调整

# 确认当前目录正确
pwd
# 应该输出：/home/xxx/Documents/kinova-hil-serl

# 查看目录结构
ls
# 应该看到：tools/  docs/  kinova_rl_env/  hil_serl_kinova/  等文件夹
```

---

## 第二步：测试硬件

```bash
# 一键测试所有硬件
python tools/quick_verify.py

# 如果某些硬件不可用，可以跳过
python tools/quick_verify.py --skip-vp        # 跳过 VisionPro
python tools/quick_verify.py --skip-robot     # 跳过机械臂
python tools/quick_verify.py --skip-camera    # 跳过相机
```

**预期结果：**
```
✓ VisionPro 连接
✓ Kinova 机械臂
✓ USB 相机
✓ 环境配置
```

---

## 第三步：选择任务

系统提供3个预定义任务：

| 任务 | 难度 | 说明 |
|------|------|------|
| reaching | ⭐ | 简单到达 |
| socket_insertion | ⭐⭐⭐ | 插插座 |
| flip_egg | ⭐⭐⭐⭐⭐ | 翻鸡蛋 |

**建议从 `socket_insertion` 开始！**

---

## 第四步：配置任务

```bash
# 编辑配置文件
vim kinova_rl_env/config/task_socket_insertion.yaml

# 修改这两个关键参数：
# socket_position: [0.45, 0.15, 0.25]  # 插座实际位置
# socket_orientation: [0.0, 0.707, 0.0, 0.707]  # 插座方向
```

**如何测量位置：**
```bash
# 启动监控
python tools/live_monitor.py --robot-ip 192.168.8.10

# 用 VisionPro 移动机械臂到插座位置
# 记录显示的 TCP 坐标
# 填入配置文件
```

---

## 第五步：采集演示

```bash
# 终端1：启动 ROS2
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.8.10

# 终端2：采集演示（至少20条）
python hil_serl_kinova/record_teleop_demos.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --num-demos 20
```

**操作要点：**
- 用 VisionPro 控制机械臂完成插入
- 动作要流畅
- 采集 20-30 条成功演示

---

## 第六步：训练模型

```bash
# 检查数据
python hil_serl_kinova/check_demos.py --demos-dir ./demos/socket_insertion

# 训练 BC 模型（约30分钟）
python hil_serl_kinova/train_bc_kinova.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --demos-dir ./demos/socket_insertion \
    --num-epochs 100
```

---

## 第七步：评估模型

```bash
# 部署到真实机器人
python hil_serl_kinova/deploy_policy.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --checkpoint ./checkpoints/socket_insertion_bc/best.pt \
    --num-episodes 5
```

---

## 常见问题

### Q1: 命令找不到文件？
**A:** 确保在项目根目录 `kinova-hil-serl/` 下运行！
```bash
cd ~/Documents/kinova-hil-serl
pwd  # 确认路径
```

### Q2: 硬件测试失败？
**A:** 查看详细指南：
```bash
cat docs/HARDWARE_TESTING_GUIDE.md
```

### Q3: 训练完整流程？
**A:** 查看完整文档：
```bash
cat docs/TRAINING_WORKFLOW.md
```

---

## 完整文档

- **硬件测试**: `docs/HARDWARE_TESTING_GUIDE.md`
- **训练流程**: `docs/TRAINING_WORKFLOW.md`
- **任务设计**: `docs/TASKS.md`
- **工具说明**: `tools/README.md`

---

## 下一步

完成第一个任务后：

1. **优化模型**：收集更多演示，调整参数
2. **RLPD训练**：如果 BC 成功率 < 80%
3. **新任务**：尝试 `flip_egg` 任务

---

**需要帮助？** 所有文档都在 `docs/` 目录下！
