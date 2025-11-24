# 完整训练工作流程

从任务定义到模型部署的完整步骤指南。

**⚠️ 重要：所有命令都需要在项目根目录 `kinova-hil-serl/` 下运行！**

```bash
# 首先进入项目根目录
cd ~/Documents/kinova-hil-serl  # 根据你的实际路径调整

# 确认当前目录正确
pwd
# 应该输出类似：/home/xxx/Documents/kinova-hil-serl
```

---

## 📋 目录

1. [准备工作](#准备工作)
2. [场景设置](#场景设置)
3. [数据采集](#数据采集)
4. [训练 BC 模型](#训练-bc-模型)
5. [评估模型](#评估模型)
6. [在线训练 (RLPD)](#在线训练-rlpd)
7. [模型迭代](#模型迭代)
8. [故障排除](#故障排除)

---

## 准备工作

### 1. 硬件检查

在开始之前，确保所有硬件正常工作：

```bash
# 运行硬件快速验证
python tools/quick_verify.py

# 预期输出：所有组件都显示 ✓ 通过
```

如果有任何失败，参考 `docs/HARDWARE_TESTING_GUIDE.md`。

### 2. 选择任务

系统提供了3个预定义任务：

| 任务名称 | 难度 | 说明 | 配置文件 |
|---------|------|------|---------|
| `reaching` | ⭐ | 简单到达任务 | `kinova_config.yaml` |
| `socket_insertion` | ⭐⭐⭐ | 插2孔插座 | `task_socket_insertion.yaml` |
| `flip_egg` | ⭐⭐⭐⭐⭐ | 反转鸡蛋 | `task_flip_egg.yaml` |

**建议学习路径：**
1. 先用 `reaching` 熟悉整个流程
2. 再尝试 `socket_insertion`
3. 最后挑战 `flip_egg`

### 3. 配置任务参数

根据你的实际场景修改配置文件。

**示例：配置 socket_insertion 任务**

```bash
# 编辑配置文件
vim kinova_rl_env/config/task_socket_insertion.yaml
```

**关键参数（必须根据实际场景调整！）：**

```yaml
task:
  # !! 重要：测量你的插座实际位置 !!
  socket_position: [0.45, 0.15, 0.25]  # [x, y, z] 单位：米

  # !! 重要：测量插座朝向 !!
  socket_orientation: [0.0, 0.707, 0.0, 0.707]  # [qx, qy, qz, qw]

  # 插头初始位置
  plug_initial_position: [0.3, -0.2, 0.15]
```

**如何测量位置：**

```bash
# 启动监控工具
python tools/live_monitor.py --robot-ip 192.168.8.10

# 用VisionPro遥操作移动机械臂到插座位置
# 记录显示的 TCP 位置坐标
# 将坐标填入配置文件
```

---

## 场景设置

### Socket Insertion 任务场景

**需要准备：**
1. 2孔插座（固定在工作台上）
2. 插头（已安装在夹爪上，或放置在固定位置）
3. 相机视野包含插座和插头

**布置建议：**
```
  [机械臂]
     |
     |
  [插头] -----> [插座]
     ↓
  [相机1]      [相机2]
  (腕部)       (第三视角，可选)
```

**标定步骤：**

1. **测量插座位置：**
   ```bash
   # 用VisionPro遥操作移动到插座中心
   python tools/live_monitor.py --robot-ip 192.168.8.10

   # 记录 TCP 位置，例如：
   # TCP=[ 0.450,  0.150,  0.250]
   ```

2. **更新配置：**
   ```yaml
   task:
     socket_position: [0.450, 0.150, 0.250]
   ```

3. **测试配置：**
   ```bash
   # 创建测试脚本
   python -c "
   from kinova_rl_env import KinovaEnv

   env = KinovaEnv(config_path='kinova_rl_env/config/task_socket_insertion.yaml')
   obs, info = env.reset()

   print('插座目标位置:', info['socket_position'])
   print('当前 TCP 位置:', obs['state']['tcp_pose'][:3])

   env.close()
   "
   ```

### Flip Egg 任务场景

**需要准备：**
1. 鸡蛋（建议初期使用塑料鸡蛋或类似物体）
2. 防滑桌面（避免鸡蛋滚动）
3. 顶视图相机（强烈推荐，用于检测鸡蛋位置）

**布置建议：**
```
     [顶视图相机]
          ↓
  [机械臂]
     |
     |
  [鸡蛋] ←------ [防滑垫]
     ↓
  [腕部相机]
```

**场景标定：**

同样使用 `live_monitor.py` 测量鸡蛋位置，并更新 `task_flip_egg.yaml`。

---

## 数据采集

### 采集前检查

```bash
# 1. 硬件检查
python tools/quick_verify.py

# 2. 启动监控（可选，便于观察）
python tools/live_monitor.py --all
```

### 开始采集演示数据

**Socket Insertion 任务：**

```bash
# 启动kortex_bringup（终端1）
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.8.10

# 采集演示（终端2）
python hil_serl_kinova/record_teleop_demos.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --num-demos 20 \
    --output-dir ./demos/socket_insertion
```

**参数说明：**
- `--num-demos 20`: 采集20条演示
- `--output-dir`: 保存路径

**操作流程（每条演示）：**

1. **准备**：系统会自动重置环境
2. **开始演示**：用VisionPro控制机械臂完成任务
   - 移动到插头附近
   - 抓取插头
   - 接近插座
   - 对齐插座孔
   - 插入
3. **结束**：任务成功或手动停止（按 `Ctrl+C`）
4. **保存**：系统自动保存这条演示
5. **重复**：继续下一条演示

**采集技巧：**

✅ **好的演示：**
- 动作流畅、速度适中
- 成功完成任务
- 轨迹合理（不绕弯路）
- 包含不同起始位置和角度

❌ **避免：**
- 动作过快或过慢
- 中途失败
- 不必要的抖动
- 总是相同的轨迹

**最低采集量：**
- Reaching: 5-10条
- Socket Insertion: 20-30条
- Flip Egg: 30-50条

### 验证采集的数据

```bash
# 查看采集的演示
python -c "
import pickle
import glob

demo_files = glob.glob('./demos/socket_insertion/*.pkl')
print(f'共采集 {len(demo_files)} 条演示')

for f in demo_files[:3]:
    with open(f, 'rb') as file:
        demo = pickle.load(file)
    print(f'{f}: {len(demo[\"observations\"])} 步')
"
```

**预期输出：**
```
共采集 20 条演示
demo_0.pkl: 342 步
demo_1.pkl: 315 步
demo_2.pkl: 298 步
```

---

## 训练 BC 模型

Behavior Cloning (BC) 是第一阶段训练，从演示数据中学习。

### 1. 准备训练数据

```bash
# 检查数据完整性
python hil_serl_kinova/check_demos.py --demos-dir ./demos/socket_insertion

# 预期输出：
# ✓ 共 20 条演示
# ✓ 平均长度: 318 步
# ✓ 成功率: 100%
# ✓ 观测空间: state(22,), images(128,128,3)
# ✓ 动作空间: (7,)
```

### 2. 开始训练

```bash
# 训练 BC 模型
python hil_serl_kinova/train_bc_kinova.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --demos-dir ./demos/socket_insertion \
    --output-dir ./checkpoints/socket_insertion_bc \
    --num-epochs 100 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --val-split 0.1
```

**参数说明：**
- `--num-epochs 100`: 训练100个epoch
- `--batch-size 256`: 批大小256
- `--learning-rate 3e-4`: 学习率
- `--val-split 0.1`: 10%数据用于验证

**训练过程：**

```
Epoch [1/100]
  Train Loss: 0.1234  Val Loss: 0.1456

Epoch [10/100]
  Train Loss: 0.0543  Val Loss: 0.0621

Epoch [50/100]
  Train Loss: 0.0123  Val Loss: 0.0189
  ✓ 新的最佳模型！保存到 ./checkpoints/socket_insertion_bc/best.pt

Epoch [100/100]
  Train Loss: 0.0089  Val Loss: 0.0156

训练完成！
最佳验证损失: 0.0156 (Epoch 87)
```

**训练时间估计：**
- Reaching: ~5分钟
- Socket Insertion: ~15-30分钟
- Flip Egg: ~30-60分钟

（根据演示数量和硬件性能）

### 3. 监控训练

```bash
# 查看训练曲线（如果使用TensorBoard）
tensorboard --logdir ./checkpoints/socket_insertion_bc/logs
```

**好的训练曲线：**
- Train loss 持续下降
- Val loss 下降并趋于平稳
- Val loss 不应该远高于 train loss（过拟合）

---

## 评估模型

训练完成后，需要评估模型性能。

### 1. 离线评估（模拟）

```bash
# 在验证集上评估
python hil_serl_kinova/evaluate_policy.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --checkpoint ./checkpoints/socket_insertion_bc/best.pt \
    --demos-dir ./demos/socket_insertion \
    --num-episodes 10 \
    --mode offline
```

**输出示例：**
```
评估 10 个episode...

Episode 1: Reward=8.5, Success=True, Steps=235
Episode 2: Reward=9.2, Success=True, Steps=198
...
Episode 10: Reward=6.3, Success=False, Steps=500

========================================
【评估结果】
========================================
成功率: 8/10 (80%)
平均奖励: 7.8 ± 1.2
平均步数: 287 ± 98
========================================
```

### 2. 在线评估（真实机器人）

**⚠️ 重要：在线评估前的安全检查！**

- [ ] 清理工作空间，移除障碍物
- [ ] 确认机械臂活动范围安全
- [ ] 准备紧急停止按钮
- [ ] 以低速模式开始测试

```bash
# 启动 kortex_bringup（终端1）
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.8.10

# 部署策略（终端2）
python hil_serl_kinova/deploy_policy.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --checkpoint ./checkpoints/socket_insertion_bc/best.pt \
    --num-episodes 5 \
    --visualize  # 可视化相机图像和动作
```

**在线评估流程：**

1. **首次运行**：系统会加载模型并重置环境
2. **观察运行**：
   - 查看可视化窗口
   - 监控机械臂动作
   - 准备随时按紧急停止
3. **评估指标**：
   - 成功率
   - 动作流畅度
   - 是否有异常行为
4. **记录结果**：保存视频或日志

**评估标准：**

| 成功率 | 评价 | 建议 |
|-------|------|------|
| 0-20% | 差 | 需要更多演示数据或调整任务参数 |
| 20-50% | 一般 | 继续训练或进入RLPD阶段 |
| 50-80% | 良好 | 可以进入RLPD阶段优化 |
| 80-100% | 优秀 | BC已足够，可以直接部署 |

---

## 在线训练 (RLPD)

如果BC模型性能不够好（成功率<80%），使用RLPD进行在线强化学习。

### 什么是RLPD？

RLPD (Reinforcement Learning with Prior Data) 结合了：
- **演示数据**：人类专家的轨迹
- **在线数据**：策略自己探索收集的数据

### 1. 准备RLPD训练

**前提条件：**
- BC模型已训练（成功率至少20%）
- 机器人硬件正常
- 足够的训练时间（可能需要几小时到几天）

### 2. 启动RLPD训练

```bash
# 启动 kortex_bringup（终端1）
ros2 launch kortex_bringup gen3_lite.launch.py robot_ip:=192.168.8.10

# 启动 RLPD 训练（终端2）
python hil_serl_kinova/train_rlpd_kinova.py \
    --config kinova_rl_env/config/task_socket_insertion.yaml \
    --demos-dir ./demos/socket_insertion \
    --bc-checkpoint ./checkpoints/socket_insertion_bc/best.pt \
    --output-dir ./checkpoints/socket_insertion_rlpd \
    --num-steps 100000 \
    --demo-ratio 0.5 \
    --batch-size 256
```

**参数说明：**
- `--bc-checkpoint`: BC模型作为初始化
- `--num-steps 100000`: 训练10万步
- `--demo-ratio 0.5`: 50%演示数据 + 50%在线数据

### 3. RLPD训练过程

```
Step [1000/100000]
  Episode Reward: 3.2
  Success Rate (last 10): 30%
  Train Loss: 0.234
  Buffer Size: 5000

Step [10000/100000]
  Episode Reward: 6.8
  Success Rate (last 10): 60%
  Train Loss: 0.089
  Buffer Size: 50000
  ✓ 性能提升！保存模型

Step [50000/100000]
  Episode Reward: 8.5
  Success Rate (last 10): 85%
  Train Loss: 0.043
  Buffer Size: 100000

Step [100000/100000]
  Episode Reward: 9.1
  Success Rate (last 10): 90%
  Train Loss: 0.021

训练完成！
最佳成功率: 92% (Step 87000)
```

**训练时间估计：**
- Socket Insertion: 2-4小时
- Flip Egg: 6-12小时

### 4. 监控RLPD训练

**实时监控（终端3）：**
```bash
python tools/live_monitor.py --all
```

**TensorBoard：**
```bash
tensorboard --logdir ./checkpoints/socket_insertion_rlpd/logs
```

**关键指标：**
- Episode Reward（上升趋势）
- Success Rate（上升到80%+）
- Train Loss（下降并稳定）

### 5. 早停条件

如果满足以下条件之一，可以提前停止：

- ✅ 成功率连续100个episode > 90%
- ✅ 性能不再提升（plateau）超过20000步
- ❌ 训练不稳定（loss突然暴涨）
- ❌ 策略崩溃（成功率降到0%）

---

## 模型迭代

### 迭代流程

```
采集演示 → 训练BC → 评估 → (不满意？) → RLPD → 评估 → 部署
    ↑                                              |
    └──────────────(还需改进？)────────────────────┘
                收集更多演示或调整参数
```

### 何时收集更多演示？

**收集更多演示的情况：**
- BC成功率<20%
- RLPD训练不稳定
- 策略有明显的坏习惯
- 任务场景改变（如插座位置变化）

**不需要更多演示的情况：**
- BC成功率>50%
- RLPD能够持续提升
- 只是需要更多训练时间

### 参数调整策略

**如果成功率低：**

1. **检查任务定义**：
   ```python
   # 降低成功阈值
   success_threshold:
     insertion: 0.02  # 从0.01改为0.02
   ```

2. **增加奖励权重**：
   ```python
   reward:
     insertion_weight: 10.0  # 从5.0提高
   ```

3. **收集更高质量的演示**

**如果训练不稳定：**

1. **降低学习率**：
   ```bash
   --learning-rate 1e-4  # 从3e-4降低
   ```

2. **增加批大小**：
   ```bash
   --batch-size 512  # 从256增加
   ```

3. **增加demo比例**：
   ```bash
   --demo-ratio 0.7  # 从0.5提高
   ```

---

## 故障排除

### 问题1: 数据采集时机器人不动

**可能原因：**
- VisionPro连接断开
- kortex_bringup未启动
- ROS2话题异常

**解决方法：**
```bash
# 检查VisionPro连接
python tests/test_visionpro_connection.py --vp-ip 192.168.1.125

# 检查ROS2话题
ros2 topic list | grep joint

# 重启kortex_bringup
# Ctrl+C 停止，然后重新启动
```

### 问题2: 训练loss不下降

**可能原因：**
- 学习率太低
- 演示质量差
- 网络结构不合适

**解决方法：**
```bash
# 提高学习率
--learning-rate 1e-3

# 检查演示数据
python hil_serl_kinova/check_demos.py --demos-dir ./demos/xxx

# 增加网络深度（修改配置文件）
```

### 问题3: 模型在真实机器人上表现差

**可能原因：**
- 过拟合演示数据
- 演示数据多样性不足
- 视觉输入与训练时不同

**解决方法：**
1. 收集更多样化的演示
2. 使用数据增强
3. 进入RLPD阶段进行在线学习

### 问题4: RLPD训练中策略崩溃

**可能原因：**
- 探索过度
- 奖励函数设计问题
- 学习率太高

**解决方法：**
```bash
# 降低学习率
--learning-rate 1e-4

# 提高demo比例
--demo-ratio 0.7

# 减小探索噪声（修改配置）
```

---

## 检查清单

### 数据采集前

- [ ] 硬件测试通过（quick_verify.py）
- [ ] 场景已设置并测量位置
- [ ] 配置文件已更新
- [ ] VisionPro已校准
- [ ] 工作空间安全

### 训练前

- [ ] 演示数据已收集（达到最低数量）
- [ ] 数据完整性检查通过
- [ ] 配置文件正确
- [ ] 有足够的磁盘空间和计算资源

### 部署前

- [ ] 模型训练完成
- [ ] 离线评估通过
- [ ] 安全措施就位
- [ ] 工作空间清理
- [ ] 紧急停止按钮可用

---

## 下一步

完成第一个任务的完整训练后，你可以：

1. **优化当前任务**：
   - 收集更多演示
   - 调整奖励函数
   - 尝试不同的网络结构

2. **尝试新任务**：
   - 定义新的任务
   - 参考现有任务代码
   - 遵循相同的训练流程

3. **系统集成**：
   - 将模型集成到应用中
   - 添加任务调度器
   - 实现多任务切换

---

## 相关文档

- **任务设计**：`docs/TASKS.md` - 如何设计新任务
- **API参考**：`docs/API.md` - 编程接口
- **配置说明**：`docs/CONFIGURATION.md` - 配置参数详解
- **硬件测试**：`docs/HARDWARE_TESTING_GUIDE.md` - 硬件连接测试

---

**祝训练顺利！** 🚀
