# HIL-SERL 任务设计指南

## 当前实现的任务

### 1. Kinova Reaching（到达任务）

**位置**: `hil_serl_kinova/experiments/kinova_reaching/`

**描述**: 机械臂末端到达目标位置

**配置**:
```python
config.task_name = "kinova_reaching"
config.target_pose = [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]  # 目标位姿
config.success_threshold = 0.02  # 成功判定：距离 < 2cm
```

**奖励函数**:
- **Sparse**: 到达目标 +1，否则 0
- **Dense**: `-distance_to_target`（负的距离）

**观测空间**:
- 状态: 关节位置(7) + TCP位姿(7) = 14 维
- 图像: 1 个相机（wrist_1）128x128 RGB

**动作空间**:
- Delta pose: [dx, dy, dz, drx, dry, drz, gripper] (7 维)

---

## 如何添加新任务

### 示例：Pick and Place（抓取放置）

#### Step 1: 创建任务目录

```bash
mkdir -p hil_serl_kinova/experiments/kinova_pick_place
```

#### Step 2: 创建配置文件

`hil_serl_kinova/experiments/kinova_pick_place/config.py`:

```python
from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    # 任务信息
    config.task_name = "kinova_pick_place"
    config.description = "抓取物体并放置到目标位置"

    # 环境配置
    config.env_name = "KinovaEnv"
    config.robot_dof = 7

    # 任务特定参数
    config.object_position = [0.4, 0.0, 0.1]  # 物体位置
    config.grasp_pose = [0.4, 0.0, 0.15, 0.0, 1.0, 0.0, 0.0]  # 抓取位姿
    config.place_pose = [0.3, 0.2, 0.2, 0.0, 1.0, 0.0, 0.0]   # 放置位姿

    # 成功判定（需要完成抓取和放置两个阶段）
    config.success_criteria = {
        'grasp_threshold': 0.01,  # 抓取精度
        'place_threshold': 0.02,  # 放置精度
        'object_grasped': True,   # 物体被抓住
    }

    # 观测空间（需要多个相机观察物体）
    config.obs_config = ConfigDict()
    config.obs_config.state_dim = 21  # 7 joint + 7 tcp + 7 object_pose
    config.obs_config.num_cameras = 2  # 手腕相机 + 第三人称相机
    config.obs_config.camera_names = ["wrist_1", "overhead"]

    # 动作空间（包含夹爪控制）
    config.action_config = ConfigDict()
    config.action_config.dim = 7
    config.action_config.gripper_enabled = True  # 启用夹爪

    # BC 训练配置
    config.bc_config = ConfigDict()
    config.bc_config.epochs = 100  # 复杂任务需要更多训练
    config.bc_config.batch_size = 256
    config.bc_config.learning_rate = 3e-4

    # RLPD 配置
    config.rlpd_config = ConfigDict()
    config.rlpd_config.offline_steps = 20000
    config.rlpd_config.online_steps = 100000

    # 奖励配置
    config.reward_config = ConfigDict()
    config.reward_config.type = "shaped"  # 使用 shaped reward
    config.reward_config.grasp_reward = 10.0  # 抓取成功奖励
    config.reward_config.place_reward = 20.0  # 放置成功奖励

    return config
```

#### Step 3: 实现自定义奖励函数

`hil_serl_kinova/experiments/kinova_pick_place/reward.py`:

```python
import numpy as np

class PickPlaceReward:
    """抓取放置任务的奖励函数"""

    def __init__(self, config):
        self.config = config
        self.phase = "reach"  # reach -> grasp -> lift -> move -> place

    def compute_reward(self, obs, action, next_obs, info):
        """计算奖励"""
        reward = 0.0
        done = False

        tcp_pose = next_obs['state']['tcp_pose'][:3]
        object_pose = info.get('object_pose', None)
        gripper_state = next_obs['state']['gripper_position']

        if self.phase == "reach":
            # 阶段 1: 到达物体
            dist_to_object = np.linalg.norm(tcp_pose - object_pose)
            reward = -dist_to_object

            if dist_to_object < 0.02:
                self.phase = "grasp"
                reward += 5.0

        elif self.phase == "grasp":
            # 阶段 2: 抓取
            if gripper_state < 0.3 and info.get('object_grasped', False):
                reward = 10.0
                self.phase = "lift"

        elif self.phase == "lift":
            # 阶段 3: 抬起
            if tcp_pose[2] > 0.2:  # Z 高度 > 0.2m
                reward = 5.0
                self.phase = "move"

        elif self.phase == "move":
            # 阶段 4: 移动到目标
            dist_to_target = np.linalg.norm(tcp_pose - self.config.place_pose[:3])
            reward = -dist_to_target

            if dist_to_target < 0.02:
                self.phase = "place"

        elif self.phase == "place":
            # 阶段 5: 放置
            if gripper_state > 0.8:  # 松开夹爪
                reward = 20.0
                done = True

        return reward, done
```

#### Step 4: 创建数据收集脚本

`hil_serl_kinova/experiments/kinova_pick_place/collect_demos.py`:

```python
#!/usr/bin/env python3
"""收集 Pick and Place 演示数据"""

from kinova_rl_env import KinovaEnv, KinovaConfig
from vision_pro_control.core import VisionProBridge
from .config import get_config

def main():
    config = get_config()

    # 创建环境
    env = KinovaEnv(config=config)

    # 启动 VisionPro
    vp_bridge = VisionProBridge(config.visionpro_ip)

    # 收集演示
    demos = []
    for i in range(config.data_config.demos_num):
        print(f"\n【演示 {i+1}】")
        print("提示用户:")
        print("  1. 移动到物体上方")
        print("  2. 捏合手指抓取")
        print("  3. 抬起物体")
        print("  4. 移动到目标位置")
        print("  5. 松开手指放置")

        demo = collect_one_demo(env, vp_bridge)
        demos.append(demo)

    # 保存
    save_demos(demos, config.data_config.demos_dir)

if __name__ == '__main__':
    main()
```

---

## 更多任务示例

### 2. 插入任务（Insertion）

**目标**: 将销钉插入孔中

**关键点**:
- 需要高精度位姿控制
- 可能需要力觉反馈
- 接触检测

**配置示例**:
```python
config.task_name = "kinova_insertion"
config.peg_diameter = 0.01  # 1cm
config.hole_diameter = 0.012  # 1.2cm（间隙 1mm）
config.insertion_depth = 0.05  # 5cm
config.force_threshold = 10.0  # N
```

### 3. 旋拧任务（Screwing）

**目标**: 旋转螺丝

**关键点**:
- 需要螺旋运动控制
- 旋转力矩感知
- 多步骤操作

### 4. 组装任务（Assembly）

**目标**: 组装多个零件

**关键点**:
- 多物体追踪
- 顺序规划
- 精确对齐

---

## 任务设计建议

### 1. 从简单开始
- ✅ Reaching（最简单）
- ⬜ Pick and Place
- ⬜ Insertion
- ⬜ Assembly（最复杂）

### 2. 分解为子任务
```python
# 复杂任务分解为阶段
task_phases = [
    "approach",  # 接近
    "grasp",     # 抓取
    "lift",      # 抬起
    "transport", # 运输
    "place",     # 放置
]
```

### 3. 定义清晰的成功条件
```python
def is_success(obs, info):
    """明确的成功判定"""
    # 位置误差
    pos_error = np.linalg.norm(tcp_pos - target_pos)

    # 姿态误差
    ori_error = rotation_error(tcp_ori, target_ori)

    # 物体状态
    object_placed = info['object_on_target']

    return (pos_error < 0.01 and
            ori_error < 0.05 and
            object_placed)
```

### 4. 设计渐进式奖励
```python
# Sparse（简单但难训练）
reward = 1.0 if success else 0.0

# Dense（引导学习）
reward = -distance_to_target  # 越接近越好

# Shaped（最有效）
reward = (
    -0.1 * distance_to_target +      # 接近目标
    -0.05 * velocity_penalty +        # 平滑运动
    +5.0 if grasped else 0.0 +       # 抓取奖励
    +10.0 if placed else 0.0          # 完成奖励
)
```

---

## 任务目录结构

```
hil_serl_kinova/experiments/
├── kinova_reaching/         # 到达任务
│   ├── __init__.py
│   ├── config.py           # 配置
│   └── README.md
│
├── kinova_pick_place/      # 抓取放置（待实现）
│   ├── __init__.py
│   ├── config.py
│   ├── reward.py           # 自定义奖励
│   ├── collect_demos.py    # 数据收集
│   └── README.md
│
└── your_custom_task/       # 你的任务
    ├── __init__.py
    ├── config.py
    └── ...
```

---

## 使用自定义任务

```bash
# 1. 收集演示
python hil_serl_kinova/experiments/kinova_pick_place/collect_demos.py \
    --num_demos 20

# 2. 训练 BC
python hil_serl_kinova/train_bc_kinova.py \
    --config hil_serl_kinova/experiments/kinova_pick_place/config.py \
    --demos_dir ./demos/pick_place

# 3. 训练 RLPD
python hil_serl_kinova/train_rlpd_kinova.py \
    --config hil_serl_kinova/experiments/kinova_pick_place/config.py \
    --demos_dir ./demos/pick_place
```

---

## 下一步

1. 从 **Reaching** 任务开始熟悉流程
2. 根据你的应用场景设计新任务
3. 参考 HIL-SERL 论文中的任务设计
4. 逐步增加任务复杂度

## 参考资源

- [HIL-SERL 论文](https://arxiv.org/abs/2304.09870)
- [原始 HIL-SERL 仓库](https://github.com/youliangtan/hil-serl)
- 任务示例：Franka 环境的多个任务实现
