# 相机配置对比

## HIL-SERL 原版 (Franka) vs 当前实现 (Kinova)

### 原版 HIL-SERL 配置

**机械臂**: Franka Panda
**相机数量**: **2 个** RealSense
**位置**:
- `wrist_1`: 手腕相机 1（序列号: 130322274175）
- `wrist_2`: 手腕相机 2（序列号: 127122270572）

**代码示例** (`hil-serl/serl_robot_infra/franka_env/envs/franka_env.py`):

```python
REALSENSE_CAMERAS: Dict = {
    "wrist_1": "130322274175",  # RealSense 序列号
    "wrist_2": "127122270572",  # RealSense 序列号
}

# 初始化相机
self.init_cameras(config.REALSENSE_CAMERAS)

# 获取图像
images = {
    "wrist_1": ...,  # 128x128x3
    "wrist_2": ...,  # 128x128x3
}
```

**为什么用两个相机？**
- 👁️ 多视角观察：不同角度看物体
- 🎯 提高泛化：视觉特征更丰富
- 🔍 遮挡处理：一个被遮挡时另一个可见

---

### 当前实现（简化版）

**机械臂**: Kinova Gen3
**相机数量**: **1 个**（可扩展到多个）
**配置**: `wrist_1`

**配置文件** (`hil_serl_kinova/experiments/kinova_reaching/config.py`):

```python
config.obs_config = ConfigDict()
config.obs_config.num_cameras = 1  # 👈 只有 1 个
config.obs_config.camera_names = ["wrist_1"]
```

**为什么只用一个？**
- ✅ 简化入门：降低硬件要求
- ✅ 快速原型：基础任务（reaching）只需一个视角
- ⚠️ 可扩展：架构支持多相机，随时可添加

---

## 如何添加多个相机（USB 或 RealSense）

### 方案 1: 多个 USB 相机

**步骤 1**: 连接多个 USB 相机

```bash
# 检查可用相机
ls /dev/video*
# 输出: /dev/video0  /dev/video1  /dev/video2
```

**步骤 2**: 修改配置

`hil_serl_kinova/experiments/kinova_reaching/config.py`:

```python
config.obs_config.num_cameras = 3  # 改为 3 个
config.obs_config.camera_names = ["wrist_1", "wrist_2", "overhead"]

# 相机设备映射
config.camera_mapping = {
    "wrist_1": 0,    # /dev/video0
    "wrist_2": 1,    # /dev/video1
    "overhead": 2,   # /dev/video2（俯视相机）
}
```

**步骤 3**: 环境中初始化

`kinova_rl_env/kinova_env/kinova_env.py`:

```python
def _setup_cameras(self):
    """设置多个 USB 相机"""
    from kinova_rl_env import WebCamera

    for cam_name, cam_id in self.config.camera_mapping.items():
        camera = WebCamera(
            camera_id=cam_id,
            target_size=self.config.obs_config.image_size
        )
        camera.start()
        self.cameras[cam_name] = camera
        print(f"✓ 相机 {cam_name} (ID={cam_id}) 已启动")
```

### 方案 2: 多个 RealSense 相机

**步骤 1**: 连接多个 RealSense

```bash
# 查看序列号
rs-enumerate-devices

# 输出示例:
# Device 0: Intel RealSense D435 (SN: 123456789)
# Device 1: Intel RealSense D435 (SN: 987654321)
```

**步骤 2**: 配置序列号

```python
config.camera_mapping = {
    "wrist_1": "123456789",  # RealSense 序列号
    "wrist_2": "987654321",
}
```

**步骤 3**: 使用 RealSenseCamera

```python
from kinova_rl_env import RealSenseCamera

for cam_name, serial_num in self.config.camera_mapping.items():
    camera = RealSenseCamera(
        camera_name=cam_name,
        serial_number=serial_num,
        image_size=self.config.obs_config.image_size
    )
    camera.start()
    self.cameras[cam_name] = camera
```

### 方案 3: 混合配置

**同时使用 USB 和 RealSense**:

```python
config.camera_config = {
    "wrist_1": {"type": "realsense", "serial": "123456789"},
    "wrist_2": {"type": "webcam", "device_id": 0},
    "overhead": {"type": "webcam", "device_id": 1},
}

# 在环境中
for cam_name, cam_cfg in config.camera_config.items():
    if cam_cfg["type"] == "realsense":
        camera = RealSenseCamera(...)
    elif cam_cfg["type"] == "webcam":
        camera = WebCamera(...)

    self.cameras[cam_name] = camera
```

---

## 观测空间变化

### 单相机观测

```python
obs = {
    'state': np.array([...]),  # (14,)
    'images': {
        'wrist_1': np.array([...])  # (128, 128, 3)
    }
}
```

### 多相机观测

```python
obs = {
    'state': np.array([...]),  # (14,)
    'images': {
        'wrist_1': np.array([...]),   # (128, 128, 3)
        'wrist_2': np.array([...]),   # (128, 128, 3)
        'overhead': np.array([...])   # (128, 128, 3)
    }
}
```

### BC 网络输入变化

**单相机**:
```python
class BCPolicy(nn.Module):
    def __init__(self):
        self.image_encoder = CNN(in_channels=3)  # 单相机

    def forward(self, state, image):
        # image: (B, 3, 128, 128)
        image_feat = self.image_encoder(image)
        ...
```

**多相机**:
```python
class BCPolicy(nn.Module):
    def __init__(self, num_cameras=3):
        # 方案 A: 独立编码器
        self.image_encoders = nn.ModuleList([
            CNN(in_channels=3) for _ in range(num_cameras)
        ])

        # 方案 B: 共享编码器
        self.shared_encoder = CNN(in_channels=3)

    def forward(self, state, images):
        # images: dict with keys ['wrist_1', 'wrist_2', 'overhead']

        # 方案 A: 独立编码
        feats = [
            self.image_encoders[i](images[name])
            for i, name in enumerate(self.camera_names)
        ]
        image_feat = torch.cat(feats, dim=1)

        # 方案 B: 共享编码（推荐）
        feats = [
            self.shared_encoder(images[name])
            for name in self.camera_names
        ]
        image_feat = torch.cat(feats, dim=1)  # 拼接特征
        ...
```

---

## 推荐配置

### 入门阶段（你现在）
- ✅ **1 个 USB 相机**
- 简单、快速、易调试
- 适合 Reaching 任务

### 进阶阶段
- ✅ **2 个 USB 相机**
- wrist_1（手腕视角）+ overhead（俯视）
- 适合 Pick and Place

### 完整配置（对标原版）
- ✅ **2-3 个 RealSense**
- 多视角 + 深度信息
- 适合复杂操作任务

---

## 快速示例

### 测试你的多个 USB 相机

```bash
# 测试相机 0
python tests/test_camera.py --backend webcam --webcam-id 0

# 测试相机 1
python tests/test_camera.py --backend webcam --webcam-id 1

# 同时显示多个相机
python -c "
import cv2
import numpy as np

# 打开两个相机
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0 and ret1:
        # 并排显示
        combined = np.hstack([frame0, frame1])
        cv2.imshow('Cameras', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
"
```

---

## 总结

| 特性 | 原版 HIL-SERL | 当前实现 | 你的建议 |
|------|--------------|---------|---------|
| 相机数量 | 2 个 | 1 个 | 1-2 个 USB |
| 相机类型 | RealSense | 可配置 | USB WebCam ✅ |
| 硬件要求 | 高 | 低 | 适中 |
| 适合场景 | 复杂任务 | 入门/基础 | 逐步扩展 |

**建议路径**:
1. ✅ 先用 1 个 USB 相机跑通流程
2. ⬜ 添加第 2 个 USB 相机（俯视）
3. ⬜ 如果需要深度，升级到 RealSense
