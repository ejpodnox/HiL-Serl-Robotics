# USB 相机配置指南

## 使用 USB 相机（WebCamera）

如果你的相机是从 USB 读入的，需要修改配置文件。

### 方法 1: 修改 YAML 配置

编辑 `kinova_rl_env/config/kinova_config.yaml`:

```yaml
camera:
  enabled: true
  backend: "webcam"  # 改为 webcam（而不是 realsense）

  # USB 相机配置
  webcam_cameras:
    wrist_1:
      device_id: 0  # USB 设备 ID（通常是 0、1、2...）
      image_size: [128, 128]

    # 如果有多个 USB 相机
    wrist_2:
      device_id: 1
      image_size: [128, 128]
```

### 方法 2: 代码中直接使用

```python
from kinova_rl_env import WebCamera

# 创建 USB 相机
camera = WebCamera(
    camera_id=0,  # /dev/video0
    target_size=(128, 128)
)

camera.start()
image = camera.get_image()  # 获取图像 (128, 128, 3)
camera.stop()
```

### 方法 3: 在环境中使用

修改 `kinova_rl_env/kinova_env/kinova_env.py`:

```python
def _setup_cameras(self):
    """设置相机"""
    if self.config.camera.backend == "webcam":
        # USB 相机
        for cam_name, cam_id in [("wrist_1", 0), ("wrist_2", 1)]:
            camera = WebCamera(
                camera_id=cam_id,
                target_size=self.config.camera.image_size
            )
            camera.start()
            self.cameras[cam_name] = camera
```

## 检测可用的 USB 相机

```bash
# Linux 下查看可用相机设备
ls /dev/video*

# 输出示例：
# /dev/video0  /dev/video1
```

或使用 Python 测试：

```python
import cv2

# 测试相机 ID 0-4
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ 相机 {i} 可用")
        cap.release()
    else:
        print(f"✗ 相机 {i} 不可用")
```

## RealSense vs USB 相机对比

| 特性 | RealSense | USB WebCam |
|------|-----------|------------|
| 接口 | ROS2 话题订阅 | OpenCV 直接读取 |
| 深度信息 | ✅ 有 | ✗ 无 |
| 配置复杂度 | 高（需要 ROS2） | 低 |
| 适合场景 | 需要深度的任务 | 纯视觉任务 |
| 你的情况 | | ✅ 推荐 |

## 快速测试

```bash
# 测试 USB 相机
python tests/test_camera.py --backend webcam --webcam-id 0
```
