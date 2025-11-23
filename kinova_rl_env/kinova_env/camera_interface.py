#!/usr/bin/env python3
"""
相机接口抽象层（可插拔设计）

支持多种相机后端：
- RealSenseCamera: Intel RealSense (ROS2)
- WebCamera: USB 网络摄像头
- DummyCamera: 模拟相机（用于无相机测试）
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2


class CameraInterface(ABC):
    """相机接口抽象类"""

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        获取图像

        Returns:
            image: RGB 图像，shape (H, W, 3)，dtype=uint8
        """
        pass

    @abstractmethod
    def start(self):
        """启动相机"""
        pass

    @abstractmethod
    def stop(self):
        """停止相机"""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """相机是否就绪"""
        pass


class RealSenseCamera(CameraInterface):
    """
    Intel RealSense 相机（通过 ROS2）

    订阅话题: /camera/{camera_name}/color/image_raw
    """

    def __init__(self, camera_name: str, topic: str = None, target_size=(128, 128)):
        """
        Args:
            camera_name: 相机名称（如 'wrist_1'）
            topic: ROS2 话题名称（默认 /camera/{camera_name}/color/image_raw）
            target_size: 目标图像尺寸 (H, W)
        """
        self.camera_name = camera_name
        self.topic = topic or f"/camera/{camera_name}/color/image_raw"
        self.target_size = target_size

        self.latest_image = None
        self.cv_bridge = None
        self.subscription = None
        self._ready = False

        print(f"RealSenseCamera '{camera_name}' 初始化 (topic: {self.topic})")

    def start(self):
        """启动相机订阅"""
        try:
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
            import rclpy
            from rclpy.node import Node

            self.cv_bridge = CvBridge()

            # 创建 ROS2 节点（如果还没有）
            if not rclpy.ok():
                rclpy.init()

            # 创建临时节点用于订阅
            class CameraNode(Node):
                def __init__(self, parent):
                    super().__init__(f'{parent.camera_name}_camera_node')
                    self.parent = parent
                    self.subscription = self.create_subscription(
                        Image,
                        parent.topic,
                        self.image_callback,
                        10
                    )

                def image_callback(self, msg):
                    # 转换 ROS Image 到 OpenCV
                    cv_image = self.parent.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
                    # Resize
                    resized = cv2.resize(cv_image, self.parent.target_size[::-1])
                    self.parent.latest_image = resized
                    self.parent._ready = True

            self.node = CameraNode(self)
            print(f"✓ RealSenseCamera '{self.camera_name}' 已启动")

        except ImportError as e:
            print(f"✗ RealSense 依赖缺失: {e}")
            print("  请安装: pip install cv_bridge")
            raise

    def stop(self):
        """停止相机"""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        self._ready = False
        print(f"✓ RealSenseCamera '{self.camera_name}' 已停止")

    def get_image(self) -> np.ndarray:
        """获取最新图像"""
        if self.latest_image is None:
            # 返回黑色图像
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        return self.latest_image.copy()

    def is_ready(self) -> bool:
        """相机是否就绪"""
        return self._ready


class WebCamera(CameraInterface):
    """
    USB 网络摄像头

    使用 OpenCV VideoCapture
    """

    def __init__(self, camera_id: int = 0, target_size=(128, 128)):
        """
        Args:
            camera_id: 相机设备 ID（默认 0）
            target_size: 目标图像尺寸 (H, W)
        """
        self.camera_id = camera_id
        self.target_size = target_size
        self.cap = None
        self._ready = False

        print(f"WebCamera (id={camera_id}) 初始化")

    def start(self):
        """启动相机"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开相机 {self.camera_id}")

        # 预热
        for _ in range(5):
            self.cap.read()

        self._ready = True
        print(f"✓ WebCamera (id={self.camera_id}) 已启动")

    def stop(self):
        """停止相机"""
        if self.cap is not None:
            self.cap.release()
        self._ready = False
        print(f"✓ WebCamera (id={self.camera_id}) 已停止")

    def get_image(self) -> np.ndarray:
        """获取最新图像"""
        if self.cap is None or not self._ready:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        resized = cv2.resize(frame_rgb, self.target_size[::-1])

        return resized

    def is_ready(self) -> bool:
        """相机是否就绪"""
        return self._ready


class DummyCamera(CameraInterface):
    """
    模拟相机（用于无相机测试）

    返回随机噪声或纯色图像
    """

    def __init__(self, target_size=(128, 128), mode='noise'):
        """
        Args:
            target_size: 目标图像尺寸 (H, W)
            mode: 'noise' (随机噪声) 或 'solid' (纯色)
        """
        self.target_size = target_size
        self.mode = mode
        self._ready = False

        print(f"DummyCamera (mode={mode}) 初始化")

    def start(self):
        """启动相机"""
        self._ready = True
        print("✓ DummyCamera 已启动")

    def stop(self):
        """停止相机"""
        self._ready = False
        print("✓ DummyCamera 已停止")

    def get_image(self) -> np.ndarray:
        """获取模拟图像"""
        if self.mode == 'noise':
            # 随机噪声
            return np.random.randint(0, 256, (*self.target_size, 3), dtype=np.uint8)
        else:
            # 灰色
            return np.full((*self.target_size, 3), 128, dtype=np.uint8)

    def is_ready(self) -> bool:
        """相机是否就绪"""
        return self._ready


class CameraManager:
    """
    相机管理器（支持多相机）

    配置驱动，自动选择相机后端
    """

    def __init__(self, camera_config: dict):
        """
        Args:
            camera_config: 相机配置字典

        示例配置:
        {
            'enabled': True,
            'backend': 'realsense',  # 'realsense', 'webcam', 'dummy'
            'cameras': {
                'wrist_1': {
                    'type': 'realsense',
                    'topic': '/camera/wrist_1/color/image_raw'
                }
            },
            'image_size': [128, 128]
        }
        """
        self.config = camera_config
        self.cameras = {}
        self.enabled = camera_config.get('enabled', False)

        if not self.enabled:
            print("⚠️  相机已禁用，使用 DummyCamera")
            self._init_dummy_cameras()
        else:
            self._init_cameras()

    def _init_cameras(self):
        """根据配置初始化相机"""
        backend = self.config.get('backend', 'dummy')
        image_size = tuple(self.config.get('image_size', [128, 128]))
        cameras_cfg = self.config.get('cameras', {})

        for camera_name, camera_cfg in cameras_cfg.items():
            camera_type = camera_cfg.get('type', backend)

            if camera_type == 'realsense':
                camera = RealSenseCamera(
                    camera_name=camera_name,
                    topic=camera_cfg.get('topic'),
                    target_size=image_size
                )
            elif camera_type == 'webcam':
                camera_id = camera_cfg.get('id', 0)
                camera = WebCamera(
                    camera_id=camera_id,
                    target_size=image_size
                )
            else:  # dummy
                camera = DummyCamera(
                    target_size=image_size,
                    mode=camera_cfg.get('mode', 'noise')
                )

            self.cameras[camera_name] = camera

    def _init_dummy_cameras(self):
        """初始化虚拟相机（用于测试）"""
        image_size = tuple(self.config.get('image_size', [128, 128]))

        # 默认创建一个 wrist_1 相机
        self.cameras['wrist_1'] = DummyCamera(target_size=image_size, mode='solid')

    def start(self):
        """启动所有相机"""
        for camera_name, camera in self.cameras.items():
            camera.start()

    def stop(self):
        """停止所有相机"""
        for camera_name, camera in self.cameras.items():
            camera.stop()

    def get_images(self) -> dict:
        """
        获取所有相机的图像

        Returns:
            images: dict，键为相机名称，值为图像
        """
        images = {}
        for camera_name, camera in self.cameras.items():
            images[camera_name] = camera.get_image()
        return images

    def get_image(self, camera_name: str) -> np.ndarray:
        """
        获取指定相机的图像

        Args:
            camera_name: 相机名称
        Returns:
            image: RGB 图像
        """
        if camera_name not in self.cameras:
            raise KeyError(f"相机 '{camera_name}' 不存在")
        return self.cameras[camera_name].get_image()

    def is_ready(self) -> bool:
        """所有相机是否就绪"""
        return all(camera.is_ready() for camera in self.cameras.values())


# ============ 使用示例 ============

if __name__ == '__main__':
    # 示例 1: 使用 DummyCamera
    print("=" * 60)
    print("示例 1: DummyCamera")
    print("=" * 60)

    config_dummy = {
        'enabled': False,
        'image_size': [128, 128]
    }

    manager = CameraManager(config_dummy)
    manager.start()

    image = manager.get_image('wrist_1')
    print(f"✓ 获取图像: shape={image.shape}, dtype={image.dtype}")

    manager.stop()

    # 示例 2: 使用 WebCamera
    print("\n" + "=" * 60)
    print("示例 2: WebCamera")
    print("=" * 60)

    config_webcam = {
        'enabled': True,
        'backend': 'webcam',
        'cameras': {
            'wrist_1': {
                'type': 'webcam',
                'id': 0
            }
        },
        'image_size': [128, 128]
    }

    try:
        manager2 = CameraManager(config_webcam)
        manager2.start()

        image = manager2.get_image('wrist_1')
        print(f"✓ 获取图像: shape={image.shape}, dtype={image.dtype}")

        # 保存图像
        cv2.imwrite('/tmp/test_camera.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("✓ 图像已保存到 /tmp/test_camera.jpg")

        manager2.stop()

    except Exception as e:
        print(f"✗ WebCamera 测试失败: {e}")

    print("\n✓ 相机接口测试完成")
