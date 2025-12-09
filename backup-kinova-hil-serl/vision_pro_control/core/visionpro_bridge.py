import numpy as np
from avp_stream import VisionProStreamer
import threading
import time

class VisionProBridge:

    def __init__(self, avp_ip: str, use_right_hand: bool = True):
        self.avp_ip = avp_ip
        self.use_right_hand = use_right_hand
        self.streamer = None
        self.running = False
        self.thread = None

        self.latest_data = {
            'head_pose' : np.eye(4),
            'wrist_pose' : np.eye(4),
            'pinch_distance' : 1.0,
            'wrist_roll' : 0.0,
            'timestamp' : 0  # 初始化为0，接收到数据后更新
        }
        self.data_lock = threading.Lock()

    def start(self):
        print(f"Connecting VisionPro(ip:{self.avp_ip})")
        self.streamer = VisionProStreamer(ip=self.avp_ip, record=False)
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("VisionPro data stream started!")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("VisionPro data stream stopped!")

    def _update_loop(self):
        while self.running:
            try:
                r = self.streamer.latest
                
                if not self._validate_data(r):
                    continue  # 跳过格式不正确的数据包

                with self.data_lock:
                    self.latest_data['head_pose'] = r['head'][0]

                    if self.use_right_hand:
                        self.latest_data['wrist_pose'] = r['right_wrist'][0]
                        self.latest_data['pinch_distance'] = r['right_pinch_distance']
                        self.latest_data['wrist_roll'] = r['right_wrist_roll']
                    else:
                        self.latest_data['wrist_pose'] = r['left_wrist'][0]
                        self.latest_data['pinch_distance'] = r['left_pinch_distance']
                        self.latest_data['wrist_roll'] = r['left_wrist_roll']
                self.latest_data['timestamp'] = time.time()

            except Exception as e:
                print("Update Error!")
            
            time.sleep(0.05)

    def get_hand_relative_to_head(self):
        with self.data_lock:
            head_pose = self.latest_data['head_pose'].copy()
            wrist_pose = self.latest_data['wrist_pose'].copy()

        head_inv = np.linalg.inv(head_pose)
        relative_pose = head_inv @ wrist_pose

        position = relative_pose[:3,3]
        rotation = relative_pose[:3,:3]

        return position, rotation
    
    def get_pinch_state(self, threshold: float = 0.02):

        with self.data_lock:
            pinch_distance = self.latest_data['pinch_distance']

        return pinch_distance < threshold
    
    def get_pinch_distance(self) -> float:
        """
        获取手指捏合距离（连续值）
        Returns:
            pinch_distance: 捏合距离 (m)，范围通常 [0.0, 0.1]
        """
        with self.data_lock:
            pinch_distance = self.latest_data['pinch_distance']

        return pinch_distance

    def get_latest_data(self) -> dict:
        """
        获取最新的 VisionPro 数据
        Returns:
            data: 包含 head_pose, wrist_pose, pinch_distance, wrist_roll, timestamp
        """
        with self.data_lock:
            return self.latest_data.copy()

    def _validate_data(self, r: dict) -> bool:
        """
        验证 VisionPro 数据格式
        Args:
            r: VisionPro 数据字典
        Returns:
            bool: 数据格式是否正确
        """
        try:
            # 检查必需的键
            required_keys = ['head', 'right_wrist', 'left_wrist', 
                            'right_pinch_distance', 'left_pinch_distance',
                            'right_wrist_roll', 'left_wrist_roll']
            
            for key in required_keys:
                if key not in r:
                    return False
            
            # 检查 head 格式
            if not isinstance(r['head'], np.ndarray) or r['head'].shape != (1, 4, 4):
                return False
            
            # 检查 wrist 格式
            if not isinstance(r['right_wrist'], np.ndarray) or r['right_wrist'].shape != (1, 4, 4):
                return False
            
            if not isinstance(r['left_wrist'], np.ndarray) or r['left_wrist'].shape != (1, 4, 4):
                return False
            
            # 检查 pinch_distance 格式
            if not isinstance(r['right_pinch_distance'], (float, int)):
                return False
            
            if not isinstance(r['left_pinch_distance'], (float, int)):
                return False
            
            # 检查 wrist_roll 格式
            if not isinstance(r['right_wrist_roll'], (float, int)):
                return False
            
            if not isinstance(r['left_wrist_roll'], (float, int)):
                return False
            
            return True
            
        except Exception as e:
            print(f"数据验证错误: {e}")
            return False
        
    