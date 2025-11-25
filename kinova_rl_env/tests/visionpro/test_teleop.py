# test_visionpro_connection.py
import time
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from avp_stream import VisionProStreamer

def test_connection(ip):
    print(f"测试连接到 VisionPro: {ip}")

    try:
        # 创建streamer
        streamer = VisionProStreamer(ip=ip)
        print("✓ Streamer 创建成功")

        # 等待数据
        print("\n等待接收数据 (10秒)...")
        for i in range(10):
            time.sleep(1)

            data = streamer.latest

            # 获取hand pose
            hand_data = data['head']
            head_data = data['right_wrist']

            print(f"[{i+1}s] Hand: {hand_data is not None and hand_data.size > 0}, "
                  f"Head: {head_data is not None and head_data.size > 0}")

            if hand_data is not None and hand_data.size > 0:
                print(f"  Hand shape: {hand_data.shape}")
            if head_data is not None and head_data.size > 0:
                print(f"  Head shape: {head_data.shape}")

        print("\n✓ 测试完成")
        streamer.stop()

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    # 从配置文件读取默认 VisionPro IP
    default_vp_ip = '192.168.1.125'
    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        config = KinovaConfig.from_yaml('kinova_rl_env/config/kinova_config.yaml')
        if hasattr(config, 'visionpro') and hasattr(config.visionpro, 'ip'):
            default_vp_ip = config.visionpro.ip
    except Exception:
        pass

    ip = sys.argv[1] if len(sys.argv) > 1 else default_vp_ip
    test_connection(ip)