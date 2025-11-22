#!/bin/bash
# 一键运行所有测试

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Kinova HIL-SERL 集成测试套件"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查ROS2环境
echo -e "\n${YELLOW}[1/7] 检查ROS2环境${NC}"
if ! ros2 topic list &> /dev/null; then
    echo -e "${RED}✗ ROS2环境未启动或未source${NC}"
    echo "请运行: source /opt/ros/humble/setup.bash"
    exit 1
fi
echo -e "${GREEN}✓ ROS2环境正常${NC}"

# 检查关键话题
echo -e "\n${YELLOW}[2/7] 检查ROS2话题${NC}"
if ros2 topic list | grep -q "/joint_states"; then
    echo -e "${GREEN}✓ /joint_states 存在${NC}"
else
    echo -e "${RED}✗ /joint_states 不存在${NC}"
    echo "请启动Kinova驱动: ros2 launch kortex_bringup kortex_control.launch.py"
    exit 1
fi

# 检查TF
echo -e "\n${YELLOW}[3/7] 检查TF变换${NC}"
timeout 2 ros2 run tf2_ros tf2_echo base_link tool_frame &> /dev/null && \
    echo -e "${GREEN}✓ TF: base_link → tool_frame 正常${NC}" || \
    echo -e "${YELLOW}⚠ TF查询超时，可能需要检查坐标系名称${NC}"

# 测试KinovaInterface
echo -e "\n${YELLOW}[4/7] 测试KinovaInterface${NC}"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from kinova_env.kinova_interface import KinovaInterface
import time

try:
    interface = KinovaInterface()
    interface.connect()
    time.sleep(1.0)

    pos, vel = interface.get_joint_state()
    assert pos is not None, "关节状态为None"

    tcp_pose = interface.get_tcp_pose()
    if tcp_pose is not None:
        print(f"  TCP位置: {tcp_pose[:3]}")
    else:
        print("  ⚠ TCP位姿获取失败（可能需要修改坐标系名称）")

    interface.disconnect()
    print("✓ KinovaInterface测试通过")
    sys.exit(0)
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ KinovaInterface测试通过${NC}"
else
    echo -e "${RED}✗ KinovaInterface测试失败${NC}"
    exit 1
fi

# 测试KinovaEnv
echo -e "\n${YELLOW}[5/7] 测试KinovaEnv${NC}"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from kinova_env.kinova_env import KinovaEnv
from kinova_env.config_loader import KinovaConfig

try:
    config = KinovaConfig.from_yaml('config/kinova_config.yaml')
    env = KinovaEnv(config=config)

    obs, info = env.reset()

    # 检查observation格式
    assert 'state' in obs, "缺少state"
    assert 'images' in obs, "缺少images"
    assert 'tcp_pose' in obs['state'], "缺少tcp_pose"

    print(f"  Observation格式: ✓")
    print(f"  tcp_pose shape: {obs['state']['tcp_pose'].shape}")

    # 测试reward
    reward = env._compute_reward(obs, None)
    print(f"  Reward计算: ✓")

    env.close()
    print("✓ KinovaEnv测试通过")
    sys.exit(0)
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ KinovaEnv测试通过${NC}"
else
    echo -e "${RED}✗ KinovaEnv测试失败${NC}"
    exit 1
fi

# 干运行测试
echo -e "\n${YELLOW}[6/7] 数据收集干运行测试${NC}"
if [ -f "test_integration_dry_run.py" ]; then
    python3 test_integration_dry_run.py
    echo -e "${GREEN}✓ 干运行测试通过${NC}"
else
    echo -e "${YELLOW}⚠ 跳过（test_integration_dry_run.py不存在）${NC}"
fi

# 检查demo文件（如果存在）
echo -e "\n${YELLOW}[7/7] 检查已有demo文件${NC}"
if ls demos/*/demo_*.pkl 1> /dev/null 2>&1; then
    DEMO_COUNT=$(ls demos/*/demo_*.pkl 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ 找到 ${DEMO_COUNT} 个demo文件${NC}"

    # 测试第一个demo
    FIRST_DEMO=$(ls demos/*/demo_*.pkl 2>/dev/null | head -1)
    python test_demo_format.py --demo_path "$FIRST_DEMO" > /dev/null 2>&1 && \
        echo -e "${GREEN}✓ Demo格式验证通过${NC}" || \
        echo -e "${YELLOW}⚠ Demo格式验证失败${NC}"
else
    echo -e "${YELLOW}⚠ 未找到demo文件（尚未收集数据）${NC}"
fi

echo -e "\n=========================================="
echo -e "${GREEN}✓ 所有测试完成！${NC}"
echo -e "=========================================="
echo ""
echo "下一步："
echo "  1. 收集演示数据:"
echo "     python record_kinova_demos.py --save_dir ./demos --num_demos 10"
echo ""
echo "  2. 验证demo格式:"
echo "     python test_demo_format.py --demo_path demos/reaching/demo_000.pkl"
echo ""
echo "  3. 转换为hdf5（可选）:"
echo "     python save_demo_utils.py --batch_convert demos/reaching"
echo ""
