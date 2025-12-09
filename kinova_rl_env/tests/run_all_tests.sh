#!/bin/bash
# 统一测试运行器 - Kinova HIL-SERL 项目
# 运行所有测试分类：硬件测试、单元测试、集成测试

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 测试计数器
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Kinova HIL-SERL 统一测试套件"
echo "=========================================="
echo "测试目录: $SCRIPT_DIR"
echo "项目根目录: $PROJECT_ROOT"
echo ""

# 解析参数
TEST_CATEGORY="${1:-all}"  # all, hardware, unit, integration

# ============================================================
# 辅助函数
# ============================================================

run_test() {
    local test_name="$1"
    local test_command="$2"
    local required="${3:-false}"  # 是否必需（失败时退出）

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${BLUE}[测试 $TOTAL_TESTS] $test_name${NC}"

    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ 通过${NC}"
        return 0
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ 失败${NC}"
        cat /tmp/test_output.log

        if [ "$required" = "true" ]; then
            echo -e "${RED}关键测试失败，终止测试${NC}"
            exit 1
        fi
        return 1
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
    echo -e "${YELLOW}[测试 $TOTAL_TESTS] $test_name - 跳过${NC}"
    echo -e "${YELLOW}  原因: $reason${NC}"
}

# ============================================================
# 1. 硬件测试 (Hardware Tests)
# ============================================================

run_hardware_tests() {
    echo ""
    echo "=========================================="
    echo "【1/4】硬件测试 (Hardware Tests)"
    echo "=========================================="

    # 1.1 检查ROS2环境
    run_test "检查ROS2环境" \
        "ros2 topic list" \
        "true"

    # 1.2 检查关键话题
    run_test "检查/joint_states话题" \
        "ros2 topic list | grep -q '/joint_states'" \
        "true"

    # 1.3 检查TF
    run_test "检查TF变换 (base_link → tool_frame)" \
        "timeout 2 ros2 run tf2_ros tf2_echo base_link tool_frame" \
        "false"

    # 1.4 ROS2连接测试
    if [ -f "$SCRIPT_DIR/hardware/test_ros2_connection.py" ]; then
        run_test "ROS2关节状态读取测试" \
            "cd $PROJECT_ROOT && timeout 5 python3 tests/hardware/test_ros2_connection.py" \
            "false"
    else
        skip_test "ROS2关节状态读取测试" "文件不存在"
    fi

    # 1.5 速度控制测试
    if [ -f "$SCRIPT_DIR/hardware/test_velocity_control.py" ]; then
        run_test "速度控制测试" \
            "cd $PROJECT_ROOT && python3 tests/hardware/test_velocity_control.py" \
            "false"
    else
        skip_test "速度控制测试" "文件不存在"
    fi

    # 1.6 机器人连接测试
    if [ -f "$SCRIPT_DIR/hardware/test_robot_connection.py" ]; then
        run_test "机器人连接测试" \
            "cd $PROJECT_ROOT && python3 tests/hardware/test_robot_connection.py" \
            "false"
    else
        skip_test "机器人连接测试" "文件不存在"
    fi
}

# ============================================================
# 2. 单元测试 (Unit Tests)
# ============================================================

run_unit_tests() {
    echo ""
    echo "=========================================="
    echo "【2/4】单元测试 (Unit Tests)"
    echo "=========================================="

    # 2.1 KinovaInterface测试
    run_test "KinovaInterface基础功能" \
        "cd $PROJECT_ROOT && python3 -c '
import sys
sys.path.insert(0, \".\")
from kinova_env.kinova_interface import KinovaInterface
import time

interface = KinovaInterface()
interface.connect()
time.sleep(1.0)

pos, vel = interface.get_joint_state()
assert pos is not None, \"关节状态为None\"

tcp_pose = interface.get_tcp_pose()
if tcp_pose is not None:
    print(f\"TCP位置: {tcp_pose[:3]}\")

interface.disconnect()
print(\"✓ KinovaInterface测试通过\")
'" \
        "true"

    # 2.2 KinovaEnv测试
    run_test "KinovaEnv环境测试" \
        "cd $PROJECT_ROOT && python3 -c '
import sys
sys.path.insert(0, \".\")
from kinova_env.kinova_env import KinovaEnv
from kinova_env.config_loader import KinovaConfig

config = KinovaConfig.from_yaml(\"config/kinova_config.yaml\")
env = KinovaEnv(config=config)

obs, info = env.reset()

assert \"state\" in obs, \"缺少state\"
assert \"images\" in obs, \"缺少images\"
assert \"tcp_pose\" in obs[\"state\"], \"缺少tcp_pose\"

print(f\"Observation格式: ✓\")
print(f\"tcp_pose shape: {obs[\"state\"][\"tcp_pose\"].shape}\")

reward = env._compute_reward(obs, None)
print(f\"Reward计算: ✓\")

env.close()
print(\"✓ KinovaEnv测试通过\")
'" \
        "true"

    # 2.3 Demo格式验证
    if ls $PROJECT_ROOT/demos/*/demo_*.pkl 1> /dev/null 2>&1; then
        FIRST_DEMO=$(ls $PROJECT_ROOT/demos/*/demo_*.pkl 2>/dev/null | head -1)
        run_test "Demo数据格式验证" \
            "cd $PROJECT_ROOT && python3 tests/unit/test_demo_format.py --demo_path '$FIRST_DEMO'" \
            "false"
    else
        skip_test "Demo数据格式验证" "未找到demo文件"
    fi
}

# ============================================================
# 3. 集成测试 (Integration Tests)
# ============================================================

run_integration_tests() {
    echo ""
    echo "=========================================="
    echo "【3/3】集成测试 (Integration Tests)"
    echo "=========================================="

    # 完整遥操作集成测试
    if [ -f "$SCRIPT_DIR/integration/test_teleop_all.py" ]; then
        skip_test "完整遥操作集成测试" "需要硬件（手动运行: python tests/integration/test_teleop_all.py）"
    else
        skip_test "完整遥操作集成测试" "文件不存在"
    fi

    # 数据收集流程测试
    skip_test "数据收集流程测试" "需要硬件（手动运行: python kinova_rl_env/record_spacemouse_demos.py --num_demos 1）"
}

# ============================================================
# 主测试流程
# ============================================================

case "$TEST_CATEGORY" in
    hardware)
        run_hardware_tests
        ;;
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    all)
        run_hardware_tests
        run_unit_tests
        run_integration_tests
        ;;
    *)
        echo -e "${RED}未知测试类别: $TEST_CATEGORY${NC}"
        echo "用法: $0 [all|hardware|unit|integration]"
        exit 1
        ;;
esac

# ============================================================
# 测试总结
# ============================================================

echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo -e "总计: $TOTAL_TESTS 个测试"
echo -e "${GREEN}通过: $PASSED_TESTS${NC}"
echo -e "${RED}失败: $FAILED_TESTS${NC}"
echo -e "${YELLOW}跳过: $SKIPPED_TESTS${NC}"
echo "=========================================="

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有测试通过！${NC}"
    echo ""
    echo "下一步："
    echo "  1. 收集演示数据:"
    echo "     python record_kinova_demos.py --save_dir ./demos --num_demos 10"
    echo ""
    echo "  2. 验证demo格式:"
    echo "     python tests/unit/test_demo_format.py --demo_path demos/reaching/demo_000.pkl"
    echo ""
    echo "  3. 转换为hdf5（可选）:"
    echo "     python tests/utils/save_demo_utils.py --batch_convert demos/reaching"
    echo ""
    exit 0
else
    echo -e "${RED}✗ 有 $FAILED_TESTS 个测试失败${NC}"
    exit 1
fi
