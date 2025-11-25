#!/bin/bash
###############################################################################
# Kinova HIL-SERL 完整流程一键运行脚本
#
# 自动执行：
# 1. 数据收集
# 2. BC 训练
# 3. 策略评估
# 4. (可选) Reward Classifier 训练
# 5. (可选) RLPD 在线学习
#
# 使用方法:
#   bash run_full_pipeline.sh --mode quick      # 快速原型（5条演示, BC训练）
#   bash run_full_pipeline.sh --mode standard   # 标准流程（20条演示, BC训练）
#   bash run_full_pipeline.sh --mode full       # 完整流程（包含RLPD）
###############################################################################

set -e  # 遇到错误立即退出

# ============ 配置参数 ============

# 默认值
MODE="standard"
TASK_NAME="reaching"
DEMOS_DIR="./demos/${TASK_NAME}"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"
VisionPro_IP="192.168.1.125"

# ============ 解析命令行参数 ============

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --vp_ip)
            VisionPro_IP="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ============ 根据模式设置参数 ============

case $MODE in
    quick)
        NUM_DEMOS=5
        BC_EPOCHS=20
        RUN_RLPD=false
        echo "🚀 快速原型模式"
        ;;
    standard)
        NUM_DEMOS=20
        BC_EPOCHS=50
        RUN_RLPD=false
        echo "🚀 标准流程模式"
        ;;
    full)
        NUM_DEMOS=20
        BC_EPOCHS=50
        RUN_RLPD=true
        OFFLINE_STEPS=10000
        ONLINE_STEPS=50000
        echo "🚀 完整流程模式"
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        echo "支持的模式: quick, standard, full"
        exit 1
        ;;
esac

# ============ 打印配置 ============

echo "============================================================"
echo "Kinova HIL-SERL 完整流程"
echo "============================================================"
echo "模式: $MODE"
echo "任务: $TASK_NAME"
echo "演示数量: $NUM_DEMOS"
echo "BC 训练轮数: $BC_EPOCHS"
echo "RLPD: $RUN_RLPD"
echo "VisionPro IP: $VisionPro_IP"
echo "============================================================"
echo ""

# ============ 步骤 1: 数据收集 ============

echo "📊 步骤 1/5: 数据收集"
echo "------------------------------------------------------------"

if [ -d "$DEMOS_DIR" ] && [ "$(ls -A $DEMOS_DIR)" ]; then
    echo "⚠️  演示数据已存在: $DEMOS_DIR"
    read -p "是否跳过数据收集? (y/n): " skip_data
    if [ "$skip_data" != "y" ]; then
        echo "正在收集数据..."
        python kinova_rl_env/record_kinova_demos.py \
            --save_dir "$DEMOS_DIR" \
            --num_demos "$NUM_DEMOS" \
            --vp_ip "$VisionPro_IP" \
            --task "$TASK_NAME"
    else
        echo "✅ 跳过数据收集"
    fi
else
    echo "正在收集数据..."
    python kinova_rl_env/record_kinova_demos.py \
        --save_dir "$DEMOS_DIR" \
        --num_demos "$NUM_DEMOS" \
        --vp_ip "$VisionPro_IP" \
        --task "$TASK_NAME"
fi

echo ""

# ============ 步骤 2: 验证数据 ============

echo "✅ 步骤 2/5: 验证数据"
echo "------------------------------------------------------------"

python hil_serl_kinova/tools/data_utils.py --validate "$DEMOS_DIR"
python hil_serl_kinova/tools/data_utils.py --stats "$DEMOS_DIR"

echo ""

# ============ 步骤 3: BC 训练 ============

echo "🎓 步骤 3/5: BC 训练"
echo "------------------------------------------------------------"

BC_CHECKPOINT_DIR="${CHECKPOINT_DIR}/bc_${TASK_NAME}"

python hil_serl_kinova/train_bc_kinova.py \
    --config hil_serl_kinova/experiments/kinova_reaching/config.py \
    --demos_dir "$DEMOS_DIR" \
    --checkpoint_dir "$BC_CHECKPOINT_DIR" \
    --epochs "$BC_EPOCHS"

echo "✅ BC 训练完成"
echo "检查点: ${BC_CHECKPOINT_DIR}/best_model.pt"
echo ""

# ============ 步骤 4: 策略评估 ============

echo "📈 步骤 4/5: 策略评估"
echo "------------------------------------------------------------"

python hil_serl_kinova/deploy_policy.py \
    --checkpoint "${BC_CHECKPOINT_DIR}/best_model.pt" \
    --mode evaluation \
    --num_episodes 10

echo ""

# ============ 步骤 5: 可视化 ============

echo "📊 步骤 5/5: 生成可视化"
echo "------------------------------------------------------------"

PLOT_DIR="./plots/${TASK_NAME}"
mkdir -p "$PLOT_DIR"

# 绘制数据集统计
python hil_serl_kinova/tools/visualize.py \
    --dataset "$DEMOS_DIR" \
    --output "${PLOT_DIR}/dataset_stats.png"

# 绘制训练曲线
python hil_serl_kinova/tools/visualize.py \
    --training "${LOG_DIR}/kinova_reaching/bc" \
    --output "${PLOT_DIR}/training_curves.png"

# 绘制多轨迹对比
python hil_serl_kinova/tools/visualize.py \
    --multi "$DEMOS_DIR" \
    --output "${PLOT_DIR}/trajectories.png" \
    --max_demos 5

echo "✅ 可视化完成，保存在: $PLOT_DIR"
echo ""

# ============ (可选) RLPD 训练 ============

if [ "$RUN_RLPD" = true ]; then
    echo "🔥 额外步骤: RLPD 在线学习"
    echo "------------------------------------------------------------"

    RLPD_CHECKPOINT_DIR="${CHECKPOINT_DIR}/rlpd_${TASK_NAME}"

    python hil_serl_kinova/train_rlpd_kinova.py \
        --config hil_serl_kinova/experiments/kinova_reaching/config.py \
        --demos_dir "$DEMOS_DIR" \
        --bc_checkpoint "${BC_CHECKPOINT_DIR}/best_model.pt" \
        --offline_steps "$OFFLINE_STEPS" \
        --online_steps "$ONLINE_STEPS"

    echo "✅ RLPD 训练完成"
    echo ""
fi

# ============ 总结 ============

echo "============================================================"
echo "✅ 完整流程执行成功！"
echo "============================================================"
echo ""
echo "📁 输出目录:"
echo "  - 演示数据: $DEMOS_DIR"
echo "  - 检查点: $CHECKPOINT_DIR"
echo "  - 日志: $LOG_DIR"
echo "  - 可视化: $PLOT_DIR"
echo ""
echo "🚀 下一步:"
echo "  1. 查看可视化结果: $PLOT_DIR"
echo "  2. 查看训练日志: tensorboard --logdir $LOG_DIR"
echo "  3. 部署策略: python hil_serl_kinova/deploy_policy.py --checkpoint ${BC_CHECKPOINT_DIR}/best_model.pt"
echo ""
echo "============================================================"
