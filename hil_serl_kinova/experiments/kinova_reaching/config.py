"""
Kinova Reaching 任务配置（HIL-SERL 格式）

使用方法:
    from hil_serl_kinova.experiments.kinova_reaching.config import get_config
    config = get_config()
"""

from ml_collections import ConfigDict


def get_config():
    """获取 Kinova Reaching 任务配置"""

    config = ConfigDict()

    # ============ 任务基本信息 ============
    config.task_name = "kinova_reaching"
    config.description = "Kinova Gen3 机械臂到达目标位置"

    # ============ 环境配置 ============
    config.env_name = "KinovaEnv"
    config.env_config_path = "kinova_rl_env/config/kinova_config.yaml"

    # 从配置文件读取默认 IP
    default_robot_ip = "192.168.8.10"
    try:
        from kinova_rl_env.kinova_env.config_loader import KinovaConfig
        kinova_config = KinovaConfig.from_yaml(config.env_config_path)
        default_robot_ip = kinova_config.robot.ip
    except Exception:
        pass

    # Kinova 机械臂参数
    config.robot_ip = default_robot_ip
    config.robot_dof = 7

    # 目标位姿（在 base_link 坐标系下）
    config.target_pose = [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]  # [x, y, z, qx, qy, qz, qw]
    config.reset_pose = [0.3, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0]   # 重置位置

    # 成功判定
    config.success_threshold = 0.02  # 2cm

    # ============ 观测空间 ============
    config.obs_config = ConfigDict()
    config.obs_config.state_dim = 14  # 7 (joint_pos) + 7 (tcp_pose)
    config.obs_config.image_size = [128, 128]
    config.obs_config.num_cameras = 1
    config.obs_config.camera_names = ["wrist_1"]

    # ============ 动作空间 ============
    config.action_config = ConfigDict()
    config.action_config.dim = 7  # [dx, dy, dz, drx, dry, drz, gripper]
    config.action_config.type = "delta_pose"  # 位移增量
    config.action_config.scale = 0.01  # 动作缩放因子

    # ============ 训练数据 ============
    config.data_config = ConfigDict()
    config.data_config.demos_dir = "./demos/reaching"  # 演示数据目录
    config.data_config.demos_num = 10  # 使用的演示数量
    config.data_config.train_ratio = 0.9  # 训练集比例
    config.data_config.shuffle = True  # 打乱数据
    config.data_config.augmentation = False  # 数据增强（可选）

    # ============ BC (Behavior Cloning) 配置 ============
    config.bc_config = ConfigDict()
    config.bc_config.epochs = 50  # 训练轮数
    config.bc_config.batch_size = 256  # 批大小
    config.bc_config.learning_rate = 3e-4  # 学习率
    config.bc_config.weight_decay = 1e-4  # 权重衰减
    config.bc_config.clip_grad_norm = 1.0  # 梯度裁剪

    # 策略网络结构
    config.bc_config.policy_hidden_dims = [256, 256, 256]
    config.bc_config.activation = "relu"
    config.bc_config.dropout = 0.1

    # ============ RLPD (Reinforcement Learning with Prior Data) 配置 ============
    config.rlpd_config = ConfigDict()
    config.rlpd_config.offline_steps = 10000  # 离线预训练步数
    config.rlpd_config.online_steps = 50000   # 在线训练步数
    config.rlpd_config.batch_size = 256
    config.rlpd_config.utd_ratio = 4  # Updates-to-Data ratio
    config.rlpd_config.gamma = 0.99  # 折扣因子
    config.rlpd_config.tau = 0.005  # 目标网络软更新系数

    # Critic 网络
    config.rlpd_config.critic_hidden_dims = [256, 256, 256]
    config.rlpd_config.critic_lr = 3e-4

    # Actor 网络
    config.rlpd_config.actor_hidden_dims = [256, 256, 256]
    config.rlpd_config.actor_lr = 3e-4

    # 温度参数（SAC）
    config.rlpd_config.temp_lr = 3e-4
    config.rlpd_config.init_temp = 1.0
    config.rlpd_config.target_entropy = None  # 自动设置

    # ============ Reward Classifier 配置 ============
    config.classifier_config = ConfigDict()
    config.classifier_config.enabled = False  # 是否启用
    config.classifier_config.checkpoint_path = None  # 模型检查点
    config.classifier_config.threshold = 0.5  # 分类阈值

    # 训练参数
    config.classifier_config.epochs = 20
    config.classifier_config.batch_size = 64
    config.classifier_config.learning_rate = 1e-4
    config.classifier_config.hidden_dims = [256, 256]

    # ============ 日志和检查点 ============
    config.logging = ConfigDict()
    config.logging.log_dir = "./logs/kinova_reaching"
    config.logging.checkpoint_dir = "./checkpoints/kinova_reaching"
    config.logging.log_frequency = 100  # 每 N 步记录一次
    config.logging.save_frequency = 1000  # 每 N 步保存检查点
    config.logging.eval_frequency = 500  # 每 N 步评估一次

    # Wandb 日志（可选）
    config.logging.use_wandb = False
    config.logging.wandb_project = "kinova-hil-serl"
    config.logging.wandb_entity = None

    # ============ 评估配置 ============
    config.eval_config = ConfigDict()
    config.eval_config.num_episodes = 10  # 评估回合数
    config.eval_config.max_steps = 200  # 每回合最大步数
    config.eval_config.render = False  # 是否渲染
    config.eval_config.save_video = False  # 是否保存视频

    # ============ 安全配置 ============
    config.safety = ConfigDict()
    config.safety.max_linear_velocity = 0.05  # 最大线速度 (m/s)
    config.safety.max_angular_velocity = 0.1  # 最大角速度 (rad/s)
    config.safety.enable_collision_check = False  # 碰撞检测（可选）
    config.safety.workspace_limits = None  # 工作空间限制（可选）

    # ============ 其他配置 ============
    config.seed = 42  # 随机种子
    config.device = "cuda"  # "cuda" 或 "cpu"
    config.num_workers = 4  # 数据加载线程数

    return config


def get_bc_config():
    """
    获取纯 BC 训练配置（快速测试用）

    Returns:
        config: BC 配置
    """
    config = get_config()

    # 简化配置
    config.bc_config.epochs = 20  # 减少训练轮数
    config.data_config.demos_num = 5  # 减少演示数量

    return config


def get_rlpd_config():
    """
    获取完整 RLPD 训练配置

    Returns:
        config: RLPD 配置
    """
    config = get_config()

    # 启用所有功能
    config.classifier_config.enabled = True

    return config


# ============ 配置验证 ============

def validate_config(config):
    """
    验证配置的合法性

    Args:
        config: ConfigDict

    Raises:
        ValueError: 配置不合法
    """
    # 检查必需字段
    required_fields = [
        'task_name',
        'env_name',
        'robot_ip',
        'target_pose',
    ]

    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"配置缺少必需字段: {field}")

    # 检查数值范围
    if config.data_config.demos_num < 1:
        raise ValueError("demos_num 必须 >= 1")

    if config.bc_config.epochs < 1:
        raise ValueError("bc_epochs 必须 >= 1")

    if not 0 <= config.data_config.train_ratio <= 1:
        raise ValueError("train_ratio 必须在 [0, 1] 范围内")

    print("✓ 配置验证通过")


# ============ 使用示例 ============

if __name__ == '__main__':
    # 获取配置
    config = get_config()

    # 验证配置
    validate_config(config)

    # 打印配置
    print("\n" + "=" * 60)
    print("Kinova Reaching 任务配置")
    print("=" * 60)
    print(f"任务名称: {config.task_name}")
    print(f"机械臂 IP: {config.robot_ip}")
    print(f"目标位姿: {config.target_pose}")
    print(f"演示数量: {config.data_config.demos_num}")
    print(f"BC 训练轮数: {config.bc_config.epochs}")
    print("=" * 60)

    # 保存配置为 YAML（可选）
    import yaml
    config_dict = config.to_dict()

    with open('/tmp/kinova_reaching_config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print("\n✓ 配置已保存到 /tmp/kinova_reaching_config.yaml")
