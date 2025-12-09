#!/usr/bin/env python3
"""
Kinova HIL-SERL Setup Script

安装方法:
    pip install -e .              # 开发模式安装
    pip install -e .[dev]         # 安装开发依赖
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="kinova-hil-serl",
    version="1.0.0",
    author="Kinova HIL-SERL Team",
    description="Human-in-the-Loop Self-supervised RL for Kinova Gen3 (ROS2 control)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kinova-hil-serl",
    packages=find_packages(exclude=["tests", "scripts", "demos", "logs", "checkpoints", "plots"]),
    python_requires=">=3.8",

    install_requires=[
        # Core dependencies
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "ml-collections>=0.1.0",

        # Deep Learning
        "torch>=2.0.0",
        "torchvision>=0.15.0",

        # RL
        "gymnasium>=0.28.0",

        # Computer Vision
        "opencv-python>=4.5.0",
        "pyrealsense2>=2.50.0",

        # Visualization
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.8.0",

        # Data Processing
        "h5py>=3.0.0",
        "pillow>=8.0.0",

        # Utilities
        "tqdm>=4.60.0",
        "tabulate>=0.8.9",
        "pyspacemouse>=1.0.3",
        "hidapi>=0.14.0",
    ],

    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
        ],
    },

    entry_points={
        "console_scripts": [
            # Data Collection
            "kinova-record-demos=kinova_rl_env.record_kinova_demos:main",
            "kinova-record-spacemouse=kinova_rl_env.record_spacemouse_demos:main",
            "kinova-record-labels=hil_serl_kinova.record_success_fail_demos:main",

            # Training
            "kinova-train-bc=hil_serl_kinova.train_bc_kinova:main",
            "kinova-train-classifier=hil_serl_kinova.train_reward_classifier:main",
            "kinova-train-rlpd=hil_serl_kinova.train_rlpd_kinova:main",

            # Deployment
            "kinova-deploy=hil_serl_kinova.deploy_policy:main",

            # Tools
            "kinova-data-utils=hil_serl_kinova.tools.data_utils:main",
            "kinova-visualize=hil_serl_kinova.tools.visualize:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],

    include_package_data=True,
    package_data={
        "kinova_rl_env": ["config/*.yaml"],
        "hil_serl_kinova": ["experiments/*/config.py"],
    },
)
