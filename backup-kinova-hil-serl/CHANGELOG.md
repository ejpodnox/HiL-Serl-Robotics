# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-23

### Added

#### Core Features
- **VisionPro Teleoperation**: Complete Apple Vision Pro integration for intuitive robot control
  - Real-time hand tracking data streaming via gRPC
  - Coordinate mapping from VisionPro space to robot workspace
  - Safety limits and velocity constraints
  - Workspace calibration system

- **Kinova Gen3 Control**: Full ROS2 interface for Kinova robotic arm
  - Twist (velocity) control mode
  - Joint position control
  - Gripper control
  - Real-time state monitoring

- **Data Collection System**:
  - Independent teleoperation recorder (vision_pro_control/record_teleop_demos.py)
  - RL environment-based demo collector (kinova_rl_env/record_kinova_demos.py)
  - HIL-SERL format data export
  - Success/failure labeling for reward classifier

- **Gymnasium Environment**:
  - KinovaEnv with standard Gym interface
  - Pluggable camera backends (RealSense/WebCam/Dummy)
  - Configurable observation and action spaces
  - YAML-based configuration system

- **HIL-SERL Training Pipeline**:
  - **BC (Behavior Cloning)**:
    - Vision-based policy with CNN image encoder
    - State-action supervised learning
    - Data augmentation support
    - TensorBoard logging

  - **Reward Classifier**:
    - Binary success/failure classifier
    - Replaces hand-crafted reward functions
    - Supports both state and image inputs

  - **RLPD (RL with Prior Data)**:
    - SAC (Soft Actor-Critic) algorithm
    - Offline pretraining + online learning
    - Replay buffer with demo mixing
    - Entropy-regularized policy optimization

- **Policy Deployment**:
  - Multiple deployment modes (policy-only, hybrid, evaluation)
  - Interactive control with VisionPro assistance
  - Configurable human-AI mixing ratio

- **Tools and Utilities**:
  - Data validation and statistics
  - Format conversion (pkl ↔ hdf5)
  - Visualization tools for trajectories and training curves
  - One-click automation script

#### Documentation
- Comprehensive README with feature overview
- QUICKSTART guide for new users
- Detailed INSTALLATION guide
- Complete API documentation
- CONFIGURATION reference
- IMPLEMENTATION_SUMMARY with technical details

#### Package Structure
- Proper Python package with `__init__.py` files
- setuptools-based installation (setup.py)
- pyproject.toml for modern Python packaging
- Console script entry points for all tools

### Changed
- Reorganized documentation into docs/ directory
- Fixed all import paths to use package-relative imports
- Removed sys.path manipulations in favor of proper package imports
- Updated README to reference new documentation structure

### Fixed
- Fixed typo: `lastest_data` → `latest_data` in VisionProBridge
- Fixed absolute imports to relative imports in kinova_env package
- Fixed config path in record_kinova_demos.py
- Removed duplicate argparse imports in main functions

## [Unreleased]

### Planned Features
- [ ] Sim-to-Real transfer with Isaac Sim integration
- [ ] Multi-task learning support
- [ ] Distributed training with Ray
- [ ] Web-based monitoring dashboard
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions

---

## Version History

- **1.0.0** (2025-01-23): Initial release with complete HIL-SERL pipeline
  - Full VisionPro teleoperation
  - Complete training framework (BC + Classifier + RLPD)
  - Production-ready package structure
  - Comprehensive documentation

---

## Contributors

- Initial implementation and framework design
- VisionPro integration
- HIL-SERL training pipeline
- Documentation and examples

---

## Acknowledgments

- [HIL-SERL](https://github.com/youliangtan/hil-serl) for the original framework
- [Kinova Robotics](https://www.kinovarobotics.com/) for Gen3 robotic arm
- [Apple](https://www.apple.com/apple-vision-pro/) for Vision Pro platform
