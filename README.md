# Kinova HIL-SERL Integration

Human-in-the-Loop Self-supervised Reinforcement Learning for Kinova Gen3 7-DOF robot.

## Quick Start - SpaceMouse Teleoperation

### Option 1: Simple Mode (Recommended)
Joint velocity control - works immediately in conda environment.

```bash
conda activate kinova_hilserl
./run_spacemouse_simple.sh
```

**What it does**: Maps SpaceMouse to joint velocities (pseudo-Cartesian feel)

### Option 2: IK Mode (True Cartesian Control)
Uses inverse kinematics for precise Cartesian control.

```bash
# One-time setup
sudo apt install python3-pip
/usr/bin/python3 -m pip install --user easyhid numpy scipy

# Run
./run_spacemouse_ik.sh
```

**What it does**: Solves IK to convert Cartesian commands → joint positions

## Controls

- **Move SpaceMouse** → Robot TCP follows
- **Left button** → Close gripper
- **Right button** → Open gripper
- **Press 'q'** → Exit

## Utilities

```bash
# Move robot to random safe pose (escape singularities)
python3 kinova_rl_env/move_to_random_pose.py

# Test gripper control
python3 tools/test_gripper.py
```

## Key Files

### Teleoperation
- `run_spacemouse_simple.sh` - Joint velocity launcher (conda)
- `run_spacemouse_ik.sh` - IK mode launcher (system Python)
- `kinova_rl_env/teleop_spacemouse.py` - Dual-mode teleop script
- `kinova_rl_env/teleop_spacemouse_ik_simple.py` - IK-based teleop
- `kinova_rl_env/move_to_random_pose.py` - Recovery tool

### Core Environment
- `kinova_rl_env/kinova_env/` - Gym environment implementation
- `kinova_rl_env/kinova_env/kinova_interface.py` - ROS2 robot interface
- `kinova_rl_env/kinova_env/camera_interface.py` - RealSense camera
- `kinova_rl_env/config/kinova_config.yaml` - Configuration

## Troubleshooting

### cffi version mismatch
```bash
unset PYTHONPATH
./run_spacemouse_simple.sh
```

### Controller not active
```bash
ros2 control list_controllers
# joint_trajectory_controller should be 'active'
```

### SpaceMouse not detected
```bash
lsusb | grep 3Dconnexion
# Check USB connection and permissions
```

## Documentation

See `START_HERE.md` for detailed setup and usage guide.

## System Requirements

- Ubuntu 22.04
- ROS2 Humble
- Kinova Gen3 7-DOF robot
- RealSense D435i camera
- SpaceMouse (3Dconnexion)
- Python 3.10 (conda environment)
