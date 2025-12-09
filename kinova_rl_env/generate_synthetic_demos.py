#!/usr/bin/env python3
"""
Generate synthetic Kinova demos (offline) for pipeline testing.

Format matches `record_spacemouse_demos.py` outputs:
    {
        "observations": [obs_t dict],
        "actions": [np.ndarray],
        "rewards": [float],
        "terminals": [bool],
        "truncations": [bool],
        "success": bool,
    }

Observations contain dummy state + random image:
    obs["state"]["tcp_pose"] shape (7,)
    obs["state"]["tcp_vel"]  shape (6,)
    obs["state"]["gripper_pose"] shape (1,)
    obs["images"]["wrist_1"] shape (128, 128, 3)
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


def load_limits(config_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    robot_cfg = cfg.get("robot", {})
    limits = robot_cfg.get("joint_limits", {})
    pos_min = np.array(limits.get("position_min", [-3.14] * 7), dtype=np.float32)
    pos_max = np.array(limits.get("position_max", [3.14] * 7), dtype=np.float32)
    action_dim = cfg.get("action", {}).get("dim", 7)
    return pos_min, pos_max, action_dim


def random_quat():
    """Uniform random unit quaternion."""
    u1, u2, u3 = np.random.rand(3)
    return np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], dtype=np.float32)


def generate_demo(
    horizon: int,
    pos_min: np.ndarray,
    pos_max: np.ndarray,
    action_dim: int,
) -> dict:
    target = np.random.uniform(low=[0.4, -0.2, 0.1], high=[0.7, 0.2, 0.4]).astype(np.float32)

    q = np.random.uniform(pos_min, pos_max)
    trajectory = []
    gripper = 0.0
    for t in range(horizon):
        action = np.clip(np.random.normal(scale=0.2, size=(action_dim,)), -1.0, 1.0).astype(np.float32)

        # Simple random walk in joint space (respect limits)
        dq = 0.05 * np.random.randn(pos_min.shape[0])
        q = np.clip(q + dq, pos_min, pos_max)

        # Fake TCP pose/vel (just placeholders)
        tcp_pos = np.random.uniform(low=[0.4, -0.2, 0.1], high=[0.7, 0.2, 0.5]).astype(np.float32)
        tcp_rot = random_quat()
        tcp_pose = np.concatenate([tcp_pos, tcp_rot])
        tcp_vel = np.random.normal(scale=0.01, size=(6,)).astype(np.float32)
        gripper = float(np.clip(gripper + np.random.uniform(-0.05, 0.05), 0.0, 1.0))

        # Dummy image
        img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)

        # Sparse success if near target
        dist = float(np.linalg.norm(tcp_pos - target))
        reward = 1.0 if dist < 0.02 else 0.0
        terminated = reward > 0
        truncated = (t == horizon - 1) and not terminated

        obs = {
            "state": {
                "tcp_pose": tcp_pose,
                "tcp_vel": tcp_vel,
                "gripper_pose": np.array([gripper], dtype=np.float32),
            },
            "images": {
                "wrist_1": img,
            },
        }

        trajectory.append(
            {
                "observation": obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        if terminated:
            break

    success = any(step["reward"] > 0 for step in trajectory)
    return {
        "observations": [t["observation"] for t in trajectory],
        "actions": [t["action"] for t in trajectory],
        "rewards": [t["reward"] for t in trajectory],
        "terminals": [t["terminated"] for t in trajectory],
        "truncations": [t["truncated"] for t in trajectory],
        "success": success,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Kinova demos (offline).")
    parser.add_argument("--config", type=str, default="kinova_rl_env/config/kinova_config.yaml", help="Config for limits")
    parser.add_argument("--save_dir", type=str, default="./demos_synth", help="Output directory")
    parser.add_argument("--task", type=str, default="synthetic", help="Task name folder")
    parser.add_argument("--num_demos", type=int, default=10, help="Number of demos to generate")
    parser.add_argument("--horizon", type=int, default=150, help="Max steps per demo")
    args = parser.parse_args()

    pos_min, pos_max, action_dim = load_limits(args.config)
    save_root = Path(args.save_dir) / args.task
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"Saving synthetic demos to: {save_root} (count={args.num_demos}, horizon={args.horizon})")

    for i in range(args.num_demos):
        demo = generate_demo(args.horizon, pos_min, pos_max, action_dim)
        out_path = save_root / f"demo_{i:03d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(demo, f)
        print(f"  ✓ demo_{i:03d}.pkl | steps={len(demo['actions'])} | success={demo['success']}")

    print("Done.")


if __name__ == "__main__":
    main()
