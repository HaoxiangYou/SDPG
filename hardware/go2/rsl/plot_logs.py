#!/usr/bin/env python3
"""
Plot saved evaluation logs from go2_deploy.py.
Usage:
  python plot_logs.py [--log-dir DIR] [--prefix PREFIX] [--out FILE]
  If PREFIX is not given, uses the latest run in log-dir.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

# Obs layout: gyro(3), gravity(3), command(3), joint_angles(12), joint_vel(12), action(12)
DT = 0.02  # 50 Hz


def find_latest_prefix(log_dir):
    """Return the eval_* prefix for the most recent run in log_dir."""
    pattern = os.path.join(log_dir, "eval_*_*_obs.npy")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No eval logs found in {log_dir}")
    # Sort by mtime, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    path = files[0]
    # path is like .../eval_20260211_022952_obs.npy -> prefix = eval_20260211_022952
    return os.path.basename(path).replace("_obs.npy", "")


def load_run(log_dir, prefix):
    """Load all arrays for one run. Returns dict with obs, velocity_cmd, action, motor_targets."""
    base = os.path.join(log_dir, prefix)
    return {
        "obs": np.load(f"{base}_obs.npy"),
        "velocity_cmd": np.load(f"{base}_velocity_cmd.npy"),
        "action": np.load(f"{base}_action.npy"),
        "motor_targets": np.load(f"{base}_motor_targets.npy"),
    }


def plot_run(data, out_path=None):
    T = data["obs"].shape[0]
    t = np.arange(T) * DT

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle("Deployment logs", fontsize=12)

    # Velocity command
    ax = axes[0, 0]
    v = data["velocity_cmd"]
    ax.plot(t, v[:, 0], label="vx")
    ax.plot(t, v[:, 1], label="vy")
    ax.plot(t, v[:, 2], label="vyaw")
    ax.set_ylabel("velocity cmd")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity command")

    # Policy action (joint deltas)
    ax = axes[0, 1]
    for i in range(data["action"].shape[1]):
        ax.plot(t, data["action"][:, i], alpha=0.7, label=f"j{i}")
    ax.set_ylabel("action")
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Policy action (joint delta)")

    # Motor targets
    ax = axes[1, 0]
    for i in range(data["motor_targets"].shape[1]):
        ax.plot(t, data["motor_targets"][:, i], alpha=0.7, label=f"j{i}")
    ax.set_ylabel("rad")
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Motor targets")

    # Obs: gyro, gravity, command
    ax = axes[1, 1]
    obs = data["obs"]
    ax.plot(t, obs[:, 0:3], label=["gx", "gy", "gz"])
    ax.plot(t, obs[:, 3:6], label=["grx", "gry", "grz"], alpha=0.8)
    ax.plot(t, obs[:, 6:9], label=["cx", "cy", "cyaw"], alpha=0.8)
    ax.set_ylabel("obs (scaled)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Obs: gyro, gravity, command")

    # Obs: joint angles (scaled)
    ax = axes[2, 0]
    for i in range(12):
        ax.plot(t, obs[:, 9 + i], alpha=0.7)
    ax.set_ylabel("joint pos (scaled)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Obs: joint angles")

    # Obs: joint velocities (scaled)
    ax = axes[2, 1]
    for i in range(12):
        ax.plot(t, obs[:, 21 + i], alpha=0.7)
    ax.set_ylabel("joint vel (scaled)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Obs: joint velocities")

    # Obs: prev action
    ax = axes[3, 0]
    for i in range(12):
        ax.plot(t, obs[:, 33 + i], alpha=0.7)
    ax.set_ylabel("prev action")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Obs: previous action")

    axes[3, 1].set_visible(False)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot go2 deployment logs")
    parser.add_argument("--log-dir", default="logs", help="Directory containing eval_* .npy files")
    parser.add_argument("--prefix", default=None, help="Run prefix (e.g. eval_20260211_022952). Default: latest")
    parser.add_argument("--out", "-o", default=None, help="Output figure path (e.g. plot.png)")
    args = parser.parse_args()

    prefix = args.prefix or find_latest_prefix(args.log_dir)
    print(f"Loading run: {prefix}")
    data = load_run(args.log_dir, prefix)
    print(f"  Steps: {data['obs'].shape[0]}, time: {data['obs'].shape[0] * DT:.2f} s")
    plot_run(data, out_path=args.out)


if __name__ == "__main__":
    main()
