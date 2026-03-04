import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main(path: str) -> None:
    """Load obs_buf.npy and plot joint trajectories in subplots.

    Assumes obs_buf has shape [T, D] and that the last 12 columns are the
    motor joint positions in Genesis ordering (FR, FL, RR, RL).
    """
    obs_path = os.path.abspath(path)
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"obs buffer not found: {obs_path}")

    obs_buf = np.load(obs_path)
    if obs_buf.ndim != 2 or obs_buf.shape[1] < 12:
        raise ValueError(f"Expected obs_buf with shape [T, >=12], got {obs_buf.shape}")

    T = obs_buf.shape[0]
    time = np.arange(T)

    # Take last 12 dims as joints
    joints = obs_buf[:, -12:]

    joint_names = [
        "FR_abd",
        "FR_hip",
        "FR_knee",
        "FL_abd",
        "FL_hip",
        "FL_knee",
        "RR_abd",
        "RR_hip",
        "RR_knee",
        "RL_abd",
        "RL_hip",
        "RL_knee",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
    axes = axes.reshape(-1)

    for i in range(12):
        ax = axes[i]
        ax.plot(time, joints[:, i])
        ax.set_ylabel(joint_names[i])
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Joint trajectories from {os.path.basename(obs_path)}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="go2_hardware/rsl/obs_buf.npy",
        help="Path to obs_buf.npy (default: obs_buf.npy in current directory)",
    )
    args = parser.parse_args()
    main(args.path)
