"""
This script is used to test and adjust the rendering settings of the Genesis environment.
"""

import argparse
import importlib

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from envs.genesis_env.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase
from utils.tensor_utils import enumerate_states

env_name = "walker_hurtle"
num_envs = 4
device = "cuda"
sim_options = gs.options.SimOptions(dt=0.01, substeps=1)

terrain_args = {
    "mesh_type": "heightfield",
    "curriculum": True,
    "selected": False,
    "border_size": 0.0,
    "border_height": 0.0,
    "terrain_length": 5.0,
    "terrain_width": 2.0,
    "platform_size": 0.0,
    "num_rows": 5,  # number of terrain rows (levels)
    "num_cols": 1,  # number of terrain cols (types)
    "num_subterrains": 5,
    "horizontal_scale": 0.05,  # [m] distance between height samples in x and y direction
    "vertical_scale": 0.005,  # [m] distance between height samples in z direction
    "static_friction": 1.0,  # TODO currently not implemented, coefficient of static friction of the terrain
    "dynamic_friction": 1.0,  # TODO currently not implemented, coefficient of dynamic friction of the terrain
    "restitution": 0.0,  # TODO currently not implemented, coefficient of restitution of the terrain
    "max_init_terrain_level": 1,  # starting curriculum level
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, hurtle, stepping stones, gap, pit]
    "terrain_proportions": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    # Optional: hurtle terrain (offset + scale*difficulty).
    "hurtle_args": {
        "num_stones": 2,
        "x_range": [2.4, 2.5],  # range of x position between the stone
        "stone_len": {"offset": 0.2, "scale": 0.0},
        "hurtle_height_range": {
            "low": {"offset": 0.20, "scale": 1.0},
            "high": {"offset": 0.30, "scale": 1.5},
        },
    },
}

sensors_args = {
    "heightfield": {
        "res": 0.2,
        "ahead": 5.0,
        "backward": 2.0,
    },
    "camera": {
        "type": "rgb",
        "res": (84, 84),
        "pos": (0.2, 0.0, -0.1),
        "lookat": (0.8, 0.0, -0.3),
        "fov": 80.0,
        "lights": {
            "pos": (8.0, 0.0, 2.0),
            "intensity": 0.8,
            "color": (1.0, 1.0, 1.0),
            "dir": (0.0, 0.0, -1.0),
            "cutoff": 100,
            "directional": True,
            "castshadow": False,
        },
        "near": 0.01,
        "far": 5.0,
    },
}

env_kwargs = {
    "show_viewer": True,
    "randomize_init": False,  # Set to False when loading states
    "vis_obs": True,
    "nominal_env_ids": None,
    "sensors_args": sensors_args,
    "terrain_args": terrain_args,
    "debug": True,
}


def main(traj_path: str = None):
    # Create environment
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))

    if traj_path is not None:
        # Load state history (nested dict with tensors (batch, time, ...))
        states = torch.load(traj_path, weights_only=False)
        env: GenesisEnv = env_fn(num_envs=1, device=device, seed=0, sim_options=sim_options, **env_kwargs)

        frames = []
        for batch_idx, time_idx, state in enumerate_states(states):
            if batch_idx != 0:
                continue  # only render trajectory for first batch
            env.set_states(state)
            env._scene.step()
            img = env.render().cpu().numpy()[0]
            frames.append(img)

        # Display video as animation in matplotlib
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        im = ax.imshow(frames[0])
        title = ax.set_title(f"Frame 0/{len(frames)}")

        def update(frame_idx):
            im.set_array(frames[frame_idx])
            title.set_text(f"Frame {frame_idx}/{len(frames)}")
            return [im, title]

        # Keep animation object alive by assigning to variable
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / 30, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()

        # Keep reference to prevent garbage collection
        return anim
    else:
        # No trajectory file, just reset
        env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)
        env.reset()

        # Render
        imgs = env.render().cpu().numpy()

        n_images = imgs.shape[0]
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes = axes.flatten()

        # Display each image
        for i in range(n_images):
            axes[i].imshow(imgs[i])
            axes[i].axis("off")
            axes[i].set_title(f"Env {i}", fontsize=10)

        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, default=None, help="Path to the .pt file containing state history")
    args = parser.parse_args()

    main(args.traj_path)
