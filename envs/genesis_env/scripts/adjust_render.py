"""
This script is used to test and adjust the rendering settings of the Genesis environment.
"""

import importlib

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np

from envs.genesis_env.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase

env_name = "humanoid"
num_envs = 4
device = "cuda"
sim_options = gs.options.SimOptions(dt=1e-2, substeps=1)
env_kwargs = {
    "show_viewer": False,
    "randomize_init": True,
    "vis_obs": True,
    "sensors_args": {
        "envs_idx": None,
        "camera": {
            "res": (84, 84),
            "pos": (0.0, -2.0, -0.5),
            "lookat": (0.0, 0.0, -0.5),
            "fov": 60.0,
            "lights": {
                "pos": (0.0, 0.0, 2.0),
                "dir": (0.0, 0.0, -1.0),
                "cutoff": 100,
                "directional": True,
                "castshadow": False,
            },
        },
    },
}


def main():
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    env.reset()
    imgs = env.render().cpu().numpy()

    n_images = imgs.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
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
    main()
