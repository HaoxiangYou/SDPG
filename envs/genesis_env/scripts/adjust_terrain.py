import importlib

# from typing import Any, Dict, Tuple
import genesis as gs
import torch

from envs.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase

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
}

env_kwargs = {
    "show_viewer": True,
    "randomize_init": True,
    "terrain_args": terrain_args,
    "debug": True,
    "sensors_args": sensors_args,
}


def main():
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    env.reset()

    for _ in range(1000):
        env.step(torch.randn(num_envs, env.num_actions))


if __name__ == "__main__":
    main()
