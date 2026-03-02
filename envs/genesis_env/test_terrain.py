import importlib

# from typing import Any, Dict, Tuple
import genesis as gs

from envs.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase

env_name = "go2_terrain"
num_envs = 20
device = "cuda"
sim_options = gs.options.SimOptions(dt=0.02, substeps=4)

domain_rand_options = {
    "randomize_friction": False,
    "friction_range": [0.2, 1.5],
    "randomize_base_mass": False,
    "added_mass_range": [-1.0, 3.0],
    "randomize_com_displacement": False,
    "com_displacement_range": [-0.01, 0.01],
    "randomize_motor_strength": False,
    "motor_strength_range": [0.9, 1.1],
    "randomize_motor_offset": False,
    "motor_offset_range": [-0.02, 0.02],
    "randomize_kp_scale": False,
    "kp_scale_range": [0.8, 1.2],
    "randomize_kd_scale": False,
    "kd_scale_range": [0.8, 1.2],
    "use_terrain": True,
    "terrain_cfg": {
        "mesh_type": "heightfield",
        "curriculum": True,
        "selected": False,
        "obtain_terrain_info_around_feet": True,
        "measure_heights": True,
        "measured_points_x": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],  # 9x9=81
        "measured_points_y": [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        "border_size": 5.0,
        "border_height": 0.5,
        "terrain_length": 8.0,
        "terrain_width": 2.0,
        "platform_size": 1.0,
        "num_rows": 5,  # number of terrain rows (levels)
        "num_cols": 1,  # number of terrain cols (types)
        "num_subterrains": 5,
        "horizontal_scale": 0.05,  # [m] distance between height samples in x and y direction
        "vertical_scale": 0.005,  # [m] distance between height samples in z direction
        "static_friction": 1.0,  # coefficient of static friction of the terrain
        "dynamic_friction": 1.0,  # coefficient of dynamic friction of the terrain
        "restitution": 0.0,  # coefficient of restitution of the terrain
        "max_init_terrain_level": 1,  # starting curriculum level
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, hurtle, stepping stones, gap, pit]
        "terrain_proportions": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    },
}

env_kwargs = {"show_viewer": True, "randomize_init": True, "domain_rand_options": domain_rand_options}


def main():
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    env.reset()


if __name__ == "__main__":
    main()
