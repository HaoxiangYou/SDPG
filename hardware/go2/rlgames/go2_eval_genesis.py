import importlib
import os
import time

import genesis as gs
import onnxruntime as ort
import torch

from envs.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase

env_name = "go2"
num_envs = 1
device = "cpu"
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
    "use_terrain": False,
    "terrain_cfg": {
        "n_subterrains": [2, 2],
        "horizontal_scale": 0.25,
        "vertical_scale": 0.005,
        "subterrain_size": [6.0, 6.0],
        "subterrain_types": [
            ["flat_terrain", "random_uniform_terrain"],
            ["pyramid_sloped_terrain", "discrete_obstacles_terrain"],
        ],
    },
}

env_kwargs = {"show_viewer": True, "randomize_init": False, "domain_rand_options": domain_rand_options}


def main():
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_checkpoints/go2_walking.onnx")

    ort_model = ort.InferenceSession(onnx_path)

    # outputs = ort_model.run(
    #     None,
    #     {"obs": np.zeros((1, 45)).astype(np.float32)},
    # )
    # print(outputs)

    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    obs, _ = env.reset()
    while True:
        now = time.time()
        actions = ort_model.run(
            None,
            {"obs": obs["observations"].detach().numpy()},
        )

        print(time.time() - now)
        # obs dof expect FR FL RR RL
        mu = actions[0].squeeze()
        obs, rew, terminated, truncated, info = env.step(torch.tensor(mu), auto_reset=False)
        # import ipdb; ipdb.set_trace()
        time.sleep(0.02)

        if terminated or truncated:
            break

    return


if __name__ == "__main__":
    main()
