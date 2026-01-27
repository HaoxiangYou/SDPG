import importlib
import os
import time

import genesis as gs
import numpy as np
import onnxruntime as ort
import torch

from envs.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase

env_name = "go2"
num_envs = 1
device = "cpu"
sim_options = gs.options.SimOptions(dt=0.02, substeps=2)
env_kwargs = {"show_viewer": True, "randomize_init": True}


def main():
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2.onnx")

    ort_model = ort.InferenceSession(onnx_path)

    outputs = ort_model.run(
        None,
        {"obs": np.zeros((1, 45)).astype(np.float32)},
    )
    print(outputs)

    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    obs, _ = env.reset()

    while True:
        now = time.time()
        actions = ort_model.run(
            None,
            {"obs": obs["privileged_observations"].detach().numpy()},
        )
        print(time.time() - now)
        obs, rew, terminated, truncated, info = env.step(torch.tensor(actions[0]), auto_reset=False)
        time.sleep(0.02)

        if terminated or truncated:
            break

    return


if __name__ == "__main__":
    main()
