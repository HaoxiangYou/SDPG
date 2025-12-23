import importlib
import os
from pathlib import Path

import torch
from rewarped_env import RewarpedEnv

from utils.common_utils import snakecase_to_pascalcase
from utils.tensor_utils import check_groups_same, duplicate_entries, select_entries

env_suite = "dflex"
env_name = "humanoid"
num_envs = 4
device = "cuda"
env_kwargs = {"randomize": True, "no_grad": True, "render": True, "no_env_offset": False, "render_mode": "usd"}
os.environ["WARP_RENDER_DIR"] = str(
    Path(__file__).parent.parent.parent / "logs" / "test_envs" / "rewarped_env" / env_name
)


def main():
    ENV = importlib.import_module(f"rewarped.envs.{env_suite}.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env = RewarpedEnv(env_fn(num_envs=num_envs, device=device, seed=0, **env_kwargs))

    env.reset()

    # make the robot move randomly
    for i in range(32):
        actions = torch.randn(num_envs, env.num_actions).to(device)
        obs, rew, terminated, truncated, info = env.step(actions, auto_reset=False)

    # for each group of 2 envs, set the states to the same
    states = env.get_states()
    states = duplicate_entries(select_entries(states, range(0, num_envs, 2)), 2)
    env.set_states(states)

    for i in range(32):
        actions = torch.randn(num_envs // 2, env.num_actions).repeat_interleave(2, dim=0).to(device)
        obs, rew, terminated, truncated, info = env.step(actions, auto_reset=False)

    env.save_video()

    # make sure the observations, rewards, and states are the same for each group of 2 envs
    assert check_groups_same(obs, 2, rtol=0.05)
    assert check_groups_same(rew, 2, rtol=0.05)
    assert check_groups_same(env.get_states(), 2, rtol=0.05)


if __name__ == "__main__":
    main()
