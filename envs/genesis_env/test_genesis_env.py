import importlib

import genesis as gs
import torch
from genesis_env import GenesisEnv

from utils.common_utils import snakecase_to_pascalcase
from utils.tensor_utils import all_dict_values_true, check_groups_same, duplicate_entries, select_entries

env_name = "humanoid"
num_envs = 4
device = "cuda"
sim_options = gs.options.SimOptions(dt=1e-1, substeps=1)
env_kwargs = {"render": False, "show_viewer": False, "randomize_init": True}


def main():
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    env.reset()
    for i in range(10):
        obs, rew, terminated, truncated, info = env.step(
            torch.ones(num_envs, env.num_actions).to(device), auto_reset=False
        )
    env._reset_idx([0])
    obs, rew, terminated, truncated, info = env.step(torch.ones(num_envs, env.num_actions).to(device), auto_reset=False)
    import pdb

    pdb.set_trace()

    for i in range(32):
        actions = torch.randn(num_envs, env.num_actions).to(device)
        env.step(actions, auto_reset=False)

    # for each group of 2 envs, set the states to the same
    states = env.get_states()
    states = duplicate_entries(select_entries(states, range(0, num_envs, 2)), 2)
    env.set_states(states)

    for i in range(32):
        actions = torch.randn(num_envs // 2, env.num_actions).repeat_interleave(2, dim=0).to(device)
        obs, rew, terminated, truncated, info = env.step(actions, auto_reset=False)

    # make sure the observations, rewards, and states are the same for each group of 2 envs
    assert check_groups_same(obs, 2)
    assert check_groups_same(rew, 2)
    assert all_dict_values_true(check_groups_same(env.get_states(), 2))


if __name__ == "__main__":
    main()
