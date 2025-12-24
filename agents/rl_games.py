from omegaconf import DictConfig, OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from envs.base_env import BaseEnv
from utils.common_utils import make_envs


class RlGamesGpuEnv(vecenv.IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
        """
        self.env: BaseEnv = env_configurations.configurations[config_name]["env_creator"](**kwargs)

        self.full_state = {}

        self.full_state["obs"], _ = self.env.reset()

    def step(self, actions):
        self.full_state["obs"], reward, terminated, truncated, info = self.env.step(actions, auto_reset=True)

        is_done = terminated | truncated

        return self.full_state["obs"], reward, is_done, info

    def reset(self):
        self.full_state["obs"], _ = self.env.reset()

        return self.full_state["obs"]

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space

        print(info["action_space"], info["observation_space"])

        return info


def make_runner(config: DictConfig):
    # Following the IsaacLab implementation:
    # First initialize the environment and use wrapper to fit RL-Games runner.
    env = make_envs(config)
    vecenv.register(
        "AFRLEnv", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("afrl_env", {"env_creator": lambda **kwargs: env, "vecenv_type": "AFRLEnv"})

    agent_config = OmegaConf.to_container(config.agent.config.rl_games, resolve=True)
    # Overwrite attributes based on parents config.
    agent_config["seed"] = config.seed

    runner = Runner()
    runner.load({"params": agent_config})
    runner.reset()

    return runner
