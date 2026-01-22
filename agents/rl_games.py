from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from envs.base_env import BaseEnv
from utils.common_utils import make_envs


class RlGamesGpuEnv(vecenv.IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    # TODO: currently only support for full state observation

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
        """
        self.env: BaseEnv = env_configurations.configurations[config_name]["env_creator"](**kwargs)

        self.env.reset()

    def step(self, actions):
        observations, reward, terminated, truncated, info = self.env.step(actions, auto_reset=True)

        is_done = terminated | truncated

        return {"obs": observations["privileged_observations"]}, reward, is_done, info

    def reset(self):
        observations, _ = self.env.reset()

        return {"obs": observations["privileged_observations"]}

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space["privileged_observations"]

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
    agent_config["config"]["num_actors"] = config.agent.config.num_envs
    agent_config["config"]["player"]["num_actors"] = config.agent.config.num_envs
    agent_config["config"]["player"]["games_num"] = config.agent.config.num_envs

    # Get Hydra's output directory - this uses the already-resolved path with timestamp
    # from when Hydra initialized, not generating a new timestamp
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        # hydra_cfg.run.dir contains the already-resolved path (with timestamp from initialization)
        output_dir = hydra_cfg.run.dir
        agent_config["config"]["log_path"] = output_dir
        agent_config["config"]["train_dir"] = output_dir
        print(f"Output directory: {output_dir}")
        agent_config["config"]["full_experiment_name"] = "training_logs"

    runner = Runner()
    runner.load({"params": agent_config})
    runner.reset()

    return runner
