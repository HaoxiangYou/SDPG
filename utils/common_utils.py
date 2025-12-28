from omegaconf import DictConfig

import envs
from envs.base_env import BaseEnv


def snakecase_to_pascalcase(s: str) -> str:
    components = s.split("_")
    return "".join(word.capitalize() for word in components)


def make_envs(config: DictConfig) -> BaseEnv:
    """Create environment based on task backend."""
    backend = config.task.get("backend")
    TaskSuite = getattr(envs, backend + "_env")
    return TaskSuite.make_envs(config)
