import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

import envs


def compute_num_envs(num_base_envs: int, num_action_perturbations: int) -> int:
    """Compute total number of environments for AFRl.

    Formula: num_base_envs * (num_action_perturbations + 1)
    """
    return num_base_envs * (num_action_perturbations + 1)


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)


def make_envs(config: DictConfig):
    """Create environment based on task suite."""
    task_suite = config.task.get("suite")
    TaskSuite = getattr(envs, task_suite + "_env")
    return TaskSuite.make_envs(config)


@hydra.main(version_base=None, config_path="../cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that runs with Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    # Resolve all interpolations in the config (resolves ${...} references)
    OmegaConf.resolve(cfg)

    env = make_envs(cfg)
    env.reset()


if __name__ == "__main__":
    main()
