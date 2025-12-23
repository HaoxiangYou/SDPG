import hydra
from omegaconf import DictConfig, OmegaConf


def compute_num_envs(num_base_envs: int, num_action_perturbations: int) -> int:
    """Compute total number of environments for AFRl.

    Formula: num_base_envs * (num_action_perturbations + 1)
    """
    return num_base_envs * (num_action_perturbations + 1)


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)


def make_runner(config: DictConfig):
    """Create agent"""
    agent_name = config.agent.get("name")
    match agent_name:
        case _:
            raise ValueError(f"Invalid agent name: {agent_name}")


@hydra.main(version_base=None, config_path="../cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that runs with Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    # Resolve all interpolations in the config (resolves ${...} references)
    OmegaConf.resolve(cfg)

    # runner = make_runner(cfg)


if __name__ == "__main__":
    main()
