import hydra
from omegaconf import DictConfig, OmegaConf


def compute_num_envs(num_base_envs: int, num_action_perturbations: int) -> int:
    """Compute total number of environments for AFRL.

    Formula: num_base_envs * (num_action_perturbations + 1)
    """
    return num_base_envs * (num_action_perturbations + 1)


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)


def make_runner(config: DictConfig):
    """Create agent"""
    agent_name = config.agent.get("name")
    match agent_name:
        case "ppo" | "sac":
            from agents.rl_games import make_runner

            return make_runner(config)
        case "afrl":
            from agents.afrl import make_runner

            return make_runner(config)
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

    # If not training, overwrite env config with the play mode
    if not cfg.train:
        cfg.task.config = cfg.task.play
        cfg.num_envs = cfg.task.play.num_envs
        cfg.agent.config.num_envs = cfg.task.play.num_envs

    runner = make_runner(cfg)

    runner.run({"train": cfg.train, "play": not cfg.train, "checkpoint": cfg.checkpoint})


if __name__ == "__main__":
    main()
