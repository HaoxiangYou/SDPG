import hydra
from omegaconf import DictConfig, OmegaConf


def compute_num_envs(num_base_envs: int, num_action_perturbations: int) -> int:
    """Compute total number of environments for AFRl.

    Formula: num_base_envs * (2 * num_action_perturbations + 1)
    """
    return num_base_envs * (2 * num_action_perturbations + 1)


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)


@hydra.main(version_base=None, config_path="../cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that runs with Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    # Resolve all interpolations in the config (resolves ${...} references)
    OmegaConf.resolve(cfg)

    # Add timestamp to logdir unless no_timestamp is True in config
    # if not cfg.get("no_timestamp", False):
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     cfg.logdir = f"{cfg.logdir}/{timestamp}"


if __name__ == "__main__":
    main()
