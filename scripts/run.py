from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.common_utils import TeeStdoutStderr


def compute_num_envs(num_base_envs: int, num_action_perturbations: int) -> int:
    """Compute total number of environments for AFRL.

    Formula: num_base_envs * (num_action_perturbations + 1)
    """
    return num_base_envs * (num_action_perturbations + 1)


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)
OmegaConf.register_new_resolver(
    "train_or_eval", lambda train: "train" if train else "eval"
)


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
    # Get Hydra's output directory and log file path
    hydra_cfg = HydraConfig.get()
    
    output_dir = Path(hydra_cfg.runtime.output_dir)
    log_file_path = output_dir / "run.log"

    # Redirect stdout/stderr to both console and log file
    with TeeStdoutStderr(log_file_path):
        # Resolve all interpolations in the config (resolves ${...} references)
        OmegaConf.resolve(cfg)

        # If not training, merge play config into task config
        if not cfg.train:
            cfg.task.config = OmegaConf.merge(cfg.task.config, cfg.task.play)
            cfg.num_envs = cfg.task.play.num_envs
            cfg.agent.config.num_envs = cfg.task.play.num_envs

        runner = make_runner(cfg)

        runner.run({"train": cfg.train, "play": not cfg.train, "checkpoint": cfg.checkpoint})


if __name__ == "__main__":
    main()
