from pathlib import Path
import shutil

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.common_utils import TeeStdoutStderr


def compute_num_envs(num_base_envs: int, num_action_perturbations: int) -> int:
    """Compute total number of environments for SDPG.

    Formula: num_base_envs * (num_action_perturbations + 1)
    """
    return num_base_envs * (num_action_perturbations + 1)


def run_name_or_timestamp(run_name: str | None, timestamp: str) -> str:
    """Use an explicit run name when provided, otherwise fall back to the timestamp."""
    if run_name is None:
        return timestamp
    run_name = str(run_name).strip()
    return run_name or timestamp


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("compute_num_envs", compute_num_envs)
OmegaConf.register_new_resolver(
    "train_or_eval", lambda train: "train" if train else "eval"
)
OmegaConf.register_new_resolver("run_name_or_timestamp", run_name_or_timestamp)


def copy_env_code(output_dir: Path, task_backend: str, task_name: str) -> None:
    """Save a copy of the environment Python script under output_dir/codes/env for reproducibility."""
    env_script_path = (
        Path(__file__).resolve().parent.parent
        / "envs"
        / f"{task_backend}_env"
        / f"{task_name}.py"
    )
    if not env_script_path.is_file():
        return
    codes_env_dir = output_dir / "codes" / "env"
    codes_env_dir.mkdir(parents=True, exist_ok=True)
    dst_path = codes_env_dir / env_script_path.name
    if not dst_path.exists():
        shutil.copy2(env_script_path, dst_path)


def make_runner(config: DictConfig):
    """Create agent"""
    agent_name = config.agent.get("name")
    match agent_name:
        case "ppo" | "sac":
            from agents.rl_games import make_runner

            return make_runner(config)
        case "sdpg":
            from agents.sdpg import make_runner

            return make_runner(config)
        case "drqv2":
            from agents.drqv2 import make_runner

            return make_runner(config)
        case "dreamerv3":
            from agents.dreamerv3 import make_runner

            return make_runner(config)
        case "teacher_student":
            from agents.teacher_student import make_runner

            return make_runner(config)
        case _:
            raise ValueError(f"Invalid agent name: {agent_name}")


@hydra.main(version_base=None, config_path="../cfgs", config_name="run")
def main(cfg: DictConfig) -> None:
    """Main function that runs with Hydra config.

    Args:
        cfg: Hydra configuration object.
    """
    # Get Hydra's output directory and log file path
    hydra_cfg = HydraConfig.get()

    output_dir = Path(hydra_cfg.runtime.output_dir)
    log_file_path = output_dir / "run.log"

    task_backend = cfg.task.get("backend", None)
    task_name = cfg.task.get("name", None)
    if task_backend is not None and task_name is not None:
        copy_env_code(output_dir, task_backend, task_name)

    # Redirect stdout/stderr to both console and log file
    with TeeStdoutStderr(log_file_path):
        # Resolve all interpolations in the config (resolves ${...} references)
        OmegaConf.resolve(cfg)

        # If not training, merge play config into task config
        if not cfg.train:
            cfg.task.config = OmegaConf.merge(cfg.task.config, cfg.task.play)
            cfg.num_envs = cfg.task.play.num_envs
            cfg.agent.config.num_envs = cfg.task.play.num_envs

        # Keep the timestamp-based logdir by default, but allow a single override
        # to rename both the Hydra output directory and WandB run.
        if (
            getattr(cfg, "run_name", None)
            and getattr(cfg, "wandb", None)
            and cfg.wandb.get("enable", False)
            and not cfg.wandb.get("name")
        ):
            cfg.wandb.name = cfg.run_name

        runner = make_runner(cfg)

        runner.run({"train": cfg.train, "play": not cfg.train, "checkpoint": cfg.checkpoint})


if __name__ == "__main__":
    main()
