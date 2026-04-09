import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner, _override_sigma, _restore

from envs.base_env import BaseEnv
from utils.common_utils import make_envs
from utils.statistic_utils import AverageMeter
from utils.tensor_utils import moveaxis_dict, stack_dict_list


class RlGamesGpuEnv(vecenv.IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    def __init__(self, config_name: str, num_actors: int, input_keys: Dict[str, List[Dict[str, str]]], **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
            input_keys: Dictionary with 'actor' and 'critic' keys containing lists of input key dictionaries.
        """
        self.env: BaseEnv = env_configurations.configurations[config_name]["env_creator"](**kwargs)

        # Extract input key names from input_keys config
        # input_keys structure: {actor: [{name: "observations"}], critic: [{name: "privileged_observations"}]}
        self.actor_input_key = input_keys["actor"][0]["name"]
        self.critic_input_key = input_keys["critic"][0]["name"]

        self.env.reset()

    def step(self, actions):
        observations, reward, terminated, truncated, info = self.env.step(actions, auto_reset=True)

        is_done = terminated | truncated

        info = {**info, "time_outs": truncated.float()}

        obs_dict = {
            "obs": observations[self.actor_input_key],
            "states": observations[self.critic_input_key],  # value function uses this
        }
        return obs_dict, reward, is_done, info

    def reset(self):
        observations, _ = self.env.reset()
        obs_dict = {
            "obs": observations[self.actor_input_key],
            "states": observations[self.critic_input_key],
        }
        return obs_dict

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        return getattr(self, "num_agents", 1)

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        obs_space = self.env.observation_space
        info["observation_space"] = obs_space.get("observations", obs_space["privileged_observations"])
        info["state_space"] = obs_space["privileged_observations"]
        info["use_global_observations"] = True
        return info


class WandbAlgoObserver(IsaacAlgoObserver):
    """RL-Games observer that mirrors key PPO metrics into WandB."""

    def __init__(self, enabled: bool = False):
        super().__init__()
        self._enabled = enabled

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().mean().item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(np.mean(value))
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _mean_ep_info(self, key: str):
        if not self.ep_infos:
            return None
        values = []
        for ep_info in self.ep_infos:
            if key not in ep_info:
                continue
            value = ep_info[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([value], device=self.algo.device)
            elif value.ndim == 0:
                value = value.unsqueeze(0)
            values.append(value.to(self.algo.device))
        if not values:
            return None
        return float(torch.cat(values).mean().item())

    def after_print_stats(self, frame, epoch_num, total_time):
        rewards = None
        episode_lengths = None

        try:
            algo = self.algo
            if getattr(algo, "game_rewards", None) is not None and algo.game_rewards.current_size > 0:
                raw = algo.game_rewards.get_mean()
                rewards = self._to_float(raw)
            if getattr(algo, "game_lengths", None) is not None and algo.game_lengths.current_size > 0:
                raw = algo.game_lengths.get_mean()
                episode_lengths = self._to_float(raw)
        except Exception as e:
            print(f"[WandbAlgoObserver] error reading algo stats: {e}")

        if rewards is None and hasattr(self, "mean_scores") and self.mean_scores.current_size > 0:
            rewards = self._to_float(self.mean_scores.get_mean())
        if episode_lengths is None:
            for key in ("length", "episode_length", "episode_lengths", "len"):
                episode_lengths = self._mean_ep_info(key)
                if episode_lengths is not None:
                    break
        if rewards is None:
            for key in ("reward", "rewards", "score"):
                rewards = self._mean_ep_info(key)
                if rewards is not None:
                    break

        super().after_print_stats(frame, epoch_num, total_time)

        if not self._enabled:
            return
        if wandb.run is None:
            print(f"[WandbAlgoObserver] wandb.run is None at epoch {epoch_num}, skipping")
            return
        wandb_metrics = {
            "env_step": int(frame),
            "time": float(total_time),
        }
        if rewards is not None:
            wandb_metrics["rewards"] = rewards
        if episode_lengths is not None:
            wandb_metrics["episode_lengths"] = episode_lengths
        wandb.log(wandb_metrics, step=epoch_num)


def _resolve_wandb_name(config: DictConfig) -> str | None:
    if not getattr(config, "wandb", None):
        return None
    base_name = config.wandb.get("name")
    if base_name is None:
        run_name = getattr(config, "run_name", None)
        if run_name:
            base_name = run_name
    if base_name is None and HydraConfig.initialized():
        base_name = HydraConfig.get().runtime.output_dir.split("/")[-1]
    if base_name is None:
        return None
    base_name = str(base_name).strip()
    return base_name or None


def _init_wandb(config: DictConfig) -> bool:
    if not getattr(config, "wandb", None) or not config.wandb.get("enable", False):
        return False
    if wandb.run is not None:
        return False

    wandb_config = config.wandb
    wandb_kwargs = {
        "project": wandb_config.get("project", "ppo"),
        "entity": wandb_config.get("entity"),
        "group": wandb_config.get("group"),
        "job_type": wandb_config.get("job_type"),
        "name": _resolve_wandb_name(config),
        "tags": wandb_config.get("tags", []),
        "notes": wandb_config.get("notes"),
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)
    wandb.define_metric("env_step")
    wandb.define_metric("*", step_metric="env_step")
    if wandb_config.get("log_config", True):
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.config.update(config_dict)
    print("Wandb logging enabled")
    return True


class RlGamesRunnerWrapper:
    def __init__(self, runner: Runner, own_wandb_run: bool = False, log_dir: str | None = None):
        self._runner = runner
        self._own_wandb_run = own_wandb_run
        self._log_dir = log_dir

    def run(self, args: Dict[str, Any]):
        try:
            if args.get("play"):
                self._run_play(args)
            else:
                return self._runner.run(args)
        finally:
            if self._own_wandb_run and wandb.run is not None:
                wandb.finish()

    @torch.no_grad()
    def _run_play(self, args: Dict[str, Any]):
        """Evaluate the policy and save trajectory (mirrors AFRLRunner.evaluate_policy)."""
        print("Started to play")
        player = self._runner.create_player()
        _restore(player, args)
        _override_sigma(player, args)

        genesis_env: BaseEnv = player.env.env
        device = player.device
        num_envs = genesis_env.num_envs
        max_steps = genesis_env.episode_length

        ep_length = torch.zeros(num_envs, device=device)
        ep_length_meter = AverageMeter(1, 100).to(device)
        ep_reward = torch.zeros(num_envs, device=device)
        ep_reward_meter = AverageMeter(1, 100).to(device)

        states_history = []

        obses = player.env_reset(player.env)
        player.get_batch_size(obses, 1)
        states_history.append(genesis_env.get_states())

        for _ in range(max_steps):
            action = player.get_action(obses, is_deterministic=True)
            obses, r, done, info = player.env_step(player.env, action)
            states_history.append(genesis_env.get_states())

            ep_length += 1
            ep_reward += r
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                ep_length_meter.update(ep_length[done_ids])
                ep_reward_meter.update(ep_reward[done_ids])
                ep_length[done_ids] = 0.0
                ep_reward[done_ids] = 0.0

        print(
            f"Episode length: {ep_length_meter.get_mean().item():.1f}, "
            f"Episode reward: {ep_reward_meter.get_mean().item():.2f}"
        )

        if self._log_dir:
            eval_dir = os.path.join(self._log_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            save_path = os.path.join(eval_dir, "trajectory.pt")
            states_history = stack_dict_list(states_history)
            states_history = moveaxis_dict(states_history, source=0, destination=1)
            torch.save(states_history, save_path)
            print(f"Trajectory saved to {save_path}")


def make_runner(
    config: DictConfig,
    env: Optional[BaseEnv] = None,
    algo_observer: Optional[IsaacAlgoObserver] = None,
):
    """Create PPO runner. If env is provided (e.g. teacher's env from teacher_student), use it; else create from config."""
    if env is None:
        env = make_envs(config)

    # Extract input_keys from config
    input_keys = OmegaConf.to_container(config.agent.config.input_keys, resolve=True)

    # NOTE: both actor and critic current only support one input key
    assert (
        len(input_keys["actor"]) == 1 and len(input_keys["critic"]) == 1
    ), "Both actor and critic current only support one input key"

    vecenv.register(
        "AFRLEnv",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
            config_name, num_actors, input_keys=input_keys, **kwargs
        ),
    )
    env_configurations.register("afrl_env", {"env_creator": lambda **kwargs: env, "vecenv_type": "AFRLEnv"})

    agent_config = OmegaConf.to_container(config.agent.config.rl_games, resolve=True)
    # Overwrite attributes based on parents config.
    agent_config["seed"] = config.seed
    agent_config["config"]["num_actors"] = config.agent.config.num_envs

    # Output directory: use config.log_dir if set (e.g. teacher_student teacher subdir), else Hydra run dir
    output_dir = getattr(config, "log_dir", None)
    if not output_dir and HydraConfig.get() is not None:
        output_dir = HydraConfig.get().run.dir
    if output_dir:
        agent_config["config"]["log_path"] = output_dir
        agent_config["config"]["train_dir"] = output_dir
        print(f"Output directory: {output_dir}")
        agent_config["config"]["full_experiment_name"] = "training_logs"

    own_wandb_run = _init_wandb(config)
    if algo_observer is None:
        algo_observer = WandbAlgoObserver(enabled=bool(getattr(config, "wandb", None) and config.wandb.get("enable", False)))

    runner = Runner(algo_observer=algo_observer)
    runner.load({"params": agent_config})
    runner.reset()

    return RlGamesRunnerWrapper(runner, own_wandb_run=own_wandb_run, log_dir=output_dir)
