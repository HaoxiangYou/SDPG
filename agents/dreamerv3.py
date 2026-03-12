"""
DreamerV3 runner for Genesis pixel environments.
"""

import functools
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import gym
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter

from envs.base_env import BaseEnv
from utils.common_utils import make_envs
from utils.statistic_utils import AverageMeter

os.environ.setdefault("MUJOCO_GL", "egl")

_DREAMER_DIR = Path(__file__).resolve().parent.parent / "externals" / "dreamerv3-torch"
if str(_DREAMER_DIR) not in sys.path:
    sys.path.insert(0, str(_DREAMER_DIR))

import dreamer as _dreamer_mod
import tools as dreamer_tools

Dreamer = _dreamer_mod.Dreamer


class DreamerLogger:
    """Logger with TensorBoard naming aligned to other agents."""

    def __init__(self, logdir: Path, step: int, use_wandb: bool = False):
        self._logdir = Path(logdir)
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._scalars = {}
        self._images = {}
        self._last_step = None
        self._last_time = None
        self._start_time = time.time()
        self._iter = 0
        self._use_wandb = use_wandb
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0.0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / max(duration, 1e-6)

    def _normalize_scalars(self, scalars):
        metrics = {}
        info_scalars = {}
        for name, value in scalars:
            if name == "train_return":
                metrics["rewards"] = value
            elif name == "train_length":
                metrics["episode_lengths"] = value
            elif name == "eval_return":
                metrics["eval/rewards"] = value
            elif name == "eval_length":
                metrics["eval/episode_lengths"] = value
            elif name in {"dataset_size", "train_episodes", "eval_episodes", "fps"}:
                info_scalars[name] = value
            else:
                metrics[name] = value
        return metrics, info_scalars

    def write(self, fps=False, step=None):
        step = self.step if step in (None, False) else step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))

        metrics, info_scalars = self._normalize_scalars(scalars)
        has_non_fps_info = any(key != "fps" for key in info_scalars)
        has_meaningful_log = bool(metrics) or has_non_fps_info or bool(self._images)
        if not has_meaningful_log:
            self._scalars = {}
            self._images = {}
            return

        self._iter += 1
        iter_idx = self._iter
        time_elapse = time.time() - self._start_time

        record = {"iter": iter_idx, "step": step, **metrics}
        record.update({f"info/{key}": value for key, value in info_scalars.items()})
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

        for name, value in metrics.items():
            self._writer.add_scalar(f"{name}/iter", value, iter_idx)
            self._writer.add_scalar(f"{name}/step", value, step)
            self._writer.add_scalar(f"{name}/time", value, time_elapse)
        for key, value in info_scalars.items():
            self._writer.add_scalar(f"info/{key}", value, iter_idx)

        for name, value in self._images.items():
            self._writer.add_image(name, value, step)

        if self._use_wandb:
            wandb_metrics = dict(metrics)
            wandb_metrics["env_step"] = step
            wandb_metrics["time"] = time_elapse
            for key, value in info_scalars.items():
                wandb_metrics[f"info/{key}"] = value
            wandb.log(wandb_metrics, step=iter_idx)

        parts = [f"iter {iter_idx}", f"step {step}"]
        if "rewards" in metrics:
            parts.append(f"ep reward {metrics['rewards']:.2f}")
        if "episode_lengths" in metrics:
            parts.append(f"ep len {metrics['episode_lengths']:.1f}")
        if "fps" in info_scalars:
            parts.append(f"fps {info_scalars['fps']:.1f}")
        if "dataset_size" in info_scalars:
            parts.append(f"buffer {int(info_scalars['dataset_size'])}")
        if "train_episodes" in info_scalars:
            parts.append(f"episode {int(info_scalars['train_episodes'])}")
        print(" | ".join(parts))

        self._writer.flush()
        self._scalars = {}
        self._images = {}

    def offline_scalar(self, name, value, step):
        metrics, info_scalars = self._normalize_scalars([(name, value)])
        for metric_name, metric_value in metrics.items():
            self._writer.add_scalar(f"{metric_name}/step", metric_value, step)
        for key, info_value in info_scalars.items():
            self._writer.add_scalar(f"info/{key}", info_value, step)

class DreamerGenesisVecEnv(gym.Env):
    """Vectorized Genesis wrapper that exposes Dreamer-style image observations."""

    metadata = {}

    def __init__(
        self,
        base_env: BaseEnv,
        img_size: int = 64,
        action_repeat: int = 2,
    ):
        obs_spaces = getattr(base_env.observation_space, "spaces", {})
        assert "RGB" in obs_spaces, "DreamerV3 requires pixel observations (`task.config.vis_obs=True`)."

        self._env = base_env
        self._img_size = int(img_size)
        self._action_repeat = int(action_repeat)
        self._num_envs = int(getattr(base_env, "num_envs", 1))
        self._num_actions = int(base_env.num_actions)
        self.reward_range = [-np.inf, np.inf]

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._img_size, self._img_size, 3),
                    dtype=np.uint8,
                )
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_actions,),
            dtype=np.float32,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def _extract_images(self, obs_dict: dict) -> np.ndarray:
        rgb = obs_dict["RGB"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach()
            rgb = rgb.cpu().numpy()
        else:
            rgb = np.asarray(rgb)

        if rgb.ndim == 3:
            rgb = np.expand_dims(rgb, 0)

        images = []
        for frame in rgb:
            if frame.shape[0] == 9:
                frame = frame[-3:]
            elif frame.shape[0] != 3:
                frame = frame[:3]
            image = np.transpose(frame, (1, 2, 0)).astype(np.uint8)
            if image.shape[:2] != (self._img_size, self._img_size):
                image = cv2.resize(
                    image,
                    (self._img_size, self._img_size),
                    interpolation=cv2.INTER_AREA,
                )
            images.append(image)
        return np.stack(images, axis=0)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids_torch = None
            env_ids_np = np.arange(self._num_envs, dtype=np.int32)
        elif torch.is_tensor(env_ids):
            env_ids_torch = env_ids.to(device=self._env.device, dtype=torch.int32)
            env_ids_np = env_ids.detach().cpu().numpy().astype(np.int32)
        else:
            env_ids_np = np.asarray(env_ids, dtype=np.int32)
            env_ids_torch = torch.as_tensor(env_ids_np, device=self._env.device, dtype=torch.int32)
        obs_dict, _ = self._env.reset(env_ids=env_ids_torch)
        batch = len(env_ids_np)
        return {
            "image": self._extract_images(obs_dict),
            "is_first": np.ones(batch, dtype=bool),
            "is_terminal": np.zeros(batch, dtype=bool),
        }

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self._env.device)
        else:
            action = action.float().to(self._env.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = torch.clamp(action, -1.0, 1.0)

        total_reward = np.zeros(self._num_envs, dtype=np.float32)
        done = np.zeros(self._num_envs, dtype=bool)
        terminated_any = np.zeros(self._num_envs, dtype=bool)
        truncated_any = np.zeros(self._num_envs, dtype=bool)
        final_image = None
        terminal_image = np.zeros(
            (self._num_envs, self._img_size, self._img_size, 3), dtype=np.uint8
        )
        has_terminal_image = np.zeros(self._num_envs, dtype=bool)
        for _ in range(self._action_repeat):
            step_action = action
            if done.any():
                # Genesis steps the full batched scene at once, so keep finished envs
                # inert while letting unfinished envs complete the remaining repeats.
                step_action = action.clone()
                done_mask = torch.from_numpy(done).to(device=self._env.device, dtype=torch.bool)
                step_action[done_mask] = 0.0

            obs_dict, reward, terminated, truncated, _ = self._env.step(
                step_action, auto_reset=False
            )
            image = self._extract_images(obs_dict)
            reward_np = reward.detach().cpu().numpy().reshape(-1).astype(np.float32)
            terminated_np = terminated.detach().cpu().numpy().reshape(-1).astype(bool)
            truncated_np = truncated.detach().cpu().numpy().reshape(-1).astype(bool)
            step_done = terminated_np | truncated_np
            newly_done = step_done & ~done

            if newly_done.any():
                terminal_image[newly_done] = image[newly_done]
                has_terminal_image[newly_done] = True

            total_reward += reward_np * (~done)
            done |= step_done
            terminated_any |= terminated_np
            truncated_any |= truncated_np
            final_image = image
            if done.all():
                break

        assert final_image is not None
        final_image[has_terminal_image] = terminal_image[has_terminal_image]

        obs = {
            "image": final_image,
            "is_first": np.zeros(self._num_envs, dtype=bool),
            "is_terminal": terminated_any,
        }
        discount = np.ones(self._num_envs, dtype=np.float32)
        discount[terminated_any] = 0.0
        discount[truncated_any] = 1.0
        info = {"discount": discount}
        return obs, total_reward, done, info


def count_steps(folder: Path) -> int:
    return sum(int(str(name).split("-")[-1][:-4]) - 1 for name in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = dreamer_tools.sample_episodes(episodes, config.batch_length)
    return dreamer_tools.from_generator(generator, config.batch_size)


def _episode_id(slot: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-env{slot}-{uuid.uuid4().hex}"


def _slice_obs_batch(obs_batch: dict, index: int) -> dict:
    return {k: v[index] for k, v in obs_batch.items()}


def _simulate_vectorized(
    agent,
    env,
    cache,
    directory,
    logger,
    num_envs,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
    reward_meter=None,
    length_meter=None,
):
    if state is None:
        step, episode = 0, 0
        done = np.ones(num_envs, dtype=bool)
        length = np.zeros(num_envs, dtype=np.int32)
        obs = None
        agent_state = None
        reward = np.zeros(num_envs, dtype=np.float32)
        episode_ids = [_episode_id(i) for i in range(num_envs)]
    else:
        step, episode, done, length, obs, agent_state, reward, episode_ids = state

    eval_scores = []
    eval_lengths = []
    while (steps and step < steps) or (episodes and episode < episodes):
        if done.any():
            indices = np.where(done)[0].astype(np.int32)
            reset_obs = env.reset(env_ids=indices)
            if obs is None:
                obs = {
                    k: np.zeros((num_envs,) + tuple(v.shape[1:]), dtype=v.dtype)
                    for k, v in reset_obs.items()
                }
            for local_idx, env_idx in enumerate(indices):
                episode_ids[env_idx] = _episode_id(int(env_idx))
                first_obs = _slice_obs_batch(reset_obs, local_idx)
                transition = {k: dreamer_tools.convert(v) for k, v in first_obs.items()}
                transition["reward"] = 0.0
                transition["discount"] = 1.0
                dreamer_tools.add_to_cache(cache, episode_ids[env_idx], transition)
                for key, value in reset_obs.items():
                    obs[key][env_idx] = value[local_idx]

        action_out, agent_state = agent(obs, done, agent_state)
        if isinstance(action_out, dict):
            action_np = {
                k: np.array(v.detach().cpu()) if torch.is_tensor(v) else np.array(v)
                for k, v in action_out.items()
            }
            env_action = action_np["action"]
        else:
            env_action = np.array(action_out)
            action_np = None

        next_obs, reward, done, info = env.step(env_action)
        episode += int(done.sum())
        length += 1
        step += num_envs
        current_length = length.copy()
        length *= 1 - done.astype(np.int32)

        for i in range(num_envs):
            transition = {k: dreamer_tools.convert(v[i]) for k, v in next_obs.items()}
            if action_np is not None:
                for key, value in action_np.items():
                    transition[key] = value[i]
            else:
                transition["action"] = env_action[i]
            transition["reward"] = reward[i]
            transition["discount"] = info.get("discount", np.array(1.0 - float(done[i])))[i]
            dreamer_tools.add_to_cache(cache, episode_ids[i], transition)

        obs = next_obs

        if done.any():
            indices = np.where(done)[0].astype(np.int32)
            for i in indices:
                dreamer_tools.save_episodes(directory, {episode_ids[i]: cache[episode_ids[i]]})
                score = float(np.array(cache[episode_ids[i]]["reward"]).sum())
                ep_length = int(current_length[i])
                video = cache[episode_ids[i]]["image"]

                if not is_eval:
                    step_in_dataset = dreamer_tools.erase_over_episodes(cache, limit)
                    if reward_meter is not None:
                        reward_meter.update(
                            torch.tensor([[score]], dtype=torch.float32, device=reward_meter.mean.device)
                        )
                    if length_meter is not None:
                        length_meter.update(
                            torch.tensor([[ep_length]], dtype=torch.float32, device=length_meter.mean.device)
                        )
                    policy_reward = (
                        float(reward_meter.get_mean().item()) if reward_meter is not None and reward_meter.current_size > 0
                        else score
                    )
                    episode_lengths = (
                        float(length_meter.get_mean().item()) if length_meter is not None and length_meter.current_size > 0
                        else ep_length
                    )
                    logger.scalar("dataset_size", step_in_dataset)
                    logger.scalar("train_return", policy_reward)
                    logger.scalar("train_length", episode_lengths)
                    logger.scalar("train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    eval_scores.append(score)
                    eval_lengths.append(ep_length)
                    if len(eval_scores) >= episodes:
                        logger.scalar("eval_return", sum(eval_scores) / len(eval_scores))
                        logger.scalar("eval_length", sum(eval_lengths) / len(eval_lengths))
                        logger.scalar("eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        break

        if is_eval and len(eval_scores) >= episodes:
            break

    if is_eval:
        while len(cache) > 1:
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward, episode_ids)


def _filter_consistent_episodes(episodes):
    """Ignore stale malformed episodes from earlier buggy runs."""
    filtered = OrderedDict()
    for key, episode in episodes.items():
        episode_keys = [name for name in episode.keys() if not name.startswith("log_")]
        if not episode_keys:
            continue
        length = len(episode[episode_keys[0]])
        if all(len(episode[name]) == length for name in episode_keys):
            filtered[key] = episode
    return filtered


def _recursive_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _resolve_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _dreamer_config_from_hydra(config: DictConfig) -> SimpleNamespace:
    yaml_loader = YAML(typ="safe", pure=True)
    with (_DREAMER_DIR / "configs.yaml").open() as f:
        configs_yaml = yaml_loader.load(f)

    flat = dict(configs_yaml.get("defaults", {}))
    _recursive_update(flat, configs_yaml.get("dmc_vision", {}))
    _recursive_update(flat, OmegaConf.to_container(config.agent.config, resolve=True))

    flat["seed"] = int(config.seed)
    flat["device"] = str(config.device)
    flat["size"] = [int(v) for v in flat.get("size", [64, 64])]
    num_envs = int(_resolve_value(config.task.config.get("num_envs", flat.get("envs", 1))))
    flat["envs"] = num_envs
    flat["parallel"] = False

    flat["steps"] = int(flat.get("steps", 1_000_000))
    flat["log_every"] = int(flat.get("log_every", 10_000))
    flat["action_repeat"] = int(flat.get("action_repeat", 1))
    flat["prefill"] = int(flat.get("prefill", 0))
    flat["batch_size"] = int(flat.get("batch_size", 16))
    flat["batch_length"] = int(flat.get("batch_length", 64))
    flat["dataset_size"] = int(flat.get("dataset_size", 1_000_000))
    flat["pretrain"] = int(flat.get("pretrain", 1))
    flat["eval_episode_num"] = int(flat.get("eval_episode_num", num_envs))
    flat["steps"] //= flat["action_repeat"]
    flat["log_every"] //= flat["action_repeat"]

    return SimpleNamespace(**flat)


class DreamerWorkspace:
    def __init__(
        self,
        config: SimpleNamespace,
        base_env: BaseEnv,
        work_dir: Path,
        full_config: DictConfig | None = None,
    ):
        self.config = config
        self.work_dir = Path(work_dir)
        self.training_logs_dir = self.work_dir / "training_logs"
        self.summaries_dir = self.training_logs_dir / "summaries"
        self.nn_dir = self.training_logs_dir / "nn"
        self.buffer_dir = self.training_logs_dir / "buffer"
        self.traindir = self.buffer_dir / "train_eps"
        self.evaldir = self.buffer_dir / "eval_eps"
        for d in (self.summaries_dir, self.nn_dir, self.buffer_dir, self.traindir, self.evaldir):
            d.mkdir(parents=True, exist_ok=True)

        self.config.logdir = self.summaries_dir
        self.config.traindir = self.traindir
        self.config.evaldir = self.evaldir

        dreamer_tools.set_seed_everywhere(self.config.seed)
        if getattr(self.config, "deterministic_run", False):
            dreamer_tools.enable_deterministic_run()

        self.use_wandb = self._init_wandb(full_config) if full_config is not None else False
        print("Logdir", self.summaries_dir)
        print("Create envs.")

        self.env = self._wrap_env(base_env)
        self.num_envs = self.env.num_envs
        print("Action Space", self.env.action_space)

        self.config.num_actions = self.env.action_space.shape[0]
        step = count_steps(self.traindir)
        self.logger = DreamerLogger(
            self.summaries_dir,
            self.config.action_repeat * step,
            use_wandb=self.use_wandb,
        )

        directory = self.config.offline_traindir or self.traindir
        self.train_eps = _filter_consistent_episodes(
            dreamer_tools.load_episodes(directory, limit=self.config.dataset_size)
        )

        self.train_dataset = make_dataset(self.train_eps, self.config)
        self.agent = Dreamer(
            self.env.observation_space,
            self.env.action_space,
            self.config,
            self.logger,
            self.train_dataset,
        ).to(self.config.device)
        self.agent.requires_grad_(requires_grad=False)
        self._state = None
        self.episode_reward_meter = AverageMeter(1, 100).to(self.config.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.config.device)
        self._iter_count = 0
        self._best_policy_reward = -float("inf")

        latest = self.nn_dir / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location=self.config.device, weights_only=False)
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            dreamer_tools.recursively_load_optim_state_dict(
                self.agent,
                checkpoint["optims_state_dict"],
            )
            self.agent._should_pretrain._once = False
            self.logger.step = checkpoint.get("logger_step", self.logger.step)
            self.logger._iter = checkpoint.get("logger_iter", self.logger._iter)
            self._iter_count = checkpoint.get("iter_count", self._iter_count)
            self._best_policy_reward = checkpoint.get("best_policy_reward", self._best_policy_reward)

    def save_snapshot(self, filename=None):
        path = self.nn_dir / (filename or "snapshot.pt")
        payload = {
            "agent_state_dict": self.agent.state_dict(),
            "optims_state_dict": dreamer_tools.recursively_collect_optim_state_dict(self.agent),
            "logger_step": self.logger.step,
            "logger_iter": self.logger._iter,
            "iter_count": self._iter_count,
            "best_policy_reward": self._best_policy_reward,
        }
        torch.save(payload, path)

    def _init_wandb(self, config: DictConfig) -> bool:
        if not getattr(config, "wandb", None) or not config.wandb.get("enable", False):
            return False
        wandb_kwargs = {
            "project": config.wandb.get("project", "approximate-forl"),
            "entity": config.wandb.get("entity"),
            "group": config.wandb.get("group"),
            "job_type": config.wandb.get("job_type"),
            "name": config.wandb.get("name"),
            "tags": config.wandb.get("tags", []),
            "notes": config.wandb.get("notes"),
        }
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        wandb.init(**wandb_kwargs)
        if config.wandb.get("log_config", True):
            wandb.config.update(OmegaConf.to_container(config, resolve=True))
        return True

    def _wrap_env(self, base_env: BaseEnv):
        return DreamerGenesisVecEnv(
            base_env,
            img_size=self.config.size[0],
            action_repeat=self.config.action_repeat,
        )

    def eval(self):
        print("Start evaluation.")
        directory = self.config.offline_evaldir or self.evaldir
        eval_eps = _filter_consistent_episodes(dreamer_tools.load_episodes(directory, limit=1))
        eval_policy = functools.partial(self.agent, training=False)
        eval_episode_num = max(1, int(getattr(self.config, "eval_episode_num", self.num_envs)))
        _simulate_vectorized(
            eval_policy,
            self.env,
            eval_eps,
            self.evaldir,
            self.logger,
            self.num_envs,
            is_eval=True,
            episodes=eval_episode_num,
        )

    def train(self):
        save_snapshot = getattr(self.config, "save_snapshot", True)
        save_snapshot_every_frames = getattr(self.config, "save_snapshot_every_frames", 50_000)
        if save_snapshot:
            initial_path = self.nn_dir / "initial_snapshot.pt"
            if not initial_path.exists():
                self.save_snapshot("initial_snapshot.pt")
        next_save_step = (
            ((self.logger.step // save_snapshot_every_frames) + 1) * save_snapshot_every_frames
            if save_snapshot_every_frames > 0
            else None
        )

        if not self.config.offline_traindir:
            prefill = max(0, self.config.prefill - count_steps(self.traindir))
            if not self.train_eps and prefill == 0:
                prefill = self.config.prefill
            print(f"Prefill dataset ({prefill} steps).")
            if prefill > 0:
                acts = self.env.action_space
                low = torch.tensor(acts.low, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
                high = torch.tensor(acts.high, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
                random_actor = torch.distributions.independent.Independent(
                    torch.distributions.uniform.Uniform(
                        low,
                        high,
                    ),
                    1,
                )

                def random_agent(obs, reset, state):
                    action = random_actor.sample()
                    logprob = random_actor.log_prob(action)
                    self.logger.step += len(reset) * self.config.action_repeat
                    return {"action": action, "logprob": logprob}, None

                self._state = _simulate_vectorized(
                    random_agent,
                    self.env,
                    self.train_eps,
                    self.traindir,
                    self.logger,
                    self.num_envs,
                    limit=self.config.dataset_size,
                    steps=prefill,
                    reward_meter=self.episode_reward_meter,
                    length_meter=self.episode_length_meter,
                )
                print(f"Logger: ({self.logger.step} steps).")
                self.train_dataset = make_dataset(self.train_eps, self.config)
                self.agent._dataset = self.train_dataset
                self.agent._step = self.logger.step // self.config.action_repeat

        print("Simulate agent.")
        while self.agent._step < self.config.steps:
            self._iter_count += 1
            print("Start training.")
            steps_left = self.config.steps - self.agent._step
            chunk_steps = min(max(1, self.config.log_every), steps_left)
            self._state = _simulate_vectorized(
                self.agent,
                self.env,
                self.train_eps,
                self.traindir,
                self.logger,
                self.num_envs,
                limit=self.config.dataset_size,
                steps=chunk_steps,
                state=self._state,
                reward_meter=self.episode_reward_meter,
                length_meter=self.episode_length_meter,
            )
            self.save_snapshot("latest.pt")

            policy_reward = (
                float(self.episode_reward_meter.get_mean().item())
                if self.episode_reward_meter.current_size > 0
                else None
            )
            if policy_reward is not None and policy_reward > self._best_policy_reward:
                self._best_policy_reward = policy_reward
                if save_snapshot:
                    self.save_snapshot("best_policy.pt")
                    print(f"Save best policy with reward: {self._best_policy_reward:.2f}")

            if (
                save_snapshot
                and next_save_step is not None
                and self.logger.step >= next_save_step
            ):
                reward_for_name = policy_reward if policy_reward is not None else 0.0
                self.save_snapshot(
                    f"iter_{self._iter_count}_reward_{reward_for_name:.2f}.pt"
                )
                next_save_step = (
                    (self.logger.step // save_snapshot_every_frames) + 1
                ) * save_snapshot_every_frames

        if save_snapshot:
            final_reward = (
                float(self.episode_reward_meter.get_mean().item())
                if self.episode_reward_meter.current_size > 0
                else 0.0
            )
            self.save_snapshot(f"iter_{self._iter_count}_reward_{final_reward:.2f}.pt")

        if self.use_wandb:
            wandb.finish()
        try:
            self.env._env.close()
        except Exception:
            pass

    def load_snapshot(self, path=None):
        path = Path(path or (self.nn_dir / "latest.pt"))
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        if "optims_state_dict" in checkpoint:
            dreamer_tools.recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])
        self.logger.step = checkpoint.get("logger_step", self.logger.step)
        self.logger._iter = checkpoint.get("logger_iter", self.logger._iter)
        self._iter_count = checkpoint.get("iter_count", self._iter_count)
        self._best_policy_reward = checkpoint.get("best_policy_reward", self._best_policy_reward)


def make_runner(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir) if hydra_cfg is not None else Path.cwd()

    OmegaConf.set_struct(config, False)
    config.log_dir = str(output_dir)
    config.task.config.vis_obs = True

    size = OmegaConf.to_container(config.agent.config.get("size", [64, 64]), resolve=True)
    size = [int(v) for v in size]
    if config.task.config.get("sensors_args") is None:
        config.task.config.sensors_args = {}
    if config.task.config.sensors_args.get("camera") is None:
        config.task.config.sensors_args["camera"] = {}
    config.task.config.sensors_args.camera.res = size

    num_envs = int(_resolve_value(config.task.config.get("num_envs", config.agent.config.get("num_envs", 1))))
    config.task.config.num_envs = num_envs
    config.agent.config.num_envs = num_envs
    OmegaConf.set_struct(config, True)

    base_env = make_envs(config)
    dreamer_cfg = _dreamer_config_from_hydra(config)
    workspace = DreamerWorkspace(
        dreamer_cfg,
        base_env,
        output_dir,
        full_config=config,
    )

    class Runner:
        def run(self, args):
            if args.get("checkpoint"):
                workspace.load_snapshot(args["checkpoint"])
            if args.get("train", False):
                workspace.train()
            elif args.get("play", False):
                workspace.eval()

    return Runner()
