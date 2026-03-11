"""
DreamerV3 runner for Genesis pixel environments.
"""

import functools
import os
import sys
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

from envs.base_env import BaseEnv
from utils.common_utils import make_envs

os.environ.setdefault("MUJOCO_GL", "egl")

_DREAMER_DIR = Path(__file__).resolve().parent.parent / "externals" / "dreamerv3-torch"
if str(_DREAMER_DIR) not in sys.path:
    sys.path.insert(0, str(_DREAMER_DIR))

import dreamer as _dreamer_mod
import tools as dreamer_tools

Dreamer = _dreamer_mod.Dreamer

class DreamerGenesisVecEnv(gym.Env):
    """Vectorized Genesis wrapper that exposes Dreamer-style image observations."""

    metadata = {}

    def __init__(
        self,
        base_env: BaseEnv,
        img_size: int = 64,
        action_repeat: int = 2,
        time_limit: int = 1000,
    ):
        obs_spaces = getattr(base_env.observation_space, "spaces", {})
        assert "RGB" in obs_spaces, "DreamerV3 requires pixel observations (`task.config.vis_obs=True`)."

        self._env = base_env
        self._img_size = int(img_size)
        self._action_repeat = int(action_repeat)
        self._time_limit = int(time_limit)
        self._num_envs = int(getattr(base_env, "num_envs", 1))
        self._num_actions = int(base_env.num_actions)
        self.reward_range = [-np.inf, np.inf]
        self._episode_steps = np.zeros(self._num_envs, dtype=np.int32)

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
        self._episode_steps[env_ids_np] = 0
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
        obs_dict = None
        for _ in range(self._action_repeat):
            obs_dict, reward, terminated, truncated, _ = self._env.step(action, auto_reset=False)
            reward_np = reward.detach().cpu().numpy().reshape(-1).astype(np.float32)
            step_done = (terminated | truncated).detach().cpu().numpy().reshape(-1).astype(bool)
            total_reward += reward_np * (~done)
            done |= step_done
            if done.all():
                break

        assert obs_dict is not None
        self._episode_steps += 1
        timeout = self._episode_steps >= self._time_limit
        done |= timeout

        obs = {
            "image": self._extract_images(obs_dict),
            "is_first": np.zeros(self._num_envs, dtype=bool),
            "is_terminal": np.zeros(self._num_envs, dtype=bool),
        }
        info = {"discount": np.ones(self._num_envs, dtype=np.float32)}
        return obs, total_reward / self._action_repeat, done, info


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
                    logger.scalar("dataset_size", step_in_dataset)
                    logger.scalar("train_return", score)
                    logger.scalar("train_length", ep_length)
                    logger.scalar("train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    eval_scores.append(score)
                    eval_lengths.append(ep_length)
                    logger.video("eval_policy", np.array(video)[None])
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
    flat["eval_every"] = int(flat.get("eval_every", 10_000))
    flat["log_every"] = int(flat.get("log_every", 10_000))
    flat["time_limit"] = int(flat.get("time_limit", 1000))
    flat["action_repeat"] = int(flat.get("action_repeat", 1))
    flat["prefill"] = int(flat.get("prefill", 0))
    flat["batch_size"] = int(flat.get("batch_size", 16))
    flat["batch_length"] = int(flat.get("batch_length", 64))
    flat["dataset_size"] = int(flat.get("dataset_size", 1_000_000))
    flat["eval_episode_num"] = int(flat.get("eval_episode_num", 10))
    flat["pretrain"] = int(flat.get("pretrain", 1))
    flat["steps"] //= flat["action_repeat"]
    flat["eval_every"] //= flat["action_repeat"]
    flat["log_every"] //= flat["action_repeat"]
    flat["time_limit"] //= flat["action_repeat"]

    return SimpleNamespace(**flat)


class DreamerWorkspace:
    def __init__(
        self,
        config: SimpleNamespace,
        train_base_env: BaseEnv,
        eval_base_env: BaseEnv,
        work_dir: Path,
        full_config: DictConfig | None = None,
    ):
        self.config = config
        self.work_dir = Path(work_dir)
        self.training_logs_dir = self.work_dir / "training_logs"
        self.logdir = self.training_logs_dir / "dreamerv3"
        self.traindir = self.logdir / "train_eps"
        self.evaldir = self.logdir / "eval_eps"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.traindir.mkdir(parents=True, exist_ok=True)
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.config.logdir = self.logdir
        self.config.traindir = self.traindir
        self.config.evaldir = self.evaldir

        dreamer_tools.set_seed_everywhere(self.config.seed)
        if getattr(self.config, "deterministic_run", False):
            dreamer_tools.enable_deterministic_run()

        self.use_wandb = self._init_wandb(full_config) if full_config is not None else False
        print("Logdir", self.logdir)
        print("Create envs.")

        self.train_env = self._wrap_env(train_base_env)
        self.eval_env = self._wrap_env(eval_base_env)
        self.num_envs = self.train_env.num_envs
        print("Action Space", self.train_env.action_space)

        self.config.num_actions = self.train_env.action_space.shape[0]
        step = count_steps(self.traindir)
        self.logger = dreamer_tools.Logger(self.logdir, self.config.action_repeat * step)

        directory = self.config.offline_traindir or self.traindir
        self.train_eps = _filter_consistent_episodes(
            dreamer_tools.load_episodes(directory, limit=self.config.dataset_size)
        )
        directory = self.config.offline_evaldir or self.evaldir
        self.eval_eps = _filter_consistent_episodes(dreamer_tools.load_episodes(directory, limit=1))

        self.train_dataset = make_dataset(self.train_eps, self.config)
        self.eval_dataset = make_dataset(self.eval_eps, self.config)
        self.agent = Dreamer(
            self.train_env.observation_space,
            self.train_env.action_space,
            self.config,
            self.logger,
            self.train_dataset,
        ).to(self.config.device)
        self.agent.requires_grad_(requires_grad=False)
        self._state = None

        latest = self.logdir / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location=self.config.device, weights_only=False)
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            dreamer_tools.recursively_load_optim_state_dict(
                self.agent,
                checkpoint["optims_state_dict"],
            )
            self.agent._should_pretrain._once = False

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
            time_limit=self.config.time_limit,
        )

    def eval(self):
        if self.config.eval_episode_num <= 0:
            return
        print("Start evaluation.")
        eval_policy = functools.partial(self.agent, training=False)
        _simulate_vectorized(
            eval_policy,
            self.eval_env,
            self.eval_eps,
            self.evaldir,
            self.logger,
            self.num_envs,
            is_eval=True,
            episodes=self.config.eval_episode_num,
        )
        if getattr(self.config, "video_pred_log", False) and self.eval_eps:
            video_pred = self.agent._wm.video_pred(next(self.eval_dataset))
            self.logger.video("eval_openl", video_pred.detach().cpu().numpy())

    def train(self):
        if not self.config.offline_traindir:
            prefill = max(0, self.config.prefill - count_steps(self.traindir))
            if not self.train_eps and prefill == 0:
                prefill = self.config.prefill
            print(f"Prefill dataset ({prefill} steps).")
            if prefill > 0:
                acts = self.train_env.action_space
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
                    return {"action": action, "logprob": logprob}, None

                self._state = _simulate_vectorized(
                    random_agent,
                    self.train_env,
                    self.train_eps,
                    self.traindir,
                    self.logger,
                    self.num_envs,
                    limit=self.config.dataset_size,
                    steps=prefill,
                )
                self.logger.step += prefill * self.config.action_repeat
                print(f"Logger: ({self.logger.step} steps).")
                self.train_dataset = make_dataset(self.train_eps, self.config)
                self.agent._dataset = self.train_dataset

        print("Simulate agent.")
        while self.agent._step < self.config.steps + self.config.eval_every:
            self.logger.write()
            self.eval()
            print("Start training.")
            self._state = _simulate_vectorized(
                self.agent,
                self.train_env,
                self.train_eps,
                self.traindir,
                self.logger,
                self.num_envs,
                limit=self.config.dataset_size,
                steps=self.config.eval_every,
                state=self._state,
            )
            torch.save(
                {
                    "agent_state_dict": self.agent.state_dict(),
                    "optims_state_dict": dreamer_tools.recursively_collect_optim_state_dict(self.agent),
                },
                self.logdir / "latest.pt",
            )

        if self.use_wandb:
            wandb.finish()
        for env in (self.train_env, self.eval_env):
            try:
                env._env.close()
            except Exception:
                pass

    def load_snapshot(self, path=None):
        path = Path(path or (self.logdir / "latest.pt"))
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        if "optims_state_dict" in checkpoint:
            dreamer_tools.recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])


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

    train_base_env = make_envs(config)
    eval_base_env = make_envs(config)
    dreamer_cfg = _dreamer_config_from_hydra(config)
    workspace = DreamerWorkspace(
        dreamer_cfg,
        train_base_env,
        eval_base_env,
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
