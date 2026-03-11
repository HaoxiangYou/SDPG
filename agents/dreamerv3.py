"""
DreamerV3 runner for Genesis pixel environments.

This mirrors the reference `train_dreamerv3.py` flow, but loads Genesis envs
through this repo's `make_envs()` utility, similar to `agents/drqv2.py`.
"""

import functools
import importlib.util
import os
import sys
from collections import OrderedDict
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
from parallel import Damy

Dreamer = _dreamer_mod.Dreamer

_wrappers_spec = importlib.util.spec_from_file_location(
    "dreamerv3_wrappers",
    _DREAMER_DIR / "envs" / "wrappers.py",
)
dreamer_wrappers = importlib.util.module_from_spec(_wrappers_spec)
assert _wrappers_spec.loader is not None
_wrappers_spec.loader.exec_module(dreamer_wrappers)


class DreamerGenesisEnv(gym.Env):
    """Gym-style wrapper that exposes Genesis RGB observations as Dreamer images."""

    metadata = {}

    def __init__(
        self,
        base_env: BaseEnv,
        img_size: int = 64,
        action_repeat: int = 2,
    ):
        assert getattr(base_env, "num_envs", 1) == 1, "DreamerV3 currently expects a single Genesis env."
        obs_spaces = getattr(base_env.observation_space, "spaces", {})
        assert "RGB" in obs_spaces, "DreamerV3 requires pixel observations (`task.config.vis_obs=True`)."

        self._env = base_env
        self._img_size = int(img_size)
        self._action_repeat = int(action_repeat)
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

    def _extract_image(self, obs_dict: dict) -> np.ndarray:
        rgb = obs_dict["RGB"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach()
            if rgb.ndim == 4:
                rgb = rgb[0]
            rgb = rgb.cpu().numpy()
        else:
            rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]

        if rgb.shape[0] == 9:
            rgb = rgb[-3:]
        elif rgb.shape[0] != 3:
            rgb = rgb[:3]

        image = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
        if image.shape[:2] != (self._img_size, self._img_size):
            image = cv2.resize(
                image,
                (self._img_size, self._img_size),
                interpolation=cv2.INTER_AREA,
            )
        return image

    def reset(self):
        obs_dict, _ = self._env.reset(env_ids=None)
        return {
            "image": self._extract_image(obs_dict),
            "is_first": np.array(True),
            "is_terminal": np.array(False),
        }

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self._env.device)
        else:
            action = action.float().to(self._env.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        total_reward = 0.0
        obs_dict = None
        done = False
        for _ in range(self._action_repeat):
            obs_dict, reward, terminated, truncated, _ = self._env.step(action, auto_reset=False)
            total_reward += float(reward.reshape(-1)[0].item())
            done = bool((terminated | truncated).reshape(-1)[0].item())
            if done:
                break

        assert obs_dict is not None
        obs = {
            "image": self._extract_image(obs_dict),
            "is_first": np.array(False),
            "is_terminal": np.array(False),
        }
        info = {"discount": np.array(1.0, np.float32)}
        return obs, total_reward / self._action_repeat, done, info


def count_steps(folder: Path) -> int:
    return sum(int(str(name).split("-")[-1][:-4]) - 1 for name in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = dreamer_tools.sample_episodes(episodes, config.batch_length)
    return dreamer_tools.from_generator(generator, config.batch_size)


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
    flat["envs"] = 1
    flat["num_envs"] = 1
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
        self.train_envs = [Damy(self.train_env)]
        self.eval_envs = [Damy(self.eval_env)]
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
        env = DreamerGenesisEnv(
            base_env,
            img_size=self.config.size[0],
            action_repeat=self.config.action_repeat,
        )
        env = dreamer_wrappers.NormalizeActions(env)
        env = dreamer_wrappers.TimeLimit(env, self.config.time_limit)
        env = dreamer_wrappers.SelectAction(env, key="action")
        env = dreamer_wrappers.UUID(env)
        return env

    def eval(self):
        if self.config.eval_episode_num <= 0:
            return
        print("Start evaluation.")
        eval_policy = functools.partial(self.agent, training=False)
        dreamer_tools.simulate(
            eval_policy,
            self.eval_envs,
            self.eval_eps,
            self.evaldir,
            self.logger,
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
                random_actor = torch.distributions.independent.Independent(
                    torch.distributions.uniform.Uniform(
                        torch.tensor(acts.low).unsqueeze(0),
                        torch.tensor(acts.high).unsqueeze(0),
                    ),
                    1,
                )

                def random_agent(obs, reset, state):
                    action = random_actor.sample()
                    logprob = random_actor.log_prob(action)
                    return {"action": action, "logprob": logprob}, None

                self._state = dreamer_tools.simulate(
                    random_agent,
                    self.train_envs,
                    self.train_eps,
                    self.traindir,
                    self.logger,
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
            self._state = dreamer_tools.simulate(
                self.agent,
                self.train_envs,
                self.train_eps,
                self.traindir,
                self.logger,
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
        for env in self.train_envs + self.eval_envs:
            try:
                env.close()
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

    config.task.config.num_envs = 1
    config.agent.config.num_envs = 1
    config.agent.config.envs = 1
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
