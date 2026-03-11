"""
DreamerV3 agent: uses externals/dreamerv3-torch (PyTorch Dreamer v3).
Wraps a BaseEnv (pixel obs) as a gym-style env with dict obs (image, is_first, is_terminal)
for the dreamerv3-torch simulate/training loop. Aligned with agents/drqv2.py env usage.
"""
import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import dreamer as _dreamer_pkg
Dreamer = _dreamer_pkg.Dreamer
import tools as dreamer_tools
from parallel import Damy

# Load dreamerv3 envs.wrappers by path so project's "envs" package is not used
_wrappers_spec = importlib.util.spec_from_file_location(
    "dreamerv3_wrappers",
    Path(_dreamer_pkg.__file__).resolve().parent / "envs" / "wrappers.py",
)
dreamer_wrappers = importlib.util.module_from_spec(_wrappers_spec)
_wrappers_spec.loader.exec_module(dreamer_wrappers)

from envs.base_env import BaseEnv
from utils.common_utils import make_envs

# --- Gym env wrapper for DreamerV3 (dict obs: image, is_first, is_terminal) ---

class DreamerEnv(gym.Env):
    """
    Wraps a BaseEnv (single env, num_envs=1) as a gym.Env for DreamerV3.
    observation_space: Dict(image=Box(0,255,(img_size,img_size,3)), is_first=Box(0,1,(1,)), is_terminal=Box(0,1,(1,)))
    action_space: Box(-1, 1, (num_actions,))
    reset() -> obs dict. step(action) -> (obs_dict, reward, done, info).
    """

    def __init__(
        self,
        base_env: BaseEnv,
        img_size: int = 84,
        discount: float = 0.99,
    ):
        assert getattr(base_env, "num_envs", base_env.num_envs) == 1, (
            "DreamerV3 wrapper expects num_envs=1 (single env)."
        )
        self._env = base_env
        self._img_size = img_size
        self._discount = discount
        self._num_actions = base_env.num_actions

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32
        )

    def _get_image(self, obs_dict: dict) -> np.ndarray:
        """Return (H, W, 3) uint8 from env RGB (C, H, W) or (9, H, W)."""
        rgb = obs_dict["RGB"]
        if torch.is_tensor(rgb):
            rgb = rgb[0].cpu().numpy()  # (9, H, W) or (3, H, W)
        else:
            rgb = np.asarray(rgb, dtype=np.uint8)
            if rgb.ndim == 3 and rgb.shape[0] == 9:
                rgb = rgb[0]
            elif rgb.ndim == 2:
                rgb = np.expand_dims(rgb, 0)
        if rgb.shape[0] == 9:
            rgb = rgb[-3:]  # last 3 frames as (3,H,W)
        if rgb.shape[0] != 3:
            rgb = rgb[:3]
        # (3, H, W) -> (H, W, 3)
        img = np.transpose(rgb, (1, 2, 0))
        if img.shape[0] != self._img_size or img.shape[1] != self._img_size:
            import cv2
            img = cv2.resize(img, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def reset(self):
        obs_dict, _ = self._env.reset(env_ids=None)
        img = self._get_image(obs_dict)
        return {
            "image": img,
            "is_first": np.array([1.0], dtype=np.float32),
            "is_terminal": np.array([0.0], dtype=np.float32),
        }

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self._env.device)
        else:
            action = action.float().to(self._env.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        action = torch.clamp(action, -1.0, 1.0)
        obs_dict, reward, terminated, truncated, _ = self._env.step(action, auto_reset=False)
        done = terminated | truncated
        if done.any():
            self._env.reset(env_ids=torch.where(done)[0])
        img = self._get_image(obs_dict)
        r = float(reward.cpu().item())
        d = bool(done.cpu().item())
        discount = 0.0 if d else self._discount
        obs = {
            "image": img,
            "is_first": np.array([0.0], dtype=np.float32),
            "is_terminal": np.array([1.0 if d else 0.0], dtype=np.float32),
        }
        info = {"discount": np.array(discount, dtype=np.float32)}
        return obs, r, d, info


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    return dreamer_tools.from_generator(
        dreamer_tools.sample_episodes(episodes, config.batch_length),
        config.batch_size,
    )


# --- Workspace and runner (aligned with drqv2 + dreamer.py main) ---


class Dreamerv3Workspace:
    """
    DreamerV3 workspace: uses DreamerEnv (from make_envs), dreamerv3-torch's
    Dreamer, tools.simulate, load_episodes, make_dataset. Train and play (eval) are
    mutually exclusive like DrQv2Workspace.
    """

    def __init__(self, config: SimpleNamespace, work_dir: Path, full_config: DictConfig = None):
        self.work_dir = Path(work_dir)
        self.training_logs_dir = self.work_dir / "training_logs"
        self.logdir = self.training_logs_dir / "dreamerv3"
        self.traindir = self.logdir / "train_eps"
        self.evaldir = self.logdir / "eval_eps"
        for d in (self.logdir, self.traindir, self.evaldir):
            d.mkdir(parents=True, exist_ok=True)

        self.config = config
        dreamer_tools.set_seed_everywhere(config.seed)
        self.device = torch.device(config.device)

        self.use_wandb = self._init_wandb(full_config) if full_config else False

        # Build train/eval envs (list of one env each for now; DreamerV3 uses list of envs)
        self.train_env = self._wrap_env(config.train_env)
        self.eval_env = self._wrap_env(config.eval_env)
        self.train_envs = [Damy(self.train_env)]
        self.eval_envs = [Damy(self.eval_env)]

        config.num_actions = self.train_env.action_space.shape[0]
        step = count_steps(self.traindir)
        config.traindir = self.traindir
        config.evaldir = self.evaldir
        self.logger = dreamer_tools.Logger(self.logdir, config.action_repeat * step)

        train_eps = dreamer_tools.load_episodes(self.traindir, limit=config.dataset_size)
        eval_eps = dreamer_tools.load_episodes(self.evaldir, limit=1)
        self.train_eps = train_eps
        self.eval_eps = eval_eps

        self.train_dataset = make_dataset(train_eps, config)
        self.eval_dataset = make_dataset(eval_eps, config)
        self.agent = Dreamer(
            self.train_env.observation_space,
            self.train_env.action_space,
            config,
            self.logger,
            self.train_dataset,
        ).to(config.device)
        self.agent.requires_grad_(requires_grad=False)

        latest = self.logdir / "latest.pt"
        if latest.exists():
            checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            dreamer_tools.recursively_load_optim_state_dict(
                self.agent, checkpoint["optims_state_dict"]
            )
            self.agent._should_pretrain._once = False

        self._state = None

    def _wrap_env(self, base_env: BaseEnv):
        env = DreamerEnv(
            base_env,
            img_size=getattr(self.config, "size", [84, 84])[0],
            discount=getattr(self.config, "discount", 0.997),
        )
        env = dreamer_wrappers.TimeLimit(env, self.config.time_limit)
        env = dreamer_wrappers.NormalizeActions(env)
        env = dreamer_wrappers.SelectAction(env, key="action")
        env = dreamer_wrappers.UUID(env)
        return env

    def _init_wandb(self, config: DictConfig) -> bool:
        if not getattr(config, "wandb", None) or not config.wandb.get("enable", False):
            return False
        w = config.wandb
        kwargs = {
            "project": w.get("project", "approximate-forl"),
            "entity": w.get("entity"),
            "group": w.get("group"),
            "job_type": w.get("job_type"),
            "name": w.get("name"),
            "tags": w.get("tags", []),
            "notes": w.get("notes"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        wandb.init(**kwargs)
        if w.get("log_config", True):
            wandb.config.update(OmegaConf.to_container(config, resolve=True))
        return True

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
        if getattr(self.config, "video_pred_log", False):
            video_pred = self.agent._wm.video_pred(next(self.eval_dataset))
            self.logger.video("eval_openl", video_pred.detach().cpu().numpy())

    def train(self):
        config = self.config
        if not getattr(config, "offline_traindir", None) or not config.offline_traindir:
            prefill = max(0, config.prefill - count_steps(self.traindir))
            if prefill > 0:
                print(f"Prefill dataset ({prefill} steps).")
                from torch import distributions as torchd
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(self.train_env.action_space.low).unsqueeze(0),
                        torch.tensor(self.train_env.action_space.high).unsqueeze(0),
                    ),
                    1,
                )

                def random_agent(o, d, s):
                    action = random_actor.sample()
                    logprob = random_actor.log_prob(action)
                    return {"action": action, "logprob": logprob}, None

                self._state = dreamer_tools.simulate(
                    random_agent,
                    self.train_envs,
                    self.train_eps,
                    self.traindir,
                    self.logger,
                    limit=config.dataset_size,
                    steps=prefill,
                )
                self.logger.step += prefill * config.action_repeat

        while self.agent._step < config.steps + config.eval_every:
            self.logger.write()
            self.eval()
            print("Start training.")
            self._state = dreamer_tools.simulate(
                self.agent,
                self.train_envs,
                self.train_eps,
                self.traindir,
                self.logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=self._state,
            )
            torch.save(
                {
                    "agent_state_dict": self.agent.state_dict(),
                    "optims_state_dict": dreamer_tools.recursively_collect_optim_state_dict(
                        self.agent
                    ),
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
        path = path or self.logdir / "latest.pt"
        if not Path(path).exists():
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        if "optims_state_dict" in ckpt:
            dreamer_tools.recursively_load_optim_state_dict(
                self.agent, ckpt["optims_state_dict"]
            )


def _recursive_update(base: dict, update: dict):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _recursive_update(base[k], v)
        else:
            base[k] = v


def _dreamer_config_from_hydra(config: DictConfig) -> SimpleNamespace:
    """Build dreamerv3-torch config (SimpleNamespace) from Hydra agent.config + defaults."""
    from ruamel.yaml import YAML
    defaults_path = Path(_dreamer_pkg.__file__).resolve().parent / "configs.yaml"
    yaml_loader = YAML(typ="safe", pure=True)
    with defaults_path.open() as f:
        configs_yaml = yaml_loader.load(f)
    flat = dict(configs_yaml.get("defaults", {}))
    _recursive_update(flat, configs_yaml.get("dmc_vision", {}))
    # Override with our agent.config
    ac = config.agent.get("config", {})
    ac_dict = OmegaConf.to_container(ac, resolve=True) if ac else {}
    if ac_dict:
        _recursive_update(flat, ac_dict)

    flat.setdefault("logdir", None)
    flat.setdefault("traindir", None)
    flat.setdefault("evaldir", None)
    flat.setdefault("size", [84, 84])
    flat.setdefault("envs", 1)
    flat.setdefault("action_repeat", 2)
    flat.setdefault("time_limit", 1000)
    flat.setdefault("steps", 1_000_000)
    flat.setdefault("eval_every", 10_000)
    flat.setdefault("eval_episode_num", 10)
    flat.setdefault("log_every", 10_000)
    flat.setdefault("prefill", 2500)
    flat.setdefault("dataset_size", 1_000_000)
    flat.setdefault("device", str(config.device))
    flat.setdefault("seed", config.seed)
    flat.setdefault("num_actions", None)

    # Keep nested encoder/actor-style dicts as dicts so **config.encoder etc. work in dreamerv3-torch.
    # Top-level has "act"/"norm" too, so we only treat as config dict if it has cnn/mlp keys or (layers+dist).
    def _is_config_dict(d):
        if not isinstance(d, dict):
            return False
        has_cnn_mlp = "cnn_keys" in d or "mlp_keys" in d
        has_layers_dist = "layers" in d and "dist" in d
        return has_cnn_mlp or has_layers_dist

    def recursive_ns(d):
        if isinstance(d, dict):
            if _is_config_dict(d):
                return d
            return SimpleNamespace(**{k: recursive_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [recursive_ns(x) for x in d]
        return d

    return recursive_ns(flat)


def make_runner(config: DictConfig):
    """Build DreamerV3 runner using DreamerEnv from make_envs (same pattern as drqv2)."""
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        output_dir = hydra_cfg.runtime.output_dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)

    OmegaConf.set_struct(config, False)
    # DreamerV3 uses single env (num_envs=1) for the simulate loop
    config.task.config.vis_obs = True
    config.task.config.num_envs = 1
    if "sensors_args" in config.task.config:
        config.task.config.sensors_args.camera.res = [84, 84]
    else:
        config.task.config.setdefault("sensors_args", {})
        if "camera" not in config.task.config.sensors_args:
            config.task.config.sensors_args["camera"] = {}
        config.task.config.sensors_args["camera"]["res"] = [84, 84]
    OmegaConf.set_struct(config, True)

    base_env_train = make_envs(config)
    base_env_eval = make_envs(config)

    dreamer_cfg = _dreamer_config_from_hydra(config)
    dreamer_cfg.logdir = None
    dreamer_cfg.traindir = None
    dreamer_cfg.evaldir = None

    work_dir = Path(getattr(config, "log_dir", Path.cwd()))
    work_dir.mkdir(parents=True, exist_ok=True)

    dreamer_cfg.train_env = base_env_train
    dreamer_cfg.eval_env = base_env_eval
    dreamer_cfg.size = list(getattr(dreamer_cfg, "size", [84, 84]))
    dreamer_cfg.envs = 1
    dreamer_cfg.action_repeat = getattr(dreamer_cfg, "action_repeat", 2)
    dreamer_cfg.time_limit = getattr(dreamer_cfg, "time_limit", 1000)
    dreamer_cfg.steps = int(getattr(dreamer_cfg, "steps", 1_000_000) or 1_000_000)
    dreamer_cfg.steps //= dreamer_cfg.action_repeat
    dreamer_cfg.eval_every = int(getattr(dreamer_cfg, "eval_every", 10_000) or 10_000)
    dreamer_cfg.eval_every //= dreamer_cfg.action_repeat
    dreamer_cfg.log_every = int(getattr(dreamer_cfg, "log_every", 10_000) or 10_000)
    dreamer_cfg.log_every //= dreamer_cfg.action_repeat
    dreamer_cfg.prefill = getattr(dreamer_cfg, "prefill", 2500)
    dreamer_cfg.dataset_size = getattr(dreamer_cfg, "dataset_size", 1_000_000)
    dreamer_cfg.eval_episode_num = getattr(dreamer_cfg, "eval_episode_num", 10)
    dreamer_cfg.device = config.device
    dreamer_cfg.seed = config.seed

    workspace = Dreamerv3Workspace(dreamer_cfg, work_dir, full_config=config)

    class Runner:
        def run(self, args):
            if args.get("checkpoint") and args["checkpoint"]:
                workspace.load_snapshot(args["checkpoint"])
            if args.get("train", False):
                workspace.train()
            elif args.get("play", False):
                workspace.eval()

    return Runner()
