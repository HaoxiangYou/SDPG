"""
DrQ-v2 agent: uses the drqv2 package (externals/drqv2, installed as dependency).
Wraps a pixel-observation backend env as dm_env inside this module (wrapper lives here, not in envs).
"""

import datetime
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, List, NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dm_env import StepType, specs
from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from envs.base_env import BaseEnv
from utils.common_utils import make_envs
from utils.statistic_utils import AverageMeter

# DrQ-v2 uses MKL and MUJOCO_GL; we don't use MuJoCo so set for headless
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

# Use installed drqv2 package (path dependency from main pyproject.toml)
from drqv2 import (
    ReplayBufferStorage,
    TrainVideoRecorder,
    VideoRecorder,
    make_replay_loader,
    utils as drqv2_utils,
)


# --- dm_env wrapper for pixel-observation backends (lives here; not in envs) ---


class ExtendedTimeStep(NamedTuple):
    """Time step with observation, reward, discount, step_type, action (for replay buffer)."""

    step_type: Any
    reward: Any
    raw_reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


class ActionRepeatWrapper:
    """Repeat a batch of actions while keeping episode boundaries per env intact."""

    def __init__(self, env, num_repeats: int):
        self._env = env
        self._num_repeats = num_repeats

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    def step(self, actions):
        # actions: (num_envs, action_dim)
        rewards = np.zeros(self._env.num_envs, dtype=np.float64)
        raw_rewards = np.zeros(self._env.num_envs, dtype=np.float64)
        discounts = np.ones(self._env.num_envs, dtype=np.float64)
        done = np.zeros(self._env.num_envs, dtype=bool)
        time_steps = None
        terminal_time_steps = [None] * self._env.num_envs
        for _ in range(self._num_repeats):
            step_actions = actions
            if done.any():
                if isinstance(actions, np.ndarray):
                    step_actions = np.array(actions, copy=True)
                    step_actions[done] = 0.0
                else:
                    step_actions = actions.clone()
                    done_mask = torch.from_numpy(done).to(
                        device=step_actions.device, dtype=torch.bool
                    )
                    step_actions[done_mask] = 0.0

            time_steps = self._env.step(step_actions)
            for j, ts in enumerate(time_steps):
                if done[j]:
                    continue
                raw_rewards[j] += float(ts.raw_reward or 0.0)
                # Keep the replay reward discounted within the repeated action,
                # but track raw episodic returns separately for logging.
                rewards[j] += float(ts.reward or 0.0) * discounts[j]
                discounts[j] *= float(ts.discount)
                if ts.last():
                    done[j] = True
                    terminal_time_steps[j] = ts._replace(
                        reward=float(rewards[j]),
                        raw_reward=float(raw_rewards[j]),
                        discount=float(discounts[j]),
                    )
            if done.all():
                break

        assert time_steps is not None
        if done.any():
            self._env.reset(env_ids=np.where(done)[0].astype(np.int32))
        return [
            terminal_time_steps[j]
            if terminal_time_steps[j] is not None
            else ts._replace(
                reward=float(rewards[j]),
                raw_reward=float(raw_rewards[j]),
                discount=float(discounts[j]),
            )
            for j, ts in enumerate(time_steps)
        ]

    def reset(self, env_ids=None):
        return self._env.reset(env_ids=env_ids)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class DrQV2EnvWrapper:
    """
    Wrap a batched BaseEnv that provides pixel (RGB) observations as dm_env for DrQ-v2.
    reset() and step() always return a list of ExtendedTimeStep (length num_envs).
    Expects RGB (9, 84, 84) per env; action is [-1, 1].
    """

    def __init__(
        self,
        base_env: BaseEnv,
        img_size: int = 84,
        discount: float = 0.99,
    ):
        # Dict space keys are in .spaces (gym/gymnasium)
        obs_spaces = getattr(base_env.observation_space, "spaces", {})
        assert "RGB" in obs_spaces, (
            "Env must have RGB in observation_space (e.g. vis_obs=True for the task)."
        )
        self._env = base_env
        self._img_size = img_size
        self._discount = discount
        self._num_envs = getattr(base_env, "_num_envs", base_env.num_envs)
        self._action_spec = specs.BoundedArray(
            shape=(base_env.num_actions,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )
        self._obs_spec = specs.BoundedArray(
            shape=(9, img_size, img_size),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._last_obs = None  # (9,H,W) first env, for render

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def _get_obs(
        self, obs_dict: dict, to_cpu: bool = True
    ) -> List:
        """Return observations as list of (9,H,W), one per env. to_cpu=True: numpy (replay); to_cpu=False: tensor on env device (act/step loop)."""
        rgb = obs_dict["RGB"]
        if isinstance(rgb, torch.Tensor):
            on_device = rgb.device
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            n = rgb.shape[0]
            if not to_cpu:
                out = []
                for i in range(n):
                    frame = rgb[i : i + 1]
                    if frame.shape[2] != self._img_size or frame.shape[3] != self._img_size:
                        frame = F.interpolate(
                            frame.float(),
                            size=(self._img_size, self._img_size),
                            mode="area",
                        ).clamp(0, 255).round().to(torch.uint8)
                    out.append(frame.squeeze(0))
                return out
            rgb = rgb.cpu().numpy()
        else:
            rgb = np.asarray(rgb, dtype=np.uint8, order="C")
            if rgb.ndim == 3:
                rgb = np.expand_dims(rgb, 0)
            n = rgb.shape[0]
        out = []
        for i in range(n):
            frame = rgb[i]
            if frame.shape[1] != self._img_size or frame.shape[2] != self._img_size:
                frame = np.stack(
                    [
                        cv2.resize(
                            frame[j],
                            (self._img_size, self._img_size),
                            interpolation=cv2.INTER_AREA,
                        )
                        for j in range(frame.shape[0])
                    ],
                    axis=0,
                )
            out.append(frame.astype(np.uint8))
        return out

    def reset(self, env_ids=None) -> List[ExtendedTimeStep]:
        env_ids_np = None
        if env_ids is not None:
            if torch.is_tensor(env_ids):
                env_ids = env_ids.to(device=self._env.device, dtype=torch.int64).view(-1)
            else:
                env_ids = torch.as_tensor(
                    np.asarray(env_ids, dtype=np.int64).ravel(),
                    device=self._env.device,
                    dtype=torch.int64,
                )
            env_ids_np = env_ids.detach().cpu().numpy()
        obs_dict, _ = self._env.reset(env_ids=env_ids)
        obs_list = self._get_obs(obs_dict, to_cpu=False)
        if env_ids is None:
            self._last_obs = obs_list[0] if obs_list else None
            num_reset_envs = self._num_envs
        else:
            num_reset_envs = len(env_ids_np)
            if num_reset_envs > 0:
                zero_idx = np.where(env_ids_np == 0)[0]
                if len(zero_idx) > 0:
                    self._last_obs = obs_list[int(zero_idx[0])]

        zero_action = torch.zeros(
            num_reset_envs,
            *self._action_spec.shape,
            dtype=torch.float32,
            device=self._env.device,
        )
        return [
            ExtendedTimeStep(
                step_type=StepType.FIRST,
                reward=0.0,
                raw_reward=0.0,
                discount=1.0,
                observation=obs_list[i],
                action=zero_action[i],
            )
            for i in range(num_reset_envs)
        ]

    def step(self, action) -> List[ExtendedTimeStep]:
        """action: (num_envs, action_dim) numpy or tensor on env device."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self._env.device)
        else:
            action = action.float().to(self._env.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = torch.clamp(action, -1.0, 1.0)
        obs_dict, reward, terminated, truncated, _ = self._env.step(
            action, auto_reset=False
        )
        done = terminated | truncated
        obs_list = self._get_obs(obs_dict, to_cpu=False)
        done_np = done.cpu().numpy().ravel()
        rewards = reward.cpu().numpy().ravel()
        self._last_obs = obs_list[0] if obs_list else None
        return [
            ExtendedTimeStep(
                step_type=StepType.LAST if done_np[i] else StepType.MID,
                reward=float(rewards[i]),
                raw_reward=float(rewards[i]),
                discount=0.0 if done_np[i] else self._discount,
                observation=obs_list[i],
                action=action[i],
            )
            for i in range(self._num_envs)
        ]

    def render(self, height: int = 84, width: int = 84) -> np.ndarray:
        """Return current observation as (H, W, 3) for video recorder (first env)."""
        last = self._last_obs
        if last is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        if torch.is_tensor(last):
            last = last.cpu().numpy()
        frame = last[-3:].transpose(1, 2, 0)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame


# --- DrQ-v2 Workspace and runner ---


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    import hydra
    return hydra.utils.instantiate(cfg)


class DrQv2Workspace:
    """
    DrQ-v2 Workspace that uses a single pixel dm_env (no separate train/eval envs).
    Train and play (eval) are mutually exclusive: train=True runs training only;
    play=True loads checkpoint and runs eval (see scripts/run.py and Runner.run).
    """

    def __init__(
        self,
        cfg,
        dm_env,
        work_dir: Path,
        full_config=None,
    ):
        self.work_dir = Path(work_dir)
        self.training_logs_dir = self.work_dir / "training_logs"
        self.summaries_dir = self.training_logs_dir / "summaries"
        self.nn_dir = self.training_logs_dir / "nn"
        self.buffer_dir = self.training_logs_dir / "buffer"
        for d in (self.summaries_dir, self.nn_dir, self.buffer_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.cfg = cfg
        drqv2_utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.env = dm_env
        self.num_envs = getattr(self.env, "num_envs", 1)
        # Agent updates per env step (reference: num_envs; can override via config)
        self.updates_per_step = getattr(cfg, "updates_per_step", None) or self.num_envs

        self.use_wandb = self._init_wandb(full_config) if full_config else False

        use_tb = getattr(cfg, "use_tb", True)
        self.summary_writer = SummaryWriter(str(self.summaries_dir)) if use_tb else None
        self.data_specs = (
            self.env.observation_spec(),
            self.env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )
        self.replay_storage = ReplayBufferStorage(
            self.data_specs,
            self.buffer_dir,
        )
        self._current_episodes = [defaultdict(list) for _ in range(self.num_envs)]
        # num_workers: use config value; with CUDA we rely on spawn (set in make_runner) so workers don't fork
        _num_workers = getattr(cfg, "replay_buffer_num_workers", 4)
        self.replay_loader = make_replay_loader(
            self.buffer_dir,
            getattr(cfg, "replay_buffer_size", 100_000),
            getattr(cfg, "batch_size", 256),
            _num_workers,
            getattr(cfg, "save_snapshot", False),
            getattr(cfg, "nstep", 3),
            getattr(cfg, "discount", 0.99),
        )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if getattr(cfg, "save_video", False) else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if getattr(cfg, "save_train_video", False) else None
        )

        self.agent = make_agent(
            self.env.observation_spec(),
            self.env.action_spec(),
            self.cfg.agent,
        )
        self.timer = drqv2_utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._iter_count = 0  # number of while-loop iterations in train()
        self.episode_raw_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

    def _init_wandb(self, config: DictConfig) -> bool:
        """Init Weights & Biases if config.wandb.enable is True (same pattern as agents/afrl.py)."""
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

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * getattr(self.cfg, "action_repeat", 1)

    @property
    def iter_count(self):
        """Number of completed training loop iterations (while train_until_step)."""
        return self._iter_count

    def write_stats(
        self,
        iter: int,
        step: int,
        time_elapse: float | None = None,
        policy_reward: float | None = None,
        episode_lengths: float | None = None,
        infos: dict | None = None,
        **extra_metrics,
    ):
        """Write training statistics to TensorBoard and wandb (same pattern as afrl.write_stats).
        Main metrics use iter/step/time axes; infos use 'info/' prefix and iter as x-axis."""
        metrics = dict(extra_metrics)
        if policy_reward is not None:
            metrics["rewards"] = policy_reward
        if episode_lengths is not None:
            metrics["episode_lengths"] = episode_lengths

        # Normalize infos to scalars (do not merge into metrics; log with info/ prefix and step as x-axis)
        info_scalars = {}
        if infos is not None:
            for key, value in infos.items():
                if isinstance(value, (int, float)):
                    info_scalars[key] = value
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    info_scalars[key] = value.item()

        if not metrics and not info_scalars:
            return

        if self.summary_writer is not None:
            for name, value in metrics.items():
                self.summary_writer.add_scalar(f"{name}/iter", value, iter)
                self.summary_writer.add_scalar(f"{name}/step", value, step)
                if time_elapse is not None:
                    self.summary_writer.add_scalar(f"{name}/time", value, time_elapse)
            for key, value in info_scalars.items():
                self.summary_writer.add_scalar(f"info/{key}", value, iter)

        if self.use_wandb:
            wandb_metrics = dict(metrics)
            wandb_metrics["env_step"] = step
            if time_elapse is not None:
                wandb_metrics["time"] = time_elapse
            for key, value in info_scalars.items():
                wandb_metrics[f"info/{key}"] = value
            wandb.log(wandb_metrics, step=iter)

        # Print key stats to terminal (similar to afrl)
        parts = [f"iter {iter}", f"step {step}"]
        if policy_reward is not None:
            parts.append(f"ep reward {policy_reward:.2f}")
        if episode_lengths is not None:
            parts.append(f"ep len {episode_lengths:.1f}")
        if infos:
            if "fps" in infos:
                parts.append(f"fps {infos['fps']:.1f}")
            if "buffer_size" in infos:
                parts.append(f"buffer {infos['buffer_size']}")
            if "episode" in infos:
                parts.append(f"episode {infos['episode']}")
        if time_elapse is not None:
            parts.append(f"time {datetime.timedelta(seconds=int(time_elapse))}")
        if extra_metrics:
            for k, v in extra_metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k} {v:.3g}")
                else:
                    parts.append(f"{k} {v}")
        print(" | ".join(parts))

    def process_time_steps(
        self, time_steps: List[ExtendedTimeStep]
    ) -> int:
        """Process step output (list of TimeSteps); store to replay; return number of episodes finished. Converts GPU tensors to CPU numpy for storage."""
        num_done = 0
        for idx, ts in enumerate(time_steps):
            for spec in self.data_specs:
                value = ts[spec.name]
                if torch.is_tensor(value):
                    value = value.cpu().numpy()
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                value = np.asarray(value, dtype=spec.dtype)
                assert spec.shape == value.shape
                self._current_episodes[idx][spec.name].append(value)
            if ts.last():
                episode = {
                    spec.name: np.array(
                        self._current_episodes[idx][spec.name], spec.dtype
                    )
                    for spec in self.data_specs
                }
                self.replay_storage.store_episode(episode)
                self._current_episodes[idx] = defaultdict(list)
                num_done += 1
        return num_done

    def eval(self):
        """Eval runs in parallel (all envs); collect episodes until num_eval_episodes."""
        num_eval_episodes = getattr(self.cfg, "num_eval_episodes", 10)
        action_repeat = getattr(self.cfg, "action_repeat", 1)
        eval_until = drqv2_utils.Until(num_eval_episodes)

        episode_count = 0
        total_reward = 0.0
        total_episode_length = 0  # in env steps (before action_repeat)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        video_initialized = False

        time_steps = self.env.reset()
        while eval_until(episode_count):
            obs_batch = torch.stack([ts.observation for ts in time_steps])
            with torch.no_grad(), drqv2_utils.eval_mode(self.agent):
                actions = self.agent.act(
                    obs_batch, self.global_step, eval_mode=True, return_numpy=False
                )
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

            if not video_initialized:
                self.video_recorder.init(self.env, enabled=True)
                video_initialized = True
            self.video_recorder.record(self.env)

            time_steps = self.env.step(actions)
            for j in range(self.num_envs):
                episode_lengths[j] += 1
                if time_steps[j].last():
                    episode_count += 1
                    total_reward += time_steps[j].reward
                    total_episode_length += episode_lengths[j] * action_repeat
                    episode_lengths[j] = 0

            if episode_count > 0 and episode_count % max(1, self.num_envs) == 0:
                self.video_recorder.save(f"{self.global_frame}.mp4")

        eval_reward = total_reward / max(1, episode_count)
        eval_ep_len = total_episode_length / max(1, episode_count)
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("eval/rewards/step", eval_reward, self.global_frame)
            self.summary_writer.add_scalar("eval/episode_lengths/step", eval_ep_len, self.global_frame)
        if self.use_wandb:
            wandb.log(
                {"eval/rewards": eval_reward, "eval/episode_lengths": eval_ep_len, "env_step": self.global_step},
                step=self.global_frame,
            )

    def train(self):
        action_repeat = getattr(self.cfg, "action_repeat", 1)
        train_until_step = drqv2_utils.Until(
            getattr(self.cfg, "num_train_frames", 1_000_000),
            action_repeat,
        )
        seed_until_step = drqv2_utils.Until(
            getattr(self.cfg, "num_seed_frames", 4000),
            action_repeat,
        )
        save_snapshot_every_frames = getattr(
            self.cfg, "save_snapshot_every_frames", 50_000
        )

        time_steps = self.env.reset()
        num_done = self.process_time_steps(time_steps)
        self._global_episode += num_done
        episode_raw_rewards = np.zeros(self.num_envs)
        episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        metrics = None

        best_policy_reward = -float("inf")
        last_policy_reward = 0.0  # for snapshot filenames when we save outside write_stats
        if getattr(self.cfg, "save_snapshot", False):
            self.save_snapshot("initial_snapshot.pt")

        next_save_step = save_snapshot_every_frames
        while train_until_step(self.global_step):
            self._iter_count += 1
            # Batch act (obs/actions stay on GPU; CPU copy only when storing to replay)
            obs_batch = torch.stack([ts.observation for ts in time_steps])
            with torch.no_grad(), drqv2_utils.eval_mode(self.agent):
                actions = self.agent.act(
                    obs_batch, self.global_step, eval_mode=False, return_numpy=False
                )
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

            # Multiple agent updates per step (configurable via updates_per_step; default num_envs)
            if not seed_until_step(self.global_step):
                for i in range(self.updates_per_step):
                    metrics = self.agent.update(self.replay_iter, i)
                if metrics:
                    scalar_metrics = {
                        k: (v.item() if isinstance(v, torch.Tensor) else v)
                        for k, v in metrics.items()
                    }
                    if self.summary_writer is not None:
                        for k, v in scalar_metrics.items():
                            self.summary_writer.add_scalar(f"{k}/step", v, self.global_frame)
                            self.summary_writer.add_scalar(f"{k}/iter", v, self.iter_count)
                    if self.use_wandb:
                        wandb.log(scalar_metrics, step=self.iter_count)

            time_steps = self.env.step(actions)
            for j, ts in enumerate(time_steps):
                episode_raw_rewards[j] += ts.raw_reward
                episode_steps[j] += 1
            num_done = self.process_time_steps(time_steps)
            self._global_episode += num_done

            self._global_step += self.num_envs

            if (
                getattr(self.cfg, "save_snapshot", False)
                and not seed_until_step(self.global_step)
                and self.global_step >= next_save_step
            ):
                self.save_snapshot(
                    "iter_{}_reward_{:.2f}.pt".format(self.iter_count, last_policy_reward)
                )
                next_save_step = (
                    self.global_step // save_snapshot_every_frames + 1
                ) * save_snapshot_every_frames

            # Compute done mask every time so we always reset episode counters when an episode ends.
            # (Otherwise envs that finish during seed phase never get reset and ep_len accumulates.)
            done_mask = np.array([ts.last() for ts in time_steps])
            if np.any(done_mask):
                done_ids = np.where(done_mask)[0]
                raw_rewards_done = torch.tensor(
                    [episode_raw_rewards[j] for j in done_ids],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1)
                lengths_done = torch.tensor(
                    [episode_steps[j] * action_repeat for j in done_ids],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1)
                self.episode_raw_reward_meter.update(raw_rewards_done)
                self.episode_length_meter.update(lengths_done)
            if num_done > 0 and metrics is not None and np.any(done_mask):
                elapsed_time, total_time = self.timer.reset()
                policy_reward = (
                    self.episode_raw_reward_meter.get_mean().item()
                    if self.episode_raw_reward_meter.current_size > 0
                    else None
                )
                episode_lengths = (
                    self.episode_length_meter.get_mean().item()
                    if self.episode_length_meter.current_size > 0
                    else None
                )
                fps = self.num_envs * action_repeat / max(elapsed_time, 1e-6)
                if policy_reward is not None:
                    last_policy_reward = policy_reward
                self.write_stats(
                    iter=self.iter_count,
                    step=self.global_step,
                    time_elapse=total_time,
                    policy_reward=policy_reward,
                    episode_lengths=episode_lengths,
                    infos={
                        "fps": fps,
                        "buffer_size": len(self.replay_storage),
                        "episode": self.global_episode,
                    },
                )
                # Save best policy (same pattern as afrl)
                if policy_reward is not None and policy_reward > best_policy_reward:
                    best_policy_reward = policy_reward
                    if getattr(self.cfg, "save_snapshot", False):
                        self.save_snapshot("best_policy.pt")
                        print(f"Save best policy with reward: {best_policy_reward:.2f}")
            if np.any(done_mask):
                for j in range(self.num_envs):
                    if done_mask[j]:
                        episode_raw_rewards[j] = 0.0
                        episode_steps[j] = 0

        if getattr(self.cfg, "save_snapshot", False):
            self.save_snapshot(
                "iter_{}_reward_{:.2f}.pt".format(self.iter_count, last_policy_reward)
            )

        if self.use_wandb:
            wandb.finish()

    def save_snapshot(self, filename=None):
        """Save agent and state. filename=None -> snapshot.pt; else nn_dir/filename (e.g. initial_snapshot.pt, last_snapshot.pt)."""
        path = self.nn_dir / (filename or "snapshot.pt")
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode", "_iter_count"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, path)

    def load_snapshot(self, path=None):
        path = path or self.nn_dir / "snapshot.pt"
        if not Path(path).exists():
            return
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v


def make_runner(config: DictConfig):
    """Build DrQ-v2 runner using pixel-observation env (make_envs) and drqv2 package."""
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        output_dir = hydra_cfg.runtime.output_dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)

    OmegaConf.set_struct(config, False)
    # Use 'spawn' for DataLoader workers when on CUDA so workers don't inherit CUDA context
    # (Backend env may init CUDA at creation; fork-after-CUDA is unsafe.)
    _device = str(getattr(config, "device", "cpu")).lower()
    _num_workers = getattr(config.agent.config, "replay_buffer_num_workers", 4)
    if ("cuda" in _device or _device == "cuda") and _num_workers > 0:
        import torch.multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except TypeError:
            try:
                _mp.set_start_method("spawn")  # Python < 3.8 has no force=
            except RuntimeError:
                pass
        except RuntimeError:
            pass  # already set
    # DrQ-v2 requires vis_obs and 84x84 RGB; num_envs from config (supports parallel simulation)
    config.task.config.vis_obs = True
    if "sensors_args" in config.task.config:
        config.task.config.sensors_args.camera.res = [84, 84]
    else:
        config.task.config.setdefault("sensors_args", {})
        if "camera" not in config.task.config.sensors_args:
            config.task.config.sensors_args["camera"] = {}
        config.task.config.sensors_args["camera"]["res"] = [84, 84]
    _num_envs = getattr(config.task.config, "num_envs", 1)
    config.agent.config.num_envs = _num_envs
    OmegaConf.set_struct(config, True)

    base_env = make_envs(config)
    dm_env = DrQV2EnvWrapper(
        base_env,
        img_size=84,
        discount=float(getattr(config.agent.config, "discount", 0.99)),
    )
    action_repeat = max(1, int(getattr(config.agent.config, "action_repeat", 1)))
    dm_env = ActionRepeatWrapper(dm_env, action_repeat)

    work_dir = Path(getattr(config, "log_dir", Path.cwd()))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build DrQ-v2 style config (agent expects hydra-instantiate style)
    cfg = OmegaConf.create({
        "seed": config.seed,
        "device": config.device,
        "use_tb": getattr(config.agent.config, "use_tb", True),
        "save_video": getattr(config.agent.config, "save_video", False),
        "save_train_video": False,
        "save_snapshot": getattr(config.agent.config, "save_snapshot", True),
        "save_snapshot_every_frames": getattr(
            config.agent.config, "save_snapshot_every_frames", 50_000
        ),
        "replay_buffer_size": getattr(config.agent.config, "replay_buffer_size", 100_000),
        "replay_buffer_num_workers": getattr(config.agent.config, "replay_buffer_num_workers", 4),
        "batch_size": getattr(config.agent.config, "batch_size", 256),
        "nstep": getattr(config.agent.config, "nstep", 3),
        "discount": getattr(config.agent.config, "discount", 0.99),
        "action_repeat": getattr(config.agent.config, "action_repeat", 1),
        "num_train_frames": getattr(config.agent.config, "num_train_frames", 1_000_000),
        "num_seed_frames": getattr(config.agent.config, "num_seed_frames", 4000),
        "updates_per_step": getattr(config.agent.config, "updates_per_step", None),
        "agent": {
            "_target_": "drqv2.agent.DrQV2Agent",
            "obs_shape": None,
            "action_shape": None,
            "device": "${device}",
            "lr": getattr(config.agent.config, "lr", 1e-4),
            "feature_dim": getattr(config.agent.config, "feature_dim", 50),
            "hidden_dim": getattr(config.agent.config, "hidden_dim", 1024),
            "critic_target_tau": getattr(config.agent.config, "critic_target_tau", 0.01),
            "num_expl_steps": getattr(config.agent.config, "num_expl_steps", 2000),
            "update_every_steps": getattr(config.agent.config, "update_every_steps", 2),
            "stddev_schedule": getattr(
                config.agent.config,
                "stddev_schedule",
                "linear(0.2,0.05,500000)",
            ),
            "stddev_clip": getattr(config.agent.config, "stddev_clip", 0.3),
            "use_tb": getattr(config.agent.config, "use_tb", True),
        },
    })
    OmegaConf.resolve(cfg)

    workspace = DrQv2Workspace(cfg, dm_env, work_dir, full_config=config)

    class Runner:
        def run(self, args):
            if args.get("checkpoint") and args["checkpoint"]:
                workspace.load_snapshot(args["checkpoint"])
            if args.get("train", False):
                workspace.train()
            elif args.get("play", False):
                workspace.eval()

    return Runner()
