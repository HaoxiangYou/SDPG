"""
Asymmetric SAC agent (visual RL baseline of arXiv:2412.13662, idea from Pinto et al. 2017):
the actor operates on pixel observations (drqv2-style encoder) while the twin Q critics
consume the low-dimensional privileged state. Off-policy SAC with autotuned entropy,
GPU-batched parallel simulation, and a CPU-resident replay buffer that stores the
frame-stacked pixel observation once per transition (the next stack is reconstructed
from the stored stack plus the newest next frame).
"""

import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import models
from utils.common_utils import TimeReport, make_envs, print_info
from utils.model_utils import get_activation_func, init_module
from utils.statistic_utils import AverageMeter, RunningMeanStd


class DoubleQCritic(nn.Module):
    """Twin Q networks on (privileged state, action); no visual encoder (asymmetric)."""

    def __init__(self, state_dim, num_actions, units, activation, use_layernorm=True, device="cuda:0"):
        super().__init__()

        def make_q():
            dims = [state_dim + num_actions] + list(units) + [1]
            mods = []
            for i in range(len(dims) - 1):
                lin = nn.Linear(dims[i], dims[i + 1])
                init_module(lin, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
                mods.append(lin)
                if i < len(dims) - 2:
                    mods.append(get_activation_func(activation))
                    if use_layernorm:
                        mods.append(nn.LayerNorm(dims[i + 1]))
            return nn.Sequential(*mods)

        self.q1 = make_q().to(device)
        self.q2 = make_q().to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    """Ring buffer over transitions from batched envs.

    Pixel memory layout: stores the full frame stack of obs (C, H, W) once and only the
    newest frame (C/num_stack channels) of next_obs; next stack = cat(obs[frame_ch:], next_frame).
    """

    def __init__(self, capacity, rgb_shape, num_image_stack, state_dim, num_actions, device="cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.idx = 0
        self.full = False

        c, h, w = rgb_shape
        assert c % num_image_stack == 0, "RGB channels must be divisible by frame stack"
        self.frame_channels = c // num_image_stack

        self.rgb = torch.zeros(self.capacity, c, h, w, dtype=torch.uint8, device=self.device)
        self.next_frame = torch.zeros(self.capacity, self.frame_channels, h, w, dtype=torch.uint8, device=self.device)
        self.state = torch.zeros(self.capacity, state_dim, dtype=torch.float32, device=self.device)
        self.next_state = torch.zeros(self.capacity, state_dim, dtype=torch.float32, device=self.device)
        self.action = torch.zeros(self.capacity, num_actions, dtype=torch.float32, device=self.device)
        self.reward = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.terminated = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, rgb, state, action, reward, terminated, next_rgb, next_state):
        n = rgb.shape[0]
        ids = (self.idx + torch.arange(n)) % self.capacity
        self.rgb[ids] = rgb.to(self.device)
        self.next_frame[ids] = next_rgb[:, -self.frame_channels :].to(self.device)
        self.state[ids] = state.to(self.device)
        self.next_state[ids] = next_state.to(self.device)
        self.action[ids] = action.to(self.device)
        self.reward[ids] = reward.to(self.device)
        self.terminated[ids] = terminated.to(self.device, dtype=torch.float32)
        self.idx = (self.idx + n) % self.capacity
        self.full = self.full or self.idx < n

    def sample(self, batch_size, device):
        ids = torch.randint(0, len(self), (batch_size,), device=self.device)
        rgb = self.rgb[ids]
        next_rgb = torch.cat([rgb[:, self.frame_channels :], self.next_frame[ids]], dim=1)
        return (
            rgb.to(device),
            self.state[ids].to(device),
            self.action[ids].to(device),
            self.reward[ids].to(device),
            self.terminated[ids].to(device),
            next_rgb.to(device),
            self.next_state[ids].to(device),
        )


class AsymSACRunner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.seed = config.seed
        self.device = self.config.device

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.agent_config = config.agent.config
        self.num_envs = self.agent_config.num_envs

        # make the environments
        self.make_envs()

        # make the models
        self.make_models()

        # Logger directory
        self.log_dir = config.log_dir

    def _init_wandb(self, config: DictConfig) -> bool:
        if not hasattr(config, "wandb") or not config.wandb.get("enable", False):
            return False

        wandb_config = config.wandb
        wandb_kwargs = {
            "project": wandb_config.get("project", "sdpg"),
            "entity": wandb_config.get("entity"),
            "group": wandb_config.get("group"),
            "job_type": wandb_config.get("job_type"),
            "name": wandb_config.get("name"),
            "tags": wandb_config.get("tags", []),
            "notes": wandb_config.get("notes"),
        }
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}

        wandb.init(**wandb_kwargs)

        if wandb_config.get("log_config", True):
            config_dict = OmegaConf.to_container(config, resolve=True)
            wandb.config.update(config_dict)

        print_info("Wandb logging enabled")
        return True

    def make_envs(self):
        self.env = make_envs(self.config)
        self.num_actions = self.env.num_actions

    def make_models(self):
        self.model_config = self.agent_config.model
        self.actor_config = self.model_config.actor
        self.critic_config = self.model_config.critic

        self.actor_input_keys = [input.name for input in self.actor_config.inputs]
        self.critic_input_keys = [input.name for input in self.critic_config.inputs]
        assert self.critic_input_keys == ["privileged_observations"], (
            "asym_sac critic expects a single privileged_observations input"
        )
        assert self.actor_input_keys == ["RGB"], "asym_sac actor expects a single RGB input"

        self.inputs_dim = {}
        for key in dict.fromkeys(self.actor_input_keys + self.critic_input_keys):
            self.inputs_dim[key] = self.env.observation_space[key].shape

        # Running statistics for the privileged state (critic input); RGB is normalized in the encoder
        self.obs_rms = {}
        for input in self.critic_config.inputs:
            if input.apply_rms:
                self.obs_rms[input.name] = RunningMeanStd(shape=self.inputs_dim[input.name], device=self.device)

        self.actor = models.actor.build_actor(
            actor_config=self.actor_config,
            inputs_dim=self.inputs_dim,
            num_actions=self.num_actions,
            device=self.device,
        )

        value_head = self.critic_config.value_head
        self.critic = DoubleQCritic(
            state_dim=int(np.prod(self.inputs_dim["privileged_observations"])),
            num_actions=self.num_actions,
            units=value_head.network.units,
            activation=value_head.network.activation,
            use_layernorm=value_head.network.get("layernorm", True),
            device=self.device,
        )
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        print("Critic (twin Q):")
        print(f"  Input keys: {self.critic_input_keys} + action")
        print(f"  Q network: {self.critic.q1}")

        # Entropy temperature (autotuned)
        init_temperature = float(self.agent_config.get("init_temperature", 0.1))
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(init_temperature), dtype=torch.float32, device=self.device)
        )
        self.target_entropy = float(self.agent_config.get("target_entropy", -self.num_actions))

        log_std_bounds = self.agent_config.get("log_std_bounds", [-5.0, 2.0])
        self.log_std_bounds = [float(log_std_bounds[0]), float(log_std_bounds[1])]

    # --- SAC action sampling (tanh-squashed Gaussian) ---

    def _actor_dist(self, actor_obs):
        out = self.actor(actor_obs)
        mean = out["mean"]
        log_std = torch.clamp(out["log_std"], self.log_std_bounds[0], self.log_std_bounds[1])
        return mean, log_std

    def sample_action(self, actor_obs, deterministic=False, with_log_prob=False):
        mean, log_std = self._actor_dist(actor_obs)
        if deterministic:
            action = torch.tanh(mean)
            return (action, None) if with_log_prob else action
        std = torch.exp(log_std)
        pre_tanh = mean + std * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if not with_log_prob:
            return action
        log_prob = (-0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # tanh correction: log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        log_prob -= (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(-1)
        return action, log_prob

    def rsample_action(self, actor_obs):
        """Reparameterized sample with log_prob (for actor/alpha updates)."""
        mean, log_std = self._actor_dist(actor_obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)
        log_prob = (-0.5 * eps**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob -= (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(-1)
        return action, log_prob

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def normalize_state(self, state):
        rms = self.obs_rms.get("privileged_observations")
        if rms is None:
            return state
        return rms.normalize(state)

    def get_actor_obs(self, obs):
        return {"RGB": obs["RGB"]}

    # --- Environment interaction ---

    @torch.no_grad()
    def step_env(self, actions):
        """Step the env `action_repeat` times with the same action.

        Envs that finish mid-repeat keep stepping with zero actions but their reward,
        termination flags and next observation are frozen at the step they finished
        (same semantics as the drqv2 ActionRepeatWrapper). No auto-reset: the returned
        observation for done envs is the pre-reset (terminal) observation.
        """
        acc_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        terminated_acc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated_acc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_obs = None

        for _ in range(self.action_repeat):
            step_actions = actions if not done.any() else torch.where(done.unsqueeze(-1), 0.0, actions)
            obs, rewards, terminated, truncated, infos = self.env.step(step_actions, auto_reset=False)
            if isinstance(infos, dict):
                self.infos.update(infos)
            active = ~done
            acc_reward += rewards * active
            terminated_acc |= terminated & active
            truncated_acc |= truncated & active
            if next_obs is None:
                next_obs = {k: v.clone() for k, v in obs.items()}
            else:
                for k in next_obs:
                    next_obs[k][active] = obs[k][active]
            done |= terminated | truncated
            if done.all():
                break

        return next_obs, acc_reward, terminated_acc, truncated_acc, done

    # --- Training ---

    def _train_init(self):
        self.max_env_steps = int(self.agent_config.total_env_steps)
        self.action_repeat = int(self.agent_config.get("action_repeat", 1))
        self.gamma = self.agent_config.gamma
        self.batch_size = self.agent_config.batch_size
        self.reward_scale = float(self.agent_config.get("reward_scale", 1.0))
        self.critic_tau = float(self.agent_config.get("critic_tau", 0.005))
        self.num_seed_steps = int(self.agent_config.get("num_seed_steps", 4000))
        self.truncate_grads = self.agent_config.get("truncate_grads", False)
        self.grad_norm = self.agent_config.get("grad_norm", 10.0)
        self.log_interval = int(self.agent_config.get("log_interval", 25))

        # Gradient updates per iteration: utd_ratio is counted in env steps (incl. action repeat)
        env_steps_per_iter = self.num_envs * self.action_repeat
        updates_per_iter = self.agent_config.get("updates_per_iter", None)
        if updates_per_iter is None:
            updates_per_iter = max(1, round(float(self.agent_config.get("utd_ratio", 0.25)) * env_steps_per_iter))
        self.updates_per_iter = int(updates_per_iter)

        # Optimizers
        lr = float(self.agent_config.get("lr", 3e-4))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.agent_config.get("actor_lr", lr)))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(self.agent_config.get("critic_lr", lr))
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(self.agent_config.get("alpha_lr", lr)))

        # Replay buffer
        rgb_shape = self.inputs_dim["RGB"]
        num_image_stack = int(getattr(self.env, "_num_image_stack", 3))
        self.buffer = ReplayBuffer(
            capacity=int(self.agent_config.replay_buffer_size),
            rgb_shape=rgb_shape,
            num_image_stack=num_image_stack,
            state_dim=int(np.prod(self.inputs_dim["privileged_observations"])),
            num_actions=self.num_actions,
            device=self.agent_config.get("buffer_device", "cpu"),
        )

        # Performance metrics recorder
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
        self.step_count = 0
        self.iter_count = 0
        self.infos = {}

        self.use_wandb = self._init_wandb(self.config)

        self.time_report = TimeReport()

        self.train_dir = os.path.join(self.log_dir, "training_logs")
        self.nn_dir = os.path.join(self.train_dir, "nn")
        self.summary_dir = os.path.join(self.train_dir, "summaries")
        if not os.path.exists(self.nn_dir):
            os.makedirs(self.nn_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.summary_writer = SummaryWriter(self.summary_dir)
        self.save_frequency = self.agent_config.save_frequency

    def write_stats(
        self,
        actor_loss: float,
        critic_loss: float,
        alpha_loss: float,
        q_value: float,
        entropy: float,
        temperature: float,
        iter: int,
        step: int,
        time_elapse: float | None = None,
        policy_reward: float | None = None,
        episode_lengths: float | None = None,
        best_policy_reward: float | None = None,
        buffer_size: int | None = None,
        fps: float | None = None,
        time_report: TimeReport | None = None,
        infos: dict | None = None,
    ):
        """Write training statistics to both TensorBoard and wandb (same pattern as sdpg.write_stats)."""
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "q_value": q_value,
            "entropy": entropy,
            "temperature": temperature,
        }

        if policy_reward is not None:
            metrics["rewards"] = policy_reward
        if episode_lengths is not None:
            metrics["episode_lengths"] = episode_lengths
        if best_policy_reward is not None:
            metrics["best_policy"] = best_policy_reward
        if buffer_size is not None:
            metrics["buffer_size"] = buffer_size
        if fps is not None:
            metrics["fps"] = fps

        if infos is not None:
            for key, value in infos.items():
                if isinstance(value, torch.Tensor):
                    if len(value.shape) == 0:
                        value = value.item()
                    else:
                        value = value.float().mean().item()
                elif isinstance(value, (list, tuple)):
                    value = sum(value) / len(value) if len(value) > 0 else float("nan")

                if isinstance(value, (float, int)):
                    if isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf")):
                        continue
                    metrics[f"{key}"] = value

        for metric_name, value in metrics.items():
            self.summary_writer.add_scalar(f"{metric_name}/iter", value, iter)
            self.summary_writer.add_scalar(f"{metric_name}/step", value, step)
            if time_elapse is not None:
                self.summary_writer.add_scalar(f"{metric_name}/time", value, time_elapse)

        if time_report is not None:
            for timer_name in time_report.timers.keys():
                self.summary_writer.add_scalar(
                    f"performance/{timer_name}_time", time_report.timers[timer_name].time_total, iter
                )

        if self.use_wandb:
            wandb_metrics = dict(metrics)
            wandb_metrics["env_step"] = step
            if time_elapse is not None:
                wandb_metrics["time"] = time_elapse
            wandb.log(wandb_metrics, step=iter)

    def update(self):
        """One SAC gradient update on a sampled batch. Returns metrics dict."""
        rgb, state, action, reward, terminated, next_rgb, next_state = self.buffer.sample(self.batch_size, self.device)
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        reward = reward * self.reward_scale

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.rsample_action({"RGB": next_rgb})
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2).squeeze(-1) - self.alpha.detach() * next_log_prob
            target_q = reward + self.gamma * (1.0 - terminated) * target_v

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1.squeeze(-1), target_q) + F.mse_loss(q2.squeeze(-1), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
        self.critic_optimizer.step()

        # Actor update (asymmetric: Q evaluated at the privileged state paired with the pixels)
        new_action, log_prob = self.rsample_action({"RGB": rgb})
        q1_pi, q2_pi = self.critic(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi).squeeze(-1)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        # Temperature update
        alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Target critic polyak update
        with torch.no_grad():
            for param, param_targ in zip(self.critic.parameters(), self.critic_target.parameters(), strict=False):
                param_targ.data.mul_(1.0 - self.critic_tau)
                param_targ.data.add_(self.critic_tau * param.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "q_value": q_pi.mean().item(),
            "entropy": -log_prob.mean().item(),
        }

    def train(self):
        self._train_init()

        self.time_report.add_timer("rollout")
        self.time_report.add_timer("train")

        self.save(filename="initial_policy")

        obs, _ = self.env.reset()
        rms = self.obs_rms.get("privileged_observations")
        if rms is not None:
            rms.update(obs["privileged_observations"])

        best_policy_reward = -float("inf")
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "alpha_loss": 0.0, "q_value": 0.0, "entropy": 0.0}
        start_time = time.time()
        last_log_time = start_time
        last_log_step = 0

        while self.step_count < self.max_env_steps:
            # --- Rollout ---
            self.time_report.start_timer("rollout")
            with torch.no_grad():
                if self.step_count < self.num_seed_steps:
                    actions = torch.rand(self.num_envs, self.num_actions, device=self.device) * 2.0 - 1.0
                else:
                    actions = self.sample_action(self.get_actor_obs(obs))

            next_obs, rewards, terminated, truncated, done = self.step_env(actions)

            self.buffer.add(
                rgb=obs["RGB"],
                state=obs["privileged_observations"],
                action=actions,
                reward=rewards,
                terminated=terminated,
                next_rgb=next_obs["RGB"],
                next_state=next_obs["privileged_observations"],
            )

            # Episode bookkeeping (raw rewards, env steps incl. action repeat)
            self.episode_reward += rewards
            self.episode_length += self.action_repeat
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
            if len(done_env_ids) > 0:
                self.episode_reward_meter.update(self.episode_reward[done_env_ids])
                self.episode_length_meter.update(self.episode_length[done_env_ids])
                self.episode_reward[done_env_ids] = 0.0
                self.episode_length[done_env_ids] = 0.0
                obs, _ = self.env.reset(env_ids=done_env_ids)
            else:
                obs = next_obs

            if rms is not None:
                rms.update(obs["privileged_observations"])

            self.step_count += self.num_envs * self.action_repeat
            self.iter_count += 1
            self.time_report.end_timer("rollout")

            # --- Updates ---
            self.time_report.start_timer("train")
            if self.step_count >= self.num_seed_steps:
                for _ in range(self.updates_per_iter):
                    metrics = self.update()
            self.time_report.end_timer("train")

            # --- Logging ---
            if self.iter_count % self.log_interval == 0:
                time_now = time.time()
                time_elapse = time_now - start_time
                fps = (self.step_count - last_log_step) / max(time_now - last_log_time, 1e-6)
                last_log_time, last_log_step = time_now, self.step_count

                policy_reward = None
                episode_lengths = None
                current_best_policy_reward = None
                if self.episode_length_meter.current_size > 0:
                    policy_reward = self.episode_reward_meter.get_mean().item()
                    episode_lengths = self.episode_length_meter.get_mean().item()
                    if policy_reward > best_policy_reward and self.step_count >= self.num_seed_steps:
                        best_policy_reward = policy_reward
                        current_best_policy_reward = best_policy_reward
                        print_info("Save best policy with reward: {:.2f}".format(best_policy_reward))
                        self.save(filename="best_policy")

                self.write_stats(
                    actor_loss=metrics["actor_loss"],
                    critic_loss=metrics["critic_loss"],
                    alpha_loss=metrics["alpha_loss"],
                    q_value=metrics["q_value"],
                    entropy=metrics["entropy"],
                    temperature=self.alpha.item(),
                    iter=self.iter_count,
                    step=self.step_count,
                    time_elapse=time_elapse,
                    policy_reward=policy_reward,
                    episode_lengths=episode_lengths,
                    best_policy_reward=current_best_policy_reward,
                    buffer_size=len(self.buffer),
                    fps=fps,
                    time_report=self.time_report,
                    infos=self.infos,
                )

                print(
                    "iter {}: step {}, ep reward {:.2f}, ep len {:.1f}, fps {:.1f}, buffer {}, actor loss {:.2f}, critic loss {:.2f}, q {:.2f}, entropy {:.2f}, alpha {:.3g}".format(
                        self.iter_count,
                        self.step_count,
                        policy_reward if policy_reward is not None else float("nan"),
                        episode_lengths if episode_lengths is not None else float("nan"),
                        fps,
                        len(self.buffer),
                        metrics["actor_loss"],
                        metrics["critic_loss"],
                        metrics["q_value"],
                        metrics["entropy"],
                        self.alpha.item(),
                    )
                )

            if self.iter_count % self.save_frequency == 0:
                reward_for_name = (
                    self.episode_reward_meter.get_mean().item() if self.episode_length_meter.current_size > 0 else 0.0
                )
                self.save(filename="iter_{}_reward_{:.2f}".format(self.iter_count, reward_for_name))

        self.time_report.report()
        self.save(filename="final_policy")

        if self.use_wandb:
            wandb.finish()

    # --- Evaluation ---

    @torch.no_grad()
    def evaluate_policy(self, maximum_trajectory_length=None):
        episode_length = torch.zeros(self.num_envs, device=self.device)
        episode_length_meter = AverageMeter(1, 100).to(self.device)
        episode_reward = torch.zeros(self.num_envs, device=self.device)
        episode_reward_meter = AverageMeter(1, 100).to(self.device)

        self.action_repeat = int(self.agent_config.get("action_repeat", 1))
        if maximum_trajectory_length is None:
            maximum_trajectory_length = self.env.episode_length

        obs, _ = self.env.reset()
        for _ in range(int(maximum_trajectory_length // self.action_repeat)):
            actions = self.sample_action(self.get_actor_obs(obs), deterministic=True)
            next_obs, rewards, terminated, truncated, done = self.step_env(actions)
            episode_length += self.action_repeat
            episode_reward += rewards
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
            if len(done_env_ids) > 0:
                episode_length_meter.update(episode_length[done_env_ids])
                episode_reward_meter.update(episode_reward[done_env_ids])
                episode_length[done_env_ids] = 0.0
                episode_reward[done_env_ids] = 0.0
                obs, _ = self.env.reset(env_ids=done_env_ids)
            else:
                obs = next_obs

        print_info(
            f"Episode length: {episode_length_meter.get_mean().item()}, "
            f"Episode reward: {episode_reward_meter.get_mean().item()}"
        )

    def play(self):
        self.evaluate_policy()

    def run(self, args):
        if "checkpoint" in args and args["checkpoint"] is not None and args["checkpoint"] != "":
            self.load(args["checkpoint"])

        if "train" in args and args["train"]:
            self.train()
        elif "play" in args and args["play"]:
            self.play()

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.critic_target = checkpoint[2].to(self.device)
        self.log_alpha = checkpoint[3].to(self.device)
        if checkpoint[4] is not None:
            self.obs_rms = {key: value.to(self.device) for key, value in checkpoint[4].items()}
        else:
            self.obs_rms = {}

    def save(self, filename=None, save_dir=None):
        if save_dir is None:
            save_dir = self.nn_dir
        if filename is None:
            filename = "best_policy"
        torch.save(
            [self.actor, self.critic, self.critic_target, self.log_alpha, self.obs_rms],
            os.path.join(save_dir, "{}.pt".format(filename)),
        )


def make_runner(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        # Use runtime.output_dir so multirun jobs get their actual subdir (e.g. .../0, .../1)
        output_dir = hydra_cfg.runtime.output_dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)

    # asym_sac requires pixel observations for the actor; force vis_obs and 84x84 RGB
    OmegaConf.set_struct(config, False)
    config.task.config.vis_obs = True
    if "sensors_args" in config.task.config:
        config.task.config.sensors_args.camera.res = [84, 84]
    else:
        config.task.config.setdefault("sensors_args", {})
        if "camera" not in config.task.config.sensors_args:
            config.task.config.sensors_args["camera"] = {}
        config.task.config.sensors_args["camera"]["res"] = [84, 84]
    OmegaConf.set_struct(config, True)

    return AsymSACRunner(config)
