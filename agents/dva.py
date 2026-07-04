import copy
import os
import time

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import models
import wandb
from utils.common_utils import TimeReport, make_envs, print_info
from utils.statistic_utils import AverageMeter, RunningMeanStd
from utils.tensor_utils import compute_grad_norm, moveaxis_dict, stack_dict_list


class DVARunner:
    """D.Va: first-order analytic policy gradient through a differentiable simulator.

    Identical to SHAC except the actor's input observation is detached from the
    computation graph, so policy gradients flow only through the simulator dynamics.
    """

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
        # Keep wandb init simple: if a field is null, don't pass it (wandb will auto-generate).
        wandb_kwargs = {
            "project": wandb_config.get("project", "sdpg"),
            "entity": wandb_config.get("entity"),
            "group": wandb_config.get("group"),
            "job_type": wandb_config.get("job_type"),
            "name": wandb_config.get("name"),
            "tags": wandb_config.get("tags", []),
            "notes": wandb_config.get("notes"),
        }
        # Remove None values
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}

        wandb.init(**wandb_kwargs)

        # Log config if enabled
        if wandb_config.get("log_config", True):
            # Convert OmegaConf to dict for wandb
            config_dict = OmegaConf.to_container(config, resolve=True)
            wandb.config.update(config_dict)

        print_info("Wandb logging enabled")
        return True

    def _train_init(self):
        self.max_epochs = self.agent_config.max_epochs
        self.horizon_length = self.agent_config.horizon_length
        self.gamma = self.agent_config.gamma
        self.lam = self.agent_config.lam
        self.critic_method = self.agent_config.critic_method
        self.critic_iterations = self.agent_config.critic_iterations
        self.num_critic_batches = self.agent_config.num_critic_batches
        self.batch_size = self.num_envs * self.horizon_length // self.num_critic_batches
        self.target_critic_alpha = self.agent_config.target_critic_alpha
        self.truncate_grads = self.agent_config.truncate_grads
        self.grad_norm = self.agent_config.grad_norm
        self.lr_schedule = self.agent_config.lr_schedule
        self.actor_lr = self.agent_config.actor_lr
        self.critic_lr = self.agent_config.critic_lr
        self.reward_scale = self.agent_config.get("reward_scale", 1.0)

        # initialize the optimizer
        betas = tuple(self.agent_config.betas)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas=betas, lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas=betas, lr=self.critic_lr)

        # Normalization
        # NOTE: observation normalization is currently initialized during the make_models function
        if self.agent_config.ret_rms:
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)
        else:
            self.ret_rms = None

        # Buffers for critic training, tensors are in the shape of (horizon_length, num_envs, ...)
        self.obs_buf = {
            key: torch.zeros(
                (self.horizon_length, self.num_envs, *self.inputs_dim[key]), dtype=torch.float32, device=self.device
            )
            for key in self.critic_input_keys
        }
        self.rew_buf = torch.zeros((self.horizon_length, self.num_envs), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((self.horizon_length, self.num_envs), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((self.horizon_length, self.num_envs), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((self.horizon_length, self.num_envs), dtype=torch.float32, device=self.device)
        self.ret = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Performance metrics recorder
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
        self.step_count = 0
        self.iter_count = 0

        self.actor_grad_norm = 0.0
        self.critic_grad_norm = 0.0
        self.policy_std = 0.0

        # Initialize wandb if enabled
        self.use_wandb = self._init_wandb(self.config)

        # Timer
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
        value_loss: float,
        actor_grad_norm: float,
        critic_grad_norm: float,
        policy_std: float,
        fps: float,
        iter: int,
        step: int,
        time_elapse: float | None = None,
        policy_reward: float | None = None,
        episode_lengths: float | None = None,
        best_policy_reward: float | None = None,
        time_report: TimeReport | None = None,
    ):
        """Write training statistics to both TensorBoard and wandb."""
        metrics = {
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "policy_std": policy_std,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "fps": fps,
        }

        # Add optional metrics
        if policy_reward is not None:
            metrics["rewards"] = policy_reward
        if episode_lengths is not None:
            metrics["episode_lengths"] = episode_lengths
        if best_policy_reward is not None:
            metrics["best_policy"] = best_policy_reward

        # Log to TensorBoard with different step types
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
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = dict(metrics)
            wandb_metrics["env_step"] = step
            if time_elapse is not None:
                wandb_metrics["time"] = time_elapse
            wandb.log(wandb_metrics, step=iter)

    def make_envs(self):
        # enable differentiable simulation for first-order gradients
        OmegaConf.set_struct(self.config, False)
        self.config.task.config.sim_options.requires_grad = True
        OmegaConf.set_struct(self.config, True)
        self.env = make_envs(self.config)
        self.num_actions = self.env.num_actions

    def compute_actor_loss(self):
        rew_acc = torch.zeros((self.horizon_length + 1, self.num_envs), dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros((self.horizon_length + 1, self.num_envs), dtype=torch.float32, device=self.device)

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Freeze a copy of the running statistics: normalize with the frozen copy for the
            # whole horizon while the live statistics keep being updated with raw observations.
            obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between horizons (truncated BPTT)
        obs, _ = self.env.initialize_trajectory()
        self.update_running_statistics(obs)
        obs = self.process_observations(obs, obs_rms)

        for i in range(self.horizon_length):
            # collect data for critic training
            with torch.no_grad():
                for key in self.critic_input_keys:
                    self.obs_buf[key][i] = obs[key].clone()

            # D.Va: detach the actor's input observation from the computation graph,
            # gradients flow into the action only (through the simulator dynamics)
            out = self.actor(self.get_actor_obs(obs, detach=True))
            dist = torch.distributions.Normal(out["mean"], torch.exp(out["log_std"]))
            actions = dist.rsample()

            obs, rew, terminated, truncated, info = self.env.step(torch.tanh(actions), auto_reset=True)
            done = terminated | truncated

            with torch.no_grad():
                raw_rew = rew.clone()

            # scale the reward
            rew = rew * self.reward_scale

            # process the observations with the running statistics
            self.update_running_statistics(obs)
            obs = self.process_observations(obs, obs_rms)

            if self.ret_rms is not None:
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            # NOTE: keep graph-attached, terminal values backpropagate through the simulator
            next_values[i + 1] = self.target_critic(self.get_critic_obs(obs)).squeeze(-1)

            # handle terminated environments
            if len(done_env_ids) > 0:
                obs_before_reset = info["observations_before_reset"]
                for id in done_env_ids:
                    invalid_obs = False
                    for key in self.critic_input_keys:
                        obs_key = obs_before_reset[key][id]
                        if (
                            torch.isnan(obs_key).sum() > 0
                            or torch.isinf(obs_key).sum() > 0
                            or (torch.abs(obs_key) > 1e6).sum() > 0
                        ):  # ugly fix for nan values
                            invalid_obs = True
                            break
                    if invalid_obs:
                        next_values[i + 1, id] = 0.0
                    elif truncated[id]:  # timeout: bootstrap with the terminal value from the pre-reset observation
                        # (timeout takes precedence over termination on the same step, as in reference D.Va)
                        real_obs = {key: obs_before_reset[key][id : id + 1] for key in self.critic_input_keys}
                        real_obs = self.process_observations(real_obs, obs_rms)
                        next_values[i + 1, id] = self.target_critic(real_obs).squeeze(-1)
                    else:  # early termination
                        next_values[i + 1, id] = 0.0

            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print("next value error")
                raise ValueError("next value error")

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.horizon_length - 1:
                actor_loss = (
                    actor_loss
                    + (
                        -rew_acc[i + 1, done_env_ids]
                        - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                    ).sum()
                )
            else:
                # terminate all envs at the end of the optimization iteration
                actor_loss = actor_loss + (-rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.horizon_length - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

            # record the performance metrics (raw, unscaled rewards)
            with torch.no_grad():
                self.episode_reward += raw_rew
                if len(done_env_ids) > 0:
                    self.episode_reward_meter.update(self.episode_reward[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    self.episode_reward[done_env_ids] = 0.0
                    self.episode_length[done_env_ids] = 0.0

        actor_loss /= self.horizon_length * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        with torch.no_grad():
            self.policy_std = torch.exp(out["log_std"]).mean().item()

        self.step_count += self.horizon_length * self.num_envs

        return actor_loss

    def train_actor(self):
        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss()
        actor_loss.backward()

        with torch.no_grad():
            self.actor_grad_norm = compute_grad_norm(self.actor.parameters())
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
            # sanity check
            if torch.isnan(self.actor_grad_norm) or self.actor_grad_norm > 1e8:
                print("NaN gradient")
                raise ValueError("NaN gradient")

        self.actor_optimizer.step()
        return actor_loss.item()

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.horizon_length)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                    self.lam * self.gamma * Ai
                    + self.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                )
                Bi = (
                    self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i]))
                    + self.rew_buf[i]
                )
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError(f"Unknown critic method: {self.critic_method}")

    def train_critic(self):
        # Targets are computed once and frozen across all critic iterations
        with torch.no_grad():
            self.compute_target_values()
            obs = {key: value.flatten(0, 1).clone() for key, value in self.obs_buf.items()}
            target_values = self.target_values.flatten().clone()

        dataset_size = self.horizon_length * self.num_envs
        value_loss = 0.0
        for j in range(self.critic_iterations):
            total_critic_loss = 0.0
            batch_cnt = 0
            # sequential (unshuffled) minibatch split
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                obs_batch = {key: value[start_idx:end_idx] for key, value in obs.items()}
                target_values_batch = target_values[start_idx:end_idx]

                self.critic_optimizer.zero_grad()
                pred_values = self.critic(obs_batch).squeeze(-1)
                critic_loss = ((pred_values - target_values_batch) ** 2).mean()
                critic_loss.backward()

                # ugly fix for simulation nan problem
                for params in self.critic.parameters():
                    if params.grad is not None:
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                self.critic_grad_norm = compute_grad_norm(self.critic.parameters())
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                self.critic_optimizer.step()

                total_critic_loss += critic_loss
                batch_cnt += 1

            value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()

        return value_loss

    def train_epoch(self):
        # linear learning rate decay
        if self.lr_schedule == "linear":
            actor_lr = (1e-5 - self.actor_lr) * float(self.iter_count / self.max_epochs) + self.actor_lr
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = actor_lr
            critic_lr = (1e-5 - self.critic_lr) * float(self.iter_count / self.max_epochs) + self.critic_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = critic_lr

        # Train the actor
        self.time_report.start_timer("train_actor")
        actor_loss = self.train_actor()
        self.time_report.end_timer("train_actor")

        # Train the critic
        self.time_report.start_timer("train_critic")
        value_loss = self.train_critic()
        self.time_report.end_timer("train_critic")

        # update target critic (alpha multiplies the old target)
        with torch.no_grad():
            alpha = self.target_critic_alpha
            for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters(), strict=False):
                param_targ.data.mul_(alpha)
                param_targ.data.add_((1.0 - alpha) * param.data)

        self.iter_count += 1

        return actor_loss, value_loss

    @torch.no_grad()
    def evaluate_policy(self, maximum_trajectory_length=None, save_trajectory=True):
        """
        TODO currently this function is only for play mode to evaluate the trained policy.
        """
        episode_length = torch.zeros(self.num_envs, device=self.device)
        episode_length_meter = AverageMeter(1, 100).to(self.device)
        episode_reward = torch.zeros(self.num_envs, device=self.device)
        episode_reward_meter = AverageMeter(1, 100).to(self.device)
        save_trajectory = save_trajectory and hasattr(self.env, "get_states")
        if save_trajectory:
            states_history = []
        if maximum_trajectory_length is None:
            maximum_trajectory_length = self.env.episode_length

        obs, _ = self.env.reset()
        if save_trajectory:
            states_history.append(self.env.get_states())
        for t in range(maximum_trajectory_length):
            # process the observations with the running statistics
            obs = self.process_observations(obs, self.obs_rms)
            actions = self.actor(self.get_actor_obs(obs))["mean"]
            obs, rewards, terminated, truncated, info = self.env.step(torch.tanh(actions), auto_reset=True)
            dones = terminated | truncated
            done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
            episode_length += 1
            episode_reward += rewards
            if save_trajectory:
                states_history.append(self.env.get_states())
            if len(done_env_ids) > 0:
                episode_length_meter.update(episode_length[done_env_ids])
                episode_reward_meter.update(episode_reward[done_env_ids])
                episode_length[done_env_ids] = 0.0
                episode_reward[done_env_ids] = 0.0

        print_info(
            f"Episode length: {episode_length_meter.get_mean().item()}, Episode reward: {episode_reward_meter.get_mean().item()}"
        )

        if save_trajectory:
            eval_dir = os.path.join(self.log_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            save_path = os.path.join(eval_dir, "trajectory.pt")
            states_history = stack_dict_list(states_history)
            states_history = moveaxis_dict(states_history, source=0, destination=1)
            torch.save(states_history, save_path)

    def train(self):
        self._train_init()

        self.time_report.add_timer("train_actor")
        self.time_report.add_timer("train_critic")

        self.save(filename="initial_policy")

        self.env.reset()
        self.episode_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, device=self.device)
        self.step_count = 0
        self.iter_count = 0
        best_policy_reward = -float("inf")
        start_time = time.time()

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()
            actor_loss, value_loss = self.train_epoch()
            time_end_epoch = time.time()
            time_elapse = time.time() - start_time
            fps = self.horizon_length * self.num_envs / (time_end_epoch - time_start_epoch)

            # Prepare metrics for logging
            policy_reward = None
            episode_lengths = None
            current_best_policy_reward = None

            if self.episode_length_meter.current_size > 0:
                policy_reward = self.episode_reward_meter.get_mean().item()
                episode_lengths = self.episode_length_meter.get_mean().item()
                if policy_reward > best_policy_reward:
                    best_policy_reward = policy_reward
                    current_best_policy_reward = best_policy_reward
                    print_info("Save best policy with reward: {:.2f}".format(best_policy_reward))
                    self.save(filename="best_policy")
            else:
                policy_reward = float("inf")
                episode_lengths = 0

            # Logging to both TensorBoard and wandb
            self.write_stats(
                actor_loss=actor_loss,
                value_loss=value_loss,
                actor_grad_norm=self.actor_grad_norm,
                critic_grad_norm=self.critic_grad_norm,
                policy_std=self.policy_std,
                fps=fps,
                iter=self.iter_count,
                step=self.step_count,
                time_elapse=time_elapse,
                policy_reward=policy_reward if policy_reward != float("inf") else None,
                episode_lengths=episode_lengths if episode_lengths > 0 else None,
                best_policy_reward=current_best_policy_reward,
                time_report=self.time_report,
            )

            print(
                "iter {}: ep reward {:.2f}, ep len {:.1f}, actor loss {:.2f}, value loss {:.2f}, fps total {:.3g}, actor grad norm {:.2f}, critic grad norm {:.2f}, std: {:.2g}".format(
                    self.iter_count,
                    policy_reward,
                    episode_lengths,
                    actor_loss,
                    value_loss,
                    fps,
                    self.actor_grad_norm,
                    self.critic_grad_norm,
                    self.policy_std,
                )
            )

            if self.iter_count % self.save_frequency == 0 or self.iter_count == self.max_epochs - 1:
                self.save(filename="iter_{}_reward_{:.2f}".format(self.iter_count, policy_reward))

        self.time_report.report()

        if self.use_wandb:
            wandb.finish()

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
        self.target_critic = checkpoint[2].to(self.device)
        if checkpoint[3] is not None:
            self.obs_rms = {key: value.to(self.device) for key, value in checkpoint[3].items()}
        else:
            self.obs_rms = checkpoint[3]
        self.ret_rms = checkpoint[4].to(self.device) if checkpoint[4] is not None else checkpoint[4]

    def save(self, filename=None, save_dir=None):
        if save_dir is None:
            save_dir = self.nn_dir
        if filename is None:
            filename = "best_policy"
        torch.save(
            [self.actor, self.critic, self.target_critic, self.obs_rms, self.ret_rms],
            os.path.join(save_dir, "{}.pt".format(filename)),
        )

    def process_observations(self, obs, obs_rms=None):
        """
        This function processes the observations with the running statistics.
        NOTE: normalization must stay differentiable (statistics are constants),
        the gradient path through the observations is preserved.
        """
        if obs_rms is None:
            obs_rms = self.obs_rms

        for key in obs_rms.keys():
            if key in obs.keys():
                obs[key] = obs_rms[key].normalize(obs[key])
        return obs

    @torch.no_grad()
    def update_running_statistics(self, obs):
        """
        This function updates the running statistics.
        """
        for key in self.obs_rms.keys():
            self.obs_rms[key].update(obs[key])

    def get_actor_obs(self, obs, detach=False):
        """
        This function gets the actor observations.
        """
        actor_obs = {}
        for key in self.actor_input_keys:
            actor_obs[key] = obs[key].detach() if detach else obs[key]
        return actor_obs

    def get_critic_obs(self, obs):
        """
        This function gets the critic observations.
        """
        critic_obs = {}
        for key in self.critic_input_keys:
            critic_obs[key] = obs[key]
        return critic_obs

    def make_models(self):
        """
        This function makes the models for the runner.
        """

        self.model_config = self.agent_config.model
        self.actor_config = self.model_config.actor
        self.critic_config = self.model_config.critic

        # load the input keys and dimensions
        self.actor_input_keys = [input.name for input in self.actor_config.inputs]
        self.critic_input_keys = [input.name for input in self.critic_config.inputs]
        self.all_input_keys = list(dict.fromkeys(self.actor_input_keys + self.critic_input_keys))
        self.inputs_dim = {}
        for key in self.all_input_keys:
            self.inputs_dim[key] = self.env.observation_space[key].shape

        # Helper to find input config by name
        def find_input(inputs_list, name):
            return next(input for input in inputs_list if input.name == name)

        # load the running statistics
        self.obs_rms = {}
        for key in self.all_input_keys:
            if key in self.actor_input_keys:
                actor_input = find_input(self.actor_config.inputs, key)
                # sensitive checking, the running statistics for the same key should be the same for actor and critic
                if key in self.critic_input_keys:
                    critic_input = find_input(self.critic_config.inputs, key)
                    if actor_input.apply_rms != critic_input.apply_rms:
                        raise ValueError(
                            f"The running statistics for the key {key} are not consistent between actor and critic"
                        )

                if actor_input.apply_rms:
                    input_dim = self.inputs_dim[key]
                    self.obs_rms[key] = RunningMeanStd(shape=input_dim, device=self.device)
            else:
                critic_input = find_input(self.critic_config.inputs, key)
                if critic_input.apply_rms:
                    input_dim = self.inputs_dim[key]
                    self.obs_rms[key] = RunningMeanStd(shape=input_dim, device=self.device)

        # Build actor and critic using factory functions
        self.actor = models.actor.build_actor(
            actor_config=self.actor_config,
            inputs_dim=self.inputs_dim,
            num_actions=self.num_actions,
            device=self.device,
        )

        self.critic = models.critic.build_critic(
            critic_config=self.critic_config,
            inputs_dim=self.inputs_dim,
            device=self.device,
        )

        self.target_critic = copy.deepcopy(self.critic)


def make_runner(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        # Use runtime.output_dir so multirun jobs get their actual subdir (e.g. .../0, .../1)
        output_dir = hydra_cfg.runtime.output_dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)

    return DVARunner(config)
