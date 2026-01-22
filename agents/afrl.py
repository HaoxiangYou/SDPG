import copy
import math
import os
import time

import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import models
import wandb
from utils.common_utils import TimeReport, make_envs, print_info
from utils.statistic_utils import AverageMeter, RunningMeanStd
from utils.tensor_utils import (
    assign_row_intervals,
    clone_dict_tensors,
    compute_grad_norm,
    flatten_dict,
    moveaxis_dict,
    select_entries,
    stack_dict_list,
)


class AFRLRunner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.seed = config.seed
        self.device = self.config.device

        self.agent_config = config.agent.config
        # number of the nominal environments
        self.num_base_envs = self.agent_config.num_base_envs
        self.num_action_perturbations = self.agent_config.num_action_perturbations
        # NOTE: for training, num_envs = num_base_envs * (num_action_perturbations + 1)
        # for evaluation only, however, num_envs may be different from num_base_envs * (num_action_perturbations + 1)
        self.num_envs = self.agent_config.num_envs
        self.nominal_env_ids = torch.arange(self.num_base_envs, device=self.device, dtype=torch.int32) * (
            self.num_action_perturbations + 1
        )
        self.max_epochs = self.agent_config.max_epochs
        self.horizon_length = self.agent_config.horizon_length
        # action perturbation factor
        self.delta = self.agent_config.delta
        self.gamma = self.agent_config.gamma
        self.lam = self.agent_config.lam
        self.reward_scale = self.agent_config.reward_scale
        self.target_critic_alpha = self.agent_config.target_critic_alpha
        self.truncated_grads = self.agent_config.truncated_grads
        self.grad_norm = self.agent_config.grad_norm
        self.mini_batch_size = self.agent_config.mini_batch_size
        self.critic_iterations = self.agent_config.critic_iterations

        # make the environments
        self.make_envs()

        # make the models
        self.make_models()

        # initialize the optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.agent_config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.agent_config.critic_lr)

        # Initialize learning rate schedulers
        self.actor_lr_scheduler = self._create_lr_scheduler(
            self.actor_optimizer, self.agent_config.get("actor_lr_schedule", {})
        )
        self.critic_lr_scheduler = self._create_lr_scheduler(
            self.critic_optimizer, self.agent_config.get("critic_lr_schedule", {})
        )

        # Normalization
        # NOTE: observation normalization is currently initialized during the make_models function
        if self.agent_config.ret_rms:
            self.ret_rms = RunningMeanStd(shape=(1,), device=self.device)
        else:
            self.ret_rms = None
        if self.agent_config.normalize_delta_J:
            self.normalize_delta_J = True
        else:
            self.normalize_delta_J = False

        # Top k perturbations for ascent direction computation
        top_k_perturbations = self.agent_config.get("top_k_perturbations", None)
        if top_k_perturbations is None:
            top_k_perturbations = self.num_action_perturbations + 1
        self.top_k_perturbations = top_k_perturbations

        # Performance metrics recorder
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length = torch.zeros(self.num_base_envs, device=self.device)
        self.episode_reward = torch.zeros(self.num_base_envs, device=self.device)
        self.step_count = 0
        self.iter_count = 0

        # Logger directory
        self.log_dir = config.log_dir
        self.train_dir = os.path.join(self.log_dir, "train")
        self.nn_dir = os.path.join(self.train_dir, "nn")
        self.summary_dir = os.path.join(self.train_dir, "summary")
        if not os.path.exists(self.nn_dir):
            os.makedirs(self.nn_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.summary_writer = SummaryWriter(self.summary_dir)
        self.save_frequency = self.agent_config.save_frequency

        # Initialize wandb if enabled
        self.use_wandb = self._init_wandb(config)

        # Buffer
        self.obs_buf = torch.zeros(
            (self.num_envs, self.horizon_length, self.num_observations), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_envs, self.horizon_length, self.num_actions), dtype=torch.float32, device=self.device
        )
        self.eps_actions = torch.zeros(
            (self.num_envs, self.horizon_length, self.num_actions), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(self.num_envs, self.horizon_length, device=self.device)
        self.ret = torch.zeros(self.num_envs, device=self.device)
        self.next_values = torch.zeros(self.num_envs, self.horizon_length, device=self.device)
        self.target_values = torch.zeros(self.num_envs, self.horizon_length, device=self.device)
        self.dones = torch.zeros(self.num_envs, self.horizon_length, device=self.device, dtype=torch.bool)
        self.delta_J = torch.zeros(self.num_envs, self.horizon_length, device=self.device)

        # Timer
        self.time_report = TimeReport()

    def _init_wandb(self, config: DictConfig) -> bool:
        if not hasattr(config, "wandb") or not config.wandb.get("enabled", False):
            return False

        wandb_config = config.wandb
        wandb_kwargs = {
            "project": wandb_config.get("project", "afrl"),
            "name": wandb_config.get("name"),
            "tags": wandb_config.get("tags", []),
            "notes": wandb_config.get("notes"),
        }
        # Remove None values
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        if wandb_config.get("entity"):
            wandb_kwargs["entity"] = wandb_config.entity

        wandb.init(**wandb_kwargs)

        # Log config if enabled
        if wandb_config.get("log_config", True):
            # Convert OmegaConf to dict for wandb
            config_dict = OmegaConf.to_container(config, resolve=True)
            wandb.config.update(config_dict)

        print_info("Wandb logging enabled")
        return True

    def _create_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, schedule_config: dict
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create a learning rate scheduler based on configuration.

        Args:
            optimizer: The optimizer to schedule
            schedule_config: Dictionary containing scheduler configuration

        Returns:
            Learning rate scheduler or None if no schedule is configured
        """
        schedule_name = schedule_config.get("name")
        if schedule_name is None or schedule_name == "null":
            return None

        if schedule_name == "cosine":
            # Cosine schedule with warmup: warmup -> cosine annealing
            warmup_epochs = schedule_config.get("warmup_epochs", 0)
            T_max = schedule_config.get("T_max", self.max_epochs)
            eta_min = schedule_config.get("eta_min", 1e-5)

            # Get initial learning rate from optimizer
            initial_lr = optimizer.param_groups[0]["lr"]
            warmup_start_lr = schedule_config.get("warmup_start_lr", 0.0)

            # Use a single LambdaLR to handle all phases (warmup, cosine, constant)
            # This avoids SequentialLR overhead from checking milestones
            eta_min_ratio = eta_min / initial_lr
            warmup_start_ratio = warmup_start_lr / initial_lr if warmup_start_lr > 0 else 0.0

            def cosine_with_warmup_lambda(epoch):
                # Phase 1: Warmup (linear from warmup_start_lr to initial_lr)
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    if warmup_epochs == 1:
                        return 1.0
                    # Linear interpolation from warmup_start_ratio to 1.0
                    return warmup_start_ratio + (1.0 - warmup_start_ratio) * epoch / (warmup_epochs - 1)

                # Phase 2: Cosine annealing (from initial_lr to eta_min)
                cosine_start_epoch = warmup_epochs
                cosine_T_max = T_max - warmup_epochs
                cosine_epoch = epoch - cosine_start_epoch

                if cosine_epoch < cosine_T_max:
                    # Cosine annealing: eta_min + (initial_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
                    cosine_factor = (1 + math.cos(math.pi * cosine_epoch / cosine_T_max)) / 2
                    return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor

                # Phase 3: Constant at eta_min
                return eta_min_ratio

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=cosine_with_warmup_lambda,
            )
        elif schedule_name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=schedule_config.get("start_factor", 1.0),
                end_factor=schedule_config.get("end_factor", 0.1),
                total_iters=schedule_config.get("T_max", self.max_epochs),
            )
        else:
            raise ValueError(f"Unknown learning rate schedule: {schedule_name}. Options: null, linear, cosine")

    def write_stats(
        self,
        actor_loss: float,
        critic_loss: float,
        rollout_reward: float,
        rollout_var: float,
        positive_rollout_ratio: float,
        actor_grad_norm: float,
        critic_grad_norm: float,
        iter: int,
        step: int,
        time_elapse: float | None = None,
        policy_reward: float | None = None,
        episode_lengths: float | None = None,
        best_policy_reward: float | None = None,
    ):
        """Write training statistics to both TensorBoard and wandb.

        Args:
            actor_loss: Actor loss value
            critic_loss: Critic loss value
            rollout_reward: Rollout reward
            rollout_var: Variance of rollout rewards
            positive_rollout_ratio: Ratio of positive rollout rewards
            actor_grad_norm: Actor gradient norm
            critic_grad_norm: Critic gradient norm
            iter: Iteration number
            step: Environment step number
            time_elapse: Elapsed time (optional)
            policy_reward: Policy reward (optional)
            episode_lengths: Episode lengths (optional)
            best_policy_reward: Best policy reward (optional)
        """
        # Prepare metrics dictionary
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "rollout_reward": rollout_reward,
            "rollout_reward_var": rollout_var,
            "rollout_reward_positive_ratio": positive_rollout_ratio,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
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

        # Log to wandb
        if self.use_wandb:
            wandb_metrics = dict(metrics)
            wandb_metrics["env_step"] = step
            if time_elapse is not None:
                wandb_metrics["time"] = time_elapse
            wandb.log(wandb_metrics, step=iter)

    def make_envs(self):
        # rewrite the nominal env ids in env config to match the num_base_envs
        self.config.task.config.nominal_env_ids = list(range(self.num_base_envs))
        self.env = make_envs(self.config)
        self.num_observations = self.env.num_observations
        self.num_actions = self.env.num_actions

    def rollout(self):
        """
        Rollout the trajectories for training and play the training dataset.
        """
        with torch.no_grad():
            # Initialize buffer for the observations
            obs_buf = []

            obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

            # initialize trajectory by resetting the auxiliary environments to the same state as the nominal environment
            obs = self.initialize_trajectory()
            self.update_running_statistics(obs)
            obs = self.process_observations(obs, obs_rms)

            # return of the short rollout (for logging purposes)
            rollout_reward = 0.0

            for i in range(self.horizon_length):
                obs_buf.append(clone_dict_tensors(obs))

                # Compute the nominal actions
                nominal_actions = self.actor(select_entries(obs, self.nominal_env_ids))
                nominal_actions = nominal_actions.repeat_interleave(self.num_action_perturbations + 1, dim=0)

                # Sample the action perturbations
                eps_actions = torch.randn_like(nominal_actions)
                eps_actions[self.nominal_env_ids] = 0.0
                actions = nominal_actions + eps_actions * self.delta
                self.actions[:, i] = actions.clone()
                self.eps_actions[:, i] = eps_actions.clone()

                # Step the environment
                # TODO: currently assume the action is bounded by [-1, 1], and we step using tanh
                obs, rewards, terminated, truncated, info = self.env.step(torch.tanh(actions), auto_reset=False)

                # Normalize the reward
                raw_rewards = rewards.clone()
                rewards = rewards * self.reward_scale
                if self.ret_rms is not None:
                    # Coarse but simple estimation of the return
                    self.ret = self.ret * self.gamma + rewards
                    self.ret_rms.update(self.ret)
                    rewards = rewards / torch.sqrt(ret_var + 1e-6)
                self.rewards[:, i] = rewards.clone()

                # Compute the next value
                next_values = torch.zeros(self.num_envs, device=self.device)
                non_terminated_env_ids = (~terminated).nonzero(as_tuple=False).squeeze(-1)
                next_values[non_terminated_env_ids] = self.target_critic(
                    self.process_observations(select_entries(obs, non_terminated_env_ids), obs_rms)
                ).squeeze(-1)
                if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                    print("next value error")
                    raise ValueError("next value error")
                self.next_values[:, i] = next_values.clone()

                # Handle the done and reset
                dones = terminated | truncated
                obs, dones = self.env_reset(dones)
                if i < self.horizon_length - 1:
                    self.dones[:, i] = dones.clone()
                else:
                    self.dones[:, i] = True

                # process the observations with the running statistics
                self.update_running_statistics(obs)
                obs = self.process_observations(obs, obs_rms)

                # Record the performance metrics
                self.episode_length += 1
                self.episode_reward += raw_rewards[self.nominal_env_ids]
                rollout_reward += raw_rewards[self.nominal_env_ids].sum().item()
                nominal_done_env_ids = dones[self.nominal_env_ids].nonzero(as_tuple=False).squeeze(-1)
                if len(nominal_done_env_ids) > 0:
                    self.episode_reward_meter.update(self.episode_reward[nominal_done_env_ids])
                    self.episode_length_meter.update(self.episode_length[nominal_done_env_ids])
                    self.episode_length[nominal_done_env_ids] = 0.0
                    self.episode_reward[nominal_done_env_ids] = 0.0

            self.step_count += self.num_envs * self.horizon_length

            # Store observation buffer for training
            # tensors in the obs_buf are in the shape of (num_envs, horizon_length, ...)
            self.obs_buf = moveaxis_dict(stack_dict_list(obs_buf, dim=0), source=0, destination=1)

            return rollout_reward / self.num_base_envs

    def compute_delta_J(self):
        """
        This function computes the incremental trajectory rewards of each action compared to the nominal action.
        """
        # Initialize the incremental trajectory rewards
        self.delta_J[:] = 0.0
        # NOTE: next value is zero if done
        curr_J = self.next_values[:, -1].clone()

        # The end of the current trajectory idx
        traj_end_ids = torch.ones(self.num_envs, dtype=torch.int, device=self.device) * self.horizon_length

        for t in reversed(range(self.horizon_length)):
            dones_env_ids = self.dones[:, t].nonzero(as_tuple=False).squeeze(-1)

            if len(dones_env_ids) > 0 and t < self.horizon_length - 1:
                assign_row_intervals(
                    tensor=self.delta_J,
                    start=torch.ones_like(dones_env_ids, device=self.device) * (t + 1),
                    end=traj_end_ids[dones_env_ids],
                    value=curr_J[dones_env_ids] - curr_J[self.get_nominal_idx_of_auxiliary_env(dones_env_ids)],
                    row_indices=dones_env_ids,
                )
                # Reset the trajectory end ids and curr_J
                curr_J[dones_env_ids] = self.next_values[
                    dones_env_ids, t
                ].clone()  # NOTE: the next value is zero if done
                traj_end_ids[dones_env_ids] = t + 1

            curr_J = self.gamma * curr_J + self.rewards[:, t]

        # Assign the remaining trajectory rewards
        assign_row_intervals(
            tensor=self.delta_J,
            start=torch.zeros(self.num_envs, device=self.device),
            end=traj_end_ids,
            value=curr_J
            - curr_J[self.get_nominal_idx_of_auxiliary_env(torch.arange(self.num_envs, device=self.device))],
        )

    def compute_action_ascent_direction(self):
        """
        This function computes the ascent direction for improving the nominal action by weighting all the perturbations.
        """
        # Reshape delta_J to group by base environments: [num_base_envs, num_action_perturbations + 1, horizon_length]
        delta_J_grouped = self.delta_J.view(self.num_base_envs, self.num_action_perturbations + 1, self.horizon_length)

        delta_J_var = delta_J_grouped.var(dim=1)
        self.rollout_var = delta_J_var.mean()
        self.positive_rollout_ratio = (self.delta_J > 0).sum() / self.delta_J.numel()

        # Normalize the delta_J for each nominal batch
        delta_J = self.delta_J
        if self.normalize_delta_J:
            delta_J = delta_J / (
                torch.sqrt(delta_J_var).repeat_interleave(self.num_action_perturbations + 1, dim=0) + 1e-6
            )

        # weighted_perturbations shape: [num_envs, horizon_length, num_actions]
        weighted_perturbations = delta_J.unsqueeze(-1) * self.eps_actions

        # reshape to group environments: [num_base_envs, num_action_perturbations + 1, horizon_length, num_actions]
        weighted_perturbations_grouped = weighted_perturbations.view(
            self.num_base_envs, self.num_action_perturbations + 1, self.horizon_length, self.num_actions
        )

        # Reshape normalized delta_J for top-k selection: [num_base_envs, num_action_perturbations + 1, horizon_length]
        delta_J_grouped_normalized = delta_J.view(
            self.num_base_envs, self.num_action_perturbations + 1, self.horizon_length
        )

        # Select top k perturbations based on delta_J values (from big to small)
        k = min(self.top_k_perturbations, self.num_action_perturbations + 1)

        if k < self.num_action_perturbations + 1:
            # Get top k indices: [num_base_envs, k, horizon_length]
            # We want to select top k for each base env and timestep based on normalized delta_J
            top_k_values, top_k_indices = torch.topk(delta_J_grouped_normalized, k=k, dim=1, largest=True)

            # Use gather to select top k weighted perturbations
            # top_k_indices: [num_base_envs, k, horizon_length]
            # Expand for gather: [num_base_envs, k, horizon_length, num_actions]
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.num_actions)

            # Gather top k weighted perturbations: [num_base_envs, k, horizon_length, num_actions]
            top_k_weighted_perturbations = torch.gather(
                weighted_perturbations_grouped, dim=1, index=top_k_indices_expanded
            )

            # Average across top k directions (dimension 1)
            action_ascent_direction = top_k_weighted_perturbations.mean(dim=1)
        else:
            # compute mean across each group (dimension 1) - use all directions
            action_ascent_direction = weighted_perturbations_grouped.mean(dim=1)

        return action_ascent_direction

    def compute_target_values(self):
        """
        This function computes the target values using TD(lambda) method.
        """
        # TD-style return estimated
        Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Monte Carlo return estimated
        Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        dones = self.dones.clone().to(torch.float32)
        for i in reversed(range(self.horizon_length)):
            lam = lam * self.lam * (1.0 - dones[:, i]) + dones[:, i]
            Ai = (1.0 - dones[:, i]) * (
                self.lam * self.gamma * Ai
                + self.gamma * self.next_values[:, i]
                + (1.0 - lam) / (1.0 - self.lam) * self.rewards[:, i]
            )
            Bi = self.gamma * (self.next_values[:, i] * dones[:, i] + Bi * (1.0 - dones[:, i])) + self.rewards[:, i]
            self.target_values[:, i] = (1.0 - self.lam) * Ai + lam * Bi

    def train_actor(self):
        # Compute the incremental trajectory rewards
        self.compute_delta_J()
        # Compute the action ascent direction for each nominal environment
        action_ascent_direction = self.compute_action_ascent_direction()

        target_actions = self.actions[self.nominal_env_ids] + action_ascent_direction
        # TODO: due to mant dictionary operations, and slicing; the pred_actions is slightly different from self.actions (approx 1e-6)
        pred_actions = self.actor(select_entries(self.obs_buf, self.nominal_env_ids))

        # Update the actor
        self.actor_optimizer.zero_grad()
        # TODO: currently we use all trajectory as single batch, and update the actor once.
        # Would be possible to use mini-batch training and run multiple updates, like PPO?
        actor_loss = F.mse_loss(pred_actions, target_actions)
        actor_loss.backward()
        self.actor_grad_norm = compute_grad_norm(self.actor.parameters())
        if self.truncated_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()
        return actor_loss.item()

    def train_critic(self):
        # NOTE: currently we use both nominal and auxiliary rollout for training the critic
        self.compute_target_values()
        # Flatten first two dimensions (num_envs, horizon_length) for all observation keys
        obs = flatten_dict(self.obs_buf, start_dim=0, end_dim=1)
        target_values = self.target_values.view(-1, 1)
        # Get dataset size from first observation key
        dataset_size = list(obs.values())[0].shape[0]

        for i in range(self.critic_iterations):
            perm = torch.randperm(dataset_size, device=self.device)
            obs_shuffled = {key: value[perm] for key, value in obs.items()}
            target_values_shuffled = target_values[perm]

            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                # Select batch for each observation key
                obs_batch = {key: value[start_idx:end_idx] for key, value in obs_shuffled.items()}
                target_values_batch = target_values_shuffled[start_idx:end_idx]

                self.critic_optimizer.zero_grad()
                pred_values = self.critic(obs_batch)
                critic_loss = F.mse_loss(pred_values, target_values_batch)
                critic_loss.backward()
                self.critic_grad_norm = compute_grad_norm(self.critic.parameters())
                if self.truncated_grads:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                self.critic_optimizer.step()

        # update target critic
        with torch.no_grad():
            alpha = self.target_critic_alpha
            for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters(), strict=False):
                param_targ.data.mul_(alpha)
                param_targ.data.add_((1.0 - alpha) * param.data)
        return critic_loss.item()

    def train_epoch(self):
        # Rollout trajectories for training
        self.time_report.start_timer("rollout")
        rollout_reward = self.rollout()
        self.time_report.end_timer("rollout")

        # Train the actor
        self.time_report.start_timer("train_actor")
        actor_loss = self.train_actor()
        self.time_report.end_timer("train_actor")

        # Train the critic
        self.time_report.start_timer("train_critic")
        critic_loss = self.train_critic()
        self.time_report.end_timer("train_critic")

        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()

        self.iter_count += 1

        return rollout_reward, actor_loss, critic_loss

    def get_nominal_idx_of_auxiliary_env(self, ids: torch.Tensor) -> torch.Tensor:
        """
        This function returns the nominal environment index for the given auxiliary environment ids.
        """
        return (ids // (self.num_action_perturbations + 1)) * (self.num_action_perturbations + 1)

    def get_auxiliary_idx_of_nominal_env(self, ids: torch.Tensor) -> torch.Tensor:
        """
        This function returns the auxiliary environment indices for the given nominal environment ids.

        Args:
            ids: Tensor of nominal environment IDs.

        Returns:
            Tensor containing all auxiliary environment IDs for the given nominal environments.
            Shape: [num_nominal_envs * num_action_perturbations]
        """
        if len(ids) == 0:
            return torch.tensor([], dtype=torch.int32, device=self.device)

        # Verify all ids are nominal environment ids
        is_nominal = torch.isin(ids, self.nominal_env_ids)
        if not torch.all(is_nominal):
            raise ValueError(
                f"All provided ids must be nominal environment ids. "
                f"Nominal env ids are: {self.nominal_env_ids.cpu().tolist()}"
            )

        # For each nominal environment, compute its auxiliary environment ids
        # If nominal env is at index i, auxiliary envs are at [i+1, i+2, ..., i+num_action_perturbations]
        auxiliary_ids_list = []
        for nominal_id in ids:
            start_idx = nominal_id + 1
            end_idx = nominal_id + self.num_action_perturbations + 1
            auxiliary_ids = torch.arange(start_idx, end_idx, dtype=torch.int32, device=self.device)
            auxiliary_ids_list.append(auxiliary_ids)

        # Concatenate all auxiliary environment ids into a single tensor
        return torch.cat(auxiliary_ids_list)

    def env_reset(self, dones: torch.Tensor):
        """
        This function handle the reset for both nominal and auxiliary environments.
        If nominal environment is reset, then all the auxiliary environments will be automatically reset to the same state to nominal

        Args:
            dones: Boolean tensor of shape [num_envs] indicating which environments are done
        """
        done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        if len(done_env_ids) > 0:
            # Find the nominal environment ids in the done env_ids
            is_nominal_done = torch.isin(done_env_ids, self.nominal_env_ids)
            done_nominal_env_ids = done_env_ids[is_nominal_done]

            # Reset the nominal environments
            if len(done_nominal_env_ids) > 0:
                self.env.reset(env_ids=done_nominal_env_ids)

            # Find the done auxiliary environment ids
            # There are two cases:
            # 1. The nominal environment is done, the auxiliary environment reset to the nominal environment state regardless of the auxiliary environment is done or not
            # 2. The nominal environment is not done, and the auxiliary environment is done
            done_auxiliary_env_ids = done_env_ids[~is_nominal_done]
            auxiliary_env_ids_for_done_nominal = (
                self.get_auxiliary_idx_of_nominal_env(done_nominal_env_ids)
                if len(done_nominal_env_ids) > 0
                else torch.tensor([], dtype=torch.int32, device=self.device)
            )
            auxiliary_env_ids_to_reset_to_nominal = (
                torch.unique(torch.cat([done_auxiliary_env_ids, auxiliary_env_ids_for_done_nominal]))
                if len(done_auxiliary_env_ids) > 0 or len(auxiliary_env_ids_for_done_nominal) > 0
                else torch.tensor([], dtype=torch.int32, device=self.device)
            )

            # Reset auxiliary environments that belong to done nominal envs (to match their nominal env state)
            if len(auxiliary_env_ids_to_reset_to_nominal) > 0:
                self.reset_auxiliary_envs(env_ids=auxiliary_env_ids_to_reset_to_nominal)

            # Update the done flags for the auxiliary environments that are reset to the nominal environment
            if len(auxiliary_env_ids_to_reset_to_nominal) > 0:
                dones[auxiliary_env_ids_to_reset_to_nominal] = True

        # Update the observation
        states = self.env.get_states(env_ids=torch.arange(self.num_envs, device=self.device, dtype=torch.int32))
        obs = self.env.compute_observations(states=states)

        return obs, dones

    def initialize_trajectory(self):
        """
        Intializing the trajectory for the rollout by resetting the auxiliary environments to the same state as the nominal environment.
        """
        self.reset_auxiliary_envs(env_ids=torch.arange(self.num_envs, device=self.device, dtype=torch.int32))
        states = self.env.get_states(env_ids=torch.arange(self.num_envs, device=self.device, dtype=torch.int32))
        obs = self.env.compute_observations(states=states)
        return obs

    def reset_auxiliary_envs(self, env_ids: torch.Tensor):
        """
        This function resets the auxiliary environments to the same state as the nominal environment.
        """
        if len(env_ids) == 0:
            return

        # Filter out nominal environments - ensure we only reset auxiliary environments
        is_nominal = torch.isin(env_ids, self.nominal_env_ids)
        auxiliary_env_ids = env_ids[~is_nominal]

        if len(auxiliary_env_ids) == 0:
            return

        nominal_env_ids_for_aux = self.get_nominal_idx_of_auxiliary_env(auxiliary_env_ids)
        nominal_states = self.env.get_states(env_ids=nominal_env_ids_for_aux)
        self.env.set_states(states=nominal_states, env_ids=auxiliary_env_ids)

    @torch.no_grad()
    def evaluate_policy(self, maximum_trajectory_length=None):
        """
        TODO currently this function is only for play mode to evaluate the trained policy.
        """
        episode_length = torch.zeros(self.num_envs, device=self.device)
        episode_length_meter = AverageMeter(1, 100).to(self.device)
        episode_reward = torch.zeros(self.num_envs, device=self.device)
        episode_reward_meter = AverageMeter(1, 100).to(self.device)
        if maximum_trajectory_length is None:
            maximum_trajectory_length = self.env.episode_length

        obs, _ = self.env.reset()
        for t in range(maximum_trajectory_length):
            # process the observations with the running statistics
            obs = self.process_observations(obs, self.obs_rms)
            actions = self.actor(obs)
            obs, rewards, terminated, truncated, info = self.env.step(torch.tanh(actions), auto_reset=True)
            dones = terminated | truncated
            done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
            episode_length += 1
            episode_reward += rewards
            if len(done_env_ids) > 0:
                episode_length_meter.update(episode_length[done_env_ids])
                episode_reward_meter.update(episode_reward[done_env_ids])
                episode_length[done_env_ids] = 0.0
                episode_reward[done_env_ids] = 0.0

        print_info(
            f"Episode length: {episode_length_meter.get_mean().item()}, Episode reward: {episode_reward_meter.get_mean().item()}"
        )

    def train(self):
        self.time_report.add_timer("rollout")
        self.time_report.add_timer("train_actor")
        self.time_report.add_timer("train_critic")

        self.save(filename="initial_policy")

        self.env.reset()
        self.episode_length = torch.zeros(self.num_base_envs, device=self.device)
        self.episode_reward = torch.zeros(self.num_base_envs, device=self.device)
        self.step_count = 0
        self.iter_count = 0
        best_policy_reward = -float("inf")
        start_time = time.time()

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()
            rollout_reward, actor_loss, critic_loss = self.train_epoch()
            time_end_epoch = time.time()
            time_elapse = time.time() - start_time

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
                critic_loss=critic_loss,
                rollout_reward=rollout_reward,
                rollout_var=self.rollout_var,
                positive_rollout_ratio=self.positive_rollout_ratio,
                actor_grad_norm=self.actor_grad_norm,
                critic_grad_norm=self.critic_grad_norm,
                iter=self.iter_count,
                step=self.step_count,
                time_elapse=time_elapse,
                policy_reward=policy_reward if policy_reward != float("inf") else None,
                episode_lengths=episode_lengths if episode_lengths > 0 else None,
                best_policy_reward=current_best_policy_reward,
            )

            print(
                "iter {}: ep reward {:.2f}, ep len {:.1f}, rollout reward {:.2f}, rollout reward std {:.2f}, rollout reward positive ratio {:.2f}, fps total {:.3g}, actor grad norm {:.2f}, critic grad norm {:.2f},".format(
                    self.iter_count,
                    policy_reward,
                    episode_lengths,
                    rollout_reward,
                    self.rollout_var.sqrt(),
                    self.positive_rollout_ratio,
                    self.step_count * self.num_envs / (time_end_epoch - time_start_epoch),
                    self.actor_grad_norm,
                    self.critic_grad_norm,
                )
            )

            if self.iter_count % self.save_frequency == 0 or self.iter_count == self.max_epochs - 1:
                self.save(filename="iter_{}_reward_{:.2f}".format(self.iter_count, policy_reward))

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
        self.obs_rms = checkpoint[3].to(self.device) if checkpoint[3] is not None else checkpoint[3]
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

    @torch.no_grad()
    def process_observations(self, obs, obs_rms=None):
        """
        This function processes the observations with the running statistics.
        """
        if obs_rms is None:
            obs_rms = self.obs_rms

        for key in obs_rms.keys():
            obs[key] = obs_rms[key].normalize(obs[key])
        return obs

    @torch.no_grad()
    def update_running_statistics(self, obs):
        """
        This function updates the running statistics.
        """
        for key in self.obs_rms.keys():
            self.obs_rms[key].update(obs[key])

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

        # TODO: currently target critic and critic are separate instead of in the same class
        # Also do I really need it?
        self.target_critic = copy.deepcopy(self.critic)


def make_runner(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        output_dir = hydra_cfg.run.dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)

    return AFRLRunner(config)
