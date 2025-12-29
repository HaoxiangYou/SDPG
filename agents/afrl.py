import copy
import os
import time

import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import models
from utils.common_utils import TimeReport, make_envs, print_info
from utils.statistic_utils import AverageMeter, RunningMeanStd
from utils.tensor_utils import assign_row_intervals, compute_grad_norm


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
        # TODO: currently assume both actor and critic are deterministic MLPs
        self.actor_config = self.agent_config.network.actor
        self.critic_config = self.agent_config.network.critic
        actor_name = self.agent_config.network.actor.name
        critic_name = self.agent_config.network.critic.name
        actor_fn = getattr(models.actor, actor_name)
        self.actor = actor_fn(self.num_observations, self.num_actions, self.actor_config, device=self.device)
        critic_fn = getattr(models.critic, critic_name)
        self.critic = critic_fn(self.num_observations, self.critic_config, device=self.device)
        self.target_critic = copy.deepcopy(self.critic)

        # initialize the optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.agent_config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.agent_config.critic_lr)

        # Running mean and std for the observations
        if self.agent_config.obs_rms:
            self.obs_rms = RunningMeanStd(shape=(self.num_observations,), device=self.device)
        else:
            self.obs_rms = None
        if self.agent_config.ret_rms:
            self.ret_rms = RunningMeanStd(shape=(1,), device=self.device)
        else:
            self.ret_rms = None

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

    def make_envs(self):
        self.env = make_envs(self.config)
        self.num_observations = self.env.num_observations
        self.num_actions = self.env.num_actions

    def rollout(self):
        """
        Rollout the trajectories for training and play the training dataset.
        """
        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

            # initialize trajectory by resetting the auxiliary environments to the same state as the nominal environment
            obs = self.initialize_trajectory()
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)

            # return of the short rollout (for logging purposes)
            rollout_reward = 0.0

            for i in range(self.horizon_length):
                self.obs_buf[:, i] = obs.clone()

                # Compute the nominal actions
                nominal_actions = self.actor(obs[self.nominal_env_ids])
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
                non_truncated_env_ids = (~truncated).nonzero(as_tuple=False).squeeze(-1)
                next_values[non_truncated_env_ids] = self.target_critic(
                    obs_rms.normalize(obs[non_truncated_env_ids])
                ).squeeze(-1)
                if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                    print("next value error")
                    raise ValueError("next value error")
                self.next_values[:, i] = next_values.clone()

                # Handle the done and reset
                dones = terminated | truncated
                obs, dones = self.env_reset(dones)

                # Normalize the observation
                if self.obs_rms is not None:
                    self.obs_rms.update(obs)
                    obs = obs_rms.normalize(obs)

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
        traj_end_ids = torch.ones(self.num_envs, dtype=torch.int, device=self.device) * (self.horizon_length + 1)

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
                curr_J[dones_env_ids] = self.next_values[dones_env_ids, t + 1]  # NOTE: the next value is zero if done
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
        weighted_perturbations = self.delta_J.unsqueeze(-1) * self.eps_actions / self.delta
        # weighted_perturbations shape: [num_envs, horizon_length, num_actions]

        # reshape to group environments: [num_base_envs, num_action_perturbations + 1, horizon_length, num_actions]
        weighted_perturbations_grouped = weighted_perturbations.view(
            self.num_base_envs, self.num_action_perturbations + 1, self.horizon_length, self.num_actions
        )

        # compute mean across each group (dimension 1)
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
        pred_actions = self.actor(self.obs_buf[self.nominal_env_ids])

        # Update the actor
        self.actor_optimizer.zero_grad()
        actor_loss = F.mse_loss(pred_actions, target_actions)
        actor_loss.backward()
        self.grad_norm_before_clip = compute_grad_norm(self.actor.parameters())
        if self.truncated_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()
        return actor_loss.item()

    def train_critic(self):
        self.compute_target_values()
        obs = self.obs_buf.view(-1, self.num_observations)
        target_values = self.target_values.view(-1, 1)
        dataset_size = obs.shape[0]

        for i in range(self.critic_iterations):
            perm = torch.randperm(dataset_size, device=self.device)
            obs_shuffled = obs[perm]
            target_values_shuffled = target_values[perm]

            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                obs_batch = obs_shuffled[start_idx:end_idx]
                target_values_batch = target_values_shuffled[start_idx:end_idx]

                self.critic_optimizer.zero_grad()
                pred_values = self.critic(obs_batch)
                critic_loss = F.mse_loss(pred_values, target_values_batch)
                critic_loss.backward()
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

        self.iter_count += 1

        return rollout_reward, actor_loss, critic_loss

    def get_nominal_idx_of_auxiliary_env(self, ids: torch.Tensor) -> torch.Tensor:
        """
        This function returns the nominal environment index for the given auxiliary environment ids.
        """
        return ids // (self.num_action_perturbations + 1)

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
        done_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            # Find the nominal environment ids in the done env_ids
            is_nominal_done = torch.isin(done_env_ids, self.nominal_env_ids)
            done_nominal_env_ids = done_env_ids[is_nominal_done]

            # Find the corresponding auxiliary environment ids for done nominal environments
            # These auxiliary envs will be reset to match their nominal env's state
            auxiliary_env_ids_to_reset_to_nominal = torch.tensor([], dtype=torch.int32, device=self.device)
            if len(done_nominal_env_ids) > 0:
                auxiliary_env_ids_to_reset_to_nominal = self.get_auxiliary_idx_of_nominal_env(done_nominal_env_ids)

            # Find the done auxiliary environment ids that do not belong to done nominal environments
            # These auxiliary envs are done but their nominal env is not done, so they need normal reset
            done_auxiliary_env_ids = done_env_ids[~is_nominal_done]
            # Filter out auxiliary envs that belong to done nominal envs (they're already handled above)
            if len(done_auxiliary_env_ids) > 0:
                nominal_ids_of_done_aux = self.get_nominal_idx_of_auxiliary_env(done_auxiliary_env_ids)
                # Keep only auxiliary envs whose nominal env is NOT done
                nominal_not_done_mask = ~torch.isin(nominal_ids_of_done_aux, done_nominal_env_ids)
                done_auxiliary_env_ids_standalone = done_auxiliary_env_ids[nominal_not_done_mask]
            else:
                done_auxiliary_env_ids_standalone = torch.tensor([], dtype=torch.int32, device=self.device)

            # Reset the nominal environments
            if len(done_nominal_env_ids) > 0:
                self.env.reset(env_ids=done_nominal_env_ids)

            # Reset auxiliary environments that belong to done nominal envs (to match their nominal env state)
            if len(auxiliary_env_ids_to_reset_to_nominal) > 0:
                self.reset_auxiliary_envs(env_ids=auxiliary_env_ids_to_reset_to_nominal)

            # Reset standalone auxiliary environments (those done but their nominal env is not done)
            if len(done_auxiliary_env_ids_standalone) > 0:
                # NOTE: When an auxiliary environment terminates early but its nominal environment continues,
                # we reset the auxiliary env independently. This creates a progress buffer mismatch:
                # the nominal env continues with its current progress, while the auxiliary env starts from 0.
                # This is acceptable because once the nominal environment is truncated, all the auxiliary environments will be done anyway.
                self.env.reset(env_ids=done_auxiliary_env_ids_standalone)

            # Update the done flags for the auxiliary environments that are reset to the nominal environment# Update the done flags for the auxiliary environments that are reset to the nominal environment
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

            # Logging
            self.summary_writer.add_scalar("actor_loss/iter", actor_loss, self.iter_count)
            self.summary_writer.add_scalar("actor_loss/step", actor_loss, self.step_count)
            self.summary_writer.add_scalar("actor_loss/time", actor_loss, time_elapse)
            self.summary_writer.add_scalar("critic_loss/iter", critic_loss, self.iter_count)
            self.summary_writer.add_scalar("critic_loss/step", critic_loss, self.step_count)
            self.summary_writer.add_scalar("critic_loss/time", critic_loss, time_elapse)
            self.summary_writer.add_scalar("rollout_reward/iter", rollout_reward, self.iter_count)
            self.summary_writer.add_scalar("rollout_reward/step", rollout_reward, self.step_count)
            self.summary_writer.add_scalar("rollout_reward/time", rollout_reward, time_elapse)

            if self.episode_length_meter.current_size > 0:
                policy_reward = self.episode_reward_meter.get_mean().item()
                length = self.episode_length_meter.get_mean().item()
                if policy_reward > best_policy_reward:
                    best_policy_reward = policy_reward
                    print_info("Save best policy with reward: {:.2f}".format(best_policy_reward))
                    self.save(filename="best_policy")
                    self.summary_writer.add_scalar("best_policy/iter", best_policy_reward, self.iter_count)
                    self.summary_writer.add_scalar("best_policy/step", best_policy_reward, self.step_count)
                    self.summary_writer.add_scalar("best_policy/time", best_policy_reward, time_elapse)
                self.summary_writer.add_scalar("rewards/iter", policy_reward, self.iter_count)
                self.summary_writer.add_scalar("rewards/step", policy_reward, self.step_count)
                self.summary_writer.add_scalar("rewards/time", policy_reward, time_elapse)
                self.summary_writer.add_scalar("length/iter", length, self.iter_count)
                self.summary_writer.add_scalar("length/step", length, self.step_count)
                self.summary_writer.add_scalar("length/time", length, time_elapse)
            else:
                policy_reward = float("inf")
                length = 0

            print(
                "iter {}: ep reward {:.2f}, rollout reward {:.2f}, ep len {:.1f}, fps total {:.2f}, grad norm before clip {:.2f},".format(
                    self.iter_count,
                    policy_reward,
                    rollout_reward,
                    length,
                    self.step_count * self.num_envs / (time_end_epoch - time_start_epoch),
                    self.grad_norm_before_clip,
                )
            )

            if epoch % self.save_frequency == 0:
                self.save(filename="iter_{}_reward_{:.2f}".format(self.iter_count, policy_reward))

    def play(self):
        pass

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


def make_runner(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None:
        output_dir = hydra_cfg.run.dir
        OmegaConf.set_struct(config, False)
        config.log_dir = output_dir
        OmegaConf.set_struct(config, True)

    return AFRLRunner(config)
