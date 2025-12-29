import copy

import torch
from omegaconf import DictConfig

import models
from utils.common_utils import make_envs
from utils.statistic_utils import AverageMeter, RunningMeanStd
from utils.tensor_utils import assign_row_intervals


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
        self.nominal_env_ids = torch.arange(self.num_base_envs, device=self.device) * (
            self.num_action_perturbations + 1
        )
        self.horizon_length = self.agent_config.horizon_length
        # action perturbation factor
        self.delta = self.agent_config.delta
        self.gamma = self.agent_config.gamma
        self.lam = self.agent_config.lam
        self.reward_scale = self.agent_config.reward_scale
        self.target_critic_alpha = self.agent_config.target_critic_alpha

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
        # reward buffer contains the reward after processing, e.g. normalization or scale.
        self.rewards = torch.zeros(self.num_envs, self.horizon_length, device=self.device)
        self.next_values = torch.zeros(self.num_envs, self.horizon_length, device=self.device)
        self.target_values = torch.zeros(self.num_envs, self.horizon_length, device=self.device)
        self.dones = torch.zeros(self.num_envs, self.horizon_length, device=self.device, dtype=torch.bool)
        self.delta_J = torch.zeros(self.num_envs, self.horizon_length, device=self.device)

    def make_envs(self):
        self.env = make_envs(self.config)
        self.num_observations = self.env.num_observations
        self.num_actions = self.env.num_actions

    def rollout(self):
        """
        Rollout the trajectories for training and play the training dataset.
        """
        pass

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
        ascent_direction = weighted_perturbations_grouped.mean(dim=1)

        return ascent_direction

    def train_actor(self):
        pass

    def train_critic(self):
        pass

    def train_epoch(self):
        pass

    def compute_target_values(self):
        """
        This function computes the target values using TD(lambda) method.
        """
        # TD-style return estimated
        Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Monte Carlo return estimated
        Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        for i in reversed(range(self.horizon_length)):
            lam = lam * self.lam * (1.0 - self.dones[i]) + self.dones[i]
            Ai = (1.0 - self.dones[i]) * (
                self.lam * self.gamma * Ai
                + self.gamma * self.next_values[i]
                + (1.0 - lam) / (1.0 - self.lam) * self.rewards[i]
            )
            Bi = self.gamma * (self.next_values[i] * self.dones[i] + Bi * (1.0 - self.dones[i])) + self.rewards[i]
            self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi

    def get_nominal_idx_of_auxiliary_env(self, ids: torch.Tensor) -> torch.Tensor:
        """
        This function returns the nominal environment index for the given auxiliary environment ids.
        """
        return ids // (self.num_action_perturbations + 1)

    def train(self):
        pass

    def run(self, args):
        pass

    def env_reset(self):
        pass


def make_runner(config: DictConfig):
    return AFRLRunner(config)
