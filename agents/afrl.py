import torch
from omegaconf import DictConfig

from utils.tensor_utils import assign_row_intervals


class AFRLRunner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.num_base_envs = config.num_base_envs
        self.num_action_perturbations = config.num_action_perturbations
        # NOTE: for training, num_envs = num_base_envs * (num_action_perturbations + 1)
        # for evaluation only, however, num_envs may be different from num_base_envs * (num_action_perturbations + 1)
        self.num_envs = config.num_envs
        self.horizon_length = config.horizon_length
        self.gamma = config.gamma
        self.device = config.device

        # Buffer
        self.next_values = torch.zeros(self.num_envs, self.horizon_length - 1, device=self.device)
        self.dones = torch.zeros(self.num_envs, self.horizon_length - 1, device=self.device, dtype=torch.bool)
        # original rewards from the environment
        self.raw_rewards = torch.zeros(self.num_envs, self.horizon_length - 1, device=self.device)
        # rewards after processing
        self.rewards = torch.zeros(self.num_envs, self.horizon_length - 1, device=self.device)

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
        self.delta_J = torch.zeros(self.num_envs, self.horizon_length, device=self.device)

        # NOTE: next value is zero if done
        curr_J = self.next_values[:, -1].clone()

        # The end of the current trajectory idx
        traj_end_ids = torch.ones(self.num_envs, dtype=torch.int, device=self.device) * (self.horizon_length + 1)

        for t in range(self.horizon_length - 1, -1, -1):
            dones_env_ids = self.dones[:, t].nonzero(as_tuple=False).squeeze(-1)

            if len(dones_env_ids) > 0 and t < self.horizon_length - 1:
                assign_row_intervals(
                    tensor=self.delta_J,
                    start=torch.ones_like(dones_env_ids, device=self.device) * (t + 1),
                    end=traj_end_ids[dones_env_ids],
                    value=curr_J[dones_env_ids] - curr_J[self.get_nominal_env_idx(dones_env_ids)],
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
            value=curr_J - curr_J[self.get_nominal_env_idx(torch.arange(self.num_envs, device=self.device))],
        )

    def compute_action_direction(self):
        """
        This function computes the direction for improving the nominal action by weighting all the perturbations.
        """
        pass

    def get_nominal_env_idx(self, ids: torch.Tensor) -> torch.Tensor:
        """
        This function returns the nominal environment index for the given environment ids.
        """
        return ids // (self.num_action_perturbations + 1)

    def train(self):
        pass

    def run(self, args):
        pass


def make_runner(config: DictConfig):
    return AFRLRunner(config)
