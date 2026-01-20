"""Base environment class for wrapping other environments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


class BaseEnv(ABC):
    """Abstract base class for wrapping other environments.

    This class provides a common interface for wrapping different types of physical simulators (e.g., rewarped, mujoco-warp, etc.) with unified and augmented function
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the base environment.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    @abstractmethod
    def reset(self, env_ids: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    @abstractmethod
    def step(
        self, actions: torch.Tensor, auto_reset: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment.
        Args:
            actions: The actions to take.
            auto_reset: Whether to automatically reset the environment if the environment is terminated.

        Returns:
            A tuple containing the observations, rewards, dones, env_ids, and info.
            - observations: The observations of the environment.
            - rewards: The rewards of the environment.
            - terminated: The early terminated flags.
            - truncated: The time-out flags.
            - info: Additional information.
        """

    @abstractmethod
    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Get the states of the environment.

        Returns:
            The states of the environment in a dictionary with following keys:
            - robot_states: A dictionary containing the states of the robot.
            - progress_buf: The progress buffer of the environment.
            - env_ids: The indices of the environments.
        """

    @abstractmethod
    def set_states(self, states: Dict[str, Any], env_ids: Optional[Sequence[int]] = None) -> None:
        """Set the states of the environment.

        Args:
            states: The states to set in a dictionary with following keys:
            - robot_states: A dictionary containing the states to set.
            - progress_buf: The progress buffer to set.
            - env_ids: The indices of the environments.
        """

    @abstractmethod
    def compute_observations(self, states: Dict[str, Any]) -> torch.Tensor:
        """Compute the observations of the environment.

        Args:
            states: The states of the environment in a dictionary with following keys:
            - robot_states: A dictionary containing the states of the robot.
            - progress_buf: The progress buffer of the environment.

        Returns:
            The observations of the environment.
        """

    @abstractmethod
    def initialize_trajectory(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        This function is used for first-order RL.
        It start a new trajectory from the current states, but cuts off the computation graph to the previous states.
        Returns:
            A tuple containing the observations and the info.
            - observations: The observations of the environment.
            - info: Additional information.
        """

    """
    Properties.
    """

    @abstractmethod
    def save_video(self) -> None:
        """For environments like rewarped, we need actively save the recorded frames.

        Returns:
            None.
        """

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        """Get whether the environment requires gradients.

        Returns:
            Whether the environment requires gradients.
        """

    @property
    @abstractmethod
    def num_observations(self) -> int:
        """Get the number of observations.

        Returns:
            The number of observations.
        """

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Get the number of actions.

        Returns:
            The number of actions.
        """

    @property
    @abstractmethod
    def device(self) -> Any:
        """Get the device.

        Returns:
            The device.
        """

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Get the number of environments.

        Returns:
            The number of environments.
        """

    @property
    @abstractmethod
    def episode_length(self) -> int:
        """Get the episode length.

        Returns:
            The maximum length of the episode.
        """

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Get the action space.

        Returns:
            The action space.
        """

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Get the observation space.

        Returns:
            The observation space.
        """

    @property
    @abstractmethod
    def nominal_env_ids(self) -> torch.Tensor:
        """Get the nominal environment indices.
        Used for AFRL agent.

        Returns:
            The nominal environment indices.
        """
