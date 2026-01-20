import importlib
import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import genesis as gs
import torch
from omegaconf import DictConfig, OmegaConf

from envs.base_env import BaseEnv
from utils.common_utils import snakecase_to_pascalcase


class GenesisEnv(BaseEnv):
    """Environment wrapper for the Genesis simulator."""

    _num_observations: int
    _num_actions: int
    _action_space: Any
    _observation_space: Any
    _nominal_env_ids: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        episode_length: int,
        early_termination: bool = False,
        seed: int = 0,
        randomize_init: bool = True,
        nominal_env_ids: Optional[Sequence[int]] = None,
        device: torch.device | None = None,
        sensors_args: Dict[str, Any] | None = None,
        sim_options: gs.options.SimOptions | None = None,
        viewer_options: gs.options.ViewerOptions | None = None,
        vis_options: gs.options.VisOptions | None = None,
        show_viewer: bool = False,
        show_FPS: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        else:
            device = torch.device(device)
        self._device = device

        self._num_envs = num_envs
        self._episode_length = episode_length
        self._early_termination = early_termination
        self._randomize_init = randomize_init
        self._seed = seed
        self._sensors_args = sensors_args
        self._nominal_env_ids = (
            torch.arange(num_envs, device=device)
            if nominal_env_ids is None
            else torch.tensor(nominal_env_ids, device=device)
        )

        if not gs._initialized:
            if self._device == torch.device("cpu"):
                gs.init(performance_mode=True, backend=gs.cpu, seed=self._seed)
            elif self._device == torch.device("cuda"):
                gs.init(performance_mode=True, backend=gs.cuda, seed=self._seed)
            else:
                raise ValueError(f"Invalid device: {self._device}")

        self._renderer = gs.renderers.Rasterizer()
        self._scene = gs.Scene(
            sim_options=sim_options,
            show_viewer=show_viewer,
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_FPS=show_FPS,
            renderer=self._renderer,
        )

        # Initialize the scene
        self.init_scene()

        # build the scene
        self.build_scene()

        # Buffers
        self._progress_buf = torch.zeros(self._num_envs, device=self._device)
        self._obs_buf = torch.zeros(self._num_envs, self._num_observations, device=self._device)
        self._truncated_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._terminated_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._reset_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._extras = {}

    @abstractmethod
    def build_scene(self) -> None:
        """Build the scene."""

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self._reset_idx(env_ids)

        self._progress_buf[env_ids] = 0

        states = self.get_states()
        self._obs_buf = self.compute_observations(states)

        return self._obs_buf, {}

    def step(
        self, actions: torch.Tensor, auto_reset: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Process actions specified by each environment.
        self._set_actions(actions)

        # Do the physics step.
        self._scene.step()
        self._progress_buf += 1

        states = self.get_states()
        self._obs_buf = self.compute_observations(states)
        self._reward_buf = self.compute_reward(states, actions)
        self._terminated_buf = self.compute_termination(states)

        # set the terminated_buf and reward involved nan ids to true and zero respectively
        nan_ids = self._find_nan_ids(states)
        self._terminated_buf[nan_ids] = True
        self._reward_buf[nan_ids] = 0.0

        self._truncated_buf = self._progress_buf >= self._episode_length
        self._reset_buf = self._terminated_buf | self._truncated_buf

        reset_env_ids = self._reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if auto_reset and len(reset_env_ids) > 0:
            obs_buf_before_reset = self._obs_buf.clone()
            self._extras["obs_before_reset"] = obs_buf_before_reset
            self._obs_buf, _ = self.reset(reset_env_ids)

        return self._obs_buf, self._reward_buf, self._terminated_buf, self._truncated_buf, {}

    def initialize_trajectory(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # TODO
        pass

    def save_video(self) -> None:
        # TODO
        pass

    def _find_nan_ids(self, states: Dict[str, Any]) -> torch.Tensor:
        """Find the ids of the environments that have NaN values in the states.

        Args:
            states: The states dictionary containing "robot_states" and "progress_buf".
                - robot_states: A dictionary containing tensors with shape (num_envs, ...).
                - progress_buf: A tensor with shape (num_envs,).

        Returns:
            A tensor containing the environment indices that have at least one NaN value.
        """
        robot_states = states.get("robot_states", {})

        def check_nan_recursive(obj: Any) -> torch.Tensor:
            """Recursively check for NaN values in nested dictionaries and tensors."""
            if isinstance(obj, torch.Tensor):
                # Check for NaN values in the tensor
                if obj.numel() == 0:
                    # Empty tensor, skip
                    return None
                elif obj.dim() == 0:
                    # Scalar tensor - if NaN, mark all environments as having NaN
                    if torch.isnan(obj):
                        return torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
                    return None
                elif obj.shape[0] == self._num_envs:
                    # Tensor with correct first dimension (num_envs, ...)
                    if obj.dim() == 1:
                        # 1D tensor: (num_envs,)
                        return torch.isnan(obj)
                    else:
                        # Multi-dimensional tensor: (num_envs, ...)
                        # Check if any element along non-first dimensions is NaN
                        return torch.isnan(obj).any(dim=tuple(range(1, obj.dim())))
                else:
                    # Tensor with unexpected shape - if it contains NaN, mark all as having NaN
                    if torch.isnan(obj).any():
                        return torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
                    return None
            elif isinstance(obj, dict):
                # Recursively check all values in the dictionary
                nan_flags = []
                for value in obj.values():
                    nan_flag = check_nan_recursive(value)
                    if nan_flag is not None:
                        nan_flags.append(nan_flag)
                if nan_flags:
                    # Combine all NaN flags with OR operation
                    return torch.stack(nan_flags, dim=0).any(dim=0)
            return None

        # Check robot_states for NaN values
        nan_mask = check_nan_recursive(robot_states)

        if nan_mask is None:
            # No NaN values found
            return torch.tensor([], dtype=torch.int32, device=self._device)

        # Return the environment indices that have NaN values
        nan_ids = nan_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        return nan_ids

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset the indices of the environment.
        This function should be called before the reset.
        """

    @abstractmethod
    def _set_actions(self, actions: torch.Tensor) -> None:
        """Set the actions of the environment.
        This function progress actions specified by each environment.
        It should be called before the physics step.
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
    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Compute the reward of the environment.

        Args:
            states: The states of the environment in a dictionary with following keys:
            - robot_states: A dictionary containing the states of the robot.
            - progress_buf: The progress buffer of the environment.

        Returns:
            The reward of the environment.
        """

    @abstractmethod
    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        """Compute the termination of the environment.

        Args:
            states: The states of the environment in a dictionary with following keys:
            - robot_states: A dictionary containing the states of the robot.
            - progress_buf: The progress buffer of the environment.

        Returns:
            The termination of the environment.
        """

    @property
    def renderer(self):
        # TODO
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def requires_grad(self) -> bool:
        # TODO
        pass

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_observations(self) -> int:
        return self._num_observations

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def episode_length(self) -> int:
        return self._episode_length

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def nominal_env_ids(self) -> torch.Tensor:
        return self._nominal_env_ids


def make_envs(config: DictConfig) -> GenesisEnv:
    env_kwargs = OmegaConf.to_container(config.task.config, resolve=True)
    env_name = config.task.name
    num_envs = env_kwargs.pop("num_envs")

    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))

    sim_kwargs = env_kwargs.pop("sim_options", None)
    sim_options = gs.options.SimOptions(**sim_kwargs) if sim_kwargs is not None else None
    viewer_kwargs = env_kwargs.pop("viewer_options", None)
    viewer_options = gs.options.ViewerOptions(**viewer_kwargs) if viewer_kwargs is not None else None
    vis_kwargs = env_kwargs.pop("vis_options", None)
    vis_options = gs.options.VisOptions(**vis_kwargs) if vis_kwargs is not None else None

    # Configure logging to prevent duplicate Genesis logs
    # Genesis logs directly, so we disable Python logging for genesis logger
    genesis_logger = logging.getLogger("genesis")
    genesis_logger.propagate = False
    genesis_logger.handlers = []  # Remove handlers to prevent duplicate logs

    env = env_fn(
        num_envs=num_envs,
        device=config.device,
        seed=config.seed,
        sim_options=sim_options,
        viewer_options=viewer_options,
        vis_options=vis_options,
        **env_kwargs,
    )

    return env
