import importlib
from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from envs.base_env import BaseEnv
from envs.mujoco_env.mjx_sim import MjxSim
from utils.common_utils import snakecase_to_pascalcase


class MujocoEnv(BaseEnv):
    """Environment wrapper for the MuJoCo (MJX) simulator.

    The simulator state exposed to tasks is (qpos, qvel) held as torch
    tensors; each control step is a pure function of them (see MjxSim). When
    built with sim_options.requires_grad, step()/reset() keep the torch
    computation graph intact (in-graph resets via clone + index-assign) so
    first-order agents (D.Va/SHAC) can backprop through rollouts;
    initialize_trajectory() cuts the graph between short horizons.

    Task state convention: DOF positions are exposed relative to the model
    reference pose (qpos - qpos0), matching the genesis backend where the
    spawn pose reads as zero.
    """

    _num_actions: int
    _action_space: Any
    _observation_space: Any
    _nominal_env_ids: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        episode_length: int,
        xml_path: str,
        early_termination: bool = False,
        seed: int = 0,
        randomize_init: bool = True,
        nominal_env_ids: Optional[Sequence[int]] = None,
        device: torch.device | None = None,
        sim_options: Dict[str, Any] | None = None,
        show_viewer: bool = False,
        show_FPS: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
            if device.type == "cuda" and not torch.cuda.is_available():
                print(f"Requested device '{device}' but CUDA is not available, falling back to CPU.")
                device = torch.device("cpu")
        self._device = device

        self._num_envs = num_envs
        self._episode_length = episode_length
        self._early_termination = early_termination
        self._randomize_init = randomize_init
        self._seed = seed
        self._show_viewer = show_viewer  # interactive viewing is handled by replay tooling
        self._show_FPS = show_FPS
        self._nominal_env_ids = (
            torch.arange(num_envs, device=device)
            if nominal_env_ids is None
            else torch.tensor(nominal_env_ids, device=device)
        )

        num_nominal_envs = self._nominal_env_ids.shape[0]
        if num_envs % num_nominal_envs != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be fully divisible by num_nominal_envs ({num_nominal_envs}). "
                f"Current remainder: {num_envs % num_nominal_envs}"
            )

        sim_options = dict(sim_options) if sim_options is not None else {}
        self._ctrl_dt = float(sim_options.get("dt", 1e-2))
        self._n_substeps = int(sim_options.get("substeps", 1))
        self._requires_grad = bool(sim_options.get("requires_grad", False))
        precision = sim_options.get("precision", None)
        if precision is None:
            # Analytic gradients through the simulator want double precision;
            # everything else runs single precision for speed.
            precision = "64" if self._requires_grad else "32"
        self._sim = MjxSim(
            xml_path=xml_path,
            sim_dt=self._ctrl_dt / self._n_substeps,
            n_substeps=self._n_substeps,
            requires_grad=self._requires_grad,
            precision=str(precision),
            solver_iterations=sim_options.get("solver_iterations", None),
            ls_iterations=sim_options.get("ls_iterations", None),
            device=device,
        )
        self._sim_dtype = self._sim.torch_dtype

        self._rng = torch.Generator(device=self._device)
        self._rng.manual_seed(seed)

        # Reference pose; task DOF states are exposed relative to it.
        self._qpos0 = torch.as_tensor(self._sim.mj_model.qpos0, dtype=self._sim_dtype, device=self._device)
        self._qpos = self._qpos0.expand(num_envs, -1).clone()
        self._qvel = torch.zeros(num_envs, self._sim.nv, dtype=self._sim_dtype, device=self._device)

        # Task-specific setup (dof groupings, reward constants, sensors, ...)
        self.init_task()

        # Buffers
        self._progress_buf = torch.zeros(self._num_envs, device=self._device)
        self._truncated_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._terminated_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._reset_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._infos = {}

    @abstractmethod
    def init_task(self) -> None:
        """Initialize task-specific attributes (dof indices, reward constants, ...)."""

    def reset(self, env_ids=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self._reset_idx(env_ids)

        self._progress_buf[env_ids] = 0

        states = self.get_states()
        observations = self.compute_observations(states)

        return observations, self._infos

    def step(
        self, actions: torch.Tensor, auto_reset: bool = True
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute a timestep. Mirrors the genesis backend pipeline:
        1. Process the actions specified by each environment.
        2. Do the actual physics step and update the progress buffer.
        3. Compute the reward and termination.
        4. Do post physics step operations.
        5. Reset the done environments if auto_reset is True (in-graph when
           requires_grad; the pre-reset observations are exposed via
           info["observations_before_reset"] for terminal-value bootstrapping).
        6. Compute the observations.
        """
        actions = actions.view(self._num_envs, self._num_actions).to(self._sim_dtype)
        actions = actions.clamp(min=-1.0, max=1.0)
        ctrl = self._action_to_ctrl(actions)

        self._qpos, self._qvel = self._sim.step(self._qpos, self._qvel, ctrl)
        self._progress_buf += 1

        states = self.get_states()
        self._reward_buf = self.compute_reward(states, actions).to(torch.float32)
        self._terminated_buf = self.compute_termination(states)
        # set the terminated_buf and reward involved nan ids to true and zero respectively
        nan_ids = self._find_nan_ids(states)
        self._terminated_buf[nan_ids] = True
        rewards = self._reward_buf
        if len(nan_ids) > 0:
            rewards = rewards.clone()
            rewards[nan_ids] = 0.0
            self._reward_buf = rewards
        self._truncated_buf = self._progress_buf >= self._episode_length
        self._reset_buf = self._terminated_buf | self._truncated_buf
        reset_env_ids = self._reset_buf.nonzero(as_tuple=False).squeeze(-1)

        self._post_physics_step()

        if self._requires_grad:
            self._infos["observations_before_reset"] = self.compute_observations(states)

        if auto_reset and len(reset_env_ids) > 0:
            _, _ = self.reset(reset_env_ids)
            states = self.get_states()

        observations = self.compute_observations(states)

        return observations, self._reward_buf, self._terminated_buf, self._truncated_buf, self._infos

    def initialize_trajectory(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Start a new trajectory from the current states, cutting the graph."""
        self._qpos = self._qpos.detach()
        self._qvel = self._qvel.detach()
        observations = self.compute_observations(self.get_states())
        return observations, self._infos

    def _set_dof_state(self, env_ids: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> None:
        """Write per-env DOF states (relative to the reference pose).

        Autograd-safe: clone + index-assign so envs that are not written keep
        their gradient path (the D.Va in-graph reset pattern).
        """
        qpos = self._qpos.clone()
        qvel = self._qvel.clone()
        qpos[env_ids] = self._qpos0 + dof_pos.to(self._sim_dtype)
        qvel[env_ids] = dof_vel.to(self._sim_dtype)
        self._qpos = qpos
        self._qvel = qvel

    def _action_to_ctrl(self, actions: torch.Tensor) -> torch.Tensor:
        """Map clamped [-1, 1] actions to actuator controls (default: identity;
        actuator gear in the MJCF applies the physical scaling)."""
        return actions

    def _find_nan_ids(self, states: Dict[str, Any]) -> torch.Tensor:
        """Find the ids of the environments that have NaN values in the states."""
        robot_states = states.get("robot_states")

        def check_nan_recursive(obj: Any) -> Optional[torch.Tensor]:
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 0:
                    return None
                elif obj.dim() == 0:
                    if torch.isnan(obj):
                        return torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
                    return None
                elif obj.shape[0] == self._num_envs:
                    if obj.dim() == 1:
                        return torch.isnan(obj)
                    else:
                        return torch.isnan(obj).any(dim=tuple(range(1, obj.dim())))
                else:
                    if torch.isnan(obj).any():
                        return torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
                    return None
            elif isinstance(obj, dict):
                nan_flags = []
                for value in obj.values():
                    nan_flag = check_nan_recursive(value)
                    if nan_flag is not None:
                        nan_flags.append(nan_flag)
                if nan_flags:
                    return torch.stack(nan_flags, dim=0).any(dim=0)
            return None

        nan_mask = check_nan_recursive(robot_states)

        if nan_mask is None:
            return torch.tensor([], dtype=torch.int32, device=self._device)

        return nan_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset the given environments (write DOF states via _set_dof_state)."""

    @abstractmethod
    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Get the states of the environment (robot_states dict + progress_buf)."""

    @abstractmethod
    def set_states(self, states: Dict[str, Any], env_ids: Optional[Sequence[int]] = None) -> None:
        """Set the states of the environment (row i of states -> env_ids[i])."""

    @abstractmethod
    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the observations (pure function of states, float32 outputs)."""

    @abstractmethod
    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Compute the reward (differentiable w.r.t. states/actions when requires_grad)."""

    @abstractmethod
    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        """Compute the early-termination flags."""

    def _post_physics_step(self) -> None:
        """Do post physics step operations (commands, observation history, ...)."""

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def num_envs(self) -> int:
        return self._num_envs

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

    @property
    def num_nominal_envs(self) -> int:
        return self._nominal_env_ids.shape[0]

    @property
    def num_auxiliary_envs(self) -> int:
        return self.num_envs // self.num_nominal_envs - 1

    def close(self) -> None:
        """Release references; jax buffers are freed by garbage collection."""
        self._qpos = None
        self._qvel = None
        self._sim = None


def make_envs(config: DictConfig) -> MujocoEnv:
    env_kwargs = OmegaConf.to_container(config.task.config, resolve=True)
    env_name = config.task.name
    num_envs = env_kwargs.pop("num_envs")

    ENV = importlib.import_module(f"envs.mujoco_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))

    env = env_fn(
        num_envs=num_envs,
        device=config.device,
        seed=config.seed,
        **env_kwargs,
    )

    return env
