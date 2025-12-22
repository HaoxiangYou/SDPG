import os
from typing import Any, Dict

import genesis as gs
import torch

from envs.genesis_env.genesis_env import GenesisEnv


class Hopper(GenesisEnv):
    """Hopper environment."""

    _num_observations = 11
    _num_actions = 3

    def __init__(
        self,
        num_envs: int,
        render: bool = False,
        seed: int = 0,
        randomize_init: bool = True,
        device: torch.device | None = None,
        show_viewer: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda")

        sim_options = gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
        )
        episode_length = 1000
        early_termination = True

        super().__init__(
            num_envs=num_envs,
            episode_length=episode_length,
            early_termination=early_termination,
            render=render,
            seed=seed,
            randomize_init=randomize_init,
            device=device,
            show_viewer=show_viewer,
            sim_options=sim_options,
        )

    def init_scene(self) -> None:
        """Initialize the scene."""

        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/hopper.xml"))
        )
        self._plane = self._scene.add_entity(gs.morphs.Plane())

        self._root_joint_names = ["rootx", "rootz", "rooty"]
        self._motor_joint_names = ["thigh_joint", "leg_joint", "foot_joint"]
        self._root_dof_idx = [self._robot.get_joint(name).dof_start for name in self._root_joint_names]
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]

        self._motor_strength = torch.tensor([200.0, 200.0, 200.0], device=self._device)

        self._default_root_dof_pos = torch.zeros(self._num_envs, len(self._root_dof_idx), device=self._device)
        self._default_motor_dof_pos = torch.zeros(self._num_envs, len(self._motors_dof_idx), device=self._device)

    def init_camera(self) -> None:
        """Initialize the camera."""
        pass

    def compute_observations(self, states: Dict[str, Any]) -> torch.Tensor:
        # TODO
        pass

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # TODO
        pass

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        # TODO
        pass

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        root_dof_pos = self._default_root_dof_pos[env_ids]
        motor_dof_pos = self._default_motor_dof_pos[env_ids]

        if self._randomize_init:
            root_dof_pos = root_dof_pos + (torch.rand_like(root_dof_pos) - 0.5) * 0.1
            motor_dof_pos = motor_dof_pos + (torch.rand_like(motor_dof_pos) - 0.5) * 0.1

        self._robot.set_dofs_position(
            position=root_dof_pos,
            dofs_idx_local=self._root_dof_idx,
            zero_velocity=True,
        )
        self._robot.set_dofs_position(
            position=motor_dof_pos,
            dofs_idx_local=self._motors_dof_idx,
            zero_velocity=True,
        )

    def _set_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(self._num_envs, self._num_actions)
        actions = actions.clamp(min=-1.0, max=1.0) * self._motor_strength
        self._robot.control_dofs_force(actions, dofs_idx_local=self._motors_dof_idx)

    def get_states(self) -> Dict[str, Any]:
        # TODO
        pass

    def set_states(self, states: Dict[str, Any]) -> None:
        # TODO
        pass
