import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import pos_lookat_up_to_T
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class Hopper(GenesisEnv):
    """Hopper environment."""

    _num_observations = 11
    _num_actions = 3
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
    _observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,))

    def __init__(
        self,
        num_envs: int,
        vis_obs: bool = False,
        seed: int = 0,
        randomize_init: bool = True,
        device: torch.device | None = None,
        sensors_args: Dict[str, Any] | None = None,
        sim_options: gs.options.SimOptions | None = None,
        viewer_options: gs.options.ViewerOptions | None = None,
        vis_options: gs.options.VisOptions | None = None,
        show_viewer: bool = False,
        show_FPS: bool = False,
    ) -> None:
        episode_length = 1000
        early_termination = True
        self._vis_obs = vis_obs

        super().__init__(
            num_envs=num_envs,
            episode_length=episode_length,
            early_termination=early_termination,
            sensors_args=sensors_args,
            seed=seed,
            randomize_init=randomize_init,
            device=device,
            show_viewer=show_viewer,
            sim_options=sim_options,
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_FPS=show_FPS,
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

        self._termination_height = -0.45
        self._termination_height_tolerance = 0.15
        self._termination_angle = torch.pi / 6.0
        self._termination_angle_tolerance = 0.05
        self._height_reward_scale = 1.0
        self._angle_reward_scale = 1.0
        self._action_penalty = -1e-1

        # Initialize the sensors
        if self._vis_obs:
            # which envs_idx to acquire the rendered rgb image
            if self._sensors_args["envs_idx"] is not None:
                self._rendered_envs_idx = self._sensors_args["envs_idx"]
            else:
                self._rendered_envs_idx = torch.arange(self._num_envs, device=self._device)

            # TODO: genesis at commit id 7db43e4caef2b185bf691d29fc545d6480cd224d only supports offset_T
            offset_T = self._sensors_args["camera"].get("offset_T", None)
            lookat = self._sensors_args["camera"].get("lookat", None)
            if offset_T is not None:
                offset_T = torch.tensor(offset_T, device=self._device)
            else:
                if lookat is not None:
                    offset_T = pos_lookat_up_to_T(
                        np.array(self._sensors_args["camera"]["pos"]), np.array(lookat), np.array((0.0, 0.0, 1.0))
                    )
                else:
                    offset_T = np.eye(4)

            self._camera = self._scene.add_sensor(
                gs.sensors.BatchRendererCameraOptions(
                    res=self._sensors_args["camera"]["res"],
                    pos=self._sensors_args["camera"]["pos"],
                    offset_T=offset_T,
                    fov=self._sensors_args["camera"]["fov"],
                    entity_idx=self._robot.idx,
                    link_idx_local=self._robot.get_link("torso").idx_local,
                    lights=[self._sensors_args["camera"]["lights"]],
                )
            )

            self._num_image_stack = 3
            self._image_buf = torch.zeros(
                len(self._rendered_envs_idx),
                self._num_image_stack,
                self._sensors_args["camera"]["res"][0],
                self._sensors_args["camera"]["res"][1],
                3,
                device=self._device,
            )

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(0.0, 1.0), n_envs_per_row=self._num_envs)

    def compute_observations(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        observations = torch.cat(
            [
                robot_states["root_joints_pos"][:, 1:],
                robot_states["motor_joints_pos"],
                robot_states["root_joints_vel"],
                robot_states["motor_joints_vel"],
            ],
            dim=-1,
        )
        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Jie Xu's reward function
        height = states["robot_states"]["root_joints_pos"][:, 1]
        height_diff = height - (self._termination_height + self._termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, 3.0)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self._height_reward_scale * height_reward, height_reward)

        angle = states["robot_states"]["root_joints_pos"][:, 2]
        # Wrap angle to [-pi, pi]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))

        angle_reward = self._angle_reward_scale * (-(angle**2) / (self._termination_angle**2) + 1.0)

        forward_vel = states["robot_states"]["root_joints_vel"][:, 0]
        forward_reward = forward_vel

        action_penalty = self._action_penalty * torch.sum(actions**2, dim=-1)

        reward = height_reward + angle_reward + forward_reward + action_penalty
        return reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            termination = robot_states["root_joints_pos"][:, 1] < self._termination_height
        return termination

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
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self._robot.set_dofs_position(
            position=motor_dof_pos,
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )

    def _set_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(self._num_envs, self._num_actions)
        actions = actions.clamp(min=-1.0, max=1.0) * self._motor_strength
        self._robot.control_dofs_force(actions, dofs_idx_local=self._motors_dof_idx)

    def render(self, env_ids: Optional[Sequence[int]] = None) -> None:
        # TODO: current camera is initialized only when vis_obs is True
        if env_ids is None:
            env_ids = self._rendered_envs_idx
        data = self._camera.read(envs_idx=env_ids)
        return data.rgb

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        root_joints_pos = self._robot.get_dofs_position(self._root_dof_idx, envs_idx=env_ids)
        motor_joints_pos = self._robot.get_dofs_position(self._motors_dof_idx, envs_idx=env_ids)
        root_joints_vel = self._robot.get_dofs_velocity(self._root_dof_idx, envs_idx=env_ids)
        motor_joints_vel = self._robot.get_dofs_velocity(self._motors_dof_idx, envs_idx=env_ids)

        robot_states = {
            "root_joints_pos": root_joints_pos.clone(),
            "motor_joints_pos": motor_joints_pos.clone(),
            "root_joints_vel": root_joints_vel.clone(),
            "motor_joints_vel": motor_joints_vel.clone(),
        }

        states = {
            "robot_states": robot_states,
            "progress_buf": self._progress_buf[env_ids].clone(),
        }

        return states

    def set_states(self, states: Dict[str, Any], env_ids: Optional[Sequence[int]] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        robot_states = states["robot_states"]

        self._robot.set_dofs_position(
            position=robot_states["root_joints_pos"],
            dofs_idx_local=self._root_dof_idx,
            envs_idx=env_ids,
            zero_velocity=False,
        )

        self._robot.set_dofs_position(
            position=robot_states["motor_joints_pos"],
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=False,
        )

        self._robot.set_dofs_velocity(
            velocity=robot_states["root_joints_vel"],
            dofs_idx_local=self._root_dof_idx,
            envs_idx=env_ids,
        )

        self._robot.set_dofs_velocity(
            velocity=robot_states["motor_joints_vel"],
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
        )

        self._progress_buf[env_ids] = states["progress_buf"].clone()
