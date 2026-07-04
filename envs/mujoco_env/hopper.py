import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from gym import spaces

from envs.mujoco_env.mujoco_env import MujocoEnv


class Hopper(MujocoEnv):
    """Hopper environment (MJX). Mirrors the genesis backend hopper task."""

    _num_actions = 3
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def __init__(
        self,
        num_envs: int,
        vis_obs: bool = False,
        seed: int = 0,
        randomize_init: bool = True,
        nominal_env_ids: Optional[Sequence[int]] = None,
        device: torch.device | None = None,
        sensors_args: Dict[str, Any] | None = None,
        sim_options: Dict[str, Any] | None = None,
        show_viewer: bool = False,
        show_FPS: bool = False,
    ) -> None:
        episode_length = 1000
        early_termination = True

        if sensors_args is None:
            sensors_args = {
                "camera": {
                    "res": [256, 256],
                    "gpu_id": 0,
                    "use_rasterizer": False,
                }
            }
        self._vis_obs = vis_obs
        if vis_obs:
            self._num_image_stack = 3
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(11,)),
                    "RGB": spaces.Box(
                        low=0,
                        high=255,
                        dtype=np.uint8,
                        shape=(
                            self._num_image_stack * 3,
                            sensors_args["camera"]["res"][0],
                            sensors_args["camera"]["res"][1],
                        ),
                    ),
                }
            )
        else:
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(11,)),
                }
            )

        super().__init__(
            num_envs=num_envs,
            episode_length=episode_length,
            xml_path=os.path.join(os.path.dirname(__file__), "../../assets/mujoco/hopper.xml"),
            early_termination=early_termination,
            seed=seed,
            randomize_init=randomize_init,
            nominal_env_ids=nominal_env_ids,
            device=device,
            sensors_args=sensors_args,
            sim_options=sim_options,
            show_viewer=show_viewer,
            show_FPS=show_FPS,
        )

    def init_task(self) -> None:
        # qpos/qvel layout follows the MJCF joint order.
        self._root_dof_idx = [0, 1, 2]  # rootx (x slide), rootz (z slide), rooty (rotation)
        self._motor_dof_idx = [3, 4, 5]  # thigh_joint, leg_joint, foot_joint

        self._default_dof_pos = torch.zeros(
            self._num_envs, self._sim.nq, dtype=self._sim_dtype, device=self._device
        )

        self._termination_height_lower_bound = -0.45
        self._termination_height_upper_bound = 15.0
        self._termination_height_tolerance = 0.15
        self._termination_angle = torch.pi / 6.0
        self._termination_angle_tolerance = 0.05
        self._extreme_vel_threshold = 100.0  # terminate if any |velocity| > this (physics blow-up)
        self._height_reward_scale = 1.0
        self._angle_reward_scale = 1.0
        self._action_penalty = -1e-1

        if self._vis_obs:
            from envs.mujoco_env.batch_renderer import MadronaBatchRenderer

            camera_args = self._sensors_args["camera"]
            # Render only the nominal environments (auxiliary envs never need
            # visual observations), matching the genesis env_idx optimization.
            self._batch_renderer = MadronaBatchRenderer(
                mj_model=self._sim.mj_model,
                env_ids=self._nominal_env_ids.cpu().tolist(),
                width=camera_args["res"][0],
                height=camera_args["res"][1],
                gpu_id=camera_args.get("gpu_id", 0),
                use_rasterizer=camera_args.get("use_rasterizer", False),
            )
            self._imgs_buf = torch.zeros(
                self.nominal_env_ids.shape[0],
                self._num_image_stack,
                camera_args["res"][0],
                camera_args["res"][1],
                3,
                device=self._device,
                dtype=torch.uint8,
            )

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}
        robot_states = states["robot_states"]
        privileged_observations = torch.cat(
            [
                robot_states["root_joints_pos"][:, 1:],
                robot_states["motor_joints_pos"],
                robot_states["root_joints_vel"],
                robot_states["motor_joints_vel"],
            ],
            dim=-1,
        )
        observations["privileged_observations"] = privileged_observations.to(torch.float32)

        if self._vis_obs:
            batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
            # NOTE: for SDPG agent, RGB observation and privileged observations may has different shapes
            # Reshape: (batch, num_stack, H, W, 3) -> (batch, num_stack * 3, H, W)
            observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                batch_size, num_stack * rgb, img_height, img_width
            )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Jie Xu's reward function
        height = states["robot_states"]["root_joints_pos"][:, 1]
        height_diff = height - (self._termination_height_lower_bound + self._termination_height_tolerance)
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
            height = robot_states["root_joints_pos"][:, 1]
            termination = height < self._termination_height_lower_bound
            # Prevent the algo exploiting physical solver by limiting the height
            termination = torch.where(height > self._termination_height_upper_bound, True, termination)
            # Terminate if any velocity is extreme (e.g. physics solver blow-up)
            extreme_vel = (torch.abs(robot_states["root_joints_vel"]) > self._extreme_vel_threshold).any(dim=-1) | (
                torch.abs(robot_states["motor_joints_vel"]) > self._extreme_vel_threshold
            ).any(dim=-1)
            termination = termination | extreme_vel
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        dof_pos = self._default_dof_pos[env_ids]

        if self._randomize_init:
            noise = torch.rand(dof_pos.shape, generator=self._rng, device=self._device, dtype=self._sim_dtype)
            dof_pos = dof_pos + (noise - 0.5) * 0.1

        dof_vel = torch.zeros_like(dof_pos)

        self._set_dof_state(env_ids, dof_pos, dof_vel)

        if self._vis_obs:
            # Find which nominal environments are being reset and refresh their
            # whole image stack with a freshly rendered frame.
            mask = torch.isin(self.nominal_env_ids, env_ids)
            nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]

            if len(nominal_idx_to_reset) > 0:
                reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
                new_img = self.render(env_ids=reset_nominal_env_ids)

                self._imgs_buf[nominal_idx_to_reset] = new_img.unsqueeze(1)

    def _post_physics_step(self) -> None:
        """Update image buffer by rolling frames and appending new image."""
        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            # Roll the buffer to shift old frames: [t-2, t-1, t-0] -> [t-1, t-0, None]
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img

    def render(self, env_ids: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        """Render the nominal environments; returns uint8 (len(env_ids), H, W, 3).

        The batch renderer always renders all nominal worlds (madrona world
        buffers are fixed at init); env_ids selects which rows are returned.
        """
        if not self._vis_obs:
            return None
        if env_ids is None:
            env_ids = self.nominal_env_ids
        rgb = self._batch_renderer.render(
            self._qpos[self.nominal_env_ids], self._qvel[self.nominal_env_ids]
        )
        rows = torch.nonzero(torch.isin(self.nominal_env_ids, env_ids), as_tuple=True)[0]
        return rgb[rows]

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        dof_pos = self._qpos[env_ids] - self._qpos0
        dof_vel = self._qvel[env_ids]

        robot_states = {
            "root_joints_pos": dof_pos[:, self._root_dof_idx].clone(),
            "motor_joints_pos": dof_pos[:, self._motor_dof_idx].clone(),
            "root_joints_vel": dof_vel[:, self._root_dof_idx].clone(),
            "motor_joints_vel": dof_vel[:, self._motor_dof_idx].clone(),
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

        dof_pos = torch.cat([robot_states["root_joints_pos"], robot_states["motor_joints_pos"]], dim=-1)
        dof_vel = torch.cat([robot_states["root_joints_vel"], robot_states["motor_joints_vel"]], dim=-1)

        self._set_dof_state(env_ids, dof_pos.detach(), dof_vel.detach())

        self._progress_buf[env_ids] = states["progress_buf"].clone()
