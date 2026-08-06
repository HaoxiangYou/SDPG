import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from gym import spaces

from envs.mujoco_env.mujoco_env import MujocoEnv


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b for wxyz quaternions (matches genesis transform_quat_by_quat)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by unit wxyz quaternions (matches genesis transform_by_quat)."""
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def _axis_angle_to_quat(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion for a rotation of angle (rad) about unit axis (matches genesis)."""
    half = (0.5 * angle).unsqueeze(-1)
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)


class Humanoid(MujocoEnv):
    """Humanoid environment (MJX). Mirrors the genesis backend humanoid task."""

    _num_actions = 21
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(21,))

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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(76,)),
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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(76,)),
                }
            )

        super().__init__(
            num_envs=num_envs,
            episode_length=episode_length,
            xml_path=os.path.join(os.path.dirname(__file__), "../../assets/mujoco/humanoid.xml"),
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
        # Free-joint root: qpos = [pos(3), quat wxyz(4), joints(21)], qvel =
        # [lin vel world(3), ang vel body(3), joints(21)] — the same frames the
        # genesis backend reports for the base (see its get_states NOTE).
        # Motor states and actions follow the genesis _motor_joint_names order,
        # which is the MJCF actuator order, NOT the kinematic-tree joint order
        # (e.g. *_ankle_x comes before *_ankle_y).
        self._motor_joint_names = [
            "abdomen_y",
            "abdomen_z",
            "abdomen_x",
            "right_hip_x",
            "right_hip_z",
            "right_hip_y",
            "right_knee",
            "right_ankle_x",
            "right_ankle_y",
            "left_hip_x",
            "left_hip_z",
            "left_hip_y",
            "left_knee",
            "left_ankle_x",
            "left_ankle_y",
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ]
        mj_model = self._sim.mj_model
        actuator_joints = [mj_model.joint(int(mj_model.actuator_trnid[i, 0])).name for i in range(mj_model.nu)]
        if actuator_joints != self._motor_joint_names:
            raise ValueError(f"MJCF actuator order {actuator_joints} does not match {self._motor_joint_names}")
        self._motor_qpos_idx = torch.tensor(
            [mj_model.joint(name).qposadr[0] for name in self._motor_joint_names],
            device=self._device,
            dtype=torch.long,
        )
        self._motor_dof_idx = torch.tensor(
            [mj_model.joint(name).dofadr[0] for name in self._motor_joint_names],
            device=self._device,
            dtype=torch.long,
        )

        # Gear in the MJCF applies the genesis _motor_strength scaling; kept here
        # only to expose the applied torques as prev_actions (part of the obs).
        self._motor_strength = torch.as_tensor(mj_model.actuator_gear[:, 0], dtype=self._sim_dtype, device=self._device)
        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, dtype=self._sim_dtype, device=self._device)

        self._default_base_pos = torch.tensor([0.0, 0.0, 1.35], dtype=self._sim_dtype, device=self._device).repeat(
            self._num_envs, 1
        )
        self._default_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self._sim_dtype, device=self._device).repeat(
            self._num_envs, 1
        )
        self._default_motor_dof_pos = torch.zeros(
            self._num_envs, self._num_actions, dtype=self._sim_dtype, device=self._device
        )

        self._target = torch.tensor([200.0, 0.0, 0.0], dtype=self._sim_dtype, device=self._device).repeat(
            self._num_envs, 1
        )
        self._basis_heading = torch.tensor([1.0, 0.0, 0.0], dtype=self._sim_dtype, device=self._device)
        self._basis_up = torch.tensor([0.0, 1.0, 0.0], dtype=self._sim_dtype, device=self._device)
        self._joint_vel_obs_scale = 0.1
        self._height_reward_scale = 10.0
        self._termination_height_lower_bound = 0.74
        self._termination_height_upper_bound = 2.0
        self._termination_height_tolerance = 0.1
        self._action_penalty = -0.002
        self._extreme_vel_threshold = 100.0  # terminate if any |velocity| > this (physics blow-up)

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
        # adapt from Jie Xu's implementation (same layout as the genesis humanoid)
        n_batch = states["progress_buf"].shape[0]
        robot_states = states["robot_states"]

        base_pose = robot_states["base_pose"]
        height = base_pose[:, 2:3]
        base_quat = base_pose[:, 3:]

        base_vel = robot_states["base_vel"]

        joints_pos = robot_states["motor_joints_pos"]
        joints_vel = robot_states["motor_joints_vel"]

        prev_actions = robot_states["prev_actions"]

        target_dirs = self._target[:n_batch] - base_pose[:, :3]
        target_dirs[:, 2] = 0
        target_dirs = torch.nn.functional.normalize(target_dirs)
        heading_vec = _quat_rotate(base_quat, self._basis_heading.expand(n_batch, 3))
        up_vec = _quat_rotate(base_quat, self._basis_up.expand(n_batch, 3))

        privileged_observations = torch.cat(
            [
                height,
                base_quat,
                base_vel,
                joints_pos,
                joints_vel * self._joint_vel_obs_scale,
                up_vec[:, 1:2],
                (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),
                prev_actions,
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
        # Jie Xu's reward function (mirrors the genesis humanoid)
        n_batch = states["progress_buf"].shape[0]

        height = states["robot_states"]["base_pose"][:, 2]
        height_diff = height - (self._termination_height_lower_bound + self._termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self._termination_height_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self._height_reward_scale * height_reward, height_reward)

        forward_vel = states["robot_states"]["base_vel"][:, 0]
        forward_reward = forward_vel

        base_pose = states["robot_states"]["base_pose"]
        base_quat = base_pose[:, 3:]
        target_dirs = self._target[:n_batch] - base_pose[:, :3]
        target_dirs[:, 2] = 0
        target_dirs = torch.nn.functional.normalize(target_dirs)
        heading_vec = _quat_rotate(base_quat, self._basis_heading.expand(n_batch, 3))
        up_vec = _quat_rotate(base_quat, self._basis_up.expand(n_batch, 3))

        up_reward = 0.1 * up_vec[:, 1]
        heading_reward = (heading_vec * target_dirs).sum(dim=-1)

        action_penalty = self._action_penalty * torch.sum(actions**2, dim=-1)

        return height_reward + forward_reward + up_reward + heading_reward + action_penalty

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            height = robot_states["base_pose"][:, 2]
            termination = height < self._termination_height_lower_bound
            # humanoid z position too high is likely due to simulation physics failure.
            termination = torch.where(height > self._termination_height_upper_bound, True, termination)
            # Terminate if any velocity is extreme (e.g. physics solver blow-up)
            extreme_vel = (torch.abs(robot_states["base_vel"]) > self._extreme_vel_threshold).any(dim=-1) | (
                torch.abs(robot_states["motor_joints_vel"]) > self._extreme_vel_threshold
            ).any(dim=-1)
            termination = termination | extreme_vel
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        base_pos = self._default_base_pos[env_ids]
        base_quat = self._default_base_quat[env_ids]
        motor_dof_pos = self._default_motor_dof_pos[env_ids]

        if self._randomize_init:
            noise = torch.rand(base_pos.shape, generator=self._rng, device=self._device, dtype=self._sim_dtype)
            base_pos = base_pos + (noise - 0.5) * 0.1
            angle = (
                torch.rand(len(env_ids), generator=self._rng, device=self._device, dtype=self._sim_dtype) - 0.5
            ) * (np.pi / 12.0)
            axis = torch.nn.functional.normalize(
                torch.rand(len(env_ids), 3, generator=self._rng, device=self._device, dtype=self._sim_dtype) - 0.5
            )
            base_quat = _quat_mul(base_quat, _axis_angle_to_quat(angle, axis))
            noise = torch.rand(motor_dof_pos.shape, generator=self._rng, device=self._device, dtype=self._sim_dtype)
            motor_dof_pos = motor_dof_pos + (noise - 0.5) * 0.1

        qpos = self._qpos0.expand(len(env_ids), -1).clone()
        qpos[:, 0:3] = base_pos
        qpos[:, 3:7] = base_quat
        qpos[:, self._motor_qpos_idx] = motor_dof_pos
        qvel = torch.zeros(len(env_ids), self._sim.nv, dtype=self._sim_dtype, device=self._device)

        self._set_qpos_qvel(env_ids, qpos, qvel)

        self._prev_actions[env_ids] = torch.zeros(
            len(env_ids), self._num_actions, dtype=self._sim_dtype, device=self._device
        )

        if self._vis_obs:
            # Find which nominal environments are being reset and refresh their
            # whole image stack with a freshly rendered frame.
            mask = torch.isin(self.nominal_env_ids, env_ids)
            nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]

            if len(nominal_idx_to_reset) > 0:
                reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
                new_img = self.render(env_ids=reset_nominal_env_ids)

                self._imgs_buf[nominal_idx_to_reset] = new_img.unsqueeze(1)

    def _set_qpos_qvel(self, env_ids: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """Write per-env absolute (qpos, qvel) rows.

        Autograd-safe like the base _set_dof_state (clone + index-assign), but
        without its qpos0-relative storage: the free joint's qpos0 is nonzero
        (spawn height, unit quat), so reconstructing absolute targets through
        the offset costs a float rounding and would break the exact SDPG
        get_states/set_states round-trip.
        """
        new_qpos = self._qpos.clone()
        new_qvel = self._qvel.clone()
        new_qpos[env_ids] = qpos.to(self._sim_dtype)
        new_qvel[env_ids] = qvel.to(self._sim_dtype)
        self._qpos = new_qpos
        self._qvel = new_qvel

    def _action_to_ctrl(self, actions: torch.Tensor) -> torch.Tensor:
        # actions arrive clamped to [-1, 1]; actuator gear in the MJCF applies the
        # motor strength. Record the applied torques as prev_actions — the genesis
        # humanoid stores the scaled actions and exposes them in the observation.
        self._prev_actions = actions * self._motor_strength
        return actions

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
        rgb = self._batch_renderer.render(self._qpos[self.nominal_env_ids], self._qvel[self.nominal_env_ids])
        rows = torch.nonzero(torch.isin(self.nominal_env_ids, env_ids), as_tuple=True)[0]
        return rgb[rows]

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        qpos = self._qpos[env_ids]
        qvel = self._qvel[env_ids]

        # The base pose is exposed absolute (not qpos0-relative): the genesis
        # backend reports the absolute base position/quaternion, and the height
        # observation/reward read the absolute z. Motor joint refs are zero in
        # the MJCF, so their absolute angles equal the genesis readings too.
        robot_states = {
            "base_pose": qpos[:, 0:7].clone(),
            "base_vel": qvel[:, 0:6].clone(),
            "motor_joints_pos": qpos[:, self._motor_qpos_idx].clone(),
            "motor_joints_vel": qvel[:, self._motor_dof_idx].clone(),
            "prev_actions": self._prev_actions[env_ids].clone(),
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

        qpos = self._qpos0.expand(len(env_ids), -1).clone()
        qpos[:, 0:7] = robot_states["base_pose"].to(self._sim_dtype)
        qpos[:, self._motor_qpos_idx] = robot_states["motor_joints_pos"].to(self._sim_dtype)
        qvel = torch.zeros(len(env_ids), self._sim.nv, dtype=self._sim_dtype, device=self._device)
        qvel[:, 0:6] = robot_states["base_vel"].to(self._sim_dtype)
        qvel[:, self._motor_dof_idx] = robot_states["motor_joints_vel"].to(self._sim_dtype)

        self._set_qpos_qvel(env_ids, qpos.detach(), qvel.detach())

        self._prev_actions[env_ids] = robot_states["prev_actions"].clone()

        self._progress_buf[env_ids] = states["progress_buf"].clone()
