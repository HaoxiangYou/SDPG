import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import axis_angle_to_quat, inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


def gs_rand(lower, upper, batch_shape):
    """Random tensor generator with shape matching."""
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2(GenesisEnv):
    """Go2 environment."""

    _num_actions = 12
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,))

    def __init__(
        self,
        num_envs: int,
        vis_obs: bool = False,
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

        episode_length = 1000  # Will be converted based on dt in reference
        early_termination = True
        self._observation_space = spaces.Dict(
            {
                "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(45,)),
            }
        )
        self._sim_dt = sim_options.dt

        super().__init__(
            num_envs=num_envs,
            episode_length=episode_length,
            early_termination=early_termination,
            sensors_args=sensors_args,
            seed=seed,
            randomize_init=randomize_init,
            nominal_env_ids=nominal_env_ids,
            device=device,
            show_viewer=show_viewer,
            sim_options=sim_options,
            viewer_options=viewer_options,
            vis_options=vis_options,
            show_FPS=show_FPS,
        )

    def init_scene(self) -> None:
        """Initialize the scene."""
        # Add plane
        self._plane = self._scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # Add go2 robot
        base_init_pos = [0.0, 0.0, 0.27]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]

        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/unitree_go2/go2_genesis.xml")),
        )

        # Joint names from go2 configuration
        self._motor_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

        self._penalized_contact_link_names = ["base", "thigh", "calf"]
        self._feet_link_names = ["foot"]
        self._base_link_name = "base"
        self._num_feet = 4

        # Get DOF indices - need to build scene first or get joint info after adding
        # Will be set after scene is built
        self._base_dof_idx = None  # Base joint DOFs (6 DOFs: 3 translation + 3 rotation)
        self._motors_dof_idx = None

        def find_link_indices(names):
            link_indices = list()
            for link in self._robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self._robot.link_start)
            return link_indices

        self._penalized_contact_link_indices = find_link_indices(self._penalized_contact_link_names)
        self._feet_link_indices = find_link_indices(self._feet_link_names)

        assert len(self._penalized_contact_link_indices) > 0
        assert len(self._feet_link_indices) > 0

        # PD control parameters
        self._kp = 20.0
        self._kd = 0.5

        # Default joint angles [rad]
        self._default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

        # Base initialization
        self._base_init_pos = torch.tensor(base_init_pos, dtype=gs.tc_float, device=self._device)
        self._base_init_quat = torch.tensor(base_init_quat, dtype=gs.tc_float, device=self._device)
        self._inv_base_init_quat = inv_quat(self._base_init_quat)

        # Action parameters
        self._action_scale = 0.25
        self._clip_actions = 100.0

        # Observation scales
        self._obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }

        # Termination parameters
        self._termination_roll_threshold = 10.0  # Convert degrees to radians
        self._termination_pitch_threshold = 10.0

        # Global gravity direction
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=self._device)

        # Reward configuration
        self._soft_dof_limit = 0.9
        self._reward_base_height_target = 0.36
        self._reward_foot_clearance_target = 0.05
        self._reward_foot_height_offset = 0.022
        self._reward_foot_clearance_tracking_sigma = 0.01
        self._only_positive_rewards = True
        self._reward_tracking_sigma = 0.25

        self._reward_scales = {
            # joint limit
            "dof_pos_limits": -10.0,
            "collision": -1.0,
            # command tracking
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            # smooth
            "lin_vel_z": -0.5,
            "base_height": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "dof_vel": -5.0e-4,
            "dof_acc": -2.0e-7,
            "action_rate": -0.01,
            # "action_smoothness": -0.01,
            "torques": -2.0e-4,
            # gait
            "feet_air_time": 1.0,
            "foot_clearance": 0.5,
        }

        # Command configuration
        self._command_cfg = {
            "num_commands": 3,
            # "lin_vel_x_range": [-1.5, 1.5],
            # "lin_vel_y_range": [-0.8, 0.8],
            # "ang_vel_range": [-1.2, 1.2],
            "lin_vel_x_range": [0.5, 0.5],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [0.0, 0.0],
        }
        self._command_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self._command_cfg["lin_vel_x_range"],
                self._command_cfg["lin_vel_y_range"],
                self._command_cfg["ang_vel_range"],
                strict=False,
            )
        ]

        # prepare list of functions
        self._reward_functions = []
        self._reward_names = []
        for name, scale in self._reward_scales.items():
            self._reward_names.append(name)
            name = "_reward_" + name
            self._reward_functions.append(getattr(self, name))

        # Buffers for observation computation
        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)
        self._last_motor_joints_vel = torch.zeros(self._num_envs, self._num_actions, device=self._device)
        self._commands = torch.zeros(
            self._num_envs, self._command_cfg["num_commands"], device=self._device
        )  # [lin_vel_x, lin_vel_y, ang_vel]
        self._link_contact_forces = torch.zeros(
            (self._num_envs, self._robot.n_links, 3), device=self._device, dtype=torch.float
        )
        self._feet_air_time = torch.zeros(
            (self._num_envs, self._num_feet),
            device=self._device,
            dtype=torch.float,
        )
        self._last_contacts = torch.zeros(
            (self._num_envs, self._num_feet),
            device=self._device,
            dtype=torch.bool,
        )

        self._foot_vel = torch.zeros(
            (self._num_envs, self._num_feet, 3),
            device=self._device,
            dtype=torch.float,
        )

        self._last_foot_vel = torch.zeros(
            (self._num_envs, self._num_feet, 3),
            device=self._device,
            dtype=torch.float,
        )

        self._feet_pos = torch.zeros(
            (self._num_envs, self._num_feet, 3),
            device=self._device,
            dtype=torch.float,
        )

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(0.0, 1.0), n_envs_per_row=self._num_envs)

        self._base_dof_idx = self._robot.base_joint.dofs_idx_local  # only use this for resetting the base velocities

        # Get motor DOF indices after scene is built
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]

        # Set PD control parameters
        self._robot.set_dofs_kp([self._kp] * self._num_actions, self._motors_dof_idx)
        self._robot.set_dofs_kv([self._kd] * self._num_actions, self._motors_dof_idx)

        # Initialize default DOF positions
        self._default_dof_pos = torch.tensor(
            [self._default_joint_angles[name] for name in self._motor_joint_names],
            dtype=gs.tc_float,
            device=self._device,
        )

        # Initialize DOF positions for all environments
        init_dof_pos = self._default_dof_pos.unsqueeze(0).repeat(self._num_envs, 1)
        self._robot.set_dofs_position(init_dof_pos, dofs_idx_local=self._motors_dof_idx, zero_velocity=True)

        # Get joint limits
        self._dof_pos_limits = torch.stack(self._robot.get_dofs_limit(self._motors_dof_idx), dim=1)
        self._torque_limits = self._robot.get_dofs_force_range(self._motors_dof_idx)[1]
        for i in range(self._dof_pos_limits.shape[0]):
            # soft limits
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = m - 0.5 * r * self._soft_dof_limit
            self._dof_pos_limits[i, 1] = m + 0.5 * r * self._soft_dof_limit

    def _post_physics_step(self) -> None:
        # resample commands when is half of the episode lenght
        self._resample_commands(self._progress_buf % int(self._episode_length / 2) == 0)
        self._link_contact_forces[:] = torch.tensor(
            self._robot.get_links_net_contact_force(),
            device=self._device,
            dtype=gs.tc_float,
        )
        self._foot_vel = self._robot.get_links_vel()[:, self._feet_link_indices, :]
        self._feet_pos = self._robot.get_links_pos()[:, self._feet_link_indices, :]

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Compute observations based on go2_env.py structure."""
        robot_states = states["robot_states"]

        # Get base angular velocity and projected gravity
        base_ang_vel = robot_states["base_ang_vel"]
        projected_gravity = robot_states["projected_gravity"]

        # Commands (velocity commands - simplified, can be extended)
        commands = robot_states.get("commands", torch.zeros(self._num_envs, 3, device=self._device))
        commands_scale = torch.tensor(
            [self._obs_scales["lin_vel"], self._obs_scales["lin_vel"], self._obs_scales["ang_vel"]],
            device=self._device,
            dtype=gs.tc_float,
        )

        # DOF positions and velocities
        dof_pos = robot_states["motor_joints_pos"]
        dof_vel = robot_states["motor_joints_vel"]

        # Actions (last applied actions)
        prev_actions = robot_states["prev_actions"]

        # Compute privileged observations (matching genesis go2_env.py structure)
        privileged_observations = torch.cat(
            (
                base_ang_vel * self._obs_scales["ang_vel"],  # 3
                projected_gravity,  # 3
                commands * commands_scale,  # 3
                (dof_pos - self._default_dof_pos) * self._obs_scales["dof_pos"],  # 12
                dof_vel * self._obs_scales["dof_vel"],  # 12
                prev_actions,  # 12
            ),
            dim=-1,
        )

        observations = {
            "privileged_observations": privileged_observations,
        }
        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Compute reward based on reward_cfg from go2_env.py."""
        robot_states = states["robot_states"]

        # Compute all reward components
        reward = torch.zeros(self._num_envs, device=self._device)

        for i in range(len(self._reward_functions)):
            name = self._reward_names[i]
            reward += self._reward_functions[i](robot_states, actions) * self._reward_scales[name]

        return reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        """Compute termination based on roll/pitch angles."""
        # Compute base Euler angles (roll, pitch, yaw)
        base_quat = self._robot.get_quat()
        transformed_quat = transform_quat_by_quat(self._inv_base_init_quat, base_quat)
        base_euler = quat_to_xyz(transformed_quat, rpy=True, degrees=True)
        self._base_euler = base_euler
        termination = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        if self._early_termination:
            # Terminate if roll or pitch exceeds threshold
            termination |= torch.abs(base_euler[:, 0]) > self._termination_roll_threshold
            termination |= torch.abs(base_euler[:, 1]) > self._termination_pitch_threshold

        return termination

    def _resample_commands(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Resample velocity commands for specified environments.

        Args:
            envs_idx: Boolean tensor mask of environments to resample commands for,
                     or tensor of environment indices (Long), or None to resample for all environments.
        """
        commands = gs_rand(*self._command_limits, (self._num_envs,))
        if envs_idx is None:
            self._commands.copy_(commands)
        else:
            # Convert indices to boolean mask if needed
            if envs_idx.dtype == torch.bool:
                # Already a boolean mask
                mask = envs_idx
            else:
                # Convert indices to boolean mask
                mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
                mask[envs_idx] = True

            # Resample commands only for specified environments using boolean mask
            torch.where(mask[:, None], commands, self._commands, out=self._commands)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environments by index."""
        if len(env_ids) == 0:
            return

        base_pos = self._base_init_pos.unsqueeze(0).repeat(len(env_ids), 1)
        base_quat = self._base_init_quat.unsqueeze(0).repeat(len(env_ids), 1)
        motor_dof_pos = self._default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)

        if self._randomize_init:
            # Add small random perturbations
            base_pos = base_pos + (torch.rand_like(base_pos) - 0.5) * 0.05
            angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
            axis = torch.nn.functional.normalize(torch.rand(len(env_ids), 3, device=self.device) - 0.5)
            base_quat = transform_quat_by_quat(base_quat, axis_angle_to_quat(angle, axis))
            motor_dof_pos = motor_dof_pos + (torch.rand_like(motor_dof_pos) - 0.5) * 0.1

        # Set base pose using set_pos and set_quat (world frame, but env_spacing handles multi-env)
        self._robot.set_pos(base_pos, envs_idx=env_ids, zero_velocity=True)
        self._robot.set_quat(base_quat, envs_idx=env_ids, zero_velocity=True)

        # Set motor DOF positions
        self._robot.set_dofs_position(
            position=motor_dof_pos,
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )

        # Reset previous actions
        self._prev_actions[env_ids] = torch.zeros(len(env_ids), self._num_actions, device=self._device)
        self._last_motor_joints_vel[env_ids] = torch.zeros(len(env_ids), self._num_actions, device=self._device)
        self._last_foot_vel[env_ids] = torch.zeros(len(env_ids), self._num_feet, 3, device=self._device)
        self._last_contacts[env_ids] = torch.zeros(len(env_ids), self._num_feet, dtype=torch.bool, device=self._device)
        self._foot_vel[env_ids] = torch.zeros(len(env_ids), self._num_feet, 3, device=self._device)

        # Resample commands for reset environments
        self._resample_commands(env_ids)

    def _set_actions(self, actions: torch.Tensor) -> None:
        """Set actions using position control (PD control)."""
        actions = actions.view(self._num_envs, self._num_actions)
        actions = torch.clip(actions, -self._clip_actions, self._clip_actions)

        # Convert actions to target DOF positions
        target_dof_pos = actions * self._action_scale + self._default_dof_pos

        # Control DOFs using position control (PD control is set in build_scene)
        self._robot.control_dofs_position(target_dof_pos, self._motors_dof_idx)

        # Store actions for observation
        self._prev_actions = actions.clone()
        self._last_motor_joints_vel = self._robot.get_dofs_velocity(self._motors_dof_idx)
        self._last_foot_vel = self._robot.get_links_vel()[:, self._feet_link_indices, :]

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Get robot states for computing observations and rewards."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        # Get base pose using get_pos and get_quat
        base_pos = self._robot.get_pos(envs_idx=env_ids)
        base_quat = self._robot.get_quat(envs_idx=env_ids)
        # Transform velocities to base frame
        inv_base_quat = inv_quat(base_quat)

        base_lin_vel = transform_by_quat(self._robot.get_vel(), inv_base_quat)
        base_ang_vel = transform_by_quat(self._robot.get_ang(), inv_base_quat)

        # Project gravity to base frame
        projected_gravity = transform_by_quat(self._global_gravity, inv_base_quat)

        # Get DOF positions and velocities
        motor_joints_pos = self._robot.get_dofs_position(self._motors_dof_idx, envs_idx=env_ids)
        motor_joints_vel = self._robot.get_dofs_velocity(self._motors_dof_idx, envs_idx=env_ids)

        robot_states = {
            "base_pos": base_pos.clone(),
            "base_quat": base_quat.clone(),
            "base_lin_vel": base_lin_vel.clone(),
            "base_ang_vel": base_ang_vel.clone(),
            "projected_gravity": projected_gravity.clone(),
            "motor_joints_pos": motor_joints_pos.clone(),
            "motor_joints_vel": motor_joints_vel.clone(),
            "prev_actions": self._prev_actions[env_ids].clone(),
            "last_motor_joints_vel": self._last_motor_joints_vel[env_ids].clone(),
            "last_foot_vel": self._last_foot_vel[env_ids].clone(),
            "last_contacts": self._last_contacts[env_ids].clone(),
            "commands": self._commands[env_ids].clone(),
        }

        states = {
            "robot_states": robot_states,
            "progress_buf": self._progress_buf[env_ids].clone(),
        }

        return states

    def set_states(self, states: Dict[str, Any], env_ids: Optional[Sequence[int]] = None) -> None:
        """Set robot states."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        robot_states = states["robot_states"]

        # Normalize quaternion to ensure consistency (q and -q represent the same rotation)
        base_quat = robot_states["base_quat"]
        # Normalize and ensure w component is non-negative (canonical form)
        base_quat_norm = torch.nn.functional.normalize(base_quat, dim=-1)
        # Ensure the quaternion has positive w component (canonical representation)
        # This helps avoid sign flips that can affect euler angle computation
        base_quat_canonical = torch.where(
            (base_quat_norm[..., 0:1] < 0).expand_as(base_quat_norm), -base_quat_norm, base_quat_norm
        )

        # Set base pose using set_pos and set_quat
        self._robot.set_pos(robot_states["base_pos"], envs_idx=env_ids)
        self._robot.set_quat(base_quat_canonical, envs_idx=env_ids)

        # Set motor DOF positions
        self._robot.set_dofs_position(
            position=robot_states["motor_joints_pos"],
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=False,
        )

        # Set base and motor DOF velocities
        base_dof_vel = torch.cat([robot_states["base_lin_vel"], robot_states["base_ang_vel"]], dim=-1)
        self._robot.set_dofs_velocity(
            velocity=torch.cat([base_dof_vel, robot_states["motor_joints_vel"]], dim=-1),
            dofs_idx_local=self._base_dof_idx + self._motors_dof_idx,
            envs_idx=env_ids,
        )

        # Update progress buffer
        self._progress_buf[env_ids] = states["progress_buf"].clone()

        # Update previous actions if provided
        self._prev_actions[env_ids] = robot_states["prev_actions"].clone()
        self._last_motor_joints_vel[env_ids] = robot_states["motor_joints_vel"].clone()
        self._last_foot_vel[env_ids] = robot_states["last_foot_vel"].clone()

    def _reward_tracking_lin_vel(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Tracking of linear velocity commands (xy axes)."""
        base_lin_vel = robot_states["base_lin_vel"]
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self._reward_tracking_sigma)

    def _reward_tracking_ang_vel(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Tracking of angular velocity commands (yaw)."""
        base_ang_vel = robot_states["base_ang_vel"]
        ang_vel_error = torch.square(self._commands[:, 2] - base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self._reward_tracking_sigma)

    def _reward_lin_vel_z(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize z axis base linear velocity."""
        base_lin_vel = robot_states["base_lin_vel"]
        return torch.square(base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize xy axis base angular velocity."""
        base_ang_vel = robot_states["base_ang_vel"]
        return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize projected gravity."""
        projected_gravity = robot_states["projected_gravity"]
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

    def _reward_dof_vel(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize DOF velocities."""
        dof_vel = robot_states["motor_joints_vel"]
        return torch.sum(torch.square(dof_vel), dim=1)

    def _reward_dof_acc(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize DOF accelerations."""
        dof_acc = (robot_states["last_motor_joints_vel"] - robot_states["motor_joints_vel"]) / self._sim_dt
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_action_rate(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize changes in actions."""
        prev_actions = robot_states["prev_actions"]
        return torch.sum(torch.square(prev_actions - actions), dim=1)

    def _reward_similar_to_default(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize joint poses far away from default pose."""
        dof_pos = robot_states["motor_joints_pos"]
        return torch.sum(torch.square(dof_pos - self._default_dof_pos), dim=1)

    def _reward_base_height(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize base height away from target."""
        base_pos = robot_states["base_pos"]
        return torch.square(base_pos[:, 2] - self._reward_base_height_target)

    def _reward_collision(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize collisions."""
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self._link_contact_forces[:, self._penalized_contact_link_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_feet_air_time(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Reward long steps
        contact = self._link_contact_forces[:, self._feet_link_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.0) * contact_filt
        self._feet_air_time += self._sim_dt
        rew_airTime = torch.sum(
            (self._feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self._commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self._feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_dof_vel_stand_still(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize DOF velocities while standing still."""
        dof_vel = robot_states["motor_joints_vel"]
        return torch.sum(torch.abs(dof_vel), dim=1) * (torch.norm(self._commands[:, :2], dim=1) < 0.1)

    def _reward_dof_pos_stand_still(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Penalize DOF positions while standing still."""
        dof_pos = robot_states["motor_joints_pos"]
        return torch.sum(torch.square(dof_pos - self._default_dof_pos), dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) < 0.1
        )

    def _reward_feet_contact_still(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Encourage feet contact with the ground at zero commands
        contacts = self._link_contact_forces[:, self._feet_link_indices, 2] > 1.0
        full_contact = torch.sum(1.0 * contacts, dim=1) == len(self._feet_link_indices)
        return 1.0 * full_contact * (torch.norm(self._commands[:, :2], dim=1) < 0.1)

    def _reward_foot_clearance(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Encourage feet to be close to desired height while swinging
        foot_vel_xy_norm = torch.norm(self._foot_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm
            * torch.square(
                self._feet_pos[:, :, 2] - self._reward_foot_clearance_target - self._reward_foot_height_offset
            ),
            dim=-1,
        )
        return torch.exp(-clearance_error / self._reward_foot_clearance_tracking_sigma)

    def _reward_dof_pos_limits(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Penalize dof positions too close to the limit
        dof_pos = robot_states["motor_joints_pos"]
        out_of_limits = -(dof_pos - self._dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (dof_pos - self._dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torques(self, robot_states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Penalize torques
        torques = self._robot.get_dofs_control_force(self._motors_dof_idx)
        return torch.sum(torch.square(torques), dim=1)
