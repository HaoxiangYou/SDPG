import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import axis_angle_to_quat, pos_lookat_up_to_T, transform_by_quat, transform_quat_by_quat, inv_quat
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class G1(GenesisEnv):
    """G1 environment."""

    _num_actions = 23
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(23,))

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
        episode_length = 1000
        early_termination = True

        if sensors_args is None:
            sensors_args = {
                "camera": {
                    "res": [256, 256],
                    "pos": [-3.0, 0.0, 1.0],
                    "lookat": [0.0, 0.0, 0.0],
                    "fov": 60.0,
                    "lights": {
                        "pos": [0.0, 0.0, 2.0],
                        "dir": [0.0, 0.0, -1.0],
                        "intensity": 0.8,
                        "color": [1.0, 1.0, 1.0],
                    },
                    "directional": True,
                    "castshadow": False,
                }
            }

        self._vis_obs = vis_obs
        if vis_obs:
            self._num_image_stack = 3
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(81,)),
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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(81,)),
                }
            )

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

        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/g1_description/g1_23dof_rev_1_0.xml")),
            surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0, 1.0)),  # Orange color from humanoid.xml default
        )
        self._plane = self._scene.add_entity(gs.morphs.Plane())

        # A record of the previous actions
        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)

        self._motor_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ]

        self._feet_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self._torso_contact_link_names = ["torso_link", "torso"]

        self._hip_knee_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
        ]
        self._ankle_joint_names = [
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        self._hip_dev_joint_names = [
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
        ]
        self._arms_dev_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ]
        self._fingers_dev_joint_names: list[str] = []
        self._torso_dev_joint_names: list[str] = []

        self._base_dof_idx = self._robot.base_joint.dofs_idx_local
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]
        self._motor_dof_to_q = {dof: q for q, dof in enumerate(self._motors_dof_idx)}
        self._all_dof_idx = [joint.dof_start for joint in self._robot.joints if getattr(joint, "dof_start", None) is not None]
        self._last_motor_joints_vel = torch.zeros(self._num_envs, len(self._motors_dof_idx), device=self._device)

        self._motor_strength = torch.tensor(
            [
                88.0,
                139.0,
                88.0,
                139.0,
                50.0,
                50.0,
                88.0,
                139.0,
                88.0,
                139.0,
                50.0,
                50.0,
                88.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
            ],
            device=self._device,
        )

        self._default_base_pos = torch.tensor([0, 0, 0.8], device=self._device).repeat(self._num_envs, 1)
        self._default_base_quat = torch.tensor([1, 0, 0, 0], device=self._device).repeat(self._num_envs, 1)
        self._default_motor_dof_pos = torch.zeros(self._num_envs, len(self._motors_dof_idx), device=self._device)
        self._current_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)
        self._action_scale = 0.5

        self._target = torch.tensor([200, 0, 0], device=self._device).repeat(self._num_envs, 1)
        # Velocity command curriculum target ranges.
        self._command_cfg = {
            "lin_vel_x_range": (0.0, 1.0),
            "lin_vel_y_range": (-0.5, 0.5),
            "ang_vel_z_range": (-1.0, 1.0),
        }
        # Start from near-zero commands and gradually expand to full ranges.
        self._command_curriculum_steps = 200_000
        self._sim_step_count = 0
        self._vel_command = torch.zeros(self._num_envs, 3, device=self._device, dtype=torch.float)
        self._height_reward_scale = 10.0
        self._action_penalty = -0.002
        self._reward_tracking_sigma = 0.5
        self._feet_air_time_threshold = 0.4

        def find_link_indices(names: list[str]) -> list[int]:
            link_indices: list[int] = []
            for link in self._robot.links:
                if any(name in link.name for name in names):
                    link_indices.append(link.idx - self._robot.link_start)
            return link_indices

        # Link sets for foot-contact based terms and termination (substring match on link.name, same as Go2).
        self._feet_link_indices = find_link_indices(self._feet_link_names)
        self._last_contacts = torch.zeros(
            (self._num_envs, len(self._feet_link_indices)), device=self._device, dtype=torch.bool
        )
        self._feet_air_time = torch.zeros((self._num_envs, len(self._feet_link_indices)), device=self._device)

        # IsaacLab terminations.base_contact: illegal_contact on torso_link, threshold 1.0 (see velocity_env_cfg.TerminationsCfg + g1 rough_env_cfg).
        self._illegal_contact_force_threshold = 1.0
        self._torso_contact_link_indices = find_link_indices(self._torso_contact_link_names)

        def motor_indices(joint_names: list[str]) -> list[int]:
            return [self._motors_dof_idx[self._motor_joint_names.index(name)] for name in joint_names]

        def motor_q_indices(dof_idx_list: list[int]) -> list[int]:
            return [self._motor_dof_to_q[d] for d in dof_idx_list]

        # dof_idx lists are in local articulation dof indexing. q_idx lists are for motor_joints_* tensor columns.
        self._hip_knee_motor_dof_idx = motor_indices(self._hip_knee_joint_names)
        self._ankle_motor_dof_idx = motor_indices(self._ankle_joint_names)
        self._hip_dev_motor_dof_idx = motor_indices(self._hip_dev_joint_names)
        self._arms_dev_motor_dof_idx = motor_indices(self._arms_dev_joint_names)
        self._fingers_dev_motor_dof_idx = motor_indices(self._fingers_dev_joint_names)
        self._torso_dev_motor_dof_idx = motor_indices(self._torso_dev_joint_names)
        self._hip_knee_motor_q_idx = motor_q_indices(self._hip_knee_motor_dof_idx)
        self._ankle_motor_q_idx = motor_q_indices(self._ankle_motor_dof_idx)
        self._hip_dev_motor_q_idx = motor_q_indices(self._hip_dev_motor_dof_idx)
        self._arms_dev_motor_q_idx = motor_q_indices(self._arms_dev_motor_dof_idx)
        self._fingers_dev_motor_q_idx = motor_q_indices(self._fingers_dev_motor_dof_idx)
        self._torso_dev_motor_q_idx = motor_q_indices(self._torso_dev_motor_dof_idx)
        assert len(self._hip_knee_motor_dof_idx) > 0
        assert len(self._ankle_motor_dof_idx) > 0
        assert len(self._hip_dev_motor_dof_idx) > 0
        assert len(self._arms_dev_motor_dof_idx) > 0

        if self._vis_obs:
            # Initialize the sensors
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
            # NOTE: A dummy link for the camera to attach to, genesis sensor camera does not support fixed rotation or axis
            self._camera_mount = self._scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True))
            self._torso_link = self._robot.get_link("torso")
            self._camera = self._scene.add_sensor(
                gs.sensors.BatchRendererCameraOptions(
                    res=self._sensors_args["camera"]["res"],
                    pos=self._sensors_args["camera"]["pos"],
                    offset_T=offset_T,
                    fov=self._sensors_args["camera"]["fov"],
                    entity_idx=self._camera_mount.idx,
                    lights=[self._sensors_args["camera"]["lights"]],
                    env_idx=self._nominal_env_ids.cpu().tolist(),
                )
            )
            self._imgs_buf = torch.zeros(
                self.nominal_env_ids.shape[0],
                self._num_image_stack,
                self._sensors_args["camera"]["res"][0],
                self._sensors_args["camera"]["res"][1],
                3,
                device=self._device,
                dtype=torch.uint8,
            )

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(0.0, 1.0), n_envs_per_row=self._num_envs)

        dof_limits = torch.stack(self._robot.get_dofs_limit(self._all_dof_idx), dim=1)
        soft_limit = 0.9
        for i in range(dof_limits.shape[0]):
            m = (dof_limits[i, 0] + dof_limits[i, 1]) / 2.0
            r = dof_limits[i, 1] - dof_limits[i, 0]
            dof_limits[i, 0] = m - 0.5 * r * soft_limit
            dof_limits[i, 1] = m + 0.5 * r * soft_limit
        self._soft_dof_pos_limits = dof_limits
        dof_row = {dof: i for i, dof in enumerate(self._all_dof_idx)}
        self._motor_soft_pos_limits = torch.stack(
            [self._soft_dof_pos_limits[dof_row[dof], :] for dof in self._motors_dof_idx],
            dim=0,
        )
        self._nominal_motor_joints_pos = self._robot.get_dofs_position(self._motors_dof_idx).clone()
        self._default_motor_dof_pos = self._nominal_motor_joints_pos.clone()

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """
        +---------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (81,)) |
        +-----------+---------------------------------+-----------+
        |   Index   | Name                            |   Shape   |
        +-----------+---------------------------------+-----------+
        |     0     | base_lin_vel                    |    (3,)   |
        |     1     | base_ang_vel                    |    (3,)   |
        |     2     | projected_gravity               |    (3,)   |
        |     3     | velocity_commands               |    (3,)   |
        |     4     | joint_pos                       |   (23,)   |
        |     5     | joint_vel                       |   (23,)   |
        |     6     | actions                         |   (23,)   |
        +-----------+---------------------------------+-----------+
        """

        observations = {}
        # adapt from Jie Xu's implementation
        n_batch = states["progress_buf"].shape[0]
        robot_states = states["robot_states"]

        base_pose = robot_states["base_pose"]
        base_quat = base_pose[:, 3:]

        base_vel = robot_states["base_vel"]

        joints_pos = robot_states["motor_joints_pos"]
        joints_vel = robot_states["motor_joints_vel"]

        prev_actions = robot_states["prev_actions"]

        inv_base_quat = inv_quat(base_quat)
        projected_gravity = transform_by_quat(torch.tensor([0.0, 0., -1.0], device=self._device).repeat(n_batch, 1), inv_base_quat)
        
        vel_command = robot_states["vel_command"]
        privileged_observations = torch.cat(
            [
                base_vel,
                joints_pos,
                joints_vel,
                projected_gravity,
                vel_command,
                prev_actions,
            ],
            dim=-1,
        )
        observations["privileged_observations"] = privileged_observations

        if self._vis_obs:
            batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
            # NOTE: for AFRL agent, RGB observation and privileged observations may has different shapes
            # Reshape: (batch, num_stack, H, W, 3) -> (batch, num_stack * 3, H, W)
            observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                batch_size, num_stack * rgb, img_height, img_width
            )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:

        """
        +------------------------------------------+
        |           Active Reward Terms            |
        +-------+-------------------------+--------+
        | Index | Name                    | Weight |
        +-------+-------------------------+--------+
        |   0   | track_lin_vel_xy_exp    |    1.0 |
        |   1   | track_ang_vel_z_exp     |    1.0 |
        |   2   | lin_vel_z_l2            |   -0.2 |
        |   3   | ang_vel_xy_l2           |  -0.05 |
        |   4   | dof_torques_l2          | -2e-06 |
        |   5   | dof_acc_l2              | -1e-07 |
        |   6   | action_rate_l2          | -0.005 |
        |   7   | feet_air_time           |   0.75 |
        |   8   | flat_orientation_l2     |   -1.0 |
        |   9   | dof_pos_limits          |   -1.0 |
        |   10  | termination_penalty     | -200.0 |
        |   11  | feet_slide              |   -0.1 |
        |   12  | joint_deviation_hip     |   -0.1 |
        |   13  | joint_deviation_arms    |   -0.1 |
        |   14  | joint_deviation_fingers |  -0.05 |
        |   15  | joint_deviation_torso   |   -0.1 |
        +-------+-------------------------+--------+
        """
        
        robot_states = states["robot_states"]
        base_vel = robot_states["base_vel"]
        cmd = robot_states["vel_command"]
        motor_joints_vel = robot_states["motor_joints_vel"]
        motor_joints_pos = robot_states["motor_joints_pos"]
        base_pose = robot_states["base_pose"]
        prev_actions = robot_states["prev_actions"]

        # 0) track_lin_vel_xy_exp
        lin_vel_error = torch.sum(torch.square(cmd[:, :2] - base_vel[:, :2]), dim=1)
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / (self._reward_tracking_sigma**2))
        # 1) track_ang_vel_z_exp
        ang_vel_error = torch.square(cmd[:, 2] - base_vel[:, 5])
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / (self._reward_tracking_sigma**2))
        # 2) lin_vel_z_l2
        lin_vel_z_l2 = torch.square(base_vel[:, 2])
        # 3) ang_vel_xy_l2
        ang_vel_xy_l2 = torch.sum(torch.square(base_vel[:, 3:5]), dim=1)
        # 4) dof_torques_l2 (hip + knee joints)
        if len(self._hip_knee_motor_dof_idx) > 0:
            dof_torques = self._robot.get_dofs_control_force(self._hip_knee_motor_dof_idx)
            dof_torques_l2 = torch.sum(torch.square(dof_torques), dim=1)
        else:
            dof_torques_l2 = torch.zeros_like(track_lin_vel_xy_exp)
        # 5) dof_acc_l2 (hip + knee joints)
        dof_acc = (motor_joints_vel - self._last_motor_joints_vel) / max(self._scene.sim.dt, 1e-6)
        dof_acc_l2 = torch.sum(torch.square(dof_acc), dim=1)

        # 6) action_rate_l2
        action_rate_l2 = torch.sum(torch.square(self._current_actions - prev_actions), dim=1)

        # 7) feet_air_time (biped-style positive reward)
        if len(self._feet_link_indices) > 0:
            link_contact_forces = torch.tensor(self._robot.get_links_net_contact_force(), device=self._device, dtype=torch.float)
            contact = torch.norm(link_contact_forces[:, self._feet_link_indices, :], dim=-1) > 1.0
            contact_filt = torch.logical_or(contact, self._last_contacts)
            first_contact = (self._feet_air_time > 0.0) * contact_filt
            self._feet_air_time += self._scene.sim.dt
            feet_air_time = torch.sum((self._feet_air_time - self._feet_air_time_threshold) * first_contact, dim=1)
            feet_air_time *= (torch.norm(cmd[:, :2], dim=1) > 0.1)
            self._feet_air_time *= ~contact_filt
            self._last_contacts = contact
        else:
            contact = torch.zeros((base_pose.shape[0], 0), device=self._device, dtype=torch.bool)
            feet_air_time = torch.zeros_like(track_lin_vel_xy_exp)

        # 8) flat_orientation_l2
        base_quat = base_pose[:, 3:]
        inv_base_quat = inv_quat(base_quat)
        projected_gravity = transform_by_quat(
            torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(base_pose.shape[0], 1), inv_base_quat
        )
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # 9) dof_pos_limits (ankle joints)
        if len(self._ankle_motor_q_idx) > 0:
            ankle_lo = self._motor_soft_pos_limits[self._ankle_motor_q_idx, 0]
            ankle_hi = self._motor_soft_pos_limits[self._ankle_motor_q_idx, 1]
            ankle_pos = motor_joints_pos[:, self._ankle_motor_q_idx]
            out_of_limits = -(ankle_pos - ankle_lo).clip(max=0.0)
            out_of_limits += (ankle_pos - ankle_hi).clip(min=0.0)
            dof_pos_limits = torch.sum(out_of_limits, dim=1)
        else:
            dof_pos_limits = torch.zeros_like(track_lin_vel_xy_exp)

        # 10) termination_penalty
        terminated = self.compute_termination(states).float()
        termination_penalty = terminated

        # 11) feet_slide
        if len(self._feet_link_indices) > 0:
            feet_vel = self._robot.get_links_vel()[:, self._feet_link_indices, :2]
            feet_slide = torch.sum(torch.norm(feet_vel, dim=-1) * contact, dim=1)
        else:
            feet_slide = torch.zeros_like(track_lin_vel_xy_exp)

        # 12-15) joint_deviation_* from nominal motor pose (indices into motor_joints_pos).
        joint_deviation_hip = (
            torch.sum(
                torch.abs(
                    motor_joints_pos[:, self._hip_dev_motor_q_idx]
                    - self._nominal_motor_joints_pos[:, self._hip_dev_motor_q_idx]
                ),
                dim=1,
            )
            if len(self._hip_dev_motor_q_idx) > 0
            else torch.zeros_like(track_lin_vel_xy_exp)
        )
        joint_deviation_arms = (
            torch.sum(
                torch.abs(
                    motor_joints_pos[:, self._arms_dev_motor_q_idx]
                    - self._nominal_motor_joints_pos[:, self._arms_dev_motor_q_idx]
                ),
                dim=1,
            )
            if len(self._arms_dev_motor_q_idx) > 0
            else torch.zeros_like(track_lin_vel_xy_exp)
        )
        joint_deviation_fingers = (
            torch.sum(
                torch.abs(
                    motor_joints_pos[:, self._fingers_dev_motor_q_idx]
                    - self._nominal_motor_joints_pos[:, self._fingers_dev_motor_q_idx]
                ),
                dim=1,
            )
            if len(self._fingers_dev_motor_q_idx) > 0
            else torch.zeros_like(track_lin_vel_xy_exp)
        )
        joint_deviation_torso = (
            torch.sum(
                torch.abs(
                    motor_joints_pos[:, self._torso_dev_motor_q_idx]
                    - self._nominal_motor_joints_pos[:, self._torso_dev_motor_q_idx]
                ),
                dim=1,
            )
            if len(self._torso_dev_motor_q_idx) > 0
            else torch.zeros_like(track_lin_vel_xy_exp)
        )

        reward = (
            1.0 * track_lin_vel_xy_exp
            + 1.0 * track_ang_vel_z_exp
            - 0.2 * lin_vel_z_l2
            - 0.05 * ang_vel_xy_l2
            - 2e-06 * dof_torques_l2
            - 1e-07 * dof_acc_l2
            - 0.005 * action_rate_l2
            + 0.75 * feet_air_time
            - 1.0 * flat_orientation_l2
            - 1.0 * dof_pos_limits
            - 200.0 * termination_penalty
            - 0.1 * feet_slide
            - 0.1 * joint_deviation_hip
            - 0.1 * joint_deviation_arms
            - 0.05 * joint_deviation_fingers
            - 0.1 * joint_deviation_torso
        )

        self._last_motor_joints_vel = motor_joints_vel.clone()
        self._prev_actions = self._current_actions.clone()

        self._infos["track_lin_vel_xy_exp"] = 1.0 * track_lin_vel_xy_exp
        self._infos["track_ang_vel_z_exp"] = 1.0 * track_ang_vel_z_exp
        self._infos["lin_vel_z_l2"] = -0.2 * lin_vel_z_l2
        self._infos["ang_vel_xy_l2"] = -0.05 * ang_vel_xy_l2
        self._infos["dof_torques_l2"] = -2e-06 * dof_torques_l2
        self._infos["dof_acc_l2"] = -1e-07 * dof_acc_l2
        self._infos["action_rate_l2"] = -0.005 * action_rate_l2
        self._infos["feet_air_time"] = 0.75 * feet_air_time
        self._infos["flat_orientation_l2"] = -1.0 * flat_orientation_l2
        self._infos["dof_pos_limits"] = -1.0 * dof_pos_limits
        self._infos["termination_penalty"] = -200.0 * termination_penalty
        self._infos["feet_slide"] = -0.1 * feet_slide
        self._infos["joint_deviation_hip"] = -0.1 * joint_deviation_hip
        self._infos["joint_deviation_arms"] = -0.1 * joint_deviation_arms
        self._infos["joint_deviation_fingers"] = -0.05 * joint_deviation_fingers
        self._infos["joint_deviation_torso"] = -0.1 * joint_deviation_torso
        return reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        # IsaacLab G1 flat: base_contact = illegal_contact on torso_link, threshold 1.0 (no height-based term).
        _ = states
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not self._early_termination or not self._torso_contact_link_indices:
            return termination
        link_contact_forces = torch.tensor(
            self._robot.get_links_net_contact_force(), device=self._device, dtype=torch.float
        )
        torso_force_norm = torch.norm(
            link_contact_forces[:, self._torso_contact_link_indices, :], dim=-1
        ).max(dim=1).values
        termination = torso_force_norm > self._illegal_contact_force_threshold
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        base_pos = self._default_base_pos[env_ids]
        base_quat = self._default_base_quat[env_ids]
        motor_dof_pos = self._default_motor_dof_pos[env_ids]

        if self._randomize_init:
            base_pos = base_pos + (torch.rand_like(base_pos) - 0.5) * 0.1
            angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
            axis = torch.nn.functional.normalize(torch.rand(len(env_ids), 3, device=self.device) - 0.5)
            base_quat = transform_quat_by_quat(base_quat, axis_angle_to_quat(angle, axis))
            motor_dof_pos = motor_dof_pos + (torch.rand_like(motor_dof_pos) - 0.5) * 0.1

        self._robot.set_pos(base_pos, envs_idx=env_ids, zero_velocity=True)
        self._robot.set_quat(base_quat, envs_idx=env_ids, zero_velocity=True)
        self._robot.set_dofs_position(
            position=motor_dof_pos,
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )

        self._prev_actions[env_ids] = torch.zeros(len(env_ids), self._num_actions, device=self._device)
        self._current_actions[env_ids] = torch.zeros(len(env_ids), self._num_actions, device=self._device)
        self._last_motor_joints_vel[env_ids] = torch.zeros(len(env_ids), len(self._motors_dof_idx), device=self._device)
        self._last_contacts[env_ids] = torch.zeros(
            len(env_ids), len(self._feet_link_indices), device=self._device, dtype=torch.bool
        )
        self._feet_air_time[env_ids] = torch.zeros(len(env_ids), len(self._feet_link_indices), device=self._device)
        self._resample_vel_commands(env_ids)

        if self._vis_obs:
            # Find which nominal environments are being reset
            # self.nominal_env_ids contains the global env_ids of nominal environments
            # We need to find the indices within nominal_env_ids that match env_ids
            mask = torch.isin(self.nominal_env_ids, env_ids)
            nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]

            if len(nominal_idx_to_reset) > 0:
                # Render fresh images for the reset nominal environments
                reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
                new_img = self.render(env_ids=reset_nominal_env_ids)

                # Initialize the image buffer for these environments
                self._imgs_buf[nominal_idx_to_reset] = new_img.unsqueeze(1)

    def _set_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(self._num_envs, self._num_actions)
        actions = actions.clamp(min=-1.0, max=1.0)
        target_dof_pos = actions * self._action_scale + self._default_motor_dof_pos
        self._robot.control_dofs_position(target_dof_pos, self._motors_dof_idx)
        self._current_actions = actions.clone()

    def _current_command_ranges(self):
        # Linear ramp [0, 1] over command_curriculum_steps.
        progress = min(1.0, self._sim_step_count / float(self._command_curriculum_steps))
        x_lo, x_hi = self._command_cfg["lin_vel_x_range"]
        y_lo, y_hi = self._command_cfg["lin_vel_y_range"]
        z_lo, z_hi = self._command_cfg["ang_vel_z_range"]
        return (
            (x_lo * progress, x_hi * progress),
            (y_lo * progress, y_hi * progress),
            (z_lo * progress, z_hi * progress),
        )

    def _resample_vel_commands(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        (x_range, y_range, z_range) = self._current_command_ranges()
        self._vel_command[env_ids, 0] = (x_range[1] - x_range[0]) * torch.rand(
            len(env_ids), device=self._device
        ) + x_range[0]
        self._vel_command[env_ids, 1] = (y_range[1] - y_range[0]) * torch.rand(
            len(env_ids), device=self._device
        ) + y_range[0]
        self._vel_command[env_ids, 2] = (z_range[1] - z_range[0]) * torch.rand(
            len(env_ids), device=self._device
        ) + z_range[0]

    def _post_physics_step(self) -> None:
        """Update image buffer by rolling frames and appending new image."""
        self._sim_step_count += self._num_envs
        # Resample commands a few times per episode, following the Go2 pattern.
        env_ids = (self._progress_buf % int(self._episode_length / 5) == 0).nonzero(as_tuple=False).flatten()
        self._resample_vel_commands(env_ids)

        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            # Roll the buffer to shift old frames: [t-2, t-1, t-0] -> [t-1, t-0, None]
            # This moves older frames "to the left" and makes room for the new frame
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img

    def render(self, env_ids: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        if self._vis_obs:
            if env_ids is None:
                env_ids = self.nominal_env_ids
            # Attach the camera to the torso pose
            pos = self._torso_link.get_pos()
            self._camera_mount.set_pos(pos)

            # TODO: genesis will refresh the image when the scene._dt is different from the last render time
            # TODO: temporarily we hack by setting the last render time to 0 to force render the new image
            self._camera._shared_metadata.last_render_timestep = 0
            data = self._camera.read(envs_idx=env_ids)
            return data.rgb
        else:
            return None

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        base_pos = self._robot.get_pos(envs_idx=env_ids)
        base_quat = self._robot.get_quat(envs_idx=env_ids)
        base_pose = torch.cat([base_pos, base_quat], dim=-1)
        # NOTE: the angular velocity of the base is in the body frame
        base_vel = self._robot.get_dofs_velocity(self._base_dof_idx, envs_idx=env_ids)
        motor_joints_pos = self._robot.get_dofs_position(self._motors_dof_idx, envs_idx=env_ids)
        motor_joints_vel = self._robot.get_dofs_velocity(self._motors_dof_idx, envs_idx=env_ids)

        robot_states = {
            "base_pose": base_pose.clone(),
            "base_vel": base_vel.clone(),
            "motor_joints_pos": motor_joints_pos.clone(),
            "motor_joints_vel": motor_joints_vel.clone(),
            "vel_command": self._vel_command[env_ids].clone(),
            "prev_actions": self._prev_actions[env_ids].clone(),
        }

        # TODO: shall we treat the image buffer as part of the robot states?

        states = {
            "robot_states": robot_states,
            "progress_buf": self._progress_buf[env_ids].clone(),
        }

        return states

    def set_states(self, states: Dict[str, Any], env_ids: Optional[Sequence[int]] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        robot_states = states["robot_states"]

        self._robot.set_pos(robot_states["base_pose"][:, :3], envs_idx=env_ids)
        self._robot.set_quat(robot_states["base_pose"][:, 3:], envs_idx=env_ids)

        self._robot.set_dofs_position(
            position=robot_states["motor_joints_pos"],
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=False,
        )

        self._robot.set_dofs_velocity(
            velocity=torch.cat([robot_states["base_vel"], robot_states["motor_joints_vel"]], dim=-1),
            dofs_idx_local=self._base_dof_idx + self._motors_dof_idx,
            envs_idx=env_ids,
        )

        self._prev_actions[env_ids] = robot_states["prev_actions"].clone()
        self._vel_command[env_ids] = robot_states["vel_command"].clone()

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
