import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import (
    axis_angle_to_quat,
    inv_quat,
    pos_lookat_up_to_T,
    transform_quat_by_quat,
)
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class ShadowHand(GenesisEnv):
    """Shadow Hand in-hand cube reorientation environment.

    Loads the Shadow Hand URDF from Genesis's asset bundle and a cube mesh
    from the local assets. Reward / termination / observation layout follows
    IsaacLab's ShadowHandEnvCfg (full obs).
    """

    # 24 = 1 forearm + 1 wrist + 5 thumb + 4 (index/middle/ring) * 3 + 5 little.
    # The Genesis URDF doesn't model the FFJ0/FFJ1 tendon coupling that
    # IsaacLab uses, so we actuate every revolute joint directly.
    _num_actions = 24
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(_num_actions,))

    _HAND_MOTOR_JOINT_NAMES = [
        "forearm_joint",
        "wrist_joint",
        "thumb_joint1",
        "thumb_joint2",
        "thumb_joint3",
        "thumb_joint4",
        "thumb_joint5",
        "index_finger_joint1",
        "index_finger_joint2",
        "index_finger_joint3",
        "index_finger_joint4",
        "middle_finger_joint1",
        "middle_finger_joint2",
        "middle_finger_joint3",
        "middle_finger_joint4",
        "ring_finger_joint1",
        "ring_finger_joint2",
        "ring_finger_joint3",
        "ring_finger_joint4",
        "little_finger_joint1",
        "little_finger_joint2",
        "little_finger_joint3",
        "little_finger_joint4",
        "little_finger_joint5",
    ]

    _FINGER_TIP_LINK_NAMES = [
        "thumb_distal",
        "index_finger_distal",
        "middle_finger_distal",
        "ring_finger_distal",
        "little_finger_distal",
    ]

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
        rigid_options: gs.options.RigidOptions | None = None,
        vis_options: gs.options.VisOptions | None = None,
        show_viewer: bool = False,
        show_FPS: bool = False,
        init_goal_rotation: Optional[Dict[str, Any]] = None,
    ) -> None:
        dt = sim_options.dt
        # Match IsaacLab ShadowHandEnvCfg.episode_length_s = 10.0.
        episode_length = int(10.0 / dt)

        early_termination = True

        self._vis_obs = vis_obs

        if sensors_args is None:
            sensors_args = {
                "camera": {
                    "res": [256, 256],
                    "pos": [0.40, 0.05, 0.55],
                    "lookat": [0.0, 0.1, 0.40],
                    "fov": 80.0,
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

        n_finger = len(self._FINGER_TIP_LINK_NAMES)
        # 24 dof_pos + 24 dof_vel + 7 cube pose + 6 cube vel
        # + 3 in_hand_pos + 4 target_quat + 4 rot_diff
        # + n_finger * (3 pos + 4 quat + 3 lin_vel + 3 ang_vel) + 24 prev_actions
        priv_obs_dim = (
            self._num_actions * 2
            + 7 + 6
            + 3 + 4 + 4
            + n_finger * (3 + 4 + 3 + 3)
            + self._num_actions
        )
        # Drops cube state — cube info is recovered from images for vis_obs actor.
        proprio_obs_dim = (
            self._num_actions * 2
            + 3 + 4
            + n_finger * (3 + 4 + 3 + 3)
            + self._num_actions
        )

        if vis_obs:
            self._num_image_stack = 3
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(priv_obs_dim,)),
                    "proprioception_and_target": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(proprio_obs_dim,)
                    ),
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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(priv_obs_dim,)),
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
            rigid_options=rigid_options,
            vis_options=vis_options,
            show_FPS=show_FPS,
        )

        # Goal-orientation sampling. When enabled, every episode targets a single
        # achievable rotation (one world axis, bounded angle) that stays fixed for
        # the episode -- this keeps the reward a smooth function of the trajectory,
        # which AFRL's first-order gradient estimate relies on. When disabled, the
        # env falls back to the IsaacLab-style full-SO(3) random target.
        igr = init_goal_rotation or {}
        self._init_goal_rotation_enabled = bool(igr.get("enable", True))
        self._init_goal_angle_min_deg = float(igr.get("angle_deg_min", 60.0))
        self._init_goal_angle_max_deg = float(igr.get("angle_deg_max", 90.0))

    def init_scene(self) -> None:
        """Initialize the scene."""

        # The URDF's natural pose has the forearm pointing along +z with the
        # palm facing the -y direction. Rotate -90 deg around X so that the
        # palm-normal points +z (up) and fingers extend along +y. The forearm
        # base is offset to -y so the rotated palm sits at world (0, 0, ~0.06).
        # If the palm appears to face down on screen, flip the sign of the X
        # quaternion component (i.e. rotate +90 deg instead).
        _SQRT2_2 = 2 ** 0.5 / 2
        # The hand is mounted high enough that a cube falling out of the
        # palm reaches the `fall_distance` (0.24) termination threshold
        # well before it hits the floor (the in_hand target sits at z=0.40
        # so the cube terminates around z=0.16).
        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file="urdf/shadow_hand/shadow_hand.urdf",
                fixed=True,
                pos=(0.0, -0.247, 0.35),
                quat=(_SQRT2_2, -_SQRT2_2, 0.0, 0.0),
            ),
        )

        self._plane = self._scene.add_entity(gs.morphs.Plane())

        # After the rotation+offset above, the palm center sits at world
        # (0, 0.1, ~0.36). Drop the cube directly above the palm so it
        # lands on the cupped fingers.
        self._in_hand_pos = torch.tensor([0.0, 0.1, 0.40], device=self._device).repeat(self._num_envs, 1)

        self._cube = self._scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/meshes/cube.obj"),
                scale=0.03,
                pos=(0.0, 0.1, 0.40),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/textures/cube.png")
                )
            ),
            material=gs.materials.Rigid(friction=1.0, rho=567.0),
        )

        # Visual-only goal cube floating beside the hand. The target quat is
        # resampled at every episode reset and every time the cube reaches
        # `success_tolerance` (matches IsaacLab's _reset_target_pose).
        self._target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)
        self._x_unit = torch.tensor([1.0, 0.0, 0.0], device=self._device)
        self._y_unit = torch.tensor([0.0, 1.0, 0.0], device=self._device)
        self._target = self._scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/meshes/cube.obj"),
                scale=0.03,
                collision=False,
                pos=(0.0, -0.3, 0.55),
            ),
            material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/textures/cube.png")
                )
            ),
        )

        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)

        self._hand_motor_joint_names = list(self._HAND_MOTOR_JOINT_NAMES)
        self._hand_motors_dof_idx: list[int] = []
        for name in self._hand_motor_joint_names:
            self._hand_motors_dof_idx.extend(self._robot.get_joint(name).dofs_idx_local)

        self._cube_dof_idx = self._cube.base_joint.dofs_idx_local

        self._finger_tip_link_names = list(self._FINGER_TIP_LINK_NAMES)
        self._finger_tip_link_idx = [
            self._robot.get_link(name).idx_local for name in self._finger_tip_link_names
        ]

        # Default cube/goal poses.
        self._default_cube_pos = torch.tensor([0.0, 0.1, 0.40], device=self._device).repeat(self._num_envs, 1)
        self._default_cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)
        # Neutral hand pose: zero across all 24 joints.
        self._default_hand_dof_pos = torch.zeros(
            self._num_envs, self._num_actions, device=self._device
        )

        # Reward / termination scales. Smooth quadratic-orientation preset
        # (mirrors AllegroHand) so the return is differentiable for AFRL:
        #   reward = -dist - rot_dist^2 - action_penalty + healthy
        # No success bonus / target resampling -- those discontinuities bias
        # first-order gradients and let the policy settle on just holding the cube.
        self._vel_obs_scale = 0.2
        self._fall_distance = 0.24
        self._dist_reward_scale = -10.0
        self._rot_reward_scale = 1.0
        self._action_penalty_scale = -0.0002
        self._healthy_reward = 3.0

        if self._vis_obs:
            offset_T = self._sensors_args["camera"].get("offset_T", None)
            lookat = self._sensors_args["camera"].get("lookat", None)
            if offset_T is not None:
                offset_T = torch.tensor(offset_T, device=self._device)
            else:
                if lookat is not None:
                    offset_T = pos_lookat_up_to_T(
                        np.array(self._sensors_args["camera"]["pos"]),
                        np.array(lookat),
                        np.array((0.0, 0.0, 1.0)),
                    )
                else:
                    offset_T = np.eye(4)
            self._camera_mount = self._scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True))
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

    def _sample_target_quat(self, n: int) -> torch.Tensor:
        """Random quat = rot(rand0*pi, x) * rot(rand1*pi, y), matching IsaacLab."""
        rand = (torch.rand(n, 2, device=self._device) * 2.0 - 1.0) * np.pi
        x_axis = self._x_unit.unsqueeze(0).expand(n, -1)
        y_axis = self._y_unit.unsqueeze(0).expand(n, -1)
        qx = axis_angle_to_quat(rand[:, 0], x_axis)
        qy = axis_angle_to_quat(rand[:, 1], y_axis)
        # transform_quat_by_quat(v, u) == quat_mul(u, v), so this is quat_mul(qx, qy).
        return transform_quat_by_quat(qy, qx)

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(1.0 / self._num_envs, 1.0/self._num_envs))
        self._hand_motors_ctrl_lower, self._hand_motors_ctrl_upper = self._robot.get_dofs_limit(
            dofs_idx_local=self._hand_motors_dof_idx
        )

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}

        robot_states = states["robot_states"]
        hand_dof_pos = robot_states["hand_dof_pos"]
        scaled_hand_dof_pos = (2.0 * hand_dof_pos - self._hand_motors_ctrl_lower - self._hand_motors_ctrl_upper) / (
            self._hand_motors_ctrl_upper - self._hand_motors_ctrl_lower
        )
        hand_dof_vel = robot_states["hand_dof_vel"]
        scaled_hand_dof_vel = hand_dof_vel * self._vel_obs_scale

        cube_pos = robot_states["cube_pos"]
        cube_quat = robot_states["cube_quat"]
        cube_vel = robot_states["cube_vel"]
        cube_linear_vel = cube_vel[:, :3]
        cube_angular_vel = cube_vel[:, 3:]
        scaled_cube_angular_vel = cube_angular_vel * self._vel_obs_scale

        target_quat = robot_states["target_quat"]
        rot_diff = transform_quat_by_quat(inv_quat(target_quat), cube_quat)

        prev_actions = robot_states["prev_actions"]

        finger_tip_pos = self._robot.get_links_pos(links_idx_local=self._finger_tip_link_idx).view(
            self.num_envs, len(self._finger_tip_link_names) * 3
        )
        finger_tip_quat = self._robot.get_links_quat(links_idx_local=self._finger_tip_link_idx).view(
            self.num_envs, len(self._finger_tip_link_names) * 4
        )
        finger_tip_vel = self._robot.get_links_vel(links_idx_local=self._finger_tip_link_idx).view(
            self.num_envs, len(self._finger_tip_link_names) * 3
        )
        finger_tip_angular_vel = self._robot.get_links_ang(links_idx_local=self._finger_tip_link_idx).view(
            self.num_envs, len(self._finger_tip_link_names) * 3
        )

        privileged_observations = torch.cat(
            [
                scaled_hand_dof_pos,
                scaled_hand_dof_vel,
                cube_pos,
                cube_quat,
                cube_linear_vel,
                scaled_cube_angular_vel,
                self._in_hand_pos,
                target_quat,
                rot_diff,
                finger_tip_pos,
                finger_tip_quat,
                finger_tip_vel,
                finger_tip_angular_vel,
                prev_actions,
            ],
            dim=-1,
        )
        observations["privileged_observations"] = privileged_observations

        if self._vis_obs:
            proprioception_and_target = torch.cat(
                [
                    scaled_hand_dof_pos,
                    scaled_hand_dof_vel,
                    self._in_hand_pos,
                    target_quat,
                    finger_tip_pos,
                    finger_tip_quat,
                    finger_tip_vel,
                    finger_tip_angular_vel,
                    prev_actions,
                ],
                dim=-1,
            )

            observations["proprioception_and_target"] = proprioception_and_target

            batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
            observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                batch_size, num_stack * rgb, img_height, img_width
            )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        robot_states = states["robot_states"]
        cube_pos = robot_states["cube_pos"]
        cube_quat = robot_states["cube_quat"]
        target_quat = robot_states["target_quat"]

        # Position term: penalize distance of cube to in-hand target.
        goal_dist = torch.norm(cube_pos - self._in_hand_pos, p=2, dim=-1)
        dist_rew = self._dist_reward_scale * goal_dist

        # Orientation term: smooth quadratic penalty on the rotation distance,
        # giving a clean gradient toward the goal everywhere (unlike 1/(d+eps),
        # which is nearly flat far from the goal).
        quat_diff = transform_quat_by_quat(inv_quat(target_quat), cube_quat)
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
        rot_rew = -(rot_dist ** 2) * self._rot_reward_scale

        # Action penalty.
        action_penalty = self._action_penalty_scale * torch.sum(actions ** 2, dim=-1)

        reward = dist_rew + rot_rew + action_penalty + self._healthy_reward

        self._infos["angle_diff"] = torch.rad2deg(torch.abs(rot_dist)).mean().item()
        self._infos["goal_dist"] = goal_dist.mean().item()

        return reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            termination = (
                torch.norm(robot_states["cube_pos"] - self._in_hand_pos, p=2, dim=-1) > self._fall_distance
            )
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        hand_dof_pos = self._default_hand_dof_pos[env_ids]
        cube_pos = self._default_cube_pos[env_ids]
        cube_quat = self._default_cube_quat[env_ids]
        target_quat = self._sample_target_quat(len(env_ids))

        if self._randomize_init:
            cube_pos = cube_pos + (torch.rand_like(cube_pos) - 0.5) * 0.02

            if self._init_goal_rotation_enabled:
                # One world-axis rotation per env: pitch (+Y) or roll (+X), angle
                # uniform in [min, max] degrees. The cube starts at its default
                # orientation, so this is a single achievable, smooth reorientation.
                n = env_ids.shape[0]
                dev = self.device
                span = max(self._init_goal_angle_max_deg - self._init_goal_angle_min_deg, 0.0)
                angles_deg = self._init_goal_angle_min_deg + torch.rand(n, device=dev) * span
                angles = torch.deg2rad(angles_deg)
                pick_pitch = torch.rand(n, device=dev) >= 0.5
                x_axis = self._x_unit.unsqueeze(0).expand(n, 3)
                y_axis = self._y_unit.unsqueeze(0).expand(n, 3)
                axes = torch.where(pick_pitch.unsqueeze(-1), y_axis, x_axis)
                target_quat = axis_angle_to_quat(angles, axes)
                self._infos["init_goal_angle_deg_mean"] = float(angles_deg.mean().item())
                self._infos["init_goal_pitch_frac"] = float(pick_pitch.float().mean().item())

            ctrl_range = self._hand_motors_ctrl_upper - self._hand_motors_ctrl_lower
            hand_dof_pos = hand_dof_pos + (torch.rand_like(hand_dof_pos) - 0.5) * ctrl_range * 0.2
            hand_dof_pos = torch.clamp(
                hand_dof_pos,
                self._hand_motors_ctrl_lower,
                self._hand_motors_ctrl_upper,
            )

        prev_actions = (hand_dof_pos - self._hand_motors_ctrl_lower) / (
            self._hand_motors_ctrl_upper - self._hand_motors_ctrl_lower
        ) * 2 - 1.0

        self._robot.set_dofs_position(
            position=hand_dof_pos,
            dofs_idx_local=self._hand_motors_dof_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self._robot.control_dofs_position(
            position=hand_dof_pos,
            dofs_idx_local=self._hand_motors_dof_idx,
            envs_idx=env_ids,
        )

        self._cube.set_pos(cube_pos, envs_idx=env_ids, zero_velocity=True)
        self._cube.set_quat(cube_quat, envs_idx=env_ids, zero_velocity=True)
        self._target_quat[env_ids] = target_quat
        self._target.set_quat(target_quat, envs_idx=env_ids, zero_velocity=True)

        self._prev_actions[env_ids] = prev_actions

        if self._vis_obs:
            mask = torch.isin(self.nominal_env_ids, env_ids)
            nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]
            if len(nominal_idx_to_reset) > 0:
                reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
                new_img = self.render(env_ids=reset_nominal_env_ids)
                self._imgs_buf[nominal_idx_to_reset] = new_img.unsqueeze(1)

    def _set_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(self._num_envs, self._num_actions)
        actions = actions.clamp(min=-1.0, max=1.0)
        self._prev_actions = actions.clone()
        target_pos = self._hand_motors_ctrl_lower + (actions + 1.0) * 0.5 * (
            self._hand_motors_ctrl_upper - self._hand_motors_ctrl_lower
        )
        self._robot.control_dofs_position(
            target_pos,
            dofs_idx_local=self._hand_motors_dof_idx,
        )

    def _post_physics_step(self) -> None:
        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img

    def render(self, env_ids: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        if self._vis_obs:
            if env_ids is None:
                env_ids = self.nominal_env_ids
            self._camera._shared_metadata.last_render_timestep = 0
            data = self._camera.read(envs_idx=env_ids)
            return data.rgb
        else:
            return None

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        hand_dof_pos = self._robot.get_dofs_position(self._hand_motors_dof_idx, envs_idx=env_ids)
        hand_dof_vel = self._robot.get_dofs_velocity(self._hand_motors_dof_idx, envs_idx=env_ids)
        cube_pos = self._cube.get_pos(envs_idx=env_ids)
        cube_quat = self._cube.get_quat(envs_idx=env_ids)
        cube_vel = self._cube.get_dofs_velocity(self._cube_dof_idx, envs_idx=env_ids)

        robot_states = {
            "hand_dof_pos": hand_dof_pos.clone(),
            "hand_dof_vel": hand_dof_vel.clone(),
            "cube_pos": cube_pos.clone(),
            "cube_quat": cube_quat.clone(),
            "cube_vel": cube_vel.clone(),
            "target_quat": self._target_quat[env_ids].clone(),
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

        self._robot.set_dofs_position(
            position=robot_states["hand_dof_pos"],
            dofs_idx_local=self._hand_motors_dof_idx,
            envs_idx=env_ids,
        )

        self._robot.set_dofs_velocity(
            velocity=robot_states["hand_dof_vel"],
            dofs_idx_local=self._hand_motors_dof_idx,
            envs_idx=env_ids,
        )

        self._cube.set_pos(robot_states["cube_pos"], envs_idx=env_ids)
        self._cube.set_quat(robot_states["cube_quat"], envs_idx=env_ids)
        self._cube.set_dofs_velocity(robot_states["cube_vel"], envs_idx=env_ids, dofs_idx_local=self._cube_dof_idx)

        self._target_quat[env_ids] = robot_states["target_quat"]
        self._target.set_quat(robot_states["target_quat"], envs_idx=env_ids)

        self._prev_actions[env_ids] = robot_states["prev_actions"].clone()

        self._progress_buf[env_ids] = states["progress_buf"].clone()
