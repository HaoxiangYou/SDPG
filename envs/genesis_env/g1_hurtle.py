import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import axis_angle_to_quat, pos_lookat_up_to_T, transform_by_quat, transform_quat_by_quat, inv_quat
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv
from utils.terrain import Terrain


def _quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by yaw component only of quaternion (Genesis w,x,y,z convention)."""
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0  # zero x, y to keep only yaw (w, z)
    quat_yaw = quat_yaw / quat_yaw.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return transform_by_quat(vec, quat_yaw)


class G1Hurtle(GenesisEnv):
    """G1 hurtle environment."""

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
        terrain_args: Dict[str, Any] | None = None,
        sim_options: gs.options.SimOptions | None = None,
        viewer_options: gs.options.ViewerOptions | None = None,
        vis_options: gs.options.VisOptions | None = None,
        debug: bool = False,
        show_viewer: bool = False,
        show_FPS: bool = False,
    ) -> None:
        episode_length = 1000
        early_termination = True

        self._debug = debug
        self._vis_obs = vis_obs
        self._terrain_args = terrain_args

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
            surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0, 1.0)),
        )

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

        self._base_dof_idx = self._robot.base_joint.dofs_idx_local
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]

        self._motor_strength = torch.tensor(
            [
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                88.0,
                25.0, 25.0, 25.0, 25.0, 25.0,
                25.0, 25.0, 25.0, 25.0, 25.0,
            ],
            device=self._device,
        )

        self._default_joint_angles = {
            "left_hip_pitch_joint": -0.312,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.669,
            "left_ankle_pitch_joint": -0.363,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.312,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": -0.669,
            "right_ankle_pitch_joint": 0.363,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.6,
            "right_wrist_roll_joint": 0.0,
        }

        self._termination_height_lower_bound = 0.6
        self._termination_height_upper_bound = 1.0
        self._default_base_pos = torch.tensor([0.2, 0, 0.8], device=self._device).repeat(self._num_envs, 1)
        self._default_base_quat = torch.tensor([1, 0, 0, 0], device=self._device).repeat(self._num_envs, 1)
        self._default_joint_angles = torch.tensor(
            [self._default_joint_angles[name] for name in self._motor_joint_names],
            device=self._device,
        )
        self._default_motor_dof_pos = self._default_joint_angles.repeat(self._num_envs, 1)

        self._target = torch.tensor([200.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)
        self._height_reward_scale = 10.0
        self._termination_height_tolerance = 0.1
        self._up_reward_scale = 0.1
        self._action_penalty = -0.002

        # proprioceptive: base_height(1) + projected_gravity(3) + base_vel(6) + motor_pos(23) + motor_vel(23) = 56
        proprioceptive_observations_dim = 56
        privileged_observations_dim = proprioceptive_observations_dim

        observation_space_dict = {
            "proprioceptive_observations": spaces.Box(
                low=-np.inf, high=np.inf, shape=(proprioceptive_observations_dim,)
            ),
        }

        self._env_spacing = 1.0

        if self._terrain_args is not None:
            self._create_terrain()
            self._terrain_y_half_width = self._terrain_args["terrain_width"] / 2.0
            self._env_spacing += self._terrain_y_half_width * 2.0
            if self._sensors_args is not None and "heightfield" in self._sensors_args:
                self._init_height_points()
                self._measured_heights = torch.zeros(
                    self._num_envs,
                    self._num_height_points,
                    device=self._device,
                    dtype=torch.float,
                )
                observation_space_dict["height_field"] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._num_height_points,)
                )
                privileged_observations_dim += self._num_height_points
        else:
            self._plane = self._scene.add_entity(gs.morphs.Plane())

        observation_space_dict["privileged_observations"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(privileged_observations_dim,)
        )

        if self._vis_obs:
            camera_cfg = self._sensors_args.get("camera", {}) if self._sensors_args is not None else {}
            self._camera_type = camera_cfg.get("type", "rgb")
            self._num_image_stack = 3
            image_res = self._sensors_args["camera"]["res"]
            if self._camera_type == "depth":
                image_shape = (self._num_image_stack, image_res[0], image_res[1])
            else:
                image_shape = (self._num_image_stack * 3, image_res[0], image_res[1])
            observation_space_dict["ego_centric_camera_observation"] = spaces.Box(
                low=0,
                high=255,
                dtype=np.uint8,
                shape=image_shape,
            )

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

            self._camera_mount = self._scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True))
            self._torso_link = self._robot.get_link("torso_link")
            camera_cfg = self._sensors_args["camera"]

            self._camera = self._scene.add_sensor(
                gs.sensors.BatchRendererCameraOptions(
                    res=camera_cfg["res"],
                    pos=camera_cfg["pos"],
                    offset_T=offset_T,
                    fov=camera_cfg["fov"],
                    near=camera_cfg.get("near", 0.01),
                    far=camera_cfg.get("far", 100.0),
                    entity_idx=self._camera_mount.idx,
                    lights=[camera_cfg["lights"]],
                    env_idx=self._nominal_env_ids.cpu().tolist(),
                    render_rgb=(self._camera_type == "rgb"),
                    render_depth=(self._camera_type == "depth"),
                )
            )
            if self._camera_type == "depth":
                self._imgs_buf = torch.zeros(
                    self.nominal_env_ids.shape[0],
                    self._num_image_stack,
                    image_res[0],
                    image_res[1],
                    device=self._device,
                    dtype=torch.uint8,
                )
            else:
                self._imgs_buf = torch.zeros(
                    self.nominal_env_ids.shape[0],
                    self._num_image_stack,
                    image_res[0],
                    image_res[1],
                    3,
                    device=self._device,
                    dtype=torch.uint8,
                )

        self._observation_space = spaces.Dict(observation_space_dict)

    def _create_terrain_surface(self):
        """Build a surface for the terrain."""
        hf = self._terrain.height_field_raw
        rows, cols = hf.shape
        ground_color = np.array((255, 255, 255), dtype=np.uint8)
        wall_color = np.array((220, 50, 50), dtype=np.uint8)
        wall_threshold = 0
        arr = np.zeros((rows, cols, 3), dtype=np.uint8)
        is_wall = hf > wall_threshold
        arr[~is_wall] = ground_color
        arr[is_wall] = wall_color
        texture_array = np.transpose(arr, (1, 0, 2))[::-1, :, :].copy()
        texture = gs.textures.ImageTexture(image_array=texture_array)
        uv_scale = 1.0
        surface = gs.surfaces.Smooth(diffuse_texture=texture, smooth=False)
        return surface, uv_scale

    def _create_terrain(self):
        self._terrain = Terrain(self._terrain_args)
        terrain_surface, texture_uv_scale = self._create_terrain_surface()
        self._gs_terrain = self._scene.add_entity(
            gs.morphs.Terrain(
                pos=(0, -(self._terrain_args["border_size"] + self._terrain_args["terrain_width"] / 2.0), 0.0),
                horizontal_scale=self._terrain_args["horizontal_scale"],
                vertical_scale=self._terrain_args["vertical_scale"],
                height_field=self._terrain.height_field_raw,
                uv_scale=texture_uv_scale,
            ),
            surface=terrain_surface,
        )
        self._height_samples = (
            torch.tensor(self._terrain.heightsamples)
            .view(self._terrain.tot_rows, self._terrain.tot_cols)
            .to(self._device)
        )

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(0.0, self._env_spacing), n_envs_per_row=self._num_envs)

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}
        robot_states = states["robot_states"]
        base_pose = robot_states["base_pose"]
        base_vel = robot_states["base_vel"]
        base_quat = base_pose[:, 3:]
        inv_base_quat = inv_quat(base_quat)
        projected_gravity = transform_by_quat(
            torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(base_pose.shape[0], 1),
            inv_base_quat,
        )

        proprioceptive_observations = torch.cat(
            [
                base_pose[:, 2:3],
                projected_gravity,
                base_vel,
                robot_states["motor_joints_pos"],
                robot_states["motor_joints_vel"],
            ],
            dim=-1,
        )
        observations["proprioceptive_observations"] = proprioceptive_observations

        if hasattr(self, "_measured_heights"):
            height_field = self._measured_heights.clone()
            privileged_observations = torch.cat([proprioceptive_observations, height_field], dim=-1)
            observations["height_field"] = height_field
        else:
            privileged_observations = proprioceptive_observations

        observations["privileged_observations"] = privileged_observations

        if self._vis_obs:
            if self._camera_type == "depth":
                observations["ego_centric_camera_observation"] = self._imgs_buf
            else:
                batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
                observations["ego_centric_camera_observation"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                    batch_size, num_stack * rgb, img_height, img_width
                )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Adapt from Jie Xu's humanoid reward
        n_batch = states["progress_buf"].shape[0]
        robot_states = states["robot_states"]

        base_pose = robot_states["base_pose"]
        base_quat = base_pose[:, 3:]
        height = base_pose[:, 2]
        height_diff = height - (self._termination_height_lower_bound + self._termination_height_tolerance)
        height_reward = torch.clip(height_diff, -1.0, self._termination_height_tolerance)
        height_reward = torch.where(height_reward < 0.0, -200.0 * height_reward * height_reward, height_reward)
        height_reward = torch.where(height_reward > 0.0, self._height_reward_scale * height_reward, height_reward)

        forward_reward = robot_states["base_vel"][:, 0]

        target_dirs = self._target - base_pose[:, :3]
        target_dirs[:, 2] = 0.0
        target_dirs = torch.nn.functional.normalize(target_dirs, dim=-1)
        heading_vec = transform_by_quat(
            torch.tensor([1.0, 0.0, 0.0], device=self._device).repeat(n_batch, 1), base_quat
        )
        up_vec = transform_by_quat(torch.tensor([0.0, 0.0, 1.0], device=self._device).repeat(n_batch, 1), base_quat)

        up_reward = self._up_reward_scale * up_vec[:, 2]
        heading_reward = (heading_vec * target_dirs).sum(dim=-1)
        action_penalty = self._action_penalty * torch.sum(actions**2, dim=-1)

        return height_reward + forward_reward + up_reward + heading_reward + action_penalty

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        height = robot_states["base_pose"][:, 2]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            termination = torch.where(height < self._termination_height_lower_bound, True, termination)
            termination = torch.where(height > self._termination_height_upper_bound, True, termination)
            if hasattr(self, "_terrain_y_half_width"):
                y_pos = robot_states["base_pose"][:, 1]
                termination = torch.where(torch.abs(y_pos) > self._terrain_y_half_width, True, termination)
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

        self._update_height_measurements(env_ids)

        if self._vis_obs:
            mask = torch.isin(self.nominal_env_ids, env_ids)
            nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]

            if len(nominal_idx_to_reset) > 0:
                reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
                new_img = self.render(env_ids=reset_nominal_env_ids)
                self._imgs_buf[nominal_idx_to_reset] = new_img.unsqueeze(1)

    def _set_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(self._num_envs, self._num_actions)
        actions = actions.clamp(min=-1.0, max=1.0) * self._motor_strength
        self._robot.control_dofs_force(actions, dofs_idx_local=self._motors_dof_idx)

    def _init_height_points(self) -> None:
        """Initialize a 2D grid of height sample points in the robot body frame."""
        hf = self._sensors_args["heightfield"]
        x = torch.tensor(hf["measured_points_x"], device=self._device, dtype=torch.float)
        y = torch.tensor(hf["measured_points_y"], device=self._device, dtype=torch.float)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self._num_height_points = grid_x.numel()
        self._height_points = torch.zeros(
            self._num_envs, self._num_height_points, 3, device=self._device, dtype=torch.float
        )
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()

    def _update_height_measurements(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Sample terrain heights on a 2D grid rotated by the robot's yaw heading."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        if not hasattr(self, "_measured_heights") or len(env_ids) == 0:
            return

        base_pos = self._robot.get_pos(envs_idx=env_ids)[:, :3]
        base_quat = self._robot.get_quat(envs_idx=env_ids)  # (w, x, y, z)

        # Rotate body-frame grid offsets by yaw, then shift to world frame
        n_pts = self._num_height_points
        rotated_offsets = _quat_apply_yaw(
            base_quat.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, 4),
            self._height_points[env_ids].reshape(-1, 3),
        ).reshape(len(env_ids), n_pts, 3)

        points_world = base_pos.unsqueeze(1) + rotated_offsets

        # Shift Y into terrain-grid coordinate frame
        points_terrain = points_world.clone()
        points_terrain[:, :, 1] += self._terrain_args["border_size"] + self._terrain_args["terrain_width"] / 2.0

        points_grid = (points_terrain / self._terrain_args["horizontal_scale"]).long()
        px = points_grid[:, :, 0].view(-1).clamp(0, self._height_samples.shape[0] - 2)
        py = points_grid[:, :, 1].view(-1).clamp(0, self._height_samples.shape[1] - 2)

        heights1 = self._height_samples[px, py]
        heights2 = self._height_samples[px + 1, py]
        heights3 = self._height_samples[px, py + 1]
        heights = torch.min(torch.min(heights1, heights2), heights3) * self._terrain_args["vertical_scale"]

        self._measured_heights[env_ids] = heights.view(len(env_ids), -1)

        if self._debug:
            # Draw measurement points at sampled terrain height in viewer world frame
            points_to_draw = points_world.clone()
            points_to_draw[:, :, 2] = heights.view(len(env_ids), -1)

            center_env_id = (self._num_envs - 1) / 2.0
            y_offset = (env_ids.float() - center_env_id) * self._env_spacing
            points_to_draw[:, :, 1] += y_offset.unsqueeze(1)

            self._scene.clear_debug_objects()
            self._scene.draw_debug_spheres(
                points_to_draw.reshape(-1, 3), radius=0.02, color=(1.0, 0.0, 0.0, 0.5)
            )

    def _post_physics_step(self) -> None:
        """Post physics step."""
        self._update_height_measurements()
        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img

    def render(self, env_ids: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        if self._vis_obs:
            if env_ids is None:
                env_ids = self.nominal_env_ids
            pos = self._torso_link.get_pos()
            self._camera_mount.set_pos(pos)

            self._camera._shared_metadata.last_render_timestep = 0
            data = self._camera.read(envs_idx=env_ids)

            if self._camera_type == "depth":
                depth_image = data.depth
                if depth_image.ndim == 2:
                    depth_image = depth_image.unsqueeze(0)
                img = self._depth_to_uint8(depth_image)
            else:
                img = data.rgb
            return img
        else:
            return None

    def _depth_to_uint8(self, depth_image: torch.Tensor) -> torch.Tensor:
        near = float(self._camera._options.near)
        far = float(self._camera._options.far)
        depth_norm = (depth_image - near) / max(far - near, 1e-6)
        depth_norm = depth_norm.clamp(0.0, 1.0)
        return torch.round(depth_norm * 255.0).to(torch.uint8)

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)

        base_pos = self._robot.get_pos(envs_idx=env_ids)
        base_quat = self._robot.get_quat(envs_idx=env_ids)
        base_pose = torch.cat([base_pos, base_quat], dim=-1)
        base_vel = self._robot.get_dofs_velocity(self._base_dof_idx, envs_idx=env_ids)
        motor_joints_pos = self._robot.get_dofs_position(self._motors_dof_idx, envs_idx=env_ids)
        motor_joints_vel = self._robot.get_dofs_velocity(self._motors_dof_idx, envs_idx=env_ids)

        robot_states = {
            "base_pose": base_pose.clone(),
            "base_vel": base_vel.clone(),
            "motor_joints_pos": motor_joints_pos.clone(),
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

        self._progress_buf[env_ids] = states["progress_buf"].clone()
