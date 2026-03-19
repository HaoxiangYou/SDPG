import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import pos_lookat_up_to_T
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv
from utils.geom_utils import lookat_to_depth_euler
from utils.terrain import Terrain


class WalkerHurtle(GenesisEnv):
    """Walker hurtle environment."""

    _num_actions = 6
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))

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
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/walker.xml")),
            surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0, 1.0)),
        )

        self._root_joint_names = ["rootx", "rooty", "rootz"]
        self._motor_joint_names = ["right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
        self._root_dof_idx = [self._robot.get_joint(name).dof_start for name in self._root_joint_names]
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]

        self._motor_strength = torch.tensor([100.0, 50.0, 20.0, 100.0, 50.0, 20.0], device=self._device)

        self._default_root_dof_pos = torch.zeros(self._num_envs, len(self._root_dof_idx), device=self._device)
        self._default_motor_dof_pos = torch.zeros(self._num_envs, len(self._motors_dof_idx), device=self._device)

        self._termination_height_lower_bound = -0.5
        self._termination_height_upper_bound = 0.7
        self._termination_angle = 1.0
        self._forward_reward_scale = 10.0
        self._health_bonus = 1.0
        self._action_penalty_scale = -1e-1

        privileged_observations_dim = 17

        observation_space_dict = {
            "proprioceptive_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(17,)),
        }

        if self._terrain_args is not None:
            self._create_terrain()
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

        observation_space_dict["privileged_observations"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(privileged_observations_dim,)
        )

        if self._vis_obs:
            camera_cfg = self._sensors_args.get("camera", {}) if self._sensors_args is not None else {}
            self._camera_type = camera_cfg.get("type", "rgb")
            self._depth_camera_cfg = camera_cfg.get("depth", {})
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
            if self._camera_type == "depth":
                depth_cfg = self._depth_camera_cfg
                if lookat is not None:
                    euler_offset = lookat_to_depth_euler(self._sensors_args["camera"]["pos"], lookat)
                else:
                    euler_offset = (0.0, 0.0, 0.0)
                depth_camera_kwargs = dict(
                    pattern=gs.sensors.DepthCameraPattern(
                        res=(image_res[1], image_res[0]),
                        fx=depth_cfg.get("fx", None),
                        fy=depth_cfg.get("fy", None),
                        cx=depth_cfg.get("cx", None),
                        cy=depth_cfg.get("cy", None),
                        fov_horizontal=depth_cfg.get("fov_horizontal", self._sensors_args["camera"]["fov"]),
                        fov_vertical=depth_cfg.get("fov_vertical", None),
                    ),
                    entity_idx=self._camera_mount.idx,
                    pos_offset=tuple(self._sensors_args["camera"]["pos"]),
                    euler_offset=euler_offset,
                    min_range=depth_cfg.get("min_range", 0.0),
                    max_range=depth_cfg.get("max_range", 5.0),
                    return_world_frame=True,
                    draw_debug=self._debug,
                )
                if "no_hit_value" in depth_cfg:
                    depth_camera_kwargs["no_hit_value"] = depth_cfg["no_hit_value"]
                self._camera = self._scene.add_sensor(
                    gs.sensors.DepthCamera(**depth_camera_kwargs)
                )
                self._imgs_buf = torch.zeros(
                    self.nominal_env_ids.shape[0],
                    self._num_image_stack,
                    image_res[0],
                    image_res[1],
                    device=self._device,
                    dtype=torch.uint8,
                )
            else:
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
                    image_res[0],
                    image_res[1],
                    3,
                    device=self._device,
                    dtype=torch.uint8,
                )

        self._observation_space = spaces.Dict(observation_space_dict)

    def _create_terrain_surface(self):
        """Build a surface for the terrain."""
        hf = self._terrain.height_field_raw  # (tot_rows, tot_cols), int16
        rows, cols = hf.shape
        ground_color = np.array((255, 255, 255), dtype=np.uint8)  # white
        wall_color = np.array((220, 50, 50), dtype=np.uint8)  # red
        wall_threshold = 0  # any height above 0 is considered a wall
        arr = np.zeros((rows, cols, 3), dtype=np.uint8)
        is_wall = hf > wall_threshold
        arr[~is_wall] = ground_color
        arr[is_wall] = wall_color
        # Mesh UVs: u = normalized x (row), v = normalized y (col). Texture (u,v) -> image col=u, row=v.
        # So image[row, col] must be color for mesh (row, col) => image[j, i] = arr[i, j]: transpose.
        # Many renderers put v=0 at bottom, so flip rows to match.
        texture_array = np.transpose(arr, (1, 0, 2))[::-1, :, :].copy()
        texture = gs.textures.ImageTexture(image_array=texture_array)
        uv_scale = 1.0
        # Smooth Plastic: low roughness (0.1) and ior=1.5 for a bright, shiny look like the default terrain style
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
        self._scene.build(n_envs=self._num_envs, env_spacing=(0.0, 2.0), n_envs_per_row=self._num_envs)

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}
        robot_states = states["robot_states"]
        proprioceptive_observations = torch.cat(
            [
                robot_states["root_joints_pos"][:, 1:],
                robot_states["motor_joints_pos"],
                robot_states["root_joints_vel"],
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
                # Reshape: (batch, num_stack, H, W, 3) -> (batch, num_stack * 3, H, W)
                observations["ego_centric_camera_observation"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                    batch_size, num_stack * rgb, img_height, img_width
                )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        forward_vel = states["robot_states"]["root_joints_vel"][:, 0]
        forward_reward = forward_vel * self._forward_reward_scale

        action_penalty = torch.sum(actions**2, dim=-1) * self._action_penalty_scale

        reward = forward_reward + action_penalty + self._health_bonus
        return reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        height = robot_states["root_joints_pos"][:, 2]
        angle = robot_states["root_joints_pos"][:, 1]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            termination = torch.where(height < self._termination_height_lower_bound, True, termination)
            termination = torch.where(height > self._termination_height_upper_bound, True, termination)
            termination = torch.where(torch.abs(angle) > self._termination_angle, True, termination)
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

        self._update_height_measurements(env_ids)

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
        actions = actions.clamp(min=-1.0, max=1.0) * self._motor_strength
        self._robot.control_dofs_force(actions, dofs_idx_local=self._motors_dof_idx)

    def _init_height_points(self) -> None:
        """Initialize height sample points along X (2D motion). Uses sensors_args['heightfield']."""
        hf = self._sensors_args["heightfield"]
        res = float(hf.get("res", 0.2))
        ahead = float(hf.get("ahead", 8.0))
        backward = float(hf.get("backward", 2.0))
        n = int(round((ahead + backward) / res)) + 1
        x_offsets = torch.linspace(-backward, ahead, n, device=self._device, dtype=torch.float)
        self._num_height_points = n
        # (num_envs, num_points, 3): X offsets along forward, Y=0, Z=0 (robot frame, 2D)
        self._height_points = torch.zeros(
            self._num_envs, self._num_height_points, 3, device=self._device, dtype=torch.float
        )
        self._height_points[:, :, 0] = x_offsets.unsqueeze(0).expand(self._num_envs, -1)

    def _update_height_measurements(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Sample terrain height along X at current base pose (2D: same Y as base)."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        if not hasattr(self, "_measured_heights") or len(env_ids) == 0:
            return
        base_pos = self._robot.get_pos(envs_idx=env_ids)[:, :3]
        # World positions: base + Y offsets
        points_continuous = base_pos.unsqueeze(1) + self._height_points[env_ids]
        points_continuous[:, :, 1] += self._terrain_args["border_size"] + self._terrain_args["terrain_width"] / 2.0

        points_grid = (points_continuous / self._terrain_args["horizontal_scale"]).long()
        px = points_grid[:, :, 0].view(-1).clamp(0, self._height_samples.shape[0] - 2)
        py = points_grid[:, :, 1].view(-1).clamp(0, self._height_samples.shape[1] - 2)

        heights1 = self._height_samples[px, py]
        heights2 = self._height_samples[px + 1, py]
        heights3 = self._height_samples[px, py + 1]
        heights = torch.min(torch.min(heights1, heights2), heights3) * self._terrain_args["vertical_scale"]

        self._measured_heights[env_ids] = heights.view(env_ids.shape[0], -1)

        if self._debug:
            points_to_draw = points_continuous.clone()
            points_to_draw[:, :, 2] = heights.view(env_ids.shape[0], -1)
            points_to_draw[:, :, 1] -= self._terrain_args["border_size"] + self._terrain_args["terrain_width"] / 2.0

            center_env_id = (self._num_envs - 1) / 2.0
            y_offset = (env_ids - center_env_id) * 2.0
            points_to_draw[:, :, 1] += y_offset.unsqueeze(1)

            points_to_draw = points_to_draw.view(-1, 3)

            self._scene.clear_debug_objects()
            self._scene.draw_debug_spheres(points_to_draw, radius=0.01, color=(1.0, 0.0, 0.0, 0.5))

    def _post_physics_step(self) -> None:
        """Post physics step"""
        self._update_height_measurements()
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

            if self._camera_type == "depth":
                self._scene.sim._sensor_manager.step()
                # TODO: Genesis Depth Camera sensor requires much less memory than the gs-Madrona RGB render
                # TODO: No additional changes are yet applied to Genesis Source Code to per nominal environment rendering
                # TODO: Memory usage are good enough to run in 3070 or 4080 GPU
                depth_image = self._camera.read_image()
                if depth_image.ndim == 2:
                    depth_image = depth_image.unsqueeze(0)
                img = self._depth_to_uint8(depth_image[env_ids])
            else:
                # TODO: genesis will refresh the image when the scene._dt is different from the last render time
                # TODO: temporarily we hack by setting the last render time to 0 to force render the new image
                self._camera._shared_metadata.last_render_timestep = 0
                data = self._camera.read(envs_idx=env_ids)
                img = data.rgb
            return img
        else:
            return None

    def _depth_to_uint8(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_cfg = self._depth_camera_cfg
        min_range = float(depth_cfg.get("min_range", 0.0))
        max_range = float(depth_cfg.get("max_range", 5.0))

        depth_norm = (depth_image - min_range) / max(max_range - min_range, 1e-6)
        depth_norm = depth_norm.clamp(0.0, 1.0)
        return torch.round(depth_norm * 255.0).to(torch.uint8)

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

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
