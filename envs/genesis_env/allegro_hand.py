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


class AllegroHand(GenesisEnv):
    """Allegro hand in-hand manipulationenvironment."""

    _num_actions = 16
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,))

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
        episode_length = int(10.0 / dt)

        early_termination = True

        self._vis_obs = vis_obs

        if sensors_args is None:
            sensors_args = {
                "camera": {
                    "res": [256, 256],
                    "pos": [0.40, 0.05, 0.425],
                    "lookat": [0.25, -0.10, 0.275],
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

        if vis_obs:
            self._num_image_stack = 3
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(124,)),
                    # observation that ignores cube information (infer from images)
                    "proprioception_and_target": spaces.Box(low=-np.inf, high=np.inf, shape=(107,)),
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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(124,)),
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

        igr = init_goal_rotation or {}
        self._init_goal_rotation_enabled = bool(igr.get("enable", True))
        self._init_goal_angle_min_deg = float(igr.get("angle_deg_min", 60.0))
        self._init_goal_angle_max_deg = float(igr.get("angle_deg_max", 90.0))

    def init_scene(self) -> None:
        """Initialize the scene."""

        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/allegro_hand/right_hand.xml")),
        )

        self._plane = self._scene.add_entity(gs.morphs.Plane())

        self._cube = self._scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/meshes/cube.obj"), scale=0.03
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/textures/cube.png")
                )
            ),
            material=gs.materials.Rigid(friction=0.3, rho=600.0),
        )

        # target quat
        self._target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)

        self._target = self._scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/meshes/cube.obj"),
                scale=0.03,
                collision=False,
                pos=(0.325, 0.3, 0.2475),
            ),
            material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=os.path.join(os.path.dirname(__file__), "../../assets/dexcube/textures/cube.png")
                )
            ),
        )

        # # A record of the previous actions
        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)

        self._hand_motor_joint_names = [
            "ffj0",
            "ffj1",
            "ffj2",
            "ffj3",
            "mfj0",
            "mfj1",
            "mfj2",
            "mfj3",
            "rfj0",
            "rfj1",
            "rfj2",
            "rfj3",
            "thj0",
            "thj1",
            "thj2",
            "thj3",
        ]

        self._hand_motors_dof_idx = []
        for name in self._hand_motor_joint_names:
            self._hand_motors_dof_idx.extend(self._robot.get_joint(name).dofs_idx_local)
        self._cube_dof_idx = self._cube.get_joint("cube_obj_baselink_joint").dofs_idx_local

        self._finger_tip_link_names = [
            "ff_tip",
            "mf_tip",
            "rf_tip",
            "th_tip",
        ]

        self._finger_tip_link_idx = [self._robot.get_link(name).idx_local for name in self._finger_tip_link_names]

        self._default_target_quat = torch.tensor([2**0.5 / 2, 2**0.5 / 2, 0.0, 0.0], device=self._device).repeat(
            self._num_envs, 1
        )
        self._default_cube_pos = torch.tensor([0.25, 0.0, 0.275], device=self._device).repeat(self._num_envs, 1)
        self._in_hand_pos = torch.tensor([0.25, 0.0, 0.25], device=self._device).repeat(self._num_envs, 1)
        self._default_cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)
        self._default_hand_dof_pos = torch.tensor(
            [
                0.0,
                0.58058,
                0.701595,
                0.538675,
                0.0,
                0.60767,
                0.758085,
                0.741625,
                0.0,
                0.8876,
                0.720425,
                0.5848,
                0.263,
                0.32612,
                1.08493,
                0.806715,
            ],
            device=self._device,
        ).repeat(self._num_envs, 1)

        self._vel_obs_scale = 0.2
        self._fall_distance = 0.2
        self._dist_reward_scale = -10.0
        self._rot_reward_scale = 1.0
        self._healthy_reward = 3.0
        self._action_penalty = -0.0002

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
            # NOTE: A dummy link for the camera to attach to; without entity_idx the batch renderer
            # uses a single world pose, only env 0 is in view and other envs render black.
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

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(1.0, 1.0))
        # Control range (low, high) for hand motors.
        self._hand_motors_ctrl_lower, self._hand_motors_ctrl_upper = self._robot.get_dofs_limit(
            dofs_idx_local=self._hand_motors_dof_idx
        )

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}

        # Privileged observations follow the IsaacLab's implementation
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
        # TODO, in IsaacLab, the observation contains figer tip pose and velocity;
        # Figer tip is not part of the state, but can be derived from the state
        # We currently obtain the finger tip pose directly from simulation, which may be problematic
        # However, since compute_observations is called after set_states, this implementation may be fine for now.
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
                # hand
                scaled_hand_dof_pos,
                scaled_hand_dof_vel,
                # object
                cube_pos,
                cube_quat,
                cube_linear_vel,
                scaled_cube_angular_vel,
                # goal
                self._in_hand_pos,
                target_quat,
                rot_diff,
                # finger_tip
                finger_tip_pos,
                finger_tip_quat,
                finger_tip_vel,
                finger_tip_angular_vel,
                # actions
                prev_actions,
            ],
            dim=-1,
        )
        observations["privileged_observations"] = privileged_observations

        if self._vis_obs:
            proprioception_and_target = torch.cat(
                [
                    # hand
                    scaled_hand_dof_pos,
                    scaled_hand_dof_vel,
                    # goal
                    self._in_hand_pos,
                    target_quat,
                    # finger_tip
                    finger_tip_pos,
                    finger_tip_quat,
                    finger_tip_vel,
                    finger_tip_angular_vel,
                    # actions
                    prev_actions,
                ],
                dim=-1,
            )

            observations["proprioception_and_target"] = proprioception_and_target

            batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
            # NOTE: for AFRL agent, RGB observation and privileged observations may has different shapes
            # Reshape: (batch, num_stack, H, W, 3) -> (batch, num_stack * 3, H, W)
            observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                batch_size, num_stack * rgb, img_height, img_width
            )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        robot_states = states["robot_states"]
        cube_pos = robot_states["cube_pos"]
        cube_quat = robot_states["cube_quat"]
        target_quat = robot_states["target_quat"]
        goal_dis = torch.norm(cube_pos - self._in_hand_pos, p=2, dim=-1)
        dist_reward = self._dist_reward_scale * goal_dis

        quat_diff = transform_quat_by_quat(inv_quat(target_quat), cube_quat)
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

        rot_rew = -(rot_dist**2) * self._rot_reward_scale

        action_penalty = self._action_penalty * torch.sum(actions**2, dim=-1)

        # restore the average angle difference between the cube and the target in degrees
        self._infos["angle_diff"] = torch.rad2deg(2 * torch.norm(quat_diff[:, 1:4], p=2, dim=-1)).mean().item()

        return dist_reward + rot_rew + action_penalty + self._healthy_reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            termination = torch.norm(robot_states["cube_pos"] - self._in_hand_pos, p=2, dim=-1) > self._fall_distance
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        hand_dof_pos = self._default_hand_dof_pos[env_ids]
        cube_pos = self._default_cube_pos[env_ids]
        cube_quat = self._default_cube_quat[env_ids]
        target_quat = self._default_target_quat[env_ids]

        if self._randomize_init:
            cube_pos = cube_pos + (torch.rand_like(cube_pos) - 0.5) * 0.02
            if self._init_goal_rotation_enabled:
                # One world-axis rotation per env: pitch (+Y) or roll (+X), angle uniform in [min, max] degrees.
                n = env_ids.shape[0]
                dev = self.device
                span = max(self._init_goal_angle_max_deg - self._init_goal_angle_min_deg, 0.0)
                angles_deg = self._init_goal_angle_min_deg + torch.rand(n, device=dev) * span
                angles = torch.deg2rad(angles_deg)
                pick_pitch = torch.rand(n, device=dev) >= 0.5
                x_axis = torch.tensor([1.0, 0.0, 0.0], device=dev).unsqueeze(0).expand(n, 3)
                y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev).unsqueeze(0).expand(n, 3)
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
        # Map actions from [-1, 1] to target position [ctrl_lower, ctrl_upper]
        target_pos = self._hand_motors_ctrl_lower + (actions + 1.0) * 0.5 * (
            self._hand_motors_ctrl_upper - self._hand_motors_ctrl_lower
        )
        self._robot.control_dofs_position(
            target_pos,
            dofs_idx_local=self._hand_motors_dof_idx,
        )

    def _post_physics_step(self) -> None:
        # TODO:
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

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
