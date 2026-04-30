import os
from typing import Any, Dict, Optional, Sequence, Tuple

import genesis as gs
import numpy as np
import torch
import torch.nn.functional as F
from genesis.utils.geom import axis_angle_to_quat, inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv
from utils.geom_utils import lookat_to_depth_euler
from utils.terrain import Terrain
from genesis.utils.geom import pos_lookat_up_to_T

def torch_rand_float(lower: float, upper: float, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


class Go2Terrain(GenesisEnv):
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
        terrain_args: Dict[str, Any] | None = None,
        sim_options: gs.options.SimOptions | None = None,
        viewer_options: gs.options.ViewerOptions | None = None,
        rigid_options: gs.options.RigidOptions | None = None,
        vis_options: gs.options.VisOptions | None = None,
        show_viewer: bool = False,
        show_FPS: bool = False,
        domain_rand_options: Dict[str, Any] | None = None,
        debug: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        episode_length = 1000  # Will be converted based on dt in reference
        early_termination = True

        # self._num_single_obs = 45
        # self._obs_frame_stack = 5
        # self._num_obs = self._num_single_obs * self._obs_frame_stack
        
        # self._num_single_privileged_obs = self._num_single_obs + 31 + 81 + 17 + 3
        # self._privileged_frame_stack = 5
        # self._num_privileged_obs = self._num_single_privileged_obs * self._privileged_frame_stack

        self._num_height_points = 81

        
        self._num_single_obs = 45
        self._num_history_obs = 20
        self._num_obs = self._num_single_obs * self._num_history_obs


        self._num_single_privileged_obs = self._num_single_obs + 3 # + 81  + 31 + 17 + 3 # 60
        self._privileged_frame_stack = 1
        self._num_privileged_obs = self._num_single_privileged_obs * self._privileged_frame_stack

        self._observation_space = spaces.Dict(
            {
                "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_privileged_obs,)),
                "observations": spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_obs,)),
                "height_field": spaces.Box(low=-np.inf, high=np.inf, shape=(self._num_height_points,)),
            }
        )

        self._dt = sim_options.dt
        self._domain_rand_options = domain_rand_options
        self._terrain_cfg = terrain_args
        self._train = False
        self._debug = debug
        self._vis_obs = vis_obs

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

    def init_scene(self) -> None:
        """Initialize the scene."""

        # Action parameters
        self._action_scale = 0.5
        self._clip_actions = 100.0
        self._clip_obs = 100.0

        # Observation scales
        self._obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        }
        self._obs_noise_cfg = {
            "ang_vel": 0.1,
            "gravity": 0.02,
            "dof_pos": 0.01,
            "dof_vel": 0.5,
        }

        # Base initialization
        base_init_pos = [0.0, 0.0, 0.42]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]

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

        self._base_init_pos = torch.tensor(base_init_pos, dtype=gs.tc_float, device=self._device)
        self._base_init_quat = torch.tensor(base_init_quat, dtype=gs.tc_float, device=self._device)
        self._inv_base_init_quat = inv_quat(self._base_init_quat)

        # Command configuration
        self._command_cfg = {
            "num_commands": 4,
            "lin_vel_x_range": [-1.0, 1.0],
            "lin_vel_y_range": [-1.0, 1.0],
            "ang_vel_range": [-1.0, 1.0],
            "heading_range": [-3.14, 3.14],
        }

        self._commands_scale = torch.tensor(
            [
                self._obs_scales["lin_vel"],
                self._obs_scales["lin_vel"],
                self._obs_scales["ang_vel"],
            ],
            device=self._device,
            dtype=torch.float,
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
        self._termination_contact_link_names = ["base"]
        self._num_feet = 4

        # Add plane
        self._terrain = Terrain(self._terrain_cfg)
        self._create_heightfield()

        # Add go2 robot
        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/unitree_go2/go2_genesis.xml")),
        )
        from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

        for solver in self._scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self._rigid_solver = solver

        self._friction_value_offset = (
            self._domain_rand_options["friction_range"][0] + self._domain_rand_options["friction_range"][1]
        ) / 2  # mean value
        self._kp_scale_offset = (
            self._domain_rand_options["kp_scale_range"][0] + self._domain_rand_options["kp_scale_range"][1]
        ) / 2  # mean value
        self._kd_scale_offset = (
            self._domain_rand_options["kd_scale_range"][0] + self._domain_rand_options["kd_scale_range"][1]
        ) / 2  # mean value
        self._push_interval = np.ceil(self._domain_rand_options["push_interval_s"] / self._dt)

        if self._vis_obs:
            camera_cfg = self._sensors_args.get("camera", {}) if self._sensors_args is not None else {}
            self._camera_type = camera_cfg.get("type", "rgb")
            self._depth_camera_cfg = camera_cfg.get("depth", {})
            self._use_bvh_depth = self._camera_type == "depth" and camera_cfg.get("depth_backend", "madrona") == "bvh"
            self._camera_near = float(self._depth_camera_cfg.get("min_range", camera_cfg.get("near", 0.01)))
            self._camera_far = float(self._depth_camera_cfg.get("max_range", camera_cfg.get("far", 5.0)))

            image_res = camera_cfg["res"]
            if self._camera_type == "depth":
                self._num_image_stack = 1
                # camera.res in YAML is [H, W] (e.g. 60×106 for parkour-aligned depth). Genesis DepthCameraPattern takes (width, height).
                self._depth_res_hw = (int(image_res[0]), int(image_res[1]))
                # obs_res: (H, W) after crop + bicubic resize (default 58×87, same as IsaacGym resized (87,58) in width×height order).
                obs_res = camera_cfg["obs_res"]
                self._depth_out_h = int(obs_res[0])
                self._depth_out_w = int(obs_res[1])
                image_shape = (self._num_image_stack, self._depth_out_h, self._depth_out_w)
                ego_centric_camera_observation_space = spaces.Box(
                    low=-0.5, high=0.5, dtype=np.float32, shape=image_shape,
                )
            else:
                self._num_image_stack = 1
                image_shape = (self._num_image_stack * 3, image_res[0], image_res[1])
                ego_centric_camera_observation_space = spaces.Box(
                    low=0, high=255, dtype=np.uint8, shape=image_shape,
                )

            self._camera_mount = self._scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True))
            self._torso_link = self._robot.get_link("base")
            lookat = camera_cfg.get("lookat", None)
            env_idx = self._nominal_env_ids.cpu().tolist()

            if self._use_bvh_depth:
                depth_cfg = self._depth_camera_cfg
                euler_offset = lookat_to_depth_euler(camera_cfg["pos"], lookat) if lookat else (0.0, 0.0, 0.0)
                depth_camera_kwargs = dict(
                    pattern=gs.sensors.DepthCameraPattern(
                        res=(image_res[1], image_res[0]),
                        fx=depth_cfg.get("fx", None), fy=depth_cfg.get("fy", None),
                        cx=depth_cfg.get("cx", None), cy=depth_cfg.get("cy", None),
                        fov_horizontal=depth_cfg.get("fov_horizontal", camera_cfg.get("fov", 80.0)),
                        fov_vertical=depth_cfg.get("fov_vertical", None),
                    ),
                    entity_idx=self._camera_mount.idx,
                    pos_offset=tuple(camera_cfg["pos"]),
                    euler_offset=euler_offset,
                    min_range=self._camera_near,
                    max_range=self._camera_far,
                    return_world_frame=True,
                    draw_debug=self._debug,
                    env_idx=env_idx,
                )
                if "no_hit_value" in depth_cfg:
                    depth_camera_kwargs["no_hit_value"] = depth_cfg["no_hit_value"]
                self._camera = self._scene.add_sensor(gs.sensors.DepthCamera(**depth_camera_kwargs))
            else:
                offset_T = camera_cfg.get("offset_T", None)
                if offset_T is not None:
                    offset_T = torch.tensor(offset_T, device=self._device)
                elif lookat is not None:
                    offset_T = pos_lookat_up_to_T(np.array(camera_cfg["pos"]), np.array(lookat), np.array((0.0, 0.0, 1.0)))
                else:
                    offset_T = np.eye(4)
                self._camera = self._scene.add_sensor(gs.sensors.BatchRendererCameraOptions(
                    res=camera_cfg["res"], pos=camera_cfg["pos"], offset_T=offset_T, fov=camera_cfg["fov"],
                    near=camera_cfg.get("near", 0.01), far=camera_cfg.get("far", 100.0),
                    entity_idx=self._camera_mount.idx, lights=[camera_cfg["lights"]], env_idx=env_idx,
                    render_rgb=(self._camera_type == "rgb"), render_depth=(self._camera_type == "depth"),
                ))

            if self._camera_type == "depth":
                self._imgs_buf = torch.zeros(
                    self.nominal_env_ids.shape[0], self._num_image_stack, self._depth_out_h, self._depth_out_w,
                    device=self._device, dtype=torch.float32,
                )
            else:
                self._imgs_buf = torch.zeros(
                    self.nominal_env_ids.shape[0], self._num_image_stack, image_res[0], image_res[1], 3,
                    device=self._device, dtype=torch.uint8,
                )
            self._observation_space["ego_centric_camera_observation"] = ego_centric_camera_observation_space


    def _compare_reward_functions(self):
        # prepare list of functions
        self._reward_functions = []
        self._reward_names = []
        for name, scale in self._reward_scales.items():
            self._reward_names.append(name)
            name = "_reward_" + name
            self._reward_functions.append(getattr(self, name))

    def _init_buffers(self):
        self._common_step_counter = 0
        self._base_euler = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_lin_vel = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_ang_vel = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_lin_vel_world = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_ang_vel_world = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)   
        self._projected_gravity = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._global_gravity = torch.tensor(np.array([0.0, 0.0, -1.0]), device=self._device, dtype=torch.float)
        self._forward_vec = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._forward_vec[:, 0] = 1.0

        self._obs_history_buf = torch.zeros(
            (self._num_envs, self._num_obs), device=self._device, dtype=torch.float
        )
        self._obs_noise = torch.zeros((self._num_envs, self._num_single_obs), device=self._device, dtype=torch.float)
        self._privileged_obs_buf = torch.zeros((self._num_envs, self._num_privileged_obs), device=self._device, dtype=torch.float)

        # Only use this for resetting the base velocities
        self._base_dof_idx = self._robot.base_joint.dofs_idx_local  # only use this for resetting the base velocities

        # Get motor DOF indices after scene is built
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]

        self._prepare_obs_noise()

        # Termination parameters
        self._termination_roll_threshold = 0.4
        self._termination_pitch_threshold = 0.4
        self._max_projected_gravity = -0.1

        # Reward configuration
        self._soft_dof_limit = 0.9
        self._reward_base_height_target = 0.3
        self._only_positive_rewards = True
        self._reward_tracking_sigma = 0.25
        self._foot_clearance_tracking_sigma = 0.01
        self._foot_clearance_target = 0.09  # desired foot clearance above ground [m]
        self._foot_height_offset = 0.022  # height of the foot coordinate origin above ground [m]

        self._reward_scales = {
            # limitation
            "dof_pos_limits": -2.0,
            "collision": -1.0,
            # command tracking
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 1.5,
            # smoothness
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            # "dof_power": -2.e-4,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "action_smoothness": -0.01,
            # gait
            # "stand_still": -0.5,
            "feet_air_time": 1.0,
            "feet_contact_stand_still": 0.5,
            "feet_clearance": 0.2,
            # "feet_distance": -1.0,
            "hip_pos": -0.05,
        }

        # PD control parameters
        self._kp = 30.0
        self._kd = 1.5

        self._p_gains, self._d_gains = [self._kp] * 12, [self._kd] * 12
        self._p_gains = torch.tensor(self._p_gains, device=self._device)
        self._d_gains = torch.tensor(self._d_gains, device=self._device)
        self._batched_p_gains = self._p_gains[None, :].repeat(self._num_envs, 1)
        self._batched_d_gains = self._d_gains[None, :].repeat(self._num_envs, 1)

        self._robot.set_dofs_kp(self._p_gains, self._motors_dof_idx)
        self._robot.set_dofs_kv(self._d_gains, self._motors_dof_idx)

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
        self._termination_contact_link_indices = find_link_indices(self._termination_contact_link_names)
        if self._domain_rand_options["obtain_link_contact_states"]:
            self._contact_state_link_indices = find_link_indices(self._domain_rand_options["contact_state_link_names"])

        assert len(self._penalized_contact_link_indices) > 0
        assert len(self._feet_link_indices) > 0
        assert len(self._termination_contact_link_indices) > 0
        assert len(self._contact_state_link_indices) > 0
        self._feet_link_indices_world_frame = [i + 1 for i in self._feet_link_indices]

        # Buffers for observation computation
        self._actions = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_last_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._dof_pos = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._dof_vel = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_dof_vel = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._root_vel = torch.zeros(self._num_envs, 3, device=self._device, dtype=torch.float)
        self._base_pos = torch.zeros(self._num_envs, 3, device=self._device, dtype=torch.float)
        self._base_quat = torch.zeros(self._num_envs, 4, device=self._device, dtype=torch.float)

        # self._com = torch.zeros(self._num_envs, 3, device=self._device, dtype=torch.float)
        self._base_link_index = 1

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

        self._feet_max_height = torch.zeros(self._num_envs, self._num_feet, device=self._device)

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

        # Terrain parameters
        self._init_height_points()
        self._measured_heights = torch.zeros(
            self._num_envs, self._num_height_points, device=self._device, requires_grad=False
        )
        if self._domain_rand_options["obtain_link_contact_states"]:
            self._link_contact_states = torch.zeros(
                self._num_envs,
                len(self._contact_state_link_indices),
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )

        self._motor_offsets = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float)
        self._motor_strengths = torch.ones((self._num_envs, self._num_actions), device=self._device, dtype=torch.float)

        self._foot_positions = torch.zeros((self._num_envs, self._num_feet, 3), device=self._device, dtype=torch.float)
        self._foot_velocities = torch.zeros((self._num_envs, self._num_feet, 3), device=self._device, dtype=torch.float)

        # terrain related buffers
        self._env_origins = torch.zeros(self._num_envs, 3, device=self._device, requires_grad=False)
        self._custom_origins = False
        # Terrain information around feet
        if self._domain_rand_options["obtain_terrain_info_around_feet"]:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs,
                len(self._feet_link_indices) * 3,
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )
            self._height_around_feet = torch.zeros(
                self._num_envs,
                len(self._feet_link_indices),
                9,
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )

        # domain randomization parameters
        self._init_domain_params()
        # randomize friction
        if self._domain_rand_options["randomize_friction"]:
            self._randomize_link_friction(torch.arange(0, self._num_envs))
        # randomize base mass
        if self._domain_rand_options["randomize_base_mass"]:
            self._randomize_base_mass(torch.arange(0, self._num_envs))
        # randomize COM displacement
        if self._domain_rand_options["randomize_com_displacement"]:
            self._randomize_com_displacement(torch.arange(0, self._num_envs))
        # randomize pd gain
        if self._domain_rand_options["randomize_kp_scale"]:
            self._randomize_kp(torch.arange(0, self._num_envs))
        if self._domain_rand_options["randomize_kd_scale"]:
            self._randomize_kd(torch.arange(0, self._num_envs))

    def build_scene(self) -> None:
        self._set_camera()

        self._scene.build(n_envs=self._num_envs, env_spacing=(0.0, 0.0), n_envs_per_row=self._num_envs)

        self._init_buffers()
        self._compare_reward_functions()
        self._get_env_origins()

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
        self._common_step_counter += 1
        # resample commands when is half of the episode lenght
        # update buffers has been called in get_states
        env_ids = (self._progress_buf % int(self._episode_length / 5) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        forward = quat_apply(self._base_quat, self._forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self._commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self._commands[:, 3] - heading),
            self._command_cfg["ang_vel_range"][0],
            self._command_cfg["ang_vel_range"][1],
        )

        if self._domain_rand_options["push_robot"] and (self._common_step_counter % self._push_interval == 0):
            self._push_robots()
        self._randomize_rigids(env_ids)
        self._randomize_controls(env_ids)

        self._render_headless()

        # Link contact state
        if self._domain_rand_options["obtain_link_contact_states"]:
            self._link_contact_states = 1.0 * (
                torch.norm(self._link_contact_forces[:, self._contact_state_link_indices, :], dim=-1) > 1.0
            )

        if self._terrain_cfg["measure_heights"]:
            self._update_surrounding_heights()
            if self._domain_rand_options["obtain_terrain_info_around_feet"]:
                self._calc_terrain_info_around_feet()
        self._last_actions[:] = self._actions[:]
        self._last_last_actions[:] = self._last_actions[:]
        self._last_dof_vel[:] = self._dof_vel[:]

        # we have to compute the observation here becuase the compute_observation will be called multiple times
        _obs_buf = torch.cat(
            [   
                # self._base_lin_vel * self._obs_scales["lin_vel"], # 3
                self._base_ang_vel * self._obs_scales["ang_vel"],  # 3
                self._projected_gravity,  # 3
                self._commands[:, :3] * self._commands_scale,  # 3
                (self._dof_pos - self._default_dof_pos) * self._obs_scales["dof_pos"],
                self._dof_vel * self._obs_scales["dof_vel"],
                self._actions,
            ],
            axis=-1,
        )

        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            # Roll the buffer to shift old frames: [t-2, t-1, t-0] -> [t-1, t-0, None]
            # This moves older frames "to the left" and makes room for the new frame
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img
        

        # domain_randomization_info = torch.cat(
        #     (
        #         (self._friction_values - self._friction_value_offset),  # 1
        #         self._added_base_mass,  # 1
        #         self._base_com_bias,  # 3
        #         self._rand_push_vels[:, :2],  # 2
        #         (self._kp_scale - self._kp_scale_offset),  # num_actions
        #         (self._kd_scale - self._kd_scale_offset),  # num_actions
        #     ),
        #     dim=-1,
        # )

        # _privileged_obs_buf = torch.cat(
        #     (_obs_buf, domain_randomization_info, self._base_lin_vel * self._obs_scales["lin_vel"]),
        #     axis=-1,
        # )

        # _privileged_obs_buf = torch.cat(
        #         (
        #             _privileged_obs_buf,  # previous
        #             self._link_contact_states,  # contact states of thighs, calfs and feet (4+4+4)=12
        #         ),
        #         dim=-1,
        #     )

        # heights = (
        #         torch.clip(self._base_pos[:, 2].unsqueeze(1) - 0.5 - self._measured_heights, -1, 1.0)
        #         * self._obs_scales["height_measurements"]
        #     )
        
        # _privileged_obs_buf = torch.cat((_privileged_obs_buf, heights), dim=-1)


        _privileged_obs_buf = torch.cat(
            (_obs_buf, self._base_lin_vel * self._obs_scales["lin_vel"]),
            axis=-1,
        )

        _privileged_obs_buf = torch.clip(_privileged_obs_buf, -self._clip_obs, self._clip_obs)

        # add noise
        if self._train:
            _obs_buf += torch_rand_float(-1.0, 1.0, (self._num_single_obs,), self._device) * self._obs_noise

        
        _obs_buf = torch.clip(_obs_buf, -self._clip_obs, self._clip_obs)

        self._obs_history_buf = torch.cat([self._obs_history_buf[:, self._num_single_obs :], _obs_buf.detach()], dim=1)
        self._privileged_obs_buf = torch.cat([self._privileged_obs_buf[:, self._num_single_privileged_obs :], _privileged_obs_buf.detach()], dim=1)

    def render(self, env_ids: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        if not self._vis_obs:
            return None
        if env_ids is None:
            env_ids = self.nominal_env_ids

        # Attach the camera to the torso pose
        self._camera_mount.set_pos(self._torso_link.get_pos())

        if self._use_bvh_depth:
            # BVH DepthCamera path: explicit sensor step + read_image.
            self._scene.sim._sensor_manager.step()
            depth_image = self._camera.read_image(envs_idx=env_ids)
            if depth_image.ndim == 2:
                depth_image = depth_image.unsqueeze(0)
            return self._process_depth_extreme_parkour(depth_image)

        # Madrona BatchRenderer path: force refresh then read rgb/depth.
        self._camera._shared_metadata.last_render_timestep = 0
        data = self._camera.read(envs_idx=env_ids)
        if self._camera_type == "depth":
            depth_image = data.depth
            if depth_image.ndim == 2:
                depth_image = depth_image.unsqueeze(0)
            return self._process_depth_extreme_parkour(depth_image)
        return data.rgb

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Compute observations based on go2_env.py structure."""
        # Compute privileged observations (matching genesis go2_env.py structure)

        heights = (
                torch.clip(self._base_pos[:, 2].unsqueeze(1) - 0.5 - self._measured_heights, -1, 1.0)
                * self._obs_scales["height_measurements"]
            )

        observations = {
            "privileged_observations": self._privileged_obs_buf,
            "observations": self._obs_history_buf,
            "height_field": heights,
        }

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

    def get_observations(self):
        return self._obs_history_buf

    def get_privileged_observations(self):
        return self._privileged_obs_buf

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        """Compute reward based on reward_cfg from go2_env.py."""
        # Compute all reward components
        reward = torch.zeros(self._num_envs, device=self._device)

        for i in range(len(self._reward_functions)):
            name = self._reward_names[i]
            reward += self._reward_functions[i]() * self._reward_scales[name]
            self._infos[name] = self._reward_functions[i]() * self._reward_scales[name]

        if self._only_positive_rewards:
            reward = torch.clip(reward, min=0.0)
        return reward * self._dt

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        """Compute termination based on roll/pitch angles."""
        termination = torch.any(
            torch.norm(
                self._link_contact_forces[:, self._termination_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )

        if self._early_termination:
            # Terminate if roll or pitch exceeds threshold
            termination |= torch.abs(self._base_euler[:, 0]) > self._termination_roll_threshold
            termination |= torch.abs(self._base_euler[:, 1]) > self._termination_pitch_threshold
            # termination |= self._projected_gravity[:, 2] > self._max_projected_gravity

        return termination

    def _resample_commands(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Resample velocity commands for nominal environments and propagate to auxiliaries.

        Args:
            envs_idx: Environment indices to consider for resampling.
                Only nominal environments within this set are resampled;
                their auxiliary environments are then set to the same commands.
        """
        envs_idx = torch.as_tensor(envs_idx, device=self._device, dtype=torch.long)
        mask = torch.isin(envs_idx, self._nominal_env_ids)
        nominal_env_ids = envs_idx[mask]
        if nominal_env_ids.numel() == 0:
            return

        self._commands[nominal_env_ids, 0] = torch_rand_float(
            self._command_cfg["lin_vel_x_range"][0],
            self._command_cfg["lin_vel_x_range"][1],
            (len(nominal_env_ids), 1),
            self._device,
        ).squeeze(1)
        self._commands[nominal_env_ids, 1] = torch_rand_float(
            self._command_cfg["lin_vel_y_range"][0],
            self._command_cfg["lin_vel_y_range"][1],
            (len(nominal_env_ids), 1),
            self._device,
        ).squeeze(1)

        self._commands[nominal_env_ids, 3] = torch_rand_float(
            self._command_cfg["heading_range"][0],
            self._command_cfg["heading_range"][1],
            (len(nominal_env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self._commands[nominal_env_ids, :3] *= (
            torch.norm(self._commands[nominal_env_ids, :3], dim=1) > 0.2
        ).unsqueeze(1)

        num_aux = int(self.num_auxiliary_envs)
        if num_aux > 0:
            offsets = torch.arange(1, num_aux + 1, device=self._device, dtype=torch.long)
            aux_env_ids = nominal_env_ids[:, None] + offsets[None, :]
            aux_env_ids = aux_env_ids.reshape(-1)
            aux_env_ids = aux_env_ids[aux_env_ids < self._num_envs]

            nominal_repeated = nominal_env_ids.repeat_interleave(num_aux)[: aux_env_ids.numel()]
            self._commands[aux_env_ids] = self._commands[nominal_repeated]

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environments by index."""
        if len(env_ids) == 0:
            return

        if self._terrain_cfg["curriculum"]:
            self._update_terrain_curriculum(env_ids)

        # TODO: update command curriculum

        base_pos = self._base_init_pos.unsqueeze(0).repeat(len(env_ids), 1) + self._env_origins[env_ids]
        base_quat = self._base_init_quat.unsqueeze(0).repeat(len(env_ids), 1)
        motor_dof_pos = self._default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)

        if self._randomize_init:
            # Add small random perturbations
            base_pos = base_pos + (torch.rand_like(base_pos) - 0.5) * 0.05
            angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.0
            axis = torch.nn.functional.normalize(torch.rand(len(env_ids), 3, device=self.device) - 0.5)
            base_quat = transform_quat_by_quat(base_quat, axis_angle_to_quat(angle, axis))
            motor_dof_pos = motor_dof_pos + (torch.rand_like(motor_dof_pos) - 0.5)

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

        self._robot.zero_all_dofs_velocity(env_ids)

        # Reset previous actions
        self._last_foot_vel[env_ids] = torch.zeros(len(env_ids), self._num_feet, 3, device=self._device)
        self._last_contacts[env_ids] = torch.zeros(len(env_ids), self._num_feet, dtype=torch.bool, device=self._device)
        self._foot_vel[env_ids] = torch.zeros(len(env_ids), self._num_feet, 3, device=self._device)

        # update buffers
        self._base_pos[env_ids] = base_pos
        self._base_quat[env_ids] = base_quat
        inv_base_quat = inv_quat(self._base_quat)
        self._projected_gravity = transform_by_quat(self._global_gravity, inv_base_quat)
        self._dof_pos[env_ids] = motor_dof_pos
        self._dof_vel[env_ids] = torch.zeros(len(env_ids), self._num_actions, device=self._device, dtype=torch.float)

        self._base_lin_vel[env_ids] = torch.zeros(len(env_ids), 3, device=self._device, dtype=torch.float)
        self._base_ang_vel[env_ids] = torch.zeros(len(env_ids), 3, device=self._device, dtype=torch.float)
        self._base_lin_vel_world[env_ids] = torch.zeros(len(env_ids), 3, device=self._device, dtype=torch.float)
        self._base_ang_vel_world[env_ids] = torch.zeros(len(env_ids), 3, device=self._device, dtype=torch.float)

        base_vel = torch.concat([self._base_lin_vel[env_ids], self._base_ang_vel[env_ids]], dim=1)
        self._robot.set_dofs_velocity(
            velocity=base_vel,
            dofs_idx_local=[0, 1, 2, 3, 4, 5],
            envs_idx=env_ids,
        )

        # Resample commands for reset environments
        self._resample_commands(env_ids)

        self._obs_history_buf[env_ids] = torch.zeros(
            len(env_ids), self._num_obs, device=self._device, dtype=torch.float
        )
        self._privileged_obs_buf[env_ids] = torch.zeros(
            len(env_ids), self._num_privileged_obs, device=self._device, dtype=torch.float
        )
        self._actions[env_ids] = torch.zeros(len(env_ids), self._num_actions, device=self._device, dtype=torch.float)
        self._last_actions[env_ids] = torch.zeros(
            len(env_ids), self._num_actions, device=self._device, dtype=torch.float
        )
        self._last_last_actions[env_ids] = torch.zeros(
            len(env_ids), self._num_actions, device=self._device, dtype=torch.float
        )
        self._last_dof_vel[env_ids] = torch.zeros(
            len(env_ids), self._num_actions, device=self._device, dtype=torch.float
        )
        self._feet_air_time[env_ids] = torch.zeros(len(env_ids), self._num_feet, device=self._device, dtype=torch.float)
        self._feet_max_height[env_ids] = torch.zeros(
            len(env_ids), self._num_feet, device=self._device, dtype=torch.float
        )
        self._progress_buf[env_ids] = torch.zeros(len(env_ids), device=self._device, dtype=torch.float)

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
        """Set actions using position control (PD control)."""
        actions = actions.view(self._num_envs, self._num_actions)
        actions = torch.clip(actions, -self._clip_actions, self._clip_actions)

        # Convert actions to target DOF positions
        target_dof_pos = actions * self._action_scale + self._default_dof_pos

        # Control DOFs using position control (PD control is set in build_scene)
        self._robot.control_dofs_position(target_dof_pos, self._motors_dof_idx)

        # Store actions for observation
        self._actions = actions.clone()

    def _update_buffers(self):
        self._base_pos[:] = self._robot.get_pos()
        self._base_quat[:] = self._robot.get_quat()
        base_quat_rel = transform_quat_by_quat(
            self._base_quat, inv_quat(self._base_init_quat.reshape(1, -1).repeat(self._num_envs, 1))
        )
        self._base_euler = quat_to_xyz(base_quat_rel, rpy=True, degrees=False)

        inv_quat_yaw = axis_angle_to_quat(
            -self._base_euler[:, 2], torch.tensor([0, 0, 1], device=self._device, dtype=torch.float)
        )

        inv_base_quat = inv_quat(self._base_quat)
        self._base_lin_vel[:] = transform_by_quat(self._robot.get_vel(), inv_quat_yaw)
        self._base_ang_vel[:] = transform_by_quat(self._robot.get_ang(), inv_base_quat)
        self._base_lin_vel_world[:] = self._robot.get_vel()
        self._base_ang_vel_world[:] = self._robot.get_ang()
        self._projected_gravity = transform_by_quat(self._global_gravity, inv_base_quat)

        self._dof_pos[:] = self._robot.get_dofs_position(self._motors_dof_idx)
        self._dof_vel[:] = self._robot.get_dofs_velocity(self._motors_dof_idx)
        self._link_contact_forces[:] = torch.tensor(
            self._robot.get_links_net_contact_force(),
            device=self._device,
            dtype=torch.float,
        )
        
        self._foot_positions[:] = self._robot.get_links_pos()[:, self._feet_link_indices, :]
        self._foot_velocities[:] = self._robot.get_links_vel()[:, self._feet_link_indices, :]

    def _get_env_origins(self):
        max_init_level = self._terrain_cfg["max_init_terrain_level"]
        if not self._terrain_cfg["curriculum"]:
            max_init_level = 0 # self._terrain_cfg["num_rows"] - 1
        self._terrain_levels = torch.randint(0, max_init_level + 1, (self._num_envs,), device=self._device)
        num_nominal_envs = self._nominal_env_ids.shape[0]
        self._terrain_types = torch.randint(0, self._terrain_cfg["num_cols"], 
                                            (num_nominal_envs,), device=self._device).repeat_interleave(self._num_envs // num_nominal_envs)
        self._max_terrain_level = self._terrain_cfg["num_rows"]
        self._terrain_origins = torch.from_numpy(self._terrain.env_origins).to(self._device).to(torch.float)
        self._env_origins[:] = self._terrain_origins[self._terrain_levels, self._terrain_types]
        self._custom_origins = True

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int] or torch.Tensor): ids of environments being reset
        """
        # Restrict updates to nominal environments only
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        mask = torch.isin(env_ids, self._nominal_env_ids)
        nominal_env_ids = env_ids[mask]
        if nominal_env_ids.numel() == 0:
            return

        distance = torch.norm(
            self._base_pos[nominal_env_ids, :2] - self._env_origins[nominal_env_ids, :2], dim=1
        )
        max_episode_length_s = self._episode_length * self._dt
        # robots that walked far enough progress to harder terains
        move_up = distance > self._terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            distance
            < torch.norm(self._commands[nominal_env_ids, :2], dim=1) * max_episode_length_s * 0.5
        ) * ~move_up

        self._terrain_levels[nominal_env_ids] += 1 * move_up - 1 * move_down
        self._terrain_levels[nominal_env_ids] = torch.where(
            self._terrain_levels[nominal_env_ids] >= self._max_terrain_level,
            torch.randint_like(self._terrain_levels[nominal_env_ids], self._max_terrain_level),
            torch.clip(self._terrain_levels[nominal_env_ids], 0),
        )  # (the minumum level is zero)
        self._env_origins[nominal_env_ids] = self._terrain_origins[
            self._terrain_levels[nominal_env_ids], self._terrain_types[nominal_env_ids]
        ]

        # Reset auxiliary environments to match their nominal environments.
        # Auxiliary env ids are contiguous: nominal_id + 1 ... nominal_id + num_auxiliary_envs
        num_aux = int(self.num_auxiliary_envs)
        if num_aux > 0:
            offsets = torch.arange(1, num_aux + 1, device=self._device, dtype=torch.long)
            aux_env_ids = nominal_env_ids[:, None] + offsets[None, :]
            aux_env_ids = aux_env_ids.reshape(-1)
            aux_env_ids = aux_env_ids[aux_env_ids < self._num_envs]

            nominal_repeated = nominal_env_ids.repeat_interleave(num_aux)[: aux_env_ids.numel()]
            self._terrain_levels[aux_env_ids] = self._terrain_levels[nominal_repeated]
            self._terrain_types[aux_env_ids] = self._terrain_types[nominal_repeated]
            self._env_origins[aux_env_ids] = self._env_origins[nominal_repeated]
            
    def _init_height_points(self):
        y = torch.tensor(self._terrain_cfg["measured_points_y"], device=self._device, requires_grad=False)
        x = torch.tensor(self._terrain_cfg["measured_points_x"], device=self._device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self._num_height_points = grid_x.numel()
        self._height_points = torch.zeros(
            self._num_envs, self._num_height_points, 3, device=self._device, requires_grad=False
        )
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()

    def _update_surrounding_heights(self):
        points = quat_apply_yaw(self._base_quat.repeat(1, self._num_height_points), self._height_points) + (
            self._base_pos[:, :3]
        ).unsqueeze(1)

        # When acquiring heights, the points need to add border_size
        # because in the height_samples, the origin of the terrain is at (border_size, border_size)
        points += self._terrain_cfg["border_size"]
        points = (points / self._terrain_cfg["horizontal_scale"]).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self._height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self._height_samples.shape[1] - 2)

        heights1 = self._height_samples[px, py]
        heights2 = self._height_samples[px + 1, py]
        heights3 = self._height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self._measured_heights = heights.view(self._num_envs, -1) * self._terrain_cfg["vertical_scale"]

    def _init_domain_params(self):
        """Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._base_com_bias = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._joint_armature = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._joint_friction = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._joint_damping = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._kp_scale = torch.ones(
            self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self._kd_scale = torch.ones(
            self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )

    def _calc_terrain_info_around_feet(self):
        """Finds neighboring points around each foot for terrain height measurement."""
        # Foot positions
        foot_points = self._foot_positions + self._terrain_cfg["border_size"]
        foot_points = (foot_points / self._terrain_cfg["horizontal_scale"]).long()
        # px and py for 4 feet, num_envs*len(feet_indices)
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        # clip to the range of height samples
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)
        # get heights around the feet, 9 points for each foot
        heights1 = self._height_samples[px - 1, py]  # [x-0.1, y]
        heights2 = self._height_samples[px + 1, py]  # [x+0.1, y]
        heights3 = self._height_samples[px, py - 1]  # [x, y-0.1]
        heights4 = self._height_samples[px, py + 1]  # [x, y+0.1]
        heights5 = self._height_samples[px, py]  # [x, y]
        heights6 = self._height_samples[px - 1, py - 1]  # [x-0.1, y-0.1]
        heights7 = self._height_samples[px + 1, py + 1]  # [x+0.1, y+0.1]
        heights8 = self._height_samples[px - 1, py + 1]  # [x-0.1, y+0.1]
        heights9 = self._height_samples[px + 1, py - 1]  # [x+0.1, y-0.1]
        # Calculate normal vectors around feet
        dx = ((heights2 - heights1) / (self._terrain_cfg["horizontal_scale"] * 2)).view(self._num_envs, -1)
        dy = ((heights4 - heights3) / (self._terrain_cfg["horizontal_scale"] * 2)).view(self._num_envs, -1)
        for i in range(len(self._feet_link_indices)):
            normal_vector = torch.cat(
                (dx[:, i].unsqueeze(1), dy[:, i].unsqueeze(1), -1 * torch.ones_like(dx[:, i].unsqueeze(1))), dim=-1
            ).to(self._device)
            normal_vector /= torch.norm(normal_vector, dim=-1, keepdim=True)
            self._normal_vector_around_feet[:, i * 3 : i * 3 + 3] = normal_vector[:]
        # Calculate height around feet
        for i in range(9):
            self._height_around_feet[:, :, i] = (
                eval(f"heights{i+1}").view(self._num_envs, -1)[:] * self._terrain_cfg["vertical_scale"]
            )

        if self._debug:
            self._scene.clear_debug_objects()
            height_points = torch.zeros(self._num_envs, 9 * len(self._feet_link_indices), 3, device=self._device)
            for i in range(len(self._feet_link_indices)):
                height_points[0, i * 9 + 0, 0] = (px - 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 0, 1] = (py - 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 0, 2] = (
                    heights6.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 1, 0] = (px - 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 1, 1] = (
                    py.view(self._num_envs, -1)[0, i] * self._terrain_cfg["horizontal_scale"]
                    - self._terrain_cfg["border_size"]
                )
                height_points[0, i * 9 + 1, 2] = (
                    heights1.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 2, 0] = (
                    px.view(self._num_envs, -1)[0, i] * self._terrain_cfg["horizontal_scale"]
                    - self._terrain_cfg["border_size"]
                )
                height_points[0, i * 9 + 2, 1] = (py - 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 2, 2] = (
                    heights3.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 3, 0] = (
                    px.view(self._num_envs, -1)[0, i] * self._terrain_cfg["horizontal_scale"]
                    - self._terrain_cfg["border_size"]
                )
                height_points[0, i * 9 + 3, 1] = (py + 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 3, 2] = (
                    heights4.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 4, 0] = (
                    px.view(self._num_envs, -1)[0, i] * self._terrain_cfg["horizontal_scale"]
                    - self._terrain_cfg["border_size"]
                )
                height_points[0, i * 9 + 4, 1] = (
                    py.view(self._num_envs, -1)[0, i] * self._terrain_cfg["horizontal_scale"]
                    - self._terrain_cfg["border_size"]
                )
                height_points[0, i * 9 + 4, 2] = (
                    heights5.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 5, 0] = (px + 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 5, 1] = (
                    py.view(self._num_envs, -1)[0, i] * self._terrain_cfg["horizontal_scale"]
                    - self._terrain_cfg["border_size"]
                )
                height_points[0, i * 9 + 5, 2] = (
                    heights2.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 6, 0] = (px + 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 6, 1] = (py + 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 6, 2] = (
                    heights7.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 7, 0] = (px - 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 7, 1] = (py + 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 7, 2] = (
                    heights8.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
                height_points[0, i * 9 + 8, 0] = (px + 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 8, 1] = (py - 1).view(self._num_envs, -1)[0, i] * self._terrain_cfg[
                    "horizontal_scale"
                ] - self._terrain_cfg["border_size"]
                height_points[0, i * 9 + 8, 2] = (
                    heights9.view(self._num_envs, -1)[0, i] * self._terrain_cfg["vertical_scale"]
                )
            self._scene.draw_debug_spheres(height_points[0, :], radius=0.02, color=(1, 0, 0, 0.7))

    def _depth_to_uint8(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_norm = (depth_image - self._camera_near) / max(self._camera_far - self._camera_near, 1e-6)
        depth_norm = depth_norm.clamp(0.0, 1.0)
        return torch.round(depth_norm * 255.0).to(torch.uint8)

    def _process_depth_extreme_parkour(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Match extreme-parkour (IsaacGym) depth preprocessing: crop → noise → clip → bicubic resize → normalize.

        Assumes raw ``depth_image`` from Genesis is **positive** distance in meters with spatial shape
        ``self._depth_res_hw`` (from ``camera.res`` [H, W], e.g. 60×106). Uses ``self._camera_near`` /
        ``self._camera_far`` as ``near_clip`` / ``far_clip``. Output shape: ``(batch, H, W)`` with ``H,W = obs_res``.
        """
        near = self._camera_near
        far = self._camera_far
        dis_noise = float(self._depth_camera_cfg.get("dis_noise", 0.0))
        out_h, out_w = self._depth_out_h, self._depth_out_w

        d = depth_image.float()
        if d.dim() == 2:
            d = d.unsqueeze(0)
        # Same crop as legged_robot.crop_depth_image: drop last 2 rows, 4 cols each side (for 60×106 → 58×98).
        d = d[:, :-2, 4:-4]
        if dis_noise > 0.0:
            d = d + dis_noise * 2.0 * (torch.rand(1, device=d.device, dtype=d.dtype) - 0.5)
        d = torch.clamp(d, min=near, max=far)
        d = d.unsqueeze(1)
        d = F.interpolate(d, size=(out_h, out_w), mode="bicubic", align_corners=False)
        d = d.squeeze(1)
        # Same normalization as extreme-parkour for positive depths: (d - near) / (far - near) - 0.5
        d = (d - near) / max(far - near, 1e-6) - 0.5
        if d.shape[-2] != out_h or d.shape[-1] != out_w:
            raise RuntimeError(
                f"Depth tensor has shape {tuple(d.shape)}; expected last dims (H, W)=({out_h}, {out_w}) "
                f"from camera.res. If you changed res, keep YAML as [height, width] matching DepthCamera read_image."
            )
        return d

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Get robot states for computing observations and rewards."""
        self._update_buffers()

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        robot_states = {
            "base_pos": self._base_pos[env_ids].clone(),
            "base_quat": self._base_quat[env_ids].clone(),
            "base_lin_vel": self._base_lin_vel[env_ids].clone(),
            "base_ang_vel": self._base_ang_vel[env_ids].clone(),
            "base_lin_vel_world": self._base_lin_vel_world[env_ids].clone(),
            "base_ang_vel_world": self._base_ang_vel_world[env_ids].clone(),
            "projected_gravity": self._projected_gravity[env_ids].clone(),
            "motor_joints_pos": self._dof_pos[env_ids].clone(),
            "motor_joints_vel": self._dof_vel[env_ids].clone(),
            "last_foot_vel": self._last_foot_vel[env_ids].clone(),
            "last_actions": self._last_actions[env_ids].clone(),
            "last_last_actions": self._last_last_actions[env_ids].clone(),
            "last_contacts": self._last_contacts[env_ids].clone(),
            "last_dof_vel": self._last_dof_vel[env_ids].clone(),
            "commands": self._commands[env_ids].clone(),
        }

        states = {
            "robot_states": robot_states,
            "progress_buf": self._progress_buf[env_ids].clone(),
            "obs_history_buf": self._obs_history_buf[env_ids].clone(),
            "privileged_obs_buf": self._privileged_obs_buf[env_ids].clone(),
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
        base_dof_vel = torch.cat([robot_states["base_lin_vel_world"], robot_states["base_ang_vel_world"]], dim=-1)
        self._robot.set_dofs_velocity(
            velocity=torch.cat([base_dof_vel, robot_states["motor_joints_vel"]], dim=-1),
            dofs_idx_local=self._base_dof_idx + self._motors_dof_idx,
            envs_idx=env_ids,
        )

        # Update progress buffer
        self._progress_buf[env_ids] = states["progress_buf"].clone()

        # Update observation history buffer
        self._obs_history_buf[env_ids] = states["obs_history_buf"].clone()

        # Update privileged observation buffer
        # NOTE: This significantly affects the rewards when not commented out
        self._privileged_obs_buf[env_ids] = states["privileged_obs_buf"].clone()

        # Update previous actions if provided
        self._last_actions[env_ids] = robot_states["last_actions"].clone()
        self._last_last_actions[env_ids] = robot_states["last_last_actions"].clone()
        self._last_dof_vel[env_ids] = robot_states["last_dof_vel"].clone()
        self._projected_gravity[env_ids] = robot_states["projected_gravity"].clone()
        self._last_foot_vel[env_ids] = robot_states["last_foot_vel"].clone()
        self._last_contacts[env_ids] = robot_states["last_contacts"].clone()
        self._commands[env_ids] = robot_states["commands"].clone()

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        """Tracking of linear velocity commands (xy axes)."""
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self._reward_tracking_sigma)

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        """Tracking of angular velocity commands (yaw)."""
        ang_vel_error = torch.square(self._commands[:, 2] - self._base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self._reward_tracking_sigma)

    def _reward_lin_vel_z(self) -> torch.Tensor:
        """Penalize z axis base linear velocity."""
        return torch.square(self._base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        """Penalize xy axis base angular velocity."""
        return torch.sum(torch.square(self._base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self) -> torch.Tensor:
        """Penalize projected gravity."""
        return torch.sum(torch.square(self._projected_gravity[:, :2]), dim=1)

    def _reward_torques(self) -> torch.Tensor:
        # Penalize torques
        return torch.sum(torch.square(self._robot.get_dofs_control_force(self._motors_dof_idx)), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        """Penalize DOF velocities while standing still."""
        return torch.sum(torch.square(self._dof_vel), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        """Penalize DOF accelerations."""
        return torch.sum(torch.square((self._dof_vel - self._last_dof_vel) / self._dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self._last_actions - self._actions), dim=1)

    def _reward_base_height(self) -> torch.Tensor:
        """Penalize base height away from target."""
        return torch.square(self._base_pos[:, 2] - self._reward_base_height_target)

    def _reward_collision(self) -> torch.Tensor:
        """Penalize collisions."""
        return torch.sum(
            1.0 * (torch.norm(self._link_contact_forces[:, self._penalized_contact_link_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_dof_pos_limits(self) -> torch.Tensor:
        # Penalize dof positions too close to the limit
        out_of_limits = -(self._dof_pos - self._dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self._dof_pos - self._dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self) -> torch.Tensor:
        # Reward long steps
        contact = self._link_contact_forces[:, self._feet_link_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.0) * contact_filt
        self._feet_air_time += self._dt
        rew_airTime = torch.sum(
            (self._feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self._commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self._feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self._foot_velocities[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm
            * torch.square(self._foot_positions[:, :, 2] -
                torch.mean(self._height_around_feet, dim=-1) -  
                self._foot_clearance_target - 
                self._foot_height_offset),
                dim=-1,
        )
        return torch.exp(-clearance_error / self._foot_clearance_tracking_sigma)

    def _reward_action_smoothness(self):
        """Penalize action smoothness"""
        action_smoothness_cost = torch.sum(
            torch.square(self._actions - 2 * self._last_actions + self._last_last_actions), dim=-1
        )
        return action_smoothness_cost

    def _reward_stand_still(self) -> torch.Tensor:
        cmd_norm = torch.norm(self._commands, dim=1)
        return torch.sum(torch.square(self._dof_pos - self._default_dof_pos), dim=1) * (cmd_norm < 0.01)

    def _reward_feet_contact_stand_still(self):
        # Encourage feet contact with the ground at zero commands
        contacts = self._link_contact_forces[:, self._feet_link_indices, 2] > 1.0
        full_contact = torch.sum(1.0 * contacts, dim=1) == len(self._feet_link_indices)
        return 1.0 * full_contact * (torch.norm(self._commands[:, :3], dim=1) < 0.01)

    def _reward_feet_distance(self):
        cur_footsteps_translated = self._foot_positions - self._base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self._num_envs, 4, 3, device=self._device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(inv_quat(self._base_quat), cur_footsteps_translated[:, i, :])

        stance_width = 0.3 * torch.ones(
            [
                self._num_envs,
                1,
            ],
            device=self._device,
        )
        desired_ys = torch.cat([-stance_width / 2, stance_width / 2, -stance_width / 2, stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)

        return stance_diff

    def _reward_hip_pos(self):
        """Reward for the hip joint position close to default position"""
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(
            torch.square(self._dof_pos[:, hip_joint_indices] - self._default_dof_pos[hip_joint_indices]), dim=-1
        )
        return dof_pos_error

    def _randomize_rigids(self, env_ids=None):
        if not self._train:
            return

        if env_ids is None:
            env_ids = torch.arange(0, self._num_envs)
        elif len(env_ids) == 0:
            return

        if self._domain_rand_options["randomize_friction"]:
            self._randomize_link_friction(env_ids)
        if self._domain_rand_options["randomize_base_mass"]:
            self._randomize_base_mass(env_ids)
        if self._domain_rand_options["randomize_com_displacement"]:
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids=None):
        if not self._train:
            return

        if env_ids is None:
            env_ids = torch.arange(0, self._num_envs)
        elif len(env_ids) == 0:
            return

        if self._domain_rand_options["randomize_motor_strength"]:
            self._randomize_motor_strength(env_ids)
        if self._domain_rand_options["randomize_motor_offset"]:
            self._randomize_motor_offset(env_ids)
        if self._domain_rand_options["randomize_kp_scale"]:
            self._randomize_kp(env_ids)
        if self._domain_rand_options["randomize_kd_scale"]:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):
        min_friction, max_friction = self._domain_rand_options["friction_range"]
        # ratios = (
        #     gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction)
        #     + min_friction
        # )
        ratios = (
            gs.rand((len(env_ids), 1), dtype=float).repeat(1, self._robot.n_links) * (max_friction - min_friction)
            + min_friction
        )
        self._friction_values[env_ids] = ratios[:, 0].unsqueeze(1).detach().clone()
        # solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)
        self._robot.set_friction_ratio(ratios, torch.arange(0, self._robot.n_links), env_ids)

    def _randomize_base_mass(self, env_ids):
        min_mass, max_mass = self._domain_rand_options["added_mass_range"]
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self._robot.set_mass_shift(added_mass, self._base_link_index, env_ids)

    def _randomize_com_displacement(self, env_ids):
        min_displacement, max_displacement = self._domain_rand_options["com_displacement_range"]
        com_displacement = (
            gs.rand((len(env_ids), 1, 3), dtype=float) * (max_displacement - min_displacement) + min_displacement
        )

        self._base_com_bias[env_ids] = com_displacement[:, 0, :].detach().clone()
        self._robot.set_COM_shift(com_displacement, self._base_link_index, env_ids)

    def _push_robots(self):
        max_push_vel_xy = self._domain_rand_options["max_push_vel_xy"]
        # in Genesis, base link also has DOF, it's 6DOF if not fixed.
        dofs_vel = self._robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = torch_rand_float(-max_push_vel_xy, max_push_vel_xy, (self._num_envs, 2), self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        dofs_vel[:, :2] += push_vel
        self._robot.set_dofs_velocity(dofs_vel)

    def _randomize_motor_strength(self, env_ids):
        min_strength, max_strength = self._domain_rand_options["motor_strength_range"]
        self._motor_strengths[env_ids, :] = (
            gs.rand((len(env_ids), 1), dtype=float) * (max_strength - min_strength) + min_strength
        )

    def _randomize_motor_offset(self, env_ids):
        min_offset, max_offset = self._domain_rand_options["motor_offset_range"]
        self._motor_offsets[env_ids, :] = (
            gs.rand((len(env_ids), 12), dtype=float) * (max_offset - min_offset) + min_offset
        )

    def _randomize_kp(self, env_ids):
        min_scale, max_scale = self._domain_rand_options["kp_scale_range"]
        kp_scales = gs.rand((len(env_ids), 12), dtype=float) * (max_scale - min_scale) + min_scale
        self._kp_scale[env_ids, :] = kp_scales.detach().clone()
        self._batched_p_gains[env_ids, :] = kp_scales * self._p_gains[None, :]

    def _randomize_kd(self, env_ids):
        min_scale, max_scale = self._domain_rand_options["kd_scale_range"]
        kd_scales = gs.rand((len(env_ids), 12), dtype=float) * (max_scale - min_scale) + min_scale
        self._kd_scale[env_ids, :] = kd_scales.detach().clone()
        self._batched_d_gains[env_ids, :] = kd_scales * self._d_gains[None, :]

    def _prepare_obs_noise(self):
        self._obs_noise[:3] = self._obs_noise_cfg["ang_vel"]
        self._obs_noise[3:6] = self._obs_noise_cfg["gravity"]
        self._obs_noise[21:33] = self._obs_noise_cfg["dof_pos"]
        self._obs_noise[33:45] = self._obs_noise_cfg["dof_vel"]

    def _set_camera(self):
        """Set camera position and direction"""
        self._floating_camera = self._scene.add_camera(
            pos=np.array([0, -1, 1]),
            lookat=np.array([0, 0, 0]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )

        self._recording = False
        self._recorded_frames = []

    def _render_headless(self):
        if self._recording and len(self._recorded_frames) < 150:
            robot_pos = np.array(self._robot.get_pos().cpu())
            # Camera expects single position (3,) - use env 0 when multiple envs
            pos = robot_pos[0] if robot_pos.ndim > 1 else robot_pos
            self._floating_camera.set_pose(pos=pos + np.array([-1, -1, 0.5]), lookat=pos + np.array([0, 0, -0.1]))
            # import time
            # start = time.time()
            frame, _, _, _ = self._floating_camera.render()
            # end = time.time()
            # print(end-start)
            self._recorded_frames.append(frame)
            # from PIL import Image
            # img = Image.fromarray(np.uint8(frame))
            # img.save('./test.png')
            # print('save')

    def get_recorded_frames(self):
        if len(self._recorded_frames) == 150:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=True):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self._floating_camera.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self._dt))
            self._floating_camera.stop_recording(save_path, fps=int(1 / self._dt))

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        self._gs_terrain = self._scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self._terrain_cfg["border_size"], -self._terrain_cfg["border_size"], 0.0),
                horizontal_scale=self._terrain_cfg["horizontal_scale"],
                vertical_scale=self._terrain_cfg["vertical_scale"],
                height_field=self._terrain.height_field_raw,
            ),
        )
        self._height_samples = (
            torch.tensor(self._terrain.heightsamples)
            .view(self._terrain.tot_rows, self._terrain.tot_cols)
            .to(self._device)
        )

    @property
    def num_privileged_obs(self) -> int:
        return self._num_privileged_obs

    @property
    def num_single_obs(self) -> int:
        return self._num_single_obs

    @property
    def num_obs(self) -> int:
        return self._num_obs

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._progress_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self._progress_buf.copy_(value.to(device=self._device, dtype=self._progress_buf.dtype))

    @property
    def max_episode_length(self) -> int:
        return self._episode_length

    @property
    def dt(self) -> float:
        return self._dt
