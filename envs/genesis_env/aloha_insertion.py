import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import (
    inv_quat,
    pos_lookat_up_to_T,
    transform_quat_by_quat,
)
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class AlohaInsertion(GenesisEnv):
    """Aloha insertion environment."""

    _num_actions = 14
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,))

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
        dt = sim_options.dt
        episode_length = int(5.0 / dt)

        early_termination = True

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
            vis_options=vis_options,
            show_FPS=show_FPS,
        )

    def init_scene(self) -> None:
        """Initialize the scene."""

        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/aloha/aloha.xml"),),
        )

        self._table = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/aloha/table.xml"),),
        )

        self._peg = self._scene.add_entity(
        gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/aloha/peg.xml")),
        )

        self._socket = self._scene.add_entity(
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/aloha/socket.xml")),
        )

        self._floor = self._scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.75)))

        if self._vis_obs:

            # if sensors_args is None:
            #     sensors_args = {
            #         "camera": {
            #             "res": [256, 256],
            #             "pos": [0.40, 0.05, 0.425],
            #             "lookat": [0.25, -0.10, 0.275],
            #             "fov": 80.0,
            #             "lights": {
            #                 "pos": [0.0, 0.0, 2.0],
            #                 "dir": [0.0, 0.0, -1.0],
            #                 "intensity": 0.8,
            #                 "color": [1.0, 1.0, 1.0],
            #             },
            #             "directional": True,
            #             "castshadow": False,
            #         }
            #     }
            # self._num_image_stack = 3
            # self._observation_space = spaces.Dict(
            #     {
            #         "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(124,)),
            #         # observation that ignores cube information (infer from images)
            #         "proprioception_and_target": spaces.Box(low=-np.inf, high=np.inf, shape=(107,)),
            #         "RGB": spaces.Box(
            #             low=0,
            #             high=255,
            #             dtype=np.uint8,
            #             shape=(
            #                 self._num_image_stack * 3,
            #                 sensors_args["camera"]["res"][0],
            #                 sensors_args["camera"]["res"][1],
            #             ),
            #         ),
            #     }
            # )
            pass
        else:
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(82,)),
                }
            )

        self._arm_joint_names = [
            "left/waist",
            "left/shoulder",
            "left/elbow",
            "left/forearm_roll",
            "left/wrist_angle",
            "left/wrist_rotate",
            "right/waist",
            "right/shoulder",
            "right/elbow",
            "right/forearm_roll",
            "right/wrist_angle",
            "right/wrist_rotate",
        ]

        self._gripper_active_joint_names = [
            "left/left_finger",
            "right/left_finger",
        ]
        self._gripper_passive_joint_names = [
            "left/right_finger",
            "right/right_finger",
        ]
        self._gripper_joint_names = (
            self._gripper_active_joint_names + self._gripper_passive_joint_names
        )

        self._motor_joint_names = (
            self._arm_joint_names + self._gripper_active_joint_names
        )
        self._robot_joint_names = self._arm_joint_names + self._gripper_joint_names

        self._arm_dofs_idx = [
            self._robot.get_joint(name).dof_start for name in self._arm_joint_names
        ]
        self._gripper_active_dofs_idx = [
            self._robot.get_joint(name).dof_start
            for name in self._gripper_active_joint_names
        ]
        self._gripper_passive_dofs_idx = [
            self._robot.get_joint(name).dof_start
            for name in self._gripper_passive_joint_names
        ]
        self._motors_dof_idx = self._arm_dofs_idx + self._gripper_active_dofs_idx

        self._robot_dofs_idx = (
            self._arm_dofs_idx
            + self._gripper_active_dofs_idx
            + self._gripper_passive_dofs_idx
        )

        _kp_per_arm = [43.0, 265.0, 227.0, 78.0, 37.0, 10.4]
        _kd_per_arm = [5.76, 20.0, 18.49, 6.78, 6.28, 1.2]
        _kp_gripper = 365.0
        _kd_gripper = 40.0
        self._kp = torch.tensor(
            _kp_per_arm + _kp_per_arm + [_kp_gripper, _kp_gripper],
            device=self._device,
        )
        self._kd = torch.tensor(
            _kd_per_arm + _kd_per_arm + [_kd_gripper, _kd_gripper],
            device=self._device,
        )
        
        self._peg_dofs_idx = self._peg.get_joint("peg").dofs_idx_local

        self._socket_dofs_idx = self._socket.get_joint("socket").dofs_idx_local

        self._default_motor_dof_pos = torch.tensor(
            [
                # left arm: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
                0.083383, -0.122008,  0.950168,  0.108187, -0.869224, -0.0731298,
                # right arm
                -0.0862348, -0.109522,  0.949474, -0.113041, -0.887378,  0.0754333,
                # grippers (finger opening, meters). 0.0305 ~ open, 0.002 closed.
                0.0305,  0.0186,
            ],
            device=self._device,
        ).repeat(self._num_envs, 1)

        self._ctrl = self._default_motor_dof_pos.clone()

        self._default_peg_pos = torch.tensor(
            [0.136459, 0.0, 0.0107945], device=self._device
        ).repeat(self._num_envs, 1)
        self._default_peg_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self._device
        ).repeat(self._num_envs, 1)

        self._default_socket_pos = torch.tensor(
            [-0.146984, 0.0, 0.0227945], device=self._device
        ).repeat(self._num_envs, 1)
        self._default_socket_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self._device
        ).repeat(self._num_envs, 1)

        self._reset_xy_noise = 0.1

        self._action_scale = 0.005

        self._socket_entrance_goal_pos = torch.tensor(
            [-0.05, 0.0, 0.15], device=self._device
        ).repeat(self._num_envs, 1)
        self._peg_end2_goal_pos = torch.tensor(
            [0.05, 0.0, 0.15], device=self._device
        ).repeat(self._num_envs, 1)

        # if self._vis_obs:
        #     # Initialize the sensors
        #     # TODO: genesis at commit id 7db43e4caef2b185bf691d29fc545d6480cd224d only supports offset_T
        #     offset_T = self._sensors_args["camera"].get("offset_T", None)
        #     lookat = self._sensors_args["camera"].get("lookat", None)
        #     if offset_T is not None:
        #         offset_T = torch.tensor(offset_T, device=self._device)
        #     else:
        #         if lookat is not None:
        #             offset_T = pos_lookat_up_to_T(
        #                 np.array(self._sensors_args["camera"]["pos"]), np.array(lookat), np.array((0.0, 0.0, 1.0))
        #             )
        #         else:
        #             offset_T = np.eye(4)
        #     # NOTE: A dummy link for the camera to attach to; without entity_idx the batch renderer
        #     # uses a single world pose, only env 0 is in view and other envs render black.
        #     self._camera_mount = self._scene.add_entity(gs.morphs.Sphere(radius=0.01, collision=False, fixed=True))
        #     self._camera = self._scene.add_sensor(
        #         gs.sensors.BatchRendererCameraOptions(
        #             res=self._sensors_args["camera"]["res"],
        #             pos=self._sensors_args["camera"]["pos"],
        #             offset_T=offset_T,
        #             fov=self._sensors_args["camera"]["fov"],
        #             entity_idx=self._camera_mount.idx,
        #             lights=[self._sensors_args["camera"]["lights"]],
        #             env_idx=self._nominal_env_ids.cpu().tolist(),
        #         )
        #     )

        #     self._imgs_buf = torch.zeros(
        #         self.nominal_env_ids.shape[0],
        #         self._num_image_stack,
        #         self._sensors_args["camera"]["res"][0],
        #         self._sensors_args["camera"]["res"][1],
        #         3,
        #         device=self._device,
        #         dtype=torch.uint8,
        #     )

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(1.5, 1.5))

        self._robot.set_dofs_kp(self._kp, self._motors_dof_idx)
        self._robot.set_dofs_kv(self._kd, self._motors_dof_idx)
        self._robot.set_dofs_damping(
            torch.zeros_like(self._kd), self._motors_dof_idx
        )

        self._ctrl_lower, self._ctrl_upper = self._robot.get_dofs_limit(
            dofs_idx_local=self._motors_dof_idx
        )

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        pass
        # observations = {}

        # # Privileged observations follow the IsaacLab's implementation
        # robot_states = states["robot_states"]
        # hand_dof_pos = robot_states["hand_dof_pos"]
        # scaled_hand_dof_pos = (2.0 * hand_dof_pos - self._hand_motors_ctrl_lower - self._hand_motors_ctrl_upper) / (
        #     self._hand_motors_ctrl_upper - self._hand_motors_ctrl_lower
        # )
        # hand_dof_vel = robot_states["hand_dof_vel"]
        # scaled_hand_dof_vel = hand_dof_vel * self._vel_obs_scale

        # cube_pos = robot_states["cube_pos"]
        # cube_quat = robot_states["cube_quat"]
        # cube_vel = robot_states["cube_vel"]
        # cube_linear_vel = cube_vel[:, :3]
        # cube_angular_vel = cube_vel[:, 3:]
        # scaled_cube_angular_vel = cube_angular_vel * self._vel_obs_scale

        # target_quat = robot_states["target_quat"]
        # rot_diff = transform_quat_by_quat(inv_quat(target_quat), cube_quat)

        # prev_actions = robot_states["prev_actions"]
        # # TODO, in IsaacLab, the observation contains figer tip pose and velocity;
        # # Figer tip is not part of the state, but can be derived from the state
        # # We currently obtain the finger tip pose directly from simulation, which may be problematic
        # # However, since compute_observations is called after set_states, this implementation may be fine for now.
        # finger_tip_pos = self._robot.get_links_pos(links_idx_local=self._finger_tip_link_idx).view(
        #     self.num_envs, len(self._finger_tip_link_names) * 3
        # )
        # finger_tip_quat = self._robot.get_links_quat(links_idx_local=self._finger_tip_link_idx).view(
        #     self.num_envs, len(self._finger_tip_link_names) * 4
        # )
        # finger_tip_vel = self._robot.get_links_vel(links_idx_local=self._finger_tip_link_idx).view(
        #     self.num_envs, len(self._finger_tip_link_names) * 3
        # )
        # finger_tip_angular_vel = self._robot.get_links_ang(links_idx_local=self._finger_tip_link_idx).view(
        #     self.num_envs, len(self._finger_tip_link_names) * 3
        # )

        # privileged_observations = torch.cat(
        #     [
        #         # hand
        #         scaled_hand_dof_pos,
        #         scaled_hand_dof_vel,
        #         # object
        #         cube_pos,
        #         cube_quat,
        #         cube_linear_vel,
        #         scaled_cube_angular_vel,
        #         # goal
        #         self._in_hand_pos,
        #         target_quat,
        #         rot_diff,
        #         # finger_tip
        #         finger_tip_pos,
        #         finger_tip_quat,
        #         finger_tip_vel,
        #         finger_tip_angular_vel,
        #         # actions
        #         prev_actions,
        #     ],
        #     dim=-1,
        # )
        # observations["privileged_observations"] = privileged_observations

        # if self._vis_obs:
        #     proprioception_and_target = torch.cat(
        #         [
        #             # hand
        #             scaled_hand_dof_pos,
        #             scaled_hand_dof_vel,
        #             # goal
        #             self._in_hand_pos,
        #             target_quat,
        #             # finger_tip
        #             finger_tip_pos,
        #             finger_tip_quat,
        #             finger_tip_vel,
        #             finger_tip_angular_vel,
        #             # actions
        #             prev_actions,
        #         ],
        #         dim=-1,
        #     )

        #     observations["proprioception_and_target"] = proprioception_and_target

        #     batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
        #     # NOTE: for AFRL agent, RGB observation and privileged observations may has different shapes
        #     # Reshape: (batch, num_stack, H, W, 3) -> (batch, num_stack * 3, H, W)
        #     observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
        #         batch_size, num_stack * rgb, img_height, img_width
        #     )

        # return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # TODO
        return torch.zeros(self._num_envs, device=self._device)
        # robot_states = states["robot_states"]
        # cube_pos = robot_states["cube_pos"]
        # cube_quat = robot_states["cube_quat"]
        # target_quat = robot_states["target_quat"]
        # goal_dis = torch.norm(cube_pos - self._in_hand_pos, p=2, dim=-1)
        # dist_reward = self._dist_reward_scale * goal_dis

        # quat_diff = transform_quat_by_quat(inv_quat(target_quat), cube_quat)
        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

        # rot_rew = -(rot_dist**2) * self._rot_reward_scale

        # action_penalty = self._action_penalty * torch.sum(actions**2, dim=-1)

        # # restore the average angle difference between the cube and the target in degrees
        # self._infos["angle_diff"] = torch.rad2deg(2 * torch.norm(quat_diff[:, 1:4], p=2, dim=-1)).mean().item()

        # return dist_reward + rot_rew + action_penalty + self._healthy_reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            # TODO
            pass
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        motor_dof_pos = self._default_motor_dof_pos[env_ids]

        peg_pos = self._default_peg_pos[env_ids].clone()
        peg_quat = self._default_peg_quat[env_ids]
        socket_pos = self._default_socket_pos[env_ids].clone()
        socket_quat = self._default_socket_quat[env_ids]

        if self._randomize_init:
            peg_pos[:, :2] += (
                (torch.rand(n, 2, device=self._device) * 2.0 - 1.0)
                * self._reset_xy_noise
            )
            socket_pos[:, :2] += (
                (torch.rand(n, 2, device=self._device) * 2.0 - 1.0)
                * self._reset_xy_noise
            )

        robot_dof_pos = torch.cat(
            [
                motor_dof_pos,                       # (n, 14)
                motor_dof_pos[:, 12:14],             # passive finger mirrors active finger
            ],
            dim=-1,
        )  # -> (n, 16), in the same joint order as self._robot_dofs_idx
        self._robot.set_dofs_position(
            position=robot_dof_pos,
            dofs_idx_local=self._robot_dofs_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )


        self._ctrl[env_ids] = motor_dof_pos
        self._robot.control_dofs_position(
            position=motor_dof_pos,
            dofs_idx_local=self._motors_dof_idx,
            envs_idx=env_ids,
        )

        self._peg.set_pos(peg_pos, envs_idx=env_ids, zero_velocity=True)
        self._peg.set_quat(peg_quat, envs_idx=env_ids, zero_velocity=True)
        self._socket.set_pos(socket_pos, envs_idx=env_ids, zero_velocity=True)
        self._socket.set_quat(socket_quat, envs_idx=env_ids, zero_velocity=True)

        # if self._vis_obs:
        #     # Find which nominal environments are being reset
        #     # self.nominal_env_ids contains the global env_ids of nominal environments
        #     # We need to find the indices within nominal_env_ids that match env_ids
        #     mask = torch.isin(self.nominal_env_ids, env_ids)
        #     nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]

        #     if len(nominal_idx_to_reset) > 0:
        #         # Render fresh images for the reset nominal environments
        #         reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
        #         new_img = self.render(env_ids=reset_nominal_env_ids)

        #         # Initialize the image buffer for these environments
        #         self._imgs_buf[nominal_idx_to_reset] = new_img.unsqueeze(1)

    def _set_actions(self, actions: torch.Tensor) -> None:
        actions = actions.view(self._num_envs, self._num_actions)
        actions = actions.clamp(min=-1.0, max=1.0)

        self._ctrl = torch.clamp(
            self._ctrl + actions * self._action_scale,
            self._ctrl_lower,
            self._ctrl_upper,
        )
        self._robot.control_dofs_position(
            self._ctrl,
            dofs_idx_local=self._motors_dof_idx,
        )

    def _post_physics_step(self) -> None:
        # TODO:
        if self._vis_obs:
            pass
            # new_img = self.render(env_ids=self.nominal_env_ids)
            # # Roll the buffer to shift old frames: [t-2, t-1, t-0] -> [t-1, t-0, None]
            # # This moves older frames "to the left" and makes room for the new frame
            # self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            # self._imgs_buf[:, -1] = new_img

    def render(self, env_ids: Optional[Sequence[int]] = None) -> Optional[torch.Tensor]:
        pass
        # if self._vis_obs:
        #     if env_ids is None:
        #         env_ids = self.nominal_env_ids

        #     # TODO: genesis will refresh the image when the scene._dt is different from the last render time
        #     # TODO: temporarily we hack by setting the last render time to 0 to force render the new image
        #     self._camera._shared_metadata.last_render_timestep = 0
        #     data = self._camera.read(envs_idx=env_ids)
        #     return data.rgb
        # else:
        #     return None

    def get_states(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        robot_pos = self._robot.get_dofs_position(self._robot_dofs_idx, envs_idx=env_ids)
        robot_vel = self._robot.get_dofs_velocity(self._robot_dofs_idx, envs_idx=env_ids)
        
        peg_pos = self._peg.get_pos(envs_idx=env_ids)
        peg_quat = self._peg.get_quat(envs_idx=env_ids)
        peg_vel = self._peg.get_dofs_velocity(self._peg_dofs_idx, envs_idx=env_ids)
        
        socket_pos = self._socket.get_pos(envs_idx=env_ids)
        socket_quat = self._socket.get_quat(envs_idx=env_ids)
        socket_vel = self._socket.get_dofs_velocity(self._socket_dofs_idx, envs_idx=env_ids)

        robot_states = {
            "robot_pos": robot_pos.clone(),
            "robot_vel": robot_vel.clone(),
            "peg_pos": peg_pos.clone(),
            "peg_quat": peg_quat.clone(),
            "peg_vel": peg_vel.clone(),
            "socket_pos": socket_pos.clone(),
            "socket_quat": socket_quat.clone(),
            "socket_vel": socket_vel.clone(),
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
            position=robot_states["robot_pos"],
            dofs_idx_local=self._robot_dofs_idx,
            envs_idx=env_ids,
        )

        self._robot.set_dofs_velocity(
            velocity=robot_states["robot_vel"],
            dofs_idx_local=self._robot_dofs_idx,
            envs_idx=env_ids,
        )

        self._peg.set_pos(robot_states["peg_pos"], envs_idx=env_ids)
        self._peg.set_quat(robot_states["peg_quat"], envs_idx=env_ids)
        self._peg.set_dofs_velocity(robot_states["peg_vel"], envs_idx=env_ids, dofs_idx_local=self._peg_dofs_idx)

        self._socket.set_pos(robot_states["socket_pos"], envs_idx=env_ids)
        self._socket.set_quat(robot_states["socket_quat"], envs_idx=env_ids)
        self._socket.set_dofs_velocity(robot_states["socket_vel"], envs_idx=env_ids, dofs_idx_local=self._socket_dofs_idx)

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
