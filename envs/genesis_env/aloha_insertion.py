import math
import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import (
    inv_quat,
    inv_transform_by_quat,
    pos_lookat_up_to_T,
    transform_quat_by_quat,
)
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


def _tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """Torch port of `mujoco_playground._src.reward.tolerance`.

    Returns 1.0 inside `bounds`, smoothly decays to `value_at_margin` at
    distance `margin` from the nearest bound, then (for `"linear"`) keeps
    decaying linearly to 0. Only the sigmoid kinds actually used by
    SinglePegInsertion are implemented (`"linear"`, `"gaussian"`).
    """
    lower, upper = bounds
    in_bounds = (x >= lower) & (x <= upper)

    if margin == 0.0:
        return in_bounds.to(x.dtype)

    d = torch.where(x < lower, lower - x, x - upper) / margin

    if sigmoid == "linear":
        scale = 1.0 - value_at_margin
        value = torch.clamp(1.0 - d * scale, min=0.0)
    elif sigmoid == "gaussian":
        scale = math.sqrt(-2.0 * math.log(value_at_margin))
        value = torch.exp(-0.5 * (d * scale) ** 2)
    else:
        raise ValueError(f"Unsupported sigmoid: {sigmoid!r}")

    return torch.where(in_bounds, torch.ones_like(x), value)


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
        episode_length = int(2.5 / dt)

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

        self._left_gripper_site_link = self._robot.get_link("left/gripper_site")
        self._right_gripper_site_link = self._robot.get_link("right/gripper_site")
        self._peg_end1_site_link = self._peg.get_link("peg_end1_site")
        self._peg_end2_site_link = self._peg.get_link("peg_end2_site")
        self._socket_entrance_site_link = self._socket.get_link("socket_entrance_site")
        self._socket_rear_site_link = self._socket.get_link("socket_rear_site")

        self._finger_link_names = [
            "left/left_finger_link",
            "left/right_finger_link",
            "right/left_finger_link",
            "right/right_finger_link",
        ]
        self._finger_link_global_indices = torch.tensor(
            [self._robot.get_link(n).idx for n in self._finger_link_names],
            device=self._device,
            dtype=torch.long,
        )

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

        self._out_of_bounds_threshold = 1.0

        self._socket_entrance_goal_pos = torch.tensor(
            [-0.05, 0.0, 0.15], device=self._device
        ).repeat(self._num_envs, 1)
        self._peg_end2_goal_pos = torch.tensor(
            [0.05, 0.0, 0.15], device=self._device
        ).repeat(self._num_envs, 1)

        self._reward_scales = {
            "left_reward": 1.0,
            "right_reward": 1.0,
            "left_target_qpos": 0.3,
            "right_target_qpos": 0.3,
            "no_table_collision": 0.3,
            "socket_z_up": 0.5,
            "peg_z_up": 0.5,
            "socket_entrance_reward": 4.0,
            "peg_end2_reward": 4.0,
            "peg_insertion_reward": 8.0,
        }
        self._reward_scale_sum = float(sum(self._reward_scales.values()))

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
        robot_states = states["robot_states"]
        batch = robot_states["robot_pos"].shape[0]

        left_gripper_pos    = self._left_gripper_site_link.get_pos()
        right_gripper_pos   = self._right_gripper_site_link.get_pos()
        socket_entrance_pos = self._socket_entrance_site_link.get_pos()
        peg_end2_pos        = self._peg_end2_site_link.get_pos()

        # TODO, duplicate information, but follow mujoco_playground implementation
        peg_pos    = robot_states["peg_pos"]
        socket_pos = robot_states["socket_pos"]

        world_z = torch.tensor([0.0, 0.0, 1.0], device=self._device).expand(batch, 3)
        peg_z    = inv_transform_by_quat(world_z, robot_states["peg_quat"])
        socket_z = inv_transform_by_quat(world_z, robot_states["socket_quat"])

        privileged_observations = torch.cat(
            [
                robot_states["robot_pos"],          # (N, 16)
                robot_states["peg_pos"],            # (N, 3)
                robot_states["peg_quat"],           # (N, 4)
                robot_states["socket_pos"],         # (N, 3)
                robot_states["socket_quat"],        # (N, 4)
                robot_states["robot_vel"],          # (N, 16)
                robot_states["peg_vel"],            # (N, 6)
                robot_states["socket_vel"],         # (N, 6)
                left_gripper_pos,
                socket_pos,
                right_gripper_pos,
                peg_pos,
                socket_entrance_pos,
                peg_end2_pos,
                socket_z,
                peg_z,
            ],
            dim=-1,
        )  # (N, 82)

        observations = {
            "privileged_observations": privileged_observations,
        }

        # TODO: add RGB / wrist-cam observations here when self._vis_obs is
        # enabled, to mirror the rest of the genesis_env codebase.
        return observations

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
        robot_states = states["robot_states"]
        n = robot_states["robot_pos"].shape[0]

        # ---- world-frame positions (via dummy-body links) -------------
        left_gripper_pos    = self._left_gripper_site_link.get_pos()
        right_gripper_pos   = self._right_gripper_site_link.get_pos()
        socket_entrance_pos = self._socket_entrance_site_link.get_pos()
        socket_rear_pos     = self._socket_rear_site_link.get_pos()
        peg_end2_pos        = self._peg_end2_site_link.get_pos()

        socket_pos = robot_states["socket_pos"]
        peg_pos    = robot_states["peg_pos"]

        # ---- (1) / (2) grip rewards -----------------------------------
        left_socket_dist = torch.norm(socket_pos - left_gripper_pos, dim=-1)
        left_reward = _tolerance(
            left_socket_dist, (0.0, 0.001), margin=0.3, sigmoid="linear"
        )

        right_peg_dist = torch.norm(peg_pos - right_gripper_pos, dim=-1)
        right_reward = _tolerance(
            right_peg_dist, (0.0, 0.001), margin=0.3, sigmoid="linear"
        )

        # ---- (3) arm-qpos-close-to-init -------------------------------
        # robot_pos order is [arm_12, gripper_active_2, gripper_passive_2];
        # _default_motor_dof_pos order is [arm_12, gripper_active_2]. The
        # first 12 slots line up, so we slice them directly.
        arm_qpos = robot_states["robot_pos"][:, :12]
        init_arm_qpos = self._default_motor_dof_pos[:, :12]
        robot_qpos_diff = arm_qpos - init_arm_qpos
        left_pose  = _tolerance(
            torch.norm(robot_qpos_diff[:, :6],  dim=-1), (0.0, 0.01), margin=2.0
        )  # default sigmoid is "gaussian", matching the reference
        right_pose = _tolerance(
            torch.norm(robot_qpos_diff[:, 6:12], dim=-1), (0.0, 0.01), margin=2.0
        )

        # ---- (4) lift rewards -----------------------------------------
        socket_dist = torch.norm(self._socket_entrance_goal_pos - socket_pos, dim=-1)
        socket_lift = _tolerance(
            socket_dist, (0.0, 0.01), margin=0.15, sigmoid="linear"
        )

        peg_dist = torch.norm(self._peg_end2_goal_pos - peg_pos, dim=-1)
        peg_lift = _tolerance(
            peg_dist, (0.0, 0.01), margin=0.15, sigmoid="linear"
        )

        # ---- (5) no table collision -----------------------------------
        # TODO: contact information may be stale without a physics step in between
        contacts = self._robot.get_contacts(with_entity=self._table)
        link_a = contacts["link_a"]        # (n_envs, n_contacts) int64
        link_b = contacts["link_b"]
        valid = contacts["valid_mask"]     # (n_envs, n_contacts) bool

        finger_ids = self._finger_link_global_indices  # (4,)
        a_is_finger = (link_a.unsqueeze(-1) == finger_ids).any(dim=-1)
        b_is_finger = (link_b.unsqueeze(-1) == finger_ids).any(dim=-1)
        finger_table_contact = (a_is_finger | b_is_finger) & valid
        table_collision = finger_table_contact.any(dim=-1).float()
        no_table_collision = 1.0 - table_collision

        # ---- (6) z-up orientation -------------------------------------
        world_z = torch.tensor([0.0, 0.0, 1.0], device=self._device).expand(n, 3)
        peg_up_cos    = inv_transform_by_quat(world_z, robot_states["peg_quat"])[:, 2]
        socket_up_cos = inv_transform_by_quat(world_z, robot_states["socket_quat"])[:, 2]
        peg_orientation = _tolerance(
            peg_up_cos, (0.99, 1.0), margin=0.03, sigmoid="linear"
        )
        socket_orientation = _tolerance(
            socket_up_cos, (0.99, 1.0), margin=0.03, sigmoid="linear"
        )

        # ---- (7) peg insertion reward ---------------------------------
        socket_ab = socket_entrance_pos - socket_rear_pos
        socket_t = torch.sum((peg_end2_pos - socket_rear_pos) * socket_ab, dim=-1)
        socket_t = socket_t / (torch.sum(socket_ab * socket_ab, dim=-1) + 1e-6)
        nearest_pt = socket_rear_pos + socket_t.unsqueeze(-1) * socket_ab
        peg_end2_dist_to_line = torch.norm(peg_end2_pos - nearest_pt, dim=-1)
        use_peg_insertion_reward = (peg_end2_dist_to_line < 0.005).to(peg_end2_pos.dtype)

        peg_insertion_dist = torch.norm(peg_end2_pos - socket_rear_pos, dim=-1)
        peg_insertion_reward = _tolerance(
            peg_insertion_dist, (0.0, 0.001), margin=0.1, sigmoid="linear"
        ) * use_peg_insertion_reward

        raw = {
            "left_reward": left_reward,
            "right_reward": right_reward,
            # Soft AND logic, the target qpos reward is given when left and right reward are both high
            "left_target_qpos":  left_pose  * left_reward * right_reward,
            "right_target_qpos": right_pose * left_reward * right_reward,
            "no_table_collision": no_table_collision,
            "socket_entrance_reward": socket_lift,
            "peg_end2_reward": peg_lift,
            # Same soft AND logic for socket_z_up and peg_z_up
            "socket_z_up": socket_orientation * socket_lift,
            "peg_z_up":    peg_orientation    * peg_lift,
            "peg_insertion_reward": peg_insertion_reward,
        }
        total = sum(self._reward_scales[k] * v for k, v in raw.items())
        return total / self._reward_scale_sum

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self._early_termination:
            peg_oob = (robot_states["peg_pos"].abs() > self._out_of_bounds_threshold).any(dim=-1)
            socket_oob = (robot_states["socket_pos"].abs() > self._out_of_bounds_threshold).any(dim=-1)
            termination |= peg_oob | socket_oob

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
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device, dtype=torch.int32)
        
        robot_pos = self._robot.get_dofs_position(self._robot_dofs_idx, envs_idx=env_ids)
        robot_vel = self._robot.get_dofs_velocity(self._robot_dofs_idx, envs_idx=env_ids)
        
        peg_pos = self._peg.get_pos(envs_idx=env_ids)
        peg_quat = self._peg.get_quat(envs_idx=env_ids)
        peg_vel = self._peg.get_dofs_velocity(self._peg_dofs_idx, envs_idx=env_ids)
        
        socket_pos = self._socket.get_pos(envs_idx=env_ids)
        socket_quat = self._socket.get_quat(envs_idx=env_ids)
        socket_vel = self._socket.get_dofs_velocity(self._socket_dofs_idx, envs_idx=env_ids)

        # self._ctrl is the stateful delta-position command buffer used by
        # `_set_actions` (ctrl += action * scale each step). It lives in
        # Python and is NOT reflected in Genesis' internal state, so it
        # must be round-tripped explicitly or two envs starting from the
        # same DOF state but different accumulated ctrls will diverge.
        ctrl = self._ctrl if env_ids is None else self._ctrl[env_ids]

        robot_states = {
            "robot_pos": robot_pos.clone(),
            "robot_vel": robot_vel.clone(),
            "peg_pos": peg_pos.clone(),
            "peg_quat": peg_quat.clone(),
            "peg_vel": peg_vel.clone(),
            "socket_pos": socket_pos.clone(),
            "socket_quat": socket_quat.clone(),
            "socket_vel": socket_vel.clone(),
            "ctrl": ctrl.clone(),
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

        # Restore the stateful delta-position ctrl buffer (see note in
        # get_states). Without this, two envs with identical DOF state
        # but different accumulated ctrls will command different targets
        # on the next step and diverge.
        self._ctrl[env_ids] = robot_states["ctrl"].clone()

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
