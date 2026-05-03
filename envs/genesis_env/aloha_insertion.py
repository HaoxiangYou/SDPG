import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

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
        rigid_options: gs.options.RigidOptions | None = None,
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
            rigid_options=rigid_options,
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

            res = self._sensors_args["camera"]["res"]
            rgb_spaces = {
                cam_name: spaces.Box(
                    low=0, high=255, dtype=np.uint8,
                    shape=(3, res[0], res[1]),
                )
                for cam_name in self._sensors_args["camera"]["cameras"].keys()
            }
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(90,)),
                    # observation that ignores object information (infer from images)
                    "proprioception": spaces.Box(low=-np.inf, high=np.inf, shape=(52,)),
                    **rgb_spaces,
                }
            )
        else:
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(90,)),
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
        _damping_per_arm = [5.76, 20.0, 18.49, 6.78, 6.28, 1.2]
        _force_upper_per_arm = [35.0, 144.0, 59.0, 22.0, 35.0, 35.0]
        _kp_gripper = 1200.0
        _damping_gripper = 60.0
        _force_upper_gripper = 80.0
        self._kp = torch.tensor(
            _kp_per_arm + _kp_per_arm + [_kp_gripper, _kp_gripper],
            device=self._device,
        )
        self._kv = torch.zeros_like(self._kp)
        self._damping = torch.tensor(
            _damping_per_arm + _damping_per_arm + [_damping_gripper, _damping_gripper],
            device=self._device,
        )
        self._force_upper = torch.tensor(
            _force_upper_per_arm
            + _force_upper_per_arm
            + [_force_upper_gripper, _force_upper_gripper],
            device=self._device,
        )
        self._force_lower = -self._force_upper
        
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
                0.0305,  0.0305,
            ],
            device=self._device,
        ).repeat(self._num_envs, 1)

        self._ctrl = self._default_motor_dof_pos.clone()

        self._default_peg_pos = torch.tensor(
            [0.136459, 0.0, 0.0157945], device=self._device
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

        self._action_scale = 0.02

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

        if self._vis_obs:
            # Compute the offset_T for each configured camera. Result is a
            # dict keyed by camera name; consumed by the per-camera
            # `scene.add_sensor(...)` calls below.
            self._camera_cfgs = self._sensors_args["camera"]["cameras"]
            self._camera_offset_Ts: Dict[str, torch.Tensor] = {}
            for cam_name, cam_cfg in self._camera_cfgs.items():
                offset_T = cam_cfg.get("offset_T", None)
                if offset_T is not None:
                    offset_T = torch.tensor(offset_T, device=self._device)
                else:
                    lookat = cam_cfg.get("lookat", None)
                    if lookat is not None:
                        offset_T = pos_lookat_up_to_T(
                            np.array(cam_cfg["pos"]),
                            np.array(lookat),
                            np.array((0.0, 0.0, 1.0)),
                        )
                    else:
                        offset_T = np.eye(4)
                self._camera_offset_Ts[cam_name] = offset_T

            # NOTE: A dummy link for static (world-frame) cameras to attach to;
            # without entity_idx the batch renderer uses a single world pose,
            # only env 0 is in view and other envs render black.
            self._table_camera_mount = self._scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.001,
                    collision=False,
                    fixed=True,
                    pos=(0.0, -0.377167, 0.0316055),
                    quat=(0.672659, 0.739953, 0.0, 0.0),
                )
            )

            # Mount spec per camera: (entity_idx, link_idx_local).
            #   - `table`       -> dummy world-frame sphere (static)
            #   - `wrist_left`  -> robot's left gripper_link  (tracks the hand)
            #   - `wrist_right` -> robot's right gripper_link (tracks the hand)
            camera_mounts = {
                "table": (
                    self._table_camera_mount.idx,
                    0,
                ),
                "wrist_left": (
                    self._robot.idx,
                    self._robot.get_link("left/gripper_link").idx_local,
                ),
                "wrist_right": (
                    self._robot.idx,
                    self._robot.get_link("right/gripper_link").idx_local,
                ),
            }

            shared_res = self._sensors_args["camera"]["res"]
            shared_lights = self._sensors_args["camera"]["lights"]
            shared_env_idx = self._nominal_env_ids.cpu().tolist()

            self._cameras: Dict[str, Any] = {}
            for i, (cam_name, cam_cfg) in enumerate(self._camera_cfgs.items()):
                if cam_name not in camera_mounts:
                    gs.raise_exception(
                        f"Unknown camera '{cam_name}' in sensors_args.camera.cameras. "
                        f"Supported: {list(camera_mounts.keys())}"
                    )
                entity_idx, link_idx_local = camera_mounts[cam_name]
                # Lights are scene-global in the Madrona batch renderer; attach
                # them once (first camera only) to avoid duplicates.
                cam_lights = [shared_lights] if i == 0 else []

                self._cameras[cam_name] = self._scene.add_sensor(
                    gs.sensors.BatchRendererCameraOptions(
                        res=shared_res,
                        pos=cam_cfg["pos"],
                        offset_T=self._camera_offset_Ts[cam_name],
                        fov=cam_cfg["fov"],
                        near=cam_cfg["near"],
                        far=cam_cfg["far"],
                        entity_idx=entity_idx,
                        link_idx_local=link_idx_local,
                        lights=cam_lights,
                        env_idx=shared_env_idx,
                    )
                )

            # One HWC buffer per camera (no history / stacking).
            res = self._sensors_args["camera"]["res"]
            self._imgs_buf: Dict[str, torch.Tensor] = {
                cam_name: torch.zeros(
                    self.nominal_env_ids.shape[0],
                    res[0], res[1], 3,
                    device=self._device,
                    dtype=torch.uint8,
                )
                for cam_name in self._cameras.keys()
            }

    def build_scene(self) -> None:
        self._scene.build(n_envs=self._num_envs, env_spacing=(1.5, 1.5))

        self._robot.set_dofs_kp(self._kp, self._motors_dof_idx)
        self._robot.set_dofs_kv(self._kv, self._motors_dof_idx)
        self._robot.set_dofs_force_range(
            lower=self._force_lower,
            upper=self._force_upper,
            dofs_idx_local=self._motors_dof_idx,
        )
        self._robot.set_dofs_damping(
            self._damping, self._motors_dof_idx
        )

        # Genesis has no torsional friction, increasing the sliding friction instead to prevent slipping and rotating.
        finger_friction = 4.0
        peg_friction = 4.0
        for fname in self._finger_link_names:
            self._robot.get_link(fname).set_friction(finger_friction)
        self._peg.set_friction(peg_friction)
        self._socket.set_friction(peg_friction)

        self._ctrl_lower, self._ctrl_upper = self._robot.get_dofs_limit(
            dofs_idx_local=self._motors_dof_idx
        )
        self._ctrl_lower[..., -2:] = 0.002
        self._ctrl_upper[..., -2:] = 0.037

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        robot_states = states["robot_states"]
        batch = robot_states["robot_pos"].shape[0]

        left_gripper_pos    = self._left_gripper_site_link.get_pos()
        right_gripper_pos   = self._right_gripper_site_link.get_pos()
        socket_entrance_pos = self._socket_entrance_site_link.get_pos()
        peg_end2_pos        = self._peg_end2_site_link.get_pos()

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
                robot_states["ctrl"],               # (N, 14)
                left_gripper_pos,
                right_gripper_pos,
                socket_entrance_pos,
                peg_end2_pos,
                socket_z,
                peg_z,
            ],
            dim=-1,
        )  # (N, 90)

        observations = {
            "privileged_observations": privileged_observations,
        }

        if self._vis_obs:
            proprioception = torch.cat(
                [
                    robot_states["robot_pos"],          # (N, 16)
                    robot_states["robot_vel"],          # (N, 16)
                    robot_states["ctrl"],               # (N, 14)
                    left_gripper_pos,
                    right_gripper_pos,
                ],
                dim=-1,
            )

            observations["proprioception"] = proprioception

            # Expose each camera as its own observation key in channels-first
            # (N, 3, H, W) layout expected by CNN encoders.
            for cam_name, img_buf in self._imgs_buf.items():
                observations[cam_name] = img_buf.permute(0, 3, 1, 2)

        return observations

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
            "left_target_qpos":  left_pose,
            "right_target_qpos": right_pose,
            "no_table_collision": no_table_collision,
            "socket_entrance_reward": socket_lift,
            "peg_end2_reward": peg_lift,
            # Soft AND logic for socket_z_up and peg_z_up, reward is high when object is lifted
            "socket_z_up": socket_orientation * socket_lift,
            "peg_z_up":    peg_orientation    * peg_lift,
            "peg_insertion_reward": peg_insertion_reward,
        }
        
        self._infos.update(
            {
                "left_gripper_socket_dist": left_socket_dist.detach().mean().item(),
                "right_gripper_peg_dist": right_peg_dist.detach().mean().item(),
                "socket_target_dist": socket_dist.detach().mean().item(),
                "peg_target_dist": peg_dist.detach().mean().item(),
            }
        )

        total = sum(self._reward_scales[k] * v for k, v in raw.items())
        return total

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

        if self._vis_obs:
            # Find which nominal environments are being reset
            # self.nominal_env_ids contains the global env_ids of nominal environments
            # We need to find the indices within nominal_env_ids that match env_ids
            mask = torch.isin(self.nominal_env_ids, env_ids)
            nominal_idx_to_reset = torch.nonzero(mask, as_tuple=True)[0]

            if len(nominal_idx_to_reset) > 0:
                # Render fresh images for the reset nominal environments
                reset_nominal_env_ids = self.nominal_env_ids[nominal_idx_to_reset]
                new_imgs = self.render(env_ids=reset_nominal_env_ids)

                # Write the fresh frame into each camera's buffer slice.
                for cam_name, new_img in zip(self._imgs_buf.keys(), new_imgs):
                    self._imgs_buf[cam_name][nominal_idx_to_reset] = new_img

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
        if self._vis_obs:
            # No history / stacking — overwrite each camera's buffer with the
            # freshly rendered frame for the current step.
            new_imgs = self.render(env_ids=self.nominal_env_ids)
            for cam_name, new_img in zip(self._imgs_buf.keys(), new_imgs):
                self._imgs_buf[cam_name].copy_(new_img)

    def render(
        self, env_ids: Optional[Sequence[int]] = None
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if not self._vis_obs:
            return None

        if env_ids is None:
            env_ids = self.nominal_env_ids

        # TODO: genesis will refresh the image when the scene._dt is different
        # from the last render time; force a re-render by invalidating the
        # timestamp on the shared metadata (which is shared across all batch-
        # renderer cameras, so setting it on any one camera is enough).
        first_cam = next(iter(self._cameras.values()))
        first_cam._shared_metadata.last_render_timestep = 0

        return tuple(cam.read(envs_idx=env_ids).rgb for cam in self._cameras.values())

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

        self._ctrl[env_ids] = robot_states["ctrl"].clone()

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
