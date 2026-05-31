import os
import warnings
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import (
    pos_lookat_up_to_T,
    quat_to_R,
    transform_by_quat,
)
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class Franka(GenesisEnv):
    """Franka Emika Panda pick-a-cube environment.

    Adapted from mujoco_playground's PandaPickCube: the arm must drive its
    gripper to a cube spawned on the table, then lift the cube to a target
    pose in free space.
    """

    _num_actions = 8
    _action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))

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
        debug_viz: bool = True,
        show_target: bool = True,
    ) -> None:
        dt = sim_options.dt
        # Match reference episode length of 150 ctrl steps at 0.02s = 3s rollout.
        episode_length = int(3.0 / dt)
        early_termination = True

        self._vis_obs = vis_obs
        self._show_target = show_target
        # Debug spheres for the reward target vs. actual gripper midpoint.
        # Requires a viewer to be visible.
        self._debug_viz = debug_viz and show_viewer

        if sensors_args is None:
            sensors_args = {
                "camera": {
                    "res": [256, 256],
                    # Hand-link camera transform, matching
                    # scripts/test_genesis_camera.py. The camera rotates with
                    # the hand because this offset is attached to the hand link.
                    "pos": [-0.2, 0.0, -0.1],
                    "lookat": [0.0, 0.0, 0.0],
                    "up": [1.0, 0.0, 0.0],
                    "fov": 60.0,
                    "near": 0.01,
                    "far": 5.0,
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

        # Observation layout (privileged), matching pick.py:
        #   qpos (7 arm + 2 fingers + 7 cube freejoint = 16)
        # + qvel (7 arm + 2 fingers + 6 cube freejoint = 15)
        # + gripper_pos (3) + gripper_mat[3:] (6)
        # + cube_mat[3:] (6)
        # + (cube_pos - gripper_pos) (3) + (target_pos - cube_pos) (3)
        # + (target_mat[:6] - cube_mat[:6]) (6)
        # + (ctrl - qpos[robot_qposadr[:-1]]) (8) = 66
        obs_dim = 66

        # Actor observation layout (no cube state — matches real-world access):
        #   arm_qpos (7) + finger_qpos (2)
        # + arm_qvel (7) + finger_qvel (2)
        # + gripper_pos (3) + gripper_mat[3:] (6)
        # + (ctrl - qpos[robot_qposadr[:-1]]) (8) = 35
        actor_obs_dim = 35

        if vis_obs:
            self._num_image_stack = 1
            self._observation_space = spaces.Dict(
                {
                    "actor_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_obs_dim,)),
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
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

    def init_scene(self) -> None:
        """Initialize the scene."""
        # Genesis's own MJCF loader chokes on mjx_panda.xml (the mujoco-playground
        # single-cube flavor) because of geom class overrides / capsule tags.
        # Fall back to Genesis's bundled panda.xml, which shares the same joint
        # and link names so the rest of the task definition carries over.
        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )

        self._plane = self._scene.add_entity(gs.morphs.Plane())

        # Pickable cube (matches mjx_single_cube.xml keyframe: half-size
        # 0.02 0.02 0.03, free joint, green, friction 1 .03 .003).
        # Visual is hidden — the peach mesh below overlays it.
        self._cube = self._scene.add_entity(
            gs.morphs.Box(
                size=(0.08, 0.08, 0.08),
                pos=(0.6, 0.0, 0.03),
                collision=True,
                visualization=False,
                batch_fixed_verts=True,
            ),
            material=gs.materials.Rigid(friction=1.0),
        )

        # Visual-only peach mesh teleported onto the cube each step.
        # Mesh extent in peach.obj is ~5.9 cm; scale up so it roughly
        # envelopes the 8 cm cube. The diffuse texture is wired in via
        # the sibling material.mtl exported with the OBJ.
        self._peach_visual = self._scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(
                    os.path.dirname(__file__), "../../assets/peach/peach.obj"
                ),
                pos=(0.6, 0.0, 0.03),
                scale=1.36,
                collision=False,
            ),
            material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(),
        )

        # Visual-only target cube (matches mjx_single_cube.xml mocap_target).
        self._target = None
        if self._show_target:
            self._target = self._scene.add_entity(
                gs.morphs.Box(
                    size=(0.08, 0.08, 0.08),
                    pos=(0.6, 0.0, 0.33),
                    collision=False,
                ),
                surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0), opacity=0.0),
                material=gs.materials.Rigid(gravity_compensation=1),
            )

        # Joints and actuators.
        self._arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self._finger_joint_names = ["finger_joint1", "finger_joint2"]

        self._arm_dof_idx = []
        for name in self._arm_joint_names:
            self._arm_dof_idx.extend(self._robot.get_joint(name).dofs_idx_local)
        self._finger_dof_idx = []
        for name in self._finger_joint_names:
            self._finger_dof_idx.extend(self._robot.get_joint(name).dofs_idx_local)
        # Control dofs: 7 arm joints + one finger (actuator8 drives finger_joint1,
        # finger_joint2 is coupled via equality in the XML).
        self._motor_dof_idx = self._arm_dof_idx + [self._finger_dof_idx[0]]

        # Links.
        self._hand_link = self._robot.get_link("hand")
        self._left_finger_link = self._robot.get_link("left_finger")
        self._right_finger_link = self._robot.get_link("right_finger")

        self._cube_dof_idx = self._cube.base_joint.dofs_idx_local

        # Local offset from the finger midpoint to the reward gripper point.
        self._gripper_midpoint_offset = torch.tensor(
            [0.0, 0.0, 0.05], device=self._device
        )

        # Per-step debug sphere handles (cleared & redrawn each step).
        self._debug_sphere_handles: list = []

        # Default keyframe "home" from mjx_single_cube.xml.
        self._default_arm_dof_pos = torch.tensor(
            [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853],
            device=self._device,
        ).repeat(self._num_envs, 1)
        self._default_finger_dof_pos = torch.tensor(
            [0.04, 0.04], device=self._device
        ).repeat(self._num_envs, 1)
        # We drive the finger via `control_dofs_position`, which takes a DOF
        # position target (joint range [0, 0.04]) rather than the tendon
        # actuator's ctrl range. 0.04 is the fully-open finger position.
        self._default_ctrl = torch.tensor(
            [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04],
            device=self._device,
        ).repeat(self._num_envs, 1)

        # Keyframe "home" cube pose from mjx_single_cube.xml (qpos[9:16]):
        # pos=(0.6, 0, 0.03), quat=(1, 0, 0, 0). pick.py uses this single anchor
        # for both cube and target randomization. We lift by +0.005 so the cube
        # bottom clears the plane at spawn instead of sitting flush, which
        # occasionally caused the rigid contact solver to blow up (NaN cube_pos
        # on roughly 1 in ~2000 envs per reset).
        self._init_obj_pos = torch.tensor([0.6, 0.0, 0.035], device=self._device)
        self._default_cube_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self._device
        ).repeat(self._num_envs, 1)
        self._default_target_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self._device
        ).repeat(self._num_envs, 1)

        # Per-environment target pose buffers (mocap replacement).
        self._target_pos = self._init_obj_pos.clone().repeat(self._num_envs, 1)
        self._target_quat = self._default_target_quat.clone()

        # Running ctrl state (ctrl is integrated with the policy delta per step).
        self._ctrl = self._default_ctrl.clone()
        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)
        # Tracks whether the gripper has reached the box within an episode
        # (used to gate the box->target reward term).
        self._reached_box = torch.zeros(self._num_envs, device=self._device)

        # Per-dof action scale. pick.py uses 0.04 for all eight ctrl entries
        # (gripper ctrlrange is [0, 0.04]). We drive the gripper at the DOF
        # level, which uses the same joint range, so the scales match exactly.
        self._action_scale = torch.tensor(
            [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
            device=self._device,
        )

        # Reward scales (from default_config in pick.py).
        self._reward_scale_gripper_box = 4.0
        self._reward_scale_box_target = 8.0
        self._reward_scale_no_floor_collision = 0.25
        self._reward_scale_robot_target_qpos = 0.3
        self._reward_scale_grasp = 1.0
        self._reward_scale_lift = 3.0
        self._reward_scale_box_target_lifted = 16.0
        self._reward_scale_box_target_fine = 5.0

        # Termination thresholds. The cube spawns with its bottom flush with
        self._reached_box_threshold = 0.02
        # Use the collision-body AABB directly for floor-contact termination.
        # A tiny non-negative margin helps catch shallow penetrations robustly.
        self._hand_floor_height = 0.001

        if self._vis_obs:
            offset_T = self._sensors_args["camera"].get("offset_T", None)
            lookat = self._sensors_args["camera"].get("lookat", None)
            if offset_T is not None:
                if torch.is_tensor(offset_T):
                    offset_T = offset_T.detach().cpu().numpy()
                else:
                    offset_T = np.asarray(offset_T, dtype=np.float32)
            else:
                if lookat is not None:
                    offset_T = pos_lookat_up_to_T(
                        np.array(self._sensors_args["camera"]["pos"]),
                        np.array(lookat),
                        np.array(self._sensors_args["camera"].get("up", (0.0, 0.0, 1.0))),
                    ).astype(np.float32)
                else:
                    offset_T = np.eye(4, dtype=np.float32)
            self._camera = self._scene.add_sensor(
                gs.sensors.BatchRendererCameraOptions(
                    res=self._sensors_args["camera"]["res"],
                    pos=self._sensors_args["camera"]["pos"],
                    offset_T=offset_T,
                    fov=self._sensors_args["camera"]["fov"],
                    near=self._sensors_args["camera"].get("near", 0.01),
                    far=self._sensors_args["camera"].get("far", 5.0),
                    entity_idx=self._robot.idx,
                    link_idx_local=self._hand_link.idx_local,
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
        self._scene.build(n_envs=self._num_envs, env_spacing=(1.0/self._num_envs, 1.0/self._num_envs))
        self._arm_ctrl_lower, self._arm_ctrl_upper = self._robot.get_dofs_limit(
            dofs_idx_local=self._arm_dof_idx
        )
        finger_lower, finger_upper = self._robot.get_dofs_limit(
            dofs_idx_local=[self._finger_dof_idx[0]]
        )
        self._ctrl_lower = torch.cat([self._arm_ctrl_lower, finger_lower], dim=-1)
        self._ctrl_upper = torch.cat([self._arm_ctrl_upper, finger_upper], dim=-1)

    def _gripper_pos(self) -> torch.Tensor:
        """Gripper position in world frame as midpoint plus rotated offset."""
        left_finger_pos = self._left_finger_link.get_pos()
        right_finger_pos = self._right_finger_link.get_pos()
        midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        hand_quat = self._hand_link.get_quat()
        offset = self._gripper_midpoint_offset.repeat(midpoint.shape[0], 1)
        return midpoint + transform_by_quat(offset, hand_quat)

    def _link_collision_min_z(self, link) -> torch.Tensor:
        """Minimum world-frame z over a link's collision geometry."""
        return link.get_AABB()[:, 0, 2]

    def _floor_collision_mask(self) -> torch.Tensor:
        """Detect floor contact using hand/finger collision-body bounds."""
        hand_min_z = self._link_collision_min_z(self._hand_link)
        left_min_z = self._link_collision_min_z(self._left_finger_link)
        right_min_z = self._link_collision_min_z(self._right_finger_link)
        self._infos["hand_min_z"] = torch.nan_to_num(hand_min_z).mean().item()
        self._infos["left_finger_min_z"] = torch.nan_to_num(left_min_z).mean().item()
        self._infos["right_finger_min_z"] = torch.nan_to_num(right_min_z).mean().item()
        return (
            (hand_min_z <= self._hand_floor_height)
            | (left_min_z <= self._hand_floor_height)
            | (right_min_z <= self._hand_floor_height)
        )

    def _invalid_cube_state_mask(self, robot_states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mask envs where any tensor in robot_states contains a NaN value."""
        num_envs = next(iter(robot_states.values())).shape[0]
        nan_mask = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        for value in robot_states.values():
            if not torch.is_tensor(value):
                continue
            flat = value.reshape(num_envs, -1) if value.dim() > 1 else value.unsqueeze(-1)
            nan_mask = nan_mask | torch.isnan(flat).any(dim=-1)
        return nan_mask

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}
        robot_states = states["robot_states"]

        arm_qpos = robot_states["arm_dof_pos"]
        arm_qvel = robot_states["arm_dof_vel"]
        finger_qpos = robot_states["finger_dof_pos"]
        finger_qvel = robot_states["finger_dof_vel"]

        gripper_pos = robot_states["gripper_pos"]
        gripper_quat = robot_states["gripper_quat"]
        cube_pos = robot_states["cube_pos"]
        cube_quat = robot_states["cube_quat"]
        cube_vel = robot_states["cube_vel"]
        target_pos = robot_states["target_pos"]
        target_quat = robot_states["target_quat"]
        ctrl = robot_states["ctrl"]

        # Reference pick.py stacks MuJoCo's full qpos/qvel, which includes the
        # cube's free-joint state: position(3) + quat(4) for qpos, linvel(3) +
        # angvel(3) for qvel.
        qpos = torch.cat([arm_qpos, finger_qpos, cube_pos, cube_quat], dim=-1)  # 16
        qvel = torch.cat([arm_qvel, finger_qvel, cube_vel], dim=-1)  # 15

        gripper_mat = quat_to_R(gripper_quat).reshape(gripper_quat.shape[0], 9)
        cube_mat = quat_to_R(cube_quat).reshape(cube_quat.shape[0], 9)
        target_mat = quat_to_R(target_quat).reshape(target_quat.shape[0], 9)

        # ctrl is length 8; robot qpos (arm + fingers) is length 9 so the
        # reference drops the last finger entry to align shapes.
        ctrl_minus_qpos = ctrl - torch.cat([arm_qpos, finger_qpos[:, :1]], dim=-1)

        privileged_observations = torch.cat(
            [
                qpos,
                qvel,
                gripper_pos,
                gripper_mat[:, 3:],
                cube_mat[:, 3:],
                cube_pos - gripper_pos,
                target_pos - cube_pos,
                target_mat[:, :6] - cube_mat[:, :6],
                ctrl_minus_qpos,
            ],
            dim=-1,
        )
        # Sim occasionally produces NaN/Inf in robot_states; replace with finite
        # values so the policy network stays finite on the terminal step (the
        # env will be reset next step via compute_termination).
        privileged_observations = torch.nan_to_num(
            privileged_observations, nan=0.0, posinf=0.0, neginf=0.0
        )
        observations["privileged_observations"] = privileged_observations

        if self._vis_obs:
            # Actor obs excludes cube state (unavailable in the real world).
            # Critic uses privileged_observations (full state, computed above).
            actor_observations = torch.cat(
                [
                    arm_qpos,         # 7
                    finger_qpos,      # 2
                    arm_qvel,         # 7
                    finger_qvel,      # 2
                    gripper_pos,      # 3
                    gripper_mat[:, 3:],  # 6
                    ctrl_minus_qpos,  # 8
                ],
                dim=-1,
            )
            actor_observations = torch.nan_to_num(
                actor_observations, nan=0.0, posinf=0.0, neginf=0.0
            )
            observations["actor_observations"] = actor_observations

            batch_size, num_stack, img_height, img_width, rgb = self._imgs_buf.shape
            observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                batch_size, num_stack * rgb, img_height, img_width
            )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        robot_states = states["robot_states"]
        invalid = self._invalid_cube_state_mask(robot_states)
        cube_pos = robot_states["cube_pos"]
        cube_quat = robot_states["cube_quat"]
        gripper_pos = robot_states["gripper_pos"]
        target_pos = robot_states["target_pos"]
        target_quat = robot_states["target_quat"]
        arm_dof_pos = robot_states["arm_dof_pos"]
        finger_qpos = robot_states["finger_dof_pos"]

        pos_err = torch.norm(target_pos - cube_pos, p=2, dim=-1)
        box_mat = quat_to_R(cube_quat).reshape(cube_quat.shape[0], 9)
        target_mat = quat_to_R(target_quat).reshape(target_quat.shape[0], 9)
        rot_err = torch.norm(target_mat[:, :6] - box_mat[:, :6], p=2, dim=-1)

        box_target = 1.0 - torch.tanh(5.0 * (0.9 * pos_err + 0.1 * rot_err))
        gripper_box_dist = torch.norm(cube_pos - gripper_pos, p=2, dim=-1)
        gripper_box = 1.0 - torch.tanh(5.0 * gripper_box_dist)

        default_arm = self._default_arm_dof_pos[: arm_dof_pos.shape[0]]
        robot_target_qpos = 1.0 - torch.tanh(
            torch.norm(arm_dof_pos - default_arm, p=2, dim=-1)
        )

        # Floor-collision proxy: penalize when the hand or fingers dip below
        # a safety height above the floor (true contact sensors are not exposed
        # through the Genesis MJCF loader).
        floor_collision = self._floor_collision_mask()
        no_floor_collision = (~floor_collision).float()

        # Gate the box->target term on gripper having reached the box at least
        # once within the current episode (mirrors the reference implementation).
        self._reached_box = torch.maximum(
            self._reached_box,
            (gripper_box_dist < self._reached_box_threshold).float(),
        )
        gripper_closed = (finger_qpos.sum(dim=-1) < 0.045).float()
        grasp = (self._reached_box > 0.0).float() * gripper_closed
        lifted = (cube_pos[:, 2] > self._init_obj_pos[2] + 0.04).float()
        box_target_lifted = (1.0 - torch.tanh(pos_err / 0.3)) * lifted
        box_target_fine = (1.0 - torch.tanh(pos_err / 0.10)) * lifted
        action_penalty = torch.sum(actions**2, dim=-1)

        rewards = (
            self._reward_scale_gripper_box * gripper_box
            + self._reward_scale_box_target * box_target * self._reached_box
            + self._reward_scale_no_floor_collision * no_floor_collision
            + self._reward_scale_robot_target_qpos * robot_target_qpos
            + self._reward_scale_grasp * grasp
            + self._reward_scale_lift * lifted
            + self._reward_scale_box_target_lifted * box_target_lifted
            + self._reward_scale_box_target_fine * box_target_fine
            - 0.01 * action_penalty
        )
        # Zero the reward for envs whose sim state is corrupt; then scrub any
        # residual NaN/Inf so PPO's advantage/return math stays finite.
        num_invalid = int(invalid.sum().item())
        if num_invalid > 0:
            warnings.warn(
                f"compute_reward: {num_invalid}/{invalid.numel()} envs returned "
                f"invalid sim state; reward zeroed out for those envs.",
                RuntimeWarning,
                stacklevel=2,
            )
        rewards = torch.where(invalid, torch.zeros_like(rewards), rewards)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        rewards = torch.clamp(rewards, -1e4, 1e4)

        self._infos["box_target_dist"] = torch.nan_to_num(pos_err).mean().item()
        self._infos["gripper_box_dist"] = torch.nan_to_num(gripper_box_dist).mean().item()
        self._infos["robot_target_qpos"] = torch.nan_to_num(robot_target_qpos).mean().item()
        self._infos["reached_box"] = self._reached_box.mean().item()
        self._infos["gripper_closed"] = gripper_closed.mean().item()
        self._infos["grasp_reward"] = grasp.mean().item()
        self._infos["lifted"] = lifted.mean().item()
        self._infos["box_target_lifted"] = torch.nan_to_num(box_target_lifted).mean().item()
        self._infos["box_target_fine"] = torch.nan_to_num(box_target_fine).mean().item()
        self._infos["action_penalty"] = torch.nan_to_num(action_penalty).mean().item()
        self._infos["invalid_robot_state"] = invalid.float().mean().item()

        return rewards

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            robot_states = states["robot_states"]
            invalid_cube = self._invalid_cube_state_mask(robot_states)
            cube_pos = robot_states["cube_pos"]
            out_of_bounds = torch.any(torch.abs(cube_pos) > 1.0, dim=-1) | (cube_pos[:, 2] < 0.0)
            floor_collision = self._floor_collision_mask()
            termination = invalid_cube | out_of_bounds | floor_collision
            self._infos["term_invalid_cube"] = invalid_cube.float().mean().item()
            self._infos["term_out_of_bounds"] = out_of_bounds.float().mean().item()
            self._infos["term_floor_collision"] = floor_collision.float().mean().item()
        return termination

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        arm_dof_pos = self._default_arm_dof_pos[env_ids].clone()
        finger_dof_pos = self._default_finger_dof_pos[env_ids].clone()
        ctrl = self._default_ctrl[env_ids].clone()

        init_obj_pos = self._init_obj_pos.unsqueeze(0).repeat(len(env_ids), 1)
        cube_pos = init_obj_pos.clone()
        cube_quat = self._default_cube_quat[env_ids].clone()
        # Default target sits 0.3m above the keyframe cube (center of pick.py's
        # target z-range [0.23, 0.43] is 0.33, close to +0.3).
        target_pos = init_obj_pos + torch.tensor([0.0, 0.0, 0.3], device=self._device)
        target_quat = self._default_target_quat[env_ids].clone()

        if self._randomize_init:
            # pick.py: cube_pos = init_obj_pos + U([-0.2, -0.2, 0.0], [0.2, 0.2, 0.0])
            cube_min = torch.tensor([-0.1, -0.1, 0.0], device=self._device)
            cube_max = torch.tensor([0.1, 0.1, 0.0], device=self._device)
            cube_pos = init_obj_pos + cube_min + torch.rand(
                len(env_ids), 3, device=self._device
            ) * (cube_max - cube_min)

            # pick.py: target_pos = init_obj_pos + U([-0.2, -0.2, 0.2], [0.2, 0.2, 0.4])
            target_min = torch.tensor([-0.1, -0.1, 0.2], device=self._device)
            target_max = torch.tensor([0.1, 0.1, 0.4], device=self._device)
            target_pos = init_obj_pos + target_min + torch.rand(
                len(env_ids), 3, device=self._device
            ) * (target_max - target_min)

        # Robot.
        self._robot.set_dofs_position(
            position=arm_dof_pos,
            dofs_idx_local=self._arm_dof_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self._robot.set_dofs_position(
            position=finger_dof_pos,
            dofs_idx_local=self._finger_dof_idx,
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self._robot.control_dofs_position(
            position=ctrl,
            dofs_idx_local=self._motor_dof_idx,
            envs_idx=env_ids,
        )

        # Cube.
        self._cube.set_pos(cube_pos, envs_idx=env_ids, zero_velocity=True)
        self._cube.set_quat(cube_quat, envs_idx=env_ids, zero_velocity=True)

        # Visual peach follows the cube.
        self._peach_visual.set_pos(cube_pos, envs_idx=env_ids, zero_velocity=True)
        self._peach_visual.set_quat(cube_quat, envs_idx=env_ids, zero_velocity=True)

        # Target visualization.
        if self._target is not None:
            self._target.set_pos(target_pos, envs_idx=env_ids, zero_velocity=True)
            self._target.set_quat(target_quat, envs_idx=env_ids, zero_velocity=True)

        # Buffers.
        self._target_pos[env_ids] = target_pos
        self._target_quat[env_ids] = target_quat
        self._ctrl[env_ids] = ctrl
        self._prev_actions[env_ids] = 0.0
        self._reached_box[env_ids] = 0.0

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

        # Policy action is a delta around the current ctrl; clip to actuator ranges.
        delta = actions * self._action_scale
        self._ctrl = torch.clamp(
            self._ctrl + delta,
            self._ctrl_lower,
            self._ctrl_upper,
        )
        binary_gripper_ctrl = torch.where(
            self._ctrl[:, -1] < 0.02,
            self._ctrl_lower[-1],
            self._ctrl_upper[-1],
        )
        self._ctrl[:, -1] = torch.where(
            self._reached_box > 0.0,
            binary_gripper_ctrl,
            self._ctrl_upper[-1],
        )

        self._robot.control_dofs_position(
            self._ctrl,
            dofs_idx_local=self._motor_dof_idx,
        )

    def _post_physics_step(self) -> None:
        # Keep the visual peach overlay aligned with the cube each step.
        self._peach_visual.set_pos(self._cube.get_pos())
        self._peach_visual.set_quat(self._cube.get_quat())

        if self._debug_viz:
            self._draw_fingertip_debug_spheres()

        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img

    def _draw_fingertip_debug_spheres(self) -> None:
        """Visualize the gripper midpoint (red) and the pick cube position
        (yellow) in world frame for env 0.
        """
        for handle in self._debug_sphere_handles:
            self._scene.clear_debug_object(handle)
        self._debug_sphere_handles.clear()

        gripper_pt = self._gripper_pos()[0].detach().cpu().numpy()
        cube_pt = self._cube.get_pos()[0].detach().cpu().numpy()
        if not (np.all(np.isfinite(gripper_pt)) and np.all(np.isfinite(cube_pt))):
            return
        gripper_color = (0.0, 1.0, 0.0, 0.9) if self._reached_box[0] > 0 else (1.0, 0.0, 0.0, 0.9)

        self._debug_sphere_handles.append(
            self._scene.draw_debug_sphere(
                pos=gripper_pt, radius=0.02, color=gripper_color
            )
        )
        self._debug_sphere_handles.append(
            self._scene.draw_debug_sphere(
                pos=cube_pt, radius=0.02, color=(1.0, 1.0, 0.0, 0.9)
            )
        )

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

        arm_dof_pos = self._robot.get_dofs_position(self._arm_dof_idx, envs_idx=env_ids)
        arm_dof_vel = self._robot.get_dofs_velocity(self._arm_dof_idx, envs_idx=env_ids)
        finger_dof_pos = self._robot.get_dofs_position(self._finger_dof_idx, envs_idx=env_ids)
        finger_dof_vel = self._robot.get_dofs_velocity(self._finger_dof_idx, envs_idx=env_ids)

        cube_pos = self._cube.get_pos(envs_idx=env_ids)
        cube_quat = self._cube.get_quat(envs_idx=env_ids)
        cube_vel = self._cube.get_dofs_velocity(self._cube_dof_idx, envs_idx=env_ids)

        gripper_pos = self._gripper_pos()[env_ids]
        gripper_quat = self._hand_link.get_quat()[env_ids]

        robot_states = {
            "arm_dof_pos": arm_dof_pos.clone(),
            "arm_dof_vel": arm_dof_vel.clone(),
            "finger_dof_pos": finger_dof_pos.clone(),
            "finger_dof_vel": finger_dof_vel.clone(),
            "cube_pos": cube_pos.clone(),
            "cube_quat": cube_quat.clone(),
            "cube_vel": cube_vel.clone(),
            "gripper_pos": gripper_pos.clone(),
            "gripper_quat": gripper_quat.clone(),
            "target_pos": self._target_pos[env_ids].clone(),
            "target_quat": self._target_quat[env_ids].clone(),
            "ctrl": self._ctrl[env_ids].clone(),
            "prev_actions": self._prev_actions[env_ids].clone(),
            "reached_box": self._reached_box[env_ids].clone(),
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
            position=robot_states["arm_dof_pos"],
            dofs_idx_local=self._arm_dof_idx,
            envs_idx=env_ids,
            zero_velocity=False,
        )
        self._robot.set_dofs_velocity(
            velocity=robot_states["arm_dof_vel"],
            dofs_idx_local=self._arm_dof_idx,
            envs_idx=env_ids,
        )
        self._robot.set_dofs_position(
            position=robot_states["finger_dof_pos"],
            dofs_idx_local=self._finger_dof_idx,
            envs_idx=env_ids,
            zero_velocity=False,
        )
        self._robot.set_dofs_velocity(
            velocity=robot_states["finger_dof_vel"],
            dofs_idx_local=self._finger_dof_idx,
            envs_idx=env_ids,
        )

        self._cube.set_pos(robot_states["cube_pos"], envs_idx=env_ids)
        self._cube.set_quat(robot_states["cube_quat"], envs_idx=env_ids)
        self._cube.set_dofs_velocity(
            robot_states["cube_vel"],
            envs_idx=env_ids,
            dofs_idx_local=self._cube_dof_idx,
        )

        self._peach_visual.set_pos(robot_states["cube_pos"], envs_idx=env_ids)
        self._peach_visual.set_quat(robot_states["cube_quat"], envs_idx=env_ids)

        self._target_pos[env_ids] = robot_states["target_pos"]
        self._target_quat[env_ids] = robot_states["target_quat"]
        if self._target is not None:
            self._target.set_pos(robot_states["target_pos"], envs_idx=env_ids)
            self._target.set_quat(robot_states["target_quat"], envs_idx=env_ids)

        self._ctrl[env_ids] = robot_states["ctrl"].clone()
        self._robot.control_dofs_position(
            position=self._ctrl[env_ids],
            dofs_idx_local=self._motor_dof_idx,
            envs_idx=env_ids,
        )
        self._prev_actions[env_ids] = robot_states["prev_actions"].clone()
        self._reached_box[env_ids] = robot_states["reached_box"].clone()

        self._progress_buf[env_ids] = states["progress_buf"].clone()