import os
from typing import Any, Dict, Optional, Sequence

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import axis_angle_to_quat, pos_lookat_up_to_T, transform_by_quat, transform_quat_by_quat
from gym import spaces

from envs.genesis_env.genesis_env import GenesisEnv


class Ant(GenesisEnv):
    """Ant environment."""

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
                    "pos": [0.0, -2.0, -0.5],
                    "lookat": [0.0, 0.0, -0.5],
                    "fov": 60.0,
                    "lights": {
                        "pos": [0.0, 0.0, 1.3],
                        "dir": [0.0, 0.0, -1.0],
                        "intensity": 0.8,
                        "color": [1.0, 1.0, 1.0],
                        "cutoff": 100,
                        "directional": True,
                        "castshadow": False,
                    },
                },
                "directional": True,
                "castshadow": False,
            }
        self._vis_obs = vis_obs
        if vis_obs:
            self._num_image_stack = 3
            self._observation_space = spaces.Dict(
                {
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(37,)),
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
                    "privileged_observations": spaces.Box(low=-np.inf, high=np.inf, shape=(37,)),
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
            gs.morphs.MJCF(file=os.path.join(os.path.dirname(__file__), "../../assets/ant.xml")),
            surface=gs.surfaces.Default(color=(1.0, 0.5, 0.0, 1.0)),  # Orange color from ant.xml default
        )
        self._plane = self._scene.add_entity(gs.morphs.Plane())

        self._motor_joint_names = ["hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4"]
        self._motors_dof_idx = [self._robot.get_joint(name).dof_start for name in self._motor_joint_names]
        self._base_dof_idx = self._robot.base_joint.dofs_idx_local

        self._default_base_pos = torch.tensor([0.0, 0.0, 0.75], device=self._device).repeat(self._num_envs, 1)
        self._default_base_quat = torch.tensor([1, 0, 0, 0], device=self._device).repeat(self._num_envs, 1)
        self._default_motor_dof_pos = torch.tensor(
            [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0], device=self._device
        ).repeat(self._num_envs, 1)
        self._target = torch.tensor([10000.0, 0, 0], device=self._device).repeat(self._num_envs, 1)

        self._motor_strength = torch.tensor(
            [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0], device=self._device
        )
        self._prev_actions = torch.zeros(self._num_envs, self._num_actions, device=self._device)

        self._termination_height_lower_bound = 0.27
        self._termination_height_upper_bound = 1.3
        self._joint_vel_obs_scale = 0.1
        self._action_penalty = 0.0

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

    def compute_observations(self, states: Dict[str, Any]) -> Dict[str, Any]:
        observations = {}
        # adapt from Jie Xu's implementation
        n_batch = states["progress_buf"].shape[0]
        robot_states = states["robot_states"]

        base_pose = robot_states["base_pose"]
        height = base_pose[:, 2:3]
        base_quat = base_pose[:, 3:]

        base_vel = robot_states["base_vel"]

        joints_pos = robot_states["motor_joints_pos"]
        joints_vel = robot_states["motor_joints_vel"]

        prev_actions = robot_states["prev_actions"]

        target_dirs = self._target - base_pose[:, :3]
        target_dirs[:, 2] = 0
        target_dirs = torch.nn.functional.normalize(target_dirs)
        heading_vec = transform_by_quat(torch.tensor([1.0, 0, 0], device=self._device).repeat(n_batch, 1), base_quat)
        up_vec = transform_by_quat(torch.tensor([0.0, 1, 0], device=self._device).repeat(n_batch, 1), base_quat)

        privileged_observations = torch.cat(
            [
                height,
                base_quat,
                base_vel,
                joints_pos,
                joints_vel * self._joint_vel_obs_scale,
                up_vec[:, 1:2],
                (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),
                prev_actions,
            ],
            dim=-1,
        )
        observations["privileged_observations"] = privileged_observations

        if self._vis_obs:
            batch_size, num_stack, height, width, rgb = self._imgs_buf.shape
            # NOTE: for AFRL agent, RGB observation and privileged observations may has different shapes
            # Reshape: (batch, num_stack, H, W, 3) -> (batch, num_stack * 3, H, W)
            observations["RGB"] = self._imgs_buf.permute(0, 1, 4, 2, 3).reshape(
                batch_size, num_stack * rgb, height, width
            )

        return observations

    def compute_reward(self, states: Dict[str, Any], actions: torch.Tensor) -> torch.Tensor:
        # Jie Xu's reward function
        n_batch = states["progress_buf"].shape[0]

        base_pose = states["robot_states"]["base_pose"]
        base_quat = base_pose[:, 3:]

        # up reward
        up_vec = transform_by_quat(torch.tensor([0.0, 1.0, 0.0], device=self._device).repeat(n_batch, 1), base_quat)
        up_reward = 0.1 * up_vec[:, 1]

        # height reward
        height = states["robot_states"]["base_pose"][:, 2]
        height_reward = height - self._termination_height_lower_bound

        # progress reward
        progress_reward = states["robot_states"]["base_vel"][:, 0]

        # heading reward
        target_dirs = self._target - base_pose[:, :3]
        target_dirs[:, 2] = 0
        target_dirs = torch.nn.functional.normalize(target_dirs)
        heading_vec = transform_by_quat(
            torch.tensor([1.0, 0.0, 0.0], device=self._device).repeat(n_batch, 1), base_quat
        )
        heading_reward = (heading_vec * target_dirs).sum(dim=-1)

        # action penalty
        action_penalty = self._action_penalty * torch.sum(actions**2, dim=-1)

        reward = progress_reward + up_reward + heading_reward + height_reward + action_penalty

        return reward

    def compute_termination(self, states: Dict[str, Any]) -> torch.Tensor:
        robot_states = states["robot_states"]
        termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._early_termination:
            termination = robot_states["base_pose"][:, 2] < self._termination_height_lower_bound
            # ant z position too high is likely due to simulation physics failure.
            termination = torch.where(robot_states["base_pose"][:, 2] > self._termination_height_upper_bound, True, termination)
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
        self._prev_actions = actions.clone()
        self._robot.control_dofs_force(actions, dofs_idx_local=self._motors_dof_idx)

    def _post_physics_step(self) -> None:
        """Update image buffer by rolling frames and appending new image."""
        if self._vis_obs:
            new_img = self.render(env_ids=self.nominal_env_ids)
            # Roll the buffer to shift old frames: [t-2, t-1, t-0] -> [t-1, t-0, None]
            # This moves older frames "to the left" and makes room for the new frame
            self._imgs_buf = torch.roll(self._imgs_buf, shifts=-1, dims=1)
            self._imgs_buf[:, -1] = new_img

    def render(self, env_ids: Optional[Sequence[int]] = None) -> None:
        if self._vis_obs:
            if env_ids is None:
                env_ids = self.nominal_env_ids
            # Attach the camera to the torso pose
            pos = self._torso_link.get_pos()
            self._camera_mount.set_pos(pos)

            # TODO: genesis will refresh the image when the scene._dt is different from the last render time
            # TODO: temporarily we hack by setting the last render time to 0 to force render the new image
            self._camera._shared_metadata.last_render_timestep = 0
            # TODO: the batch renderer will first render for all envs, and then return the data for the envs_idx, this may significantly increase the render memory
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
        base_vel = self._robot.get_dofs_velocity(self._base_dof_idx, envs_idx=env_ids)
        motor_joints_pos = self._robot.get_dofs_position(self._motors_dof_idx, envs_idx=env_ids)
        motor_joints_vel = self._robot.get_dofs_velocity(self._motors_dof_idx, envs_idx=env_ids)

        robot_states = {
            "base_pose": base_pose.clone(),
            "base_vel": base_vel.clone(),
            "motor_joints_pos": motor_joints_pos.clone(),
            "motor_joints_vel": motor_joints_vel.clone(),
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

        # TODO: shall we update the image buffer here?

        self._progress_buf[env_ids] = states["progress_buf"].clone()
