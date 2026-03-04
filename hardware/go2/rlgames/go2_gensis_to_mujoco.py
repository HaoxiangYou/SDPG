# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

import os
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Match envs/genesis_env/go2.py: observation scales, action scale, clip, defaults, command scale.
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0

# PD gains for torque control (Genesis go2: Kp=30, Kd=0.5). tau = Kp*(target - q) - Kd*qd.
KP = 30.0
KD = 1.5

# Torque limits per actuator (go2.xml motors: abduction/hip ±23.7, knee ±45.43). MuJoCo order FL,FR,RL,RR.
TORQUE_LIMIT_ABDUCTION_HIP = 23.7
TORQUE_LIMIT_KNEE = 45.43
TORQUE_LIMITS_MJ = np.array(
    [TORQUE_LIMIT_ABDUCTION_HIP, TORQUE_LIMIT_ABDUCTION_HIP, TORQUE_LIMIT_KNEE] * 4,
    dtype=np.float32,
)
# Default DOF positions [rad] in Genesis order: FR, FL, RR, RL (each hip, thigh, calf).
DEFAULT_DOF_POS_GENESIS = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float32,
)
COMMANDS_SCALE = np.array([OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]], dtype=np.float32)

# MuJoCo go2_mjx actuator order: FL, FR, RL, RR. Genesis motor order: FR, FL, RR, RL.
# gen_to_mj[j] = MuJoCo index for Genesis joint j.
GEN_TO_MJ = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)
# gen_from_mj[i] = Genesis index for MuJoCo joint i; used to reorder obs (mj -> gen) and ctrl (gen -> mj).
GEN_FROM_MJ = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)

# Joint limits (low, high) in Genesis order; from go2_mjx classes (abduction, hip, knee) per leg.
JOINT_LIMITS_LOW = np.array(
    [-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227],
    dtype=np.float32,
)
JOINT_LIMITS_HIGH = np.array(
    [1.0472, 3.4907, -0.8378, 1.0472, 3.4907, -0.8378, 1.0472, 4.5379, -0.8378, 1.0472, 4.5379, -0.8378],
    dtype=np.float32,
)


class OnnxController:
    """ONNX controller for the Go-2 robot. Obs and control match envs/genesis_env/go2.py."""

    def __init__(
        self,
        policy_path: str,
        n_substeps: int,
        commands: np.ndarray,
    ):
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self._n_substeps = n_substeps
        self._commands = np.asarray(commands, dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._counter = 0
        self._global_gravity = np.array([0, 0, -1], dtype=np.float32)

    def get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Build privileged obs matching go2.compute_observations: [ang_vel*0.25, gravity, cmd*scale, dof_pos, dof_vel, prev_actions]."""
        # Angular velocity (body frame). Gyro = base_ang_vel; scale like Genesis.
        gyro = data.sensor("gyro").data.copy().astype(np.float32)
        base_ang_vel = gyro * OBS_SCALES["ang_vel"]

        # Projected gravity: same as go2 get_states — transform_by_quat(_global_gravity, inv_base_quat).
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        proj_gravity = imu_xmat.T @ np.array([0, 0, -1])

        # DOF pos/vel in MuJoCo order (FL, FR, RL, RR), then reorder to Genesis (FR, FL, RR, RL)
        qpos = data.qpos[7:19].astype(np.float32)
        qvel = data.qvel[6:18].astype(np.float32)
        dof_pos_gen = qpos[GEN_TO_MJ]
        dof_vel_gen = qvel[GEN_TO_MJ]
        cmd_scaled = self._commands * COMMANDS_SCALE
        dof_pos_obs = (dof_pos_gen - DEFAULT_DOF_POS_GENESIS) * OBS_SCALES["dof_pos"]
        dof_vel_obs = dof_vel_gen * OBS_SCALES["dof_vel"]
        prev_actions = self._last_action.copy()

        obs = np.concatenate(
            [base_ang_vel, proj_gravity, cmd_scaled, dof_pos_obs, dof_vel_obs, prev_actions],
            dtype=np.float32,
        )
        return obs

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        onnx_input = {"obs": obs.reshape(1, -1)}
        outs = self._policy.run(None, onnx_input)
        mu = outs[0].squeeze()
        sigma = np.exp(outs[1].squeeze())
        raw = np.random.normal(mu, sigma, size=(12,))
        actions = np.clip(raw.astype(np.float32), -CLIP_ACTIONS, CLIP_ACTIONS)
        return actions

    def apply_control(self, model: mujoco.MjModel, data: mujoco.MjData, actions: np.ndarray) -> None:
        """Compute PD torques tau = Kp*(target - q) - Kd*qd and set ctrl = tau (motor actuators)."""

        # Target DOF positions in Genesis order, clamp to joint limits
        target_gen = actions * ACTION_SCALE + DEFAULT_DOF_POS_GENESIS
        target_gen = np.clip(target_gen, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        target_mj = target_gen[GEN_FROM_MJ]
        self._last_action = actions.copy()
        # Current q, qd in MuJoCo actuator order (FL, FR, RL, RR)
        q = data.qpos[7:19].astype(np.float32)
        qd = data.qvel[6:18].astype(np.float32)

        # PD torques: tau = Kp*(target - q) - Kd*qd
        tau = KP * (target_mj - q) - KD * qd
        tau = np.clip(tau, -TORQUE_LIMITS_MJ, TORQUE_LIMITS_MJ)

        data.ctrl[:] = tau


def load_model_and_policy():
    """Load MuJoCo model, data, and ONNX policy (no control callback)."""
    scene_path = os.path.join("assets/unitree_go2/scene_genesis.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl_dt = 0.02
    sim_dt = 0.005
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy_path = os.path.join(_HERE, "onnx_checkpoints/go2_walking.onnx")
    # commands [lin_vel_x, lin_vel_y, ang_vel]; e.g. [0,0,0] stand, [1,0,0] forward
    commands = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    policy = OnnxController(policy_path=policy_path, n_substeps=n_substeps, commands=commands)

    mujoco.mj_forward(model, data)
    return model, data, policy


if __name__ == "__main__":
    model, data, policy = load_model_and_policy()

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            obs = policy.get_obs(model, data)
            actions = policy.get_actions(obs)
            for i in range(4):
                policy.apply_control(model, data, actions)
                mujoco.mj_step(model, data)
            time.sleep(0.02)
            v.sync()
