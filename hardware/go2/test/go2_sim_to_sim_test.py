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

import importlib
import os

import genesis as gs
import mujoco
import numpy as np
import onnxruntime as ort
import onnxruntime as rt
import torch
from genesis.utils.geom import inv_quat, transform_by_quat
from genesis.utils.misc import ti_to_torch

from envs.genesis_env import GenesisEnv
from utils.common_utils import snakecase_to_pascalcase

_HERE = os.path.dirname(os.path.abspath(__file__))

# Match envs/genesis_env/go2.py: observation scales, action scale, clip, defaults, command scale.
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
ACTION_SCALE = 0.5
CLIP_ACTIONS = 100.0

# PD gains for torque control (Genesis go2: Kp=30, Kd=0.5). tau = Kp*(target - q) - Kd*qd.
KP = 30.0
KD = 0.5

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

env_name = "go2"
num_envs = 1
device = "cpu"
sim_options = gs.options.SimOptions(dt=0.005, substeps=1)
env_kwargs = {"show_viewer": True, "randomize_init": False}


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
        # gyro = data.sensor("gyro").data.copy().astype(np.float32)
        # base_ang_vel = gyro * OBS_SCALES["ang_vel"]

        # # Projected gravity: same as go2 get_states — transform_by_quat(_global_gravity, inv_base_quat).
        # imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        # proj_gravity = imu_xmat.T @ np.array([0, 0, -1])

        base_quat = data.qpos[3:7].astype(np.float32)
        inv_base_quat = inv_quat(base_quat)
        base_ang_vel = transform_by_quat(data.qvel[3:6].astype(np.float32), inv_base_quat) * OBS_SCALES["ang_vel"]
        proj_gravity = transform_by_quat(self._global_gravity, inv_base_quat)

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
        raw = outs[0][0]
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

        return tau


def load_model_and_policy():
    """Load MuJoCo model, data, and ONNX policy (no control callback)."""
    scene_path = os.path.join(_HERE, "..", "mujoco_menagerie", "unitree_go2", "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.005
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy_path = os.path.join(_HERE, "go2_walking.onnx")
    # commands [lin_vel_x, lin_vel_y, ang_vel]; e.g. [0,0,0] stand, [1,0,0] forward
    commands = np.array([1.5, 0.0, 0.0], dtype=np.float32)
    policy = OnnxController(policy_path=policy_path, n_substeps=n_substeps, commands=commands)

    return model, data, policy


if __name__ == "__main__":
    model, data, policy = load_model_and_policy()
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    genesis_env: GenesisEnv = env_fn(num_envs=num_envs, device=device, seed=0, sim_options=sim_options, **env_kwargs)

    genesis_obs, _ = genesis_env.reset()
    genesis_states = genesis_env.get_states()

    mujoco_obs = policy.get_obs(model, data)
    # Check the initial state
    assert np.allclose(data.qpos[:3], genesis_states["robot_states"]["base_pos"])
    assert np.allclose(data.qpos[3:7], genesis_states["robot_states"]["base_quat"])
    assert np.allclose(data.qpos[7:], genesis_states["robot_states"]["motor_joints_pos"])
    assert np.allclose(data.qvel[:3], genesis_states["robot_states"]["base_lin_vel"])
    assert np.allclose(data.qvel[3:6], genesis_states["robot_states"]["base_ang_vel"])
    assert np.allclose(data.qvel[6:], genesis_states["robot_states"]["motor_joints_vel"])

    # Then check initial observations
    # assert np.allclose(mujoco_obs, genesis_obs["privileged_observations"].detach().numpy().squeeze())

    actions_list = []

    genesis_obs, _ = genesis_env.reset()
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2_walking.onnx")
    ort_model = ort.InferenceSession(onnx_path)
    # First get action trajectory from genesis env
    for i in range(100):
        # actions = policy.get_actions(genesis_obs["privileged_observations"].detach().numpy().squeeze())
        actions = ort_model.run(
            None,
            {"obs": genesis_obs["privileged_observations"].detach().numpy()},
        )
        genesis_obs, rew, terminated, truncated, info = genesis_env.step(torch.tensor(actions[0]), auto_reset=False)
        # print("Genesis torques:", genesis_env.torques)
        # print("Genesis applied torques", genesis_env._robot.get_dofs_control_force(genesis_env._motors_dof_idx))
        # print("Genesis applied actuator forces", genesis_env._robot.get_dofs_force(genesis_env._motors_dof_idx))

        # Genesis Newton's second law: M @ qacc = force (total = qf_passive - qf_bias + qf_applied + qf_constraint)
        robot = genesis_env._robot
        solver = robot._solver
        dofs_idx_global = list(range(robot._dof_start, robot._dof_start + robot.n_dofs))
        M_gen = robot.get_mass_mat()
        force_gen = robot.get_dofs_force()

        qacc_gen = ti_to_torch(
            solver.dofs_state.acc,
            None,
            dofs_idx_global,
            transpose=True,
            copy=True,
        )
        if M_gen.dim() == 3:
            M_gen = M_gen.squeeze(0)
        if force_gen.dim() == 2:
            force_gen = force_gen.squeeze(0)
        if qacc_gen.dim() == 2:
            qacc_gen = qacc_gen.squeeze(0)

        lhs_gen = torch.mv(M_gen, qacc_gen)
        eom_err_gen = (lhs_gen - force_gen).abs()
        eom_ok_gen = torch.allclose(lhs_gen, force_gen, atol=1e-4, rtol=1e-3)
        assert eom_ok_gen, "Genesis Newton's second law not satisfied"

        # get external force components
        qf_passive = ti_to_torch(
            solver.dofs_state.qf_passive,
            None,  # row_mask: all envs
            dofs_idx_global,  # col_mask: robot's DOFs
            transpose=True,
            copy=True,
        )
        if qf_passive.dim() == 2:
            qf_passive = qf_passive.squeeze(0)

        qf_bias = ti_to_torch(
            solver.dofs_state.qf_bias,
            None,
            dofs_idx_global,
            transpose=True,
            copy=True,
        )
        if qf_bias.dim() == 2:
            qf_bias = qf_bias.squeeze(0)

        qf_applied = ti_to_torch(
            solver.dofs_state.qf_applied,
            None,
            dofs_idx_global,
            transpose=True,
            copy=True,
        )
        if qf_applied.dim() == 2:
            qf_applied = qf_applied.squeeze(0)

        # Also read constraint forces
        qf_constraint = ti_to_torch(
            solver.dofs_state.qf_constraint,
            None,
            dofs_idx_global,
            transpose=True,
            copy=True,
        )
        if qf_constraint.dim() == 2:
            qf_constraint = qf_constraint.squeeze(0)

        assert force_gen.shape == qf_passive.shape == qf_bias.shape == qf_applied.shape == qf_constraint.shape
        qf_sum = qf_passive - qf_bias + qf_applied + qf_constraint
        assert torch.allclose(force_gen, qf_sum, atol=1e-5, rtol=1e-4), (
            f"force_gen does not match qf_passive - qf_bias + qf_applied + qf_constraint:\n"
            f"  max error: {(force_gen - qf_sum).abs().max().item():.2e}\n"
            f"  force_gen sample [0:6]:   {force_gen[:6].cpu().numpy()}\n"
            f"  qf_sum sample [0:6]:      {qf_sum[:6].cpu().numpy()}"
        )

        genesis_states = genesis_env.get_states()
        data.qpos[:3] = genesis_states["robot_states"]["base_pos"].detach().numpy()
        data.qpos[3:7] = genesis_states["robot_states"]["base_quat"].detach().numpy()
        data.qpos[7:] = (
            genesis_states["robot_states"]["motor_joints_pos"].squeeze(0).detach().cpu().numpy()[GEN_FROM_MJ]
        )
        data.qvel[:3] = genesis_states["robot_states"]["base_lin_vel"].detach().numpy()
        data.qvel[3:6] = genesis_states["robot_states"]["base_ang_vel"].detach().numpy()
        data.qvel[6:] = (
            genesis_states["robot_states"]["motor_joints_vel"].squeeze(0).detach().cpu().numpy()[GEN_FROM_MJ]
        )
        data.ctrl[:] = genesis_env.torques[0, GEN_TO_MJ].detach().numpy()

        mujoco.mj_forward(model, data)
        # print("Mujoco torques", genesis_env.torques[0, :].detach().numpy())
        # print("Mujoco applied torques", data.actuator_force[GEN_TO_MJ])

        # verify Newton's second law
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M, data.qM)

        qacc = data.qacc
        qfrc_applied = data.qfrc_applied
        qfrc_bias = data.qfrc_bias
        tau = data.qfrc_passive + data.qfrc_actuator + data.qfrc_applied
        jac = data.efc_J.reshape(-1, model.nv)
        lhs = M @ qacc

        rhs = tau + jac.T @ data.efc_force - qfrc_bias
        eom_err = np.abs(lhs - rhs)
        eom_ok = np.allclose(lhs, rhs, atol=1e-5, rtol=1e-4)
        actual_torques = tau - qfrc_bias + jac.T @ data.efc_force
        # print("Mujoco actuator forces", actual_torques[6:][GEN_TO_MJ])
        assert eom_ok, "Mujoco Newton's second law not satisfied"
        actions_list.append(actions[0])

        # So here we will know the following relations:
        # MuJoCo: M @ qacc = tau - qfrc_bias + jac.T @ data.efc_force
        # Genesis: M @ qacc = qf_passive + qf_applied - qf_bias + qf_constraint
        print("Mujoco and Genesis Newton's second law check", eom_ok_gen, eom_ok)
        # Reorder MuJoCo vectors to Genesis order (base 0:6 same; joints 6:18 via GEN_TO_MJ)
        tau_genesis_order = np.concatenate([tau[:6], tau[6:][GEN_TO_MJ]])
        qfrc_bias_genesis = np.concatenate([qfrc_bias[:6], qfrc_bias[6:][GEN_TO_MJ]])
        qfrc_constraint_genesis = np.concatenate(
            [(jac.T @ data.efc_force)[:6], (jac.T @ data.efc_force)[6:][GEN_TO_MJ]]
        )
        qf_tau_gen = (qf_passive + qf_applied).detach().cpu().numpy()
        qf_bias_gen = qf_bias.detach().cpu().numpy()
        qf_constraint_gen = qf_constraint.detach().cpu().numpy()
        print("tau difference", np.abs(tau_genesis_order - qf_tau_gen))
        print("bias difference", np.abs(qfrc_bias_genesis - qf_bias_gen))
        print("contact force difference", np.abs(qfrc_constraint_genesis - qf_constraint_gen))

    # with viewer.launch_passive(model, data) as v:
    #   for i in range(100):
    #     # obs = policy.get_obs(model, data)
    #     # actions = policy.get_actions(obs)
    #     for _ in range(4):
    #       tau = policy.apply_control(model, data, actions_list[i].squeeze())
    #       data.ctrl[:] = tau
    #       mujoco.mj_step(model, data)
    #     time.sleep(0.02)
    #     v.sync()
