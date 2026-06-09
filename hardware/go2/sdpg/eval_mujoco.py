"""
SDPG MuJoCo evaluation for Go2: load JIT policy, run with keyboard commands.
Observation is a dict (e.g. {"observations": ...}) to match hardware/go2/sdpg/eval_genesis.py.
Reuses keyboard_reader and MuJoCo/obs constants from hardware/go2/rsl/eval_mujoco.py.
"""
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch

# Reuse keyboard controller from RSL (path added at runtime)
# TODO: make the keyboard_reader in a common folder
_RSL_DIR = Path(__file__).resolve().parent.parent / "rsl"
if str(_RSL_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_DIR))
from keyboard_reader import KeyboardController  # type: ignore[import-untyped]

# MuJoCo/obs constants (same as hardware/go2/rsl/eval_mujoco.py)
MOTOR_INDEX = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.intp)
OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
ACTION_SCALE = 0.5
CLIP_OBS = 100.0

KP = 30.0
KD = 1.5

TORQUE_LIMIT_ABDUCTION_HIP = 23.7
TORQUE_LIMIT_KNEE = 45.43
TORQUE_LIMITS_MJ = np.array(
    [TORQUE_LIMIT_ABDUCTION_HIP, TORQUE_LIMIT_ABDUCTION_HIP, TORQUE_LIMIT_KNEE] * 4,
    dtype=np.float32,
)
DEFAULT_DOF_POS_GENESIS = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float32,
)
COMMANDS_SCALE = np.array(
    [OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]],
    dtype=np.float32,
)

JOINT_LIMITS_LOW = np.array(
    [
        -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227,
        -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227,
    ],
    dtype=np.float32,
)
JOINT_LIMITS_HIGH = np.array(
    [
        1.0472, 3.4907, -0.8378, 1.0472, 3.4907, -0.8378,
        1.0472, 4.5379, -0.8378, 1.0472, 4.5379, -0.8378,
    ],
    dtype=np.float32,
)

# SDPG Go2 actor input key (consistent with eval_genesis / genesis_go2.yaml)
OBS_KEY = "observations"


class SDPGController:
    """SDPG controller for Go2. Obs is a dict keyed by actor input key(s), matching eval_genesis."""

    def __init__(
        self,
        policy: torch.nn.Module,
        commands: np.ndarray,
        obs_key: str = OBS_KEY,
        device: str = "cuda:0",
    ):
        self._policy = policy
        self._commands = np.asarray(commands, dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._obs_key = obs_key
        self._device = device
        self._obs_buf = []

    def set_commands(self, commands: np.ndarray) -> None:
        self._commands = np.asarray(commands, dtype=np.float32)

    def get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        """Build observation dict: {obs_key: array} matching Genesis go2 compute_observations."""
        gyro = data.sensor("gyro").data.copy().astype(np.float32)
        base_ang_vel = gyro * OBS_SCALES["ang_vel"]

        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        proj_gravity = imu_xmat.T @ np.array([0, 0, -1])

        qpos = data.qpos[7:19].astype(np.float32)
        qvel = data.qvel[6:18].astype(np.float32)
        dof_pos_gen = qpos[MOTOR_INDEX]
        dof_vel_gen = qvel[MOTOR_INDEX]
        cmd_scaled = self._commands * COMMANDS_SCALE
        dof_pos_obs = (dof_pos_gen - DEFAULT_DOF_POS_GENESIS) * OBS_SCALES["dof_pos"]
        dof_vel_obs = dof_vel_gen * OBS_SCALES["dof_vel"]
        prev_actions = self._last_action.copy()

        obs = np.concatenate(
            [
                base_ang_vel,
                proj_gravity,
                cmd_scaled,
                dof_pos_obs,
                dof_vel_obs,
                prev_actions,
            ],
            dtype=np.float32,
        )
        obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
        self._obs_buf.append(obs)
        return {self._obs_key: obs}

    def get_actions(self, obs_dict: dict) -> np.ndarray:
        """Policy expects dict of tensors with batch dim; returns (12,) numpy."""
        device = self._device
        batch = {
            k: torch.from_numpy(v).float().unsqueeze(0).to(device)
            for k, v in obs_dict.items()
        }
        with torch.no_grad():
            actions = self._policy(batch)
        return actions.squeeze(0).cpu().numpy()

    def apply_control(
        self, model: mujoco.MjModel, data: mujoco.MjData, actions: np.ndarray
    ) -> None:
        target_gen = actions * ACTION_SCALE + DEFAULT_DOF_POS_GENESIS
        target_gen = np.clip(target_gen, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        target_mj = target_gen[MOTOR_INDEX]
        self._last_action = actions.copy()

        q = data.qpos[7:19].astype(np.float32)
        qd = data.qvel[6:18].astype(np.float32)
        tau = KP * (target_mj - q) - KD * qd
        tau = np.clip(tau, -TORQUE_LIMITS_MJ, TORQUE_LIMITS_MJ)
        data.ctrl[:] = tau


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate SDPG JIT policy on MuJoCo Go2 with keyboard."
    )
    parser.add_argument(
        "jit_model",
        type=str,
        help="Path to jit_model.pt (e.g. from eval_genesis export: .../train/nn/jit_model.pt).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_obs", type=str, default=None, help="Save obs_buf to this .npy path.")
    args = parser.parse_args()

    rsl_dir = Path(__file__).resolve().parent.parent / "rsl"
    scene_path = rsl_dir / "assets" / "unitree_go2" / "scene_genesis.xml"
    if not scene_path.exists():
        scene_path = Path("assets/unitree_go2/scene_genesis.xml")
    scene_path = str(scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.timestep = 0.005

    jit_path = args.jit_model
    if not os.path.exists(jit_path):
        raise FileNotFoundError(
            f"JIT model not found: {jit_path}. Run eval_genesis.py first to export jit_model.pt."
        )
    policy = torch.jit.load(jit_path)
    policy.to(device=args.device)

    keyboard = KeyboardController(vel_scale_x=1.0, vel_scale_y=1.0, vel_scale_rot=1.0)
    controller = SDPGController(
        policy=policy,
        commands=np.zeros(3, dtype=np.float32),
        obs_key=OBS_KEY,
        device=args.device,
    )

    try:
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                controller.set_commands(keyboard.get_command())
                obs_dict = controller.get_obs(model, data)
                actions = controller.get_actions(obs_dict)
                for _ in range(4):
                    controller.apply_control(model, data, actions)
                    mujoco.mj_step(model, data)
                time.sleep(0.02)
                v.sync()
    finally:
        keyboard.stop()
        if args.save_obs:
            np.save(args.save_obs, controller._obs_buf)


if __name__ == "__main__":
    main()
