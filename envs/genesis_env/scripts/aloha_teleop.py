"""ALOHA bimanual teleoperation using to debug the AlohaInsertion env.

Keyboard controls:
    Tab            Switch active arm (left <-> right).
    Up / Down      Move active EE -x / +x.
    Left / Right   Move active EE -y / +y.
    n / m          Move active EE +z / -z.
    j / k          Yaw rotate active EE +z / -z about local z.
    t / g          Pitch rotate active EE +y / -y about local y.
    r / f          Roll  rotate active EE +x / -x about local x.
    space          Toggle gripper close/open for active arm (press to toggle).
    u              Reset env.
    esc            Quit.
"""

import threading
import time
from pathlib import Path
from typing import Tuple

import genesis as gs
import numpy as np
import torch
from omegaconf import OmegaConf
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from envs.genesis_env.aloha_insertion import AlohaInsertion


GRIP_OPEN = 0.037
GRIP_CLOSED = 0.002

TASK_CFG_PATH = (
    Path(__file__).resolve().parents[3] / "cfgs" / "task" / "genesis" / "aloha_insertion.yaml"
)


def _build_env(cfg_path: Path, device: str = "cuda", seed: int = 0) -> AlohaInsertion:
    """Load aloha_insertion task YAML and instantiate the env with teleop overrides.

    Reads `cfgs/task/genesis/aloha_insertion.yaml` for sim/viewer/vis options and
    forces `num_envs=1`, `randomize_init=False`, `show_viewer=True` for teleop.
    """
    cfg = OmegaConf.load(cfg_path)
    env_kwargs = OmegaConf.to_container(cfg.config, resolve=False)

    # Drop the (unresolved) interpolation if present and apply teleop overrides.
    env_kwargs.pop("num_envs", None)
    env_kwargs["randomize_init"] = False
    env_kwargs["show_viewer"] = True

    sim_kwargs = env_kwargs.pop("sim_options", None) or {}
    viewer_kwargs = env_kwargs.pop("viewer_options", None) or {}
    vis_kwargs = env_kwargs.pop("vis_options", None) or {}

    sim_options = gs.options.SimOptions(**sim_kwargs) if sim_kwargs else None
    viewer_options = gs.options.ViewerOptions(**viewer_kwargs) if viewer_kwargs else None
    vis_options = gs.options.VisOptions(**vis_kwargs) if vis_kwargs else None

    return AlohaInsertion(
        num_envs=1,
        device=device,
        seed=seed,
        sim_options=sim_options,
        viewer_options=viewer_options,
        vis_options=vis_options,
        rigid_options=rigid_options,
        **env_kwargs,
    )


class KeyboardDevice:
    """Background keyboard listener using pynput."""

    def __init__(self):
        self.pressed_keys: set = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def start(self):
        self.listener.start()

    def stop(self):
        try:
            self.listener.stop()
        except NotImplementedError:
            pass
        self.listener.join()

    def _on_press(self, key):
        with self.lock:
            self.pressed_keys.add(key)

    def _on_release(self, key):
        with self.lock:
            self.pressed_keys.discard(key)

    def snapshot(self) -> set:
        with self.lock:
            return self.pressed_keys.copy()


def _quat_wxyz_to_R(q: np.ndarray) -> R:
    return R.from_quat([q[1], q[2], q[3], q[0]])


def _R_to_quat_wxyz(rot: R) -> Tuple[float, float, float, float]:
    q = rot.as_quat()  # xyzw
    return float(q[3]), float(q[0]), float(q[1]), float(q[2])


def _read_link_pose(link, env_id: int = 0) -> Tuple[np.ndarray, R]:
    pos = link.get_pos()[env_id].detach().cpu().numpy().astype(np.float64).copy()
    quat = link.get_quat()[env_id].detach().cpu().numpy().astype(np.float64).copy()
    return pos, _quat_wxyz_to_R(quat)


def main():
    env = _build_env(TASK_CFG_PATH)
    env.reset()

    device = env.device
    robot = env._robot
    left_link = env._left_gripper_site_link
    right_link = env._right_gripper_site_link
    left_arm_dofs = env._arm_dofs_idx[:6]
    right_arm_dofs = env._arm_dofs_idx[6:12]
    action_scale = float(env._action_scale)

    state = {
        "active_arm": "left",
        "left_pos": None,
        "left_R": None,
        "right_pos": None,
        "right_R": None,
        "left_grip_closed": False,
        "right_grip_closed": False,
    }

    def reset_targets():
        state["left_pos"], state["left_R"] = _read_link_pose(left_link)
        state["right_pos"], state["right_R"] = _read_link_pose(right_link)
        state["left_grip_closed"] = False
        state["right_grip_closed"] = False

    reset_targets()

    kb = KeyboardDevice()
    kb.start()

    print(__doc__)
    print(f"Active arm: {state['active_arm']}")

    prev_pressed: set = set()
    dpos = 0.005   # m per frame when a translation key is held
    drot = 0.05    # rad per frame when a rotation key is held
    sleep_dt = 0.02
    STEPS_PER_LOOP = max(1, int(round(sleep_dt / env._scene.dt)))
    target_marker = None

    K_U = keyboard.KeyCode.from_char("u")
    K_N = keyboard.KeyCode.from_char("n")
    K_M = keyboard.KeyCode.from_char("m")
    K_J = keyboard.KeyCode.from_char("j")
    K_K = keyboard.KeyCode.from_char("k")
    K_T = keyboard.KeyCode.from_char("t")
    K_G = keyboard.KeyCode.from_char("g")
    K_R = keyboard.KeyCode.from_char("r")
    K_F = keyboard.KeyCode.from_char("f")

    try:
        while True:
            pressed = kb.snapshot()
            new_pressed = pressed - prev_pressed
            prev_pressed = pressed

            if keyboard.Key.esc in pressed:
                break

            if K_U in new_pressed:
                env.reset()
                reset_targets()
                print(f"Env reset. Active arm: {state['active_arm']}")
                continue

            if keyboard.Key.tab in new_pressed:
                state["active_arm"] = (
                    "right" if state["active_arm"] == "left" else "left"
                )
                print(f"Active arm: {state['active_arm']}")

            arm = state["active_arm"]
            tp = state[f"{arm}_pos"]
            tR = state[f"{arm}_R"]

            for k in pressed:
                if k == keyboard.Key.up:
                    tp[0] -= dpos
                elif k == keyboard.Key.down:
                    tp[0] += dpos
                elif k == keyboard.Key.right:
                    tp[1] += dpos
                elif k == keyboard.Key.left:
                    tp[1] -= dpos
                elif k == K_N:
                    tp[2] += dpos
                elif k == K_M:
                    tp[2] -= dpos
                elif k == K_J:
                    tR = tR * R.from_euler("z", drot)
                elif k == K_K:
                    tR = tR * R.from_euler("z", -drot)
                elif k == K_T:
                    tR = tR * R.from_euler("y", drot)
                elif k == K_G:
                    tR = tR * R.from_euler("y", -drot)
                elif k == K_R:
                    tR = tR * R.from_euler("x", drot)
                elif k == K_F:
                    tR = tR * R.from_euler("x", -drot)

            if keyboard.Key.space in new_pressed:
                key = f"{arm}_grip_closed"
                state[key] = not state[key]
                print(f"{arm} gripper:", "closed" if state[key] else "open")

            state[f"{arm}_pos"] = tp
            state[f"{arm}_R"] = tR

            link = left_link if arm == "left" else right_link
            arm_dofs = left_arm_dofs if arm == "left" else right_arm_dofs
            target_quat = _R_to_quat_wxyz(tR)

            target_q = robot.inverse_kinematics(
                link=link,
                pos=torch.from_numpy(np.asarray(tp, dtype=np.float32)).unsqueeze(0).to(device),
                quat=torch.tensor([list(target_quat)], device=device, dtype=torch.float32),
                dofs_idx_local=arm_dofs,
                max_samples=1,
                max_solver_iters=8,
                respect_joint_limit=True,
            )  # full qpos: (1, n_robot_dofs)

            ctrl = env._ctrl[0]  # (14,)
            arm_dofs_t = torch.as_tensor(arm_dofs, device=device, dtype=torch.long)
            target_q_active = target_q[0].to(ctrl.dtype).index_select(0, arm_dofs_t)
            actions = torch.zeros(14, device=device, dtype=ctrl.dtype)
            slc = slice(0, 6) if arm == "left" else slice(6, 12)
            actions[slc] = (target_q_active - ctrl[slc]) / action_scale

            target_left_grip = GRIP_CLOSED if state["left_grip_closed"] else GRIP_OPEN
            target_right_grip = GRIP_CLOSED if state["right_grip_closed"] else GRIP_OPEN
            actions[12] = (target_left_grip - ctrl[12].item()) / action_scale
            actions[13] = (target_right_grip - ctrl[13].item()) / action_scale

            actions = actions.clamp(-1.0, 1.0).unsqueeze(0)
            for _ in range(STEPS_PER_LOOP):
                env.step(actions, auto_reset=False)

            # Visualize active arm's target as a small RGB axis frame.
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = tR.as_matrix()
            T[:3, 3] = tp
            if target_marker is not None:
                env._scene.clear_debug_object(target_marker)
            target_marker = env._scene.draw_debug_frame(
                T, axis_length=0.05, origin_size=0.006, axis_radius=0.0025
            )

            time.sleep(sleep_dt)
    finally:
        if target_marker is not None:
            try:
                env._scene.clear_debug_object(target_marker)
            except Exception:
                pass
        kb.stop()


if __name__ == "__main__":
    main()
