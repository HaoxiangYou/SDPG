"""Open-loop motion-planning debug script for ALOHA insertion (Genesis).

Drives the bi-manual ALOHA arm through a hand-crafted state machine:
    1. APPROACH -- descend from the default pose straight to the grasp pose.
    2. CLOSE    -- close the grippers (no arm motion).
    3. LIFT     -- lift each grasped object to its task goal.

The plan is executed with *differential IK*: every env step we re-solve
Genesis's `inverse_kinematics` for each arm against a slowly-moving EE
target, convert the joint solution into a delta-position action via the
env's own `_action_scale`, clip to [-1, 1], and call `env.step(action)`.
That goes through the *exact same* control path the RL agent uses, so any
config / kinematics / control issue surfaces here the same way it does
during training.

Env initialisation reuses `cfgs/task/genesis/aloha_insertion.yaml` so the
sim/rigid options and 14-d delta-action space are guaranteed to match the
training env.

Usage:
    python -m envs.genesis_env.scripts.aloha_motion_planning
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import genesis as gs
import numpy as np
import torch
from omegaconf import OmegaConf

from envs.genesis_env.aloha_insertion import AlohaInsertion


GRIP_OPEN = 0.037
GRIP_CLOSED = 0.002

TASK_CFG_PATH = (
    Path(__file__).resolve().parents[3]
    / "cfgs" / "task" / "genesis" / "aloha_insertion.yaml"
)


def _build_env(cfg_path: Path, device: str = "cuda", seed: int = 0) -> AlohaInsertion:
    """Load the aloha_insertion task YAML and instantiate a single-env scene
    with `randomize_init=False` (so the script targets the deterministic
    default object poses) and `show_viewer=True`."""
    cfg = OmegaConf.load(str(cfg_path))
    env_kwargs = OmegaConf.to_container(cfg.config, resolve=False)

    env_kwargs.pop("num_envs", None)
    env_kwargs["randomize_init"] = False
    env_kwargs["show_viewer"] = True

    sim_kwargs = env_kwargs.pop("sim_options", None) or {}
    viewer_kwargs = env_kwargs.pop("viewer_options", None) or {}
    vis_kwargs = env_kwargs.pop("vis_options", None) or {}
    rigid_kwargs = env_kwargs.pop("rigid_options", None) or {}

    sim_options = gs.options.SimOptions(**sim_kwargs) if sim_kwargs else None
    viewer_options = (
        gs.options.ViewerOptions(**viewer_kwargs) if viewer_kwargs else None
    )
    vis_options = gs.options.VisOptions(**vis_kwargs) if vis_kwargs else None
    rigid_options = (
        gs.options.RigidOptions(**rigid_kwargs) if rigid_kwargs else None
    )

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


def _link_pos(link, env_id: int = 0) -> np.ndarray:
    return link.get_pos()[env_id].detach().cpu().numpy().astype(np.float64).copy()


def _link_quat(link, env_id: int = 0) -> np.ndarray:
    return link.get_quat()[env_id].detach().cpu().numpy().astype(np.float64).copy()


def _waypoint_step(curr: np.ndarray, target: np.ndarray, step_size: float) -> np.ndarray:
    """Advance `curr` by at most `step_size` (Euclidean) toward `target`.
    If we're already within `step_size` of the target, snap to the target."""
    delta = target - curr
    d = float(np.linalg.norm(delta))
    if d <= step_size:
        return target.copy()
    return curr + (step_size / d) * delta


def _solve_arm_ik(
    robot,
    link,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    arm_dofs,
    device,
    rot_mask_z_only: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Solve IK for one arm at (target_pos, target_quat). Returns the
    qpos for just the 6 active arm dofs (arm_dofs), and the residual
    pos/rot norms in meters and radians.

    Defaults are tightened relative to the Genesis defaults:
      * `max_step_size=0.1` rad/step (default 0.5) so a single Newton
        iteration cannot ping the joints across many radians.
      * `max_solver_iters=20` (default) gives Newton time to converge.
      * `damping=0.05` stabilises the DLS pseudo-inverse near singularities.
      * `max_samples=1` keeps us on the IK branch closest to the live qpos
        (no random restarts -> no jumpy targets).

    If `rot_mask_z_only=True`, only the gripper Z-axis is constrained to
    align with the target quat's Z-axis, which is the standard relaxation
    for top-down picking and avoids unreachable-orientation rejections.
    """
    pos_t = torch.from_numpy(np.asarray(target_pos, dtype=np.float32)).unsqueeze(0).to(device)
    quat_t = torch.from_numpy(np.asarray(target_quat, dtype=np.float32)).unsqueeze(0).to(device)
    rot_mask = [False, False, True] if rot_mask_z_only else [True, True, True]
    q, err = robot.inverse_kinematics(
        link=link,
        pos=pos_t,
        quat=quat_t,
        dofs_idx_local=arm_dofs,
        max_samples=1,
        max_solver_iters=20,
        max_step_size=0.1,
        damping=0.05,
        rot_mask=rot_mask,
        respect_joint_limit=True,
        return_error=True,
    )
    arm_dofs_t = torch.as_tensor(arm_dofs, device=device, dtype=torch.long)
    q_active = q[0].index_select(0, arm_dofs_t).contiguous()
    err_np = err[0].detach().cpu().numpy()
    return q_active, float(np.linalg.norm(err_np[:3])), float(np.linalg.norm(err_np[3:]))


def _build_action(
    env,
    q_left: torch.Tensor,
    q_right: torch.Tensor,
    grip_closed: bool,
    max_arm_action: float = 1.0,
) -> torch.Tensor:
    """Pack a 14-d delta-position action driving the env's stateful
    `_ctrl` toward (q_left, q_right, grip).

    Action follows the env's update rule: ctrl_new = ctrl + action *
    action_scale, so we send `action = (q_target - ctrl) / action_scale`
    and let env.step clamp to [-1, 1]. `max_arm_action` is an additional
    per-phase cap (slower descents, full-speed lifts).
    """
    device = env.device
    action_scale = float(env._action_scale)
    ctrl = env._ctrl[0]  # (14,)
    actions = torch.zeros(14, device=device, dtype=ctrl.dtype)
    actions[0:6]  = (q_left.to(ctrl.dtype)  - ctrl[0:6])  / action_scale
    actions[6:12] = (q_right.to(ctrl.dtype) - ctrl[6:12]) / action_scale
    grip_target = GRIP_CLOSED if grip_closed else GRIP_OPEN
    actions[12] = (grip_target - ctrl[12].item()) / action_scale
    actions[13] = (grip_target - ctrl[13].item()) / action_scale
    actions[0:12] = actions[0:12].clamp(-max_arm_action, max_arm_action)
    return actions.clamp(-1.0, 1.0).unsqueeze(0)


class DiffIKPlanner:
    """Differential-IK trajectory executor for the bi-manual ALOHA arm.

    Each phase, the planner advances an internal *micro-target* in EE
    space by `step_size` per frame toward the final waypoint, solves IK
    against the micro-target, and converts the joint solution into a
    14-d delta-position action.

    Critically, the micro-target advances from *its own previous value*
    each frame, NOT from the live EE position. This means:
      * At frame 0 the micro-target is just `step_size` ahead of the
        starting EE pose, so IK only has to solve a small problem and
        the action is small -- no initial blast.
      * If the arm is contact-pinned, the micro-target keeps moving
        anyway, so the IK joint target keeps shifting and the action
        eventually saturates to push the arm off the contact (the old
        version that advanced from `live EE` would stall here).
    """

    def __init__(
        self,
        env: AlohaInsertion,
        step_size: float = 0.005,           # 0.5 cm/frame EE advance
        ik_pos_tol_accept: float = 0.05,   # accept IK unless residual >> 5 cm
        ik_rot_tol_accept: float = 0.30,   # ~17 deg
        rot_mask_z_only: bool = False,
    ):
        self.env = env
        self.device = env.device
        self.robot = env._robot
        self.left_link = env._left_gripper_site_link
        self.right_link = env._right_gripper_site_link
        self.left_arm_dofs = env._arm_dofs_idx[:6]
        self.right_arm_dofs = env._arm_dofs_idx[6:12]
        self.step_size = step_size
        self.ik_pos_tol = ik_pos_tol_accept
        self.ik_rot_tol = ik_rot_tol_accept
        self.rot_mask_z_only = rot_mask_z_only
        # Lock orientation: hold whatever quat each gripper site has at
        # construction time (the down-facing default pose).
        self.left_quat0 = _link_quat(self.left_link)
        self.right_quat0 = _link_quat(self.right_link)
        # Internal micro-targets in EE space; reset at the start of each
        # phase by `reset_phase()`.
        self._l_micro: np.ndarray | None = None
        self._r_micro: np.ndarray | None = None
        # Last *accepted* IK joint solution -- fallback if IK ever diverges.
        self._last_q_left: torch.Tensor | None = None
        self._last_q_right: torch.Tensor | None = None

    def reset_phase(self) -> None:
        """Snap the internal micro-targets to the live EE positions.
        Called at the start of every phase so the trajectory is
        continuous in EE space across phases."""
        self._l_micro = _link_pos(self.left_link)
        self._r_micro = _link_pos(self.right_link)

    def _ik_targets(self, l_target: np.ndarray, r_target: np.ndarray):
        """Advance the internal micro-targets by `step_size` toward
        (l_target, r_target), then solve IK against the micro-targets.
        Returns (q_left_active, q_right_active, info_dict)."""
        # Advance micro-targets independently of the live EE -- this is
        # what makes the planner contact-robust.
        self._l_micro = _waypoint_step(self._l_micro, l_target, self.step_size)
        self._r_micro = _waypoint_step(self._r_micro, r_target, self.step_size)

        ql, e_lp, e_lr = _solve_arm_ik(
            self.robot, self.left_link, self._l_micro, self.left_quat0,
            self.left_arm_dofs, self.device,
            rot_mask_z_only=self.rot_mask_z_only,
        )
        qr, e_rp, e_rr = _solve_arm_ik(
            self.robot, self.right_link, self._r_micro, self.right_quat0,
            self.right_arm_dofs, self.device,
            rot_mask_z_only=self.rot_mask_z_only,
        )

        l_ok = (e_lp < self.ik_pos_tol) and (e_lr < self.ik_rot_tol)
        r_ok = (e_rp < self.ik_pos_tol) and (e_rr < self.ik_rot_tol)

        if l_ok or self._last_q_left is None:
            self._last_q_left = ql.detach().clone()
        if r_ok or self._last_q_right is None:
            self._last_q_right = qr.detach().clone()

        info = {
            "l_now": _link_pos(self.left_link),
            "r_now": _link_pos(self.right_link),
            "l_micro": self._l_micro.copy(),
            "r_micro": self._r_micro.copy(),
            "e_lp": e_lp, "e_lr": e_lr, "e_rp": e_rp, "e_rr": e_rr,
            "l_ok": l_ok, "r_ok": r_ok,
        }
        return self._last_q_left, self._last_q_right, info

    def execute(
        self,
        name: str,
        l_target: np.ndarray,
        r_target: np.ndarray,
        grip_closed: bool,
        max_steps: int = 600,
        settle_steps: int = 0,
        sleep_dt: float = 0.0,
        log_every: int = 50,
        verbose: bool = False,
        done_pos_tol: float = 0.015,  # 1.5 cm: stop when both EEs are this close
        max_arm_action: float = 1.0,  # per-phase arm action cap (slow descents)
    ) -> None:
        """Drive both EEs toward (l_target, r_target). Stops when each EE
        is within `done_pos_tol` of its target, then optionally holds for
        `settle_steps` more frames (so e.g. the gripper has time to close
        around the object before we move on).

        Diagnostics:
          * Every `log_every` frames a snapshot is printed (frame #, IK
            residuals, IK accept/reject, EE positions, error to target,
            action norm, ctrl-vs-IK gap).
          * When IK is rejected, a one-liner is also printed (rate-limited
            to once every 25 rejections per arm to avoid log floods).
          * If `verbose=True`, prints a snapshot every frame.
          * A summary at the end shows total IK rejections per arm, max
            EE error during the phase, and the final action norm.
        """
        print(
            f"[plan] phase {name}: target_l={l_target.round(3).tolist()} "
            f"target_r={r_target.round(3).tolist()} "
            f"grip={'closed' if grip_closed else 'open'}"
        )
        # Snap micro-targets to live EE -> first IK problem is small.
        self.reset_phase()

        l_reject_count = 0
        r_reject_count = 0
        max_l_err = 0.0
        max_r_err = 0.0
        last_action_norm = 0.0
        n_done = 0
        # Track signed overshoot per axis for both arms: how far past the
        # target the EE went *in the direction of the original approach*.
        # This is the cleanest single-number diagnostic for "PD too hot".
        l_now0 = _link_pos(self.left_link)
        r_now0 = _link_pos(self.right_link)
        l_dir = l_target - l_now0
        l_dir_norm = float(np.linalg.norm(l_dir)) + 1e-9
        l_unit = l_dir / l_dir_norm
        r_dir = r_target - r_now0
        r_dir_norm = float(np.linalg.norm(r_dir)) + 1e-9
        r_unit = r_dir / r_dir_norm
        l_max_overshoot = 0.0
        r_max_overshoot = 0.0

        def _snapshot(step_idx: int, info: dict, action_t: torch.Tensor):
            l_now = info["l_now"]
            r_now = info["r_now"]
            l_err_v = float(np.linalg.norm(l_target - l_now))
            r_err_v = float(np.linalg.norm(r_target - r_now))
            # Lag = how far the live EE trails behind the planner's micro-target.
            l_lag = float(np.linalg.norm(info["l_micro"] - l_now))
            r_lag = float(np.linalg.norm(info["r_micro"] - r_now))
            ctrl = self.env._ctrl[0]
            arm_action_max = float(action_t[0, :12].abs().max().item())
            grip_action = (
                float(action_t[0, 12].item()),
                float(action_t[0, 13].item()),
            )
            print(
                f"  [{name} f{step_idx:4d}] "
                f"L pos={l_now.round(3).tolist()} err={l_err_v:.4f} lag={l_lag:.4f} "
                f"IK({'ok' if info['l_ok'] else 'REJ'} p={info['e_lp']:.4f},r={info['e_lr']:.4f})\n"
                f"             "
                f"R pos={r_now.round(3).tolist()} err={r_err_v:.4f} lag={r_lag:.4f} "
                f"IK({'ok' if info['r_ok'] else 'REJ'} p={info['e_rp']:.4f},r={info['e_rr']:.4f})\n"
                f"             arm_action_max={arm_action_max:.3f} "
                f"grip_action=({grip_action[0]:+.3f},{grip_action[1]:+.3f})"
            )

        for step in range(max_steps):
            ql, qr, info = self._ik_targets(l_target, r_target)
            actions = _build_action(
                self.env, ql, qr, grip_closed,
                max_arm_action=max_arm_action,
            )
            last_action_norm = float(actions[0, :12].abs().max().item())

            if not info["l_ok"]:
                l_reject_count += 1
                if l_reject_count % 25 == 1:
                    pos_bad = info['e_lp'] >= self.ik_pos_tol
                    rot_bad = info['e_lr'] >= self.ik_rot_tol
                    reasons = []
                    if pos_bad: reasons.append(f"pos {info['e_lp']:.4f}>{self.ik_pos_tol}")
                    if rot_bad: reasons.append(f"rot {info['e_lr']:.4f}>{self.ik_rot_tol}")
                    print(
                        f"  [!] {name} f{step}: LEFT IK rejected ({', '.join(reasons)}); "
                        f"holding last accepted q. count={l_reject_count}"
                    )
            if not info["r_ok"]:
                r_reject_count += 1
                if r_reject_count % 25 == 1:
                    pos_bad = info['e_rp'] >= self.ik_pos_tol
                    rot_bad = info['e_rr'] >= self.ik_rot_tol
                    reasons = []
                    if pos_bad: reasons.append(f"pos {info['e_rp']:.4f}>{self.ik_pos_tol}")
                    if rot_bad: reasons.append(f"rot {info['e_rr']:.4f}>{self.ik_rot_tol}")
                    print(
                        f"  [!] {name} f{step}: RIGHT IK rejected ({', '.join(reasons)}); "
                        f"holding last accepted q. count={r_reject_count}"
                    )

            if verbose or (step % log_every == 0):
                _snapshot(step, info, actions)

            self.env.step(actions, auto_reset=False)
            if sleep_dt > 0:
                time.sleep(sleep_dt)

            l_now_step = _link_pos(self.left_link)
            r_now_step = _link_pos(self.right_link)
            l_err = float(np.linalg.norm(l_target - l_now_step))
            r_err = float(np.linalg.norm(r_target - r_now_step))
            max_l_err = max(max_l_err, l_err)
            max_r_err = max(max_r_err, r_err)
            # Signed projection onto the approach direction. If the EE has
            # gone past `l_target` in that direction, this is positive.
            l_proj = float(np.dot(l_now_step - l_target, l_unit))
            r_proj = float(np.dot(r_now_step - r_target, r_unit))
            l_max_overshoot = max(l_max_overshoot, l_proj)
            r_max_overshoot = max(r_max_overshoot, r_proj)
            if l_err < done_pos_tol and r_err < done_pos_tol:
                n_done += 1
                if n_done >= 3:  # debounce
                    break
            else:
                n_done = 0

        # Hold pose while the (closing) gripper has time to settle.
        for _ in range(settle_steps):
            ql, qr, _ = self._ik_targets(l_target, r_target)
            actions = _build_action(
                self.env, ql, qr, grip_closed,
                max_arm_action=max_arm_action,
            )
            self.env.step(actions, auto_reset=False)
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        l_final = _link_pos(self.left_link)
        r_final = _link_pos(self.right_link)
        total_steps = step + 1
        print(
            f"  [{name} stats] IK rejects L/R={l_reject_count}/{total_steps} | "
            f"{r_reject_count}/{total_steps} ; max_err L={max_l_err:.3f} m, "
            f"R={max_r_err:.3f} m ; last_action_max={last_action_norm:.3f}"
        )
        # Overshoot summary: how far past target each EE went along the
        # original approach direction. Big positive numbers => PD overshoot.
        print(
            f"  [{name} overshoot] L={l_max_overshoot*100:+.1f} cm past target | "
            f"R={r_max_overshoot*100:+.1f} cm past target  "
            f"(<=0 means no overshoot)"
        )
        print(f"[plan] {name} finished after {step + 1} steps. "
              f"L err={np.linalg.norm(l_target - l_final):.4f} m, "
              f"R err={np.linalg.norm(r_target - r_final):.4f} m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action-scale", type=float, default=None,
        help="Override env._action_scale at runtime (default: use env value). "
             "Try 0.05 (env default), 0.01, 0.005 to A/B test PD overshoot.",
    )
    args = parser.parse_args()

    env = _build_env(TASK_CFG_PATH)
    env.reset()

    if args.action_scale is not None:
        print(f"[plan] OVERRIDE env._action_scale: {env._action_scale} -> {args.action_scale}")
        env._action_scale = args.action_scale

    # Read object & goal positions in world frame *after* reset.
    socket_pos = _link_pos_from_entity(env._socket)
    peg_pos = _link_pos_from_entity(env._peg)
    socket_goal = env._socket_entrance_goal_pos[0].detach().cpu().numpy().astype(np.float64)
    peg_goal = env._peg_end2_goal_pos[0].detach().cpu().numpy().astype(np.float64)

    # Phase target poses. Reward layout:
    #   left_socket_dist  -> left arm grasps the socket
    #   right_peg_dist    -> right arm grasps the peg
    grasp_offset_z = 0.0  # gripper site height above object COM at grasp
    lift_offset_z  = grasp_offset_z

    left_grasp  = socket_pos + np.array([0.0, 0.0, grasp_offset_z])
    right_grasp = peg_pos    + np.array([0.0, 0.0, grasp_offset_z])
    left_lift  = socket_goal + np.array([0.0, 0.0, lift_offset_z])
    right_lift = peg_goal    + np.array([0.0, 0.0, lift_offset_z])

    print(
        f"[plan] socket pos={socket_pos.round(4).tolist()} -> goal={socket_goal.round(4).tolist()}\n"
        f"[plan] peg    pos={peg_pos.round(4).tolist()} -> goal={peg_goal.round(4).tolist()}\n"
        f"[plan] env dt={env._scene.dt}, action_scale={env._action_scale}, "
        f"action_repeat={getattr(env, '_action_repeat', 1)}"
    )

    planner = DiffIKPlanner(env)

    # ----- Execute the plan -----
    # The micro-target advances 1 cm/frame in EE space; that's the rate
    # limit. Each phase just calls execute() with its EE-space waypoint.
    planner.execute(
        "APPROACH", left_grasp, right_grasp, grip_closed=False,
        max_steps=600, log_every=50,
    )
    planner.execute(
        "CLOSE", left_grasp, right_grasp, grip_closed=True,
        max_steps=10, settle_steps=30, log_every=5,
    )
    planner.execute(
        "LIFT", left_lift, right_lift, grip_closed=True,
        max_steps=600, log_every=25,
    )

    # Hold at the final pose so the user can inspect the scene.
    print("[plan] holding at goal; Ctrl+C to exit.")
    try:
        while True:
            ql = planner._last_q_left
            qr = planner._last_q_right
            actions = _build_action(env, ql, qr, grip_closed=True)
            env.step(actions, auto_reset=False)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def _link_pos_from_entity(entity, env_id: int = 0) -> np.ndarray:
    """Free-floating peg/socket entities expose `get_pos()` directly on the
    entity (they're a single body with a free joint)."""
    return entity.get_pos()[env_id].detach().cpu().numpy().astype(np.float64).copy()


if __name__ == "__main__":
    main()
