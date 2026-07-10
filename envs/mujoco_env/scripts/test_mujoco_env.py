"""Consistency and gradient checks for the mujoco (MJX) backend.

Checks, in order:
1. get_states/set_states round-trip: teleporting auxiliary envs onto nominal
   states and stepping with identical actions must produce identical next
   states (the core SDPG requirement).
2. Rollout sanity: random-action rollout stays finite, episodes terminate.
3. requires_grad mode: gradients flow through a short rollout back to actions,
   and are finite.
4. Throughput report for both modes.

Run: python envs/mujoco_env/scripts/test_mujoco_env.py [--env hopper|walker]
"""

import argparse
import time

import torch

from envs.mujoco_env.hopper import Hopper
from envs.mujoco_env.walker import Walker

ENVS = {"hopper": Hopper, "walker": Walker}


def flatten_states(states):
    return torch.cat([v for v in states["robot_states"].values()], dim=-1)


def test_round_trip(env_cls, device):
    num_envs = 8
    env = env_cls(num_envs=num_envs, seed=0, device=device, sim_options={"dt": 1e-2, "substeps": 2})
    env.reset()

    # random warmup steps so states differ across envs
    for _ in range(10):
        actions = torch.rand(num_envs, env.num_actions, device=env.device) * 2 - 1
        env.step(actions, auto_reset=False)

    # teleport envs 4..7 onto states of envs 0..3, then step both groups with the same actions
    nominal_ids = torch.arange(0, 4, device=env.device, dtype=torch.int32)
    aux_ids = torch.arange(4, 8, device=env.device, dtype=torch.int32)
    states = env.get_states(nominal_ids)
    env.set_states(states, aux_ids)

    actions = torch.rand(4, env.num_actions, device=env.device) * 2 - 1
    actions = torch.cat([actions, actions], dim=0)
    env.step(actions, auto_reset=False)

    after = env.get_states()
    nominal_flat = flatten_states({"robot_states": {k: v[:4] for k, v in after["robot_states"].items()}})
    aux_flat = flatten_states({"robot_states": {k: v[4:] for k, v in after["robot_states"].items()}})
    max_err = (nominal_flat - aux_flat).abs().max().item()
    assert max_err == 0.0, f"set_states(get_states()) round-trip not exact: max err {max_err}"
    print(f"[1/4] round-trip exactness OK (max err {max_err})")


def test_rollout(env_cls, device):
    num_envs = 64
    env = env_cls(num_envs=num_envs, seed=0, device=device, sim_options={"dt": 1e-2, "substeps": 2})
    env.reset()
    total_terminated = 0
    for i in range(200):
        actions = torch.rand(num_envs, env.num_actions, device=env.device) * 2 - 1
        obs, reward, terminated, truncated, info = env.step(actions)
        assert torch.isfinite(obs["privileged_observations"]).all(), f"non-finite obs at step {i}"
        assert torch.isfinite(reward).all(), f"non-finite reward at step {i}"
        total_terminated += int(terminated.sum())
    assert total_terminated > 0, "random rollout never terminated an episode; termination logic suspicious"
    print(f"[2/4] rollout sanity OK ({total_terminated} terminations over 200 random steps x {num_envs} envs)")


def test_gradients(env_cls, device):
    num_envs = 16
    horizon = 16
    env = env_cls(
        num_envs=num_envs,
        seed=0,
        device=device,
        sim_options={"dt": 1e-2, "substeps": 2, "requires_grad": True},
    )
    assert env.requires_grad
    env.reset()
    obs, _ = env.initialize_trajectory()

    actions = [
        torch.zeros(num_envs, env.num_actions, device=env.device, requires_grad=True) for _ in range(horizon)
    ]
    total_reward = 0.0
    for t in range(horizon):
        obs, reward, terminated, truncated, info = env.step(torch.tanh(actions[t]), auto_reset=True)
        assert "observations_before_reset" in info
        total_reward = total_reward + reward.sum()
    total_reward.backward()

    grad_norms = torch.stack([a.grad.norm() for a in actions])
    assert torch.isfinite(grad_norms).all(), f"non-finite action gradients: {grad_norms}"
    assert (grad_norms > 0).all(), f"zero gradients for some steps: {grad_norms}"

    # graph must be cut by initialize_trajectory
    env.initialize_trajectory()
    states = env.get_states()
    assert not flatten_states(states).requires_grad, "initialize_trajectory failed to cut the graph"
    print(f"[3/4] gradient flow OK (per-step action grad norms {grad_norms.min():.2e}..{grad_norms.max():.2e})")


def test_throughput(env_cls, device):
    for requires_grad, num_envs in ((False, 4096), (True, 64)):
        env = env_cls(
            num_envs=num_envs,
            seed=0,
            device=device,
            sim_options={"dt": 1e-2, "substeps": 2, "requires_grad": requires_grad},
        )
        env.reset()
        actions = torch.zeros(num_envs, env.num_actions, device=env.device)
        env.step(actions)  # compile
        torch.cuda.synchronize()
        t0 = time.time()
        steps = 100
        for _ in range(steps):
            env.step(actions)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(
            f"[4/4] throughput requires_grad={requires_grad} num_envs={num_envs}: "
            f"{steps / dt:.1f} steps/s, {steps * num_envs / dt:.0f} env-steps/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env", default="hopper", choices=sorted(ENVS))
    args = parser.parse_args()

    env_cls = ENVS[args.env]
    test_round_trip(env_cls, args.device)
    test_rollout(env_cls, args.device)
    test_gradients(env_cls, args.device)
    test_throughput(env_cls, args.device)
    print("ALL CHECKS PASSED")
