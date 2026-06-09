import argparse
import copy
import os
import sys
from pathlib import Path

import genesis as gs
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Repo root so we can import envs, agents, utils
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Resolver for agent config num_envs
OmegaConf.register_new_resolver(
    "compute_num_envs",
    lambda n_base, n_pert: n_base * (n_pert + 1),
)

env_name = "go2"
num_envs = 20
device = "cuda"
sim_options = gs.options.SimOptions(dt=0.02, substeps=4)
env_kwargs = {"show_viewer": True, "randomize_init": True}

domain_rand_options = {
    "randomize_friction": False,
    "friction_range": [0.2, 1.5],
    "randomize_base_mass": False,
    "added_mass_range": [-1.0, 3.0],
    "randomize_com_displacement": False,
    "com_displacement_range": [-0.01, 0.01],
    "randomize_motor_strength": False,
    "motor_strength_range": [0.9, 1.1],
    "randomize_motor_offset": False,
    "motor_offset_range": [-0.02, 0.02],
    "randomize_kp_scale": False,
    "kp_scale_range": [0.8, 1.2],
    "randomize_kd_scale": False,
    "kd_scale_range": [0.8, 1.2],
}


def _buffer_name(key: str, prefix: str) -> str:
    """Valid buffer attribute name from observation key (e.g. 'observations' -> 'mean_observations')."""
    return prefix + key.replace("/", "_")


class SDPGPolicyWrapper(nn.Module):
    """Wraps SDPG actor + obs normalization for all actor input keys.
    Accepts a dict of tensors keyed by actor_input_keys; normalizes each with its obs_rms; returns actor(obs)['mean']."""

    def __init__(self, actor, obs_rms, actor_input_keys, mean_bounds=None):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.actor_input_keys = list(actor_input_keys)
        for key in self.actor_input_keys:
            rms = obs_rms.get(key) if obs_rms else None
            if rms is not None:
                self.register_buffer(_buffer_name(key, "mean_"), rms.mean.clone())
                self.register_buffer(_buffer_name(key, "var_"), rms.var.clone())
            else:
                self.register_buffer(_buffer_name(key, "mean_"), torch.zeros(1))
                self.register_buffer(_buffer_name(key, "var_"), torch.ones(1))
        # Action mean bounds (as in sdpg.py); clamp then tanh for env.step
        if mean_bounds is not None:
            low, high = mean_bounds[0], mean_bounds[1]
            self.register_buffer("mean_bound_low", torch.tensor(float(low), dtype=torch.float32))
            self.register_buffer("mean_bound_high", torch.tensor(float(high), dtype=torch.float32))
        else:
            self.register_buffer("mean_bound_low", torch.tensor(float("-inf"), dtype=torch.float32))
            self.register_buffer("mean_bound_high", torch.tensor(float("inf"), dtype=torch.float32))

    def forward(self, obs_dict):
        normalized = {}
        for key in self.actor_input_keys:
            t = obs_dict[key]
            mean = getattr(self, _buffer_name(key, "mean_"))
            var = getattr(self, _buffer_name(key, "var_"))
            if mean.numel() > 1 or var.numel() > 1:
                normalized[key] = (t - mean) / torch.sqrt(var + 1e-5)
            else:
                normalized[key] = t
        mean = self.actor(normalized)["mean"]
        mean = torch.clamp(mean, self.mean_bound_low, self.mean_bound_high)
        return torch.tanh(mean)


def build_config(log_dir: str, checkpoint: str):
    """Build OmegaConf config for SDPG + Genesis Go2 (eval: train=False, play num_envs)."""
    # TODO: currently use the cfgs file; may be better directly load the config in the log dir?
    cfgs_path = REPO_ROOT / "cfgs"
    task_cfg = OmegaConf.load(cfgs_path / "task" / "genesis" / "go2.yaml")
    agent_cfg = OmegaConf.load(cfgs_path / "agent" / "sdpg" / "genesis_go2.yaml")

    task_cfg.config = OmegaConf.merge(task_cfg.config, task_cfg.get("play", {}))
    task_cfg.config.num_envs = num_envs

    agent_cfg.config.num_base_envs = num_envs
    agent_cfg.config.num_action_perturbations = 0
    agent_cfg.config.num_envs = num_envs

    cfg = OmegaConf.create(
        {
            "task": task_cfg,
            "agent": agent_cfg,
            "seed": 0,
            "device": device,
            "log_dir": log_dir,
            "train": False,
            "checkpoint": checkpoint,
        }
    )
    OmegaConf.resolve(cfg)
    return cfg


def export_policy_as_jit(runner, path: str, name: str = "jit_model"):
    """Export SDPG actor + obs_rms as a TorchScript JIT model. Input is a dict of tensors keyed by actor_input_keys."""
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"{name}.pt")

    wrapper = SDPGPolicyWrapper(
        runner.actor,
        runner.obs_rms,
        actor_input_keys=runner.actor_input_keys,
        mean_bounds=getattr(runner, "mean_bounds", None),
    ).to("cpu")
    wrapper.eval()
    example_obs_dict = {
        key: torch.zeros((1,) + tuple(runner.inputs_dim[key]), dtype=torch.float32, device="cpu")
        for key in runner.actor_input_keys
    }
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_obs_dict)
    traced.save(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SDPG policy on Genesis Go2; export/load JIT model.")
    parser.add_argument("checkpoint", type=str, help="Path to the SDPG checkpoint (.pt file).")
    parser.add_argument(
        "--output_directory",
        type=str,
        default=None,
        help="Directory to save the exported JIT model. Default: directory of checkpoint_path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint
    output_directory = args.output_directory
    if output_directory is None:
        output_directory = os.path.dirname(checkpoint)

    log_dir = os.path.dirname(checkpoint)
    jit_ckpt_path = os.path.join(output_directory, "jit_model.pt")

    # Build config and create runner (env + models)
    cfg = build_config(log_dir, checkpoint)
    from agents.sdpg import SDPGRunner

    runner = SDPGRunner(cfg)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    runner.load(checkpoint)
    export_policy_as_jit(runner, output_directory, "jit_model")
    policy = torch.jit.load(jit_ckpt_path)
    policy.to(device=device)

    env = runner.env
    obs, _ = env.reset()

    with torch.no_grad():
        for _ in range(1000):
            actions = policy(obs)
            obs, rewards, terminated, truncated, info = env.step(actions, auto_reset=True)

if __name__ == "__main__":
    main()

