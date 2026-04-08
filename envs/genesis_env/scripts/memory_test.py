"""Report GPU memory usage after initializing a Genesis environment.

Usage:
    python envs/genesis_env/scripts/memory_test.py genesis/hopper
    python envs/genesis_env/scripts/memory_test.py genesis/walker_hurtle --num_envs 512
    python envs/genesis_env/scripts/memory_test.py genesis/go2_terrain --num_envs 2048 --vis_obs
    python envs/genesis_env/scripts/memory_test.py genesis/walker_hurtle --vis_obs config.sensors_args.camera.type=depth config.sensors_args.camera.res="[256, 256]"
"""

import argparse
import importlib
import sys
from pathlib import Path

import genesis as gs
import torch
import yaml
from omegaconf import OmegaConf

from utils.common_utils import snakecase_to_pascalcase


def _gpu_used_gb() -> float:
    """Return total GPU memory used in GB (matches nvidia-smi)."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def _print_memory(label: str, baseline_gb: float) -> float:
    """Print GPU memory delta from baseline and return current used GB."""
    current = _gpu_used_gb()
    delta = current - baseline_gb
    print(f"[{label}]  +{delta:.3f} GB")
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis environment GPU memory test")
    parser.add_argument("task", help="Task config path, e.g. genesis/hopper")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--vis_obs", action="store_true")
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args()

    cfgs_dir = Path(__file__).resolve().parent.parent.parent.parent / "cfgs"
    task_yaml = cfgs_dir / "task" / f"{args.task}.yaml"
    if not task_yaml.is_file():
        raise FileNotFoundError(f"Task config not found: {task_yaml}")

    task_cfg = OmegaConf.load(task_yaml)
    task_cfg.config.num_envs = args.num_envs
    for ov in overrides:
        key, _, val = ov.partition("=")
        if not val:
            raise ValueError(f"Invalid override (expected key=value): {ov}")
        # Walk the key path to verify every segment exists
        parts = key.split(".")
        node = task_cfg
        for i, part in enumerate(parts):
            if not OmegaConf.is_config(node) or part not in node:
                valid = list(node.keys()) if OmegaConf.is_config(node) else []
                tried = ".".join(parts[: i + 1])
                raise KeyError(f"Key '{tried}' not found. Available keys at '{'.'.join(parts[:i]) or 'root'}': {valid}")
            node = node[part]
        new = yaml.safe_load(val)
        print(f"Override: {key}: {node} -> {new}")
        OmegaConf.update(task_cfg, key, new)
    OmegaConf.resolve(task_cfg)

    env_name = task_cfg.name
    env_kwargs = OmegaConf.to_container(task_cfg.config, resolve=True)

    env_kwargs.pop("num_envs", None)
    env_kwargs["vis_obs"] = args.vis_obs

    sim_kwargs = env_kwargs.pop("sim_options", None)
    sim_options = gs.options.SimOptions(**sim_kwargs) if sim_kwargs else None
    viewer_kwargs = env_kwargs.pop("viewer_options", None)
    viewer_options = gs.options.ViewerOptions(**viewer_kwargs) if viewer_kwargs else None
    vis_kwargs = env_kwargs.pop("vis_options", None)
    vis_options = gs.options.VisOptions(**vis_kwargs) if vis_kwargs else None

    env_kwargs["show_viewer"] = False

    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))

    print(f"Task: {env_name}  |  num_envs: {args.num_envs}  |  vis_obs: {args.vis_obs}")
    print("-" * 60)

    baseline = _gpu_used_gb()

    env = env_fn(
        num_envs=args.num_envs,
        device=args.device,
        sim_options=sim_options,
        viewer_options=viewer_options,
        vis_options=vis_options,
        **env_kwargs,
    )
    _print_memory("after build", baseline)

    env.reset()
    _print_memory("after reset", baseline)

    if args.vis_obs:
        env.render(env_ids=torch.arange(args.num_envs, device=args.device))
        _print_memory("after render", baseline)

    print("-" * 60)
    total = _gpu_used_gb() - baseline
    print(f"Total GPU cost: {total:.3f} GB")


if __name__ == "__main__":
    main()
