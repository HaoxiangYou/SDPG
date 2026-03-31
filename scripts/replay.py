"""Replay saved trajectories in the Genesis viewer.

Usage:
    python scripts/replay.py task=genesis/walker_hurtle traj_path=path/to/trajectory.pt
    python scripts/replay.py task=genesis/walker_hurtle traj_path=traj.pt replay_num_envs=8
    python scripts/replay.py task=genesis/walker_hurtle traj_path=traj.pt max_frames=500
"""

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import genesis as gs
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.common_utils import snakecase_to_pascalcase
from utils.tensor_utils import select_entries


def _slice_all_batches_at_time(states: Dict[str, Any], t: int) -> Dict[str, Any]:
    """Slice all batches at timestep t: (batch, time, ...) -> (batch, ...)."""
    result = {}
    for key, val in states.items():
        if isinstance(val, dict):
            result[key] = _slice_all_batches_at_time(val, t)
        elif isinstance(val, torch.Tensor):
            result[key] = val[:, t] if val.ndim >= 2 else val
        else:
            result[key] = val
    return result


def _infer_batch_time(states: Dict[str, Any]) -> tuple[int, int]:
    """Infer (batch_size, time_steps) from a nested trajectory dict."""
    for val in states.values():
        if isinstance(val, dict):
            return _infer_batch_time(val)
        if isinstance(val, torch.Tensor) and val.ndim >= 2:
            return int(val.shape[0]), int(val.shape[1])
    return 1, 1


def main() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "cfgs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="replay", overrides=sys.argv[1:])

    if cfg.traj_path is None:
        raise ValueError("Trajectory path required. Use: traj_path=path/to/trajectory.pt")

    OmegaConf.resolve(cfg.task)
    if "play" in cfg.task:
        cfg.task.config = OmegaConf.merge(cfg.task.config, cfg.task.play)

    # Load trajectory and infer dimensions
    states = torch.load(cfg.traj_path, weights_only=False)
    batch_size, time_steps = _infer_batch_time(states)
    if cfg.max_frames is not None:
        time_steps = min(time_steps, int(cfg.max_frames))
    print(f"Trajectory: {batch_size} batch(es) x {time_steps} timesteps")

    # Build env kwargs from task config (play settings already merged)
    env_name = cfg.task.name
    env_kwargs = OmegaConf.to_container(cfg.task.config, resolve=True)

    # num_envs: replay_num_envs override or trajectory batch_size, capped at batch_size
    env_kwargs.pop("num_envs", None)
    num_envs = min(int(cfg.replay_num_envs) if cfg.replay_num_envs is not None else batch_size, batch_size)

    sim_kwargs = env_kwargs.pop("sim_options", None)
    sim_options = gs.options.SimOptions(**sim_kwargs) if sim_kwargs else None
    viewer_kwargs = env_kwargs.pop("viewer_options", None)
    viewer_options = gs.options.ViewerOptions(**viewer_kwargs) if viewer_kwargs else None
    vis_kwargs = env_kwargs.pop("vis_options", None)
    vis_options = gs.options.VisOptions(**vis_kwargs) if vis_kwargs else None

    env_kwargs.setdefault("show_viewer", True)
    env_kwargs.setdefault("show_FPS", True)
    env_kwargs["randomize_init"] = False

    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))

    print(f"Creating environment '{env_name}' with {num_envs} env(s)...")
    env = env_fn(
        num_envs=num_envs,
        device=cfg.device,
        sim_options=sim_options,
        viewer_options=viewer_options,
        vis_options=vis_options,
        **env_kwargs,
    )
    env.reset()

    print("Replaying...")
    for t in range(time_steps):
        state_t = _slice_all_batches_at_time(states, t)
        if batch_size > num_envs:
            state_t = select_entries(state_t, list(range(num_envs)))
        env.set_states(state_t)
        env._scene.step()

    print(f"Replay complete ({time_steps} frames).")


if __name__ == "__main__":
    main()
