"""
This script is used to test and adjust the rendering settings of the Genesis environment.

Usage:

    1.Quick debug camera settings:
        Tweak the hardcoded env setting in-place and run with:
            python envs/genesis_env/scripts/adjust_render.py

    2. Check how camera follows the moving entity in a saved trajectory:
        python envs/genesis_env/scripts/adjust_render.py --traj_path path/to/trajectory.pt

    3. Use a saved config:
        python envs/genesis_env/scripts/adjust_render.py --config path/to/config.yaml
"""

import argparse
import importlib
from pathlib import Path

import genesis as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf

from utils.common_utils import snakecase_to_pascalcase
from utils.tensor_utils import enumerate_states

# ---------------------------------------------------------------------------
# Quick-debug. Tweak these in-place when --config is not used.
# ---------------------------------------------------------------------------
env_name = "walker_hurtle"
num_envs = 4
device = "cuda"
seed = 0

terrain_args = {
    "mesh_type": "heightfield",
    "curriculum": True,
    "selected": False,
    "border_size": 0.0,
    "border_height": 0.0,
    "terrain_length": 5.0,
    "terrain_width": 2.0,
    "platform_size": 0.0,
    "num_rows": 5,
    "num_cols": 1,
    "num_subterrains": 5,
    "horizontal_scale": 0.05,
    "vertical_scale": 0.005,
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "max_init_terrain_level": 1,
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, hurtle, stepping stones, gap, pit]
    "terrain_proportions": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "hurtle_args": {
        "num_stones": 2,
        "x_range": [2.4, 2.5],
        "stone_len": {"offset": 0.2, "scale": 0.0},
        "hurtle_height_range": {
            "low": {"offset": 0.20, "scale": 1.0},
            "high": {"offset": 0.30, "scale": 1.5},
        },
    },
}

sensors_args = {
    "heightfield": {
        "res": 0.2,
        "ahead": 5.0,
        "backward": 2.0,
    },
    "camera": {
        "type": "rgb",
        "res": (84, 84),
        "pos": (0.2, 0.0, -0.1),
        "lookat": (0.8, 0.0, -0.3),
        "fov": 80.0,
        "lights": {
            "pos": (8.0, 0.0, 2.0),
            "intensity": 0.8,
            "color": (1.0, 1.0, 1.0),
            "dir": (0.0, 0.0, -1.0),
            "cutoff": 100,
            "directional": True,
            "castshadow": False,
        },
        "near": 0.01,
        "far": 5.0,
    },
}

env_kwargs = {
    "num_envs": num_envs,
    "device": device,
    "seed": seed,
    "sim_options": gs.options.SimOptions(dt=0.01, substeps=1),
    "randomize_init": False,  # deterministic resets when loading states
    "sensors_args": sensors_args,
    "terrain_args": terrain_args,
    "debug": True,
}


def main(traj_path: str | None = None, config_path: str | None = None):
    global env_name, env_kwargs

    num_envs_override = 1 if traj_path is not None else env_kwargs["num_envs"]

    # --- Replace env_name / env_kwargs from a yaml if --config was given ----
    if config_path is not None:
        cfg = OmegaConf.create(
            {"num_envs": num_envs_override, "task": OmegaConf.load(config_path)}
        )
        OmegaConf.resolve(cfg)
        task_cfg = cfg.task
        if "play" in task_cfg:
            task_cfg.config = OmegaConf.merge(task_cfg.config, task_cfg.play)

        env_name = task_cfg.name
        env_kwargs = OmegaConf.to_container(task_cfg.config, resolve=True)

        for key, cls in (
            ("sim_options", gs.options.SimOptions),
            ("viewer_options", gs.options.ViewerOptions),
            ("rigid_options", gs.options.RigidOptions),
            ("vis_options", gs.options.VisOptions),
        ):
            kw = env_kwargs.pop(key, None)
            env_kwargs[key] = cls(**kw) if kw else None

        env_kwargs["device"] = device
        env_kwargs["seed"] = seed

    env_kwargs["num_envs"] = num_envs_override
    env_kwargs["show_viewer"] = True
    env_kwargs["vis_obs"] = True
    env_kwargs["nominal_env_ids"] = None

    # --- Instantiate --------------------------------------------------------
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env = getattr(ENV, snakecase_to_pascalcase(env_name))(**env_kwargs)

    # --- Run ----------------------------------------------------------------
    if traj_path is not None:
        states = torch.load(traj_path, weights_only=False)

        # frames_per_cam[name] -> list of (H, W, 3) numpy frames for env 0
        frames_per_cam: dict[str, list] = {}
        cam_names: list[str] = []

        for batch_idx, _time_idx, state in enumerate_states(states):
            if batch_idx != 0:
                continue  # only render the first batch
            env.set_states(state)
            env._scene.step()
            pairs = _render_as_named_batches(env)
            if not cam_names:
                cam_names = [n for n, _ in pairs]
                frames_per_cam = {n: [] for n in cam_names}
            for name, img_batch in pairs:
                frames_per_cam[name].append(img_batch.cpu().numpy()[0])

        # One animation window per camera. Keep refs alive so FuncAnimation
        # doesn't get garbage-collected before plt.show() returns.
        anims = []
        for name in cam_names:
            frames = frames_per_cam[name]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axis("off")
            im = ax.imshow(frames[0])
            title = ax.set_title(f"[{name}] Frame 0/{len(frames)}")

            # Factory to avoid Python's late-binding closure bug in loops.
            def _make_update(im, title, name, frames):
                def update(frame_idx):
                    im.set_array(frames[frame_idx])
                    title.set_text(f"[{name}] Frame {frame_idx}/{len(frames)}")
                    return [im, title]
                return update

            anim = FuncAnimation(
                fig,
                _make_update(im, title, name, frames),
                frames=len(frames),
                interval=1000 / 30,
                blit=False,
                repeat=True,
            )
            fig.tight_layout()
            anims.append(anim)

        plt.show()
        return anims

    # No trajectory: reset and snapshot. One grid-figure per camera.
    env.reset()
    pairs = _render_as_named_batches(env)

    for name, img_batch in pairs:
        imgs = img_batch.cpu().numpy()
        n_images = imgs.shape[0]
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(f"Camera: {name}")
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_images):
            axes[i].imshow(imgs[i])
            axes[i].axis("off")
            axes[i].set_title(f"Env {i}", fontsize=10)
        for i in range(n_images, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()

    plt.show()


def _render_as_named_batches(env):
    """Normalize `env.render()` into a list of (camera_name, (N, H, W, 3)) pairs.

    Some envs (e.g. `aloha_insertion`) return a tuple of per-camera image
    batches now that we support multiple cameras; others still return a
    single tensor. This helper smooths over both shapes.
    """
    out = env.render()
    if out is None:
        return []
    if isinstance(out, (tuple, list)):
        # Prefer the env's own camera ordering when available.
        if hasattr(env, "_cameras") and isinstance(env._cameras, dict):
            names = list(env._cameras.keys())
        else:
            names = [f"cam{i}" for i in range(len(out))]
        return list(zip(names, out))
    return [("cam0", out)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, default=None,
                        help="Path to a .pt file containing state history (optional).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a task yaml (e.g. cfgs/task/genesis/g1_hurtle.yaml). "
                             "If given, replaces the hardcoded env_name/env_kwargs above.")
    args = parser.parse_args()

    if args.config is not None and not Path(args.config).is_file():
        raise FileNotFoundError(f"--config path does not exist: {args.config}")

    main(traj_path=args.traj_path, config_path=args.config)
