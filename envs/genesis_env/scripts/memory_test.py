import argparse
import importlib
import os
import subprocess
import sys

import genesis as gs
import matplotlib.pyplot as plt
import torch

from utils.common_utils import snakecase_to_pascalcase


def _gpu_memory_gb_nvml(device_index: int = 0):
    """Return (used_gb, total_gb) from NVML (matches nvidia-smi). Returns (None, None) if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / 1024**3, info.total / 1024**3
    except Exception:
        return None, None


def get_gpu_used_gb() -> float | None:
    """Return current GPU used memory in GB (nvidia-smi), or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    device_index = torch.cuda.current_device()
    used_gb, _ = _gpu_memory_gb_nvml(device_index)
    return used_gb


def print_gpu_memory(label: str):
    """Print GPU memory in GB. PyTorch stats are allocator-only; 'GPU used' matches nvidia-smi."""
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available, skipping GPU memory")
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[{label}] PyTorch (GB): allocated={allocated:.3f}, reserved={reserved:.3f}, max_allocated={max_allocated:.3f}"
    )
    used_gb, total_gb = _gpu_memory_gb_nvml(torch.cuda.current_device())
    if used_gb is not None:
        print(f"[{label}] GPU total used (nvidia-smi): {used_gb:.3f} GB / {total_gb:.3f} GB")


env_name = "hopper"
device = "cuda"
sim_options = gs.options.SimOptions(dt=1e-2, substeps=1)
env_kwargs_base = {
    "show_viewer": False,
    "randomize_init": True,
    "sensors_args": {
        "camera": {
            "res": (256, 256),
            "pos": (0.40, 0.05, 0.425),
            "lookat": (0.25, -0.10, 0.275),
            "fov": 80.0,
            "lights": {
                "pos": (0.0, 0.0, 2.0),
                "intensity": 0.8,
                "color": (1.0, 1.0, 1.0),
                "dir": (0.0, 0.0, -1.0),
                "cutoff": 100,
                "directional": True,
                "castshadow": False,
            },
        },
    },
}

NUM_ENVS_LIST = [1, 64, 512, 1024, 2048]
MEMORY_PREFIX = "MEMORY_GB="


def run_single_measurement(num_envs: int, vis_obs: bool) -> float | None:
    """Create env, reset, (render if vis_obs). Return *additional* GPU used in GB (after - before env), or None."""

    gb_before = get_gpu_used_gb()
    if gb_before is None:
        return None

    kwargs = {**env_kwargs_base, "vis_obs": vis_obs}
    nominal_env_ids = torch.arange(num_envs, device=device)
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env = env_fn(
        num_envs=num_envs,
        nominal_env_ids=nominal_env_ids,
        device=device,
        seed=0,
        sim_options=sim_options,
        **kwargs,
    )
    env.reset()
    if vis_obs:
        env.render(env_ids=nominal_env_ids)
    gb_after = get_gpu_used_gb()
    if gb_after is None:
        return None
    return max(0.0, gb_after - gb_before)


def run_plot():
    """Run each (num_envs, vis_obs) in a subprocess for clean GPU, collect results, plot."""
    script = os.path.abspath(__file__)
    # Project root (parent of envs/) so subprocess imports resolve
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script)))
    mem_state = []
    mem_rgb = []
    for num_envs in NUM_ENVS_LIST:
        # state: vis_obs=False
        out = subprocess.run(
            [sys.executable, script, "--num_envs", str(num_envs), "--vis_obs", "0"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_root,
        )

        def parse_gb(stdout: str) -> float:
            for line in (stdout or "").strip().splitlines():
                if line.strip().startswith(MEMORY_PREFIX):
                    try:
                        return float(line.strip()[len(MEMORY_PREFIX) :].strip())
                    except ValueError:
                        pass
            return float("nan")

        gb = parse_gb(out.stdout or "")
        mem_state.append(gb)
        if out.returncode != 0:
            print(f"num_envs={num_envs} vis_obs=0 exit={out.returncode}, stderr: {(out.stderr or '')[:300]}")

        # RGB: vis_obs=True
        out = subprocess.run(
            [sys.executable, script, "--num_envs", str(num_envs), "--vis_obs", "1"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_root,
        )
        gb = parse_gb(out.stdout or "")
        mem_rgb.append(gb)
        if out.returncode != 0:
            print(f"num_envs={num_envs} vis_obs=1 exit={out.returncode}, stderr: {(out.stderr or '')[:300]}")

    print("mem_state (GB):", mem_state)
    print("mem_rgb (GB):", mem_rgb)

    plt.figure(figsize=(8, 5))
    plt.plot(NUM_ENVS_LIST, mem_state, "o-", label="state", linewidth=2, markersize=8)
    plt.plot(NUM_ENVS_LIST, mem_rgb, "s-", label="RGB", linewidth=2, markersize=8)
    plt.xlabel("num_envs")
    plt.ylabel("GPU memory added by env (GB)")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.legend()
    plt.title("Genesis Hopper env: GPU memory vs num_envs")
    plt.xticks(NUM_ENVS_LIST)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("memory_test_plot.png", dpi=150)
    print("Saved memory_test_plot.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Genesis env memory test")
    parser.add_argument("--num_envs", type=int, default=None, help="Single run: num_envs")
    parser.add_argument("--vis_obs", type=int, default=None, choices=[0, 1], help="Single run: 0=state, 1=RGB")
    parser.add_argument("--plot", action="store_true", help="Run all (num_envs × vis_obs) and plot")
    args = parser.parse_args()

    if args.plot:
        run_plot()
        return

    if args.num_envs is not None and args.vis_obs is not None:
        # Single run for subprocess: print only one parseable line (for parsing)
        import genesis as gs  # noqa: F401

        gb = run_single_measurement(args.num_envs, vis_obs=bool(args.vis_obs))
        val = gb if gb is not None else float("nan")
        print(f"{MEMORY_PREFIX}{val}")
        return

    # Default: one interactive run with printed stats

    num_envs = 64
    nominal_env_ids = torch.arange(num_envs, device=device)
    ENV = importlib.import_module(f"envs.genesis_env.{env_name}")
    env_fn = getattr(ENV, snakecase_to_pascalcase(env_name))
    env_kwargs = {**env_kwargs_base, "vis_obs": True}
    env = env_fn(
        num_envs=num_envs,
        nominal_env_ids=nominal_env_ids,
        device=device,
        seed=0,
        sim_options=sim_options,
        **env_kwargs,
    )
    env.reset()
    print_gpu_memory("after reset")
    env.render(env_ids=nominal_env_ids)
    print_gpu_memory("after render")


if __name__ == "__main__":
    main()
