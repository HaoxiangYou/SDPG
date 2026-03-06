
# Externals

This folder contains vendored third-party dependencies (typically as **git subtrees**).

## Conventions (recommended)

- **Location**: `externals/<RepoName>/`
- **Pinning**: record the upstream URL + pinned commit/tag here.
- **Local modifications**: list each intentional patch (what/why + exact files changed).
- **Patch files (optional but useful)**: you can also save reproducible diffs under `externals/patches/<RepoName>/...`
  - Create/update a patch file:

```bash
mkdir -p externals/patches/Genesis
git diff -- externals/Genesis > externals/patches/Genesis/local.patch
```

## Genesis

Genesis is vendored under `externals/Genesis/` as a git subtree (upstream: `https://github.com/Genesis-Embodied-AI/Genesis.git`).

Pinned upstream commit:
- `fdb04477f3b00b693e84f7720050115cb1c8c0c8`

### Local patches

#### Patch 01: allow NaN to propagate (handled by env)

We intentionally **do not hard-crash** the simulator on NaNs inside Genesis. Instead, NaNs are allowed to propagate so the **environment** can detect them and handle them explicitly (e.g., terminate/reset an episode, log diagnostics, etc.).

Changes in `externals/Genesis/genesis/engine/solvers/rigid/rigid_solver.py`:

- **Downgrade NaN errors from exception to warning** (around `_report_error`):
  - `"Invalid constraint forces causing 'nan'..."` changed from `gs.raise_exception(...)` → `gs.warn(...)`
  - `"Invalid accelerations causing 'nan'..."` changed from `gs.raise_exception(...)` → `gs.warn(...)`

- **Disable Genesis-side “prevent NaN propagation” guard** (in `func_copy_next_to_curr`):
  - The per-DOF / per-qpos `ti.math.isnan(...)` checks that would block copying `*_next` → `*_curr` were commented out.

If you update Genesis in the future, re-apply (or rebase) this patch to keep NaN handling centralized in the env.

#### Patch 02: fix camera dimension mismatch (may be fixed upstream)

Fixes a dimension mismatch in the batch-renderer camera wrapper when `self._pos` / `self.transform` are not already batched.

Changes in `externals/Genesis/genesis/engine/sensors/camera.py`:
- `get_pos`: use `self._pos.dim()` to decide whether to `unsqueeze(0)` vs return the already-batched tensor (instead of branching on `n_envs`).
- `get_quat`: ensure `transform` is 3D by conditionally `unsqueeze(0)` before calling `T_to_trans_quat(...)`.

Note: this is a bug-fix and **may be resolved by future Genesis updates**. If it gets fixed upstream, this patch can be dropped.

#### Patch 03: camera `env_idx` and subset rendering (save memory)

Patch 03 incorporates the changes from commits `0ceb3ed2` (camera sensor `env_idx` / cache) and `7f2ae64` (batch render subset). Batch-renderer camera supports an optional **`env_idx`** so that only a **subset of envs** is rendered and stored, reducing GPU memory for visual observations.

- **Options**: `externals/Genesis/genesis/options/sensors/camera.py` — `BatchRendererCameraOptions.env_idx` (optional sequence of int).
- **Engine**: `externals/Genesis/genesis/engine/sensors/camera.py` — shared metadata `env_idx`, cache allocation and `read()` mapping; `_camera_read_from_image_cache` uses `renderer._rendered_envs_idx` then `env_idx` for cache indexing; CUDA tensor handling.
- **Batch renderer**: `externals/Genesis/genesis/vis/batch_renderer.py` — `BatchRenderer` uses `rendered_envs_idx`, passes it to the adapter and slices camera poses; `GenesisGeomRetriever` takes `env_ids` and `retrieve_rigid_state_torch()` returns state only for those envs.

Reference commits: `0ceb3ed2b1dc4a6d220c02df3a116c3c7ffcd879`, `db92b374144f77df6270bc579ade0232bf4680c2`,`7f2ae642fd55bd18ac7532cd55df6e63924815b5`.

Example update command (adjust ref as needed):

```bash
git subtree pull --prefix=externals/Genesis https://github.com/Genesis-Embodied-AI/Genesis.git <ref> --squash
```

## rl_games

`rl_games` is vendored under `externals/rl_games/` as a git subtree (upstream: `https://github.com/Denys88/rl_games.git`).

Pinned upstream commit:
- `208b9f9464b8a4ae6fcb17a2d8ee7b6ee44a417b`

### Local patches (Genesis env compatibility; temporary workaround)

These changes are a **temporary workaround** to make `rl_games` compatible with this repo’s `genesis_env` integration. There may be better long-term solutions (e.g., a cleaner adapter layer between env outputs and `rl_games` expectations, or upstream fixes).

#### Patch 01: keep tensor rewards/dones (avoid unconditional `.cpu()`)

In `externals/rl_games/rl_games/common/player.py`, when `self.is_tensor_obses` is true, we keep `rewards` / `dones` as returned (instead of forcing `.cpu()`), to avoid device / type mismatches when the env already produces tensors.

#### Patch 02: make `neglogp` constant device/dtype-safe under torch compile

In `externals/rl_games/rl_games/algos_torch/models.py`, replace the NumPy scalar log constant with a Torch expression on the correct device/dtype. This avoids device/dtype issues in compiled codepaths.

Example update command (adjust ref as needed):

```bash
git subtree pull --prefix=externals/rl_games https://github.com/Denys88/rl_games.git <ref> --squash
```

## drqv2

DrQ-v2 is vendored under `externals/drqv2/` (upstream: `https://github.com/facebookresearch/drqv2.git`). It is used as an **installable package** so the project’s `utils` package is not shadowed when DataLoader workers (spawn) re-run the main script.

### Package layout and install

- **Package root**: `externals/drqv2/` contains a `pyproject.toml` and a `drqv2/` subpackage (the importable package).
- **Modules**: The runnable code lives under `externals/drqv2/drqv2/` (e.g. `agent.py`, `utils.py`, `replay_buffer.py`, `logger.py`, `video.py`, `dmc.py`). The original flat `.py` files at the root of `externals/drqv2/` may be kept for reference or removed.
- **Dependencies**: `externals/drqv2/pyproject.toml` declares `dm-env`, `torchvision`, `termcolor`, plus `torch`, `numpy`, `hydra-core`, `omegaconf`, `imageio`, `opencv-python`. Optional extra: `dm_control` for `drqv2.dmc.make()`.
- **Main project**: The root `pyproject.toml` depends on drqv2 via `drqv2 @ file:./externals/drqv2`. Install from repo root with `pip install -e .`.
- **Hydra**: The agent is instantiated as `drqv2.agent.DrQV2Agent` (see `agents/drqv2.py` and the built config).

### Local patches

All of the following refer to files under **`externals/drqv2/drqv2/`** (the package), unless noted.

#### Patch 01: DataLoader worker seed type (Python 3.10+)

In Python 3.10+, `random.seed()` only accepts built-in types (`int`, `float`, etc.) and rejects NumPy scalars. The replay buffer’s worker init used `np.random.get_state()[1][0]`, which is a NumPy scalar, and passed it to `random.seed()`, causing a `TypeError`.

**Change** in `externals/drqv2/drqv2/replay_buffer.py`, in `_worker_init_fn`:

- Use `seed = int(np.random.get_state()[1][0]) + worker_id` so the value passed to `random.seed(seed)` is a Python `int`.

#### Patch 02: CPU collate for pin_memory

When the main process uses CUDA, the DataLoader’s default collate can create tensors on the default device (GPU). `pin_memory` only works on CPU tensors, which caused `RuntimeError: cannot pin 'torch.cuda.ByteTensor'`.

**Change** in `externals/drqv2/drqv2/replay_buffer.py`:

- Add `_replay_collate_cpu(batch)` that builds all batch tensors with `torch.from_numpy(...)` (CPU only).
- Pass `collate_fn=_replay_collate_cpu` into `torch.utils.data.DataLoader` in `make_replay_loader`.

#### Patch 03: Agent as package submodule and relative imports

To avoid the name collision between the project’s `utils` package and drqv2’s `utils` module, drqv2 is used as an installable package. The agent lives in `drqv2/agent.py` and uses `from . import utils` instead of `import utils`. Hydra target is `drqv2.agent.DrQV2Agent`. The main project’s `agents/drqv2.py` imports from the installed `drqv2` package and no longer mutates `sys.path` or `sys.modules["utils"]`.

Example update command (adjust ref as needed):

```bash
git subtree pull --prefix=externals/drqv2 https://github.com/facebookresearch/drqv2.git main --squash
```

After updating, re-apply the package layout (the `drqv2/drqv2/` structure and `pyproject.toml`) and the patches above to the updated files.

