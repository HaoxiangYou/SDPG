
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

#### Patch 03: camera `env_idx` to save memory (subset of envs for image cache)

Batch-renderer camera supports an optional **`env_idx`** so that only a subset of envs are stored in the image cache (saves GPU memory). Render compute still runs for all envs unless the backend supports subset rendering.

- **Options**: `externals/Genesis/genesis/options/sensors/camera.py` — `BatchRendererCameraOptions.env_idx` (optional sequence of int).
- **Engine**: `externals/Genesis/genesis/engine/sensors/camera.py` — shared metadata `env_idx`, cache allocation and `read()` mapping when `env_idx` is set; CUDA tensor handling in `_camera_read_from_image_cache`.

Introduced in commit: `0ceb3ed2b1dc4a6d220c02df3a116c3c7ffcd879`.

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

