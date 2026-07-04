
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

#### Patch 04: depth camera / raycaster `env_idx` and subset rendering (save memory)

Mirrors Patch 03's approach for the **depth camera** (ray-casting path). When an `env_idx` subset is provided, the BVH, AABB, and output cache are allocated only for the subset of environments, drastically reducing GPU memory (e.g., 4 envs instead of 1024).

- **Options**: `externals/Genesis/genesis/options/sensors/options.py` — `RaycasterOptions.env_idx` (`Optional[Sequence[int]]`). Inherited by `DepthCamera`.
- **Raycaster engine**: `externals/Genesis/genesis/engine/sensors/raycaster.py`:
  - `RaycasterSharedMetadata`: added `env_idx`, `_env_idx_map_t`, `_compact_ground_truth_cache`, `_cls_cache_start_idx` fields for subset tracking.
  - `kernel_update_aabbs`: takes `env_idx_map`; loops over `n_subset` compact rows, reads solver vertex positions with real env index (`i_b`), writes AABB with compact index (`i_local`).
  - `kernel_cast_rays`: BVH nodes and morton codes indexed by `i_local` (compact); solver state (`links_pos`, `links_quat`, `free_verts_state`) indexed by `i_b` (real env); `output_hits` indexed by `i_local`.
  - `build()`: allocates `AABB(n_batches=len(env_idx))` and `LBVH(...)` with the compact batch count. Always sets `_env_idx_map_t` (identity for non-subset, real mapping for subset).
  - `_update_bvh()`: passes `_env_idx_map_t` to `kernel_update_aabbs`.
  - `_update_shared_ground_truth_cache()`: when `env_idx` is set, ray-casts into `_compact_ground_truth_cache` (subset-sized), then scatters back into the full `SensorManager` cache for ring-buffer / delay compatibility.
  - `read()` / `read_ground_truth()`: overridden to read from the compact cache when available, with `_real_to_compact_idx` for transparent index remapping.
- **Depth camera**: `externals/Genesis/genesis/engine/sensors/depth_camera.py` — `build()` sets `_shape` based on `len(env_idx)`; `read_image(envs_idx=None)` delegates to `self.read()` and reshapes.
- **SensorManager**: `externals/Genesis/genesis/engine/sensors/sensor_manager.py` — `# TODO` comment noting that `_ground_truth_cache` is still allocated at full `n_envs` size per dtype (shared across all sensor classes of that dtype). Per-sensor-class cache allocation would be needed for full memory reclamation from the `SensorManager` side.

Reference commit: `de76b998625450c18711b4274f5c873cf73f3cf7`.

#### Patch 05: BatchRenderer depth and segmentation output

The Madrona batch renderer already computes depth and segmentation internally but previously only exposed RGB. This patch adds per-camera control over which outputs to render and cache.

- **Options**: `externals/Genesis/genesis/options/sensors/camera.py` — added `render_rgb` (default `True`), `render_depth` (default `False`), `render_segmentation` (default `False`) to `BatchRendererCameraOptions`. Validation ensures at least one is enabled.
- **Data**: `externals/Genesis/genesis/engine/sensors/camera.py`:
  - `CameraData`: fields are now `rgb`, `depth`, `segmentation` (all `Optional[torch.Tensor]`, default `None`). Backward-compatible — existing code accessing `data.rgb` works unchanged.
  - `BatchRendererCameraSharedMetadata`: added `depth_cache` and `segmentation_cache` dicts alongside `image_cache`.
  - `_camera_read_from_image_cache`: refactored with `_select_from_cache` helper; accepts optional `cached_depth` and `cached_seg` kwargs.
  - `BatchRendererCameraSensor.build()`: conditionally allocates RGB (`uint8`), depth (`float`), and segmentation (`int32`) caches based on the render flags.
  - `BatchRendererCameraSensor._render_current_state()`: passes `render_rgb`/`render_depth`/`render_segmentation` flags to `renderer.render()`; stores results in the appropriate caches. Removed leftover debug code (matplotlib/pdb).
  - `BatchRendererCameraSensor.read()`: overrides base to pass depth and segmentation caches to the read helper.

Backward-compatible: cameras without explicit `render_depth`/`render_segmentation` behave identically to before (RGB only). Depth clipping is controlled by `near`/`far` on the camera options (passed to Madrona as `cam_znears`/`cam_zfars`).

Example update command (adjust ref as needed):

```bash
git subtree pull --prefix=externals/Genesis https://github.com/Genesis-Embodied-AI/Genesis.git <ref> --squash
```

## mujoco (MJX)

MuJoCo is vendored under `externals/mujoco/` as a git subtree (upstream: `https://github.com/google-deepmind/mujoco.git`).

Pinned upstream ref:
- tag `3.3.7` (commit `f1d45bd5`), matching the `mujoco==3.3.7` wheel already installed in the conda env (Genesis requires `mujoco>=3.3.6,<3.4.0`).

Only the **MJX** python package is installed from the subtree (the C engine comes from the PyPI wheel):

```bash
pip install "jax[cuda13]"            # jax 0.10.2 at time of vendoring
pip install -e externals/mujoco/mjx  # editable mujoco-mjx 3.3.7
```

### Local patches

#### Patch 01: reverse-mode differentiable constraint solver

`mjx.step` was not reverse-mode differentiable when `opt.iterations > 1`: the
main solver loop used `jax.lax.while_loop`, which JAX cannot
reverse-differentiate. The line search already used the scan-based
`_while_loop_scan` helper (fixed trip count + done mask, numerically
identical), so the patch applies the same helper to the main loop.

Changes in `externals/mujoco/mjx/mujoco/mjx/_src/solver.py` (in `solve`):

- `ctx = jax.lax.while_loop(cond, body, ctx)` → `ctx = _while_loop_scan(cond, body, ctx, m.opt.iterations)`

Cost note: the scan always runs `opt.iterations` body evaluations (no early
exit), so keep `iterations` small in MJX task XMLs (e.g. `iterations="4"`,
`ls_iterations="8"`, the mujoco_playground convention) rather than relying on
the default 100.

Example update command (adjust ref as needed):

```bash
git subtree pull --prefix=externals/mujoco https://github.com/google-deepmind/mujoco.git <ref> --squash
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

#### Patch 04: Agent batch act and optional tensor return (GPU-friendly)

The main project runs the act/step loop on GPU (Genesis env and agent on same device) to avoid repeated GPU↔CPU transfers. The agent is extended to support this.

**Changes** in `externals/drqv2/drqv2/agent.py`, in `act()`:

- **Batch observations**: If `obs` is already a 4D tensor `(N, C, H, W)`, it is not unsqueezed; only 3D single-obs is unsqueezed to 4D. Return shape is `(N, action_dim)` when `N > 1`, else `(action_dim,)` for backward compatibility.
- **Tensor obs**: If `obs` is a tensor on a different device, it is moved to `self.device` instead of re-creating from numpy.
- **`return_numpy=True` (default)**: Unchanged behavior — action is returned as numpy (e.g. for replay storage).
- **`return_numpy=False`**: Return the action tensor on the agent device so the caller can pass it directly to the env without a CPU round-trip.

#### Patch 05: ReplayBufferStorage.store_episode (multi-env)

When using multiple envs, the workspace maintains per-env episode buffers and flushes a full episode to replay when an env hits `last()`. The storage only exposed `add(time_step)` for single transitions.

**Change** in `externals/drqv2/drqv2/replay_buffer.py`:

- Add a public method `store_episode(self, episode)` that calls the existing `_store_episode(episode)`, so the workspace can push a complete episode dict (e.g. from multi-env `process_time_steps`).

### Integration notes (main project, not in externals)

The following are implemented in **`agents/drqv2.py`** and config (not as patches under `externals/drqv2`), but depend on the package patches above:

- **Backend-agnostic naming**: `PixelDMCEnv`, `DrQv2Workspace`, `ActionRepeatWrapper` (no Genesis-specific names) so other backends can be added later.
- **Logs under Hydra output dir**: TensorBoard, `snapshot.pt`, `train.csv`, and replay buffer are written to Hydra's `runtime.output_dir` (e.g. `logs/.../drqv2/train/<timestamp>/`), same pattern as `agents/afrl.py`.
- **Parallel simulation**: `num_envs >= 1` supported; env returns a list of `ExtendedTimeStep`; batch act, multiple agent updates per step, and `process_time_steps` for per-env episode storage.
- **GPU act/step loop**: Obs and actions stay on GPU between env and agent; conversion to CPU numpy only when storing to replay and for video/logging.
- **Wandb**: Optional Weights & Biases logging via `config.wandb.enable`; same config shape as AFRL (`cfgs/config.yaml`).

After updating, re-apply the package layout (the `drqv2/drqv2/` structure and `pyproject.toml`) and the patches above to the updated files.

