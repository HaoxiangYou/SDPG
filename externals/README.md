
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

Example update command (adjust ref as needed):

```bash
git subtree pull --prefix=externals/Genesis https://github.com/Genesis-Embodied-AI/Genesis.git <ref> --squash
```

