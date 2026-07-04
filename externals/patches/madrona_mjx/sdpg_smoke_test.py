"""Minimal madrona_mjx BatchRenderer smoke test (hopper, 4 worlds, 64x64).

Mirrors the mujoco_playground MadronaWrapper usage pattern:
- tile model visual fields (geom_rgba/matid/size, light_*) across worlds
- vmap renderer.init / renderer.render over worlds
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "MADRONA_MWGPU_KERNEL_CACHE",
    "/home/haoxiang/.cache/madrona_mjx_kernels/cache.bin",
)

import subprocess
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from madrona_mjx.renderer import BatchRenderer

NUM_WORLDS = 4
WIDTH = HEIGHT = 64
XML = "/home/haoxiang/Documents/Research/SDPG/assets/mujoco/hopper.xml"

mj_model = mujoco.MjModel.from_xml_path(XML)
m = mjx.put_model(mj_model)
print(f"model: ngeom={mj_model.ngeom} ncam={mj_model.ncam} nlight={mj_model.nlight}")

t0 = time.time()
renderer = BatchRenderer(
    m=m,
    gpu_id=0,
    num_worlds=NUM_WORLDS,
    batch_render_view_width=WIDTH,
    batch_render_view_height=HEIGHT,
    enabled_geom_groups=np.asarray([0, 1, 2]),
    enabled_cameras=np.asarray([0]),
    add_cam_debug_geo=False,
    use_rasterizer=False,
    viz_gpu_hdls=None,
)
print(f"BatchRenderer constructed in {time.time() - t0:.1f}s")

# Tile visual model fields across worlds (mujoco_playground pattern).
in_axes = jax.tree_util.tree_map(lambda x: None, m)
in_axes = in_axes.tree_replace({
    "geom_rgba": 0,
    "geom_matid": 0,
    "geom_size": 0,
    "light_pos": 0,
    "light_dir": 0,
    "light_type": 0,
    "light_castshadow": 0,
    "light_cutoff": 0,
})
m_v = m.tree_replace({
    "geom_rgba": jp.repeat(jp.expand_dims(m.geom_rgba, 0), NUM_WORLDS, axis=0),
    "geom_matid": jp.repeat(
        jp.expand_dims(jp.repeat(-1, m.geom_matid.shape[0], 0), 0),
        NUM_WORLDS,
        axis=0,
    ),
    "geom_size": jp.repeat(jp.expand_dims(m.geom_size, 0), NUM_WORLDS, axis=0),
    "light_pos": jp.repeat(jp.expand_dims(m.light_pos, 0), NUM_WORLDS, axis=0),
    "light_dir": jp.repeat(jp.expand_dims(m.light_dir, 0), NUM_WORLDS, axis=0),
    "light_type": jp.repeat(jp.expand_dims(m.light_type, 0), NUM_WORLDS, axis=0),
    "light_castshadow": jp.repeat(
        jp.expand_dims(m.light_castshadow, 0), NUM_WORLDS, axis=0
    ),
    "light_cutoff": jp.repeat(
        jp.expand_dims(m.light_cutoff, 0), NUM_WORLDS, axis=0
    ),
})

# Per-world data with slightly perturbed qpos so worlds differ.
def make_world(rng):
  data = mjx.make_data(m)
  # Perturb the thigh joint so worlds differ visually even with a
  # torso-tracking camera (index 3 = thigh_joint; 0-2 are root DOFs).
  qpos = data.qpos.at[3].add(jax.random.uniform(rng, (), minval=-1.0, maxval=0.0))
  data = data.replace(qpos=qpos)
  return mjx.forward(m, data)

rngs = jax.random.split(jax.random.PRNGKey(0), NUM_WORLDS)
data = jax.vmap(make_world)(rngs)

t0 = time.time()
render_token, rgb, depth = jax.vmap(renderer.init, in_axes=(0, in_axes))(data, m_v)
rgb.block_until_ready()
print(f"renderer.init done in {time.time() - t0:.1f}s (includes megakernel JIT)")

t0 = time.time()
_, rgb2, depth2 = jax.vmap(renderer.render, in_axes=(0, 0))(render_token, data)
rgb2.block_until_ready()
print(f"renderer.render done in {time.time() - t0:.3f}s")

rgb_np = np.asarray(rgb2)
depth_np = np.asarray(depth2)
print("rgb shape:", rgb_np.shape, "dtype:", rgb_np.dtype)
print("depth shape:", depth_np.shape, "dtype:", depth_np.dtype)
assert rgb_np.shape == (NUM_WORLDS, 1, HEIGHT, WIDTH, 4), rgb_np.shape
assert rgb_np.dtype == np.uint8
print("rgb min/max/mean:", rgb_np.min(), rgb_np.max(), rgb_np[..., :3].mean())
finite = np.isfinite(depth_np)
print(
    "depth finite fraction:", finite.mean(),
    "finite min/max:",
    depth_np[finite].min() if finite.any() else None,
    depth_np[finite].max() if finite.any() else None,
)
assert rgb_np[..., :3].max() > 0, "rgb is all zeros!"
n_unique = len(np.unique(rgb_np[..., :3]))
print("unique rgb values:", n_unique)
# worlds should differ slightly (perturbed qpos)
diff = np.abs(
    rgb_np[0, ..., :3].astype(int) - rgb_np[1, ..., :3].astype(int)
).mean()
print("mean |world0 - world1| rgb:", diff)

# Save a PNG strip of the 4 worlds for eyeballing.
try:
  import imageio.v2 as imageio
  strip = np.concatenate([rgb_np[i, 0, :, :, :3] for i in range(NUM_WORLDS)], axis=1)
  out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_rgb.png")
  imageio.imwrite(out_png, strip)
  print("saved", out_png)
except Exception as e:  # imageio may not be installed
  print("png save skipped:", e)

# GPU memory used by this process.
pid = os.getpid()
out = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
    capture_output=True, text=True,
).stdout
for line in out.strip().splitlines():
  if line.startswith(str(pid)):
    print("GPU memory used by this process:", line.split(",")[1].strip())

print("SMOKE TEST PASSED")
