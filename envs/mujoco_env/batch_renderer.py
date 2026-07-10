"""Madrona-MJX batch renderer adapter for the mujoco backend.

Renders only a subset of environments (the SDPG nominal envs): the Madrona
renderer is created with num_worlds = len(env_ids) and fed poses computed for
those envs only, mirroring the genesis backend's BatchRendererCameraOptions
env_idx optimization (auxiliary envs are never rendered).

The renderer keeps its own float32 MJX model/data (posing runs kinematics
only, no physics), so it works next to a float64 differentiable sim: the
process-global jax x64 flag is asserted off around every posing/render call
(jit caches key on the x64 context, so both sides keep their own compiled
traces). Rendering is never differentiated — visual first-order agents (D.Va)
detach the actor input, which is the whole point of the algorithm.
"""

import os
from typing import Optional, Sequence, Tuple

import jax
import numpy as np
import torch

# Cache the compiled madrona megakernels across processes (first compile ~30s+).
_DEFAULT_KERNEL_CACHE = os.path.expanduser("~/.cache/madrona_mjx_kernels/cache.bin")
os.makedirs(os.path.dirname(_DEFAULT_KERNEL_CACHE), exist_ok=True)
os.environ.setdefault("MADRONA_MWGPU_KERNEL_CACHE", _DEFAULT_KERNEL_CACHE)


class MadronaBatchRenderer:
    def __init__(
        self,
        mj_model,
        env_ids: Sequence[int],
        width: int,
        height: int,
        gpu_id: int = 0,
        use_rasterizer: bool = False,
        enabled_geom_groups: Sequence[int] = (0, 1, 2),
    ) -> None:
        from madrona_mjx.renderer import BatchRenderer  # deferred: requires compiled madrona

        self._num_worlds = len(env_ids)

        # Float32 posing pipeline, regardless of the sim precision.
        jax.config.update("jax_enable_x64", False)
        from mujoco import mjx

        self._mjx_model = mjx.put_model(mj_model)
        data_template = mjx.make_data(self._mjx_model)

        self._renderer = BatchRenderer(
            m=self._mjx_model,
            gpu_id=gpu_id,
            num_worlds=self._num_worlds,
            batch_render_view_width=width,
            batch_render_view_height=height,
            enabled_geom_groups=np.asarray(enabled_geom_groups),
            enabled_cameras=np.asarray([0]),
            add_cam_debug_geo=False,
            use_rasterizer=use_rasterizer,
            viz_gpu_hdls=None,
        )

        def pose(qpos: jax.Array, qvel: jax.Array):
            d = data_template.replace(qpos=qpos, qvel=qvel)
            d = mjx.kinematics(self._mjx_model, d)
            d = mjx.com_pos(self._mjx_model, d)
            d = mjx.camlight(self._mjx_model, d)
            return d

        self._pose_fn = jax.jit(jax.vmap(pose))
        self._render_token = None

        # renderer.init/render are written to be called per world under
        # jax.vmap (the mujoco_playground MadronaWrapper pattern): the
        # underlying custom call reads raw device buffers with a fixed
        # (num_worlds, ...) layout and does no shape checking, so an
        # unvmapped call on batched data silently renders garbage (and can
        # fault reading past the too-small camera buffers). Model visual
        # fields consumed by init must be tiled across worlds.
        import jax.numpy as jp

        model_in_axes = jax.tree_util.tree_map(lambda x: None, self._mjx_model)
        model_in_axes = model_in_axes.tree_replace(
            {
                "geom_rgba": 0,
                "geom_matid": 0,
                "geom_size": 0,
                "light_pos": 0,
                "light_dir": 0,
                "light_type": 0,
                "light_castshadow": 0,
                "light_cutoff": 0,
            }
        )

        def tile(x):
            return jp.repeat(jp.expand_dims(x, 0), self._num_worlds, axis=0)

        m = self._mjx_model
        self._tiled_model = m.tree_replace(
            {
                "geom_rgba": tile(m.geom_rgba),
                "geom_matid": tile(jp.repeat(-1, m.geom_matid.shape[0], 0)),
                "geom_size": tile(m.geom_size),
                "light_pos": tile(m.light_pos),
                "light_dir": tile(m.light_dir),
                "light_type": tile(m.light_type),
                "light_castshadow": tile(m.light_castshadow),
                "light_cutoff": tile(m.light_cutoff),
            }
        )
        self._init_fn = jax.vmap(self._renderer.init, in_axes=(0, model_in_axes))
        self._render_fn = jax.vmap(self._renderer.render, in_axes=(0, 0))

    def _pose(self, qpos: torch.Tensor, qvel: torch.Tensor):
        # Assert the f32 trace context (a float64 sim in the same process
        # flips the global flag before its own jitted calls).
        jax.config.update("jax_enable_x64", False)
        qpos = jax.dlpack.from_dlpack(qpos.detach().to(torch.float32).contiguous())
        qvel = jax.dlpack.from_dlpack(qvel.detach().to(torch.float32).contiguous())
        return self._pose_fn(qpos, qvel)

    def init(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Initialize madrona world buffers from the first poses; returns RGB."""
        data = self._pose(qpos, qvel)
        self._render_token, rgb, _ = self._init_fn(data, self._tiled_model)
        return self._rgb_to_torch(rgb)

    def render(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Render all worlds from (qpos, qvel) of the rendered envs; returns
        uint8 RGB of shape (num_worlds, H, W, 3)."""
        if self._render_token is None:
            return self.init(qpos, qvel)
        data = self._pose(qpos, qvel)
        _, rgb, _ = self._render_fn(self._render_token, data)
        return self._rgb_to_torch(rgb)

    @staticmethod
    def _rgb_to_torch(rgb: jax.Array) -> torch.Tensor:
        # madrona returns (num_worlds, num_cams, H, W, 4); a single camera is
        # enabled, so drop the cam axis and the alpha channel.
        return torch.utils.dlpack.from_dlpack(rgb).clone()[:, 0, :, :, :3]

    @property
    def num_worlds(self) -> int:
        return self._num_worlds
