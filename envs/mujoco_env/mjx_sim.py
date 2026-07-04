"""JAX/MJX batched simulation core with a torch autograd bridge.

The physics step is a pure function of (qpos, qvel, ctrl): the mjx.Data is
rebuilt from a template inside the jitted function every control step. This
costs the solver warm-start, but makes the dynamics an exact deterministic
function of the exposed state, so the SDPG get_states/set_states round-trip
is exact by construction and the same function can be differentiated for
first-order agents (D.Va/SHAC).
"""

import os

# Torch owns most of the GPU; keep XLA from preallocating its default 75%.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from typing import Tuple

import jax
import mujoco
import torch


def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    # Clone so that later in-place torch ops can never alias an XLA buffer.
    return torch.utils.dlpack.from_dlpack(x).clone()


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(x.detach().contiguous())


class _MjxStepFunction(torch.autograd.Function):
    """Differentiable (qpos, qvel, ctrl) -> (qpos', qvel') through mjx.step.

    The backward pass re-runs the forward through jax.vjp (the pullback is not
    storable across the torch/jax boundary), which roughly doubles the cost of
    differentiable rollouts. This is inherent to first-order methods here and
    is intentionally left unoptimized.
    """

    @staticmethod
    def forward(ctx, sim: "MjxSim", qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor):
        ctx.sim = sim
        ctx.save_for_backward(qpos, qvel, ctrl)
        next_qpos, next_qvel = sim._step_fn(_torch_to_jax(qpos), _torch_to_jax(qvel), _torch_to_jax(ctrl))
        return _jax_to_torch(next_qpos), _jax_to_torch(next_qvel)

    @staticmethod
    def backward(ctx, grad_qpos: torch.Tensor, grad_qvel: torch.Tensor):
        ctx.sim._assert_precision()  # backward may run after another env flipped the x64 flag
        qpos, qvel, ctrl = ctx.saved_tensors
        d_qpos, d_qvel, d_ctrl = ctx.sim._vjp_fn(
            _torch_to_jax(qpos),
            _torch_to_jax(qvel),
            _torch_to_jax(ctrl),
            _torch_to_jax(grad_qpos.contiguous()),
            _torch_to_jax(grad_qvel.contiguous()),
        )
        return None, _jax_to_torch(d_qpos), _jax_to_torch(d_qvel), _jax_to_torch(d_ctrl)


class MjxSim:
    """Batched MJX simulation stepped from torch tensors."""

    def __init__(
        self,
        xml_path: str,
        sim_dt: float,
        n_substeps: int,
        requires_grad: bool = False,
        precision: str = "32",
        solver_iterations: int | None = None,
        ls_iterations: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        if precision not in ("32", "64"):
            raise ValueError(f"precision must be '32' or '64', got {precision!r}")
        # Global for the process; mjx then builds float64 models/data. The
        # flag is re-asserted before every jitted call (jit traces lazily) so
        # envs of different precision can coexist in one process (e.g. tests).
        self._enable_x64 = precision == "64"
        jax.config.update("jax_enable_x64", self._enable_x64)
        if device is not None and torch.device(device).type == "cpu":
            jax.config.update("jax_platform_name", "cpu")

        self._requires_grad = requires_grad
        self._n_substeps = n_substeps

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = sim_dt
        if solver_iterations is not None:
            self._mj_model.opt.iterations = solver_iterations
        if ls_iterations is not None:
            self._mj_model.opt.ls_iterations = ls_iterations

        from mujoco import mjx  # deferred: honors jax config set above

        self._mjx_model = mjx.put_model(self._mj_model)
        self._data_template = mjx.make_data(self._mjx_model)

        def single_step(qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array):
            data = self._data_template.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

            def substep(d, _):
                return mjx.step(self._mjx_model, d), None

            data, _ = jax.lax.scan(substep, data, None, length=self._n_substeps)
            return data.qpos, data.qvel

        batched_step = jax.vmap(single_step)
        self._step_fn = jax.jit(batched_step)

        def step_vjp(qpos, qvel, ctrl, grad_qpos, grad_qvel):
            _, pullback = jax.vjp(batched_step, qpos, qvel, ctrl)
            return pullback((grad_qpos, grad_qvel))

        self._vjp_fn = jax.jit(step_vjp)

    def step(self, qpos: torch.Tensor, qvel: torch.Tensor, ctrl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance one control step (n_substeps physics steps).

        Tensors are (num_envs, nq/nv/nu) in the sim dtype. Differentiable
        w.r.t. all three inputs when the sim was built with requires_grad.
        """
        self._assert_precision()
        if self._requires_grad:
            return _MjxStepFunction.apply(self, qpos, qvel, ctrl)
        with torch.no_grad():
            next_qpos, next_qvel = self._step_fn(_torch_to_jax(qpos), _torch_to_jax(qvel), _torch_to_jax(ctrl))
            return _jax_to_torch(next_qpos), _jax_to_torch(next_qvel)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def nq(self) -> int:
        return self._mj_model.nq

    @property
    def nv(self) -> int:
        return self._mj_model.nv

    @property
    def nu(self) -> int:
        return self._mj_model.nu

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def _assert_precision(self) -> None:
        """Re-assert this sim's x64 mode; jit traces lazily and the flag is
        process-global, so another env of different precision may have
        flipped it since construction."""
        if jax.config.jax_enable_x64 != self._enable_x64:
            jax.config.update("jax_enable_x64", self._enable_x64)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self._enable_x64 else torch.float32
