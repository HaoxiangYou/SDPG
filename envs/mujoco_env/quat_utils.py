"""Torch quaternion helpers (wxyz) shared by the free-root MJX tasks.

They match the genesis backend's geom utilities (transform_quat_by_quat,
transform_by_quat, axis_angle_to_quat) so task code ported from the genesis
envs keeps the same conventions.
"""

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b for wxyz quaternions (matches genesis transform_quat_by_quat)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(quat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by unit wxyz quaternions (matches genesis transform_by_quat)."""
    w, xyz = quat[..., :1], quat[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def axis_angle_to_quat(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion for a rotation of angle (rad) about unit axis (matches genesis)."""
    half = (0.5 * angle).unsqueeze(-1)
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)
