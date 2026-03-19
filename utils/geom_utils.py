import numpy as np
from genesis.utils.geom import R_to_quat, quat_to_xyz


def lookat_to_depth_euler(pos, lookat, up=(0.0, 0.0, 1.0)):
    """Convert lookat camera pose into DepthCamera euler_offset.

    Genesis DepthCamera uses a robotics camera frame whose forward axis is local +X,
    unlike raster cameras that typically look along local -Z.
    """
    pos = np.asarray(pos, dtype=np.float32)
    lookat = np.asarray(lookat, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    x_axis = lookat - pos
    x_axis /= np.linalg.norm(x_axis).clip(min=1e-8)

    y_axis = np.cross(up, x_axis)
    y_axis /= np.linalg.norm(y_axis).clip(min=1e-8)

    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis).clip(min=1e-8)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=-1)
    quat_offset = R_to_quat(rotation)
    return tuple(quat_to_xyz(quat_offset, rpy=True, degrees=True).tolist())
